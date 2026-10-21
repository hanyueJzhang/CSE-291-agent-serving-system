import csv
import io
import json
import socket
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from scripts.sweep import expand_runs, load_plan, main
from tests.test_run_experiment import WORKLOAD

FAKE_SERVER_CMD = f"{sys.executable} -m tests.fake_llama_server"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        workload = self.root / "tiny_mix.json"
        workload.write_text(json.dumps(WORKLOAD))
        self.env_log = self.root / "server_env.jsonl"
        self.port = free_port()
        self.plan = {
            "name": "tiny_sweep",
            "server": {"SERVER_PORT": self.port, "PARALLEL": 2, "CTX": 4096},
            "args": ["--workload", str(workload), "--skip-env-check", "--sample-interval-s", "0.01"],
            "runs": [
                {"run_id": "a", "args": ["--concurrency", "1"]},
                {"run_id": "b", "server": {"PARALLEL": 3, "EXTRA_ARGS": "--cache-ram 0"},
                 "args": ["--concurrency", "2", "--no-cache-prompt"]},
            ],
        }
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan))

    def tearDown(self):
        self.tmp.cleanup()

    def sweep(self, *extra, server_cmd=FAKE_SERVER_CMD):
        import os
        os.environ["FAKE_SERVER_ENV_LOG"] = str(self.env_log)
        try:
            with redirect_stdout(io.StringIO()):
                return main([str(self.plan_path), "--results-dir", str(self.root / "results"),
                             "--no-restore", "--ready-timeout-s", "20", "--server-cmd", server_cmd,
                             *extra])
        finally:
            del os.environ["FAKE_SERVER_ENV_LOG"]

    def server_starts(self):
        if not self.env_log.exists():
            return []
        return [json.loads(line) for line in self.env_log.read_text().splitlines()]

    def test_plan_merging(self):
        runs = expand_runs(load_plan(self.plan_path))
        self.assertEqual(runs[0]["server"]["PARALLEL"], "2")
        self.assertEqual(runs[1]["server"], {"SERVER_PORT": str(self.port), "PARALLEL": "3",
                                             "CTX": "4096", "EXTRA_ARGS": "--cache-ram 0"})
        self.assertEqual(runs[1]["args"][-3:], ["--concurrency", "2", "--no-cache-prompt"])

    def test_rejects_duplicate_run_ids(self):
        self.plan["runs"].append({"run_id": "a"})
        self.plan_path.write_text(json.dumps(self.plan))
        with self.assertRaises(ValueError):
            load_plan(self.plan_path)

    def test_runs_each_with_a_fresh_server_and_resumes(self):
        self.assertEqual(self.sweep(), 0)
        out = self.root / "results" / "tiny_sweep"

        # one server start per run, with that run's settings
        self.assertEqual(self.server_starts(), [
            {"CTX": "4096", "PARALLEL": "2", "EXTRA_ARGS": None},
            {"CTX": "4096", "PARALLEL": "3", "EXTRA_ARGS": "--cache-ram 0"},
        ])
        for run_id in ("a", "b"):
            self.assertTrue((out / run_id / "requests.csv").exists())
        config_b = json.loads((out / "b" / "run_config.json").read_text())
        self.assertEqual(config_b["server"]["total_slots"], 3)
        self.assertFalse(config_b["client"]["cache_prompt"])

        log = [json.loads(line) for line in (out / "sweep_log.jsonl").read_text().splitlines()]
        self.assertEqual([(r["run_id"], r["status"]) for r in log], [("a", "ok"), ("b", "ok")])
        with (out / "sweep.csv").open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual([(r["run_id"], r["concurrency"], r["server_slots"]) for r in rows],
                         [("a", "1", "2"), ("b", "2", "3")])

        # rerun: both done, nothing restarted
        self.assertEqual(self.sweep(), 0)
        self.assertEqual(len(self.server_starts()), 2)

    def test_incomplete_folder_is_moved_aside_and_rerun(self):
        out = self.root / "results" / "tiny_sweep"
        (out / "a").mkdir(parents=True)
        (out / "a" / "run_config.json").write_text("{}")
        self.assertEqual(self.sweep(), 0)
        self.assertTrue((out / "a" / "requests.csv").exists())
        self.assertEqual(len(list(out.glob("a.incomplete-*"))), 1)

    def test_server_that_dies_marks_run_failed(self):
        code = self.sweep(server_cmd=f"{sys.executable} -c 'raise SystemExit(3)'")
        self.assertEqual(code, 1)
        out = self.root / "results" / "tiny_sweep"
        log = [json.loads(line) for line in (out / "sweep_log.jsonl").read_text().splitlines()]
        self.assertEqual([r["status"] for r in log], ["failed", "failed"])
        self.assertIn("exited with code 3", log[0]["error"])
        self.assertFalse((out / "sweep.csv").exists())

    def test_dry_run_starts_nothing(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main([str(self.plan_path), "--dry-run", "--results-dir", str(self.root / "results")])
        self.assertIn("--run-id b", buf.getvalue())
        self.assertIn("EXTRA_ARGS=--cache-ram 0", buf.getvalue())
        self.assertEqual(self.server_starts(), [])
        self.assertFalse((self.root / "results").exists())


if __name__ == "__main__":
    unittest.main()
