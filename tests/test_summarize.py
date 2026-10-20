import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis.summarize import load_run, main, percentile, summarize_run, write_table
from tests.fake_llama_server import NORMAL_STREAM, FakeLlamaServer
from tests.test_run_experiment import WORKLOAD
from scripts.run_experiment import parse_args, run_experiment
from workload.metrics import MEMORY_COLUMNS, REQUEST_COLUMNS, TASK_COLUMNS, write_csv


def req(task_id, step_idx, t_arrive, t_dispatch, t_first, t_finish, prompt, cached, out, decode_ms,
        status="ok"):
    return {"run_id": "r", "task_id": task_id, "step_idx": step_idx, "t_arrive": t_arrive,
            "t_dispatch": t_dispatch, "t_first_token": t_first, "t_finish": t_finish,
            "prompt_tokens": prompt, "cached_tokens": cached, "output_tokens": out,
            "prefill_ms": 10, "decode_ms": decode_ms, "status": status,
            "error": "" if status == "ok" else "HTTP 400"}


REQUESTS = [
    req("t0", 0, 0.0, 0.5, 1.0, 2.0, 100, 0, 10, 500),
    req("t1", 0, 1.0, 1.0, 1.5, 2.0, 100, 20, 10, 400),
    req("t1", 2, 3.0, 4.0, 4.5, 5.0, 200, 150, 20, 400),
    req("t2", 0, 2.0, 2.5, None, 3.0, None, None, None, None, status="error"),
]
TASKS = [
    {"run_id": "r", "task_id": "t0", "task_type": "short", "t_arrive": 0.0, "t_finish": 2.0,
     "num_steps": 1, "total_queue_s": 0.5, "total_tool_s": 0.0, "status": "ok"},
    {"run_id": "r", "task_id": "t1", "task_type": "agent", "t_arrive": 1.0, "t_finish": 5.0,
     "num_steps": 2, "total_queue_s": 1.0, "total_tool_s": 1.0, "status": "ok"},
    {"run_id": "r", "task_id": "t2", "task_type": "short", "t_arrive": 2.0, "t_finish": 3.0,
     "num_steps": 1, "total_queue_s": 0.5, "total_tool_s": 0.0, "status": "failed"},
]
MEMORY = [
    {"t": 0.0, "gpu_mem_used_mb": 6000, "gpu_util_pct": 50, "server_rss_mb": 1000,
     "slot_tokens_total": 100, "server_requests_deferred": 0},
    {"t": 1.0, "gpu_mem_used_mb": 6100, "gpu_util_pct": 90, "server_rss_mb": 1200,
     "slot_tokens_total": 400, "server_requests_deferred": 0},
]
CONFIG = {"run_id": "r", "scheduler": {"name": "fifo", "max_concurrency": 4},
          "workload": {"name": "mix", "seed": 1, "arrival": {"rate_per_s": 0.5}},
          "server": {"total_slots": 4, "n_ctx_per_slot": 2048}, "warnings": ["w1"]}


def write_run(run_dir):
    run_dir.mkdir(parents=True)
    write_csv(run_dir / "requests.csv", REQUEST_COLUMNS, REQUESTS)
    write_csv(run_dir / "tasks.csv", TASK_COLUMNS, TASKS)
    write_csv(run_dir / "memory.csv", MEMORY_COLUMNS, MEMORY)
    (run_dir / "run_config.json").write_text(json.dumps(CONFIG))


class PercentileTests(unittest.TestCase):
    def test_matches_linear_interpolation(self):
        self.assertIsNone(percentile([], 50))
        self.assertEqual(percentile([3.0], 99), 3.0)
        self.assertEqual(percentile([4.0, 1.0, 3.0, 2.0], 50), 2.5)
        self.assertAlmostEqual(percentile([1.0, 2.0, 3.0, 4.0], 95), 3.85)
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 100), 4.0)


class SummarizeRunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp.name) / "r"
        write_run(self.run_dir)
        self.s = summarize_run(load_run(self.run_dir))

    def tearDown(self):
        self.tmp.cleanup()

    def test_counts_exclude_failures_from_latency(self):
        self.assertEqual(self.s["counts"], {"tasks": 3, "tasks_ok": 2, "requests": 4,
                                            "requests_ok": 3, "requests_failed": 1})
        self.assertEqual(self.s["requests"]["e2e_s"]["n"], 3)

    def test_request_latencies(self):
        r = self.s["requests"]
        # queue 0.5, 0.0, 1.0 ; ttft incl. queue 1.0, 0.5, 1.5 ; server ttft 0.5, 0.5, 0.5
        self.assertAlmostEqual(r["queue_s"]["mean"], 0.5)
        self.assertAlmostEqual(r["queue_s"]["p50"], 0.5)
        self.assertAlmostEqual(r["ttft_s"]["p50"], 1.0)
        self.assertAlmostEqual(r["ttft_s"]["max"], 1.5)
        self.assertAlmostEqual(r["server_ttft_s"]["p99"], 0.5)
        # e2e 2.0, 1.0, 2.0 ; decode 50, 40, 20 ms/token
        self.assertAlmostEqual(r["e2e_s"]["mean"], 5 / 3)
        self.assertAlmostEqual(r["decode_ms_per_token"]["p50"], 40.0)

    def test_tasks_throughput_cache_memory(self):
        s = self.s
        self.assertAlmostEqual(s["tasks"]["latency_s"]["mean"], 3.0)   # 2.0 and 4.0
        self.assertAlmostEqual(s["tasks"]["latency_excl_tool_s"]["mean"], 2.5)  # 2.0 and 4.0 - 1.0
        self.assertAlmostEqual(s["tasks"]["queue_s"]["mean"], 0.75)
        self.assertEqual(sorted(s["tasks"]["by_type"]), ["agent", "short"])
        self.assertEqual(s["tasks"]["by_type"]["short"]["latency_s"]["n"], 1)
        self.assertAlmostEqual(s["makespan_s"], 5.0)                   # 0.0 -> 5.0, all tasks
        self.assertAlmostEqual(s["throughput"]["tasks_per_s"], 2 / 5)
        self.assertAlmostEqual(s["throughput"]["output_tokens_per_s"], 40 / 5)
        self.assertAlmostEqual(s["prompt_cache"]["cached_ratio_all"], 170 / 400)
        self.assertAlmostEqual(s["prompt_cache"]["cached_ratio_followup_steps"], 150 / 200)
        self.assertEqual(s["memory"]["gpu_mem_peak_mb"], 6100)
        self.assertEqual(s["memory"]["gpu_util_mean_pct"], 70)
        self.assertEqual(s["memory"]["server_rss_peak_mb"], 1200)
        self.assertEqual(s["memory"]["server_requests_deferred_max"], 0)
        self.assertEqual((s["scheduler"], s["concurrency"], s["rate_per_s"]), ("fifo", 4, 0.5))
        self.assertEqual(s["warnings"], ["w1"])

    def test_table(self):
        path = Path(self.tmp.name) / "sweep.csv"
        write_table(path, [self.s])
        with path.open() as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["run_id"], "r")
        self.assertEqual(float(rows[0]["ttft_p50_s"]), 1.0)


class CliTests(unittest.TestCase):
    def test_writes_summary_table_and_plots(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            write_run(a)
            write_run(b)
            table = Path(tmp) / "sweep.csv"
            argv = [str(a), str(b), "--table", str(table)]
            try:
                import matplotlib  # noqa: F401
                argv.append("--plots")
            except ImportError:
                pass
            self.assertEqual(main(argv), 0)
            self.assertTrue((a / "summary.json").exists())
            with table.open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 2)
            if "--plots" in argv:
                self.assertTrue((a / "latency_cdf.png").exists())
                self.assertTrue((a / "memory.png").exists())


class RealRunTests(unittest.IsolatedAsyncioTestCase):
    """Summarize what run_experiment actually writes (fake llama-server)."""

    async def test_summarizes_run_experiment_output(self):
        fake = await FakeLlamaServer(total_slots=2).start()
        fake.handler = FakeLlamaServer.stream(NORMAL_STREAM, delay_s=0.005)
        with tempfile.TemporaryDirectory() as tmp:
            workload = Path(tmp) / "tiny_mix.json"
            workload.write_text(json.dumps(WORKLOAD))
            try:
                out_dir = await run_experiment(parse_args([
                    "--workload", str(workload), "--concurrency", "1", "--url", fake.url,
                    "--results-dir", tmp, "--run-id", "r1",
                    "--sample-interval-s", "0.01", "--skip-env-check",
                ]))
            finally:
                await fake.close()
            s = summarize_run(load_run(out_dir))

        self.assertEqual(s["counts"]["tasks_ok"], 4)
        self.assertEqual((s["scheduler"], s["concurrency"]), ("fifo", 1))
        for key in ("queue_s", "ttft_s", "server_ttft_s", "e2e_s"):
            self.assertEqual(s["requests"][key]["n"], s["counts"]["requests_ok"])
            self.assertGreaterEqual(s["requests"][key]["p50"], 0.0)
        self.assertGreater(s["throughput"]["output_tokens_per_s"], 0)
        self.assertGreater(s["memory"]["samples"], 0)


if __name__ == "__main__":
    unittest.main()
