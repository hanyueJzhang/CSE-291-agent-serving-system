import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_experiment import estimate_peak_context, parse_args, run_experiment
from tests.fake_llama_server import SLOT_STREAM, FakeLlamaServer

WORKLOAD = {
    "name": "tiny_mix",
    "seed": 3,
    "arrival": {"process": "poisson", "rate_per_s": 50.0, "num_tasks": 4},
    "task_types": [
        {"type": "short_chat", "weight": 0.5,
         "steps": [{"kind": "llm", "prompt_tokens": 8, "max_output_tokens": 4}]},
        {"type": "agent", "weight": 0.5, "steps": [
            {"kind": "llm", "prompt_tokens": 8, "max_output_tokens": 4},
            {"kind": "tool", "delay_s": {"dist": "fixed", "value": 0.02}},
            {"kind": "llm", "append_tokens": 4, "max_output_tokens": 4, "carry_context": True},
        ]},
    ],
}


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


class RunExperimentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fake = await FakeLlamaServer(total_slots=2).start()
        self.fake.handler = FakeLlamaServer.stream(SLOT_STREAM, delay_s=0.005)
        self.tmp = tempfile.TemporaryDirectory()
        self.workload_path = Path(self.tmp.name) / "tiny_mix.json"
        self.workload_path.write_text(json.dumps(WORKLOAD))

    async def asyncTearDown(self):
        await self.fake.close()
        self.tmp.cleanup()

    def args(self, *extra):
        return parse_args([
            "--workload", str(self.workload_path), "--concurrency", "1",
            "--url", self.fake.url, "--results-dir", self.tmp.name,
            "--sample-interval-s", "0.01", "--skip-env-check", *extra,
        ])

    async def test_end_to_end_writes_all_outputs(self):
        out_dir = await run_experiment(self.args("--run-id", "r1"))

        self.assertEqual(out_dir, Path(self.tmp.name) / "r1")
        tasks = read_csv(out_dir / "tasks.csv")
        requests = read_csv(out_dir / "requests.csv")
        memory = read_csv(out_dir / "memory.csv")
        config = json.loads((out_dir / "run_config.json").read_text())

        self.assertEqual(len(tasks), 4)
        self.assertTrue(all(t["status"] == "ok" for t in tasks))
        # short_chat tasks send 1 request, agent tasks 2.
        self.assertEqual(len(requests), sum(1 if t["task_type"] == "short_chat" else 2 for t in tasks))
        self.assertTrue(all(r["prompt_tokens"] == "30" and r["cached_tokens"] == "12" for r in requests))
        self.assertTrue(all(r["slot_id"] == "0" for r in requests))
        self.assertGreater(len(memory), 1)
        self.assertEqual(set(memory[0]), {"t", "gpu_mem_used_mb", "gpu_util_pct", "server_rss_mb",
                                          "slot_tokens_total", "server_requests_deferred"})

        self.assertEqual(config["status"], "done")
        self.assertEqual(config["scheduler"], {"name": "fifo", "max_concurrency": 1})
        self.assertEqual(config["server"]["total_slots"], 2)
        self.assertEqual(config["server"]["n_ctx_per_slot"], 2048)
        self.assertTrue(config["client"]["ignore_eos"])
        # Only the dirty-repo warning is allowed (it depends on the checkout running the test).
        self.assertEqual([w for w in config["warnings"] if "uncommitted" not in w], [])
        self.assertFalse((out_dir / "summary.json").exists())

        # One warmup request, then every workload prompt carries the run's salt.
        salt = config["prompt_salt"]
        workload_bodies = self.fake.bodies[1:]
        self.assertEqual(len(workload_bodies), len(requests))
        for body in workload_bodies:
            for message in body["messages"]:
                if message["role"] == "user":
                    self.assertTrue(message["content"].startswith(salt))

    async def test_overrides_and_warnings(self):
        out_dir = await run_experiment(self.args(
            "--run-id", "r2", "--num-tasks", "2", "--concurrency", "3", "--no-cache-prompt", "--warmup", "0",
        ))
        config = json.loads((out_dir / "run_config.json").read_text())

        self.assertEqual(config["workload"]["arrival"]["num_tasks"], 2)
        self.assertEqual(len(read_csv(out_dir / "tasks.csv")), 2)
        self.assertFalse(config["client"]["cache_prompt"])
        self.assertTrue(any("slots 2" in w for w in config["warnings"]))
        self.assertFalse(self.fake.bodies[0]["cache_prompt"])

    async def test_agent_scheduler_and_admission_check(self):
        out_dir = await run_experiment(self.args(
            "--run-id", "agent", "--scheduler", "agent", "--aging-s", "5", "--warmup", "0",
        ))
        config = json.loads((out_dir / "run_config.json").read_text())
        self.assertEqual(config["scheduler"], {"name": "agent", "max_concurrency": 1,
                                               "continuation_first": True, "aging_s": 5.0})
        self.assertEqual(config["admission"]["max_context"], int(0.95 * 2048))
        self.assertTrue(all(t["status"] == "ok" for t in read_csv(out_dir / "tasks.csv")))

        # A slot too small for any task: everything rejected, nothing sent.
        self.fake.n_ctx = 10
        sent = len(self.fake.bodies)
        out_dir = await run_experiment(self.args("--run-id", "tiny", "--warmup", "0"))
        tasks = read_csv(out_dir / "tasks.csv")
        self.assertTrue(all(t["status"] == "rejected" for t in tasks))
        self.assertEqual(len(self.fake.bodies), sent)
        self.assertEqual(read_csv(out_dir / "requests.csv"), [])

        # --no-admission-check restores the Phase 1 behaviour.
        out_dir = await run_experiment(self.args("--run-id", "tiny_off", "--warmup", "0",
                                                 "--no-admission-check"))
        self.assertTrue(all(t["status"] == "ok" for t in read_csv(out_dir / "tasks.csv")))

    async def test_agent_flags_need_agent_scheduler(self):
        with self.assertRaises(ValueError):
            await run_experiment(self.args("--run-id", "bad", "--aging-s", "5"))
        self.assertFalse((Path(self.tmp.name) / "bad").exists())

    async def test_refuses_to_overwrite_a_run(self):
        await run_experiment(self.args("--run-id", "same", "--num-tasks", "1"))
        with self.assertRaises(FileExistsError):
            await run_experiment(self.args("--run-id", "same", "--num-tasks", "1"))

    async def test_unreachable_server_leaves_no_results_folder(self):
        self.fake.health_statuses = [503]
        args = self.args("--run-id", "never", "--ready-timeout-s", "0.05")
        with self.assertRaises(TimeoutError):
            await run_experiment(args)
        self.assertFalse((Path(self.tmp.name) / "never").exists())

    async def test_server_errors_become_failed_tasks(self):
        from aiohttp import web

        async def reject(request):
            self.fake.bodies.append(await request.json())
            return web.json_response({"error": {"code": 400, "message": "exceeds the available context size"}},
                                     status=400)

        self.fake.handler = reject
        out_dir = await run_experiment(self.args("--run-id", "r3", "--warmup", "0"))

        tasks = read_csv(out_dir / "tasks.csv")
        self.assertTrue(all(t["status"] == "failed" for t in tasks))
        requests = read_csv(out_dir / "requests.csv")
        self.assertTrue(all(r["status"] == "error" for r in requests))
        self.assertIn("exceeds the available context size", requests[0]["error"])
        self.assertNotIn("error", tasks[0])
        self.assertEqual(len(tasks), 4)


class EstimateTests(unittest.TestCase):
    def test_estimate_peak_context(self):
        task = {"steps": [
            {"kind": "llm", "prompt_tokens": 512, "max_output_tokens": 128},
            {"kind": "tool", "delay_s": 1.0},
            {"kind": "llm", "prompt_tokens": 300, "max_output_tokens": 256, "carry_context": True},
        ]}
        # 512+8 prompt, +128+8 answer, +20+8 tool, +300+8 prompt, +256 output
        self.assertEqual(estimate_peak_context(task), 1248)


if __name__ == "__main__":
    unittest.main()
