import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from analysis.reporting.figures.capacity import render as render_capacity
from analysis.reporting.figures.concurrency import render as render_concurrency
from analysis.reporting.loader import load_manifest, load_run, load_runs, run_frame, step_frame
from analysis.reporting.style import apply_style
from analysis.reporting.validation import require_constant, select_runs


def summary(run_id="r1"):
    stats = {"n": 10, "mean": 1.0, "p50": 1.0, "p95": 2.0, "p99": 3.0, "max": 4.0}
    return {
        "run_id": run_id, "workload": "mix", "scheduler": "fifo", "concurrency": 4,
        "rate_per_s": 0.5, "seed": 1, "server_slots": 4, "ctx_per_slot": 2048,
        "cache_prompt": True, "counts": {"tasks": 10, "tasks_ok": 10, "tasks_failed": 0,
            "tasks_timeout": 0, "requests": 10, "requests_ok": 10, "requests_error": 0,
            "requests_timeout": 0},
        "makespan_s": 20.0,
        "throughput": {"tasks_per_s": 0.5, "requests_per_s": 0.5,
            "output_tokens_per_s": 50.0, "uncached_prompt_tokens_per_s": 100.0},
        "requests": {"queue_s": stats, "ttft_s": stats, "server_ttft_s": stats,
            "e2e_s": stats, "decode_ms_per_token": stats},
        "tasks": {"latency_s": stats, "latency_excl_tool_s": stats, "queue_s": stats,
            "by_type": {}},
        "prompt_cache": {"cached_ratio_all": 0.2, "cached_ratio_followup_steps": 0.3,
            "followup_steps": {"prefill_ms": stats, "uncached_tokens": stats}},
        "memory": {"gpu_mem_peak_mb": 6000, "gpu_util_mean_pct": 50,
            "server_rss_peak_mb": 800, "slot_tokens_peak": 4096,
            "active_slot_tokens_peak": 3000, "idle_slot_tokens_peak": 2000,
            "idle_slot_tokens_mean": 1000, "server_requests_deferred_max": 0},
        "by_step": {"agent/step0": {"prompt_tokens": stats, "uncached_tokens": stats,
            "prefill_ms": stats, "server_ttft_s": stats, "decode_ms_per_token": stats}},
    }


class ReportingLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "results" / "group" / "r1"
        self.run.mkdir(parents=True)
        (self.run / "summary.json").write_text(json.dumps(summary()))
        (self.run / "run_config.json").write_text(json.dumps({"dimensions": {"phase": 1}}))
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps({"version": 1, "runs": {
            "results/group/r1": {"initial_prompt_tokens": 512}
        }}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_merges_artifacts_and_builds_normalized_frames(self):
        runs = load_runs(["results/group/r1"], manifest_path=self.manifest_path, root=self.root)
        frame = run_frame(runs)
        steps = step_frame(runs)
        self.assertEqual(frame.loc[0, "initial_prompt_tokens"], 512)
        self.assertEqual(frame.loc[0, "phase"], 1)
        self.assertEqual(frame.loc[0, "task_latency_p95_s"], 2.0)
        self.assertEqual(frame.loc[0, "task_queue_p95_s"], 2.0)
        self.assertEqual(frame.loc[0, "slot_occupancy_ratio"], 0.5)
        self.assertEqual((steps.loc[0, "task_type"], steps.loc[0, "step_idx"]), ("agent", 0))

    def test_conflicting_metadata_is_rejected(self):
        manifest = load_manifest(self.manifest_path, root=self.root)
        manifest["results/group/r1"]["phase"] = 2
        with self.assertRaisesRegex(ValueError, "dimension 'phase' conflicts"):
            load_run("results/group/r1", manifest=manifest, root=self.root)


class ReportingValidationTests(unittest.TestCase):
    def test_selection_and_constant_controls(self):
        frame = pd.DataFrame([
            {"policy": "fifo", "rate": 0.5},
            {"policy": "fifo", "rate": 1.0},
        ])
        require_constant(frame, ["policy"])
        self.assertEqual(len(select_runs(frame, {"rate": [0.5]})), 1)
        with self.assertRaisesRegex(ValueError, "controlled variables differ"):
            require_constant(frame, ["rate"])

    def test_capacity_figure_smoke(self):
        frame = pd.DataFrame([
            {"rate_per_s": 0.5, "tasks_per_s": 0.48, "task_latency_p50_s": 2.0,
             "task_latency_p95_s": 4.0, "task_queue_p95_s": 1.0, "policy": "fifo"},
            {"rate_per_s": 1.0, "tasks_per_s": 0.8, "task_latency_p50_s": 5.0,
             "task_latency_p95_s": 12.0, "task_queue_p95_s": 8.0, "policy": "fifo"},
        ])
        apply_style()
        with tempfile.TemporaryDirectory() as tmp:
            paths = render_capacity(
                frame,
                {"id": "capacity", "title": "Capacity", "require_constant": ["policy"]},
                Path(tmp) / "capacity",
                ["png"],
            )
            self.assertEqual(len(paths), 1)
            self.assertGreater(paths[0].stat().st_size, 0)

    def test_concurrency_figure_smoke(self):
        frame = pd.DataFrame([
            {"rate_per_s": rate, "concurrency": concurrency, "server_slots": 4,
             "tasks_per_s": throughput, "task_latency_p95_s": latency,
             "server_requests_deferred_max": deferred, "policy": "fifo"}
            for rate, concurrency, throughput, latency, deferred in [
                (0.5, 1, 0.4, 40.0, 0),
                (0.5, 4, 0.49, 10.0, 0),
                (0.5, 8, 0.49, 10.0, 2),
                (1.0, 1, 0.5, 100.0, 0),
                (1.0, 4, 0.85, 25.0, 0),
                (1.0, 8, 0.85, 25.0, 4),
            ]
        ])
        apply_style()
        with tempfile.TemporaryDirectory() as tmp:
            paths = render_concurrency(
                frame,
                {"id": "concurrency", "title": "Concurrency",
                 "require_constant": ["policy", "server_slots"]},
                Path(tmp) / "concurrency",
                ["png"],
            )
            self.assertEqual(len(paths), 1)
            self.assertGreater(paths[0].stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
