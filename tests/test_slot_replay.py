import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis import slot_gaps, slot_replay
from analysis.slot_trace import load_run
from workload.metrics import REQUEST_COLUMNS

# Two slots. Agent A pauses for a tool between t=1 and t=3; single-step task C
# finishes at t=2; new task D arrives at t=2.5 when both slots are free.
# LRU (and llama.cpp, no prefix match) gives D A's slot (used longer ago),
# so A's follow-up loses its 110-token history. Dead-first gives D C's slot.
ROWS = [
    # task, step, arrive, dispatch, first, finish, prompt, cached, output, prefill_ms, slot
    ("A", 0, 0.0, 0.0, 0.1, 1.0, 100, 0, 10, 100.0, 0),
    ("C", 0, 0.0, 0.0, 0.1, 2.0, 50, 0, 10, 100.0, 1),
    ("D", 0, 2.5, 2.5, 2.6, 5.0, 80, 0, 10, 100.0, 0),
    ("A", 1, 3.0, 3.0, 3.2, 4.0, 210, 0, 10, 200.0, 1),
]


def write_run(root: Path, rows=ROWS, slots=2, with_slot_id=True) -> Path:
    run = root / "run"
    run.mkdir()
    (run / "run_config.json").write_text(json.dumps({
        "server": {"total_slots": slots}, "scheduler": {"name": "fifo", "max_concurrency": slots},
    }))
    with (run / "requests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQUEST_COLUMNS)
        w.writeheader()
        for task, step, arr, disp, first, fin, prompt, cached, out, pre, slot in rows:
            w.writerow({
                "run_id": "run", "task_id": task, "step_idx": step, "t_arrive": arr,
                "t_dispatch": disp, "t_first_token": first, "t_finish": fin,
                "prompt_tokens": prompt, "cached_tokens": cached, "output_tokens": out,
                "prefill_ms": pre, "decode_ms": 1.0, "status": "ok", "error": "",
                "slot_id": slot if with_slot_id else "",
            })
    return run


class SlotReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_trace_links_steps_and_estimates_start(self):
        run = load_run(write_run(self.root))
        a1 = next(r for r in run.requests if r.task_id == "A" and r.step_idx == 1)
        self.assertEqual(a1.carried, 110)
        self.assertTrue(a1.is_followup)
        self.assertAlmostEqual(a1.start, 3.0)          # max(dispatch, first - prefill)
        self.assertTrue(run.has_slots)

    def test_dead_first_keeps_history_lru_loses_it(self):
        rows = {r["policy"]: r for r in slot_replay.analyze(write_run(self.root), holds=[1.0])}
        self.assertEqual(rows["lru"]["history_kept_pct"], 0.0)
        self.assertEqual(rows["llama"]["history_kept_pct"], 0.0)
        self.assertEqual(rows["dead_first"]["history_kept_pct"], 100.0)
        self.assertEqual(rows["dead_belady"]["history_kept_pct"], 100.0)
        self.assertEqual(rows["lru"]["live_overwrites"], 1)
        self.assertEqual(rows["dead_first"]["live_overwrites"], 0)
        # The observed assignment reproduces what the server measured (cached 0).
        self.assertEqual(rows["observed"]["history_kept_pct"], 0.0)
        self.assertEqual(rows["observed"]["measured_history_kept_pct"], 0.0)
        self.assertEqual(rows["lru"]["same_slot_as_observed_pct"], 100.0)

    def test_hold_counts_returning_owner(self):
        # One slot only: D must overwrite A's live context; A returns 0.5 s later.
        rows = [
            ("A", 0, 0.0, 0.0, 0.1, 1.0, 100, 0, 10, 100.0, 0),
            ("D", 0, 2.5, 2.5, 2.6, 5.0, 80, 0, 10, 100.0, 0),
            ("A", 1, 3.0, 5.0, 5.2, 6.0, 210, 0, 10, 200.0, 0),
        ]
        out = slot_replay.analyze(write_run(self.root, rows, slots=1), holds=[0.1, 1.0])
        belady = next(r for r in out if r["policy"] == "dead_belady")
        self.assertEqual(belady["live_overwrites"], 1)
        tight, loose = belady["hold"]
        self.assertEqual(tight["holds"], 0)
        self.assertEqual(loose["holds"], 1)
        self.assertEqual(loose["tokens_saved"], 110)
        self.assertAlmostEqual(loose["slot_s_idle"], 0.5)

    def test_runs_without_slot_id_skip_observed(self):
        out = slot_replay.analyze(write_run(self.root, with_slot_id=False))
        self.assertNotIn("observed", {r["policy"] for r in out})
        self.assertNotIn("same_slot_as_observed_pct", out[0])

    def test_gaps(self):
        row = slot_gaps.analyze(write_run(self.root))
        self.assertTrue(row["has_slot_id"])
        self.assertEqual(row["follow_ups_on_own_slot_pct"], 0.0)
        # No request was ever ready while a slot sat idle.
        self.assertAlmostEqual(row["idle_slot_s_while_ready"], 0.0)
        self.assertGreater(row["slot_busy_pct"], 0)

    def test_idle_while_ready_counts_dispatch_gap(self):
        # B is ready at 1.0 when A's slot frees, but only gets it at 1.2.
        rows = [
            ("A", 0, 0.0, 0.0, 0.1, 1.0, 100, 0, 10, 100.0, 0),
            ("B", 0, 0.5, 1.2, 1.3, 2.0, 100, 0, 10, 100.0, 0),
        ]
        row = slot_gaps.analyze(write_run(self.root, rows, slots=1))
        self.assertAlmostEqual(row["idle_slot_s_while_ready"], 0.2)
        self.assertAlmostEqual(row["handoff_gap_p50_ms"], 200.0)


if __name__ == "__main__":
    unittest.main()
