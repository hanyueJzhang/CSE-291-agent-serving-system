import csv
import json
import tempfile
import unittest
from pathlib import Path

from analysis import paired
from workload.metrics import REQUEST_COLUMNS, TASK_COLUMNS


def write(root: Path, name: str, policy: str, seed: int, queue: float) -> Path:
    """Two 2-step tasks; follow-ups wait `queue` seconds."""
    run = root / name
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps(
        {"dimensions": {"policy": policy, "seed": seed, "experiment": "x", "load_pct_x": 95}}))
    reqs, tasks = [], []
    for i in range(2):
        tid = f"t{i}"
        t0 = 10.0 * i
        f_arrive = t0 + 2.0
        f_dispatch = f_arrive + queue
        reqs.append({"task_id": tid, "step_idx": 0, "t_arrive": t0, "t_dispatch": t0, "t_first_token": t0 + 0.5,
                     "t_finish": t0 + 1.0, "prompt_tokens": 100, "cached_tokens": 0, "output_tokens": 10,
                     "prefill_ms": 100, "status": "ok"})
        reqs.append({"task_id": tid, "step_idx": 2, "t_arrive": f_arrive, "t_dispatch": f_dispatch,
                     "t_first_token": f_dispatch + 0.1, "t_finish": f_dispatch + 1.0, "prompt_tokens": 200,
                     "cached_tokens": 110, "output_tokens": 10, "prefill_ms": 100, "status": "ok"})
        tasks.append({"task_id": tid, "task_type": "a", "t_arrive": t0, "t_finish": f_dispatch + 1.0,
                      "num_steps": 3, "status": "ok"})
    for fname, cols, rows in (("requests.csv", REQUEST_COLUMNS, reqs), ("tasks.csv", TASK_COLUMNS, tasks)):
        with (run / fname).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows:
                w.writerow({"run_id": name, **r})
    return run


class PairedTests(unittest.TestCase):
    def test_pairs_by_seed_and_reports_relative_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed, (qb, qp) in {42: (4.0, 2.0), 43: (8.0, 2.0)}.items():
                write(root, f"base_{seed}", "fifo", seed, qb)
                write(root, f"cf_{seed}", "cf", seed, qp)
            write(root, "cf_44", "cf", 44, 1.0)      # no baseline for seed 44
            result = paired.compare(paired.find_runs([root]), baseline="fifo")

        cf = result["policies"]["cf"]
        self.assertEqual(cf["n"], 2)
        q = cf["summary"]["followup_queue_mean_s"]
        self.assertEqual(sorted(q["per_instance_rel_pct"]), [-75.0, -50.0])
        self.assertAlmostEqual(q["mean_rel_pct"], -62.5)
        self.assertEqual(cf["summary"]["followup_cached_pct"]["mean_value"], 55.0)
        self.assertEqual(cf["summary"]["tasks_not_ok"]["mean_value"], 0.0)
        self.assertTrue(any("cf_44" in s for s in result["skipped"]))


if __name__ == "__main__":
    unittest.main()
