import csv
import tempfile
import unittest
from pathlib import Path

from analysis import prefill_stall
from analysis.prefill_stall import Sample
from workload.metrics import REQUEST_COLUMNS

# Two slots. A decodes 100 tokens over t = 1..5 and B prefills in t = 2..3, so
# one of A's four decode seconds overlaps a prefill. C decodes alone at t = 10..11.
ROWS = [
    # task, arrive, first token, finish, prompt, cached, output, prefill_ms, decode_ms
    ("A", 0.0, 1.0, 5.0, 1000, 0, 100, 1000.0, 4000.0),
    ("B", 2.0, 3.0, 5.0, 2000, 1000, 100, 1000.0, 2000.0),
    ("C", 9.5, 10.0, 11.0, 500, 0, 100, 500.0, 1000.0),
]


def write_run(root: Path) -> Path:
    run = root / "run"
    run.mkdir()
    with (run / "requests.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REQUEST_COLUMNS)
        w.writeheader()
        for task, arrive, first, finish, prompt, cached, output, prefill_ms, decode_ms in ROWS:
            w.writerow({
                "run_id": "run", "task_id": task, "step_idx": 0, "t_arrive": arrive, "t_dispatch": arrive,
                "t_first_token": first, "t_finish": finish, "prompt_tokens": prompt, "cached_tokens": cached,
                "output_tokens": output, "prefill_ms": prefill_ms, "decode_ms": decode_ms, "status": "ok",
            })
    return run


class PrefillStallTests(unittest.TestCase):
    def test_overlap_and_busy_slots_per_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples, prefill_rate = prefill_stall.load_samples(write_run(Path(tmp)))
        a, b, c = samples
        self.assertAlmostEqual(a.overlap_s, 1.0)            # B's prefill
        self.assertAlmostEqual(a.n_busy, 1 + 3 / 4)         # B holds a slot for 3 of A's 4 s
        self.assertAlmostEqual(b.overlap_s, 0.0)
        self.assertAlmostEqual(b.n_busy, 2.0)
        self.assertAlmostEqual(c.n_busy, 1.0)
        self.assertAlmostEqual(prefill_rate, 2500 / 2.5)    # uncached tokens / prefill seconds
        self.assertEqual(prefill_stall.stall_free_ms(samples), {1: (10.0, 1), 2: (20.0, 1)})

    def test_fit_recovers_the_stall(self):
        # decode_s = out * (8 + 2 * n) ms + 0.9 * overlap
        samples = [Sample(out, n, ov, out * (0.008 + 0.002 * n) + 0.9 * ov, 1.0)
                   for out, n, ov in [(100, 1, 0.0), (200, 2, 0.5), (300, 4, 0.0), (150, 3, 2.0), (250, 2, 1.0)]]
        result = prefill_stall.fit(samples)
        self.assertAlmostEqual(result["stall"], 0.9, places=6)
        self.assertAlmostEqual(result["base_ms_per_token"], 8.0, places=6)
        self.assertAlmostEqual(result["ms_per_token_per_busy_slot"], 2.0, places=6)

    def test_break_even_hold(self):
        ms = {3: 12.0, 4: 12.5}             # 250 -> 320 tokens/s: the fourth slot adds 70
        hold = prefill_stall.break_even_s(ms, 4, history=2000, prefill_rate=4000, stall=1.0)
        self.assertAlmostEqual(hold, 0.5 * 320 / 70)        # 0.5 s of prefill stops all four slots
        self.assertIsNone(prefill_stall.break_even_s(ms, 3, 2000, 4000, 1.0))


if __name__ == "__main__":
    unittest.main()
