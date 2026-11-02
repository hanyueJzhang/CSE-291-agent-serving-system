import unittest

from analysis.reprefill import avoidable_ms


def _req(prompt, cached, output, prefill_ms=0.0):
    return {"prompt_tokens": str(prompt), "cached_tokens": str(cached),
            "output_tokens": str(output), "prefill_ms": str(prefill_ms)}


class AvoidableReprefillTest(unittest.TestCase):
    def test_full_hit_has_no_avoidable_prefill(self):
        prev = _req(800, 0, 100)
        cur = _req(1200, 900, 128, prefill_ms=60.0)  # the 300 uncached tokens are all new
        self.assertEqual(avoidable_ms(prev, cur), (0.0, 0))

    def test_displaced_history_is_attributed_proportionally(self):
        prev = _req(800, 0, 100)
        cur = _req(1200, 0, 128, prefill_ms=120.0)  # 900 history + 300 new, nothing cached
        self.assertEqual(avoidable_ms(prev, cur), (90.0, 900))


if __name__ == "__main__":
    unittest.main()
