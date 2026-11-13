import unittest

from analysis.server_overhead import non_compute_ms


class NonComputeTest(unittest.TestCase):
    def test_subtracts_prefill_and_decode_from_time_in_server(self):
        row = {"t_dispatch": "10.0", "t_finish": "12.5", "prefill_ms": "300", "decode_ms": "1900"}
        self.assertAlmostEqual(non_compute_ms(row), 300.0)


if __name__ == "__main__":
    unittest.main()
