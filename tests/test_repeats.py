import unittest

from analysis.repeats import CONFIGS, collect, fmt


class RepeatsTest(unittest.TestCase):
    def test_fmt_shows_mean_sd_and_seeds(self):
        self.assertEqual(fmt([10.0, 20.0, 30.0]), "20.0 ± 10.0 (10.0/20.0/30.0)")

    def test_every_config_has_three_seeds_in_results(self):
        table = collect()
        self.assertEqual(set(table), set(CONFIGS))
        for metrics in table.values():
            for values in metrics.values():
                self.assertEqual(len(values), 3)


if __name__ == "__main__":
    unittest.main()
