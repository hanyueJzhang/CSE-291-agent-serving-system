import tempfile
import unittest
from pathlib import Path

from analysis.reporting import reprefill_figure as fig


class ReprefillFigureTest(unittest.TestCase):
    def test_collects_three_seeds_per_bar(self):
        data = fig.collect()
        for label, caches in data.items():
            for cache, runs in caches.items():
                self.assertEqual(len(runs), 3, (label, cache))
        self.assertNotIn("on", data["W1\n0.75X"])

    def test_render_writes_png_and_pdf(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = fig.render(Path(tmp))
            self.assertEqual({p.suffix for p in paths}, {".png", ".pdf"})
            self.assertTrue(all(p.stat().st_size > 0 for p in paths))


if __name__ == "__main__":
    unittest.main()
