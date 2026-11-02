import unittest

from analysis.overflow import classify


class OverflowClassifyTest(unittest.TestCase):
    def test_short_ok_output_is_truncated(self):
        self.assertEqual(classify({"status": "ok", "output_tokens": "72"}, 256), "truncated")

    def test_full_ok_output_is_ok(self):
        self.assertEqual(classify({"status": "ok", "output_tokens": "256"}, 256), "ok")

    def test_server_error_keeps_its_status(self):
        self.assertEqual(classify({"status": "error", "output_tokens": "0"}, 256), "error")


if __name__ == "__main__":
    unittest.main()
