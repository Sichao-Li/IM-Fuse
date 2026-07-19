import unittest
from pathlib import Path

from battery_fusion.release_check import run_release_check


class ReleaseCheckTests(unittest.TestCase):
    def test_tracked_publication_data_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report = run_release_check(root)

        self.assertEqual(report.errors, [])
        self.assertEqual(report.split_summary["average_voltage"]["n_samples"], 8088)
        self.assertEqual(report.split_summary["capacity_vol"]["n_seeds"], 5)
        self.assertTrue(report.warnings)  # Full rerun artifacts are intentionally external.


if __name__ == "__main__":
    unittest.main()
