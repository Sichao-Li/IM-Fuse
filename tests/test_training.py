import unittest

import torch

from battery_fusion.training.metrics import regression_metrics


class TrainingTests(unittest.TestCase):
    def test_regression_metrics_report_mae_mse_r2(self) -> None:
        metrics = regression_metrics(
            y_true=torch.tensor([1.0, 2.0, 3.0]),
            y_pred=torch.tensor([1.0, 2.5, 2.5]),
        )

        self.assertAlmostEqual(metrics["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["mse"], 1.0 / 6.0)
        self.assertIn("r2", metrics)


if __name__ == "__main__":
    unittest.main()
