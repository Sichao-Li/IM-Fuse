import unittest

import torch
from torch.utils.data import DataLoader, TensorDataset

from battery_fusion.models.mlp import MLPRegressor
from battery_fusion.training.metrics import regression_metrics
from battery_fusion.training.runner import train_regressor


class TrainingTests(unittest.TestCase):
    def test_regression_metrics_report_mae_mse_r2(self) -> None:
        metrics = regression_metrics(
            y_true=torch.tensor([1.0, 2.0, 3.0]),
            y_pred=torch.tensor([1.0, 2.5, 2.5]),
        )

        self.assertAlmostEqual(metrics["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["mse"], 1.0 / 6.0)
        self.assertIn("r2", metrics)

    def test_train_regressor_returns_metrics(self) -> None:
        x = torch.arange(12, dtype=torch.float32).reshape(6, 2)
        y = x.sum(dim=1)
        loader = DataLoader(TensorDataset(x, y), batch_size=3, shuffle=False)
        model = MLPRegressor(input_dim=2, hidden_dim=8, dropout=0.0)

        history = train_regressor(model, loader, loader, epochs=1, learning_rate=1e-3)

        self.assertEqual(len(history), 1)
        self.assertIn("train_loss", history[0])
        self.assertIn("val_mae", history[0])


if __name__ == "__main__":
    unittest.main()
