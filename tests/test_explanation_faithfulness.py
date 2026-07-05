import unittest

import numpy as np
import pandas as pd

from battery_fusion.explain.faithfulness import (
    ablation_indices,
    ablation_schedule,
    summarize_faithfulness,
)


class ExplanationFaithfulnessTests(unittest.TestCase):
    def test_ablation_schedule_keeps_at_least_one_feature_for_positive_fraction(self) -> None:
        self.assertEqual(ablation_schedule(10, [0.0, 0.05, 0.2, 1.0]), [0, 1, 2, 10])

    def test_ablation_indices_orders_top_bottom_and_random_reproducibly(self) -> None:
        importance = np.array([0.0, -4.0, 2.0, 7.0, -1.0])

        self.assertEqual(ablation_indices(importance, "top", 3, seed=0).tolist(), [3, 1, 2])
        self.assertEqual(ablation_indices(importance, "bottom", 3, seed=0).tolist(), [0, 4, 2])
        self.assertEqual(
            ablation_indices(importance, "random", 5, seed=11).tolist(),
            ablation_indices(importance, "random", 5, seed=11).tolist(),
        )

    def test_summary_reports_auc_and_top_minus_random_delta(self) -> None:
        curves = pd.DataFrame(
            {
                "sample_id": ["a", "a", "a", "a", "a", "a"],
                "order": ["top", "top", "random", "random", "bottom", "bottom"],
                "ablation_fraction": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                "prediction_delta": [0.0, 4.0, 0.0, 2.0, 0.0, 1.0],
                "error_delta": [0.0, 3.0, 0.0, 1.0, 0.0, -1.0],
            }
        )

        summary = summarize_faithfulness(curves)

        top = summary[summary["order"] == "top"].iloc[0]
        random = summary[summary["order"] == "random"].iloc[0]
        self.assertAlmostEqual(top["prediction_delta_auc_mean"], 2.0)
        self.assertAlmostEqual(random["prediction_delta_auc_mean"], 1.0)
        self.assertAlmostEqual(top["prediction_auc_minus_random"], 1.0)


if __name__ == "__main__":
    unittest.main()
