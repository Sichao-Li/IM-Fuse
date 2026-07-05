import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from battery_fusion.experiments.classical_baselines import (
    build_composition_descriptor_frame,
    classical_model_factories,
    load_formula_vocabulary,
    regression_metrics_np,
)


class ClassicalBaselineTests(unittest.TestCase):
    def test_composition_descriptors_match_tabular_notebook_counts(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": ["mp-1", "mp-2"],
                "formula": ["LiFePO4", "NaCl"],
            }
        )

        descriptors = build_composition_descriptor_frame(frame)

        self.assertEqual(descriptors["sample_id"].tolist(), ["mp-1", "mp-2"])
        self.assertEqual(
            [column for column in descriptors.columns if column != "sample_id"],
            ["Li", "Fe", "P", "O", "Na", "Cl"],
        )
        self.assertEqual(descriptors.loc[0, "Li"], 1.0)
        self.assertEqual(descriptors.loc[0, "Fe"], 1.0)
        self.assertEqual(descriptors.loc[0, "P"], 1.0)
        self.assertEqual(descriptors.loc[0, "O"], 4.0)
        self.assertEqual(descriptors.loc[1, "Na"], 1.0)
        self.assertEqual(descriptors.loc[1, "Cl"], 1.0)

    def test_composition_descriptors_reindex_to_base_formula_vocabulary(self) -> None:
        frame = pd.DataFrame(
            {
                "sample_id": ["mp-1"],
                "formula": ["NaCl"],
            }
        )

        descriptors = build_composition_descriptor_frame(
            frame,
            base_formulas=["LiFePO4", "NaCl"],
        )

        self.assertEqual(
            [column for column in descriptors.columns if column != "sample_id"],
            ["Li", "Fe", "P", "O", "Na", "Cl"],
        )
        self.assertEqual(descriptors.loc[0, "Li"], 0.0)
        self.assertEqual(descriptors.loc[0, "Na"], 1.0)

    def test_load_formula_vocabulary_preserves_csv_order(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "raw.csv"
            pd.DataFrame(
                {
                    "id_discharge": ["mp-1", "mp-2", "mp-3"],
                    "formula_discharge": ["LiC6", "NaCl", "LiFePO4"],
                }
            ).to_csv(path, index=False)

            formulas = load_formula_vocabulary(path, "formula_discharge")

        self.assertEqual(formulas.tolist(), ["LiC6", "NaCl", "LiFePO4"])

    def test_model_factories_keep_requested_rf_and_optional_xgboost(self) -> None:
        factories = classical_model_factories(random_state=7, include_xgboost=False)

        self.assertEqual(set(factories), {"random_forest"})
        self.assertNotIn("xgboost", factories)

    def test_regression_metrics_np_reports_expected_values(self) -> None:
        metrics = regression_metrics_np(
            y_true=np.array([1.0, 2.0, 3.0]),
            y_pred=np.array([1.0, 2.5, 2.5]),
        )

        self.assertAlmostEqual(metrics["MAE"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["MSE"], 1.0 / 6.0)
        self.assertIn("R2", metrics)


if __name__ == "__main__":
    unittest.main()
