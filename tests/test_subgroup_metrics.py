import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import pandas as pd

from battery_fusion.experiments.subgroups import compute_subgroup_metrics, load_prediction_files


class SubgroupMetricTests(unittest.TestCase):
    def test_small_group_is_marked_unreliable(self) -> None:
        predictions = pd.DataFrame(
            {
                "sample_id": ["a", "b", "c"],
                "anion_family": ["oxide", "oxide", "halide"],
                "working_ion": ["Li", "Li", "Na"],
                "y_true": [1.0, 2.0, 4.0],
                "y_pred": [1.5, 2.5, 5.0],
                "model_name": ["m"] * 3,
                "modality_set": ["tabular"] * 3,
                "seed": [0] * 3,
            }
        )

        metrics = compute_subgroup_metrics(predictions, min_group_size=2)
        oxide = metrics[
            (metrics["group_type"] == "anion_family")
            & (metrics["group_name"] == "oxide")
        ].iloc[0]
        halide = metrics[
            (metrics["group_type"] == "anion_family")
            & (metrics["group_name"] == "halide")
        ].iloc[0]
        self.assertFalse(bool(oxide["unreliable"]))
        self.assertTrue(bool(halide["unreliable"]))
        self.assertEqual(halide["n_samples"], 1)

    def test_load_prediction_files_filters_split_across_roots(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "neural"
            second = root / "baselines"
            first.mkdir()
            second.mkdir()
            pd.DataFrame(
                {
                    "sample_id": ["a", "b"],
                    "split": ["train", "test"],
                    "y_true": [1.0, 2.0],
                    "y_pred": [1.1, 2.2],
                }
            ).to_csv(first / "seed_0_predictions.csv", index=False)
            pd.DataFrame(
                {
                    "sample_id": ["c", "d"],
                    "split": ["val", "test"],
                    "y_true": [3.0, 4.0],
                    "y_pred": [3.3, 4.4],
                }
            ).to_csv(second / "seed_0_predictions.csv", index=False)

            loaded = load_prediction_files([first, second], split="test")

        self.assertEqual(loaded["sample_id"].tolist(), ["b", "d"])
        self.assertEqual(loaded["split"].unique().tolist(), ["test"])


if __name__ == "__main__":
    unittest.main()
