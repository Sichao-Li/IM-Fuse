import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from battery_fusion.experiments.ood_splits import (
    assign_composition_clusters,
    create_composition_cluster_holdout_splits,
    create_working_ion_holdout_splits,
)


class OodSplitTests(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "sample_id": [f"mp-{i}" for i in range(12)],
                "formula": [
                    "LiFePO4",
                    "NaFePO4",
                    "KFePO4",
                    "LiCoO2",
                    "NaCoO2",
                    "MgCoO2",
                    "CaMnO3",
                    "ZnMnO3",
                    "Li2S",
                    "Na2S",
                    "MgF2",
                    "CaF2",
                ],
                "working_ion": [
                    "Li",
                    "Na",
                    "K",
                    "Li",
                    "Na",
                    "Mg",
                    "Ca",
                    "Zn",
                    "Li",
                    "Na",
                    "Mg",
                    "Ca",
                ],
                "anion_family": [
                    "phosphate_or_polyanion",
                    "phosphate_or_polyanion",
                    "phosphate_or_polyanion",
                    "oxide",
                    "oxide",
                    "oxide",
                    "oxide",
                    "oxide",
                    "sulfide",
                    "sulfide",
                    "halide",
                    "halide",
                ],
                "target": [float(i) for i in range(12)],
            }
        )

    def test_composition_clusters_do_not_use_target_values(self) -> None:
        frame = self._frame()
        changed_targets = frame.copy()
        changed_targets["target"] = list(reversed(frame["target"].tolist()))

        first = assign_composition_clusters(frame, n_clusters=3, seed=7)
        second = assign_composition_clusters(changed_targets, n_clusters=3, seed=7)

        pd.testing.assert_series_equal(
            first["composition_cluster"],
            second["composition_cluster"],
            check_names=False,
        )

    def test_composition_cluster_holdout_excludes_cluster_from_train_and_val(self) -> None:
        frame = self._frame()
        with TemporaryDirectory() as tmp:
            result = create_composition_cluster_holdout_splits(
                frame,
                output_dir=Path(tmp),
                seeds=[0],
                n_clusters=3,
                min_test_size=1,
                val_ratio=0.25,
            )

            self.assertGreaterEqual(len(result.created), 1)
            split = result.created[0]
            train = pd.read_csv(split["train"])
            val = pd.read_csv(split["val"])
            test = pd.read_csv(split["test"])
            heldout = int(test["composition_cluster"].iloc[0])

            self.assertEqual(set(test["composition_cluster"]), {heldout})
            self.assertNotIn(heldout, set(train["composition_cluster"]))
            self.assertNotIn(heldout, set(val["composition_cluster"]))

    def test_composition_cluster_membership_is_stable_across_model_seeds(self) -> None:
        frame = self._frame()
        with TemporaryDirectory() as tmp:
            create_composition_cluster_holdout_splits(
                frame,
                output_dir=Path(tmp),
                seeds=[0, 1],
                n_clusters=3,
                cluster_seed=11,
                min_test_size=1,
                val_ratio=0.25,
            )

            seed0 = pd.read_csv(Path(tmp) / "cluster_0" / "seed_0" / "test.csv")
            seed1 = pd.read_csv(Path(tmp) / "cluster_0" / "seed_1" / "test.csv")
            self.assertEqual(set(seed0["sample_id"]), set(seed1["sample_id"]))

    def test_na_working_ion_holdout_excludes_na_from_train_and_val(self) -> None:
        with TemporaryDirectory() as tmp:
            result = create_working_ion_holdout_splits(
                self._frame(),
                heldout_ions=["Na"],
                output_dir=Path(tmp),
                seeds=[0],
                min_test_size=1,
                val_ratio=0.25,
            )

            train = pd.read_csv(result[0]["train"])
            val = pd.read_csv(result[0]["val"])
            test = pd.read_csv(result[0]["test"])
            self.assertEqual(set(test["working_ion"]), {"Na"})
            self.assertNotIn("Na", set(train["working_ion"]))
            self.assertNotIn("Na", set(val["working_ion"]))

    def test_multivalent_working_ion_holdout_excludes_all_requested_ions(self) -> None:
        heldout = {"Mg", "Ca", "Zn"}
        with TemporaryDirectory() as tmp:
            result = create_working_ion_holdout_splits(
                self._frame(),
                heldout_ions=sorted(heldout),
                output_dir=Path(tmp),
                seeds=[0],
                min_test_size=1,
                val_ratio=0.25,
            )

            train = pd.read_csv(result[0]["train"])
            val = pd.read_csv(result[0]["val"])
            test = pd.read_csv(result[0]["test"])
            self.assertEqual(set(test["working_ion"]), heldout)
            self.assertTrue(heldout.isdisjoint(set(train["working_ion"])))
            self.assertTrue(heldout.isdisjoint(set(val["working_ion"])))


if __name__ == "__main__":
    unittest.main()
