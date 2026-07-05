import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from battery_fusion.experiments.chemistry_splits import create_anion_holdout_splits


class ChemistrySplitTests(unittest.TestCase):
    def test_holdout_family_never_appears_in_train_or_val(self) -> None:
        assignments = pd.DataFrame(
            {
                "sample_id": [f"mp-{i}" for i in range(8)],
                "anion_family": ["halide", "halide", "oxide", "oxide", "oxide", "sulfide", "sulfide", "other"],
                "target": list(range(8)),
            }
        )
        with TemporaryDirectory() as tmp:
            result = create_anion_holdout_splits(
                assignments=assignments,
                heldout_family="halide",
                output_dir=Path(tmp),
                seeds=[0],
                min_test_samples=2,
                val_ratio=0.25,
            )

            train = pd.read_csv(result[0]["train"])
            val = pd.read_csv(result[0]["val"])
            test = pd.read_csv(result[0]["test"])
            self.assertNotIn("halide", set(train["anion_family"]))
            self.assertNotIn("halide", set(val["anion_family"]))
            self.assertEqual(set(test["anion_family"]), {"halide"})


if __name__ == "__main__":
    unittest.main()
