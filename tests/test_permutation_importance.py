import unittest

import numpy as np
import pandas as pd

from battery_fusion.explain.permutation import (
    build_feature_groups,
    metric_delta_rows,
    permuted_matrix,
)


class PermutationImportanceTests(unittest.TestCase):
    def test_build_feature_groups_uses_contiguous_windows(self) -> None:
        groups = build_feature_groups(5, group_size=2, prefix="rdf")

        self.assertEqual(
            groups,
            [
                ("rdf_0_1", [0, 1]),
                ("rdf_2_3", [2, 3]),
                ("rdf_4_4", [4]),
            ],
        )

    def test_permuted_matrix_shuffles_only_selected_group_columns(self) -> None:
        matrix = np.array([[1, 10, 100], [2, 20, 200], [3, 30, 300]], dtype=float)

        permuted = permuted_matrix(matrix, columns=[0, 2], seed=0)

        np.testing.assert_array_equal(permuted[:, 1], matrix[:, 1])
        self.assertCountEqual(permuted[:, 0].tolist(), matrix[:, 0].tolist())
        self.assertCountEqual(permuted[:, 2].tolist(), matrix[:, 2].tolist())
        self.assertFalse(np.array_equal(permuted[:, [0, 2]], matrix[:, [0, 2]]))

    def test_metric_delta_rows_compare_permuted_metrics_to_baseline(self) -> None:
        rows = metric_delta_rows(
            baseline={"mae": 1.0, "mse": 2.0, "rmse": 1.4, "r2": 0.8},
            permuted={"mae": 1.5, "mse": 3.0, "rmse": 1.7, "r2": 0.6},
            metadata={"feature_group": "rdf_0_9"},
        )

        frame = pd.DataFrame(rows)
        self.assertEqual(frame.loc[0, "feature_group"], "rdf_0_9")
        self.assertAlmostEqual(frame.loc[0, "delta_mae"], 0.5)
        self.assertAlmostEqual(frame.loc[0, "delta_r2"], -0.2)


if __name__ == "__main__":
    unittest.main()
