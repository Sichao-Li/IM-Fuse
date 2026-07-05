import unittest

import pandas as pd

from battery_fusion.explain.structure_ablation import summarize_atom_ablation, summarize_edge_ablation


class StructureAtomAblationTests(unittest.TestCase):
    def test_summarize_atom_ablation_groups_by_element_and_atom_index(self) -> None:
        rows = pd.DataFrame(
            {
                "sample_id": ["a", "a", "b", "b"],
                "atom_idx": [0, 1, 0, 1],
                "element": ["Li", "O", "Li", "O"],
                "prediction_delta": [1.0, 3.0, 5.0, 7.0],
                "error_delta": [0.5, 2.0, 4.0, 6.0],
            }
        )

        element_summary, site_summary = summarize_atom_ablation(rows)

        li = element_summary[element_summary["element"] == "Li"].iloc[0]
        atom_1 = site_summary[site_summary["atom_idx"] == 1].iloc[0]
        self.assertEqual(li["n_sites"], 2)
        self.assertAlmostEqual(li["prediction_delta_mean"], 3.0)
        self.assertEqual(atom_1["n_sites"], 2)
        self.assertAlmostEqual(atom_1["error_delta_mean"], 4.0)

    def test_summarize_edge_ablation_groups_by_edge_feature_dimension(self) -> None:
        rows = pd.DataFrame(
            {
                "sample_id": ["a", "b", "a", "b"],
                "edge_feature_idx": [0, 0, 1, 1],
                "prediction_delta": [1.0, 3.0, 2.0, 6.0],
                "error_delta": [0.5, 1.5, 1.0, 3.0],
            }
        )

        summary = summarize_edge_ablation(rows)

        edge_1 = summary[summary["edge_feature_idx"] == 1].iloc[0]
        self.assertEqual(edge_1["n_samples"], 2)
        self.assertAlmostEqual(edge_1["prediction_delta_mean"], 4.0)


if __name__ == "__main__":
    unittest.main()
