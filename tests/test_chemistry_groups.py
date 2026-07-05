import unittest

import pandas as pd

from battery_fusion.utils.chemistry_groups import (
    assign_anion_family,
    assign_chemistry_groups,
    normalize_working_ion,
)


class ChemistryGroupTests(unittest.TestCase):
    def test_assign_anion_family_uses_transparent_rules(self) -> None:
        self.assertEqual(assign_anion_family("LiCoO2"), "oxide")
        self.assertEqual(assign_anion_family("LiFePO4"), "phosphate_or_polyanion")
        self.assertEqual(assign_anion_family("Na2S"), "sulfide")
        self.assertEqual(assign_anion_family("LiCl"), "halide")
        self.assertEqual(assign_anion_family("LiC6"), "other")

    def test_normalize_working_ion_keeps_common_ions(self) -> None:
        self.assertEqual(normalize_working_ion("Li"), "Li")
        self.assertEqual(normalize_working_ion("Fe"), "other")
        self.assertEqual(normalize_working_ion(None), "other")

    def test_assign_chemistry_groups_emits_expected_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "id_discharge": ["mp-1"],
                "formula_discharge": ["LiFePO4"],
                "working_ion": ["Li"],
                "target": [3.5],
            }
        )

        grouped = assign_chemistry_groups(
            frame,
            sample_id_col="id_discharge",
            formula_col="formula_discharge",
            target_col="target",
            working_ion_col="working_ion",
        )

        self.assertEqual(
            list(grouped.columns),
            ["sample_id", "formula", "working_ion", "anion_family", "target"],
        )
        self.assertEqual(grouped.loc[0, "anion_family"], "phosphate_or_polyanion")


if __name__ == "__main__":
    unittest.main()
