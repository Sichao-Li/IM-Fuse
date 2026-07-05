import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from pymatgen.core import Lattice, Structure

from battery_fusion.features.rdf import build_rdf_vector
from battery_fusion.features.structure import build_crystal_graph
from battery_fusion.features.tabular import formula_vector, vocabulary_from_formulas


class FeatureBuilderTests(unittest.TestCase):
    def test_formula_vector_uses_shared_vocabulary(self) -> None:
        vocab = vocabulary_from_formulas(["LiFePO4", "NaFePO4"])
        vector = formula_vector("LiFePO4", vocab)

        self.assertEqual(vocab, ["Fe", "Li", "Na", "O", "P"])
        np.testing.assert_allclose(vector, np.array([1.0, 1.0, 0.0, 4.0, 1.0]))

    def test_rdf_vector_has_fixed_length(self) -> None:
        structure = Structure(
            Lattice.cubic(4.0),
            ["Li", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )

        rdf = build_rdf_vector(structure, bins=16, cutoff=8.0)

        self.assertEqual(rdf.shape, (16,))
        self.assertTrue(np.isfinite(rdf).all())

    def test_rdf_vector_uses_legacy_raw_rdfpy_scale(self) -> None:
        structure = Structure(
            Lattice.cubic(4.0),
            ["Li", "O"],
            [[0, 0, 0], [0.5, 0.5, 0.5]],
        )

        rdf = build_rdf_vector(structure, bins=400, cutoff=20.0, noise_std=0.0)

        self.assertEqual(rdf.shape, (400,))
        self.assertTrue(np.isfinite(rdf).all())
        self.assertGreater(float(rdf.sum()), 10.0)

    def test_crystal_graph_contains_atom_and_neighbor_features(self) -> None:
        with TemporaryDirectory() as tmp:
            atom_init = Path(tmp) / "atom_init.json"
            atom_init.write_text('{"3": [1.0, 0.0], "8": [0.0, 1.0]}')
            structure = Structure(
                Lattice.cubic(4.0),
                ["Li", "O"],
                [[0, 0, 0], [0.5, 0.5, 0.5]],
            )

            graph = build_crystal_graph(
                structure=structure,
                atom_init_path=atom_init,
                radius=6.0,
                max_neighbors=4,
                gaussian_centers=np.linspace(0, 6, 8),
                gaussian_width=0.5,
            )

            self.assertEqual(graph["atom_fea"].shape, (2, 2))
            self.assertEqual(graph["nbr_fea"].shape, (2, 4, 8))
            self.assertEqual(graph["nbr_fea_idx"].shape, (2, 4))
            self.assertTrue(np.isfinite(graph["nbr_fea"]).all())


if __name__ == "__main__":
    unittest.main()
