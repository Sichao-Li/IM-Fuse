import unittest

import torch

from battery_fusion.models.mlp import MLPRegressor
from battery_fusion.models.lstm import RdfLSTMRegressor
from battery_fusion.fusion.models import (
    EarlyFusionRegressor,
    LateFeatureFusionRegressor,
    LateFusionRegressor,
)


def cgcnn_graph_batch(
    batch_size: int = 2,
    atoms_per_crystal: int = 2,
    atom_dim: int = 4,
    nbr_dim: int = 3,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    n_atoms = batch_size * atoms_per_crystal
    atom_fea = torch.ones(n_atoms, atom_dim)
    nbr_fea = torch.ones(n_atoms, 2, nbr_dim)
    nbr_fea_idx = torch.tensor([[idx, (idx + 1) % n_atoms] for idx in range(n_atoms)], dtype=torch.long)
    crystal_atom_idx = [
        torch.arange(start, start + atoms_per_crystal)
        for start in range(0, n_atoms, atoms_per_crystal)
    ]
    return atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx


class ModelTests(unittest.TestCase):
    def test_mlp_regressor_forward_shape(self) -> None:
        model = MLPRegressor(input_dim=4, hidden_dim=8)
        output = model(torch.ones(3, 4))
        self.assertEqual(tuple(output.shape), (3,))

    def test_rdf_lstm_forward_shape(self) -> None:
        model = RdfLSTMRegressor(input_size=400, hidden_size=8, output_size=1)
        output = model(torch.ones(3, 400))
        self.assertEqual(tuple(output.shape), (3,))

    def test_rdf_lstm_uses_publication_400_feature_sequence_encoder(self) -> None:
        model = RdfLSTMRegressor(input_size=400, hidden_size=8, output_size=1)

        self.assertEqual(model.lstm_cell.input_size, 400)
        self.assertEqual(model.lstm_cell.hidden_size, 8)

    def test_early_fusion_accepts_dual_modalities(self) -> None:
        model = EarlyFusionRegressor(input_dims={"rdf": 10, "tabular": 4}, hidden_dim=8)
        batch = {"rdf": torch.ones(3, 10), "tabular": torch.ones(3, 4)}
        output = model(batch)
        self.assertEqual(tuple(output.shape), (3,))

    def test_late_fusion_learns_weighted_prediction_shape(self) -> None:
        model = LateFusionRegressor(n_modalities=2)
        predictions = torch.ones(5, 2)
        output = model(predictions)
        self.assertEqual(tuple(output.shape), (5,))

    def test_late_feature_fusion_accepts_tri_modalities(self) -> None:
        model = LateFeatureFusionRegressor(
            input_dims={"rdf": 10, "structure": 3, "tabular": 4},
            hidden_dim=8,
        )
        batch = {
            "rdf": torch.ones(3, 10),
            "structure": torch.ones(3, 3),
            "tabular": torch.ones(3, 4),
        }
        output = model(batch)
        self.assertEqual(tuple(output.shape), (3,))


if __name__ == "__main__":
    unittest.main()
