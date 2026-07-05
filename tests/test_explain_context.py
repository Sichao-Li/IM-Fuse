import unittest

import torch
from torch import nn

from battery_fusion.explain.context import FusionContextWrapper, FusionInputWrapper
from battery_fusion.explain.results import (
    group_interactions_by_modality,
    interactions_to_frame,
    split_main_effects,
)


class RecordingFusionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_batch = None

    def forward(self, batch):
        self.last_batch = batch
        graph_inputs = batch["graph_inputs"]
        atom_fea = graph_inputs[0]
        return (
            atom_fea.sum()
            + batch["tab_inputs"].sum()
            + batch["rdf_inputs"].sum()
        ).reshape(1, 1)


class ExplainContextTests(unittest.TestCase):
    def test_build_input_preserves_publication_mapping_order_and_padding(self) -> None:
        atom_fea = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        tab = torch.tensor([[10.0, 20.0]])
        rdf = torch.tensor([[30.0, 40.0, 50.0]])

        input_tensor, mapping = FusionInputWrapper(
            perturb_modalities=("graph", "tab", "rdf")
        ).build_input_and_mapping(atom_fea=atom_fea, tab=tab, rdf=rdf)

        self.assertEqual(tuple(input_tensor.shape), (3, 7))
        self.assertEqual([entry["modality"] for entry in mapping], ["graph", "graph", "tab", "tab", "rdf", "rdf", "rdf"])
        torch.testing.assert_close(input_tensor[:, 0], torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(input_tensor[:, 2], torch.tensor([10.0, 0.0, 0.0]))
        torch.testing.assert_close(input_tensor[:, 4], torch.tensor([30.0, 0.0, 0.0]))

    def test_context_wrapper_reconstructs_modalities_from_perturbed_columns(self) -> None:
        atom_fea = torch.zeros(2, 3)
        tab = torch.zeros(1, 2)
        rdf = torch.zeros(1, 3)
        input_tensor, mapping = FusionInputWrapper(
            perturb_modalities=("graph", "tab", "rdf")
        ).build_input_and_mapping(
            atom_fea=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            tab=torch.tensor([[10.0, 20.0]]),
            rdf=torch.tensor([[30.0, 40.0, 50.0]]),
        )
        model = RecordingFusionModel()
        wrapper = FusionContextWrapper(
            fusion_model=model,
            device=torch.device("cpu"),
            perturb_modalities=("graph", "tab", "rdf"),
            mapping=mapping,
        )
        wrapper.set_context(
            atom_fea=atom_fea,
            nbr_fea=torch.zeros(2, 1),
            nbr_fea_idx=torch.zeros(2, 1, dtype=torch.long),
            crystal_atom_idx=[torch.tensor([0, 1])],
            tab_inputs=tab,
            rdf_inputs=rdf,
        )

        pred = wrapper(input_tensor)

        self.assertEqual(tuple(pred.shape), (1,))
        batch = model.last_batch
        torch.testing.assert_close(batch["graph_inputs"][0], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        torch.testing.assert_close(batch["tab_inputs"], torch.tensor([[10.0, 20.0]]))
        torch.testing.assert_close(batch["rdf_inputs"], torch.tensor([[30.0, 40.0, 50.0]]))

    def test_grouped_rdf_columns_reconstruct_bin_windows(self) -> None:
        atom_fea = torch.zeros(1, 2)
        tab = torch.zeros(1, 1)
        rdf = torch.zeros(1, 5)
        input_tensor, mapping = FusionInputWrapper(
            perturb_modalities=("rdf",),
            rdf_group_size=2,
        ).build_input_and_mapping(
            atom_fea=torch.zeros(1, 2),
            tab=torch.zeros(1, 1),
            rdf=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
        )
        model = RecordingFusionModel()
        wrapper = FusionContextWrapper(
            fusion_model=model,
            device=torch.device("cpu"),
            perturb_modalities=("rdf",),
            mapping=mapping,
        )
        wrapper.set_context(
            atom_fea=atom_fea,
            nbr_fea=torch.zeros(1, 1),
            nbr_fea_idx=torch.zeros(1, 1, dtype=torch.long),
            crystal_atom_idx=[torch.tensor([0])],
            tab_inputs=tab,
            rdf_inputs=rdf,
        )

        wrapper(input_tensor)

        self.assertEqual(tuple(input_tensor.shape), (2, 3))
        self.assertEqual([entry["bin_indices"] for entry in mapping], [[0, 1], [2, 3], [4]])
        torch.testing.assert_close(model.last_batch["rdf_inputs"], torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))

    def test_split_main_effects_uses_publication_graph_tab_rdf_boundaries(self) -> None:
        effects = list(range(2 + 3 + 4))
        split = split_main_effects(effects, n_atoms=2, n_tab_features=3)

        self.assertEqual(split.graph.tolist(), [0.0, 1.0])
        self.assertEqual(split.tabular.tolist(), [2.0, 3.0, 4.0])
        self.assertEqual(split.rdf.tolist(), [5.0, 6.0, 7.0, 8.0])

    def test_interaction_helpers_preserve_min_max_scaling_and_grouping(self) -> None:
        interactions = {(0, 2): 1.0, (1, 3): 3.0, (4, 5): 2.0}

        frame = interactions_to_frame(interactions)
        grouped = group_interactions_by_modality(
            frame,
            n_atoms=2,
            n_tab_features=2,
            n_rdf_features=2,
        )

        self.assertEqual(frame["interaction"].tolist(), [0.0, 1.0, 0.5])
        self.assertEqual(grouped.loc[("graph", "tab")], 0.5)
        self.assertEqual(grouped.loc[("rdf", "rdf")], 0.5)


if __name__ == "__main__":
    unittest.main()
