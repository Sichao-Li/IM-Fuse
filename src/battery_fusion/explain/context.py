from __future__ import annotations

from typing import Any

import torch
from torch import nn


class FusionInputWrapper:
    """Build the perturbation matrix used by GRS/FIS for fusion inputs.

    This preserves the legacy explanation convention:
    graph columns are whole atom feature vectors, tabular columns are scalar
    features, and RDF columns are scalar bins. Scalar modalities are padded with
    zeros so all modalities can be concatenated column-wise.
    """

    def __init__(
        self,
        perturb_modalities: tuple[str, ...] | list[str],
        rdf_group_size: int = 1,
    ):
        self.perturb_modalities = tuple(perturb_modalities)
        if rdf_group_size <= 0:
            raise ValueError("rdf_group_size must be positive")
        self.rdf_group_size = rdf_group_size

    def build_input_and_mapping(
        self,
        atom_fea: torch.Tensor,
        tab: torch.Tensor,
        rdf: torch.Tensor,
    ) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        """Build a single-sample perturbation matrix and column mapping."""
        if tab.dim() == 2 and tab.size(0) == 1:
            tab = tab.squeeze(0)
        if rdf.dim() == 2 and rdf.size(0) == 1:
            rdf = rdf.squeeze(0)

        mapping: list[dict[str, Any]] = []
        cols: list[torch.Tensor] = []

        if "graph" in self.perturb_modalities:
            graph_part = atom_fea.T
            for atom_idx in range(atom_fea.size(0)):
                mapping.append(
                    {
                        "modality": "graph",
                        "atom_idx": atom_idx,
                        "all_dims": True,
                    }
                )
            cols.append(graph_part)

        if "tab" in self.perturb_modalities:
            tab_part = tab.unsqueeze(0)
            for feat_idx in range(tab.size(0)):
                mapping.append(
                    {
                        "modality": "tab",
                        "feat_idx": feat_idx,
                    }
                )
            cols.append(tab_part)

        if "rdf" in self.perturb_modalities:
            rdf_cols = []
            for start in range(0, rdf.size(0), self.rdf_group_size):
                end = min(rdf.size(0), start + self.rdf_group_size)
                bin_indices = list(range(start, end))
                group_col = rdf[start:end].unsqueeze(1)
                if group_col.size(0) < self.rdf_group_size:
                    pad = torch.zeros(
                        self.rdf_group_size - group_col.size(0),
                        1,
                        dtype=group_col.dtype,
                        device=group_col.device,
                    )
                    group_col = torch.cat([group_col, pad], dim=0)
                rdf_cols.append(group_col)
                mapping.append(
                    {
                        "modality": "rdf",
                        "bin_idx": start,
                        "bin_indices": bin_indices,
                    }
                )
            cols.append(torch.cat(rdf_cols, dim=1))

        if not cols:
            raise ValueError("No modalities selected for perturbation.")

        max_rows = max(col.size(0) for col in cols)
        padded_cols: list[torch.Tensor] = []
        for col in cols:
            if col.size(0) == max_rows:
                padded_cols.append(col)
                continue
            pad = torch.zeros(
                max_rows - col.size(0),
                col.size(1),
                dtype=col.dtype,
                device=col.device,
            )
            padded_cols.append(torch.cat([col, pad], dim=0))

        return torch.cat(padded_cols, dim=1), mapping


class FusionContextWrapper(nn.Module):
    """Reconstruct fusion-model batches from perturbed GRS/FIS columns."""

    def __init__(
        self,
        fusion_model: nn.Module,
        device: torch.device,
        perturb_modalities: tuple[str, ...] | list[str],
        mapping: list[dict[str, Any]],
    ):
        super().__init__()
        self.fusion_model = fusion_model
        self.device = device
        self.perturb_modalities = tuple(perturb_modalities)
        self.mapping = mapping

        self.base_atom_fea: torch.Tensor | None = None
        self.nbr_fea: torch.Tensor | None = None
        self.nbr_fea_idx: torch.Tensor | None = None
        self.crystal_atom_idx: list[torch.Tensor] | None = None
        self.base_tab: torch.Tensor | None = None
        self.base_rdf: torch.Tensor | None = None

    def set_context(
        self,
        atom_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
        crystal_atom_idx: list[torch.Tensor],
        tab_inputs: torch.Tensor,
        rdf_inputs: torch.Tensor,
    ) -> None:
        """Store the unperturbed sample context used for reconstruction."""
        self.base_atom_fea = atom_fea.to(self.device)
        self.nbr_fea = nbr_fea.to(self.device)
        self.nbr_fea_idx = nbr_fea_idx.to(self.device)
        self.crystal_atom_idx = [idx.to(self.device) for idx in crystal_atom_idx]

        tab_inputs = tab_inputs.to(self.device)
        rdf_inputs = rdf_inputs.to(self.device)
        if tab_inputs.dim() == 1:
            tab_inputs = tab_inputs.unsqueeze(0)
        if rdf_inputs.dim() == 1:
            rdf_inputs = rdf_inputs.unsqueeze(0)

        self.base_tab = tab_inputs
        self.base_rdf = rdf_inputs

    def forward(self, pert_input: torch.Tensor) -> torch.Tensor:
        if (
            self.base_atom_fea is None
            or self.nbr_fea is None
            or self.nbr_fea_idx is None
            or self.crystal_atom_idx is None
            or self.base_tab is None
            or self.base_rdf is None
        ):
            raise RuntimeError("FusionContextWrapper.set_context must be called before forward.")

        atom_fea = self.base_atom_fea.clone()
        tab = self.base_tab.clone()
        rdf = self.base_rdf.clone()

        _, n_cols = pert_input.shape
        if n_cols != len(self.mapping):
            raise ValueError(
                f"pert_input has {n_cols} columns but mapping has {len(self.mapping)} entries."
            )

        for column_idx in range(n_cols):
            col_info = self.mapping[column_idx]
            col_vals = pert_input[:, column_idx]
            modality = col_info["modality"]

            if modality == "graph":
                d_atom = atom_fea.size(1)
                atom_fea[col_info["atom_idx"]] = col_vals[:d_atom]
            elif modality == "tab":
                tab[0, col_info["feat_idx"]] = col_vals[0]
            elif modality == "rdf":
                bin_indices = col_info.get("bin_indices")
                if bin_indices is None:
                    rdf[0, col_info["bin_idx"]] = col_vals[0]
                else:
                    rdf[0, bin_indices] = col_vals[: len(bin_indices)]
            else:
                raise ValueError(f"Unsupported modality in mapping: {modality}")

        batch = {
            "graph_inputs": (
                atom_fea,
                self.nbr_fea,
                self.nbr_fea_idx,
                self.crystal_atom_idx,
            ),
            "tab_inputs": tab,
            "rdf_inputs": rdf,
        }
        return self.fusion_model(batch).squeeze(-1)
