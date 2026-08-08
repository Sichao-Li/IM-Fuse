from __future__ import annotations

from typing import Any

import torch
from torch import nn


class ConvLayer(nn.Module):
    """CGCNN-style gated neighbor convolution."""

    def __init__(self, atom_fea_len: int, nbr_fea_len: int):
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2 * atom_fea_len + nbr_fea_len, 2 * atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * atom_fea_len)
        self.bn2 = nn.BatchNorm1d(atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(
        self,
        atom_in_fea: torch.Tensor,
        nbr_fea: torch.Tensor,
        nbr_fea_idx: torch.Tensor,
    ) -> torch.Tensor:
        n_atoms, max_nbr = nbr_fea_idx.shape
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(n_atoms, max_nbr, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.reshape(-1, 2 * self.atom_fea_len)).reshape(
            n_atoms, max_nbr, 2 * self.atom_fea_len
        )
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        if nbr_sumed.shape[0] != 1:
            nbr_sumed = self.bn2(nbr_sumed)
        return self.softplus2(atom_in_fea + nbr_sumed)


class StructureNetwork(nn.Module):
    """Original-style CGCNN structural encoder."""

    def __init__(
        self,
        orig_atom_fea_len: int,
        nbr_fea_len: int,
        atom_fea_len: int = 128,
        n_conv: int = 1,
        h_fea_len: int = 256,
    ):
        super().__init__()
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len) for _ in range(n_conv)]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.out_dim = h_fea_len

    def forward(self, structure_input: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]) -> torch.Tensor:
        atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = structure_input
        atom_fea = self.embedding(atom_fea.float())
        nbr_fea = nbr_fea.float()
        nbr_fea_idx = nbr_fea_idx.long()
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        crys_fea = self.pooling(atom_fea, crystal_atom_idx)
        crys_fea = self.conv_to_fc(self.conv_to_fc_softplus(crys_fea))
        return self.conv_to_fc_softplus(crys_fea)

    @staticmethod
    def pooling(atom_fea: torch.Tensor, crystal_atom_idx: list[torch.Tensor]) -> torch.Tensor:
        assert sum(len(idx_map) for idx_map in crystal_atom_idx) == atom_fea.shape[0]
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)


class TabularEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class RDFEncoder(nn.Module):
    """RDF encoder matching the previous multimodal base: MLP over RDF vector."""

    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class MultimodalEarlyFusionRegressor(nn.Module):
    """Original-style tri-modal early fusion over tabular, RDF, and CGCNN encoders."""

    def __init__(
        self,
        tab_encoder: nn.Module,
        rdf_encoder: nn.Module,
        graph_encoder: nn.Module,
        d_joint: int = 128,
        n_targets: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tab_encoder = tab_encoder
        self.rdf_encoder = rdf_encoder
        self.graph_encoder = graph_encoder
        self.tab_proj = nn.Sequential(nn.Linear(tab_encoder.out_dim, d_joint), nn.ReLU(), nn.Dropout(dropout))
        self.rdf_proj = nn.Sequential(nn.Linear(rdf_encoder.out_dim, d_joint), nn.ReLU(), nn.Dropout(dropout))
        self.graph_proj = nn.Sequential(nn.Linear(graph_encoder.out_dim, d_joint), nn.ReLU(), nn.Dropout(dropout))
        self.fusion_head = nn.Sequential(
            nn.Linear(3 * d_joint, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_targets),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        z_tab = self.tab_proj(self.tab_encoder(batch["tabular"]))
        z_rdf = self.rdf_proj(self.rdf_encoder(batch["rdf"]))
        z_graph = self.graph_proj(self.graph_encoder(batch["structure"]))
        return self.fusion_head(torch.cat([z_tab, z_rdf, z_graph], dim=-1)).squeeze(-1)


class MultimodalMidFusionRegressor(nn.Module):
    """Original-style tri-modal middle fusion using token mixing."""

    def __init__(
        self,
        tab_encoder: nn.Module,
        rdf_encoder: nn.Module,
        graph_encoder: nn.Module,
        d_joint: int = 128,
        n_targets: int = 1,
        dropout: float = 0.1,
        n_heads: int = 2,
        n_layers: int = 3,
    ):
        super().__init__()
        self.tab_encoder = tab_encoder
        self.rdf_encoder = rdf_encoder
        self.graph_encoder = graph_encoder
        self.tab_proj = nn.Sequential(nn.Linear(tab_encoder.out_dim, d_joint), nn.ReLU(), nn.Dropout(dropout))
        self.rdf_proj = nn.Sequential(nn.Linear(rdf_encoder.out_dim, d_joint), nn.ReLU(), nn.Dropout(dropout))
        self.graph_proj = nn.Sequential(nn.Linear(graph_encoder.out_dim, d_joint), nn.ReLU(), nn.Dropout(dropout))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_joint,
            nhead=n_heads,
            dim_feedforward=4 * d_joint,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fusion_head = nn.Sequential(
            nn.Linear(d_joint, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_targets),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        z_tab = self.tab_proj(self.tab_encoder(batch["tabular"]))
        z_rdf = self.rdf_proj(self.rdf_encoder(batch["rdf"]))
        z_graph = self.graph_proj(self.graph_encoder(batch["structure"]))
        tokens = torch.stack([z_tab, z_rdf, z_graph], dim=1)
        fused = self.transformer(tokens).mean(dim=1)
        return self.fusion_head(fused).squeeze(-1)


def build_multimodal_early_fusion(
    tabular_dim: int,
    rdf_dim: int,
    atom_fea_dim: int,
    nbr_fea_dim: int,
    dropout: float = 0.1,
) -> MultimodalEarlyFusionRegressor:
    tab_encoder = TabularEncoder(in_dim=tabular_dim, hidden_dim=256, out_dim=128, dropout=dropout)
    rdf_encoder = RDFEncoder(in_dim=rdf_dim, hidden_dim=256, out_dim=128, dropout=dropout)
    graph_encoder = StructureNetwork(
        orig_atom_fea_len=atom_fea_dim,
        nbr_fea_len=nbr_fea_dim,
        atom_fea_len=128,
        n_conv=1,
        h_fea_len=256,
    )
    return MultimodalEarlyFusionRegressor(
        tab_encoder=tab_encoder,
        rdf_encoder=rdf_encoder,
        graph_encoder=graph_encoder,
        d_joint=128,
        n_targets=1,
        dropout=dropout,
    )


def build_multimodal_mid_fusion(
    tabular_dim: int,
    rdf_dim: int,
    atom_fea_dim: int,
    nbr_fea_dim: int,
    dropout: float = 0.0,
) -> MultimodalMidFusionRegressor:
    tab_encoder = TabularEncoder(in_dim=tabular_dim, hidden_dim=128, out_dim=64, dropout=dropout)
    rdf_encoder = RDFEncoder(in_dim=rdf_dim, hidden_dim=128, out_dim=64, dropout=dropout)
    graph_encoder = StructureNetwork(
        orig_atom_fea_len=atom_fea_dim,
        nbr_fea_len=nbr_fea_dim,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
    )
    return MultimodalMidFusionRegressor(
        tab_encoder=tab_encoder,
        rdf_encoder=rdf_encoder,
        graph_encoder=graph_encoder,
        d_joint=128,
        n_targets=1,
        dropout=dropout,
        n_heads=2,
        n_layers=3,
    )
