import torch
from torch import nn


class GraphPoolingRegressor(nn.Module):
    """Small CGCNN-compatible graph baseline for cached structure features."""

    def __init__(self, atom_input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.atom_encoder = nn.Sequential(
            nn.Linear(atom_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, graphs: list[dict[str, torch.Tensor]]) -> torch.Tensor:
        pooled = []
        device = next(self.parameters()).device
        for graph in graphs:
            atom_fea = graph["atom_fea"].float().to(device)
            pooled.append(self.atom_encoder(atom_fea).mean(dim=0))
        return self.head(torch.stack(pooled)).squeeze(-1)
