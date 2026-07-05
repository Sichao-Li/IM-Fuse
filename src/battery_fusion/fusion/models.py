import torch
from torch import nn


class EarlyFusionRegressor(nn.Module):
    """Concatenate selected modality vectors before regression."""

    def __init__(self, input_dims: dict[str, int], hidden_dim: int = 256):
        super().__init__()
        self.modalities = tuple(input_dims.keys())
        total_dim = sum(input_dims.values())
        self.network = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        vectors = [batch[modality].float() for modality in self.modalities]
        return self.network(torch.cat(vectors, dim=-1)).squeeze(-1)


class MidFusionRegressor(nn.Module):
    """Encode each selected modality, then fuse latent vectors."""

    def __init__(self, input_dims: dict[str, int], latent_dim: int = 128):
        super().__init__()
        self.modalities = tuple(input_dims.keys())
        self.encoders = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(input_dim, latent_dim),
                    nn.ReLU(),
                    nn.Linear(latent_dim, latent_dim),
                    nn.ReLU(),
                )
                for modality, input_dim in input_dims.items()
            }
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim * len(self.modalities), latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        encoded = [
            self.encoders[modality](batch[modality].float())
            for modality in self.modalities
        ]
        return self.head(torch.cat(encoded, dim=-1)).squeeze(-1)


class LateFusionRegressor(nn.Module):
    """Learn a weighted combination of unimodal predictions."""

    def __init__(self, n_modalities: int):
        super().__init__()
        self.linear = nn.Linear(n_modalities, 1, bias=True)

    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        return self.linear(predictions.float()).squeeze(-1)


class LateFeatureFusionRegressor(nn.Module):
    """Predict per modality, then learn a late weighted combination."""

    def __init__(self, input_dims: dict[str, int], hidden_dim: int = 128):
        super().__init__()
        self.modalities = tuple(input_dims.keys())
        self.heads = nn.ModuleDict(
            {
                modality: nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for modality, input_dim in input_dims.items()
            }
        )
        self.combiner = LateFusionRegressor(n_modalities=len(self.modalities))

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        predictions = [
            self.heads[modality](batch[modality].float()).squeeze(-1)
            for modality in self.modalities
        ]
        return self.combiner(torch.stack(predictions, dim=-1))
