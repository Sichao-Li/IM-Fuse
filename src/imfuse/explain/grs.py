from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class GRSConfig:
    epsilon_rate: float = 0.05
    loss_fn: str = "mean_squared_error"
    n_order: int = 2
    delta: float = 0.1


def run_fis_explainer(
    wrapped_model: nn.Module,
    input_tensor: torch.Tensor,
    output: torch.Tensor,
    config: GRSConfig = GRSConfig(),
) -> Any:
    """Run the GRS/FIS explainer with the publication notebook parameters."""
    try:
        import generalized_rashomon_set as grs
    except ImportError as exc:
        raise ImportError(
            "run_fis_explainer requires generalized_rashomon_set to be installed."
        ) from exc

    explainer = grs.explainers.fis_explainer(
        wrapped_model,
        input=input_tensor,
        output=output,
        epsilon_rate=config.epsilon_rate,
        loss_fn=config.loss_fn,
        n_order=config.n_order,
        delta=config.delta,
    )
    explainer.ref_explain()
    return explainer
