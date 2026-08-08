from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class TargetTransform:
    kind: str = "none"
    mean: float = 0.0
    std: float = 1.0

    @property
    def enabled(self) -> bool:
        return self.kind == "standardize"

    def transform_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not self.enabled:
            return values.copy()
        return (values - self.mean) / self.std

    def inverse_array(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not self.enabled:
            return values.copy()
        return values * self.std + self.mean

    def transform_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return values
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return (values - mean) / std

    def inverse_tensor(self, values: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return values
        mean = torch.as_tensor(self.mean, dtype=values.dtype, device=values.device)
        std = torch.as_tensor(self.std, dtype=values.dtype, device=values.device)
        return values * std + mean

    def transform_frame(self, frame: pd.DataFrame, target_col: str = "target") -> pd.DataFrame:
        output = frame.copy()
        output[target_col] = self.transform_array(output[target_col].to_numpy(dtype=float))
        return output

    def inverse_prediction_frame(
        self,
        frame: pd.DataFrame,
        inverse_y_true: bool = True,
    ) -> pd.DataFrame:
        output = frame.copy()
        if inverse_y_true and "y_true" in output.columns:
            output["y_true"] = self.inverse_array(output["y_true"].to_numpy(dtype=float))
        if "y_pred" in output.columns:
            output["y_pred"] = self.inverse_array(output["y_pred"].to_numpy(dtype=float))
        return output

    def to_config(self) -> dict[str, float | str]:
        return {"target_transform": self.kind, "target_mean": self.mean, "target_std": self.std}

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TargetTransform":
        kind = str(config.get("target_transform", "none"))
        if kind not in {"none", "standardize"}:
            kind = "none"
        return cls(
            kind=kind,
            mean=float(config.get("target_mean", 0.0)),
            std=_safe_std(float(config.get("target_std", 1.0))),
        )


def _safe_std(value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return float(value)


def fit_target_transform(values: np.ndarray | pd.Series, kind: str = "none") -> TargetTransform:
    if kind not in {"none", "standardize"}:
        raise ValueError(f"Unsupported target transform: {kind!r}")
    if kind == "none":
        return TargetTransform(kind="none", mean=0.0, std=1.0)
    array = np.asarray(values, dtype=float).reshape(-1)
    return TargetTransform(
        kind="standardize",
        mean=float(np.mean(array)),
        std=_safe_std(float(np.std(array))),
    )
