from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MainEffectSplit:
    graph: np.ndarray
    tabular: np.ndarray
    rdf: np.ndarray


def split_main_effects(
    main_effects: Iterable[float],
    n_atoms: int,
    n_tab_features: int = 76,
) -> MainEffectSplit:
    """Split GRS main effects using the publication graph, tabular, RDF order."""
    effects = np.asarray(list(main_effects), dtype=float)
    graph_end = n_atoms
    tab_end = graph_end + n_tab_features
    return MainEffectSplit(
        graph=effects[:graph_end],
        tabular=effects[graph_end:tab_end],
        rdf=effects[tab_end:],
    )


def interactions_to_frame(
    interactions: Mapping[tuple[int, int], float] | Iterable[tuple[tuple[int, int], float]],
) -> pd.DataFrame:
    """Convert interaction values to a min-max normalized DataFrame.

    The normalization mirrors the publication notebook: `(value - min) / (max - min)`.
    """
    items = interactions.items() if isinstance(interactions, Mapping) else interactions
    raw_items = list(items)
    values = [float(value) for _, value in raw_items]
    min_value = min(values)
    max_value = max(values)
    denom = max_value - min_value

    rows = []
    for (i, j), value in raw_items:
        normalized = 0.0 if denom == 0 else (float(value) - min_value) / denom
        rows.append({"i": int(i), "j": int(j), "interaction": normalized})
    return pd.DataFrame(rows)


def feature_group(
    index: int,
    n_atoms: int,
    n_tab_features: int = 76,
    n_rdf_features: int | None = None,
) -> str:
    """Map a fused explanation index back to graph, tabular, or RDF."""
    if 0 <= index < n_atoms:
        return "graph"
    if n_atoms <= index < n_atoms + n_tab_features:
        return "tab"
    rdf_start = n_atoms + n_tab_features
    if n_rdf_features is None:
        if index >= rdf_start:
            return "rdf"
    elif rdf_start <= index < rdf_start + n_rdf_features:
        return "rdf"
    return "unknown"


def group_interactions_by_modality(
    interactions: pd.DataFrame,
    n_atoms: int,
    n_tab_features: int = 76,
    n_rdf_features: int | None = None,
) -> pd.Series:
    """Average normalized interactions for each modality-pair group."""
    frame = interactions.copy()
    frame["i_group"] = frame["i"].apply(
        lambda value: feature_group(
            int(value),
            n_atoms=n_atoms,
            n_tab_features=n_tab_features,
            n_rdf_features=n_rdf_features,
        )
    )
    frame["j_group"] = frame["j"].apply(
        lambda value: feature_group(
            int(value),
            n_atoms=n_atoms,
            n_tab_features=n_tab_features,
            n_rdf_features=n_rdf_features,
        )
    )
    return frame.groupby(["i_group", "j_group"])["interaction"].mean()
