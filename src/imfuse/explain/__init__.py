"""Explanation helpers for multimodal fusion models."""

from imfuse.explain.context import FusionContextWrapper, FusionInputWrapper
from imfuse.explain.faithfulness import (
    ablation_indices,
    ablation_schedule,
    run_faithfulness_validation,
    summarize_faithfulness,
)
from imfuse.explain.permutation import run_permutation_importance
from imfuse.explain.permutation_matrix import run_permutation_importance_matrix
from imfuse.explain.results import (
    MainEffectSplit,
    group_interactions_by_modality,
    interactions_to_frame,
    split_main_effects,
)
from imfuse.explain.structure_ablation import (
    run_structure_atom_ablation,
    summarize_atom_ablation,
)

__all__ = [
    "FusionContextWrapper",
    "FusionInputWrapper",
    "MainEffectSplit",
    "ablation_indices",
    "ablation_schedule",
    "group_interactions_by_modality",
    "interactions_to_frame",
    "run_permutation_importance",
    "run_permutation_importance_matrix",
    "run_faithfulness_validation",
    "run_structure_atom_ablation",
    "split_main_effects",
    "summarize_atom_ablation",
    "summarize_faithfulness",
]
