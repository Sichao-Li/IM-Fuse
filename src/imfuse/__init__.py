"""Interpretable multimodal fusion for battery materials.

The top-level namespace re-exports the library path used to run IM-Fuse on
new data: feature builders, fusion models, dataset/collate utilities, the
training loop, and the modality-dropout probe. Imports are resolved lazily so
that ``import imfuse`` stays cheap. The publication reproduction and
attribution workflows remain available under ``imfuse.experiments`` and
``imfuse.explain``; the ``imfuse`` command line is the supported entry point
for full reruns.

Example:
    >>> from imfuse import build_multimodal_mid_fusion, train_torch_model
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "1.1.0"

_EXPORTS: dict[str, str] = {
    # Repository layout
    "ProjectPaths": "imfuse.paths",
    # Feature builders (pure functions)
    "vocabulary_from_formulas": "imfuse.features.tabular",
    "formula_vector": "imfuse.features.tabular",
    "build_rdf_vector": "imfuse.features.rdf",
    "build_crystal_graph": "imfuse.features.structure",
    # Encoders and fusion models
    "TabularEncoder": "imfuse.fusion.cgcnn_multimodal",
    "RDFEncoder": "imfuse.fusion.cgcnn_multimodal",
    "StructureNetwork": "imfuse.fusion.cgcnn_multimodal",
    "MultimodalEarlyFusionRegressor": "imfuse.fusion.cgcnn_multimodal",
    "MultimodalMidFusionRegressor": "imfuse.fusion.cgcnn_multimodal",
    "build_multimodal_early_fusion": "imfuse.fusion.cgcnn_multimodal",
    "build_multimodal_mid_fusion": "imfuse.fusion.cgcnn_multimodal",
    "RdfLSTMRegressor": "imfuse.models.lstm",
    # Cached-feature access
    "ProcessedFeatureStore": "imfuse.fusion.feature_store",
    # Datasets, collates, adapters, and the training loop
    "PublicationUnimodalDataset": "imfuse.experiments.publication",
    "PublicationTriDataset": "imfuse.experiments.publication",
    "collate_graph_features": "imfuse.experiments.publication",
    "collate_unimodal": "imfuse.experiments.publication",
    "collate_tri": "imfuse.experiments.publication",
    "unimodal_adapter": "imfuse.experiments.publication",
    "tri_adapter": "imfuse.experiments.publication",
    "infer_unimodal_dims": "imfuse.experiments.publication",
    "infer_tri_dims": "imfuse.experiments.publication",
    "train_torch_model": "imfuse.experiments.publication",
    "predict_torch_model": "imfuse.experiments.publication",
    "set_seed": "imfuse.experiments.publication",
    "resolve_device": "imfuse.experiments.publication",
    # Metrics and probes
    "regression_metrics": "imfuse.training.metrics",
    "masked_modality_forward": "imfuse.experiments.modality_dropout",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
