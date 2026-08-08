"""Train and probe a tri-modal IM-Fuse model on a small synthetic dataset.

This example runs entirely in memory, on CPU, in a few minutes. It does not
use the publication data contract (no ``mp_total.csv``, no cached splits, no
downloads) and shows the smallest complete path a new project needs:

1. build the three modality features for each structure;
2. wrap them in a minimal in-memory feature store;
3. train the intermediate-fusion model with the library training loop;
4. evaluate, then probe modality reliance with inference-time dropout.

The toy task predicts crystal density (g/cm^3) for rocksalt-type AB
structures. Density is a deliberately easy, physically meaningful target that
depends on both composition (atomic masses) and geometry (lattice volume), so
every modality carries signal. Expect a low MAE with all modalities present
and clear degradation when modalities are dropped.

Run from the repository root:

    python examples/quickstart.py
"""

from __future__ import annotations

import itertools
import json
import os
import tempfile
from pathlib import Path

# rdfpy emits benign RuntimeWarnings for empty radial shells on small toy
# cells; silence them here (the env var also reaches its worker processes).
os.environ.setdefault("PYTHONWARNINGS", "ignore::RuntimeWarning")

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Element, Lattice, Structure
from torch.utils.data import DataLoader

from imfuse import (
    PublicationTriDataset,
    build_crystal_graph,
    build_multimodal_mid_fusion,
    build_rdf_vector,
    collate_tri,
    formula_vector,
    infer_tri_dims,
    masked_modality_forward,
    predict_torch_model,
    set_seed,
    train_torch_model,
    tri_adapter,
    vocabulary_from_formulas,
)

CATIONS = ["Li", "Na", "K", "Mg", "Ca"]
ANIONS = ["O", "S", "F", "Cl"]
LATTICE_CONSTANTS = [4.2, 4.8, 5.4, 6.0]
RDF_BINS = 100  # 100 bins x 0.05 A = 5 A radial range
GRAPH_RADIUS = 6.0


class InMemoryFeatureStore:
    """Dict-backed replacement for ProcessedFeatureStore.

    The training datasets touch the feature cache through exactly one method,
    ``load(modality, sample_id) -> {"features": ...}``, so any object with
    that method can stand in for the on-disk store.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], dict] = {}

    def add(self, sample_id: str, tabular: np.ndarray, rdf: np.ndarray, graph: dict) -> None:
        self._data["tabular", sample_id] = {"features": torch.as_tensor(tabular, dtype=torch.float32)}
        self._data["rdf", sample_id] = {"features": torch.as_tensor(rdf, dtype=torch.float32)}
        self._data["structure", sample_id] = {
            "features": {
                "atom_fea": torch.as_tensor(graph["atom_fea"], dtype=torch.float32),
                "nbr_fea": torch.as_tensor(graph["nbr_fea"], dtype=torch.float32),
                "nbr_fea_idx": torch.as_tensor(graph["nbr_fea_idx"], dtype=torch.long),
            }
        }

    def load(self, modality: str, sample_id: str) -> dict:
        return self._data[modality, sample_id]


def write_one_hot_atom_init(elements: list[str], path: Path) -> None:
    """Author the atom_init.json the crystal-graph builder requires.

    IM-Fuse does not bundle atom embeddings. Any consistent width works; a
    one-hot over the elements present is the simplest valid choice.
    """
    numbers = sorted(Element(symbol).Z for symbol in elements)
    eye = np.eye(len(numbers))
    path.write_text(json.dumps({str(z): eye[i].tolist() for i, z in enumerate(numbers)}))


def build_toy_dataset(atom_init_path: Path) -> tuple[InMemoryFeatureStore, pd.DataFrame]:
    formulas = [f"{cation}{anion}" for cation, anion in itertools.product(CATIONS, ANIONS)]
    vocabulary = vocabulary_from_formulas(formulas)

    store = InMemoryFeatureStore()
    rows = []
    combinations = list(itertools.product(itertools.product(CATIONS, ANIONS), LATTICE_CONSTANTS))
    for index, ((cation, anion), a) in enumerate(combinations):
        if index % 20 == 0:
            print(f"  featurized {index}/{len(combinations)}")
        structure = Structure.from_spacegroup(
            "Fm-3m", Lattice.cubic(a), [cation, anion], [[0, 0, 0], [0.5, 0.5, 0.5]]
        )
        sample_id = f"{cation}{anion}_a{a:.1f}"
        store.add(
            sample_id,
            tabular=formula_vector(f"{cation}{anion}", vocabulary),
            rdf=build_rdf_vector(structure, bins=RDF_BINS, supercell=2, noise_std=0.0),
            graph=build_crystal_graph(structure, atom_init_path, radius=GRAPH_RADIUS),
        )
        rows.append({"sample_id": sample_id, "target": float(structure.density)})
    return store, pd.DataFrame(rows)


def probe_modalities(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Mean absolute error when only a subset of modalities is available."""
    conditions = {
        "tabular + rdf + structure": ("tabular", "rdf", "structure"),
        "tabular + rdf": ("tabular", "rdf"),
        "tabular + structure": ("tabular", "structure"),
        "rdf + structure": ("rdf", "structure"),
        "tabular only": ("tabular",),
        "rdf only": ("rdf",),
        "structure only": ("structure",),
    }
    model.eval()
    results = {}
    for label, available in conditions.items():
        errors = []
        with torch.no_grad():
            for batch in loader:
                inputs, target = tri_adapter(batch, device)
                prediction = masked_modality_forward(model, inputs, available)
                errors.append((prediction - target).abs())
        results[label] = float(torch.cat(errors).mean())
    return results


def main() -> None:
    set_seed(0)
    device = torch.device("cpu")

    with tempfile.TemporaryDirectory() as tmp:
        atom_init_path = Path(tmp) / "atom_init.json"
        write_one_hot_atom_init(sorted(set(CATIONS + ANIONS)), atom_init_path)
        print(f"Building features for {len(CATIONS) * len(ANIONS) * len(LATTICE_CONSTANTS)} structures ...")
        store, frame = build_toy_dataset(atom_init_path)

    shuffled = frame.sample(frac=1.0, random_state=0).reset_index(drop=True)
    n_val = n_test = max(4, len(shuffled) // 10)
    frames = {
        "test": shuffled.iloc[:n_test],
        "val": shuffled.iloc[n_test : n_test + n_val],
        "train": shuffled.iloc[n_test + n_val :],
    }
    datasets = {name: PublicationTriDataset(store, part) for name, part in frames.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=16, shuffle=(name == "train"), collate_fn=collate_tri)
        for name, dataset in datasets.items()
    }
    print({name: len(dataset) for name, dataset in datasets.items()})

    model = build_multimodal_mid_fusion(**infer_tri_dims(datasets["train"])).to(device)
    train_torch_model(
        model,
        loaders["train"],
        loaders["val"],
        tri_adapter,
        device,
        epochs=80,
        learning_rate=1e-3,
        weight_decay=1e-5,
        scheduler_milestone=60,
        early_stopping_patience=20,
        early_stopping_min_delta=0.0,
    )

    metrics, predictions = predict_torch_model(model, loaders["test"], tri_adapter, device)
    print("\nTest metrics (density, g/cm^3):")
    print({key: round(value, 4) for key, value in metrics.items()})
    print(predictions.head())

    print("\nModality-dropout probe (test MAE, g/cm^3):")
    for label, mae in probe_modalities(model, loaders["test"], device).items():
        print(f"  {label:<28} {mae:.3f}")


if __name__ == "__main__":
    main()
