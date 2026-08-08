import json
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.core import Structure


def gaussian_distance(
    distances: np.ndarray,
    centers: np.ndarray,
    width: float,
) -> np.ndarray:
    return np.exp(-((distances[..., None] - centers) ** 2) / (width**2))


def build_crystal_graph(
    structure: Structure,
    atom_init_path: Path,
    radius: float = 8.0,
    max_neighbors: int = 12,
    gaussian_centers: np.ndarray | None = None,
    gaussian_width: float = 0.2,
) -> dict[str, Any]:
    """Convert a pymatgen structure into CGCNN-style graph tensors."""

    centers = (
        np.asarray(gaussian_centers, dtype=np.float32)
        if gaussian_centers is not None
        else np.arange(0, radius + 0.2, 0.2, dtype=np.float32)
    )
    atom_init = json.loads(Path(atom_init_path).read_text())

    atom_features = []
    for site in structure:
        atomic_number = str(site.specie.Z)
        if atomic_number not in atom_init:
            raise KeyError(f"Missing atom feature for atomic number {atomic_number}")
        atom_features.append(atom_init[atomic_number])

    neighbor_indices = []
    neighbor_distances = []
    for i, site in enumerate(structure):
        neighbors = sorted(
            structure.get_neighbors(site, radius),
            key=lambda neighbor: float(neighbor.nn_distance),
        )
        ids = [int(neighbor.index) for neighbor in neighbors[:max_neighbors]]
        distances = [float(neighbor.nn_distance) for neighbor in neighbors[:max_neighbors]]
        while len(ids) < max_neighbors:
            ids.append(i)
            distances.append(radius + 1.0)
        neighbor_indices.append(ids)
        neighbor_distances.append(distances)

    nbr_distance_array = np.asarray(neighbor_distances, dtype=np.float32)
    return {
        "atom_fea": np.asarray(atom_features, dtype=np.float32),
        "nbr_fea": gaussian_distance(nbr_distance_array, centers, gaussian_width).astype(
            np.float32
        ),
        "nbr_fea_idx": np.asarray(neighbor_indices, dtype=np.int64),
    }
