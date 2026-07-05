import numpy as np
from pymatgen.core import Structure
from rdfpy import rdf


def build_rdf_vector(
    structure: Structure,
    bins: int = 400,
    cutoff: float = 20.0,
    supercell: int = 5,
    dr: float = 0.05,
    noise_std: float = 0.01,
    noise_seed: int | None = 0,
) -> np.ndarray:
    """Build the legacy fixed-length rdfpy RDF vector.

    This intentionally preserves the previous project representation:
    make a 5x supercell, compute rdfpy ``g(r)`` values, truncate/pad to
    400 bins, and keep the raw scale. The raw scale is part of the legacy
    signal and is not L1-normalized.
    """

    _ = cutoff
    crystal = structure.copy()
    crystal.make_supercell(supercell)
    coords = crystal.cart_coords
    if noise_std > 0:
        rng = np.random.default_rng(noise_seed)
        coords = coords + rng.normal(loc=0.0, scale=noise_std, size=coords.shape)
    values, _radii = rdf(coords, dr=dr, parallel=True, eps=1e-15, progress=False)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if len(values) >= bins:
        vector = values[:bins]
    else:
        vector = np.pad(values, (0, bins - len(values)), mode="constant")
    return np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
