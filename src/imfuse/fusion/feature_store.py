from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class ProcessedFeatureStore:
    """Load cached modality features by sample id.

    The public pipeline writes one ``.pt`` feature object per sample under
    ``data/processed/{tabular,rdf,structure}/...``.  This small helper is shared
    by training, modality-dropout evaluation, and attribution scripts.
    """

    def __init__(self, processed_root: Path, modalities: tuple[str, ...]):
        self.processed_root = Path(processed_root)
        self.paths: dict[str, dict[str, Path]] = {}
        for modality in modalities:
            modality_paths = {
                path.stem: path
                for path in sorted((self.processed_root / modality).glob("*/*.pt"))
            }
            if not modality_paths:
                raise FileNotFoundError(
                    f"No cached {modality!r} features found under {self.processed_root}"
                )
            self.paths[modality] = modality_paths

    def load(self, modality: str, sample_id: str) -> dict[str, Any]:
        try:
            path = self.paths[modality][str(sample_id)]
        except KeyError as exc:
            raise FileNotFoundError(f"Missing {modality} feature for sample {sample_id}") from exc
        return torch.load(path, map_location="cpu", weights_only=False)
