from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from battery_fusion.fusion.modality_sets import normalize_modalities


class FusionCacheDataset(Dataset):
    """Load cached features aligned by id_discharge for selected modalities."""

    def __init__(self, processed_root: Path, split: str, modalities: list[str]):
        self.processed_root = Path(processed_root)
        self.split = split
        self.modalities = normalize_modalities(modalities)
        first_dir = self.processed_root / self.modalities[0] / split
        if not first_dir.exists():
            raise FileNotFoundError(f"Missing processed modality directory: {first_dir}")
        ids = {path.stem for path in first_dir.glob("*.pt")}
        for modality in self.modalities[1:]:
            modality_dir = self.processed_root / modality / split
            if not modality_dir.exists():
                raise FileNotFoundError(
                    f"Missing processed modality directory: {modality_dir}"
                )
            ids &= {path.stem for path in modality_dir.glob("*.pt")}
        self.ids = sorted(ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.ids[index]
        sample: dict[str, Any] = {
            "id_discharge": sample_id,
            "modalities": {},
        }
        target = None
        for modality in self.modalities:
            item = torch.load(
                self.processed_root / modality / self.split / f"{sample_id}.pt",
                map_location="cpu",
                weights_only=False,
            )
            sample["modalities"][modality] = item["features"]
            target = float(item["target"])
        sample["target"] = target
        return sample


def collate_fusion_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "id_discharge": [sample["id_discharge"] for sample in samples],
        "target": torch.tensor([sample["target"] for sample in samples], dtype=torch.float32),
    }
    modalities = samples[0]["modalities"].keys()
    for modality in modalities:
        values = [sample["modalities"][modality] for sample in samples]
        if modality == "structure":
            batch[modality] = torch.stack([_structure_to_vector(value) for value in values])
        else:
            batch[modality] = torch.stack(values)
    return batch


def _structure_to_vector(graph: dict[str, torch.Tensor]) -> torch.Tensor:
    return graph["atom_fea"].float().mean(dim=0)
