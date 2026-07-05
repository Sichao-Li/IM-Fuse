import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from battery_fusion.fusion.datasets import FusionCacheDataset, collate_fusion_batch
from battery_fusion.fusion.modality_sets import normalize_modalities


class FusionDatasetTests(unittest.TestCase):
    def test_normalize_modalities_accepts_dual_and_triple_sets(self) -> None:
        self.assertEqual(normalize_modalities(["rdf", "tabular"]), ("rdf", "tabular"))
        self.assertEqual(
            normalize_modalities(["structure", "rdf", "tabular"]),
            ("rdf", "structure", "tabular"),
        )
        with self.assertRaises(ValueError):
            normalize_modalities(["rdf", "bad"])

    def test_fusion_dataset_aligns_modalities_by_id(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            base = root / "data" / "processed" / "split_seed_1"
            for modality in ["rdf", "tabular"]:
                folder = base / modality / "train"
                folder.mkdir(parents=True)
                for sample_id, value in [("mp-1", 1.0), ("mp-2", 2.0)]:
                    torch.save(
                        {
                            "id_discharge": sample_id,
                            "features": torch.tensor([value]),
                            "target": value,
                        },
                        folder / f"{sample_id}.pt",
                    )
            structure_folder = base / "structure" / "train"
            structure_folder.mkdir(parents=True)
            for sample_id, value in [("mp-1", 1.0), ("mp-2", 2.0)]:
                torch.save(
                    {
                        "id_discharge": sample_id,
                        "features": {"atom_fea": torch.ones(2, 3) * value},
                        "target": value,
                    },
                    structure_folder / f"{sample_id}.pt",
                )

            dataset = FusionCacheDataset(
                processed_root=base,
                split="train",
                modalities=["rdf", "tabular", "structure"],
            )

            self.assertEqual(len(dataset), 2)
            sample = dataset[0]
            self.assertEqual(set(sample["modalities"]), {"rdf", "tabular", "structure"})
            batch = collate_fusion_batch([dataset[0], dataset[1]])
            self.assertEqual(tuple(batch["rdf"].shape), (2, 1))
            self.assertEqual(tuple(batch["tabular"].shape), (2, 1))
            self.assertEqual(tuple(batch["structure"].shape), (2, 3))
            self.assertEqual(tuple(batch["target"].shape), (2,))


if __name__ == "__main__":
    unittest.main()
