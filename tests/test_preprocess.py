import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import torch
from pymatgen.core import Lattice, Structure

from battery_fusion.data.preprocess import preprocess_modalities


class PreprocessTests(unittest.TestCase):
    def test_preprocess_writes_aligned_modality_caches(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            raw = root / "data" / "raw"
            cifs = raw / "cifs"
            labels_dir = root / "data" / "labels"
            splits_dir = root / "data" / "splits"
            cifs.mkdir(parents=True)
            labels_dir.mkdir(parents=True)
            splits_dir.mkdir(parents=True)

            atom_init = raw / "atom_init.json"
            atom_init.write_text('{"3": [1.0, 0.0], "8": [0.0, 1.0]}')
            rows = []
            for sample_id, target in [("mp-1", 1.0), ("mp-2", 2.0)]:
                structure = Structure(
                    Lattice.cubic(4.0),
                    ["Li", "O"],
                    [[0, 0, 0], [0.5, 0.5, 0.5]],
                )
                structure.to(filename=str(cifs / f"{sample_id}.cif"))
                rows.append(
                    {
                        "id_discharge": sample_id,
                        "target": target,
                        "formula_discharge": "LiO",
                    }
                )
            labels_path = labels_dir / "labels_keep_last.csv"
            pd.DataFrame(rows).to_csv(labels_path, index=False)
            split_path = splits_dir / "split_seed_1.json"
            split_path.write_text(
                json.dumps(
                    {
                        "name": "split_seed_1",
                        "splits": {"train": ["mp-1"], "val": [], "test": ["mp-2"]},
                    }
                )
            )

            preprocess_modalities(
                root=root,
                split_path=split_path,
                labels_path=labels_path,
                modalities=["rdf", "tabular", "structure"],
                atom_init_path=atom_init,
                rdf_bins=8,
                rdf_cutoff=6.0,
                graph_radius=6.0,
                graph_max_neighbors=4,
            )

            base = root / "data" / "processed" / "split_seed_1"
            self.assertTrue((base / "rdf" / "train" / "mp-1.pt").exists())
            self.assertTrue((base / "tabular" / "test" / "mp-2.pt").exists())
            self.assertTrue((base / "structure" / "train" / "mp-1.pt").exists())
            index = pd.read_csv(base / "index.csv")
            self.assertEqual(set(index["modality"]), {"rdf", "tabular", "structure"})
            self.assertEqual(set(index["split"]), {"train", "test"})

            rdf = torch.load(base / "rdf" / "train" / "mp-1.pt", weights_only=False)
            self.assertEqual(tuple(rdf["features"].shape), (8,))
            self.assertEqual(rdf["target"], 1.0)


if __name__ == "__main__":
    unittest.main()
