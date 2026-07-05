import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from battery_fusion.data.foundation import (
    audit_cif_coverage,
    build_label_table_from_mp_total,
    copy_matching_cifs,
    prepare_data_foundation,
)
from battery_fusion.data.splits import create_split_manifest
from battery_fusion.paths import ProjectPaths


class DataFoundationTests(unittest.TestCase):
    def test_create_split_manifest_is_deterministic_and_complete(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels = pd.DataFrame(
                {
                    "id_discharge": [f"mp-{i}" for i in range(10)],
                    "target": list(range(10)),
                }
            )
            labels_path = tmp_path / "labels.csv"
            labels.to_csv(labels_path, index=False)

            first = create_split_manifest(
                labels_path=labels_path,
                output_path=tmp_path / "split_a.json",
                seed=7,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
            )
            second = create_split_manifest(
                labels_path=labels_path,
                output_path=tmp_path / "split_b.json",
                seed=7,
                train_ratio=0.6,
                val_ratio=0.2,
                test_ratio=0.2,
            )

            self.assertEqual(first["splits"], second["splits"])
            all_ids = (
                first["splits"]["train"]
                + first["splits"]["val"]
                + first["splits"]["test"]
            )
            self.assertEqual(sorted(all_ids), sorted(labels["id_discharge"].tolist()))
            self.assertEqual(len(first["splits"]["train"]), 6)
            self.assertEqual(len(first["splits"]["val"]), 2)
            self.assertEqual(len(first["splits"]["test"]), 2)

            written = json.loads((tmp_path / "split_a.json").read_text())
            self.assertEqual(written["seed"], 7)
            self.assertTrue(written["source"]["labels_sha256"])

    def test_project_paths_are_split_and_modality_aware(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            paths = ProjectPaths(root=tmp_path)

            self.assertEqual(
                paths.raw_mp_total, tmp_path / "data" / "raw" / "mp_total.csv"
            )
            self.assertEqual(paths.raw_cif_dir, tmp_path / "data" / "raw" / "cifs")
            self.assertEqual(
                paths.processed_modality_dir("split_seed_7", "rdf", "train"),
                tmp_path / "data" / "processed" / "split_seed_7" / "rdf" / "train",
            )

    def test_build_label_table_from_mp_total_standardizes_columns(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            mp_total = tmp_path / "external_mp_total.csv"
            pd.DataFrame(
                {
                    "sample_id": ["a", "b", "a", "c"],
                    "voltage": [1.0, 2.0, 1.5, None],
                    "formula": ["LiO", "NaO", "Li2O", "KO"],
                    "ion": ["Li", "Na", "Li", "K"],
                    "material_id": ["mp-1", "mp-2", "mp-3", "mp-4"],
                }
            ).to_csv(mp_total, index=False)

            labels, stats = build_label_table_from_mp_total(
                mp_total_path=mp_total,
                output_path=tmp_path / "labels.csv",
                target_col="voltage",
                id_col="sample_id",
                formula_col="formula",
                working_ion_col="ion",
                extra_metadata_cols=("material_id",),
            )

            self.assertEqual(labels["id_discharge"].tolist(), ["b", "a"])
            self.assertEqual(labels["target"].tolist(), [2.0, 1.5])
            self.assertEqual(labels["formula_discharge"].tolist(), ["NaO", "Li2O"])
            self.assertEqual(labels["working_ion"].tolist(), ["Na", "Li"])
            self.assertEqual(labels["material_id"].tolist(), ["mp-2", "mp-3"])
            self.assertEqual(stats["n_duplicate_rows_removed"], 1)
            self.assertEqual(stats["n_missing_target_rows_removed"], 1)

    def test_copy_and_audit_cif_coverage(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp).resolve()
            source = tmp_path / "source_cifs"
            target = tmp_path / "data" / "raw" / "cifs"
            source.mkdir()
            (source / "mp-1.cif").write_text("data_mp_1\n")
            (source / "mp-2.CIF").write_text("data_mp_2\n")

            copied = copy_matching_cifs(source, ["mp-1", "mp-2", "mp-3"], target)
            coverage = audit_cif_coverage(["mp-1", "mp-2", "mp-3"], target)

            self.assertEqual(copied["copied"].tolist(), [True, True, False])
            self.assertTrue((target / "mp-1.cif").exists())
            self.assertTrue((target / "mp-2.cif").exists())
            self.assertEqual(coverage["has_cif"].tolist(), [True, True, False])

    def test_prepare_data_foundation_writes_labels_splits_and_manifest(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mp_total = root / "external_mp_total.csv"
            source_cifs = root / "external_cifs"
            source_cifs.mkdir()
            (source_cifs / "mp-1.cif").write_text("data_mp_1\n")
            (source_cifs / "mp-2.cif").write_text("data_mp_2\n")
            pd.DataFrame(
                {
                    "id_discharge": ["mp-1", "mp-2", "mp-3", "mp-1"],
                    "capacity_vol": [10.0, 20.0, 30.0, 11.0],
                    "formula_discharge": ["LiO", "NaO", "KO", "Li2O"],
                    "working_ion": ["Li", "Na", "K", "Li"],
                }
            ).to_csv(mp_total, index=False)

            result = prepare_data_foundation(
                root=root,
                target_col="capacity_vol",
                mp_total=mp_total,
                cif_dir=source_cifs,
                seeds=[0],
                allow_missing_cifs=True,
            )

            self.assertTrue(Path(result.labels_path).exists())
            self.assertTrue(Path(result.cif_coverage_path).exists())
            self.assertTrue(Path(result.manifest_path).exists())
            self.assertEqual(result.n_labels, 3)
            self.assertEqual(result.n_duplicate_rows_removed, 1)
            self.assertEqual(result.n_cifs_found, 2)
            self.assertEqual(result.n_cifs_missing, 1)
            self.assertEqual(len(result.split_paths), 1)
            split = json.loads(Path(result.split_paths[0]).read_text())
            all_ids = split["splits"]["train"] + split["splits"]["val"] + split["splits"]["test"]
            self.assertEqual(sorted(all_ids), ["mp-1", "mp-2", "mp-3"])

    def test_prepare_data_foundation_fails_on_missing_cifs_by_default(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            mp_total = root / "external_mp_total.csv"
            pd.DataFrame(
                {
                    "id_discharge": ["mp-1"],
                    "capacity_vol": [10.0],
                    "formula_discharge": ["LiO"],
                }
            ).to_csv(mp_total, index=False)

            with self.assertRaises(FileNotFoundError):
                prepare_data_foundation(
                    root=root,
                    target_col="capacity_vol",
                    mp_total=mp_total,
                    seeds=[0],
                )


if __name__ == "__main__":
    unittest.main()
