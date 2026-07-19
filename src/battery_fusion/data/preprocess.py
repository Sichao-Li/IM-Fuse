import json
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from pymatgen.core import Structure

from battery_fusion.features.rdf import build_rdf_vector
from battery_fusion.features.structure import build_crystal_graph
from battery_fusion.features.tabular import formula_vector, vocabulary_from_formulas
from battery_fusion.paths import ProjectPaths

VALID_MODALITIES = {"rdf", "tabular", "structure"}


def _load_split(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _validate_modalities(modalities: Sequence[str]) -> list[str]:
    selected = list(modalities)
    invalid = sorted(set(selected).difference(VALID_MODALITIES))
    if invalid:
        raise ValueError(f"Unsupported modalities: {invalid}")
    return selected


def preprocess_modalities(
    root: Path,
    split_path: Path,
    labels_path: Path,
    modalities: Sequence[str],
    output_name: str | None = None,
    atom_init_path: Path | None = None,
    rdf_bins: int = 400,
    rdf_cutoff: float = 20.0,
    graph_radius: float = 8.0,
    graph_max_neighbors: int = 12,
) -> pd.DataFrame:
    """Build cached modality tensors aligned by a split manifest."""

    paths = ProjectPaths(root=Path(root))
    split = _load_split(split_path)
    split_name = output_name or split.get("name") or Path(split_path).stem
    selected = _validate_modalities(modalities)
    labels = pd.read_csv(labels_path)
    required = {"id_discharge", "target"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"{labels_path} is missing required columns {sorted(missing)}")
    if "formula_discharge" not in labels.columns and "tabular" in selected:
        raise ValueError("Tabular preprocessing requires formula_discharge in labels file")

    label_by_id = labels.set_index("id_discharge", drop=False)
    vocabulary = (
        vocabulary_from_formulas(label_by_id["formula_discharge"].astype(str).tolist())
        if "tabular" in selected
        else []
    )
    atom_path = Path(atom_init_path) if atom_init_path else paths.raw_dir / "atom_init.json"
    gaussian_centers = np.arange(0, graph_radius + 0.2, 0.2, dtype=np.float32)
    index_rows: list[dict[str, Any]] = []

    for split_label, ids in split["splits"].items():
        for sample_id in ids:
            if sample_id not in label_by_id.index:
                raise KeyError(f"{sample_id} from split is missing in {labels_path}")
            row = label_by_id.loc[sample_id]
            if isinstance(row, pd.DataFrame):
                raise ValueError(f"{sample_id} has duplicate rows in {labels_path}")
            target = float(row["target"])
            cif_path = paths.raw_cif_dir / f"{sample_id}.cif"
            if not cif_path.exists():
                raise FileNotFoundError(f"Missing CIF for {sample_id}: {cif_path}")
            structure = Structure.from_file(cif_path)

            if "rdf" in selected:
                features = torch.tensor(
                    build_rdf_vector(
                        structure,
                        bins=rdf_bins,
                        cutoff=rdf_cutoff,
                        noise_seed=_sample_seed(sample_id),
                    ),
                    dtype=torch.float32,
                )
                _save_feature(
                    paths, split_name, "rdf", split_label, sample_id, features, target
                )
                index_rows.append(_index_row(split_label, sample_id, "rdf", target))

            if "tabular" in selected:
                features = torch.tensor(
                    formula_vector(str(row["formula_discharge"]), vocabulary),
                    dtype=torch.float32,
                )
                _save_feature(
                    paths, split_name, "tabular", split_label, sample_id, features, target
                )
                index_rows.append(_index_row(split_label, sample_id, "tabular", target))

            if "structure" in selected:
                graph = build_crystal_graph(
                    structure=structure,
                    atom_init_path=atom_path,
                    radius=graph_radius,
                    max_neighbors=graph_max_neighbors,
                    gaussian_centers=gaussian_centers,
                )
                graph_tensors = {
                    key: torch.tensor(value)
                    for key, value in graph.items()
                }
                _save_feature(
                    paths,
                    split_name,
                    "structure",
                    split_label,
                    sample_id,
                    graph_tensors,
                    target,
                )
                index_rows.append(_index_row(split_label, sample_id, "structure", target))

    index = pd.DataFrame(index_rows)
    output_dir = paths.processed_split_dir(split_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    index.to_csv(output_dir / "index.csv", index=False)
    config = {
        "split": str(split_path),
        "output_name": split_name,
        "labels": str(labels_path),
        "modalities": selected,
        "rdf": {"bins": rdf_bins, "cutoff": rdf_cutoff},
        "structure": {
            "radius": graph_radius,
            "max_neighbors": graph_max_neighbors,
            "atom_init_path": str(atom_path),
        },
        "tabular": {"vocabulary": vocabulary},
    }
    (output_dir / "preprocessing_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return index


def _sample_seed(sample_id: str) -> int:
    digest = hashlib.sha256(str(sample_id).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _save_feature(
    paths: ProjectPaths,
    split_name: str,
    modality: str,
    split: str,
    sample_id: str,
    features: Any,
    target: float,
) -> None:
    output_dir = paths.processed_modality_dir(split_name, modality, split)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"id_discharge": sample_id, "features": features, "target": target},
        output_dir / f"{sample_id}.pt",
    )


def _index_row(split: str, sample_id: str, modality: str, target: float) -> dict[str, Any]:
    return {
        "split": split,
        "id_discharge": sample_id,
        "modality": modality,
        "target": target,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess cached modality tensors.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument(
        "--modalities", nargs="+", default=["rdf", "tabular", "structure"]
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Processed-cache directory name; defaults to the split manifest name.",
    )
    parser.add_argument("--atom-init", type=Path, default=None)
    parser.add_argument("--rdf-bins", type=int, default=400)
    parser.add_argument("--rdf-cutoff", type=float, default=20.0)
    parser.add_argument("--graph-radius", type=float, default=8.0)
    parser.add_argument("--graph-max-neighbors", type=int, default=12)
    args = parser.parse_args()

    index = preprocess_modalities(
        root=args.root,
        split_path=args.split,
        labels_path=args.labels,
        modalities=args.modalities,
        output_name=args.output_name,
        atom_init_path=args.atom_init,
        rdf_bins=args.rdf_bins,
        rdf_cutoff=args.rdf_cutoff,
        graph_radius=args.graph_radius,
        graph_max_neighbors=args.graph_max_neighbors,
    )
    print(f"Wrote {len(index)} cached modality records")


if __name__ == "__main__":
    main()
