from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load_atoms(path: Path, file_format: str):
    from jarvis.core.atoms import Atoms

    if file_format == "poscar":
        return Atoms.from_poscar(str(path))
    if file_format == "cif":
        return Atoms.from_cif(str(path))
    raise ValueError(f"Unsupported file_format={file_format!r}")


def _feature_vector_from_atoms(
    model,
    atoms,
    feature_mode: str,
    cutoff: float,
    max_neighbors: int,
    device,
) -> np.ndarray:
    import torch
    from alignn.graphs import Graph

    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=float(cutoff),
        max_neighbors=int(max_neighbors),
    )
    lat = torch.tensor(atoms.lattice_mat)
    captured: dict[str, torch.Tensor] = {}
    handle = None
    if feature_mode == "readout":
        if not hasattr(model, "readout"):
            raise ValueError("Requested readout features, but pretrained model has no readout module.")

        def _capture_readout(_module, _inputs, output):
            captured["readout"] = output.detach().cpu()

        handle = model.readout.register_forward_hook(_capture_readout)
    with torch.no_grad():
        output = model([g.to(device), lg.to(device), lat.to(device)])
    if handle is not None:
        handle.remove()
    if feature_mode == "readout":
        if "readout" not in captured:
            raise RuntimeError("Readout hook did not capture a feature vector.")
        return np.asarray(captured["readout"], dtype=np.float32).reshape(-1)
    if isinstance(output, dict):
        output = output["out"]
    return np.asarray(output.detach().cpu(), dtype=np.float32).reshape(-1)


def featurize_pretrained_alignn(
    manifest: Path,
    output_csv: Path,
    model_name: str,
    file_format: str = "poscar",
    feature_mode: str = "readout",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    overwrite: bool = False,
) -> pd.DataFrame:
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"{output_csv} exists; pass --overwrite")
    import torch
    from alignn.pretrained import get_figshare_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_figshare_model(model_name)
    model.to(device)
    model.eval()
    manifest_frame = pd.read_csv(manifest)
    required = {"sample_id", "structure_path"}
    missing = required.difference(manifest_frame.columns)
    if missing:
        raise ValueError(f"{manifest} missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for row in manifest_frame.itertuples(index=False):
        atoms = _load_atoms(Path(row.structure_path), file_format=file_format)
        vector = _feature_vector_from_atoms(
            model=model,
            atoms=atoms,
            feature_mode=feature_mode,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            device=device,
        )
        rows.append(
            {
                "sample_id": str(row.sample_id),
                **{f"feature_{idx}": float(value) for idx, value in enumerate(vector)},
            }
        )
    output = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_csv, index=False)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen pretrained ALIGNN features.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--file_format", choices=["poscar", "cif"], default="poscar")
    parser.add_argument("--feature_mode", choices=["readout", "prediction"], default="readout")
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    featurize_pretrained_alignn(
        manifest=args.manifest,
        output_csv=args.output_csv,
        model_name=args.model_name,
        file_format=args.file_format,
        feature_mode=args.feature_mode,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
