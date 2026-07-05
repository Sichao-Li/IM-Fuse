from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch

from battery_fusion.fusion.feature_store import ProcessedFeatureStore
from battery_fusion.explain.composition_importance import (
    DEFAULT_MODEL_NAME,
    load_mid_tri_model,
    resolve_device,
    select_samples,
)
from battery_fusion.explain.faithfulness import _predict_mid_tri
from battery_fusion.explain.fusion_importance import load_structure_symbols

LOGGER = logging.getLogger(__name__)


def summarize_atom_ablation(atom_importance: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-atom ablation rows by element and local atom index."""
    element_summary = (
        atom_importance.groupby("element")
        .agg(
            n_sites=("atom_idx", "size"),
            n_samples=("sample_id", "nunique"),
            prediction_delta_mean=("prediction_delta", "mean"),
            prediction_delta_median=("prediction_delta", "median"),
            prediction_delta_std=("prediction_delta", "std"),
            error_delta_mean=("error_delta", "mean"),
            error_delta_median=("error_delta", "median"),
            error_delta_std=("error_delta", "std"),
        )
        .reset_index()
        .sort_values("prediction_delta_mean", ascending=False)
    )
    site_summary = (
        atom_importance.groupby("atom_idx")
        .agg(
            n_sites=("atom_idx", "size"),
            n_samples=("sample_id", "nunique"),
            elements=("element", lambda values: ",".join(sorted(set(str(value) for value in values)))),
            prediction_delta_mean=("prediction_delta", "mean"),
            prediction_delta_median=("prediction_delta", "median"),
            prediction_delta_std=("prediction_delta", "std"),
            error_delta_mean=("error_delta", "mean"),
            error_delta_median=("error_delta", "median"),
            error_delta_std=("error_delta", "std"),
        )
        .reset_index()
        .sort_values("prediction_delta_mean", ascending=False)
    )
    return element_summary, site_summary


def summarize_edge_ablation(edge_importance: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-edge-feature ablation rows by edge feature dimension."""
    return (
        edge_importance.groupby("edge_feature_idx")
        .agg(
            n_samples=("sample_id", "nunique"),
            prediction_delta_mean=("prediction_delta", "mean"),
            prediction_delta_median=("prediction_delta", "median"),
            prediction_delta_std=("prediction_delta", "std"),
            error_delta_mean=("error_delta", "mean"),
            error_delta_median=("error_delta", "median"),
            error_delta_std=("error_delta", "std"),
        )
        .reset_index()
        .sort_values("prediction_delta_mean", ascending=False)
    )


def _ablate_atom(graph: dict[str, torch.Tensor], atom_idx: int) -> dict[str, torch.Tensor]:
    ablated = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in graph.items()
    }
    ablated["atom_fea"][atom_idx, :] = 0.0
    return ablated


def _ablate_edge_feature(graph: dict[str, torch.Tensor], edge_feature_idx: int) -> dict[str, torch.Tensor]:
    ablated = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in graph.items()
    }
    ablated["nbr_fea"][:, :, edge_feature_idx] = 0.0
    return ablated


def run_structure_atom_ablation(
    target_col: str,
    seed: int,
    split: str,
    processed_root: Path,
    split_dir: Path,
    checkpoint_path: Path | None,
    cif_dir: Path,
    output_dir: Path,
    sample_id: str | None = None,
    sample_index: int = 0,
    max_samples: int = 100,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    atom_path = output_dir / "structure_atom_ablation_importance.csv"
    element_path = output_dir / "structure_atom_ablation_element_summary.csv"
    site_path = output_dir / "structure_atom_ablation_site_summary.csv"
    edge_path = output_dir / "structure_edge_feature_ablation_importance.csv"
    edge_summary_path = output_dir / "structure_edge_feature_ablation_summary.csv"
    config_path = output_dir / "structure_atom_ablation_config.json"
    if not overwrite and any(path.exists() for path in [atom_path, element_path, site_path, edge_path, edge_summary_path]):
        raise FileExistsError(f"{output_dir} already has atom-ablation outputs; pass --overwrite")

    split_frame = pd.read_csv(Path(split_dir) / target_col / f"seed_{seed}" / f"{split}.csv")
    selected = select_samples(split_frame, sample_id=sample_id, sample_index=sample_index, max_samples=max_samples)
    store = ProcessedFeatureStore(processed_root, ("tabular", "rdf", "structure"))
    first_id = str(selected.iloc[0]["sample_id"])
    first_tab = store.load("tabular", first_id)["features"].float().unsqueeze(0)
    first_rdf = store.load("rdf", first_id)["features"].float().unsqueeze(0)
    first_graph = store.load("structure", first_id)["features"]
    device = resolve_device(device_name)
    checkpoint_path = checkpoint_path or (
        Path("results/final_publication")
        / target_col
        / "random_split"
        / "checkpoints"
        / DEFAULT_MODEL_NAME
        / f"seed_{seed}"
        / "model.pt"
    )
    model = load_mid_tri_model(
        checkpoint_path=Path(checkpoint_path),
        tabular_dim=int(first_tab.shape[-1]),
        rdf_dim=int(first_rdf.shape[-1]),
        atom_fea_dim=int(first_graph["atom_fea"].shape[-1]),
        nbr_fea_dim=int(first_graph["nbr_fea"].shape[-1]),
        device=device,
    )

    rows = []
    edge_rows = []
    for _, row in selected.iterrows():
        sample = str(row["sample_id"])
        tab = store.load("tabular", sample)["features"].float().unsqueeze(0)
        rdf = store.load("rdf", sample)["features"].float().unsqueeze(0)
        graph = store.load("structure", sample)["features"]
        n_atoms = int(graph["atom_fea"].shape[0])
        symbols = load_structure_symbols(sample, cif_dir=cif_dir, n_atoms=n_atoms)
        y_true = float(row["target"])
        y_pred = _predict_mid_tri(model, tab, rdf, graph, device=device)
        original_error = abs(y_true - y_pred)
        for atom_idx in range(n_atoms):
            ablated_graph = _ablate_atom(graph, atom_idx)
            y_pred_ablated = _predict_mid_tri(model, tab, rdf, ablated_graph, device=device)
            ablated_error = abs(y_true - y_pred_ablated)
            rows.append(
                {
                    "sample_id": sample,
                    "target_col": target_col,
                    "seed": seed,
                    "split": split,
                    "atom_idx": atom_idx,
                    "element": symbols[atom_idx] if atom_idx < len(symbols) else f"atom_{atom_idx}",
                    "n_atoms": n_atoms,
                    "y_true": y_true,
                    "y_pred_original": y_pred,
                    "y_pred_ablated": y_pred_ablated,
                    "prediction_delta": abs(y_pred - y_pred_ablated),
                    "original_error": original_error,
                    "ablated_error": ablated_error,
                    "error_delta": ablated_error - original_error,
                }
            )
        n_edge_features = int(graph["nbr_fea"].shape[-1])
        for edge_feature_idx in range(n_edge_features):
            ablated_graph = _ablate_edge_feature(graph, edge_feature_idx)
            y_pred_ablated = _predict_mid_tri(model, tab, rdf, ablated_graph, device=device)
            ablated_error = abs(y_true - y_pred_ablated)
            edge_rows.append(
                {
                    "sample_id": sample,
                    "target_col": target_col,
                    "seed": seed,
                    "split": split,
                    "edge_feature_idx": edge_feature_idx,
                    "n_edge_features": n_edge_features,
                    "y_true": y_true,
                    "y_pred_original": y_pred,
                    "y_pred_ablated": y_pred_ablated,
                    "prediction_delta": abs(y_pred - y_pred_ablated),
                    "original_error": original_error,
                    "ablated_error": ablated_error,
                    "error_delta": ablated_error - original_error,
                }
            )

    atom_importance = pd.DataFrame(rows)
    edge_importance = pd.DataFrame(edge_rows)
    element_summary, site_summary = summarize_atom_ablation(atom_importance)
    edge_summary = summarize_edge_ablation(edge_importance)
    atom_importance.to_csv(atom_path, index=False)
    edge_importance.to_csv(edge_path, index=False)
    element_summary.to_csv(element_path, index=False)
    site_summary.to_csv(site_path, index=False)
    edge_summary.to_csv(edge_summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "target_col": target_col,
                "seed": seed,
                "split": split,
                "processed_root": str(processed_root),
                "split_dir": str(split_dir),
                "checkpoint_path": str(checkpoint_path),
                "cif_dir": str(cif_dir),
                "sample_id": sample_id,
                "sample_index": sample_index,
                "max_samples": max_samples,
                "device": device_name,
            },
            indent=2,
        )
    )
    LOGGER.info("Wrote %s, %s, %s, %s, and %s", atom_path, edge_path, element_path, site_path, edge_summary_path)
    return atom_importance, element_summary, site_summary, edge_importance, edge_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fast atom-level structure ablation for final mid-tri fusion.")
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/legacy_rdf_split_seed_42"))
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits/publication"))
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--cif_dir", type=Path, default=Path("data/raw/cifs"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    _, element_summary, site_summary, _, edge_summary = run_structure_atom_ablation(
        target_col=args.target_col,
        seed=args.seed,
        split=args.split,
        processed_root=args.processed_root,
        split_dir=args.split_dir,
        checkpoint_path=args.checkpoint_path,
        cif_dir=args.cif_dir,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        sample_index=args.sample_index,
        max_samples=args.max_samples,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print("Top elements:")
    print(element_summary.head(20).to_string(index=False))
    print("\nTop local atom indices:")
    print(site_summary.head(20).to_string(index=False))
    print("\nTop edge feature dimensions:")
    print(edge_summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
