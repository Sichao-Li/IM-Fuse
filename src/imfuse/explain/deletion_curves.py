from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from imfuse.fusion.feature_store import ProcessedFeatureStore
from imfuse.explain.composition_importance import (
    DEFAULT_MODEL_NAME,
    load_mid_tri_model,
    resolve_device,
    select_samples,
)
from imfuse.explain.faithfulness import _predict_mid_tri
from imfuse.training.metrics import regression_metrics

DEFAULT_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
ORDERS = ("top", "random", "bottom")
DELETION_CURVE_COLORS = {
    "top": "#8da0cb",
    "random": "#fc8d62",
    "bottom": "#66c2a5",
}


@dataclass(frozen=True)
class FeatureGroup:
    name: str
    indices: list[int]
    importance: float


def parse_feature_indices(value: str | float | int) -> list[int]:
    text = "" if pd.isna(value) else str(value)
    if text in {"", "all"}:
        return []
    return [int(part) for part in text.split(",") if part != ""]


def ablation_count(n_items: int, fraction: float) -> int:
    if fraction < 0 or fraction > 1:
        raise ValueError("fraction must be between 0 and 1")
    if n_items == 0 or fraction == 0:
        return 0
    return max(1, min(n_items, int(np.ceil(n_items * fraction))))


def ranked_feature_groups(frame: pd.DataFrame, order: str, seed: int) -> list[FeatureGroup]:
    groups = [
        FeatureGroup(
            name=str(row["feature_group"]),
            indices=parse_feature_indices(row["feature_indices"]),
            importance=float(row["delta_mae_mean"]),
        )
        for _, row in frame.iterrows()
    ]
    if order == "top":
        return sorted(groups, key=lambda group: group.importance, reverse=True)
    if order == "bottom":
        return sorted(groups, key=lambda group: group.importance)
    if order == "random":
        indices = np.random.default_rng(seed).permutation(len(groups))
        return [groups[int(index)] for index in indices]
    raise ValueError(f"Unsupported order {order!r}")


def _load_samples(
    target_col: str,
    seed: int,
    split: str,
    processed_root: Path,
    split_dir: Path,
    sample_index: int,
    max_samples: int,
) -> tuple[pd.DataFrame, torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]], torch.Tensor]:
    split_frame = pd.read_csv(Path(split_dir) / target_col / f"seed_{seed}" / f"{split}.csv")
    if max_samples <= 0:
        max_samples = len(split_frame) - sample_index
    selected = select_samples(split_frame, sample_id=None, sample_index=sample_index, max_samples=max_samples)
    store = ProcessedFeatureStore(processed_root, ("tabular", "rdf", "structure"))
    tabs = []
    rdfs = []
    graphs = []
    targets = []
    for _, row in selected.iterrows():
        sample_id = str(row["sample_id"])
        tabs.append(store.load("tabular", sample_id)["features"].float())
        rdfs.append(store.load("rdf", sample_id)["features"].float())
        graphs.append(store.load("structure", sample_id)["features"])
        targets.append(float(row["target"]))
    return selected, torch.stack(tabs), torch.stack(rdfs), graphs, torch.tensor(targets, dtype=torch.float32)


def _load_model(
    target_col: str,
    seed: int,
    checkpoint_path: Path | None,
    tabs: torch.Tensor,
    rdfs: torch.Tensor,
    graphs: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.nn.Module:
    checkpoint_path = checkpoint_path or (
        Path("results/final_publication")
        / target_col
        / "random_split"
        / "checkpoints"
        / DEFAULT_MODEL_NAME
        / f"seed_{seed}"
        / "model.pt"
    )
    return load_mid_tri_model(
        checkpoint_path=Path(checkpoint_path),
        tabular_dim=int(tabs.shape[-1]),
        rdf_dim=int(rdfs.shape[-1]),
        atom_fea_dim=int(graphs[0]["atom_fea"].shape[-1]),
        nbr_fea_dim=int(graphs[0]["nbr_fea"].shape[-1]),
        device=device,
    )


def _predict_samples(
    model: torch.nn.Module,
    tabs: torch.Tensor,
    rdfs: torch.Tensor,
    graphs: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    preds = [
        _predict_mid_tri(model, tabs[idx : idx + 1], rdfs[idx : idx + 1], graph, device=device)
        for idx, graph in enumerate(graphs)
    ]
    return torch.tensor(preds, dtype=torch.float32)


def permute_groups(values: torch.Tensor, groups: list[FeatureGroup], seed: int) -> torch.Tensor:
    """Permute selected feature-group columns across samples."""
    output = values.clone()
    columns = sorted({index for group in groups for index in group.indices})
    if columns:
        permutation = torch.as_tensor(
            np.random.default_rng(seed).permutation(output.shape[0]),
            dtype=torch.long,
        )
        output[:, columns] = output[permutation][:, columns]
    return output


def _zero_atoms_for_sample(graph: dict[str, torch.Tensor], atom_indices: list[int]) -> dict[str, torch.Tensor]:
    output = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in graph.items()
    }
    if atom_indices:
        output["atom_fea"][atom_indices, :] = 0.0
    return output


def _curve_row(
    target_col: str,
    modality: str,
    order: str,
    fraction: float,
    n_deleted: int,
    y_true: torch.Tensor,
    baseline_pred: torch.Tensor,
    ablated_pred: torch.Tensor,
) -> dict[str, float | str | int]:
    baseline = regression_metrics(y_true, baseline_pred)
    ablated = regression_metrics(y_true, ablated_pred)
    return {
        "target_col": target_col,
        "modality": modality,
        "order": order,
        "ablation_fraction": float(fraction),
        "n_deleted": int(n_deleted),
        "baseline_mae": baseline["mae"],
        "ablated_mae": ablated["mae"],
        "delta_mae": ablated["mae"] - baseline["mae"],
        "baseline_rmse": baseline["rmse"],
        "ablated_rmse": ablated["rmse"],
        "delta_rmse": ablated["rmse"] - baseline["rmse"],
        "baseline_r2": baseline["r2"],
        "ablated_r2": ablated["r2"],
        "delta_r2": ablated["r2"] - baseline["r2"],
        "prediction_delta_mean": float(torch.mean(torch.abs(ablated_pred - baseline_pred)).item()),
    }


def compute_tabular_or_rdf_deletion_curve(
    target_col: str,
    modality: str,
    summary_path: Path,
    model: torch.nn.Module,
    tabs: torch.Tensor,
    rdfs: torch.Tensor,
    graphs: list[dict[str, torch.Tensor]],
    y_true: torch.Tensor,
    device: torch.device,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> pd.DataFrame:
    summary = pd.read_csv(summary_path)
    baseline_pred = _predict_samples(model, tabs, rdfs, graphs, device)
    rows = []
    for order in ORDERS:
        ranked = ranked_feature_groups(summary, order=order, seed=seed)
        for fraction in fractions:
            count = ablation_count(len(ranked), fraction)
            groups = ranked[:count]
            ablated_tabs = permute_groups(tabs, groups, seed=seed + count) if modality == "composition" else tabs
            ablated_rdfs = permute_groups(rdfs, groups, seed=seed + count) if modality == "rdf" else rdfs
            ablated_pred = _predict_samples(model, ablated_tabs, ablated_rdfs, graphs, device)
            rows.append(
                _curve_row(
                    target_col,
                    modality,
                    order,
                    fraction,
                    count,
                    y_true,
                    baseline_pred,
                    ablated_pred,
                )
            )
    return pd.DataFrame(rows)


def compute_structure_atom_deletion_curve(
    target_col: str,
    atom_importance_path: Path,
    model: torch.nn.Module,
    tabs: torch.Tensor,
    rdfs: torch.Tensor,
    graphs: list[dict[str, torch.Tensor]],
    sample_ids: list[str],
    y_true: torch.Tensor,
    device: torch.device,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> pd.DataFrame:
    atom_importance = pd.read_csv(atom_importance_path)
    baseline_pred = _predict_samples(model, tabs, rdfs, graphs, device)
    rows = []
    for order in ORDERS:
        for fraction in fractions:
            ablated_graphs = []
            n_deleted = 0
            for idx, sample_id in enumerate(sample_ids):
                sample_rows = atom_importance[atom_importance["sample_id"].astype(str) == str(sample_id)]
                atoms = sample_rows[["atom_idx", "prediction_delta"]].copy()
                if order == "top":
                    atoms = atoms.sort_values("prediction_delta", ascending=False)
                elif order == "bottom":
                    atoms = atoms.sort_values("prediction_delta", ascending=True)
                elif order == "random":
                    atoms = atoms.sample(frac=1.0, random_state=seed + idx)
                count = ablation_count(len(atoms), fraction)
                n_deleted += count
                ablated_graphs.append(_zero_atoms_for_sample(graphs[idx], atoms["atom_idx"].head(count).astype(int).tolist()))
            ablated_pred = _predict_samples(model, tabs, rdfs, ablated_graphs, device)
            rows.append(
                _curve_row(
                    target_col,
                    "structure",
                    order,
                    fraction,
                    n_deleted,
                    y_true,
                    baseline_pred,
                    ablated_pred,
                )
            )
    return pd.DataFrame(rows)


def compute_overall_modality_deletion_curve(
    target_col: str,
    composition_summary_path: Path,
    rdf_summary_path: Path,
    structure_summary_path: Path,
    model: torch.nn.Module,
    tabs: torch.Tensor,
    rdfs: torch.Tensor,
    graphs: list[dict[str, torch.Tensor]],
    y_true: torch.Tensor,
    device: torch.device,
    fractions: tuple[float, ...] = DEFAULT_FRACTIONS,
    seed: int = 0,
) -> pd.DataFrame:
    composition = pd.read_csv(composition_summary_path)
    rdf = pd.read_csv(rdf_summary_path)
    structure = pd.read_csv(structure_summary_path)
    modality_importance = pd.DataFrame(
        {
            "feature_group": ["composition", "rdf", "structure"],
            "feature_indices": ["all", "all", "all"],
            "delta_mae_mean": [
                float(composition["delta_mae_mean"].max()),
                float(rdf["delta_mae_mean"].max()),
                float(structure["delta_mae_mean"].max()),
            ],
        }
    )
    baseline_pred = _predict_samples(model, tabs, rdfs, graphs, device)
    rows = []
    for order in ORDERS:
        ranked = ranked_feature_groups(modality_importance, order=order, seed=seed)
        for fraction in fractions:
            count = ablation_count(len(ranked), fraction)
            selected = {group.name for group in ranked[:count]}
            ablated_tabs = torch.zeros_like(tabs) if "composition" in selected else tabs
            ablated_rdfs = torch.zeros_like(rdfs) if "rdf" in selected else rdfs
            ablated_graphs = (
                [_zero_atoms_for_sample(graph, list(range(graph["atom_fea"].shape[0]))) for graph in graphs]
                if "structure" in selected
                else graphs
            )
            ablated_pred = _predict_samples(model, ablated_tabs, ablated_rdfs, ablated_graphs, device)
            row = _curve_row(
                target_col,
                "overall",
                order,
                fraction,
                count,
                y_true,
                baseline_pred,
                ablated_pred,
            )
            row["deleted_modalities"] = ",".join(sorted(selected))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_deletion_curve(curves: pd.DataFrame, output_path: Path, ylabel: str = "MAE increase") -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for order, group in curves.groupby("order", sort=False):
        group = group.sort_values("ablation_fraction")
        ax.plot(
            group["ablation_fraction"],
            group["delta_mae"],
            marker="o",
            linewidth=2.0,
            color=DELETION_CURVE_COLORS.get(str(order)),
            label=order,
        )
    ax.axhline(0, linestyle="--", linewidth=1.0, color="black")
    ax.set_xlabel("Deleted feature fraction")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def run_deletion_curve_suite(
    target_col: str,
    input_root: Path,
    output_dir: Path,
    processed_root: Path,
    split_dir: Path,
    checkpoint_path: Path | None,
    seed: int = 0,
    split: str = "test",
    sample_index: int = 0,
    max_samples: int = 100,
    device_name: str = "cpu",
) -> dict[str, Path]:
    selected, tabs, rdfs, graphs, y_true = _load_samples(
        target_col,
        seed,
        split,
        processed_root,
        split_dir,
        sample_index,
        max_samples,
    )
    device = resolve_device(device_name)
    model = _load_model(target_col, seed, checkpoint_path, tabs, rdfs, graphs, device)
    sample_ids = selected["sample_id"].astype(str).tolist()

    paths = {
        "composition": output_dir / f"{target_col}_composition_deletion_curve.pdf",
        "rdf": output_dir / f"{target_col}_rdf_deletion_curve.pdf",
        "structure": output_dir / f"{target_col}_structure_deletion_curve.pdf",
        "overall": output_dir / f"{target_col}_overall_3modality_deletion_curve.pdf",
    }
    curve_paths = {
        "composition": input_root / "composition_permutation" / "deletion_curve.csv",
        "rdf": input_root / "rdf_permutation" / "deletion_curve.csv",
        "structure": input_root / "structure_atom_ablation" / "deletion_curve.csv",
        "overall": input_root / "overall_3modality_deletion_curve.csv",
    }

    composition_curve = compute_tabular_or_rdf_deletion_curve(
        target_col,
        "composition",
        input_root / "composition_permutation" / "permutation_importance_summary.csv",
        model,
        tabs,
        rdfs,
        graphs,
        y_true,
        device,
        seed=seed,
    )
    rdf_curve = compute_tabular_or_rdf_deletion_curve(
        target_col,
        "rdf",
        input_root / "rdf_permutation" / "permutation_importance_summary.csv",
        model,
        tabs,
        rdfs,
        graphs,
        y_true,
        device,
        seed=seed,
    )
    structure_curve = compute_structure_atom_deletion_curve(
        target_col,
        input_root / "structure_atom_ablation" / "structure_atom_ablation_importance.csv",
        model,
        tabs,
        rdfs,
        graphs,
        sample_ids,
        y_true,
        device,
        seed=seed,
    )
    overall_curve = compute_overall_modality_deletion_curve(
        target_col,
        input_root / "composition_permutation" / "permutation_importance_summary.csv",
        input_root / "rdf_permutation" / "permutation_importance_summary.csv",
        input_root / "structure_permutation" / "permutation_importance_summary.csv",
        model,
        tabs,
        rdfs,
        graphs,
        y_true,
        device,
        seed=seed,
    )

    for name, curve in {
        "composition": composition_curve,
        "rdf": rdf_curve,
        "structure": structure_curve,
        "overall": overall_curve,
    }.items():
        curve_paths[name].parent.mkdir(parents=True, exist_ok=True)
        curve.to_csv(curve_paths[name], index=False)
        plot_deletion_curve(curve, paths[name])
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate composition/RDF/structure/overall deletion-curve figures.")
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/publication"))
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits/publication"))
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = run_deletion_curve_suite(
        target_col=args.target_col,
        input_root=args.input_root,
        output_dir=args.output_dir,
        processed_root=args.processed_root,
        split_dir=args.split_dir,
        checkpoint_path=args.checkpoint_path,
        seed=args.seed,
        split=args.split,
        sample_index=args.sample_index,
        max_samples=args.max_samples,
        device_name=args.device,
    )
    for path in paths.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
