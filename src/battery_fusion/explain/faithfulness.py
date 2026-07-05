from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from battery_fusion.fusion.feature_store import ProcessedFeatureStore
from battery_fusion.explain.composition_importance import (
    DEFAULT_MODEL_NAME,
    load_mid_tri_model,
    resolve_device,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_FRACTIONS = (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0)


def ablation_schedule(n_features: int, fractions: list[float] | tuple[float, ...]) -> list[int]:
    """Convert ablation fractions to feature counts.

    Any positive fraction ablates at least one feature so small feature sets still
    exercise the deletion curve.
    """
    if n_features < 0:
        raise ValueError("n_features must be non-negative")
    counts = []
    for fraction in fractions:
        if fraction < 0 or fraction > 1:
            raise ValueError(f"Ablation fraction must be between 0 and 1, got {fraction}")
        if n_features == 0 or fraction == 0:
            counts.append(0)
        else:
            counts.append(max(1, min(n_features, int(np.ceil(n_features * fraction)))))
    return counts


def ablation_indices(
    importance: np.ndarray | list[float],
    order: str,
    n_remove: int,
    seed: int,
) -> np.ndarray:
    """Return feature indices to ablate for one ranking order."""
    values = np.asarray(importance, dtype=float)
    if n_remove < 0:
        raise ValueError("n_remove must be non-negative")
    n_remove = min(n_remove, len(values))
    if n_remove == 0:
        return np.asarray([], dtype=int)
    if order == "top":
        ranked = np.argsort(-np.abs(values), kind="stable")
    elif order == "bottom":
        ranked = np.argsort(np.abs(values), kind="stable")
    elif order == "random":
        ranked = np.random.default_rng(seed).permutation(len(values))
    else:
        raise ValueError(f"Unsupported ablation order {order!r}")
    return ranked[:n_remove].astype(int)


def _predict_mid_tri(
    model: nn.Module,
    tab: torch.Tensor,
    rdf: torch.Tensor,
    graph: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    batch = {
        "tabular": tab.to(device),
        "rdf": rdf.to(device),
        "structure": (
            graph["atom_fea"].float().to(device),
            graph["nbr_fea"].float().to(device),
            graph["nbr_fea_idx"].long().to(device),
            [torch.arange(graph["atom_fea"].shape[0], dtype=torch.long, device=device)],
        ),
    }
    with torch.no_grad():
        return float(model(batch).detach().cpu().reshape(-1)[0])


def _ablate_features(
    tab: torch.Tensor,
    rdf: torch.Tensor,
    graph: dict[str, torch.Tensor],
    feature_rows: pd.DataFrame,
    selected_positions: np.ndarray,
    modality: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    tab_out = tab.clone()
    rdf_out = rdf.clone()
    graph_out = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in graph.items()
    }
    selected_rows = feature_rows.iloc[selected_positions]
    if modality == "composition":
        idx_col = "feature_idx" if "feature_idx" in selected_rows.columns else "local_feature_idx"
        indices = selected_rows[idx_col].astype(int).tolist()
        tab_out[:, indices] = 0.0
    elif modality == "rdf":
        indices = []
        for _, row in selected_rows.iterrows():
            if "rdf_bin_indices" in row and pd.notna(row["rdf_bin_indices"]) and str(row["rdf_bin_indices"]):
                indices.extend(int(value) for value in str(row["rdf_bin_indices"]).split(",") if value != "")
            else:
                indices.append(int(row["local_feature_idx"]))
        rdf_out[:, indices] = 0.0
    elif modality == "structure":
        atom_indices = selected_rows["atom_idx"].dropna().astype(int).tolist()
        if atom_indices:
            graph_out["atom_fea"][atom_indices, :] = 0.0
    else:
        raise ValueError(f"Unsupported modality {modality!r}")
    return tab_out, rdf_out, graph_out


def _infer_target_seed_split(frame: pd.DataFrame, target_col: str | None, seed: int | None, split: str | None) -> tuple[str, int, str]:
    if target_col is None:
        target_values = frame["target_col"].dropna().astype(str).unique()
        if len(target_values) != 1:
            raise ValueError("Could not infer target_col from importance CSV; pass --target_col")
        target_col = target_values[0]
    if seed is None:
        seed_values = frame["seed"].dropna().astype(int).unique()
        if len(seed_values) != 1:
            raise ValueError("Could not infer seed from importance CSV; pass --seed")
        seed = int(seed_values[0])
    if split is None:
        split_values = frame["split"].dropna().astype(str).unique()
        if len(split_values) != 1:
            raise ValueError("Could not infer split from importance CSV; pass --split")
        split = split_values[0]
    return target_col, int(seed), split


def run_faithfulness_validation(
    importance_csv: Path,
    modality: str,
    processed_root: Path,
    checkpoint_path: Path | None,
    output_dir: Path,
    target_col: str | None = None,
    seed: int | None = None,
    split: str | None = None,
    device_name: str = "cpu",
    fractions: list[float] | tuple[float, ...] = DEFAULT_FRACTIONS,
    orders: list[str] | tuple[str, ...] = ("top", "random", "bottom"),
    random_seed: int = 0,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate local importance scores by importance-guided feature ablation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    curves_path = output_dir / "faithfulness_curves.csv"
    summary_path = output_dir / "faithfulness_summary.csv"
    config_path = output_dir / "faithfulness_config.json"
    if not overwrite and (curves_path.exists() or summary_path.exists()):
        raise FileExistsError(f"{output_dir} already has faithfulness outputs; pass --overwrite")

    importance = pd.read_csv(importance_csv)
    target_col, seed, split = _infer_target_seed_split(importance, target_col, seed, split)
    checkpoint_path = checkpoint_path or (
        Path("results/final_publication")
        / target_col
        / "random_split"
        / "checkpoints"
        / DEFAULT_MODEL_NAME
        / f"seed_{seed}"
        / "model.pt"
    )
    device = resolve_device(device_name)
    store = ProcessedFeatureStore(processed_root, ("tabular", "rdf", "structure"))

    sample_ids = importance["sample_id"].astype(str).drop_duplicates().tolist()
    first_id = sample_ids[0]
    first_tab = store.load("tabular", first_id)["features"].float().unsqueeze(0)
    first_rdf = store.load("rdf", first_id)["features"].float().unsqueeze(0)
    first_graph = store.load("structure", first_id)["features"]
    model = load_mid_tri_model(
        checkpoint_path=Path(checkpoint_path),
        tabular_dim=int(first_tab.shape[-1]),
        rdf_dim=int(first_rdf.shape[-1]),
        atom_fea_dim=int(first_graph["atom_fea"].shape[-1]),
        nbr_fea_dim=int(first_graph["nbr_fea"].shape[-1]),
        device=device,
    )

    rows = []
    for sample_offset, sample_id in enumerate(sample_ids):
        sample_rows = importance[importance["sample_id"].astype(str) == sample_id].reset_index(drop=True)
        if "modality" in sample_rows.columns:
            sample_rows = sample_rows[sample_rows["modality"].fillna(modality) == modality].reset_index(drop=True)
        if sample_rows.empty:
            LOGGER.warning("Skipping %s because no %s rows are present", sample_id, modality)
            continue
        tab = store.load("tabular", sample_id)["features"].float().unsqueeze(0)
        rdf = store.load("rdf", sample_id)["features"].float().unsqueeze(0)
        graph = store.load("structure", sample_id)["features"]
        y_true = float(sample_rows["y_true"].iloc[0])
        original_pred = _predict_mid_tri(model, tab, rdf, graph, device=device)
        original_error = abs(y_true - original_pred)
        feature_importance = sample_rows["importance"].to_numpy(dtype=float)
        counts = ablation_schedule(len(feature_importance), fractions)
        for order in orders:
            for fraction, count in zip(fractions, counts):
                selected = ablation_indices(
                    feature_importance,
                    order=order,
                    n_remove=count,
                    seed=random_seed + sample_offset,
                )
                ablated_tab, ablated_rdf, ablated_graph = _ablate_features(
                    tab=tab,
                    rdf=rdf,
                    graph=graph,
                    feature_rows=sample_rows,
                    selected_positions=selected,
                    modality=modality,
                )
                ablated_pred = _predict_mid_tri(model, ablated_tab, ablated_rdf, ablated_graph, device=device)
                ablated_error = abs(y_true - ablated_pred)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "target_col": target_col,
                        "seed": seed,
                        "split": split,
                        "modality": modality,
                        "order": order,
                        "ablation_fraction": float(fraction),
                        "n_features": len(feature_importance),
                        "n_ablated": int(count),
                        "y_true": y_true,
                        "y_pred_original": original_pred,
                        "y_pred_ablated": ablated_pred,
                        "prediction_delta": abs(original_pred - ablated_pred),
                        "original_error": original_error,
                        "ablated_error": ablated_error,
                        "error_delta": ablated_error - original_error,
                    }
                )

    curves = pd.DataFrame(rows)
    summary = summarize_faithfulness(curves)
    curves.to_csv(curves_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "importance_csv": str(importance_csv),
                "checkpoint_path": str(checkpoint_path),
                "processed_root": str(processed_root),
                "target_col": target_col,
                "seed": seed,
                "split": split,
                "modality": modality,
                "fractions": list(fractions),
                "orders": list(orders),
                "random_seed": random_seed,
            },
            indent=2,
        )
    )
    LOGGER.info("Wrote %s and %s", curves_path, summary_path)
    return curves, summary


def _auc(frame: pd.DataFrame, value_col: str) -> float:
    ordered = frame.sort_values("ablation_fraction")
    return float(np.trapz(ordered[value_col].to_numpy(), ordered["ablation_fraction"].to_numpy()))


def summarize_faithfulness(curves: pd.DataFrame) -> pd.DataFrame:
    """Summarize deletion curves as per-order AUC and final deltas."""
    per_sample_rows = []
    for (sample_id, order), group in curves.groupby(["sample_id", "order"], sort=False):
        final = group.sort_values("ablation_fraction").iloc[-1]
        per_sample_rows.append(
            {
                "sample_id": sample_id,
                "order": order,
                "prediction_delta_auc": _auc(group, "prediction_delta"),
                "error_delta_auc": _auc(group, "error_delta"),
                "final_prediction_delta": float(final["prediction_delta"]),
                "final_error_delta": float(final["error_delta"]),
            }
        )
    per_sample = pd.DataFrame(per_sample_rows)
    summary = (
        per_sample.groupby("order")
        .agg(
            n_samples=("sample_id", "nunique"),
            prediction_delta_auc_mean=("prediction_delta_auc", "mean"),
            prediction_delta_auc_std=("prediction_delta_auc", "std"),
            error_delta_auc_mean=("error_delta_auc", "mean"),
            error_delta_auc_std=("error_delta_auc", "std"),
            final_prediction_delta_mean=("final_prediction_delta", "mean"),
            final_prediction_delta_std=("final_prediction_delta", "std"),
            final_error_delta_mean=("final_error_delta", "mean"),
            final_error_delta_std=("final_error_delta", "std"),
        )
        .reset_index()
    )
    if "random" in set(summary["order"]):
        random_pred = float(summary.loc[summary["order"] == "random", "prediction_delta_auc_mean"].iloc[0])
        random_error = float(summary.loc[summary["order"] == "random", "error_delta_auc_mean"].iloc[0])
        summary["prediction_auc_minus_random"] = summary["prediction_delta_auc_mean"] - random_pred
        summary["error_auc_minus_random"] = summary["error_delta_auc_mean"] - random_error
    else:
        summary["prediction_auc_minus_random"] = np.nan
        summary["error_auc_minus_random"] = np.nan
    return summary


def plot_faithfulness_curves(
    curves: pd.DataFrame,
    output_path: Path,
    value_col: str = "prediction_delta",
) -> None:
    import matplotlib.pyplot as plt

    grouped = (
        curves.groupby(["order", "ablation_fraction"])[value_col]
        .agg(["mean", "sem"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for order, group in grouped.groupby("order", sort=False):
        ax.errorbar(
            group["ablation_fraction"],
            group["mean"],
            yerr=group["sem"].fillna(0.0),
            marker="o",
            linewidth=1.8,
            capsize=2.5,
            label=order,
        )
    ax.set_xlabel("Ablated feature fraction")
    ax.set_ylabel("Prediction change" if value_col == "prediction_delta" else "Error change")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _parse_fractions(values: list[str] | None) -> list[float]:
    if not values:
        return list(DEFAULT_FRACTIONS)
    return [float(value) for value in values]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate explanation faithfulness by importance-guided ablation.")
    parser.add_argument("--importance_csv", type=Path, required=True)
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", choices=["train", "val", "test"], default=None)
    parser.add_argument("--modality", choices=["composition", "rdf", "structure"], required=True)
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/legacy_rdf_split_seed_42"))
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--fractions", nargs="*", default=None)
    parser.add_argument("--orders", nargs="+", choices=["top", "random", "bottom"], default=["top", "random", "bottom"])
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    curves, summary = run_faithfulness_validation(
        importance_csv=args.importance_csv,
        target_col=args.target_col,
        seed=args.seed,
        split=args.split,
        modality=args.modality,
        processed_root=args.processed_root,
        checkpoint_path=args.checkpoint_path,
        output_dir=args.output_dir,
        device_name=args.device,
        fractions=_parse_fractions(args.fractions),
        orders=args.orders,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
