from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from battery_fusion.fusion.feature_store import ProcessedFeatureStore
from battery_fusion.explain.composition_importance import (
    DEFAULT_MODEL_NAME,
    load_feature_names,
    load_mid_tri_model,
    resolve_device,
    select_samples,
)
from battery_fusion.explain.faithfulness import _predict_mid_tri
from battery_fusion.training.metrics import regression_metrics

LOGGER = logging.getLogger(__name__)


def build_feature_groups(n_features: int, group_size: int, prefix: str) -> list[tuple[str, list[int]]]:
    if n_features <= 0:
        raise ValueError("n_features must be positive")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    groups = []
    for start in range(0, n_features, group_size):
        end = min(n_features, start + group_size)
        groups.append((f"{prefix}_{start}_{end - 1}", list(range(start, end))))
    return groups


def permuted_matrix(matrix: np.ndarray, columns: list[int], seed: int) -> np.ndarray:
    output = np.asarray(matrix).copy()
    if len(columns) == 0:
        return output
    permutation = np.random.default_rng(seed).permutation(output.shape[0])
    output[:, columns] = output[permutation][:, columns]
    return output


def metric_delta_rows(
    baseline: dict[str, float],
    permuted: dict[str, float],
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    row = dict(metadata)
    for key in ["mae", "mse", "rmse", "r2"]:
        row[f"baseline_{key}"] = float(baseline[key])
        row[f"permuted_{key}"] = float(permuted[key])
        row[f"delta_{key}"] = float(permuted[key] - baseline[key])
    return [row]


def _load_selected_features(
    store: ProcessedFeatureStore,
    split_frame: pd.DataFrame,
) -> tuple[list[str], torch.Tensor, torch.Tensor, list[dict[str, torch.Tensor]], torch.Tensor]:
    sample_ids = split_frame["sample_id"].astype(str).tolist()
    tabs = []
    rdfs = []
    graphs = []
    y_true = []
    for _, row in split_frame.iterrows():
        sample_id = str(row["sample_id"])
        tabs.append(store.load("tabular", sample_id)["features"].float())
        rdfs.append(store.load("rdf", sample_id)["features"].float())
        graphs.append(store.load("structure", sample_id)["features"])
        y_true.append(float(row["target"]))
    return sample_ids, torch.stack(tabs), torch.stack(rdfs), graphs, torch.tensor(y_true, dtype=torch.float32)


def _predict_samples(
    model: torch.nn.Module,
    tabs: torch.Tensor,
    rdfs: torch.Tensor,
    graphs: list[dict[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    preds = []
    for idx, graph in enumerate(graphs):
        preds.append(_predict_mid_tri(model, tabs[idx : idx + 1], rdfs[idx : idx + 1], graph, device=device))
    return torch.tensor(preds, dtype=torch.float32)


def _feature_group_names(
    modality: str,
    n_features: int,
    group_size: int,
    labels_path: Path,
) -> list[tuple[str, list[int]]]:
    if modality == "composition":
        names = load_feature_names(labels_path, expected_dim=n_features)
        base_groups = build_feature_groups(n_features, group_size=1, prefix="composition")
        return [(names[indices[0]], indices) for _, indices in base_groups]
    if modality == "rdf":
        return build_feature_groups(n_features, group_size=group_size, prefix="rdf")
    raise ValueError(f"Unsupported feature-group modality {modality!r}")


def run_permutation_importance(
    target_col: str,
    seed: int,
    split: str,
    modality: str,
    processed_root: Path,
    split_dir: Path,
    checkpoint_path: Path | None,
    labels_path: Path,
    output_dir: Path,
    sample_index: int = 0,
    max_samples: int | None = 100,
    group_size: int = 10,
    repeats: int = 5,
    random_seed: int = 0,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "permutation_importance_metrics.csv"
    summary_path = output_dir / "permutation_importance_summary.csv"
    config_path = output_dir / "permutation_importance_config.json"
    if not overwrite and (metrics_path.exists() or summary_path.exists()):
        raise FileExistsError(f"{output_dir} already has permutation outputs; pass --overwrite")

    split_frame = pd.read_csv(Path(split_dir) / target_col / f"seed_{seed}" / f"{split}.csv")
    if max_samples is None or max_samples <= 0:
        max_samples = len(split_frame) - sample_index
    selected = select_samples(split_frame, sample_id=None, sample_index=sample_index, max_samples=max_samples)
    store = ProcessedFeatureStore(processed_root, ("tabular", "rdf", "structure"))
    sample_ids, tabs, rdfs, graphs, y_true = _load_selected_features(store, selected)
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
        tabular_dim=int(tabs.shape[-1]),
        rdf_dim=int(rdfs.shape[-1]),
        atom_fea_dim=int(graphs[0]["atom_fea"].shape[-1]),
        nbr_fea_dim=int(graphs[0]["nbr_fea"].shape[-1]),
        device=device,
    )

    baseline_pred = _predict_samples(model, tabs, rdfs, graphs, device=device)
    baseline = regression_metrics(y_true, baseline_pred)
    rows = []
    if modality in {"composition", "rdf"}:
        matrix = tabs.numpy() if modality == "composition" else rdfs.numpy()
        groups = _feature_group_names(
            modality=modality,
            n_features=matrix.shape[1],
            group_size=group_size,
            labels_path=labels_path,
        )
        for group_idx, (feature_group, columns) in enumerate(groups):
            for repeat in range(repeats):
                permuted = permuted_matrix(matrix, columns=columns, seed=random_seed + group_idx * 1009 + repeat)
                perm_tabs = torch.tensor(permuted, dtype=torch.float32) if modality == "composition" else tabs
                perm_rdfs = torch.tensor(permuted, dtype=torch.float32) if modality == "rdf" else rdfs
                perm_pred = _predict_samples(model, perm_tabs, perm_rdfs, graphs, device=device)
                perm_metrics = regression_metrics(y_true, perm_pred)
                rows.extend(
                    metric_delta_rows(
                        baseline=baseline,
                        permuted=perm_metrics,
                        metadata={
                            "target_col": target_col,
                            "seed": seed,
                            "split": split,
                            "modality": modality,
                            "feature_group": feature_group,
                            "feature_indices": ",".join(str(index) for index in columns),
                            "repeat": repeat,
                            "n_samples": len(sample_ids),
                        },
                    )
                )
    elif modality == "structure":
        for repeat in range(repeats):
            permutation = np.random.default_rng(random_seed + repeat).permutation(len(graphs))
            perm_graphs = [graphs[int(idx)] for idx in permutation]
            perm_pred = _predict_samples(model, tabs, rdfs, perm_graphs, device=device)
            perm_metrics = regression_metrics(y_true, perm_pred)
            rows.extend(
                metric_delta_rows(
                    baseline=baseline,
                    permuted=perm_metrics,
                    metadata={
                        "target_col": target_col,
                        "seed": seed,
                        "split": split,
                        "modality": modality,
                        "feature_group": "whole_structure",
                        "feature_indices": "all",
                        "repeat": repeat,
                        "n_samples": len(sample_ids),
                    },
                )
            )
    else:
        raise ValueError("modality must be composition, rdf, or structure")

    metrics = pd.DataFrame(rows)
    summary = (
        metrics.groupby(["target_col", "seed", "split", "modality", "feature_group", "feature_indices", "n_samples"])
        .agg(
            repeats=("repeat", "nunique"),
            delta_mae_mean=("delta_mae", "mean"),
            delta_mae_std=("delta_mae", "std"),
            delta_rmse_mean=("delta_rmse", "mean"),
            delta_rmse_std=("delta_rmse", "std"),
            delta_r2_mean=("delta_r2", "mean"),
            delta_r2_std=("delta_r2", "std"),
            baseline_mae=("baseline_mae", "first"),
            baseline_rmse=("baseline_rmse", "first"),
            baseline_r2=("baseline_r2", "first"),
        )
        .reset_index()
        .sort_values("delta_mae_mean", ascending=False)
    )
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "target_col": target_col,
                "seed": seed,
                "split": split,
                "modality": modality,
                "processed_root": str(processed_root),
                "split_dir": str(split_dir),
                "checkpoint_path": str(checkpoint_path),
                "sample_index": sample_index,
                "max_samples": max_samples,
                "group_size": group_size,
                "repeats": repeats,
                "random_seed": random_seed,
                "device": device_name,
            },
            indent=2,
        )
    )
    LOGGER.info("Wrote %s and %s", metrics_path, summary_path)
    return metrics, summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standard permutation importance for final mid-tri fusion inputs.")
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--modality", choices=["composition", "rdf", "structure"], required=True)
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/legacy_rdf_split_seed_42"))
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits/publication"))
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--labels_path", type=Path, default=Path("data/labels/labels_keep_last.csv"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--group_size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    _, summary = run_permutation_importance(
        target_col=args.target_col,
        seed=args.seed,
        split=args.split,
        modality=args.modality,
        processed_root=args.processed_root,
        split_dir=args.split_dir,
        checkpoint_path=args.checkpoint_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        sample_index=args.sample_index,
        max_samples=args.max_samples,
        group_size=args.group_size,
        repeats=args.repeats,
        random_seed=args.random_seed,
        device_name=args.device,
        overwrite=args.overwrite,
    )
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
