from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from battery_fusion.fusion.feature_store import ProcessedFeatureStore
from battery_fusion.explain.composition_importance import DEFAULT_MODEL_NAME, load_mid_tri_model, resolve_device, select_samples
from battery_fusion.explain.faithfulness import _predict_mid_tri
from battery_fusion.explain.permutation import (
    _load_selected_features,
    metric_delta_rows,
    run_permutation_importance,
)
from battery_fusion.training.metrics import regression_metrics

LOGGER = logging.getLogger(__name__)
DEFAULT_MODALITIES = ("composition", "rdf", "structure")
FEATURE_STORE_MODALITIES = ("tabular", "rdf", "structure")


def combine_full_3modal_summaries(summaries: list[pd.DataFrame]) -> pd.DataFrame:
    """Combine per-modality permutation summaries into one cross-modal ranking."""
    if not summaries:
        return pd.DataFrame()
    frame = pd.concat(summaries, ignore_index=True).copy()
    frame["source_modality"] = frame["modality"].astype(str)
    frame["modality_feature"] = frame["source_modality"] + ":" + frame["feature_group"].astype(str)
    sort_cols = [col for col in ["delta_mae_mean", "delta_rmse_mean"] if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    return frame.reset_index(drop=True)


def aggregate_full_3modal_across_seeds(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the combined full 3-modal permutation ranking across seeds."""
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(["target_col", "split", "source_modality", "feature_group", "feature_indices", "modality_feature"])
        .agg(
            n_seeds=("seed", "nunique"),
            n_samples_mean=("n_samples", "mean"),
            delta_mae_seed_mean=("delta_mae_mean", "mean"),
            delta_mae_seed_std=("delta_mae_mean", "std"),
            delta_rmse_seed_mean=("delta_rmse_mean", "mean"),
            delta_rmse_seed_std=("delta_rmse_mean", "std"),
            delta_r2_seed_mean=("delta_r2_mean", "mean"),
            delta_r2_seed_std=("delta_r2_mean", "std"),
        )
        .reset_index()
        .sort_values("delta_mae_seed_mean", ascending=False)
    )
    return grouped


def run_all_modalities_block_permutation(
    target_col: str,
    seed: int,
    split: str,
    processed_root: Path,
    split_dir: Path,
    checkpoint_path: Path | None,
    output_dir: Path,
    sample_index: int = 0,
    max_samples: int | None = 100,
    repeats: int = 5,
    random_seed: int = 0,
    device_name: str = "cpu",
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permute all three modality inputs together as a modality-block sanity check."""
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "all_modalities_permutation_metrics.csv"
    summary_path = output_dir / "all_modalities_permutation_summary.csv"
    config_path = output_dir / "all_modalities_permutation_config.json"
    if not overwrite and (metrics_path.exists() or summary_path.exists()):
        raise FileExistsError(f"{output_dir} already has all-modality permutation outputs; pass --overwrite")

    split_frame = pd.read_csv(Path(split_dir) / target_col / f"seed_{seed}" / f"{split}.csv")
    if max_samples is None or max_samples <= 0:
        max_samples = len(split_frame) - sample_index
    selected = select_samples(split_frame, sample_id=None, sample_index=sample_index, max_samples=max_samples)
    store = ProcessedFeatureStore(processed_root, FEATURE_STORE_MODALITIES)
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
    baseline_pred = torch.tensor(
        [_predict_mid_tri(model, tabs[idx : idx + 1], rdfs[idx : idx + 1], graph, device=device) for idx, graph in enumerate(graphs)],
        dtype=torch.float32,
    )
    baseline = regression_metrics(y_true, baseline_pred)
    rows = []
    for repeat in range(repeats):
        rng = np.random.default_rng(random_seed + repeat)
        tab_perm = rng.permutation(len(sample_ids))
        rdf_perm = rng.permutation(len(sample_ids))
        graph_perm = rng.permutation(len(sample_ids))
        perm_tabs = tabs[torch.as_tensor(tab_perm, dtype=torch.long)]
        perm_rdfs = rdfs[torch.as_tensor(rdf_perm, dtype=torch.long)]
        perm_graphs = [graphs[int(idx)] for idx in graph_perm]
        perm_pred = torch.tensor(
            [_predict_mid_tri(model, perm_tabs[idx : idx + 1], perm_rdfs[idx : idx + 1], graph, device=device) for idx, graph in enumerate(perm_graphs)],
            dtype=torch.float32,
        )
        perm_metrics = regression_metrics(y_true, perm_pred)
        rows.extend(
            metric_delta_rows(
                baseline=baseline,
                permuted=perm_metrics,
                metadata={
                    "target_col": target_col,
                    "seed": seed,
                    "split": split,
                    "modality": "full_3modal",
                    "feature_group": "all_modalities_block",
                    "feature_indices": "all",
                    "repeat": repeat,
                    "n_samples": len(sample_ids),
                },
            )
        )

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
    )
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "target_col": target_col,
                "seed": seed,
                "split": split,
                "processed_root": str(processed_root),
                "split_dir": str(split_dir),
                "checkpoint_path": str(checkpoint_path),
                "sample_index": sample_index,
                "max_samples": max_samples,
                "repeats": repeats,
                "random_seed": random_seed,
                "device": device_name,
            },
            indent=2,
        )
    )
    return metrics, summary


def run_permutation_importance_matrix(
    target_col: str,
    seeds: list[int],
    output_dir: Path,
    split: str = "test",
    modalities: tuple[str, ...] = DEFAULT_MODALITIES,
    processed_root: Path = Path("data/processed/legacy_rdf_split_seed_42"),
    split_dir: Path = Path("data/splits/publication"),
    checkpoint_root: Path | None = None,
    labels_path: Path = Path("data/labels/labels_keep_last.csv"),
    sample_index: int = 0,
    max_samples: int | None = 100,
    rdf_group_size: int = 10,
    repeats: int = 5,
    random_seed: int = 0,
    device_name: str = "cpu",
    include_all_modalities_block: bool = True,
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_seed_frames = []
    output_root = output_dir / target_col
    for seed in seeds:
        seed_summaries = []
        seed_dir = output_root / f"seed_{seed}"
        checkpoint_path = (
            checkpoint_root / target_col / "random_split" / "checkpoints" / DEFAULT_MODEL_NAME / f"seed_{seed}" / "model.pt"
            if checkpoint_root is not None
            else None
        )
        for modality in modalities:
            modality_dir = seed_dir / f"{modality}_permutation"
            LOGGER.info("Running %s permutation for %s seed %s", modality, target_col, seed)
            _, summary = run_permutation_importance(
                target_col=target_col,
                seed=seed,
                split=split,
                modality=modality,
                processed_root=processed_root,
                split_dir=split_dir,
                checkpoint_path=checkpoint_path,
                labels_path=labels_path,
                output_dir=modality_dir,
                sample_index=sample_index,
                max_samples=max_samples,
                group_size=rdf_group_size if modality == "rdf" else 1,
                repeats=repeats,
                random_seed=random_seed + seed * 10000,
                device_name=device_name,
                overwrite=overwrite,
            )
            seed_summaries.append(summary)
        if include_all_modalities_block:
            _, block_summary = run_all_modalities_block_permutation(
                target_col=target_col,
                seed=seed,
                split=split,
                processed_root=processed_root,
                split_dir=split_dir,
                checkpoint_path=checkpoint_path,
                output_dir=seed_dir / "all_modalities_block_permutation",
                sample_index=sample_index,
                max_samples=max_samples,
                repeats=repeats,
                random_seed=random_seed + seed * 10000 + 500000,
                device_name=device_name,
                overwrite=overwrite,
            )
            seed_summaries.append(block_summary)
        full_summary = combine_full_3modal_summaries(seed_summaries)
        full_dir = seed_dir / "full_3modal_permutation"
        full_dir.mkdir(parents=True, exist_ok=True)
        full_summary.to_csv(full_dir / "permutation_importance_summary.csv", index=False)
        (full_dir / "permutation_importance_config.json").write_text(
            json.dumps(
                {
                    "target_col": target_col,
                    "seed": seed,
                    "split": split,
                    "modalities": list(modalities),
                    "include_all_modalities_block": include_all_modalities_block,
                    "sample_index": sample_index,
                    "max_samples": max_samples,
                    "rdf_group_size": rdf_group_size,
                    "repeats": repeats,
                    "random_seed": random_seed,
                    "device": device_name,
                },
                indent=2,
            )
        )
        all_seed_frames.append(full_summary)

    all_seed_summary = pd.concat(all_seed_frames, ignore_index=True) if all_seed_frames else pd.DataFrame()
    aggregate = aggregate_full_3modal_across_seeds(all_seed_summary)
    output_root.mkdir(parents=True, exist_ok=True)
    all_seed_summary.to_csv(output_root / "full_3modal_permutation_summary_all_seeds.csv", index=False)
    aggregate.to_csv(output_root / "full_3modal_permutation_summary_by_feature.csv", index=False)
    return all_seed_summary, aggregate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run composition/RDF/structure and full 3-modal permutation importance across seeds.")
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--modalities", nargs="+", choices=list(DEFAULT_MODALITIES), default=list(DEFAULT_MODALITIES))
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/legacy_rdf_split_seed_42"))
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits/publication"))
    parser.add_argument("--checkpoint_root", type=Path, default=Path("results/final_publication"))
    parser.add_argument("--labels_path", type=Path, default=Path("data/labels/labels_keep_last.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/explanation_validation/permutation_matrix"))
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--rdf_group_size", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--skip_all_modalities_block", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    _, aggregate = run_permutation_importance_matrix(
        target_col=args.target_col,
        seeds=args.seeds,
        split=args.split,
        modalities=tuple(args.modalities),
        processed_root=args.processed_root,
        split_dir=args.split_dir,
        checkpoint_root=args.checkpoint_root,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        sample_index=args.sample_index,
        max_samples=args.max_samples,
        rdf_group_size=args.rdf_group_size,
        repeats=args.repeats,
        random_seed=args.random_seed,
        device_name=args.device,
        include_all_modalities_block=not args.skip_all_modalities_block,
        overwrite=args.overwrite,
    )
    print(aggregate.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
