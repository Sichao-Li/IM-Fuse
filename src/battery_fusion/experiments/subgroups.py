from __future__ import annotations

import argparse
import logging
from pathlib import Path
from collections.abc import Sequence

import numpy as np
import pandas as pd

from battery_fusion.utils.chemistry_groups import (
    ANION_FAMILIES,
    WORKING_ION_GROUPS,
    load_assignments,
)

LOGGER = logging.getLogger(__name__)
UNIMODAL_MODALITY_SETS = {"composition", "tabular", "graph", "structure", "rdf"}


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(diff**2) / total) if total > 0 else np.nan
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def compute_subgroup_metrics(
    predictions: pd.DataFrame,
    min_group_size: int = 30,
) -> pd.DataFrame:
    required = {"y_true", "y_pred", "model_name", "modality_set", "seed"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")

    frame = predictions.copy()
    if "anion_family" not in frame.columns:
        frame["anion_family"] = "other"
    if "working_ion" not in frame.columns:
        frame["working_ion"] = "other"
    frame["anion_family"] = frame["anion_family"].fillna("other")
    frame["working_ion"] = frame["working_ion"].fillna("other")
    frame["modality_set"] = frame["modality_set"].astype(str)
    frame["model_name"] = frame["model_name"].astype(str)

    rows: list[dict[str, object]] = []
    group_specs = [
        ("anion_family", "anion_family", ANION_FAMILIES),
        ("working_ion", "working_ion", WORKING_ION_GROUPS),
    ]
    for group_type, column, names in group_specs:
        for (model_name, modality_set, seed), model_frame in frame.groupby(
            ["model_name", "modality_set", "seed"], dropna=False
        ):
            for group_name in names:
                subset = model_frame[model_frame[column] == group_name]
                row: dict[str, object] = {
                    "group_type": group_type,
                    "group_name": group_name,
                    "model_name": model_name,
                    "modality_set": modality_set,
                    "seed": seed,
                    "n_samples": int(len(subset)),
                    "unreliable": bool(len(subset) < min_group_size),
                    "delta_MAE_vs_best_unimodal": np.nan,
                    "delta_MAE_vs_CGCNN": np.nan,
                }
                if len(subset) >= min_group_size:
                    metrics = _regression_metrics(
                        subset["y_true"].astype(float).to_numpy(),
                        subset["y_pred"].astype(float).to_numpy(),
                    )
                    row.update(metrics)
                else:
                    row.update({"MAE": np.nan, "MSE": np.nan, "RMSE": np.nan, "R2": np.nan})
                rows.append(row)

    metrics = pd.DataFrame(rows)
    if not metrics.empty:
        metrics = _add_delta_columns(metrics)
    return metrics


def _add_delta_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    output = metrics.copy()
    key_cols = ["group_type", "group_name", "seed"]
    for key, group in output.groupby(key_cols, dropna=False):
        valid = group[group["MAE"].notna()]
        if valid.empty:
            continue
        modality_sets = valid["modality_set"].astype(str).str.lower()
        model_names = valid["model_name"].astype(str).str.lower()
        unimodal_mask = modality_sets.isin(UNIMODAL_MODALITY_SETS) & ~modality_sets.str.contains(r"\+")
        unimodal = valid[unimodal_mask]
        cgcnn_mask = (
            modality_sets.isin({"graph", "structure"})
            | model_names.str.contains("cgcnn")
            | model_names.str.contains("graph")
            | model_names.str.contains("structure")
        )
        cgcnn = valid[cgcnn_mask]
        if not unimodal.empty:
            baseline = float(unimodal["MAE"].min())
            selector = (output[key_cols] == pd.Series(key, index=key_cols)).all(axis=1)
            output.loc[selector, "delta_MAE_vs_best_unimodal"] = output.loc[selector, "MAE"] - baseline
        if not cgcnn.empty:
            baseline = float(cgcnn["MAE"].min())
            selector = (output[key_cols] == pd.Series(key, index=key_cols)).all(axis=1)
            output.loc[selector, "delta_MAE_vs_CGCNN"] = output.loc[selector, "MAE"] - baseline
    return output


def load_prediction_files(
    predictions_dir: Path | Sequence[Path],
    split: str | None = None,
) -> pd.DataFrame:
    prediction_roots = (
        [Path(predictions_dir)]
        if isinstance(predictions_dir, (str, Path))
        else [Path(path) for path in predictions_dir]
    )
    paths = [
        path
        for root in prediction_roots
        for path in sorted(root.rglob("*_predictions.csv"))
    ]
    if not paths:
        raise FileNotFoundError(
            "No *_predictions.csv files found under "
            + ", ".join(str(root) for root in prediction_roots)
        )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if split is not None:
            if "split" in frame.columns:
                frame = frame[frame["split"].astype(str) == str(split)].copy()
            elif f"_{split}_" not in path.name:
                continue
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No prediction rows matched split={split!r} under "
            + ", ".join(str(root) for root in prediction_roots)
        )
    return pd.concat(frames, ignore_index=True)


def merge_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    metadata = metadata.copy()
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    output["sample_id"] = output["sample_id"].astype(str)
    keep_cols = ["sample_id", "formula", "working_ion", "anion_family"]
    metadata = metadata[[column for column in keep_cols if column in metadata.columns]]
    for column in ["formula", "working_ion", "anion_family"]:
        if column in output.columns:
            output = output.drop(columns=[column])
    return output.merge(metadata, on="sample_id", how="left")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate subgroup metrics.")
    parser.add_argument("--predictions_dir", type=Path, nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--working_ion_col", default="working_ion")
    parser.add_argument("--target_col", default="target")
    parser.add_argument("--output_dir", type=Path, default=Path("results/subgroup_analysis"))
    parser.add_argument("--min_group_size", type=int, default=30)
    parser.add_argument("--split", default=None, help="Optional split label to filter predictions, e.g. test.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    predictions = load_prediction_files(args.predictions_dir, split=args.split)
    metadata = load_assignments(args.metadata)
    predictions = merge_metadata(predictions, metadata)
    metrics = compute_subgroup_metrics(predictions, min_group_size=args.min_group_size)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "subgroup_metrics.csv"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
    metrics.to_csv(output_path, index=False)
    LOGGER.info("Wrote subgroup metrics to %s", output_path)


if __name__ == "__main__":
    main()
