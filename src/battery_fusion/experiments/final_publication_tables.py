from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ("average_voltage", "capacity_vol")
SPLITS = ("train", "val", "test")
SUMMARY_METRICS = ("MAE", "MSE", "RMSE", "R2")

MODEL_LABELS = {
    "unimodal_rdf_sequence": "RDF sequence encoder",
    "unimodal_tabular": "Composition neural",
    "unimodal_structure": "CGCNN-style graph",
    "late_dual_rdf_tabular": "RDF + composition late fusion",
    "late_dual_rdf_structure": "RDF + graph late fusion",
    "late_dual_tabular_structure": "Composition + graph late fusion",
    "early_tri_rdf_tabular_structure": "Tri-fusion early",
    "mid_tri_rdf_tabular_structure": "Tri-fusion mid",
    "late_tri_rdf_tabular_structure": "Tri-fusion late",
    "random_forest": "Random forest composition",
    "xgboost": "XGBoost composition",
    "alignn_pretrained_rf": "ALIGNN pretrained + RF",
    "composition": "Composition",
    "graph": "Graph",
    "composition_graph": "Composition + graph",
    "full_fusion": "Full fusion",
}

SOURCE_LABELS = {
    "neural/fusion": "neural/fusion",
    "classical": "classical composition baseline",
    "alignn_pretrained": "pretrained structure baseline",
}

RANDOM_MODEL_ORDER = [
    "unimodal_rdf_sequence",
    "unimodal_tabular",
    "unimodal_structure",
    "late_dual_rdf_tabular",
    "late_dual_rdf_structure",
    "late_dual_tabular_structure",
    "early_tri_rdf_tabular_structure",
    "mid_tri_rdf_tabular_structure",
    "late_tri_rdf_tabular_structure",
    "random_forest",
    "xgboost",
    "alignn_pretrained_rf",
]

HOLDOUT_MODEL_ORDER = [
    "composition",
    "random_forest",
    "xgboost",
    "graph",
    "composition_graph",
    "full_fusion",
    "alignn_pretrained_rf",
]

MANIFEST_ROOTS = [
    Path("results/final_publication"),
    Path("results/predictions"),
    Path("results/explanation_validation"),
    Path("figures/final_publication"),
    Path("figures/explanation_validation"),
]

MANIFEST_EXCLUDE_PATTERNS = [
    "m3" + "gnet",
    "mat" + "gl",
    "target_" + "standardized",
    "standard" + "ized",
    "gated_fusion",
    "alignn_anion_holdout",
    "alignn_runs",
    "alignn_outputs",
    "diagnostic",
    "checkpoints",
    "models",
    "embeddings",
    "pretrained_features",
    "staged_structures",
    "/partial/",
    "/manifests/",
]

MANIFEST_ALLOWED_SUFFIXES = {".csv", ".json", ".pdf", ".md"}


def _metric_summary(
    frame: pd.DataFrame,
    group_cols: list[str],
    split: str | None = "test",
) -> pd.DataFrame:
    metrics = frame.copy()
    if split is not None and "split" in metrics.columns:
        metrics = metrics[metrics["split"].astype(str) == split].copy()
    if metrics.empty:
        return pd.DataFrame()
    for column in ["n_samples", "n_train", "n_val", "n_test"]:
        if column not in metrics.columns:
            metrics[column] = float("nan")
    return (
        metrics.groupby(group_cols, dropna=False)
        .agg(
            seeds=("seed", "nunique"),
            n_samples=("n_samples", "mean"),
            n_train=("n_train", "mean"),
            n_val=("n_val", "mean"),
            n_test=("n_test", "mean"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_std=("R2", "std"),
        )
        .reset_index()
    )


def _split_n_samples(frame: pd.DataFrame, split: str) -> pd.Series:
    if "n_samples" in frame.columns and not frame["n_samples"].isna().all():
        return frame["n_samples"]
    split_count_col = f"n_{split}"
    if split_count_col in frame.columns:
        return frame[split_count_col]
    return pd.Series(float("nan"), index=frame.index)


def _split_metric_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if "split" not in frame.columns:
        return pd.DataFrame(columns=group_cols)
    metrics = frame.copy()
    for metric in SUMMARY_METRICS:
        if metric not in metrics.columns:
            metrics[metric] = float("nan")
    summaries: list[pd.DataFrame] = []
    for split in SPLITS:
        split_metrics = metrics[metrics["split"].astype(str) == split].copy()
        if split_metrics.empty:
            continue
        split_metrics["n_samples_for_split"] = _split_n_samples(split_metrics, split)
        summary = (
            split_metrics.groupby(group_cols, dropna=False)
            .agg(
                **{
                    f"{split}_seeds": ("seed", "nunique"),
                    f"{split}_n_samples": ("n_samples_for_split", "mean"),
                    **{
                        f"{split}_{metric}_mean": (metric, "mean")
                        for metric in SUMMARY_METRICS
                    },
                    **{
                        f"{split}_{metric}_std": (metric, "std")
                        for metric in SUMMARY_METRICS
                    },
                }
            )
            .reset_index()
        )
        summaries.append(summary)
    if not summaries:
        return pd.DataFrame(columns=group_cols)
    output = summaries[0]
    for summary in summaries[1:]:
        output = output.merge(summary, on=group_cols, how="outer")
    return output


def _split_counts_summary(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    count_cols = [column for column in ["n_train", "n_val", "n_test"] if column in frame.columns]
    if not count_cols:
        return pd.DataFrame(columns=group_cols)
    return (
        frame.groupby(group_cols, dropna=False)[count_cols]
        .mean()
        .reset_index()
    )


def _merge_split_metrics(base: pd.DataFrame, metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    output = base.copy()
    usable_group_cols = [column for column in group_cols if column in output.columns and column in metrics.columns]
    if not usable_group_cols:
        return output
    split_counts = _split_counts_summary(metrics, usable_group_cols)
    if not split_counts.empty:
        output = output.merge(split_counts, on=usable_group_cols, how="left", suffixes=("", "_from_metrics"))
        for column in ["n_train", "n_val", "n_test"]:
            fallback = f"{column}_from_metrics"
            if fallback not in output.columns:
                continue
            if column in output.columns:
                output[column] = output[column].fillna(output[fallback])
            else:
                output[column] = output[fallback]
            output = output.drop(columns=[fallback])
    split_metrics = _split_metric_summary(metrics, usable_group_cols)
    if not split_metrics.empty:
        output = output.merge(split_metrics, on=usable_group_cols, how="left")
    return output


def _regression_metrics_from_predictions(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denominator == 0.0 else float(1.0 - np.sum(error**2) / denominator)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def _split_from_prediction_path(path: Path) -> str:
    stem = path.stem
    for split in SPLITS:
        if stem.endswith(f"_{split}_predictions"):
            return split
    return ""


def _metrics_from_prediction_files(predictions_dir: Path) -> pd.DataFrame:
    if not predictions_dir.exists():
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for path in sorted(predictions_dir.glob("*/*_predictions.csv")):
        frame = pd.read_csv(path)
        if frame.empty or not {"y_true", "y_pred"}.issubset(frame.columns):
            continue
        metrics = _regression_metrics_from_predictions(frame)
        split = str(frame["split"].iloc[0]) if "split" in frame.columns else _split_from_prediction_path(path)
        model_name = str(frame["model_name"].iloc[0]) if "model_name" in frame.columns else path.parent.name
        modality_set = str(frame["modality_set"].iloc[0]) if "modality_set" in frame.columns else ""
        seed = int(frame["seed"].iloc[0]) if "seed" in frame.columns else int(path.stem.split("_")[1])
        rows.append(
            {
                "model_name": model_name,
                "modality_set": modality_set,
                "seed": seed,
                "split": split,
                "n_samples": len(frame),
                **metrics,
            }
        )
    if not rows:
        return pd.DataFrame()
    metrics = pd.DataFrame(rows)
    counts = (
        metrics.pivot_table(
            index=["model_name", "seed"],
            columns="split",
            values="n_samples",
            aggfunc="first",
        )
        .rename(columns={split: f"n_{split}" for split in SPLITS})
        .reset_index()
    )
    return metrics.merge(counts, on=["model_name", "seed"], how="left")


def _complete_metrics_with_predictions(metrics: pd.DataFrame, predictions_dir: Path) -> pd.DataFrame:
    prediction_metrics = _metrics_from_prediction_files(predictions_dir)
    if prediction_metrics.empty:
        return metrics
    if metrics.empty:
        return prediction_metrics
    key_cols = ["model_name", "seed", "split"]
    if not all(column in metrics.columns for column in key_cols):
        return pd.concat([metrics, prediction_metrics], ignore_index=True, sort=False)
    existing_keys = set(map(tuple, metrics[key_cols].astype(str).to_numpy()))
    prediction_keys = prediction_metrics[key_cols].astype(str).apply(tuple, axis=1)
    missing_prediction_metrics = prediction_metrics[~prediction_keys.isin(existing_keys)].copy()
    if missing_prediction_metrics.empty:
        return metrics
    return pd.concat([metrics, missing_prediction_metrics], ignore_index=True, sort=False)


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = float("nan")
    return output


def _append_order_and_labels(
    frame: pd.DataFrame,
    model_col: str = "model_name",
    order: list[str] | None = None,
) -> pd.DataFrame:
    output = frame.copy()
    output["model_label"] = output[model_col].map(MODEL_LABELS).fillna(output[model_col])
    if order is not None:
        output["sort_order"] = output[model_col].map({model: idx for idx, model in enumerate(order)})
        output["sort_order"] = output["sort_order"].fillna(len(order)).astype(int)
    return output


def build_random_split_summary(results_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    predictions_root = results_root.parent / "predictions"
    for target in TARGETS:
        target_dir = results_root / target
        random_summary_path = target_dir / "random_split" / "random_seed_summary.csv"
        if random_summary_path.exists():
            random_summary = pd.read_csv(random_summary_path)
            random_summary = random_summary.rename(columns={"n_seeds": "seeds"})
            random_metrics_path = target_dir / "random_split" / "publication_metrics.csv"
            if random_metrics_path.exists():
                random_metrics = pd.read_csv(random_metrics_path)
                random_metrics = _complete_metrics_with_predictions(
                    random_metrics,
                    predictions_root / "final_publication_random" / target,
                )
                random_summary = _merge_split_metrics(
                    random_summary,
                    random_metrics,
                    ["model_name", "modality_set"],
                )
            split_test_path = Path("data") / "splits" / "publication" / target / "seed_0" / "test.csv"
            if "n_test" not in random_summary.columns:
                random_summary["n_test"] = float("nan")
            if split_test_path.exists():
                random_summary["n_test"] = random_summary["n_test"].fillna(
                    len(pd.read_csv(split_test_path))
                )
            random_summary["target"] = target
            random_summary["source"] = SOURCE_LABELS["neural/fusion"]
            rows.append(random_summary)

        classical_path = target_dir / "classical_baselines" / "classical_baseline_metrics.csv"
        if classical_path.exists():
            classical_metrics = _complete_metrics_with_predictions(
                pd.read_csv(classical_path),
                predictions_root / "final_publication_classical_random" / target,
            )
            classical = _metric_summary(
                classical_metrics,
                ["model_name", "modality_set"],
                split="test",
            )
            classical = _merge_split_metrics(
                classical,
                classical_metrics,
                ["model_name", "modality_set"],
            )
            classical["target"] = target
            classical["fusion"] = "classical"
            classical["source"] = SOURCE_LABELS["classical"]
            rows.append(classical)

        alignn_path = target_dir / "alignn_pretrained_rf" / "alignn_pretrained_rf_metrics.csv"
        if alignn_path.exists():
            alignn_metrics = _complete_metrics_with_predictions(
                pd.read_csv(alignn_path),
                predictions_root / "final_publication_alignn_pretrained" / target,
            )
            alignn = _metric_summary(
                alignn_metrics,
                ["model_name", "modality_set"],
                split="test",
            )
            alignn = _merge_split_metrics(
                alignn,
                alignn_metrics,
                ["model_name", "modality_set"],
            )
            alignn["target"] = target
            alignn["fusion"] = "pretrained_rf"
            alignn["source"] = SOURCE_LABELS["alignn_pretrained"]
            rows.append(alignn)

    if not rows:
        return pd.DataFrame()
    summary = pd.concat(rows, ignore_index=True, sort=False)
    summary = _append_order_and_labels(summary, order=RANDOM_MODEL_ORDER)
    summary["n_samples"] = summary.get("n_samples", summary.get("n_test"))
    summary["n_test"] = summary.get("n_test", summary.get("n_samples"))
    split_metric_columns = [
        f"{split}_{field}"
        for split in SPLITS
        for field in [
            "seeds",
            "n_samples",
            "MAE_mean",
            "MAE_std",
            "MSE_mean",
            "MSE_std",
            "RMSE_mean",
            "RMSE_std",
            "R2_mean",
            "R2_std",
        ]
    ]
    columns = [
        "target",
        "source",
        "model_name",
        "model_label",
        "modality_set",
        "fusion",
        "seeds",
        "n_train",
        "n_val",
        "n_test",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "R2_mean",
        "R2_std",
        *split_metric_columns,
        "sort_order",
    ]
    summary = _ensure_columns(summary, columns)
    return summary[columns].sort_values(["target", "sort_order"]).reset_index(drop=True)


def build_experiment_b_summary(results_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        path = results_root / target / "modality_dropout_mid_tri" / "modality_dropout_metrics.csv"
        if not path.exists():
            continue
        summary = _metric_summary(
            pd.read_csv(path),
            ["condition", "available_modalities"],
            split=None,
        )
        summary["target"] = target
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    columns = [
        "target",
        "condition",
        "available_modalities",
        "seeds",
        "n_samples",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "R2_mean",
        "R2_std",
    ]
    return pd.concat(rows, ignore_index=True, sort=False)[columns]


def build_experiment_c_summary(results_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        target_dir = results_root / target
        holdout_path = target_dir / "anion_holdout_halide" / "anion_holdout_metrics.csv"
        if holdout_path.exists():
            holdout = _metric_summary(
                pd.read_csv(holdout_path),
                ["heldout_family", "model_name", "modality_set"],
                split=None,
            )
            holdout["target"] = target
            holdout["source"] = "neural/fusion"
            rows.append(holdout)

        classical_path = (
            target_dir
            / "classical_anion_holdout_halide"
            / "classical_baseline_metrics.csv"
        )
        if classical_path.exists():
            classical = _metric_summary(
                pd.read_csv(classical_path),
                ["model_name", "modality_set"],
                split="test",
            )
            classical["target"] = target
            classical["heldout_family"] = "halide"
            classical["source"] = SOURCE_LABELS["classical"]
            rows.append(classical)

        alignn_path = (
            target_dir
            / "alignn_pretrained_anion_holdout_halide"
            / "alignn_pretrained_rf_metrics.csv"
        )
        if alignn_path.exists():
            alignn = _metric_summary(
                pd.read_csv(alignn_path),
                ["model_name", "modality_set"],
                split="test",
            )
            alignn["target"] = target
            alignn["heldout_family"] = "halide"
            alignn["source"] = SOURCE_LABELS["alignn_pretrained"]
            rows.append(alignn)

    if not rows:
        return pd.DataFrame()
    summary = pd.concat(rows, ignore_index=True, sort=False)
    summary = _append_order_and_labels(summary, order=HOLDOUT_MODEL_ORDER)
    columns = [
        "target",
        "heldout_family",
        "source",
        "model_name",
        "model_label",
        "modality_set",
        "seeds",
        "n_test",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "R2_mean",
        "R2_std",
        "sort_order",
    ]
    return summary[columns].sort_values(["target", "sort_order"]).reset_index(drop=True)


def build_experiment_d_summary(results_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        for source, path in [
            (
                "neural/fusion",
                results_root / target / "subgroup_analysis" / "subgroup_metrics.csv",
            ),
            (
                SOURCE_LABELS["classical"],
                results_root / target / "classical_subgroup_analysis" / "subgroup_metrics.csv",
            ),
            (
                SOURCE_LABELS["alignn_pretrained"],
                results_root / target / "alignn_pretrained_subgroup_analysis" / "subgroup_metrics.csv",
            ),
        ]:
            if not path.exists():
                continue
            metrics = pd.read_csv(path)
            summary = (
                metrics.groupby(
                    ["group_type", "group_name", "model_name", "modality_set"],
                    dropna=False,
                )
                .agg(
                    seeds=("seed", "nunique"),
                    n_samples_mean=("n_samples", "mean"),
                    n_samples_min=("n_samples", "min"),
                    n_samples_max=("n_samples", "max"),
                    unreliable_any=("unreliable", "max"),
                    MAE_mean=("MAE", "mean"),
                    MAE_std=("MAE", "std"),
                    RMSE_mean=("RMSE", "mean"),
                    RMSE_std=("RMSE", "std"),
                    R2_mean=("R2", "mean"),
                    R2_std=("R2", "std"),
                )
                .reset_index()
            )
            summary["target"] = target
            summary["source"] = source
            rows.append(summary)

    if not rows:
        return pd.DataFrame()
    summary = pd.concat(rows, ignore_index=True, sort=False)
    summary = _append_order_and_labels(summary, order=RANDOM_MODEL_ORDER)
    columns = [
        "target",
        "source",
        "group_type",
        "group_name",
        "model_name",
        "model_label",
        "modality_set",
        "seeds",
        "n_samples_mean",
        "n_samples_min",
        "n_samples_max",
        "unreliable_any",
        "MAE_mean",
        "MAE_std",
        "RMSE_mean",
        "RMSE_std",
        "R2_mean",
        "R2_std",
        "sort_order",
    ]
    return summary[columns].sort_values(
        ["target", "group_type", "group_name", "sort_order"]
    ).reset_index(drop=True)


def _is_public_manifest_path(path: Path) -> bool:
    text = str(path).lower()
    if path.suffix.lower() not in MANIFEST_ALLOWED_SUFFIXES:
        return False
    if path.parent == Path("results/final_publication") and not path.name.startswith("publication_"):
        return False
    return not any(pattern in text for pattern in MANIFEST_EXCLUDE_PATTERNS)


def _manifest_target(path: Path) -> str:
    parts = path.parts
    for target in TARGETS:
        if target in parts:
            return target
    return ""


def _manifest_group(path: Path) -> str:
    parts = list(path.parts)
    for marker in ["final_publication", "predictions", "explanation_validation", "final_publication"]:
        if marker in parts:
            idx = parts.index(marker)
            if idx + 2 < len(parts) and parts[idx + 1] in TARGETS:
                return parts[idx + 2]
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return path.parent.name


def _manifest_type(path: Path) -> str:
    text = str(path)
    if "/results/predictions/" in text:
        return "prediction_csv"
    if "/figures/" in text:
        return "figure"
    if "/results/explanation_validation/" in text:
        return "explanation_metric"
    if path.name.startswith("publication_"):
        return "summary_table"
    if path.suffix == ".json":
        return "config"
    return "metric_or_metadata"


def build_publication_manifest() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for root in MANIFEST_ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or not _is_public_manifest_path(path):
                continue
            rows.append(
                {
                    "path": str(path),
                    "artifact_type": _manifest_type(path),
                    "target": _manifest_target(path),
                    "experiment_group": _manifest_group(path),
                    "file_type": path.suffix.lstrip("."),
                    "bytes": path.stat().st_size,
                    "public_scope": "yes",
                    "role": "retained raw-target publication artifact",
                }
            )
    return pd.DataFrame(rows)


def build_all_tables(results_root: Path, output_dir: Path, overwrite: bool = False) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "publication_random_split_summary.csv": build_random_split_summary(results_root),
        "publication_experiment_b_modality_dropout_summary.csv": build_experiment_b_summary(results_root),
        "publication_experiment_c_halide_holdout_summary.csv": build_experiment_c_summary(results_root),
        "publication_experiment_d_subgroup_summary.csv": build_experiment_d_summary(results_root),
        "publication_manifest.csv": build_publication_manifest(),
    }
    written: dict[str, Path] = {}
    for filename, table in tables.items():
        path = output_dir / filename
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        table.to_csv(path, index=False)
        written[filename] = path
    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build raw-target final-publication summary tables."
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path("results/final_publication"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/final_publication"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_all_tables(
        results_root=args.results_root,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
