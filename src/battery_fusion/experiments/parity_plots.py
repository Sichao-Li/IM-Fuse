from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from battery_fusion.experiments.final_publication_tables import MODEL_LABELS, RANDOM_MODEL_ORDER, TARGETS


PREDICTION_EXPERIMENTS = (
    "final_publication_random",
    "final_publication_classical_random",
    "final_publication_alignn_pretrained",
)
DEFAULT_SPLITS = ("train", "test")
TARGET_UNITS = {
    "average_voltage": "V",
    "capacity_vol": "mAh cm$^{-3}$",
}
TARGET_LABELS = {
    "average_voltage": "Average voltage",
    "capacity_vol": "Capacity",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )


def standardize_like_plot_gth_pre(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match the legacy vis.plot_gth_pre convention: standardize y_true and y_pred separately."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    true_std = max(float(np.std(y_true, ddof=0)), eps)
    pred_std = max(float(np.std(y_pred, ddof=0)), eps)
    return (
        (y_true - float(np.mean(y_true))) / true_std,
        (y_pred - float(np.mean(y_pred))) / pred_std,
    )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_true - y_pred
    mae = float(np.mean(np.abs(error)))
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float("nan") if denominator == 0.0 else float(1.0 - np.sum(error**2) / denominator)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def discover_models(predictions_root: Path, targets: list[str], splits: list[str]) -> list[str]:
    models: set[str] = set()
    for experiment in PREDICTION_EXPERIMENTS:
        for target in targets:
            target_dir = predictions_root / experiment / target
            if not target_dir.exists():
                continue
            for model_dir in target_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                if any(model_dir.glob(f"*_{split}_predictions.csv") for split in splits):
                    models.add(model_dir.name)
    ordered = [model for model in RANDOM_MODEL_ORDER if model in models]
    ordered.extend(sorted(model for model in models if model not in ordered))
    return ordered


def _prediction_paths(predictions_root: Path, target: str, model_name: str, split: str, seeds: set[int] | None) -> list[Path]:
    paths: list[Path] = []
    for experiment in PREDICTION_EXPERIMENTS:
        model_dir = predictions_root / experiment / target / model_name
        if not model_dir.exists():
            continue
        for path in sorted(model_dir.glob(f"*_{split}_predictions.csv")):
            if seeds is not None:
                try:
                    seed = int(path.stem.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if seed not in seeds:
                    continue
            paths.append(path)
    return paths


def load_prediction_panel(
    predictions_root: Path,
    target: str,
    model_name: str,
    split: str,
    seeds: set[int] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in _prediction_paths(predictions_root, target, model_name, split, seeds):
        frame = pd.read_csv(path)
        if frame.empty or not {"y_true", "y_pred"}.issubset(frame.columns):
            continue
        if "seed" not in frame.columns:
            try:
                frame["seed"] = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                frame["seed"] = np.nan
        if "split" not in frame.columns:
            frame["split"] = split
        if "model_name" not in frame.columns:
            frame["model_name"] = model_name
        if "sample_id" not in frame.columns:
            frame["sample_id"] = np.arange(len(frame))
        frame["source_path"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = panel[panel["split"].astype(str) == split].copy()
    return panel


def seed_metric_summary(panel: pd.DataFrame) -> dict[str, float]:
    rows = []
    for seed, group in panel.groupby("seed", dropna=False):
        metrics = regression_metrics(group["y_true"].to_numpy(), group["y_pred"].to_numpy())
        metrics["seed"] = seed
        rows.append(metrics)
    seed_metrics = pd.DataFrame(rows)
    summary: dict[str, float] = {"seeds": float(seed_metrics["seed"].nunique()), "n_samples": float(len(panel))}
    for metric in ["MAE", "MSE", "RMSE", "R2"]:
        summary[f"{metric}_mean"] = float(seed_metrics[metric].mean())
        summary[f"{metric}_std"] = float(seed_metrics[metric].std(ddof=1)) if len(seed_metrics) > 1 else 0.0
    return summary


def build_seed_band(panel: pd.DataFrame, n_bins: int = 20) -> pd.DataFrame:
    if panel["seed"].nunique() < 2 or len(panel) < 10:
        return pd.DataFrame()
    working = panel[["seed", "y_true_std", "y_pred_std"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if working.empty:
        return pd.DataFrame()
    bin_count = min(n_bins, max(4, int(np.sqrt(len(working)))))
    try:
        working["bin"] = pd.qcut(working["y_true_std"], q=bin_count, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    per_seed = (
        working.groupby(["bin", "seed"], dropna=False)
        .agg(x_mean=("y_true_std", "mean"), y_mean=("y_pred_std", "mean"))
        .reset_index()
    )
    band = (
        per_seed.groupby("bin", dropna=False)
        .agg(
            x_mean=("x_mean", "mean"),
            y_mean=("y_mean", "mean"),
            y_std=("y_mean", "std"),
            seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values("x_mean")
    )
    return band[band["seeds"] >= 2].copy()


def _axis_limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.concatenate([x[np.isfinite(x)], y[np.isfinite(y)]])
    if finite.size == 0:
        return -1.0, 1.0
    lo = float(np.nanpercentile(finite, 0.5))
    hi = float(np.nanpercentile(finite, 99.5))
    padding = max(0.25, 0.08 * (hi - lo))
    return lo - padding, hi + padding


def plot_parity_panel(
    panel: pd.DataFrame,
    target: str,
    model_name: str,
    split: str,
    output_path: Path,
    overwrite: bool = False,
    max_points: int = 7000,
    seed: int = 0,
) -> dict[str, float]:
    if output_path.with_suffix(".pdf").exists() and not overwrite:
        raise FileExistsError(f"{output_path.with_suffix('.pdf')} exists; pass --overwrite")
    configure_style()
    working = panel.copy()
    working["y_true_std"], working["y_pred_std"] = standardize_like_plot_gth_pre(
        working["y_true"].to_numpy(),
        working["y_pred"].to_numpy(),
    )
    metrics = seed_metric_summary(working)

    if len(working) > max_points:
        scatter = working.sample(n=max_points, random_state=seed)
    else:
        scatter = working

    fig, ax = plt.subplots(figsize=(3.35, 3.25), constrained_layout=True)
    ax.scatter(
        scatter["y_true_std"],
        scatter["y_pred_std"],
        s=9,
        color="#5FA777",
        alpha=0.17 if split == "train" else 0.28,
        edgecolors="none",
        rasterized=True,
        label="Predictions",
    )

    band = build_seed_band(working)
    if not band.empty:
        lower = band["y_mean"] - band["y_std"].fillna(0.0)
        upper = band["y_mean"] + band["y_std"].fillna(0.0)
        ax.fill_between(
            band["x_mean"],
            lower,
            upper,
            color="#4C78A8",
            alpha=0.16,
            linewidth=0,
            label="Seed range",
        )
        ax.plot(band["x_mean"], band["y_mean"], color="#4C78A8", linewidth=1.3)

    x = working["y_true_std"].to_numpy(dtype=float)
    y = working["y_pred_std"].to_numpy(dtype=float)
    fit_mask = np.isfinite(x) & np.isfinite(y)
    if fit_mask.sum() >= 2:
        slope, intercept = np.polyfit(x[fit_mask], y[fit_mask], deg=1)
    else:
        slope, intercept = 1.0, 0.0
    lo, hi = _axis_limits(x, y)
    xx = np.linspace(lo, hi, 50)
    ax.plot(xx, slope * xx + intercept, color="#D18B47", linestyle="--", linewidth=1.3, label="Best fit")
    ax.plot([lo, hi], [lo, hi], color="#111111", linestyle="--", linewidth=1.0, alpha=0.72, label="Perfect fit")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")
    ax.grid(True, color="#D7DCE2", linewidth=0.45, alpha=0.65)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    model_label = MODEL_LABELS.get(model_name, model_name)
    ax.set_xlabel("Standardized true value")
    ax.set_ylabel("Standardized predicted value")
    unit = TARGET_UNITS.get(target, "")
    text = (
        f"MAE {metrics['MAE_mean']:.3g} +/- {metrics['MAE_std']:.2g} {unit}\n"
        f"R2 {metrics['R2_mean']:.3f} +/- {metrics['R2_std']:.2f}"
    )
    ax.text(
        0.03,
        0.97,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=6.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D7DCE2", "alpha": 0.85},
    )
    ax.legend(loc="lower right", frameon=False, handlelength=1.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)

    return {
        "target": target,
        "model_name": model_name,
        "model_label": model_label,
        "split": split,
        "n_rows": float(len(working)),
        "n_unique_samples": float(working["sample_id"].nunique()),
        "n_seeds": metrics["seeds"],
        "MAE_mean": metrics["MAE_mean"],
        "MAE_std": metrics["MAE_std"],
        "MSE_mean": metrics["MSE_mean"],
        "MSE_std": metrics["MSE_std"],
        "RMSE_mean": metrics["RMSE_mean"],
        "RMSE_std": metrics["RMSE_std"],
        "R2_mean": metrics["R2_mean"],
        "R2_std": metrics["R2_std"],
        "fit_slope": float(slope),
        "fit_intercept": float(intercept),
        "pdf_path": str(output_path.with_suffix(".pdf")),
        "png_path": str(output_path.with_suffix(".png")),
    }


def build_parity_plots(
    predictions_root: Path,
    output_dir: Path,
    summary_output: Path,
    targets: list[str],
    splits: list[str],
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    overwrite: bool = False,
    max_points: int = 7000,
) -> pd.DataFrame:
    predictions_root = Path(predictions_root)
    output_dir = Path(output_dir)
    summary_output = Path(summary_output)
    seed_set = set(seeds) if seeds else None
    model_names = models or discover_models(predictions_root, targets, splits)
    rows: list[dict[str, float]] = []
    for target in targets:
        for model_name in model_names:
            for split in splits:
                panel = load_prediction_panel(predictions_root, target, model_name, split, seed_set)
                if panel.empty:
                    continue
                output_path = output_dir / target / split / f"{model_name}_{split}_parity"
                rows.append(
                    plot_parity_panel(
                        panel=panel,
                        target=target,
                        model_name=model_name,
                        split=split,
                        output_path=output_path,
                        overwrite=overwrite,
                        max_points=max_points,
                    )
                )
    summary = pd.DataFrame(rows)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    if summary_output.exists() and not overwrite:
        raise FileExistsError(f"{summary_output} exists; pass --overwrite")
    summary.to_csv(summary_output, index=False)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final-publication train/test parity plots.")
    parser.add_argument("--predictions_root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/final_publication/parity_plots"))
    parser.add_argument(
        "--summary_output",
        type=Path,
        default=Path("results/final_publication/parity_plot_summary.csv"),
    )
    parser.add_argument("--targets", nargs="+", default=list(TARGETS))
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--max_points", type=int, default=7000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_parity_plots(
        predictions_root=args.predictions_root,
        output_dir=args.output_dir,
        summary_output=args.summary_output,
        targets=args.targets,
        splits=args.splits,
        models=args.models,
        seeds=args.seeds,
        overwrite=args.overwrite,
        max_points=args.max_points,
    )


if __name__ == "__main__":
    main()
