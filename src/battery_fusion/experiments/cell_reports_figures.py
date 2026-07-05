from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd


TARGETS = ("average_voltage", "capacity_vol")
TARGET_LABELS = {
    "average_voltage": "Average voltage",
    "capacity_vol": "Capacity",
}
TARGET_UNITS = {
    "average_voltage": "V",
    "capacity_vol": "mAh cm$^{-3}$",
}

CONDITION_LABELS = {
    "full": "Full (C+G+R)",
    "drop_composition": "Drop composition (G+R)",
    "drop_graph": "Drop graph (C+R)",
    "drop_rdf": "Drop RDF (C+G)",
    "composition_only_fallback": "Composition only",
    "graph_only_fallback": "Graph only",
    "rdf_only_fallback": "RDF only",
}
CONDITION_COLORS = {
    "full": "#2F3437",
    "drop_composition": "#4C78A8",
    "drop_graph": "#F58518",
    "drop_rdf": "#54A24B",
    "composition_only_fallback": "#A7A7A7",
    "graph_only_fallback": "#7F7F7F",
    "rdf_only_fallback": "#C7C7C7",
}

MODEL_LABELS = {
    "unimodal_tabular": "Composition neural",
    "composition": "Composition neural",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "unimodal_structure": "CGCNN-style graph",
    "graph": "CGCNN-style graph",
    "late_dual_tabular_structure": "Composition + graph",
    "composition_graph": "Composition + graph",
    "early_tri_rdf_tabular_structure": "Early tri-fusion",
    "mid_tri_rdf_tabular_structure": "Full fusion",
    "full_fusion": "Full fusion",
    "alignn_pretrained_rf": "ALIGNN + RF",
}
SUBGROUP_MODELS = [
    "unimodal_tabular",
    "random_forest",
    "xgboost",
    "unimodal_structure",
    "mid_tri_rdf_tabular_structure",
    "alignn_pretrained_rf",
]
NONFUSION_BASELINES = {
    "unimodal_tabular",
    "random_forest",
    "xgboost",
    "unimodal_structure",
    "alignn_pretrained_rf",
}
FUSION_MODEL = "mid_tri_rdf_tabular_structure"
UNIMODAL_BRANCHES = [
    "unimodal_rdf_sequence",
    "unimodal_tabular",
    "unimodal_structure",
]
FUSION_BENEFIT_ORDER = [
    "unimodal_rdf_sequence",
    "unimodal_tabular",
    "unimodal_structure",
    "overall_unimodal_branches",
]
FUSION_BENEFIT_LABELS = {
    "unimodal_rdf_sequence": "vs RDF",
    "unimodal_tabular": "vs comp.",
    "unimodal_structure": "vs graph",
    "overall_unimodal_branches": "Overall",
}
FUSION_BASELINE_COMPARATORS = [
    "unimodal_rdf_sequence",
    "unimodal_tabular",
    "unimodal_structure",
    "random_forest",
    "xgboost",
    "alignn_pretrained_rf",
]
FUSION_BASELINE_ORDER = FUSION_BASELINE_COMPARATORS + ["overall_nonfusion"]
FUSION_BASELINE_LABELS = {
    "unimodal_rdf_sequence": "RDF",
    "unimodal_tabular": "Comp.\nNN",
    "unimodal_structure": "Graph\nNN",
    "random_forest": "RF",
    "xgboost": "XGBoost",
    "alignn_pretrained_rf": "ALIGNN\n+ RF",
    "overall_nonfusion": "Overall",
}
ANION_ORDER = ["oxide", "sulfide", "halide", "phosphate_or_polyanion", "other"]
ION_ORDER = ["Li", "Na", "Mg", "K", "Zn", "Ca", "Al", "other"]
GROUP_LABELS = {
    "oxide": "Oxide",
    "sulfide": "Sulfide",
    "halide": "Halide",
    "phosphate_or_polyanion": "Phosphate/\npolyanion",
    "other": "Other",
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
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
        }
    )


def _mean_std(frame: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    return (
        frame.groupby(group_cols, dropna=False)[value_col]
        .agg(["mean", "std"])
        .rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std"})
        .reset_index()
    )


def _false_mask(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return ~series
    return ~series.astype(str).str.lower().isin({"true", "1", "yes"})


def build_modality_dropout_plot_data(results_root: Path, modality_dropout_dir: str = "modality_dropout_mid_tri") -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for target in TARGETS:
        path = results_root / target / modality_dropout_dir / "modality_dropout_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        full = (
            frame[frame["condition"] == "full"][["seed", "MAE"]]
            .rename(columns={"MAE": "full_MAE"})
            .copy()
        )
        merged = frame.merge(full, on="seed", how="left")
        merged["delta_MAE_vs_full"] = merged["MAE"] - merged["full_MAE"]
        summary = _mean_std(
            merged,
            ["condition", "available_modalities"],
            "delta_MAE_vs_full",
        )
        mae_summary = _mean_std(merged, ["condition"], "MAE")
        summary = summary.merge(mae_summary, on="condition", how="left")
        summary["target"] = target
        summary["condition_label"] = summary["condition"].map(CONDITION_LABELS).fillna(summary["condition"])
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def _load_subgroup_metrics(results_root: Path, target: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, directory in [
        ("neural/fusion", "subgroup_analysis"),
        ("classical composition baseline", "classical_subgroup_analysis"),
        ("pretrained structure baseline", "alignn_pretrained_subgroup_analysis"),
    ]:
        path = results_root / target / directory / "subgroup_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["target"] = target
        frame["source"] = source
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _subgroup_models_for_fusion(fusion_model: str) -> list[str]:
    return [fusion_model if model == FUSION_MODEL else model for model in SUBGROUP_MODELS]


def build_subgroup_delta_plot_data(results_root: Path, fusion_model: str = FUSION_MODEL) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    subgroup_models = _subgroup_models_for_fusion(fusion_model)
    for target in TARGETS:
        metrics = _load_subgroup_metrics(results_root, target)
        if metrics.empty:
            continue
        reliable = metrics[
            metrics["model_name"].isin(subgroup_models)
            & _false_mask(metrics["unreliable"])
        ].copy()
        best = (
            reliable[reliable["model_name"].isin(NONFUSION_BASELINES)]
            .groupby(["target", "group_type", "group_name", "seed"], dropna=False)["MAE"]
            .min()
            .rename("best_nonfusion_MAE")
            .reset_index()
        )
        reliable = reliable.merge(
            best,
            on=["target", "group_type", "group_name", "seed"],
            how="left",
        )
        reliable["delta_MAE_vs_best_nonfusion"] = reliable["MAE"] - reliable["best_nonfusion_MAE"]
        reliable["delta_MAE_percent_vs_best_nonfusion"] = (
            100.0 * reliable["delta_MAE_vs_best_nonfusion"] / reliable["best_nonfusion_MAE"]
        )
        summary = (
            reliable.groupby(["target", "group_type", "group_name", "model_name", "modality_set"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                n_samples_mean=("n_samples", "mean"),
                MAE_mean=("MAE", "mean"),
                best_nonfusion_MAE_mean=("best_nonfusion_MAE", "mean"),
                delta_MAE_mean=("delta_MAE_vs_best_nonfusion", "mean"),
                delta_MAE_std=("delta_MAE_vs_best_nonfusion", "std"),
                delta_MAE_percent_mean=("delta_MAE_percent_vs_best_nonfusion", "mean"),
                delta_MAE_percent_std=("delta_MAE_percent_vs_best_nonfusion", "std"),
            )
            .reset_index()
        )
        summary["model_label"] = summary["model_name"].map(MODEL_LABELS).fillna(summary["model_name"])
        summary["model_order"] = summary["model_name"].map({name: idx for idx, name in enumerate(subgroup_models)})
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_fusion_subgroup_benefit_plot_data(results_root: Path, fusion_model: str = FUSION_MODEL) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    key_cols = ["target", "group_type", "group_name", "seed"]
    for target in TARGETS:
        metrics = _load_subgroup_metrics(results_root, target)
        if metrics.empty:
            continue
        model_set = set(UNIMODAL_BRANCHES + [fusion_model])
        reliable = metrics[
            metrics["model_name"].isin(model_set)
            & _false_mask(metrics["unreliable"])
        ].copy()
        fusion = reliable[reliable["model_name"] == fusion_model][key_cols + ["MAE", "n_samples"]].rename(
            columns={"MAE": "fusion_MAE"}
        )
        comparisons: list[pd.DataFrame] = []
        for branch in UNIMODAL_BRANCHES:
            branch_frame = reliable[reliable["model_name"] == branch][key_cols + ["MAE"]].rename(
                columns={"MAE": "branch_MAE"}
            )
            merged = fusion.merge(branch_frame, on=key_cols, how="inner")
            merged["comparator_model"] = branch
            comparisons.append(merged)
        if not comparisons:
            continue
        paired = pd.concat(comparisons, ignore_index=True, sort=False)
        paired["delta_MAE_vs_branch"] = paired["fusion_MAE"] - paired["branch_MAE"]
        paired["delta_MAE_percent_vs_branch"] = 100.0 * paired["delta_MAE_vs_branch"] / paired["branch_MAE"]
        paired["fusion_wins"] = paired["delta_MAE_vs_branch"] < 0

        summary = (
            paired.groupby(["target", "group_type", "group_name", "comparator_model"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                comparisons=("fusion_wins", "size"),
                n_samples_mean=("n_samples", "mean"),
                fusion_MAE_mean=("fusion_MAE", "mean"),
                branch_MAE_mean=("branch_MAE", "mean"),
                delta_MAE_mean=("delta_MAE_vs_branch", "mean"),
                delta_MAE_std=("delta_MAE_vs_branch", "std"),
                delta_MAE_percent_mean=("delta_MAE_percent_vs_branch", "mean"),
                delta_MAE_percent_std=("delta_MAE_percent_vs_branch", "std"),
                fusion_win_rate=("fusion_wins", "mean"),
            )
            .reset_index()
        )

        overall = (
            paired.groupby(["target", "group_type", "group_name"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                comparisons=("fusion_wins", "size"),
                n_samples_mean=("n_samples", "mean"),
                fusion_MAE_mean=("fusion_MAE", "mean"),
                branch_MAE_mean=("branch_MAE", "mean"),
                delta_MAE_mean=("delta_MAE_vs_branch", "mean"),
                delta_MAE_std=("delta_MAE_vs_branch", "std"),
                delta_MAE_percent_mean=("delta_MAE_percent_vs_branch", "mean"),
                delta_MAE_percent_std=("delta_MAE_percent_vs_branch", "std"),
                fusion_win_rate=("fusion_wins", "mean"),
            )
            .reset_index()
        )
        overall["comparator_model"] = "overall_unimodal_branches"
        summary = pd.concat([summary, overall], ignore_index=True, sort=False)
        summary["fusion_win_rate_percent"] = 100.0 * summary["fusion_win_rate"]
        summary["comparator_label"] = summary["comparator_model"].map(FUSION_BENEFIT_LABELS)
        summary["comparator_order"] = summary["comparator_model"].map(
            {name: idx for idx, name in enumerate(FUSION_BENEFIT_ORDER)}
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_fusion_baseline_benefit_plot_data(results_root: Path, fusion_model: str = FUSION_MODEL) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    key_cols = ["target", "group_type", "group_name", "seed"]
    for target in TARGETS:
        metrics = _load_subgroup_metrics(results_root, target)
        if metrics.empty:
            continue
        model_set = set(FUSION_BASELINE_COMPARATORS + [fusion_model])
        reliable = metrics[
            metrics["model_name"].isin(model_set)
            & _false_mask(metrics["unreliable"])
        ].copy()
        fusion = reliable[reliable["model_name"] == fusion_model][key_cols + ["MAE", "n_samples"]].rename(
            columns={"MAE": "fusion_MAE"}
        )
        comparisons: list[pd.DataFrame] = []
        for comparator in FUSION_BASELINE_COMPARATORS:
            comparator_frame = reliable[reliable["model_name"] == comparator][key_cols + ["MAE"]].rename(
                columns={"MAE": "baseline_MAE"}
            )
            merged = fusion.merge(comparator_frame, on=key_cols, how="inner")
            merged["comparator_model"] = comparator
            comparisons.append(merged)
        if not comparisons:
            continue
        paired = pd.concat(comparisons, ignore_index=True, sort=False)
        paired["delta_MAE_vs_baseline"] = paired["fusion_MAE"] - paired["baseline_MAE"]
        paired["delta_MAE_percent_vs_baseline"] = 100.0 * paired["delta_MAE_vs_baseline"] / paired["baseline_MAE"]
        paired["fusion_wins"] = paired["delta_MAE_vs_baseline"] < 0

        summary = (
            paired.groupby(["target", "group_type", "group_name", "comparator_model"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                comparisons=("fusion_wins", "size"),
                n_samples_mean=("n_samples", "mean"),
                fusion_MAE_mean=("fusion_MAE", "mean"),
                baseline_MAE_mean=("baseline_MAE", "mean"),
                delta_MAE_mean=("delta_MAE_vs_baseline", "mean"),
                delta_MAE_std=("delta_MAE_vs_baseline", "std"),
                delta_MAE_percent_mean=("delta_MAE_percent_vs_baseline", "mean"),
                delta_MAE_percent_std=("delta_MAE_percent_vs_baseline", "std"),
                fusion_win_rate=("fusion_wins", "mean"),
            )
            .reset_index()
        )

        overall = (
            paired.groupby(["target", "group_type", "group_name"], dropna=False)
            .agg(
                seeds=("seed", "nunique"),
                comparisons=("fusion_wins", "size"),
                n_samples_mean=("n_samples", "mean"),
                fusion_MAE_mean=("fusion_MAE", "mean"),
                baseline_MAE_mean=("baseline_MAE", "mean"),
                delta_MAE_mean=("delta_MAE_vs_baseline", "mean"),
                delta_MAE_std=("delta_MAE_vs_baseline", "std"),
                delta_MAE_percent_mean=("delta_MAE_percent_vs_baseline", "mean"),
                delta_MAE_percent_std=("delta_MAE_percent_vs_baseline", "std"),
                fusion_win_rate=("fusion_wins", "mean"),
            )
            .reset_index()
        )
        overall["comparator_model"] = "overall_nonfusion"
        summary = pd.concat([summary, overall], ignore_index=True, sort=False)
        summary["fusion_win_rate_percent"] = 100.0 * summary["fusion_win_rate"]
        summary["comparator_label"] = summary["comparator_model"].map(FUSION_BASELINE_LABELS)
        summary["comparator_order"] = summary["comparator_model"].map(
            {name: idx for idx, name in enumerate(FUSION_BASELINE_ORDER)}
        )
        rows.append(summary)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def save_figure(fig: plt.Figure, output_path: Path, overwrite: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = output_path.with_suffix(".pdf")
    png_path = output_path.with_suffix(".png")
    for path in [pdf_path, png_path]:
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=600, bbox_inches="tight")


def _clean_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color="#D7DCE2", linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def plot_figure_b(plot_data: pd.DataFrame, output_dir: Path, overwrite: bool) -> None:
    configure_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), constrained_layout=True)
    for ax, target in zip(axes, TARGETS):
        subset = plot_data[plot_data["target"] == target].copy()
        subset = subset.sort_values("delta_MAE_vs_full_mean", ascending=True)
        y = np.arange(len(subset))
        colors = [CONDITION_COLORS.get(condition, "#888888") for condition in subset["condition"]]
        ax.barh(
            y,
            subset["delta_MAE_vs_full_mean"],
            xerr=subset["delta_MAE_vs_full_std"].fillna(0.0),
            color=colors,
            edgecolor="white",
            linewidth=0.6,
            capsize=2.2,
            height=0.68,
        )
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_yticks(y, subset["condition_label"])
        ax.set_xlabel(f"MAE increase vs full ({TARGET_UNITS[target]})")
        _clean_axes(ax)
    # fig.text(0.004, 0.99, "B", fontsize=12, fontweight="bold", va="top")
    save_figure(fig, output_dir / "figure_b_modality_dropout_delta_mae", overwrite)
    plt.close(fig)


def _ordered_groups(group_type: str, values: list[str]) -> list[str]:
    order = ANION_ORDER if group_type == "anion_family" else ION_ORDER
    existing = [value for value in order if value in values]
    existing.extend(sorted(value for value in values if value not in existing))
    return existing


def _format_group_label(group: str) -> str:
    return GROUP_LABELS.get(group, group)


def _heatmap_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    target: str,
    group_type: str,
    title: str,
    norm: TwoSlopeNorm,
) -> mpl.cm.ScalarMappable:
    subset = frame[(frame["target"] == target) & (frame["group_type"] == group_type)].copy()
    groups = _ordered_groups(group_type, subset["group_name"].dropna().astype(str).unique().tolist())
    models = [model for model in SUBGROUP_MODELS if model in set(subset["model_name"])]
    pivot = subset.pivot_table(
        index="group_name",
        columns="model_name",
        values="delta_MAE_percent_mean",
        aggfunc="mean",
    ).reindex(index=groups, columns=models)
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap="BrBG_r", norm=norm)
    ax.set_xticks(np.arange(len(models)), [MODEL_LABELS.get(model, model) for model in models], rotation=38, ha="right")
    ax.set_yticks(np.arange(len(groups)), [_format_group_label(group) for group in groups])
    ax.set_title(title, loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value > 0.72 * norm.vmax or value < 0.72 * norm.vmin else "#263238"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=5.8, color=color)
    return image


def plot_figure_d(plot_data: pd.DataFrame, output_dir: Path, overwrite: bool) -> None:
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7), constrained_layout=True)
    finite = plot_data["delta_MAE_percent_mean"].replace([np.inf, -np.inf], np.nan).dropna()
    upper = max(30.0, float(np.ceil(np.nanpercentile(finite, 98) / 10.0) * 10.0))
    lower = min(-10.0, float(np.floor(np.nanmin(finite) / 5.0) * 5.0))
    norm = TwoSlopeNorm(vmin=lower, vcenter=0.0, vmax=upper)
    panel_specs = [
        ("average_voltage", "anion_family", "Average voltage: anion family"),
        ("capacity_vol", "anion_family", "Capacity: anion family"),
        ("average_voltage", "working_ion", "Average voltage: working ion"),
        ("capacity_vol", "working_ion", "Capacity: working ion"),
    ]
    images: list[mpl.cm.ScalarMappable] = []
    for ax, (target, group_type, title) in zip(axes.flat, panel_specs):
        image = _heatmap_panel(ax, plot_data, target, group_type, title, norm=norm)
        images.append(image)
    cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.72, pad=0.015, extend="both")
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6.5)
    cbar.set_label(r"$\Delta$MAE vs best non-fusion (%)", fontsize=7)
    # fig.text(0.004, 0.99, "D", fontsize=12, fontweight="bold", va="top")
    save_figure(fig, output_dir / "figure_d_subgroup_delta_mae_heatmap", overwrite)
    plt.close(fig)


def _fusion_benefit_heatmap_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    target: str,
    group_type: str,
    title: str,
    norm: TwoSlopeNorm,
    cmap: mpl.colors.Colormap,
    comparator_order: list[str],
    comparator_labels: dict[str, str],
    overall_key: str,
    show_title: bool = True,
) -> mpl.cm.ScalarMappable:
    subset = frame[(frame["target"] == target) & (frame["group_type"] == group_type)].copy()
    groups = _ordered_groups(group_type, subset["group_name"].dropna().astype(str).unique().tolist())
    columns = [name for name in comparator_order if name in set(subset["comparator_model"])]
    pivot = subset.pivot_table(
        index="group_name",
        columns="comparator_model",
        values="fusion_win_rate_percent",
        aggfunc="mean",
    ).reindex(index=groups, columns=columns)
    values = pivot.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(columns)), [comparator_labels.get(column, column) for column in columns])
    ax.set_yticks(np.arange(len(groups)), [_format_group_label(group) for group in groups])
    if show_title:
        ax.set_title(title, loc="left", fontweight="bold")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(groups), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    if overall_key in columns:
        overall_idx = columns.index(overall_key)
        ax.axvline(overall_idx - 0.5, color="white", linewidth=1.6)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            color = "white" if value < 28 or value > 78 else "#263238"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=6.0, color=color)
    return image


def plot_figure_d2(plot_data: pd.DataFrame, output_dir: Path, overwrite: bool) -> None:
    configure_style()
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "fusion_win_rate",
        ["#B85C5C", "#F6F0E3", "#1B8A7A"],
    )
    norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    panel_specs = [
        ("average_voltage", "anion_family", "Average voltage: anion family"),
        ("capacity_vol", "anion_family", "Capacity: anion family"),
        ("average_voltage", "working_ion", "Average voltage: working ion"),
        ("capacity_vol", "working_ion", "Capacity: working ion"),
    ]
    images: list[mpl.cm.ScalarMappable] = []
    for ax, (target, group_type, title) in zip(axes.flat, panel_specs):
        image = _fusion_benefit_heatmap_panel(
            ax,
            plot_data,
            target,
            group_type,
            title,
            norm=norm,
            cmap=cmap,
            comparator_order=FUSION_BENEFIT_ORDER,
            comparator_labels=FUSION_BENEFIT_LABELS,
            overall_key="overall_unimodal_branches",
        )
        images.append(image)
    cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.72, pad=0.015)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6.5)
    cbar.set_label("Fusion win rate vs unimodal branches (%)", fontsize=7)
    # fig.text(0.004, 0.99, "D2", fontsize=12, fontweight="bold", va="top")
    save_figure(fig, output_dir / "figure_d2_fusion_subgroup_win_rate", overwrite)
    plt.close(fig)


def plot_figure_d3(plot_data: pd.DataFrame, output_dir: Path, overwrite: bool) -> None:
    configure_style()
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "fusion_baseline_win_rate",
        ["#B85C5C", "#F6F0E3", "#1B8A7A"],
    )
    norm = TwoSlopeNorm(vmin=0.0, vcenter=50.0, vmax=100.0)
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.35), constrained_layout=True)
    panel_specs = [
        ("average_voltage", "anion_family", "Average voltage: anion family"),
        ("capacity_vol", "anion_family", "Capacity: anion family"),
        ("average_voltage", "working_ion", "Average voltage: working ion"),
        ("capacity_vol", "working_ion", "Capacity: working ion"),
    ]
    images: list[mpl.cm.ScalarMappable] = []
    for ax, (target, group_type, title) in zip(axes.flat, panel_specs):
        image = _fusion_benefit_heatmap_panel(
            ax,
            plot_data,
            target,
            group_type,
            title,
            norm=norm,
            cmap=cmap,
            comparator_order=FUSION_BASELINE_ORDER,
            comparator_labels=FUSION_BASELINE_LABELS,
            overall_key="overall_nonfusion",
            show_title=False,
        )
        images.append(image)
    cbar = fig.colorbar(images[0], ax=axes.ravel().tolist(), shrink=0.72, pad=0.015)
    cbar.ax.tick_params(length=2, width=0.5, labelsize=6.5)
    cbar.set_label("Fusion win rate vs comparator (%)", fontsize=7)
    # fig.text(0.004, 0.99, "D3", fontsize=12, fontweight="bold", va="top")
    save_figure(fig, output_dir / "figure_d3_fusion_baseline_win_rate", overwrite)
    plt.close(fig)


def build_and_plot(
    results_root: Path,
    output_dir: Path,
    data_output_dir: Path,
    overwrite: bool = False,
    modality_dropout_dir: str = "modality_dropout_mid_tri",
    fusion_model: str = FUSION_MODEL,
) -> dict[str, Path]:
    results_root = Path(results_root)
    output_dir = Path(output_dir)
    data_output_dir = Path(data_output_dir)
    data_output_dir.mkdir(parents=True, exist_ok=True)

    figure_b_data = build_modality_dropout_plot_data(results_root, modality_dropout_dir=modality_dropout_dir)
    figure_d_data = build_subgroup_delta_plot_data(results_root, fusion_model=fusion_model)
    figure_d2_data = build_fusion_subgroup_benefit_plot_data(results_root, fusion_model=fusion_model)
    figure_d3_data = build_fusion_baseline_benefit_plot_data(results_root, fusion_model=fusion_model)

    outputs = {
        "figure_b_data": data_output_dir / "figure_b_modality_dropout_delta_mae.csv",
        "figure_d_data": data_output_dir / "figure_d_subgroup_delta_mae_heatmap.csv",
        "figure_d2_data": data_output_dir / "figure_d2_fusion_subgroup_win_rate.csv",
        "figure_d3_data": data_output_dir / "figure_d3_fusion_baseline_win_rate.csv",
    }
    for key, frame in [
        ("figure_b_data", figure_b_data),
        ("figure_d_data", figure_d_data),
        ("figure_d2_data", figure_d2_data),
        ("figure_d3_data", figure_d3_data),
    ]:
        path = outputs[key]
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite")
        frame.to_csv(path, index=False)

    plot_figure_b(figure_b_data, output_dir, overwrite=overwrite)
    plot_figure_d(figure_d_data, output_dir, overwrite=overwrite)
    plot_figure_d2(figure_d2_data, output_dir, overwrite=overwrite)
    plot_figure_d3(figure_d3_data, output_dir, overwrite=overwrite)

    for name in [
        "figure_b_modality_dropout_delta_mae",
        "figure_d_subgroup_delta_mae_heatmap",
        "figure_d2_fusion_subgroup_win_rate",
        "figure_d3_fusion_baseline_win_rate",
    ]:
        outputs[f"{name}_pdf"] = output_dir / f"{name}.pdf"
        outputs[f"{name}_png"] = output_dir / f"{name}.png"
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render retained modality-dropout and subgroup figures.")
    parser.add_argument("--results_root", type=Path, default=Path("results/final_publication"))
    parser.add_argument("--output_dir", type=Path, default=Path("figures/final_publication/cell_reports"))
    parser.add_argument("--data_output_dir", type=Path, default=Path("results/final_publication/cell_reports_figure_data"))
    parser.add_argument("--modality_dropout_dir", default="modality_dropout_mid_tri")
    parser.add_argument("--fusion_model", default=FUSION_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_and_plot(
        results_root=args.results_root,
        output_dir=args.output_dir,
        data_output_dir=args.data_output_dir,
        overwrite=args.overwrite,
        modality_dropout_dir=args.modality_dropout_dir,
        fusion_model=args.fusion_model,
    )


if __name__ == "__main__":
    main()
