from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PERMUTATION_DELETION_FRACTIONS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
PERMUTATION_DELETION_ORDERS = ("top", "random", "bottom")
PERMUTATION_DELETION_COLORS = {
    "top": "#8da0cb",
    "random": "#fc8d62",
    "bottom": "#66c2a5",
}


def prepare_top_rows(
    frame: pd.DataFrame,
    label_col: str,
    value_col: str,
    top_n: int,
) -> pd.DataFrame:
    out = (
        frame.sort_values(value_col, ascending=False)
        .head(top_n)
        .rename(columns={label_col: "label", value_col: "value"})
    )
    return out[["label", "value"]].reset_index(drop=True)


def modality_overall_frame(
    composition: pd.DataFrame,
    rdf: pd.DataFrame,
    structure: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "modality": ["composition", "rdf", "structure"],
            "importance": [
                float(composition["delta_mae_mean"].max()),
                float(rdf["delta_mae_mean"].max()),
                float(structure["delta_mae_mean"].max()),
            ],
        }
    )


def select_top_permutation_features(
    frame: pd.DataFrame,
    top_n: int = 20,
    include_full_block: bool = False,
) -> pd.DataFrame:
    """Select top cross-modal permutation features for an error-bar plot."""
    selected = frame.copy()
    if not include_full_block:
        selected = selected[selected["source_modality"] != "full_3modal"]
    selected = selected.sort_values("delta_mae_seed_mean", ascending=False).head(top_n)
    return selected.reset_index(drop=True)


def select_modality_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    """Pick one representative row for each modality for compact comparison."""
    rows = []
    order = ["composition", "rdf", "structure", "full_3modal"]
    for modality in order:
        group = frame[frame["source_modality"] == modality]
        if group.empty:
            continue
        if modality == "full_3modal":
            block = group[group["feature_group"] == "all_modalities_block"]
            chosen = block.iloc[0] if not block.empty else group.sort_values("delta_mae_seed_mean", ascending=False).iloc[0]
        elif modality == "structure":
            whole = group[group["feature_group"] == "whole_structure"]
            chosen = whole.iloc[0] if not whole.empty else group.sort_values("delta_mae_seed_mean", ascending=False).iloc[0]
        else:
            chosen = group.sort_values("delta_mae_seed_mean", ascending=False).iloc[0]
        rows.append(chosen)
    return pd.DataFrame(rows).reset_index(drop=True)


def _short_permutation_label(row: pd.Series) -> str:
    modality = str(row["source_modality"])
    feature = str(row["feature_group"])
    if modality == "full_3modal":
        return "all modalities"
    if modality == "structure" and feature == "whole_structure":
        return "structure"
    return f"{modality}: {feature}"


def _plot_errorbar_horizontal(
    frame: pd.DataFrame,
    output_path: Path,
    title: str,
    xlabel: str = "Permutation delta MAE",
) -> None:
    import matplotlib.pyplot as plt

    plot_frame = frame.copy().iloc[::-1].reset_index(drop=True)
    plot_frame["plot_label"] = plot_frame.apply(_short_permutation_label, axis=1)
    fig_height = max(3.2, 0.34 * len(plot_frame) + 1.0)
    fig, ax = plt.subplots(figsize=(6.2, fig_height))
    colors = {
        "composition": "#4C78A8",
        "rdf": "#F58518",
        "structure": "#54A24B",
        "full_3modal": "#B279A2",
    }
    bar_colors = [colors.get(value, "#6B7280") for value in plot_frame["source_modality"]]
    xerr = plot_frame["delta_mae_seed_std"].fillna(0.0)
    ax.barh(
        plot_frame["plot_label"],
        plot_frame["delta_mae_seed_mean"],
        xerr=xerr,
        color=bar_colors,
        ecolor="0.25",
        capsize=2.5,
        linewidth=0,
    )
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_permutation_matrix_errorbars(
    input_root: Path,
    output_dir: Path,
    targets: list[str] | tuple[str, ...] = ("capacity_vol", "average_voltage"),
    top_n: int = 20,
) -> dict[str, Path]:
    """Plot cross-seed permutation importance with error bars for each target."""
    outputs = {}
    for target in targets:
        summary_path = input_root / target / "full_3modal_permutation_summary_by_feature.csv"
        frame = pd.read_csv(summary_path)
        target_label = target.replace("_", " ")
        top_features = select_top_permutation_features(frame, top_n=top_n)
        representatives = select_modality_representatives(frame)
        top_path = output_dir / f"{target}_permutation_top_features_errorbar.pdf"
        modality_path = output_dir / f"{target}_permutation_modality_representatives_errorbar.pdf"
        _plot_errorbar_horizontal(
            top_features,
            top_path,
            title=f"{target_label}: top permutation features",
        )
        _plot_errorbar_horizontal(
            representatives,
            modality_path,
            title=f"{target_label}: modality representatives",
        )
        outputs[f"{target}_top_features"] = top_path
        outputs[f"{target}_modality_representatives"] = modality_path
    return outputs


def _ablation_count(n_items: int, fraction: float) -> int:
    if n_items == 0 or fraction == 0:
        return 0
    return max(1, min(n_items, int(np.ceil(n_items * fraction))))


def _rank_importance_values(values: pd.Series, order: str, seed: int) -> pd.Series:
    if order == "top":
        return values.sort_values(ascending=False)
    if order == "bottom":
        return values.sort_values(ascending=True)
    if order == "random":
        shuffled = values.sample(frac=1.0, random_state=seed)
        return shuffled
    raise ValueError(f"Unsupported order {order!r}")


def _overall_modality_rows(seed_frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality in ["composition", "rdf", "structure"]:
        group = seed_frame[seed_frame["source_modality"] == modality]
        if group.empty:
            continue
        if modality == "structure":
            whole = group[group["feature_group"] == "whole_structure"]
            chosen = whole.iloc[0] if not whole.empty else group.sort_values("delta_mae_mean", ascending=False).iloc[0]
        else:
            chosen = group.sort_values("delta_mae_mean", ascending=False).iloc[0]
        rows.append(chosen)
    return pd.DataFrame(rows)


def build_permutation_deletion_curve(
    frame: pd.DataFrame,
    modality: str,
    fractions: tuple[float, ...] = PERMUTATION_DELETION_FRACTIONS,
    orders: tuple[str, ...] = PERMUTATION_DELETION_ORDERS,
) -> pd.DataFrame:
    """Build cumulative permutation-importance deletion curves with seed error bars."""
    seed_rows = []
    for seed, seed_frame in frame.groupby("seed", sort=True):
        if modality == "overall":
            modality_frame = _overall_modality_rows(seed_frame)
        else:
            modality_frame = seed_frame[seed_frame["source_modality"] == modality]
        values = modality_frame["delta_mae_mean"].fillna(0.0).reset_index(drop=True)
        target_col = str(seed_frame["target_col"].iloc[0])
        for order in orders:
            ranked = _rank_importance_values(values, order=order, seed=int(seed))
            for fraction in fractions:
                count = _ablation_count(len(ranked), fraction)
                seed_rows.append(
                    {
                        "target_col": target_col,
                        "modality": modality,
                        "seed": int(seed),
                        "order": order,
                        "ablation_fraction": float(fraction),
                        "n_deleted": int(count),
                        "importance_sum": float(ranked.head(count).sum()) if count else 0.0,
                    }
                )
    per_seed = pd.DataFrame(seed_rows)
    return (
        per_seed.groupby(["target_col", "modality", "order", "ablation_fraction"], sort=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_deleted_mean=("n_deleted", "mean"),
            importance_mean=("importance_sum", "mean"),
            importance_std=("importance_sum", "std"),
        )
        .reset_index()
    )


def _plot_permutation_deletion_curve(curve: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for order, group in curve.groupby("order", sort=False):
        group = group.sort_values("ablation_fraction")
        x = group["ablation_fraction"].to_numpy()
        y = group["importance_mean"].to_numpy()
        std = group["importance_std"].fillna(0.0).to_numpy()
        color = PERMUTATION_DELETION_COLORS.get(str(order))
        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.0,
            color=color,
            label=order,
        )
        ax.fill_between(x, y - std, y + std, alpha=0.18, color=color)
    ax.axhline(0, linestyle="--", linewidth=1.0, color="black")
    ax.set_xlabel("Deleted feature fraction")
    ax.set_ylabel("Cumulative permutation delta MAE")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_permutation_deletion_curves(
    input_root: Path,
    output_dir: Path,
    results_dir: Path,
    targets: list[str] | tuple[str, ...] = ("capacity_vol", "average_voltage"),
) -> dict[str, Path]:
    """Plot permutation-derived deletion curves with seed error bars."""
    outputs = {}
    for target in targets:
        frame = pd.read_csv(input_root / target / "full_3modal_permutation_summary_all_seeds.csv")
        for modality in ["composition", "rdf", "structure", "overall"]:
            curve = build_permutation_deletion_curve(frame, modality=modality)
            target_results = results_dir / target
            target_results.mkdir(parents=True, exist_ok=True)
            curve.to_csv(target_results / f"{target}_{modality}_permutation_deletion_curve.csv", index=False)
            output_path = output_dir / f"{target}_{modality}_permutation_deletion_curve_errorbar.pdf"
            _plot_permutation_deletion_curve(curve, output_path)
            outputs[f"{target}_{modality}"] = output_path
    return outputs


def _plot_horizontal_bar(
    frame: pd.DataFrame,
    label_col: str,
    value_col: str,
    output_path: Path,
    xlabel: str,
    title: str | None = None,
) -> None:
    import matplotlib.pyplot as plt

    plot_frame = frame.iloc[::-1].reset_index(drop=True)
    fig_height = max(3.0, 0.32 * len(plot_frame) + 0.9)
    fig, ax = plt.subplots(figsize=(5.4, fig_height))
    ax.barh(plot_frame[label_col], plot_frame[value_col], color="#4C78A8")
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_interpretability_summary(
    target_col: str,
    input_root: Path,
    output_dir: Path,
    top_n: int = 15,
) -> dict[str, Path]:
    composition = pd.read_csv(input_root / "composition_permutation" / "permutation_importance_summary.csv")
    rdf = pd.read_csv(input_root / "rdf_permutation" / "permutation_importance_summary.csv")
    structure = pd.read_csv(input_root / "structure_permutation" / "permutation_importance_summary.csv")
    atom_elements = pd.read_csv(input_root / "structure_atom_ablation" / "structure_atom_ablation_element_summary.csv")

    outputs = {
        "rdf": output_dir / f"{target_col}_rdf_permutation_importance.pdf",
        "structure_atoms": output_dir / f"{target_col}_structure_atom_ablation_elements.pdf",
        "overall": output_dir / f"{target_col}_modality_importance_overall.pdf",
    }

    rdf_top = prepare_top_rows(rdf, label_col="feature_group", value_col="delta_mae_mean", top_n=top_n)
    _plot_horizontal_bar(
        rdf_top,
        label_col="label",
        value_col="value",
        output_path=outputs["rdf"],
        xlabel="Permutation delta MAE",
    )

    element_top = prepare_top_rows(
        atom_elements,
        label_col="element",
        value_col="prediction_delta_mean",
        top_n=top_n,
    )
    _plot_horizontal_bar(
        element_top,
        label_col="label",
        value_col="value",
        output_path=outputs["structure_atoms"],
        xlabel="Atom-ablation prediction change",
    )

    overall = modality_overall_frame(composition, rdf, structure)
    _plot_horizontal_bar(
        overall,
        label_col="modality",
        value_col="importance",
        output_path=outputs["overall"],
        xlabel="Representative delta MAE",
    )
    return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot retained interpretability summary figures.")
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--input_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_n", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = plot_interpretability_summary(
        target_col=args.target_col,
        input_root=args.input_root,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )
    for path in outputs.values():
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
