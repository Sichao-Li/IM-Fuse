from __future__ import annotations

import argparse
from pathlib import Path

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
