from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn

from battery_fusion.fusion.feature_store import ProcessedFeatureStore
from battery_fusion.explain.context import FusionContextWrapper, FusionInputWrapper
from battery_fusion.explain.grs import GRSConfig, run_fis_explainer
from battery_fusion.features.tabular import vocabulary_from_formulas
from battery_fusion.fusion.cgcnn_multimodal import build_multimodal_mid_fusion

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "mid_tri_rdf_tabular_structure"


class CleanFusionAdapter(nn.Module):
    """Adapt clean publication fusion inputs to the legacy explanation wrapper."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        return self.model(
            {
                "tabular": batch["tab_inputs"],
                "rdf": batch["rdf_inputs"],
                "structure": batch["graph_inputs"],
            }
        ).reshape(-1, 1)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested --device mps but torch.backends.mps.is_available() is False")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda but torch.cuda.is_available() is False")
    return torch.device(device)


def load_feature_names(labels_path: Path, expected_dim: int) -> list[str]:
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        formula_cols = [col for col in labels.columns if "formula" in col.lower()]
        if formula_cols:
            names = vocabulary_from_formulas(labels[formula_cols[0]])
            if len(names) == expected_dim:
                return names
            LOGGER.warning(
                "Formula vocabulary length %s does not match tabular dimension %s; using generic names.",
                len(names),
                expected_dim,
            )
    return [f"composition_{idx}" for idx in range(expected_dim)]


def load_mid_tri_model(
    checkpoint_path: Path,
    tabular_dim: int,
    rdf_dim: int,
    atom_fea_dim: int,
    nbr_fea_dim: int,
    device: torch.device,
) -> nn.Module:
    model = build_multimodal_mid_fusion(
        tabular_dim=tabular_dim,
        rdf_dim=rdf_dim,
        atom_fea_dim=atom_fea_dim,
        nbr_fea_dim=nbr_fea_dim,
        dropout=0.0,
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def select_samples(
    split_frame: pd.DataFrame,
    sample_id: str | None,
    sample_index: int,
    max_samples: int,
) -> pd.DataFrame:
    if sample_id is not None:
        selected = split_frame[split_frame["sample_id"].astype(str) == str(sample_id)]
        if selected.empty:
            raise ValueError(f"Sample {sample_id!r} not found in split frame")
        return selected.head(max_samples).reset_index(drop=True)
    if sample_index < 0 or sample_index >= len(split_frame):
        raise IndexError(f"sample_index {sample_index} outside split of length {len(split_frame)}")
    return split_frame.iloc[sample_index : sample_index + max_samples].reset_index(drop=True)


def explain_one_sample(
    model: nn.Module,
    store: ProcessedFeatureStore,
    row: pd.Series,
    feature_names: list[str],
    target_col: str,
    seed: int,
    split: str,
    device: torch.device,
    config: GRSConfig,
) -> pd.DataFrame:
    sample_id = str(row["sample_id"])
    tab = store.load("tabular", sample_id)["features"].float().unsqueeze(0)
    rdf = store.load("rdf", sample_id)["features"].float().unsqueeze(0)
    graph = store.load("structure", sample_id)["features"]

    atom_fea = graph["atom_fea"].float()
    nbr_fea = graph["nbr_fea"].float()
    nbr_fea_idx = graph["nbr_fea_idx"].long()
    crystal_atom_idx = [torch.arange(atom_fea.shape[0], dtype=torch.long)]

    input_tensor, mapping = FusionInputWrapper(("tab",)).build_input_and_mapping(
        atom_fea=atom_fea,
        tab=tab,
        rdf=rdf,
    )
    wrapped = FusionContextWrapper(
        fusion_model=CleanFusionAdapter(model),
        device=device,
        perturb_modalities=("tab",),
        mapping=mapping,
    ).to(device)
    wrapped.set_context(
        atom_fea=atom_fea,
        nbr_fea=nbr_fea,
        nbr_fea_idx=nbr_fea_idx,
        crystal_atom_idx=crystal_atom_idx,
        tab_inputs=tab,
        rdf_inputs=rdf,
    )

    y_true = float(row["target"])
    with torch.no_grad():
        y_pred = float(
            wrapped(
                input_tensor.to(device),
            )
            .detach()
            .cpu()
            .reshape(-1)[0]
        )

    explainer = run_fis_explainer(
        wrapped_model=wrapped,
        input_tensor=input_tensor.to(device),
        output=torch.tensor([y_true], device=device),
        config=config,
    )
    importance = [float(value) for value in explainer.ref_analysis["ref_main_effects"]]
    if len(feature_names) != len(importance):
        feature_names = [f"composition_{idx}" for idx in range(len(importance))]

    return pd.DataFrame(
        {
            "sample_id": sample_id,
            "target_col": target_col,
            "seed": seed,
            "split": split,
            "y_true": y_true,
            "y_pred": y_pred,
            "feature_idx": list(range(len(importance))),
            "feature_name": feature_names,
            "importance": importance,
        }
    )


def run_composition_importance(
    target_col: str,
    seed: int,
    split: str,
    processed_root: Path,
    split_dir: Path,
    checkpoint_path: Path,
    labels_path: Path,
    output_dir: Path,
    sample_id: str | None = None,
    sample_index: int = 0,
    max_samples: int = 1,
    device_name: str = "cpu",
    epsilon_rate: float = 0.05,
    n_order: int = 2,
    delta: float = 0.1,
) -> pd.DataFrame:
    device = resolve_device(device_name)
    split_frame = pd.read_csv(Path(split_dir) / target_col / f"seed_{seed}" / f"{split}.csv")
    selected = select_samples(split_frame, sample_id=sample_id, sample_index=sample_index, max_samples=max_samples)
    store = ProcessedFeatureStore(processed_root, ("tabular", "rdf", "structure"))

    first_id = str(selected.iloc[0]["sample_id"])
    first_tab = store.load("tabular", first_id)["features"].float()
    first_rdf = store.load("rdf", first_id)["features"].float()
    first_graph = store.load("structure", first_id)["features"]
    model = load_mid_tri_model(
        checkpoint_path=checkpoint_path,
        tabular_dim=int(first_tab.shape[-1]),
        rdf_dim=int(first_rdf.shape[-1]),
        atom_fea_dim=int(first_graph["atom_fea"].shape[-1]),
        nbr_fea_dim=int(first_graph["nbr_fea"].shape[-1]),
        device=device,
    )
    feature_names = load_feature_names(labels_path, expected_dim=int(first_tab.shape[-1]))
    config = GRSConfig(epsilon_rate=epsilon_rate, loss_fn="mean_squared_error", n_order=n_order, delta=delta)

    frames = []
    for _, row in selected.iterrows():
        LOGGER.info("Explaining %s", row["sample_id"])
        frames.append(
            explain_one_sample(
                model=model,
                store=store,
                row=row,
                feature_names=feature_names,
                target_col=target_col,
                seed=seed,
                split=split,
                device=device,
                config=config,
            )
        )

    output = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = sample_id if sample_id is not None else f"index_{sample_index}_n_{max_samples}"
    output_path = output_dir / f"composition_importance_{target_col}_{DEFAULT_MODEL_NAME}_seed_{seed}_{split}_{suffix}.csv"
    output.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s", output_path)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run composition-feature GRS importance for the mid-tri fusion model.")
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/legacy_rdf_split_seed_42"))
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits/publication"))
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--labels_path", type=Path, default=Path("data/labels/labels_keep_last.csv"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/explanations/composition_importance"))
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--epsilon_rate", type=float, default=0.05)
    parser.add_argument("--n_order", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = (
            Path("results/final_publication")
            / args.target_col
            / "random_split"
            / "checkpoints"
            / DEFAULT_MODEL_NAME
            / f"seed_{args.seed}"
            / "model.pt"
        )
    output = run_composition_importance(
        target_col=args.target_col,
        seed=args.seed,
        split=args.split,
        processed_root=args.processed_root,
        split_dir=args.split_dir,
        checkpoint_path=checkpoint_path,
        labels_path=args.labels_path,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        sample_index=args.sample_index,
        max_samples=args.max_samples,
        device_name=args.device,
        epsilon_rate=args.epsilon_rate,
        n_order=args.n_order,
        delta=args.delta,
    )
    print(output.sort_values(["sample_id", "importance"], ascending=[True, False]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
