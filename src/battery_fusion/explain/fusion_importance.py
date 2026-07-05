from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from battery_fusion.fusion.feature_store import ProcessedFeatureStore
from battery_fusion.explain.composition_importance import (
    CleanFusionAdapter,
    DEFAULT_MODEL_NAME,
    load_feature_names,
    load_mid_tri_model,
    resolve_device,
    select_samples,
)
from battery_fusion.explain.context import FusionContextWrapper, FusionInputWrapper
from battery_fusion.explain.grs import GRSConfig, run_fis_explainer

LOGGER = logging.getLogger(__name__)
MODALITY_ALIASES = {
    "composition": "tab",
    "tabular": "tab",
    "tab": "tab",
    "rdf": "rdf",
    "structure": "graph",
    "graph": "graph",
}
DISPLAY_MODALITY = {
    "tab": "composition",
    "rdf": "rdf",
    "graph": "structure",
}


def normalize_perturb_modalities(values: list[str]) -> tuple[str, ...]:
    modalities = []
    for value in values:
        try:
            modality = MODALITY_ALIASES[value.lower()]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported modality {value!r}; choose from composition, rdf, structure."
            ) from exc
        if modality not in modalities:
            modalities.append(modality)
    return tuple(modalities)


def load_structure_symbols(sample_id: str, cif_dir: Path, n_atoms: int) -> list[str]:
    cif_path = Path(cif_dir) / f"{sample_id}.cif"
    if cif_path.exists():
        try:
            from pymatgen.core import Structure

            structure = Structure.from_file(cif_path)
            symbols = [site.specie.symbol for site in structure]
            if len(symbols) == n_atoms:
                return symbols
            LOGGER.warning(
                "CIF atom count for %s is %s but graph has %s atoms; using generic atom names.",
                sample_id,
                len(symbols),
                n_atoms,
            )
        except Exception as exc:
            LOGGER.warning("Could not read CIF symbols for %s from %s: %s", sample_id, cif_path, exc)
    return [f"atom_{idx}" for idx in range(n_atoms)]


def mapping_metadata(
    mapping: list[dict[str, Any]],
    composition_names: list[str],
    structure_symbols: list[str],
    rdf_step: float,
) -> pd.DataFrame:
    rows = []
    for fused_idx, item in enumerate(mapping):
        modality = item["modality"]
        if modality == "tab":
            local_idx = int(item["feat_idx"])
            name = composition_names[local_idx] if local_idx < len(composition_names) else f"composition_{local_idx}"
            rows.append(
                {
                    "fused_feature_idx": fused_idx,
                    "modality": DISPLAY_MODALITY[modality],
                    "local_feature_idx": local_idx,
                    "feature_name": name,
                    "atom_idx": np.nan,
                    "element": "",
                    "rdf_distance": np.nan,
                    "rdf_bin_indices": "",
                }
            )
        elif modality == "rdf":
            bin_indices = item.get("bin_indices", [item["bin_idx"]])
            local_idx = int(bin_indices[0])
            end_idx = int(bin_indices[-1])
            distance = local_idx * rdf_step
            end_distance = end_idx * rdf_step
            name = f"rdf_{distance:.3f}" if local_idx == end_idx else f"rdf_{distance:.3f}_{end_distance:.3f}"
            rows.append(
                {
                    "fused_feature_idx": fused_idx,
                    "modality": DISPLAY_MODALITY[modality],
                    "local_feature_idx": local_idx,
                    "feature_name": name,
                    "atom_idx": np.nan,
                    "element": "",
                    "rdf_distance": distance,
                    "rdf_bin_indices": ",".join(str(index) for index in bin_indices),
                }
            )
        elif modality == "graph":
            atom_idx = int(item["atom_idx"])
            element = structure_symbols[atom_idx] if atom_idx < len(structure_symbols) else f"atom_{atom_idx}"
            rows.append(
                {
                    "fused_feature_idx": fused_idx,
                    "modality": DISPLAY_MODALITY[modality],
                    "local_feature_idx": atom_idx,
                    "feature_name": f"{element}_{atom_idx}",
                    "atom_idx": atom_idx,
                    "element": element,
                    "rdf_distance": np.nan,
                    "rdf_bin_indices": "",
                }
            )
        else:
            raise ValueError(f"Unsupported modality in mapping: {modality}")
    return pd.DataFrame(rows)


def explain_one_sample(
    model: nn.Module,
    store: ProcessedFeatureStore,
    row: pd.Series,
    composition_names: list[str],
    target_col: str,
    seed: int,
    split: str,
    device: torch.device,
    config: GRSConfig,
    perturb_modalities: tuple[str, ...],
    cif_dir: Path,
    rdf_step: float,
    rdf_group_size: int,
) -> pd.DataFrame:
    sample_id = str(row["sample_id"])
    tab = store.load("tabular", sample_id)["features"].float().unsqueeze(0)
    rdf = store.load("rdf", sample_id)["features"].float().unsqueeze(0)
    graph = store.load("structure", sample_id)["features"]

    atom_fea = graph["atom_fea"].float()
    nbr_fea = graph["nbr_fea"].float()
    nbr_fea_idx = graph["nbr_fea_idx"].long()
    crystal_atom_idx = [torch.arange(atom_fea.shape[0], dtype=torch.long)]

    input_tensor, mapping = FusionInputWrapper(
        perturb_modalities,
        rdf_group_size=rdf_group_size,
    ).build_input_and_mapping(
        atom_fea=atom_fea,
        tab=tab,
        rdf=rdf,
    )
    wrapped = FusionContextWrapper(
        fusion_model=CleanFusionAdapter(model),
        device=device,
        perturb_modalities=perturb_modalities,
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
        y_pred = float(wrapped(input_tensor.to(device)).detach().cpu().reshape(-1)[0])

    explainer = run_fis_explainer(
        wrapped_model=wrapped,
        input_tensor=input_tensor.to(device),
        output=torch.tensor([y_true], device=device),
        config=config,
    )
    importance = [float(value) for value in explainer.ref_analysis["ref_main_effects"]]
    metadata = mapping_metadata(
        mapping=mapping,
        composition_names=composition_names,
        structure_symbols=load_structure_symbols(sample_id, cif_dir=cif_dir, n_atoms=atom_fea.shape[0]),
        rdf_step=rdf_step,
    )
    if len(metadata) != len(importance):
        raise RuntimeError(f"Metadata length {len(metadata)} does not match importance length {len(importance)}")

    output = metadata.copy()
    output["sample_id"] = sample_id
    output["target_col"] = target_col
    output["seed"] = seed
    output["split"] = split
    output["y_true"] = y_true
    output["y_pred"] = y_pred
    output["importance"] = importance
    return output[
        [
            "sample_id",
            "target_col",
            "seed",
            "split",
            "y_true",
            "y_pred",
            "fused_feature_idx",
            "modality",
            "local_feature_idx",
            "feature_name",
            "atom_idx",
            "element",
            "rdf_distance",
            "rdf_bin_indices",
            "importance",
        ]
    ]


def run_fusion_importance(
    target_col: str,
    seed: int,
    split: str,
    modalities: list[str],
    processed_root: Path,
    split_dir: Path,
    checkpoint_path: Path,
    labels_path: Path,
    cif_dir: Path,
    output_dir: Path,
    sample_id: str | None = None,
    sample_index: int = 0,
    max_samples: int = 1,
    device_name: str = "cpu",
    epsilon_rate: float = 0.05,
    n_order: int = 2,
    delta: float = 0.1,
    rdf_step: float = 0.1,
    rdf_group_size: int = 1,
) -> pd.DataFrame:
    perturb_modalities = normalize_perturb_modalities(modalities)
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
    composition_names = load_feature_names(labels_path, expected_dim=int(first_tab.shape[-1]))
    config = GRSConfig(epsilon_rate=epsilon_rate, loss_fn="mean_squared_error", n_order=n_order, delta=delta)

    frames = []
    for _, row in selected.iterrows():
        LOGGER.info("Explaining %s with modalities=%s", row["sample_id"], ",".join(perturb_modalities))
        frames.append(
            explain_one_sample(
                model=model,
                store=store,
                row=row,
                composition_names=composition_names,
                target_col=target_col,
                seed=seed,
                split=split,
                device=device,
                config=config,
                perturb_modalities=perturb_modalities,
                cif_dir=cif_dir,
                rdf_step=rdf_step,
                rdf_group_size=rdf_group_size,
            )
        )

    output = pd.concat(frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = sample_id if sample_id is not None else f"index_{sample_index}_n_{max_samples}"
    modality_tag = "_".join(DISPLAY_MODALITY[modality] for modality in perturb_modalities)
    output_path = output_dir / (
        f"fusion_importance_{modality_tag}_{target_col}_{DEFAULT_MODEL_NAME}_seed_{seed}_{split}_{suffix}.csv"
    )
    output.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s", output_path)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GRS importance for selected mid-tri fusion modalities.")
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--modalities", nargs="+", default=["composition"], choices=["composition", "tabular", "tab", "rdf", "structure", "graph"])
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/publication"))
    parser.add_argument("--split_dir", type=Path, default=Path("data/splits/publication"))
    parser.add_argument("--checkpoint_path", type=Path, default=None)
    parser.add_argument("--labels_path", type=Path, default=Path("data/sample_order/sample_order_keep_last.csv"))
    parser.add_argument("--cif_dir", type=Path, default=Path("data/raw/cifs"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/explanations/fusion_importance"))
    parser.add_argument("--sample_id", default=None)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--epsilon_rate", type=float, default=0.05)
    parser.add_argument("--n_order", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--rdf_step", type=float, default=0.1)
    parser.add_argument("--rdf_group_size", type=int, default=1)
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
    output = run_fusion_importance(
        target_col=args.target_col,
        seed=args.seed,
        split=args.split,
        modalities=args.modalities,
        processed_root=args.processed_root,
        split_dir=args.split_dir,
        checkpoint_path=checkpoint_path,
        labels_path=args.labels_path,
        cif_dir=args.cif_dir,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        sample_index=args.sample_index,
        max_samples=args.max_samples,
        device_name=args.device,
        epsilon_rate=args.epsilon_rate,
        n_order=args.n_order,
        delta=args.delta,
        rdf_step=args.rdf_step,
        rdf_group_size=args.rdf_group_size,
    )
    print(output.sort_values(["sample_id", "importance"], ascending=[True, False]).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
