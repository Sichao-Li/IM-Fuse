from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from battery_fusion.experiments.publication import (
    PublicationTriDataset,
    TRI_TORCH_SPECS,
    build_torch_model,
    collate_tri,
    infer_tri_dims,
    resolve_device,
)
from battery_fusion.fusion.feature_store import ProcessedFeatureStore
from battery_fusion.fusion.cgcnn_multimodal import (
    MultimodalEarlyFusionRegressor,
    MultimodalMidFusionRegressor,
)
from battery_fusion.training.metrics import regression_metrics
from battery_fusion.training.target_transform import TargetTransform
from battery_fusion.utils.chemistry_groups import load_assignments

LOGGER = logging.getLogger(__name__)
CANONICAL_MODALITIES = ("tabular", "structure", "rdf")


@dataclass(frozen=True)
class DropoutCondition:
    name: str
    available_modalities: tuple[str, ...]


def modality_dropout_conditions() -> list[DropoutCondition]:
    return [
        DropoutCondition("full", ("tabular", "structure", "rdf")),
        DropoutCondition("drop_composition", ("structure", "rdf")),
        DropoutCondition("drop_graph", ("tabular", "rdf")),
        DropoutCondition("drop_rdf", ("tabular", "structure")),
        DropoutCondition("composition_only_fallback", ("tabular",)),
        DropoutCondition("graph_only_fallback", ("structure",)),
        DropoutCondition("rdf_only_fallback", ("rdf",)),
    ]


def _resolve_checkpoint(checkpoint_dir: Path, seed: int) -> Path:
    candidates = [
        Path(checkpoint_dir) / f"seed_{seed}" / "model.pt",
        Path(checkpoint_dir) / f"seed_{seed}.pt",
        Path(checkpoint_dir) / "model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No checkpoint found for seed {seed} under {checkpoint_dir}")


def _load_run_config(checkpoint_path: Path) -> dict[str, Any]:
    config_path = checkpoint_path.parent / "run_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing publication run config next to {checkpoint_path}")
    return json.loads(config_path.read_text())


def _move_graph(
    graph: tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx = graph
    return (
        atom_fea.to(device),
        nbr_fea.to(device),
        nbr_fea_idx.to(device),
        [idx.to(device) for idx in crystal_atom_idx],
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    return (
        {
            "tabular": batch["tabular"].to(device),
            "rdf": batch["rdf"].to(device),
            "structure": _move_graph(batch["structure"], device),
        },
        batch["target"].to(device),
    )


def masked_modality_forward(
    model: torch.nn.Module,
    inputs: dict[str, Any],
    available_modalities: tuple[str, ...],
) -> torch.Tensor:
    available = set(available_modalities)
    if isinstance(model, MultimodalMidFusionRegressor):
        z_tab = model.tab_proj(model.tab_encoder(inputs["tabular"]))
        z_rdf = model.rdf_proj(model.rdf_encoder(inputs["rdf"]))
        z_graph = model.graph_proj(model.graph_encoder(inputs["structure"]))
        if "tabular" not in available:
            z_tab = torch.zeros_like(z_tab)
        if "rdf" not in available:
            z_rdf = torch.zeros_like(z_rdf)
        if "structure" not in available:
            z_graph = torch.zeros_like(z_graph)
        tokens = torch.stack([z_tab, z_rdf, z_graph], dim=1)
        fused = model.transformer(tokens).mean(dim=1)
        return model.fusion_head(fused).squeeze(-1)
    if isinstance(model, MultimodalEarlyFusionRegressor):
        z_tab = model.tab_proj(model.tab_encoder(inputs["tabular"]))
        z_rdf = model.rdf_proj(model.rdf_encoder(inputs["rdf"]))
        z_graph = model.graph_proj(model.graph_encoder(inputs["structure"]))
        if "tabular" not in available:
            z_tab = torch.zeros_like(z_tab)
        if "rdf" not in available:
            z_rdf = torch.zeros_like(z_rdf)
        if "structure" not in available:
            z_graph = torch.zeros_like(z_graph)
        return model.fusion_head(torch.cat([z_tab, z_rdf, z_graph], dim=-1)).squeeze(-1)
    raise TypeError(f"Unsupported fusion model: {type(model)!r}")


def _metadata_columns(split_frame: pd.DataFrame, metadata_path: Path | None) -> pd.DataFrame:
    output = split_frame.copy()
    if metadata_path and metadata_path.exists():
        metadata = load_assignments(metadata_path)
        output = output.drop(columns=["formula", "working_ion", "anion_family"], errors="ignore")
        output = output.merge(metadata[["sample_id", "formula", "working_ion", "anion_family"]], on="sample_id", how="left")
    for column in ["formula", "working_ion", "anion_family"]:
        if column not in output.columns:
            output[column] = ""
    return output[["sample_id", "formula", "working_ion", "anion_family"]].drop_duplicates("sample_id")


def run_modality_dropout(
    processed_root: Path,
    checkpoint_dir: Path,
    split_dir: Path,
    output_dir: Path,
    seeds: list[int],
    target_name: str,
    model_name: str = "mid_tri_rdf_tabular_structure",
    split: str = "test",
    batch_size: int = 128,
    device: str = "auto",
    metadata: Path | None = None,
    max_samples: int | None = None,
    overwrite: bool = False,
    predictions_root: Path = Path("results/predictions"),
    experiment_name: str = "modality_dropout",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "modality_dropout_metrics.csv"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite")
    if metrics_path.exists() and overwrite:
        metrics_path.unlink()

    spec = next((candidate for candidate in TRI_TORCH_SPECS if candidate.name == model_name), None)
    if spec is None:
        raise ValueError(f"Unsupported dropout model {model_name!r}")

    device_obj = resolve_device(device)
    store = ProcessedFeatureStore(processed_root, CANONICAL_MODALITIES)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        checkpoint_path = _resolve_checkpoint(checkpoint_dir, seed)
        run_config = _load_run_config(checkpoint_path)
        target_transform = TargetTransform.from_config(run_config)
        seed_dir = Path(split_dir) / f"seed_{seed}"
        split_frame = pd.read_csv(seed_dir / f"{split}.csv")
        dataset = PublicationTriDataset(store, split_frame, preload=True)
        if max_samples is not None:
            dataset = Subset(dataset, list(range(min(max_samples, len(dataset)))))
        base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset
        dims = run_config.get("input_dims") or infer_tri_dims(base_dataset)
        model = build_torch_model(spec, dims, dropout=float(run_config.get("dropout", 0.1)))
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        model.to(device_obj)
        model.eval()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_tri)
        metadata_frame = _metadata_columns(split_frame, metadata)

        for condition in modality_dropout_conditions():
            sample_ids: list[str] = []
            y_true_parts: list[torch.Tensor] = []
            y_pred_parts: list[torch.Tensor] = []
            with torch.no_grad():
                for batch in loader:
                    inputs, targets = _move_batch(batch, device_obj)
                    predictions = masked_modality_forward(model, inputs, condition.available_modalities)
                    sample_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
                    y_true_parts.append(targets.detach().cpu())
                    y_pred_parts.append(predictions.detach().cpu())
            y_true = torch.cat(y_true_parts).reshape(-1)
            y_pred = torch.cat(y_pred_parts).reshape(-1)
            y_pred = target_transform.inverse_tensor(y_pred)
            metrics = regression_metrics(y_true, y_pred)
            predictions = pd.DataFrame(
                {
                    "sample_id": sample_ids,
                    "y_true": y_true.numpy(),
                    "y_pred": y_pred.numpy(),
                }
            ).merge(metadata_frame, on="sample_id", how="left")
            predictions["split"] = split
            predictions["model_name"] = condition.name
            predictions["modality_set"] = "+".join(condition.available_modalities)
            predictions["seed"] = seed
            prediction_path = (
                Path(predictions_root)
                / experiment_name
                / target_name
                / condition.name
                / f"seed_{seed}_predictions.csv"
            )
            if prediction_path.exists() and not overwrite:
                raise FileExistsError(f"{prediction_path} exists; pass --overwrite")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            predictions[
                [
                    "sample_id",
                    "formula",
                    "working_ion",
                    "anion_family",
                    "y_true",
                    "y_pred",
                    "split",
                    "model_name",
                    "modality_set",
                    "seed",
                ]
            ].to_csv(prediction_path, index=False)
            rows.append(
                {
                    "condition": condition.name,
                    "available_modalities": "+".join(condition.available_modalities),
                    "MAE": metrics["mae"],
                    "MSE": metrics["mse"],
                    "RMSE": metrics["rmse"],
                    "R2": metrics["r2"],
                    "n_samples": int(len(predictions)),
                    "seed": seed,
                    "split_id": f"seed_{seed}",
                    "target_name": target_name,
                    "model_name": model_name,
                    "checkpoint_path": str(checkpoint_path),
                    "prediction_path": str(prediction_path),
                    "target_transform": target_transform.kind,
                }
            )
            LOGGER.info("Seed %s %s: %s", seed, condition.name, rows[-1])
        del model
        if device_obj.type == "mps":
            torch.mps.empty_cache()

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(metrics_path, index=False)
    (output_dir / "modality_dropout_config.json").write_text(
        json.dumps(
            {
                "processed_root": str(processed_root),
                "checkpoint_dir": str(checkpoint_dir),
                "split_dir": str(split_dir),
                "output_dir": str(output_dir),
                "seeds": seeds,
                "target_name": target_name,
                "model_name": model_name,
                "split": split,
                "batch_size": batch_size,
                "device": str(device_obj),
                "metadata": str(metadata) if metadata else None,
                "max_samples": max_samples,
                "predictions_root": str(predictions_root),
                "experiment_name": experiment_name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return metrics_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate inference-time modality dropout.")
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/publication"))
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_name", required=True)
    parser.add_argument("--model_name", default="mid_tri_rdf_tabular_structure")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--predictions_root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--experiment_name", default="modality_dropout")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    run_modality_dropout(
        processed_root=args.processed_root,
        checkpoint_dir=args.checkpoint_dir,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        target_name=args.target_name,
        model_name=args.model_name,
        split=args.split,
        batch_size=args.batch_size,
        device=args.device,
        metadata=args.metadata,
        max_samples=args.max_samples,
        overwrite=args.overwrite,
        predictions_root=args.predictions_root,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
