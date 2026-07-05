from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, Dataset, Subset

from battery_fusion.experiments.anion_holdout import ProcessedFeatureStore
from battery_fusion.fusion.cgcnn_multimodal import (
    RDFEncoder,
    StructureNetwork,
    TabularEncoder,
    build_multimodal_early_fusion,
    build_multimodal_mid_fusion,
)
from battery_fusion.models.lstm import RdfLSTMRegressor
from battery_fusion.training.metrics import regression_metrics
from battery_fusion.training.target_transform import TargetTransform, fit_target_transform
from battery_fusion.utils.chemistry_groups import assign_chemistry_groups

LOGGER = logging.getLogger(__name__)
MODALITIES = ("tabular", "rdf", "structure")


@dataclass(frozen=True)
class TorchModelSpec:
    name: str
    modality_set: str
    model_type: str
    modalities: tuple[str, ...]


@dataclass(frozen=True)
class LateFusionSpec:
    name: str
    modality_set: str
    base_models: tuple[str, ...]


BASE_MODEL_SPECS = (
    TorchModelSpec("unimodal_rdf_sequence", "rdf", "rdf_sequence", ("rdf",)),
    TorchModelSpec("unimodal_tabular", "composition", "tabular", ("tabular",)),
    TorchModelSpec("unimodal_structure", "graph", "structure", ("structure",)),
)
TRI_TORCH_SPECS = (
    TorchModelSpec(
        "early_tri_rdf_tabular_structure",
        "rdf+composition+graph",
        "early_tri",
        ("rdf", "tabular", "structure"),
    ),
    TorchModelSpec(
        "mid_tri_rdf_tabular_structure",
        "rdf+composition+graph",
        "mid_tri",
        ("rdf", "tabular", "structure"),
    ),
)
LATE_FUSION_SPECS = (
    LateFusionSpec(
        "late_dual_rdf_tabular",
        "rdf+composition",
        ("unimodal_rdf_sequence", "unimodal_tabular"),
    ),
    LateFusionSpec(
        "late_dual_rdf_structure",
        "rdf+graph",
        ("unimodal_rdf_sequence", "unimodal_structure"),
    ),
    LateFusionSpec(
        "late_dual_tabular_structure",
        "composition+graph",
        ("unimodal_tabular", "unimodal_structure"),
    ),
    LateFusionSpec(
        "late_tri_rdf_tabular_structure",
        "rdf+composition+graph",
        ("unimodal_rdf_sequence", "unimodal_tabular", "unimodal_structure"),
    ),
)


class EncoderRegressor(nn.Module):
    def __init__(self, encoder: nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(encoder.out_dim, 1))

    def forward(self, features: Any) -> torch.Tensor:
        return self.head(self.encoder(features)).squeeze(-1)


class PublicationUnimodalDataset(Dataset):
    def __init__(
        self,
        store: ProcessedFeatureStore,
        split_frame: pd.DataFrame,
        modality: str,
        preload: bool = True,
    ):
        self.store = store
        self.frame = split_frame.reset_index(drop=True).copy()
        self.modality = modality
        self.items = [self._load_index(index) for index in range(len(self.frame))] if preload else None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.items is not None:
            return self.items[index]
        return self._load_index(index)

    def _load_index(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        sample_id = str(row["sample_id"])
        return {
            "sample_id": sample_id,
            "features": self.store.load(self.modality, sample_id)["features"],
            "target": float(row["target"]),
        }


class PublicationTriDataset(Dataset):
    def __init__(
        self,
        store: ProcessedFeatureStore,
        split_frame: pd.DataFrame,
        preload: bool = True,
    ):
        self.store = store
        self.frame = split_frame.reset_index(drop=True).copy()
        self.items = [self._load_index(index) for index in range(len(self.frame))] if preload else None

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.items is not None:
            return self.items[index]
        return self._load_index(index)

    def _load_index(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        sample_id = str(row["sample_id"])
        return {
            "sample_id": sample_id,
            "tabular": self.store.load("tabular", sample_id)["features"].float(),
            "rdf": self.store.load("rdf", sample_id)["features"].float(),
            "structure": self.store.load("structure", sample_id)["features"],
            "target": float(row["target"]),
        }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device == "mps" and not torch.backends.mps.is_available():
        LOGGER.warning("Requested MPS, but torch.backends.mps.is_available() is false; using CPU.")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        LOGGER.warning("Requested CUDA, but torch.cuda.is_available() is false; using CPU.")
        return torch.device("cpu")
    return torch.device(device)


def load_target_metadata(
    raw_data: Path,
    target_col: str,
    sample_id_col: str = "id_discharge",
    formula_col: str = "formula_discharge",
    working_ion_col: str = "working_ion",
    assignment_output: Path | None = None,
) -> pd.DataFrame:
    raw = pd.read_csv(raw_data)
    required = {sample_id_col, formula_col, working_ion_col, target_col}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{raw_data} is missing required columns: {sorted(missing)}")
    raw = raw.drop_duplicates(subset=sample_id_col, keep="last")
    assignments = assign_chemistry_groups(
        raw,
        sample_id_col=sample_id_col,
        formula_col=formula_col,
        target_col=target_col,
        working_ion_col=working_ion_col,
    )
    if assignment_output is not None:
        assignment_output.parent.mkdir(parents=True, exist_ok=True)
        assignments.to_csv(assignment_output, index=False)
    return assignments


def sample_pool_from_processed(processed_root: Path, modalities: tuple[str, ...] = MODALITIES) -> list[str]:
    id_sets = []
    for modality in modalities:
        paths = sorted((Path(processed_root) / modality).glob("*/*.pt"))
        if not paths:
            raise FileNotFoundError(f"No cached {modality} features under {processed_root}")
        id_sets.append({path.stem for path in paths})
    return sorted(set.intersection(*id_sets))


def sample_order_from_labels(
    labels_path: Path | None,
    metadata: pd.DataFrame,
    sample_pool: list[str],
) -> list[str]:
    pool = set(sample_pool)
    available = set(metadata["sample_id"].astype(str))
    if labels_path is not None and Path(labels_path).exists():
        labels = pd.read_csv(labels_path)
        id_col = "id_discharge" if "id_discharge" in labels.columns else "sample_id"
        ordered = [str(sample_id) for sample_id in labels[id_col].tolist()]
    else:
        ordered = metadata["sample_id"].astype(str).tolist()
    ordered = [sample_id for sample_id in ordered if sample_id in pool and sample_id in available]
    missing = (pool & available).difference(ordered)
    ordered.extend(sorted(missing))
    if not ordered:
        raise ValueError("No sample ids remain after intersecting features and metadata")
    return ordered


def create_publication_splits(
    metadata: pd.DataFrame,
    sample_order: list[str],
    output_dir: Path,
    seeds: list[int],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    overwrite: bool = False,
    target_col: str | None = None,
) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    metadata_by_id = metadata.set_index("sample_id", drop=False)
    ids = [sample_id for sample_id in sample_order if sample_id in metadata_by_id.index]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_order contains duplicate sample ids")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        shuffled = ids[:]
        random.Random(seed).shuffle(shuffled)
        train_end = int(len(shuffled) * train_ratio)
        val_end = train_end + int(len(shuffled) * val_ratio)
        split_ids = {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for split, ids_for_split in split_ids.items():
            path = seed_dir / f"{split}.csv"
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            metadata_by_id.loc[ids_for_split].reset_index(drop=True).to_csv(path, index=False)
        config = {
            "seed": seed,
            "target_col": target_col,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_samples": len(ids),
            "n_train": len(split_ids["train"]),
            "n_val": len(split_ids["val"]),
            "n_test": len(split_ids["test"]),
        }
        (seed_dir / "split_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def collate_graph_features(graphs: list[dict[str, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    atom_fea_list = []
    nbr_fea_list = []
    nbr_fea_idx_list = []
    crystal_atom_idx = []
    base_idx = 0
    for graph in graphs:
        atom_fea = graph["atom_fea"].float()
        nbr_fea = graph["nbr_fea"].float()
        nbr_fea_idx = graph["nbr_fea_idx"].long()
        n_atoms = atom_fea.shape[0]
        atom_fea_list.append(atom_fea)
        nbr_fea_list.append(nbr_fea)
        nbr_fea_idx_list.append(nbr_fea_idx + base_idx)
        crystal_atom_idx.append(torch.arange(base_idx, base_idx + n_atoms, dtype=torch.long))
        base_idx += n_atoms
    return (
        torch.cat(atom_fea_list, dim=0),
        torch.cat(nbr_fea_list, dim=0),
        torch.cat(nbr_fea_idx_list, dim=0),
        crystal_atom_idx,
    )


def collate_unimodal(samples: list[dict[str, Any]]) -> dict[str, Any]:
    features = [sample["features"] for sample in samples]
    batch: dict[str, Any] = {
        "sample_id": [sample["sample_id"] for sample in samples],
        "target": torch.tensor([sample["target"] for sample in samples], dtype=torch.float32),
    }
    if isinstance(features[0], dict):
        batch["features"] = collate_graph_features(features)
    else:
        batch["features"] = torch.stack([feature.float() for feature in features])
    return batch


def collate_tri(samples: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_id": [sample["sample_id"] for sample in samples],
        "tabular": torch.stack([sample["tabular"].float() for sample in samples]),
        "rdf": torch.stack([sample["rdf"].float() for sample in samples]),
        "structure": collate_graph_features([sample["structure"] for sample in samples]),
        "target": torch.tensor([sample["target"] for sample in samples], dtype=torch.float32),
    }


def move_graph_tuple(
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


def unimodal_adapter(modality: str) -> Callable[[dict[str, Any], torch.device], tuple[Any, torch.Tensor]]:
    def _adapter(batch: dict[str, Any], device: torch.device) -> tuple[Any, torch.Tensor]:
        features = batch["features"]
        if modality == "structure":
            features = move_graph_tuple(features, device)
        else:
            features = features.to(device)
        return features, batch["target"].to(device)

    return _adapter


def tri_adapter(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, Any], torch.Tensor]:
    inputs = {
        "tabular": batch["tabular"].to(device),
        "rdf": batch["rdf"].to(device),
        "structure": move_graph_tuple(batch["structure"], device),
    }
    return inputs, batch["target"].to(device)


def infer_unimodal_dims(dataset: PublicationUnimodalDataset, modality: str) -> dict[str, int]:
    features = dataset[0]["features"]
    if modality == "tabular":
        return {"tabular_dim": int(features.shape[-1])}
    if modality == "rdf":
        return {"rdf_dim": int(features.shape[-1])}
    if modality == "structure":
        return {
            "atom_fea_dim": int(features["atom_fea"].shape[-1]),
            "nbr_fea_dim": int(features["nbr_fea"].shape[-1]),
        }
    raise ValueError(f"Unsupported modality {modality!r}")


def infer_tri_dims(dataset: PublicationTriDataset) -> dict[str, int]:
    sample = dataset[0]
    graph = sample["structure"]
    return {
        "tabular_dim": int(sample["tabular"].shape[-1]),
        "rdf_dim": int(sample["rdf"].shape[-1]),
        "atom_fea_dim": int(graph["atom_fea"].shape[-1]),
        "nbr_fea_dim": int(graph["nbr_fea"].shape[-1]),
    }


def build_torch_model(spec: TorchModelSpec, dims: dict[str, int], dropout: float) -> nn.Module:
    if spec.model_type == "rdf_sequence":
        return RdfLSTMRegressor(input_size=dims["rdf_dim"], hidden_size=256, output_size=1)
    if spec.model_type == "tabular":
        return EncoderRegressor(
            TabularEncoder(in_dim=dims["tabular_dim"], hidden_dim=256, out_dim=128, dropout=dropout),
            dropout=dropout,
        )
    if spec.model_type == "structure":
        encoder = StructureNetwork(
            orig_atom_fea_len=dims["atom_fea_dim"],
            nbr_fea_len=dims["nbr_fea_dim"],
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
        )
        return EncoderRegressor(encoder, dropout=dropout)
    if spec.model_type == "early_tri":
        return build_multimodal_early_fusion(**dims, dropout=dropout)
    if spec.model_type == "mid_tri":
        return build_multimodal_mid_fusion(**dims, dropout=0.0)
    raise ValueError(f"Unsupported model type {spec.model_type!r}")


def resolve_epochs_for_spec(
    spec: TorchModelSpec,
    default_epochs: int,
    rdf_epochs: int | None = None,
    tabular_epochs: int | None = None,
    structure_epochs: int | None = None,
    tri_epochs: int | None = None,
) -> int:
    if spec.model_type == "rdf_sequence" and rdf_epochs is not None:
        return rdf_epochs
    if spec.model_type == "tabular" and tabular_epochs is not None:
        return tabular_epochs
    if spec.model_type == "structure" and structure_epochs is not None:
        return structure_epochs
    if spec.model_type in {"early_tri", "mid_tri"} and tri_epochs is not None:
        return tri_epochs
    return default_epochs


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    batch_adapter: Callable[[dict[str, Any], torch.device], tuple[Any, torch.Tensor]],
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    scheduler_milestone: int,
    early_stopping_patience: int | None,
    early_stopping_min_delta: float,
    target_transform: TargetTransform | None = None,
    log_every: int = 5,
) -> list[dict[str, float]]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[scheduler_milestone], gamma=0.1)
    loss_fn = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            inputs, targets = batch_adapter(batch, device)
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = loss_fn(predictions.reshape(-1), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))
        scheduler.step()

        val_metrics, _ = predict_torch_model(
            model,
            val_loader,
            batch_adapter,
            device,
            target_transform=target_transform,
        )
        improved = val_metrics["mse"] < best_val_mse - early_stopping_min_delta
        if improved:
            best_val_mse = val_metrics["mse"]
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        early_stopped = (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val_mae": val_metrics["mae"],
            "val_mse": val_metrics["mse"],
            "val_rmse": val_metrics["rmse"],
            "val_r2": val_metrics["r2"],
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
            "epochs_without_improvement": epochs_without_improvement,
            "early_stopped": early_stopped,
        }
        if epoch == 1 or epoch % log_every == 0 or early_stopped:
            LOGGER.info("epoch %s/%s %s", epoch, epochs, row)
        history.append(row)
        if early_stopped:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_torch_model(
    model: nn.Module,
    loader: DataLoader,
    batch_adapter: Callable[[dict[str, Any], torch.device], tuple[Any, torch.Tensor]],
    device: torch.device,
    target_transform: TargetTransform | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.to(device)
    model.eval()
    sample_ids: list[str] = []
    y_true_parts: list[torch.Tensor] = []
    y_pred_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch_adapter(batch, device)
            predictions = model(inputs)
            sample_ids.extend(str(sample_id) for sample_id in batch["sample_id"])
            y_true_parts.append(targets.detach().cpu())
            y_pred_parts.append(predictions.detach().cpu())
    y_true = torch.cat(y_true_parts).reshape(-1)
    y_pred = torch.cat(y_pred_parts).reshape(-1)
    if target_transform is not None:
        y_true = target_transform.inverse_tensor(y_true)
        y_pred = target_transform.inverse_tensor(y_pred)
    return regression_metrics(y_true, y_pred), pd.DataFrame(
        {"sample_id": sample_ids, "y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    )


def fit_nonnegative_late_fusion(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    weights, *_ = np.linalg.lstsq(predictions, targets, rcond=None)
    weights = np.clip(weights.astype(float), 0.0, None)
    if weights.sum() > 0:
        return weights / weights.sum()
    return np.ones(predictions.shape[1], dtype=float) / predictions.shape[1]


def _split_frame_with_metadata(
    predictions: pd.DataFrame,
    split_frame: pd.DataFrame,
    split: str,
    model_name: str,
    modality_set: str,
    seed: int,
) -> pd.DataFrame:
    metadata_cols = ["sample_id", "formula", "working_ion", "anion_family"]
    metadata = split_frame[metadata_cols].drop_duplicates("sample_id")
    output = predictions.merge(metadata, on="sample_id", how="left")
    output["split"] = split
    output["model_name"] = model_name
    output["modality_set"] = modality_set
    output["seed"] = seed
    return output[
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
    ]


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return regression_metrics(torch.as_tensor(y_true), torch.as_tensor(y_pred))


def _subset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def _save_predictions(
    predictions_root: Path,
    experiment_name: str,
    target_name: str,
    model_name: str,
    seed: int,
    predictions_by_split: dict[str, pd.DataFrame],
    overwrite: bool,
) -> dict[str, Path]:
    output_paths: dict[str, Path] = {}
    for split, frame in predictions_by_split.items():
        path = predictions_root / experiment_name / target_name / model_name / f"seed_{seed}_{split}_predictions.csv"
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        output_paths[split] = path
    return output_paths


def _append_metric_rows(
    rows: list[dict[str, object]],
    experiment_name: str,
    target_name: str,
    model_name: str,
    modality_set: str,
    model_type: str,
    seed: int,
    split_frames: dict[str, pd.DataFrame],
    predictions_by_split: dict[str, pd.DataFrame],
    checkpoint_path: Path | None,
    prediction_paths: dict[str, Path],
    extra: dict[str, object] | None = None,
) -> None:
    extra = extra or {}
    for split, predictions in predictions_by_split.items():
        metrics = _metrics_from_arrays(
            predictions["y_true"].to_numpy(dtype=float),
            predictions["y_pred"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "experiment": experiment_name,
                "target_name": target_name,
                "model_name": model_name,
                "modality_set": modality_set,
                "model_type": model_type,
                "seed": seed,
                "split": split,
                "MAE": metrics["mae"],
                "MSE": metrics["mse"],
                "RMSE": metrics["rmse"],
                "R2": metrics["r2"],
                "n_train": int(len(split_frames["train"])),
                "n_val": int(len(split_frames["val"])),
                "n_test": int(len(split_frames["test"])),
                "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
                "prediction_path": str(prediction_paths[split]),
                **extra,
            }
        )


def _prediction_matrix(
    base_predictions: dict[str, dict[str, pd.DataFrame]],
    base_models: tuple[str, ...],
    split: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    merged = base_predictions[base_models[0]][split][["sample_id", "y_true", "y_pred"]].rename(
        columns={"y_pred": base_models[0]}
    )
    for base_model in base_models[1:]:
        merged = merged.merge(
            base_predictions[base_model][split][["sample_id", "y_pred"]].rename(columns={"y_pred": base_model}),
            on="sample_id",
            how="inner",
        )
    pred_matrix = merged[list(base_models)].to_numpy(dtype=float)
    targets = merged["y_true"].to_numpy(dtype=float)
    return pred_matrix, targets, merged


def _late_fusion_predictions(
    base_predictions: dict[str, dict[str, pd.DataFrame]],
    spec: LateFusionSpec,
    split_frames: dict[str, pd.DataFrame],
    seed: int,
) -> tuple[np.ndarray, dict[str, pd.DataFrame]]:
    val_matrix, val_targets, _ = _prediction_matrix(base_predictions, spec.base_models, "val")
    weights = fit_nonnegative_late_fusion(val_matrix, val_targets)
    outputs: dict[str, pd.DataFrame] = {}
    for split in ["train", "val", "test"]:
        matrix, targets, merged = _prediction_matrix(base_predictions, spec.base_models, split)
        predictions = merged[["sample_id", "y_true"]].copy()
        predictions["y_pred"] = matrix @ weights
        outputs[split] = _split_frame_with_metadata(
            predictions,
            split_frames[split],
            split=split,
            model_name=spec.name,
            modality_set=spec.modality_set,
            seed=seed,
        )
    return weights, outputs


def run_publication_matrix(
    processed_root: Path,
    raw_data: Path,
    target_col: str,
    split_dir: Path,
    output_dir: Path,
    seeds: list[int],
    labels_path: Path | None = Path("data/labels/labels_keep_last.csv"),
    assignment_output: Path | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    epochs: int = 50,
    rdf_epochs: int | None = 300,
    tabular_epochs: int | None = None,
    structure_epochs: int | None = None,
    tri_epochs: int | None = None,
    batch_size: int = 256,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-5,
    rdf_weight_decay: float = 1e-4,
    scheduler_milestone: int = 20,
    early_stopping_patience: int | None = 50,
    early_stopping_min_delta: float = 0.0,
    dropout: float = 0.1,
    device: str = "auto",
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    overwrite: bool = False,
    skip_split_creation: bool = False,
    preload_features: bool = True,
    predictions_root: Path = Path("results/predictions"),
    experiment_name: str = "publication_random",
    models: list[str] | None = None,
    target_transform: str = "none",
) -> pd.DataFrame:
    target_name = target_col
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "publication_metrics.csv"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to replace it")
    if metrics_path.exists() and overwrite:
        metrics_path.unlink()

    metadata = load_target_metadata(raw_data, target_col=target_col, assignment_output=assignment_output)
    sample_pool = sample_pool_from_processed(processed_root)
    sample_order = sample_order_from_labels(labels_path, metadata, sample_pool)
    metadata = metadata[metadata["sample_id"].isin(sample_order)].reset_index(drop=True)
    if not skip_split_creation:
        create_publication_splits(
            metadata=metadata,
            sample_order=sample_order,
            output_dir=split_dir,
            seeds=seeds,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            overwrite=overwrite,
            target_col=target_col,
        )

    selected = set(models) if models else None
    base_specs = [spec for spec in BASE_MODEL_SPECS if selected is None or spec.name in selected]
    tri_specs = [spec for spec in TRI_TORCH_SPECS if selected is None or spec.name in selected]
    late_specs = [spec for spec in LATE_FUSION_SPECS if selected is None or spec.name in selected]
    required_base_names = {base for spec in late_specs for base in spec.base_models}
    for spec in BASE_MODEL_SPECS:
        if spec.name in required_base_names and spec not in base_specs:
            base_specs.append(spec)

    device_obj = resolve_device(device)
    LOGGER.info("Using device: %s", device_obj)
    store = ProcessedFeatureStore(processed_root, MODALITIES)
    rows: list[dict[str, object]] = []

    for seed in seeds:
        set_seed(seed)
        seed_dir = Path(split_dir) / f"seed_{seed}"
        split_frames = {
            split: pd.read_csv(seed_dir / f"{split}.csv")
            for split in ["train", "val", "test"]
        }
        seed_target_transform = fit_target_transform(split_frames["train"]["target"], target_transform)
        training_split_frames = {
            split: seed_target_transform.transform_frame(frame)
            for split, frame in split_frames.items()
        }
        LOGGER.info(
            "Seed %s split sizes: train=%s val=%s test=%s",
            seed,
            len(split_frames["train"]),
            len(split_frames["val"]),
            len(split_frames["test"]),
        )
        LOGGER.info("Seed %s train anion counts: %s", seed, split_frames["train"]["anion_family"].value_counts().to_dict())
        LOGGER.info("Seed %s train working-ion counts: %s", seed, split_frames["train"]["working_ion"].value_counts().to_dict())
        base_predictions: dict[str, dict[str, pd.DataFrame]] = {}

        for spec in base_specs:
            modality = spec.modalities[0]
            LOGGER.info("Training %s seed %s", spec.name, seed)
            datasets = {
                split: _subset(
                    PublicationUnimodalDataset(store, frame, modality=modality, preload=preload_features),
                    max_train_samples if split == "train" else max_eval_samples,
                )
                for split, frame in training_split_frames.items()
            }
            loaders = {
                split: DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                    collate_fn=collate_unimodal,
                    drop_last=False,
                )
                for split, dataset in datasets.items()
            }
            base_dataset = datasets["train"].dataset if isinstance(datasets["train"], Subset) else datasets["train"]
            dims = infer_unimodal_dims(base_dataset, modality)
            model = build_torch_model(spec, dims, dropout=dropout)
            adapter = unimodal_adapter(modality)
            wd = rdf_weight_decay if spec.model_type == "rdf_sequence" else weight_decay
            model_epochs = resolve_epochs_for_spec(
                spec,
                default_epochs=epochs,
                rdf_epochs=rdf_epochs,
                tabular_epochs=tabular_epochs,
                structure_epochs=structure_epochs,
                tri_epochs=tri_epochs,
            )
            history = train_torch_model(
                model=model,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                batch_adapter=adapter,
                device=device_obj,
                epochs=model_epochs,
                learning_rate=learning_rate,
                weight_decay=wd,
                scheduler_milestone=scheduler_milestone,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                target_transform=seed_target_transform,
            )
            checkpoint_dir = output_dir / "checkpoints" / spec.name / f"seed_{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            (checkpoint_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "model_name": spec.name,
                        "model_type": spec.model_type,
                        "modalities": list(spec.modalities),
                        "target_col": target_col,
                        "seed": seed,
                        "input_dims": dims,
                        "epochs": model_epochs,
                        "default_epochs": epochs,
                        "rdf_epochs": rdf_epochs,
                        "tabular_epochs": tabular_epochs,
                        "structure_epochs": structure_epochs,
                        "tri_epochs": tri_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": wd,
                        "scheduler_milestone": scheduler_milestone,
                        "early_stopping_patience": early_stopping_patience,
                        "early_stopping_min_delta": early_stopping_min_delta,
                        "device": str(device_obj),
                        **seed_target_transform.to_config(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            predictions_by_split = {}
            for split, loader in loaders.items():
                _metrics, predictions = predict_torch_model(
                    model,
                    loader,
                    adapter,
                    device_obj,
                    target_transform=seed_target_transform,
                )
                predictions_by_split[split] = _split_frame_with_metadata(
                    predictions,
                    split_frames[split],
                    split=split,
                    model_name=spec.name,
                    modality_set=spec.modality_set,
                    seed=seed,
                )
            prediction_paths = _save_predictions(
                predictions_root,
                experiment_name,
                target_name,
                spec.name,
                seed,
                predictions_by_split,
                overwrite=overwrite,
            )
            _append_metric_rows(
                rows,
                experiment_name,
                target_name,
                spec.name,
                spec.modality_set,
                spec.model_type,
                seed,
                split_frames,
                predictions_by_split,
                checkpoint_path,
                prediction_paths,
            )
            base_predictions[spec.name] = predictions_by_split
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            del model
            if device_obj.type == "mps":
                torch.mps.empty_cache()

        for spec in tri_specs:
            LOGGER.info("Training %s seed %s", spec.name, seed)
            datasets = {
                split: _subset(
                    PublicationTriDataset(store, frame, preload=preload_features),
                    max_train_samples if split == "train" else max_eval_samples,
                )
                for split, frame in training_split_frames.items()
            }
            loaders = {
                split: DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=(split == "train"),
                    collate_fn=collate_tri,
                    drop_last=False,
                )
                for split, dataset in datasets.items()
            }
            base_dataset = datasets["train"].dataset if isinstance(datasets["train"], Subset) else datasets["train"]
            dims = infer_tri_dims(base_dataset)
            model = build_torch_model(spec, dims, dropout=dropout)
            model_epochs = resolve_epochs_for_spec(
                spec,
                default_epochs=epochs,
                rdf_epochs=rdf_epochs,
                tabular_epochs=tabular_epochs,
                structure_epochs=structure_epochs,
                tri_epochs=tri_epochs,
            )
            history = train_torch_model(
                model=model,
                train_loader=loaders["train"],
                val_loader=loaders["val"],
                batch_adapter=tri_adapter,
                device=device_obj,
                epochs=model_epochs,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                scheduler_milestone=scheduler_milestone,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                target_transform=seed_target_transform,
            )
            checkpoint_dir = output_dir / "checkpoints" / spec.name / f"seed_{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "model.pt"
            torch.save(model.state_dict(), checkpoint_path)
            (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            (checkpoint_dir / "run_config.json").write_text(
                json.dumps(
                    {
                        "model_name": spec.name,
                        "model_type": spec.model_type,
                        "modalities": list(spec.modalities),
                        "target_col": target_col,
                        "seed": seed,
                        "input_dims": dims,
                        "epochs": model_epochs,
                        "default_epochs": epochs,
                        "rdf_epochs": rdf_epochs,
                        "tabular_epochs": tabular_epochs,
                        "structure_epochs": structure_epochs,
                        "tri_epochs": tri_epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "scheduler_milestone": scheduler_milestone,
                        "early_stopping_patience": early_stopping_patience,
                        "early_stopping_min_delta": early_stopping_min_delta,
                        "device": str(device_obj),
                        **seed_target_transform.to_config(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            predictions_by_split = {}
            for split, loader in loaders.items():
                _metrics, predictions = predict_torch_model(
                    model,
                    loader,
                    tri_adapter,
                    device_obj,
                    target_transform=seed_target_transform,
                )
                predictions_by_split[split] = _split_frame_with_metadata(
                    predictions,
                    split_frames[split],
                    split=split,
                    model_name=spec.name,
                    modality_set=spec.modality_set,
                    seed=seed,
                )
            prediction_paths = _save_predictions(
                predictions_root,
                experiment_name,
                target_name,
                spec.name,
                seed,
                predictions_by_split,
                overwrite=overwrite,
            )
            _append_metric_rows(
                rows,
                experiment_name,
                target_name,
                spec.name,
                spec.modality_set,
                spec.model_type,
                seed,
                split_frames,
                predictions_by_split,
                checkpoint_path,
                prediction_paths,
            )
            pd.DataFrame(rows).to_csv(metrics_path, index=False)
            del model
            if device_obj.type == "mps":
                torch.mps.empty_cache()

        for spec in late_specs:
            LOGGER.info("Fitting %s seed %s from validation predictions", spec.name, seed)
            weights, predictions_by_split = _late_fusion_predictions(base_predictions, spec, split_frames, seed)
            checkpoint_dir = output_dir / "checkpoints" / spec.name / f"seed_{seed}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            weights_path = checkpoint_dir / "late_fusion_weights.json"
            weights_path.write_text(
                json.dumps(
                    {
                        "model_name": spec.name,
                        "base_models": list(spec.base_models),
                        "weights": {name: float(weight) for name, weight in zip(spec.base_models, weights)},
                        "target_col": target_col,
                        "seed": seed,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            prediction_paths = _save_predictions(
                predictions_root,
                experiment_name,
                target_name,
                spec.name,
                seed,
                predictions_by_split,
                overwrite=overwrite,
            )
            _append_metric_rows(
                rows,
                experiment_name,
                target_name,
                spec.name,
                spec.modality_set,
                "late_fusion_nnls",
                seed,
                split_frames,
                predictions_by_split,
                weights_path,
                prediction_paths,
                extra={"late_fusion_weights": json.dumps({name: float(weight) for name, weight in zip(spec.base_models, weights)})},
            )
            pd.DataFrame(rows).to_csv(metrics_path, index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    (output_dir / "publication_run_config.json").write_text(
        json.dumps(
            {
                "processed_root": str(processed_root),
                "raw_data": str(raw_data),
                "target_col": target_col,
                "split_dir": str(split_dir),
                "labels_path": str(labels_path) if labels_path else None,
                "seeds": seeds,
                "epochs": epochs,
                "rdf_epochs": rdf_epochs,
                "tabular_epochs": tabular_epochs,
                "structure_epochs": structure_epochs,
                "tri_epochs": tri_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "rdf_weight_decay": rdf_weight_decay,
                "scheduler_milestone": scheduler_milestone,
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "dropout": dropout,
                "device": str(device_obj),
                "max_train_samples": max_train_samples,
                "max_eval_samples": max_eval_samples,
                "preload_features": preload_features,
                "experiment_name": experiment_name,
                "models": models,
                "target_transform": target_transform,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clean publication training matrix.")
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/legacy_rdf_split_seed_42"))
    parser.add_argument("--raw_data", type=Path, default=Path("data/raw/mp_total.csv"))
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--labels_path", type=Path, default=Path("data/labels/labels_keep_last.csv"))
    parser.add_argument("--assignment_output", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--rdf_epochs", type=int, default=300)
    parser.add_argument("--tabular_epochs", type=int, default=None)
    parser.add_argument("--structure_epochs", type=int, default=None)
    parser.add_argument("--tri_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--rdf_weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler_milestone", type=int, default=20)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no_preload", action="store_true")
    parser.add_argument("--skip_split_creation", action="store_true")
    parser.add_argument("--predictions_root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--experiment_name", default="publication_random")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--target_transform", choices=["none", "standardize"], default="none")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    run_publication_matrix(
        processed_root=args.processed_root,
        raw_data=args.raw_data,
        target_col=args.target_col,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        seeds=args.seeds,
        labels_path=args.labels_path,
        assignment_output=args.assignment_output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        rdf_epochs=args.rdf_epochs,
        tabular_epochs=args.tabular_epochs,
        structure_epochs=args.structure_epochs,
        tri_epochs=args.tri_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rdf_weight_decay=args.rdf_weight_decay,
        scheduler_milestone=args.scheduler_milestone,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        dropout=args.dropout,
        device=args.device,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        overwrite=args.overwrite,
        skip_split_creation=args.skip_split_creation,
        preload_features=not args.no_preload,
        predictions_root=args.predictions_root,
        experiment_name=args.experiment_name,
        models=args.models,
        target_transform=args.target_transform,
    )


if __name__ == "__main__":
    main()
