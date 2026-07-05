from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from battery_fusion.fusion.cgcnn_multimodal import (
    RDFEncoder,
    StructureNetwork,
    TabularEncoder,
    build_multimodal_mid_fusion,
)
from battery_fusion.models.lstm import RdfLSTMRegressor
from battery_fusion.training.metrics import regression_metrics
from battery_fusion.training.runner import train_regressor
from battery_fusion.training.target_transform import TargetTransform, fit_target_transform

LOGGER = logging.getLogger(__name__)

MODEL_MODALITIES = {
    "composition": ("tabular",),
    "graph": ("structure",),
    "rdf": ("rdf",),
    "composition_graph": ("tabular", "structure"),
    "composition_rdf": ("tabular", "rdf"),
    "graph_rdf": ("structure", "rdf"),
    "full_fusion": ("tabular", "structure", "rdf"),
}
MODALITY_SET_NAMES = {
    "composition": "composition",
    "graph": "graph",
    "rdf": "rdf",
    "composition_graph": "composition+graph",
    "composition_rdf": "composition+rdf",
    "graph_rdf": "graph+rdf",
    "full_fusion": "composition+graph+rdf",
}


class EncoderRegressor(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.head = torch.nn.Sequential(torch.nn.Dropout(dropout), torch.nn.Linear(encoder.out_dim, 1))

    def forward(self, features: Any) -> torch.Tensor:
        return self.head(self.encoder(features)).squeeze(-1)


class PublicationSubsetMidFusionRegressor(torch.nn.Module):
    """Mid-fusion model for two-modality holdout runs using publication encoders."""

    def __init__(
        self,
        encoders: dict[str, torch.nn.Module],
        d_joint: int = 128,
        dropout: float = 0.0,
        n_heads: int = 2,
        n_layers: int = 3,
    ):
        super().__init__()
        self.modalities = tuple(encoders)
        self.encoders = torch.nn.ModuleDict(encoders)
        self.projections = torch.nn.ModuleDict(
            {
                modality: torch.nn.Sequential(
                    torch.nn.Linear(encoder.out_dim, d_joint),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                )
                for modality, encoder in encoders.items()
            }
        )
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_joint,
            nhead=n_heads,
            dim_feedforward=4 * d_joint,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.fusion_head = torch.nn.Sequential(
            torch.nn.Linear(d_joint, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(256, 1),
        )

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        tokens = []
        for modality in self.modalities:
            encoded = self.encoders[modality](batch[modality])
            tokens.append(self.projections[modality](encoded))
        fused = self.transformer(torch.stack(tokens, dim=1)).mean(dim=1)
        return self.fusion_head(fused).squeeze(-1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ProcessedFeatureStore:
    def __init__(self, processed_root: Path, modalities: tuple[str, ...]):
        self.processed_root = Path(processed_root)
        self.paths: dict[str, dict[str, Path]] = {}
        for modality in modalities:
            modality_paths: dict[str, Path] = {}
            for path in sorted((self.processed_root / modality).glob("*/*.pt")):
                modality_paths[path.stem] = path
            if not modality_paths:
                raise FileNotFoundError(f"No cached features found for {modality}")
            self.paths[modality] = modality_paths

    def load(self, modality: str, sample_id: str) -> dict[str, Any]:
        try:
            path = self.paths[modality][str(sample_id)]
        except KeyError as exc:
            raise FileNotFoundError(f"Missing {modality} feature for {sample_id}") from exc
        return torch.load(path, map_location="cpu", weights_only=False)


class HoldoutUnimodalDataset(Dataset):
    def __init__(
        self,
        store: ProcessedFeatureStore,
        split_frame: pd.DataFrame,
        modality: str,
        preload: bool = False,
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
        item = self.store.load(self.modality, sample_id)
        target = float(row["target"]) if "target" in self.frame.columns and pd.notna(row["target"]) else float(item["target"])
        return {
            "id_discharge": sample_id,
            "features": item["features"],
            "target": target,
        }


class HoldoutFusionDataset(Dataset):
    def __init__(
        self,
        store: ProcessedFeatureStore,
        split_frame: pd.DataFrame,
        modalities: tuple[str, ...],
        preload: bool = False,
    ):
        self.store = store
        self.frame = split_frame.reset_index(drop=True).copy()
        self.modalities = modalities
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
        sample: dict[str, Any] = {"id_discharge": sample_id, "modalities": {}}
        target = None
        for modality in self.modalities:
            item = self.store.load(modality, sample_id)
            sample["modalities"][modality] = item["features"]
            target = float(item["target"])
        if "target" in self.frame.columns and pd.notna(row["target"]):
            target = float(row["target"])
        sample["target"] = target
        return sample


def _adapter(batch: dict[str, Any], device: torch.device) -> tuple[Any, torch.Tensor]:
    features = batch["features"]
    if isinstance(features, torch.Tensor):
        features = features.to(device)
    elif isinstance(features, tuple):
        features = _move_graph_tuple(features, device)
    return features, batch["target"].to(device)


def _collate_graph_features(
    graphs: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
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


def _move_graph_tuple(
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


def _collate_unimodal_with_id(samples: list[dict[str, Any]]) -> dict[str, Any]:
    features = [sample["features"] for sample in samples]
    if isinstance(features[0], dict):
        batch_features = _collate_graph_features(features)
    else:
        batch_features = torch.stack(features)
    return {
        "id_discharge": [sample["id_discharge"] for sample in samples],
        "features": batch_features,
        "target": torch.tensor([sample["target"] for sample in samples], dtype=torch.float32),
    }


def _collate_publication_fusion_with_id(samples: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {
        "id_discharge": [sample["id_discharge"] for sample in samples],
        "target": torch.tensor([sample["target"] for sample in samples], dtype=torch.float32),
    }
    modalities = samples[0]["modalities"].keys()
    for modality in modalities:
        values = [sample["modalities"][modality] for sample in samples]
        if modality == "structure":
            batch[modality] = _collate_graph_features(values)
        else:
            batch[modality] = torch.stack(values)
    return batch


def _fusion_adapter(batch: dict[str, Any], device: torch.device) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    inputs = {}
    for key, value in batch.items():
        if key in {"id_discharge", "target"}:
            continue
        if isinstance(value, tuple):
            inputs[key] = _move_graph_tuple(value, device)
        else:
            inputs[key] = value.to(device)
    return inputs, batch["target"].to(device)


def _build_unimodal_model(modality: str, sample: dict[str, Any]) -> torch.nn.Module:
    features = sample["features"]
    if modality == "tabular":
        return EncoderRegressor(
            TabularEncoder(in_dim=int(features.shape[-1]), hidden_dim=256, out_dim=128, dropout=0.1),
            dropout=0.1,
        )
    if modality == "rdf":
        return RdfLSTMRegressor(input_size=int(features.shape[-1]), hidden_size=256, output_size=1)
    if modality == "structure":
        if "nbr_fea" not in features or "nbr_fea_idx" not in features:
            raise ValueError("Publication structure model requires atom_fea, nbr_fea, and nbr_fea_idx")
        encoder = StructureNetwork(
            orig_atom_fea_len=int(features["atom_fea"].shape[-1]),
            nbr_fea_len=int(features["nbr_fea"].shape[-1]),
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
        )
        return EncoderRegressor(encoder, dropout=0.1)
    raise ValueError(f"Unsupported modality {modality!r}")


def _infer_fusion_dims(dataset: HoldoutFusionDataset) -> dict[str, int]:
    sample = dataset[0]
    dims = {}
    for modality, features in sample["modalities"].items():
        if modality == "structure":
            if "nbr_fea" not in features or "nbr_fea_idx" not in features:
                raise ValueError("Publication structure fusion requires atom_fea, nbr_fea, and nbr_fea_idx")
            dims["atom_fea_dim"] = int(features["atom_fea"].shape[-1])
            dims["nbr_fea_dim"] = int(features["nbr_fea"].shape[-1])
        elif modality == "tabular":
            dims["tabular_dim"] = int(features.shape[-1])
        elif modality == "rdf":
            dims["rdf_dim"] = int(features.shape[-1])
        else:
            raise ValueError(f"Unsupported modality {modality!r}")
    return dims


def _publication_encoder(modality: str, input_dims: dict[str, int]) -> torch.nn.Module:
    if modality == "tabular":
        return TabularEncoder(in_dim=input_dims["tabular_dim"], hidden_dim=128, out_dim=64, dropout=0.0)
    if modality == "rdf":
        return RDFEncoder(in_dim=input_dims["rdf_dim"], hidden_dim=128, out_dim=64, dropout=0.0)
    if modality == "structure":
        return StructureNetwork(
            orig_atom_fea_len=input_dims["atom_fea_dim"],
            nbr_fea_len=input_dims["nbr_fea_dim"],
            atom_fea_len=64,
            n_conv=3,
            h_fea_len=128,
        )
    raise ValueError(f"Unsupported modality {modality!r}")


def _build_fusion_model(
    fusion: str,
    input_dims: dict[str, int],
    modalities: tuple[str, ...],
    model_name: str | None = None,
) -> torch.nn.Module:
    if fusion != "mid":
        raise ValueError("Publication anion-holdout fusion is architecture-consistent only with --fusion mid")
    if set(modalities) == {"tabular", "structure", "rdf"}:
        return build_multimodal_mid_fusion(
            tabular_dim=input_dims["tabular_dim"],
            rdf_dim=input_dims["rdf_dim"],
            atom_fea_dim=input_dims["atom_fea_dim"],
            nbr_fea_dim=input_dims["nbr_fea_dim"],
            dropout=0.0,
        )
    return PublicationSubsetMidFusionRegressor(
        {modality: _publication_encoder(modality, input_dims) for modality in modalities},
        d_joint=128,
        dropout=0.0,
        n_heads=2,
        n_layers=3,
    )


def _subset(dataset: Dataset, max_samples: int | None) -> Dataset:
    if max_samples is None:
        return dataset
    return Subset(dataset, list(range(min(max_samples, len(dataset)))))


def _predict_unimodal(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    target_transform: TargetTransform | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()
    ids: list[str] = []
    y_true_parts: list[torch.Tensor] = []
    y_pred_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = _adapter(batch, device_obj)
            predictions = model(inputs)
            ids.extend(str(sample_id) for sample_id in batch.get("id_discharge", []))
            y_true_parts.append(targets.detach().cpu())
            y_pred_parts.append(predictions.detach().cpu())
    y_true = torch.cat(y_true_parts)
    y_pred = torch.cat(y_pred_parts)
    if target_transform is not None:
        y_true = target_transform.inverse_tensor(y_true)
        y_pred = target_transform.inverse_tensor(y_pred)
    return regression_metrics(y_true, y_pred), pd.DataFrame(
        {"sample_id": ids, "y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    )


def _predict_fusion(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    target_transform: TargetTransform | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    device_obj = torch.device(device)
    model.to(device_obj)
    model.eval()
    ids: list[str] = []
    y_true_parts: list[torch.Tensor] = []
    y_pred_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = _fusion_adapter(batch, device_obj)
            predictions = model(inputs)
            ids.extend(str(sample_id) for sample_id in batch["id_discharge"])
            y_true_parts.append(targets.detach().cpu())
            y_pred_parts.append(predictions.detach().cpu())
    y_true = torch.cat(y_true_parts)
    y_pred = torch.cat(y_pred_parts)
    if target_transform is not None:
        y_true = target_transform.inverse_tensor(y_true)
        y_pred = target_transform.inverse_tensor(y_pred)
    return regression_metrics(y_true, y_pred), pd.DataFrame(
        {"sample_id": ids, "y_true": y_true.numpy(), "y_pred": y_pred.numpy()}
    )


def _metadata_columns(split_frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["sample_id", "formula", "working_ion", "anion_family"]
    metadata = split_frame.copy()
    for column in columns:
        if column not in metadata.columns:
            metadata[column] = ""
    return metadata[columns].drop_duplicates("sample_id")


def _finalize_predictions(
    predictions: pd.DataFrame,
    split_frame: pd.DataFrame,
    split: str,
    model_name: str,
    modality_set: str,
    seed: int,
) -> pd.DataFrame:
    metadata = _metadata_columns(split_frame)
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


def run_anion_holdout_experiment(
    split_dir: Path,
    processed_root: Path,
    models: list[str],
    output_dir: Path,
    seeds: list[int],
    fusion: str = "early",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 5e-4,
    device: str = "cpu",
    early_stopping_patience: int | None = None,
    early_stopping_min_delta: float = 0.0,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    overwrite: bool = False,
    config_path: Path | None = None,
    preload_features: bool = True,
    predictions_root: Path = Path("results/predictions"),
    target_transform: str = "none",
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "anion_holdout_metrics.csv"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to replace it")
    if metrics_path.exists() and overwrite:
        metrics_path.unlink()

    unknown = sorted(set(models).difference(MODEL_MODALITIES))
    if unknown:
        raise ValueError(f"Unsupported models: {unknown}")
    all_modalities = tuple(sorted({modality for name in models for modality in MODEL_MODALITIES[name]}))
    store = ProcessedFeatureStore(processed_root, all_modalities)
    metric_rows: list[dict[str, object]] = []

    for seed in seeds:
        set_seed(seed)
        seed_dir = Path(split_dir) / f"seed_{seed}"
        train_frame = pd.read_csv(seed_dir / "train.csv")
        val_frame = pd.read_csv(seed_dir / "val.csv")
        test_frame = pd.read_csv(seed_dir / "test.csv")
        seed_target_transform = fit_target_transform(train_frame["target"], target_transform)
        train_frame_model = seed_target_transform.transform_frame(train_frame)
        val_frame_model = seed_target_transform.transform_frame(val_frame)
        test_frame_model = seed_target_transform.transform_frame(test_frame)
        heldout_family = (
            str(test_frame["anion_family"].dropna().iloc[0])
            if "anion_family" in test_frame.columns and not test_frame.empty
            else Path(split_dir).name
        )
        LOGGER.info(
            "Seed %s split sizes: train=%s val=%s test=%s",
            seed,
            len(train_frame),
            len(val_frame),
            len(test_frame),
        )
        if "anion_family" in train_frame.columns:
            LOGGER.info("Train family counts: %s", train_frame["anion_family"].value_counts().to_dict())
            LOGGER.info("Test family counts: %s", test_frame["anion_family"].value_counts().to_dict())
        if "working_ion" in train_frame.columns:
            LOGGER.info("Train working-ion counts: %s", train_frame["working_ion"].value_counts().to_dict())

        for model_name in models:
            modalities = MODEL_MODALITIES[model_name]
            modality_set = MODALITY_SET_NAMES[model_name]
            model_output = output_dir / "checkpoints" / model_name / f"seed_{seed}"
            model_output.mkdir(parents=True, exist_ok=True)
            if len(modalities) == 1:
                modality = modalities[0]
                train_dataset = _subset(
                    HoldoutUnimodalDataset(store, train_frame_model, modality, preload=preload_features),
                    max_train_samples,
                )
                val_dataset = _subset(
                    HoldoutUnimodalDataset(store, val_frame_model, modality, preload=preload_features),
                    max_eval_samples,
                )
                test_dataset = _subset(
                    HoldoutUnimodalDataset(store, test_frame_model, modality, preload=preload_features),
                    max_eval_samples,
                )
                model = _build_unimodal_model(modality, train_dataset[0])
                collate_fn = _collate_unimodal_with_id
                adapter = _adapter
                predict_fn = _predict_unimodal
            else:
                train_dataset = _subset(
                    HoldoutFusionDataset(store, train_frame_model, modalities, preload=preload_features),
                    max_train_samples,
                )
                val_dataset = _subset(
                    HoldoutFusionDataset(store, val_frame_model, modalities, preload=preload_features),
                    max_eval_samples,
                )
                test_dataset = _subset(
                    HoldoutFusionDataset(store, test_frame_model, modalities, preload=preload_features),
                    max_eval_samples,
                )
                base_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
                model = _build_fusion_model(
                    fusion,
                    _infer_fusion_dims(base_dataset),
                    modalities,
                    model_name=model_name,
                )
                collate_fn = _collate_publication_fusion_with_id
                adapter = _fusion_adapter
                predict_fn = _predict_fusion

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
            history = train_regressor(
                model,
                train_loader,
                val_loader,
                epochs=epochs,
                learning_rate=learning_rate,
                device=device,
                batch_adapter=adapter,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                restore_best=True,
                target_transform=seed_target_transform,
            )
            torch.save(model.state_dict(), model_output / "model.pt")
            (model_output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
            (model_output / "run_config.json").write_text(
                json.dumps(
                    {
                        "model_name": model_name,
                        "modalities": list(modalities),
                        "modality_set": modality_set,
                        "seed": seed,
                        "fusion": fusion,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "device": str(device),
                        "early_stopping_patience": early_stopping_patience,
                        "early_stopping_min_delta": early_stopping_min_delta,
                        **seed_target_transform.to_config(),
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            test_metrics, predictions = predict_fn(
                model,
                test_loader,
                device,
                target_transform=seed_target_transform,
            )
            predictions = _finalize_predictions(
                predictions,
                test_frame,
                split="test",
                model_name=model_name,
                modality_set=modality_set,
                seed=seed,
            )
            prediction_path = (
                Path(predictions_root)
                / "anion_holdout"
                / model_name
                / f"seed_{seed}_predictions.csv"
            )
            if prediction_path.exists() and not overwrite:
                raise FileExistsError(f"{prediction_path} exists; pass --overwrite to replace it")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(prediction_path, index=False)

            row = {
                "heldout_family": heldout_family,
                "model_name": model_name,
                "modality_set": modality_set,
                "seed": seed,
                "MAE": test_metrics["mae"],
                "MSE": test_metrics["mse"],
                "RMSE": test_metrics["rmse"],
                "R2": test_metrics["r2"],
                "n_train": int(len(train_dataset)),
                "n_val": int(len(val_dataset)),
                "n_test": int(len(test_dataset)),
            }
            LOGGER.info("%s seed %s metrics: %s", model_name, seed, row)
            metric_rows.append(row)

    metric_frame = pd.DataFrame(metric_rows)
    metric_frame.to_csv(metrics_path, index=False)
    config = {
        "split_dir": str(split_dir),
        "processed_root": str(processed_root),
        "models": models,
        "seeds": seeds,
        "fusion": fusion,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "max_train_samples": max_train_samples,
        "max_eval_samples": max_eval_samples,
        "config_path": str(config_path) if config_path else None,
        "preload_features": preload_features,
        "predictions_root": str(predictions_root),
        "target_transform": target_transform,
    }
    (output_dir / "anion_holdout_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    return metric_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate anion-family holdout models.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--processed_root", type=Path, default=Path("data/processed/split_seed_42"))
    parser.add_argument("--models", nargs="+", default=["composition", "graph", "composition_graph", "full_fusion"])
    parser.add_argument("--output_dir", type=Path, default=Path("results/anion_holdout"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--fusion", choices=["early", "mid", "late"], default="early")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--no_preload_features", action="store_true")
    parser.add_argument("--predictions_root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--target_transform", choices=["none", "standardize"], default="none")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    run_anion_holdout_experiment(
        split_dir=args.split_dir,
        processed_root=args.processed_root,
        models=args.models,
        output_dir=args.output_dir,
        seeds=args.seeds,
        fusion=args.fusion,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        overwrite=args.overwrite,
        config_path=args.config,
        preload_features=not args.no_preload_features,
        predictions_root=args.predictions_root,
        target_transform=args.target_transform,
    )


if __name__ == "__main__":
    main()
