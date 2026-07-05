from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "alignn_pretrained_rf"
MODALITY_SET = "alignn_pretrained_structure"
DEFAULT_PRETRAINED_MODELS = [
    "jv_formation_energy_peratom_alignn",
    "mp_e_form_alignn",
    "jv_optb88vdw_bandgap_alignn",
    "mp_gappbe_alignn",
]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def regression_metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    diff = y_pred - y_true
    mse = float(np.mean(diff**2))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(math.sqrt(mse))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum(diff**2) / denom) if denom > 0 else 0.0
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def load_split_frames(split_dir: Path, seed: int) -> dict[str, pd.DataFrame]:
    seed_dir = Path(split_dir) / f"seed_{seed}"
    frames = {split: pd.read_csv(seed_dir / f"{split}.csv") for split in ["train", "val", "test"]}
    for split, frame in frames.items():
        required = {"sample_id", "formula", "working_ion", "anion_family", "target"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{seed_dir / f'{split}.csv'} missing columns: {sorted(missing)}")
    return frames


def feature_cache_path(feature_cache_dir: Path, model_name: str, feature_mode: str) -> Path:
    return Path(feature_cache_dir) / f"{safe_name(model_name)}_{feature_mode}.csv"


def _prefix_feature_columns(frame: pd.DataFrame, model_name: str) -> pd.DataFrame:
    output = frame.copy()
    feature_columns = [column for column in output.columns if column != "sample_id"]
    rename = {
        column: f"{safe_name(model_name)}_{idx}"
        for idx, column in enumerate(feature_columns)
    }
    return output.rename(columns=rename)


def _write_structure_link_or_copy(
    source: Path,
    destination: Path,
    overwrite: bool,
    use_symlinks: bool,
) -> None:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        destination.unlink()
    if use_symlinks:
        destination.symlink_to(source.resolve())
    else:
        shutil.copy2(source, destination)


def _stage_structure_file(
    sample_id: str,
    source_cif: Path,
    data_dir: Path,
    file_format: str,
    overwrite: bool,
    use_symlinks: bool,
) -> str:
    if file_format == "cif":
        destination_name = f"{sample_id}.cif"
        _write_structure_link_or_copy(
            source=source_cif,
            destination=data_dir / destination_name,
            overwrite=overwrite,
            use_symlinks=use_symlinks,
        )
        return destination_name
    if file_format == "poscar":
        destination_name = f"{sample_id}.vasp"
        destination = data_dir / destination_name
        if destination.exists() and not overwrite:
            return destination_name
        if destination.exists():
            destination.unlink()
        try:
            from pymatgen.core import Structure
            from pymatgen.io.vasp import Poscar
        except Exception as exc:
            raise RuntimeError(
                "Staging ALIGNN as POSCAR requires pymatgen in the runner "
                "environment. Install pymatgen or pass --file_format cif and "
                "install cif2cell for ALIGNN/JARVIS CIF parsing."
            ) from exc
        structure = Structure.from_file(source_cif)
        Poscar(structure).write_file(destination)
        return destination_name
    raise ValueError(
        f"Unsupported ALIGNN staging file_format={file_format!r}. "
        "Use 'poscar' to avoid cif2cell, or 'cif' if cif2cell is installed."
    )


def _stage_manifest_for_missing_ids(
    sample_ids: list[str],
    cif_dir: Path,
    staging_dir: Path,
    file_format: str,
    overwrite: bool,
) -> pd.DataFrame:
    staging_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    for sample_id in sample_ids:
        source_cif = Path(cif_dir) / f"{sample_id}.cif"
        if not source_cif.exists():
            raise FileNotFoundError(source_cif)
        structure_name = _stage_structure_file(
            sample_id=sample_id,
            source_cif=source_cif,
            data_dir=staging_dir,
            file_format=file_format,
            overwrite=overwrite,
            use_symlinks=False,
        )
        rows.append(
            {
                "sample_id": sample_id,
                "structure_path": str((staging_dir / structure_name).resolve()),
            }
        )
    return pd.DataFrame(rows)


def _run_featurizer(
    manifest_path: Path,
    output_csv: Path,
    model_name: str,
    alignn_python: str,
    file_format: str,
    feature_mode: str,
    cutoff: float,
    max_neighbors: int,
    overwrite: bool,
) -> None:
    command = [
        alignn_python,
        "-m",
        "battery_fusion.experiments.alignn_pretrained_featurize",
        "--manifest",
        str(manifest_path),
        "--output_csv",
        str(output_csv),
        "--model_name",
        model_name,
        "--file_format",
        file_format,
        "--feature_mode",
        feature_mode,
        "--cutoff",
        str(cutoff),
        "--max_neighbors",
        str(max_neighbors),
    ]
    if overwrite:
        command.append("--overwrite")
    env = os.environ.copy()
    src_path = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    LOGGER.info("ALIGNN pretrained featurizer: %s", " ".join(command))
    subprocess.run(command, check=True, env=env)


def ensure_pretrained_feature_frame(
    sample_ids: list[str],
    model_name: str,
    cif_dir: Path,
    feature_cache_dir: Path,
    alignn_python: str,
    file_format: str = "poscar",
    feature_mode: str = "readout",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    overwrite: bool = False,
    skip_featurization: bool = False,
) -> pd.DataFrame:
    cache_path = feature_cache_path(feature_cache_dir, model_name, feature_mode)
    existing = pd.read_csv(cache_path) if cache_path.exists() else pd.DataFrame()
    existing_ids = set(existing["sample_id"].astype(str)) if "sample_id" in existing.columns else set()
    requested = list(dict.fromkeys(str(sample_id) for sample_id in sample_ids))
    missing = [sample_id for sample_id in requested if sample_id not in existing_ids]
    if missing:
        if skip_featurization:
            raise FileNotFoundError(
                f"{cache_path} does not contain cached features for {len(missing)} samples."
            )
        feature_cache_dir.mkdir(parents=True, exist_ok=True)
        staging_dir = feature_cache_dir / "staged_structures" / safe_name(model_name)
        manifest = _stage_manifest_for_missing_ids(
            sample_ids=missing,
            cif_dir=cif_dir,
            staging_dir=staging_dir,
            file_format=file_format,
            overwrite=overwrite,
        )
        manifest_path = feature_cache_dir / "manifests" / f"{safe_name(model_name)}_{feature_mode}_missing.csv"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(manifest_path, index=False)
        new_feature_path = feature_cache_dir / "partial" / f"{safe_name(model_name)}_{feature_mode}_latest.csv"
        _run_featurizer(
            manifest_path=manifest_path,
            output_csv=new_feature_path,
            model_name=model_name,
            alignn_python=alignn_python,
            file_format=file_format,
            feature_mode=feature_mode,
            cutoff=cutoff,
            max_neighbors=max_neighbors,
            overwrite=True,
        )
        new_features = pd.read_csv(new_feature_path)
        existing = pd.concat([existing, new_features], ignore_index=True)
        existing = existing.drop_duplicates(subset="sample_id", keep="last")
        existing.to_csv(cache_path, index=False)
    if existing.empty:
        raise FileNotFoundError(cache_path)
    return _prefix_feature_columns(existing, model_name)


def align_feature_matrix_to_frame(
    frame: pd.DataFrame,
    feature_frames: list[pd.DataFrame],
) -> tuple[np.ndarray, list[str]]:
    expected_ids = frame["sample_id"].astype(str).tolist()
    merged = pd.DataFrame({"sample_id": expected_ids})
    for feature_frame in feature_frames:
        merged = merged.merge(feature_frame, on="sample_id", how="left")
    if merged.isna().any().any():
        missing = merged.loc[merged.isna().any(axis=1), "sample_id"].tolist()
        raise ValueError(f"Missing pretrained ALIGNN features for sample ids: {missing[:10]}")
    feature_columns = [column for column in merged.columns if column != "sample_id"]
    return merged[feature_columns].to_numpy(dtype=np.float32), feature_columns


def prediction_frame(
    split_frame: pd.DataFrame,
    y_pred: np.ndarray,
    split: str,
    model_name: str,
    seed: int,
    target_col: str,
    pretrained_models: list[str],
) -> pd.DataFrame:
    output = split_frame[["sample_id", "formula", "working_ion", "anion_family", "target"]].copy()
    output = output.rename(columns={"target": "y_true"})
    output["y_pred"] = np.asarray(y_pred, dtype=float)
    output["split"] = split
    output["model_name"] = model_name
    output["modality_set"] = MODALITY_SET
    output["target_col"] = target_col
    output["seed"] = seed
    output["pretrained_models"] = ";".join(pretrained_models)
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
            "target_col",
            "seed",
            "pretrained_models",
        ]
    ]


def run_alignn_pretrained_baseline(
    split_dir: Path,
    cif_dir: Path,
    output_dir: Path,
    target_col: str,
    seeds: list[int],
    pretrained_models: list[str] | None = None,
    feature_mode: str = "readout",
    feature_cache_dir: Path | None = None,
    predictions_root: Path = Path("results/predictions"),
    experiment_name: str = "final_publication_alignn_pretrained",
    alignn_python: str = ".venv-alignn/bin/python",
    file_format: str = "poscar",
    cutoff: float = 8.0,
    max_neighbors: int = 12,
    n_estimators: int = 500,
    n_jobs: int = -1,
    skip_featurization: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "alignn_pretrained_rf_metrics.csv"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite")
    models = pretrained_models or DEFAULT_PRETRAINED_MODELS
    cache_dir = Path(feature_cache_dir) if feature_cache_dir else output_dir / "pretrained_features"
    rows: list[dict[str, object]] = []
    for seed in seeds:
        frames = load_split_frames(split_dir, seed)
        all_ids = pd.concat(
            [frames[split][["sample_id"]] for split in ["train", "val", "test"]],
            ignore_index=True,
        )["sample_id"].astype(str).tolist()
        feature_frames = [
            ensure_pretrained_feature_frame(
                sample_ids=all_ids,
                model_name=model_name,
                cif_dir=cif_dir,
                feature_cache_dir=cache_dir,
                alignn_python=alignn_python,
                file_format=file_format,
                feature_mode=feature_mode,
                cutoff=cutoff,
                max_neighbors=max_neighbors,
                overwrite=overwrite,
                skip_featurization=skip_featurization,
            )
            for model_name in models
        ]
        x_by_split: dict[str, np.ndarray] = {}
        feature_columns: list[str] = []
        for split, frame in frames.items():
            x_by_split[split], feature_columns = align_feature_matrix_to_frame(frame, feature_frames)
        y_by_split = {
            split: frame["target"].to_numpy(dtype=float)
            for split, frame in frames.items()
        }
        estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=n_jobs,
            min_samples_leaf=1,
        )
        estimator.fit(x_by_split["train"], y_by_split["train"])
        model_dir = output_dir / "models" / MODEL_NAME / f"seed_{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        if model_path.exists() and not overwrite:
            raise FileExistsError(f"{model_path} exists; pass --overwrite")
        with open(model_path, "wb") as handle:
            pickle.dump(
                {
                    "estimator": estimator,
                    "feature_columns": feature_columns,
                    "pretrained_models": models,
                    "feature_mode": feature_mode,
                },
                handle,
            )
        for split in ["train", "val", "test"]:
            predictions = estimator.predict(x_by_split[split])
            metrics = regression_metrics_np(y_by_split[split], predictions)
            pred_frame = prediction_frame(
                split_frame=frames[split],
                y_pred=predictions,
                split=split,
                model_name=MODEL_NAME,
                seed=seed,
                target_col=target_col,
                pretrained_models=models,
            )
            prediction_path = (
                Path(predictions_root)
                / experiment_name
                / target_col
                / MODEL_NAME
                / f"seed_{seed}_{split}_predictions.csv"
            )
            if prediction_path.exists() and not overwrite:
                raise FileExistsError(f"{prediction_path} exists; pass --overwrite")
            prediction_path.parent.mkdir(parents=True, exist_ok=True)
            pred_frame.to_csv(prediction_path, index=False)
            rows.append(
                {
                    "experiment_name": experiment_name,
                    "target_col": target_col,
                    "model_name": MODEL_NAME,
                    "modality_set": MODALITY_SET,
                    "seed": seed,
                    "split": split,
                    **metrics,
                    "n_samples": int(len(frames[split])),
                    "n_train": int(len(frames["train"])),
                    "n_val": int(len(frames["val"])),
                    "n_test": int(len(frames["test"])),
                    "feature_mode": feature_mode,
                    "pretrained_models": ";".join(models),
                    "n_features": int(len(feature_columns)),
                    "model_path": str(model_path),
                    "prediction_path": str(prediction_path),
                }
            )
            LOGGER.info(
                "%s seed %s %s MAE=%.4f R2=%.4f",
                MODEL_NAME,
                seed,
                split,
                metrics["MAE"],
                metrics["R2"],
            )
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(metrics_path, index=False)
    (output_dir / "alignn_pretrained_rf_config.json").write_text(
        json.dumps(
            {
                "split_dir": str(split_dir),
                "cif_dir": str(cif_dir),
                "output_dir": str(output_dir),
                "target_col": target_col,
                "seeds": seeds,
                "pretrained_models": models,
                "feature_mode": feature_mode,
                "feature_cache_dir": str(cache_dir),
                "predictions_root": str(predictions_root),
                "experiment_name": experiment_name,
                "alignn_python": alignn_python,
                "file_format": file_format,
                "cutoff": cutoff,
                "max_neighbors": max_neighbors,
                "n_estimators": n_estimators,
                "n_jobs": n_jobs,
                "skip_featurization": skip_featurization,
                "baseline_type": "frozen pretrained ALIGNN readout features with RandomForestRegressor",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train frozen-pretrained ALIGNN + RF baseline.")
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--cif_dir", type=Path, default=Path("data/raw/cifs"))
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--pretrained_models", nargs="+", default=DEFAULT_PRETRAINED_MODELS)
    parser.add_argument("--feature_mode", choices=["readout", "prediction"], default="readout")
    parser.add_argument("--feature_cache_dir", type=Path, default=None)
    parser.add_argument("--predictions_root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--experiment_name", default="final_publication_alignn_pretrained")
    parser.add_argument("--alignn_python", default=".venv-alignn/bin/python")
    parser.add_argument("--file_format", choices=["poscar", "cif"], default="poscar")
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--max_neighbors", type=int, default=12)
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--skip_featurization", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    run_alignn_pretrained_baseline(
        split_dir=args.split_dir,
        cif_dir=args.cif_dir,
        output_dir=args.output_dir,
        target_col=args.target_col,
        seeds=args.seeds,
        pretrained_models=args.pretrained_models,
        feature_mode=args.feature_mode,
        feature_cache_dir=args.feature_cache_dir,
        predictions_root=args.predictions_root,
        experiment_name=args.experiment_name,
        alignn_python=args.alignn_python,
        file_format=args.file_format,
        cutoff=args.cutoff,
        max_neighbors=args.max_neighbors,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        skip_featurization=args.skip_featurization,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
