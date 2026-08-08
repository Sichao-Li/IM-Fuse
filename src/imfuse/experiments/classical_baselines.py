from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
from pathlib import Path
from typing import Callable

from chemparse import parse_formula
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

LOGGER = logging.getLogger(__name__)


def _formula_count_frame(formulas: pd.Series) -> pd.DataFrame:
    """Match the `mendeleev_encoding` helper in single_modal/tabular.ipynb."""

    return pd.DataFrame(formulas.apply(lambda formula: parse_formula(str(formula))).values.tolist()).fillna(0)


def load_formula_vocabulary(vocabulary_csv: Path, formula_col: str) -> pd.Series:
    """Load the fixed formula pool used to define element-count columns.

    The original tabular notebook defines the element vocabulary from
    ``df["formula_discharge"]`` and then reindexes train/validation/test
    encodings to that same column set. This helper makes that choice explicit
    for reproducible publication reruns.
    """

    frame = pd.read_csv(vocabulary_csv)
    if formula_col not in frame.columns:
        raise ValueError(f"{vocabulary_csv} is missing formula column: {formula_col}")
    return frame[formula_col].dropna().astype(str).reset_index(drop=True)


def build_composition_descriptor_frame(
    frame: pd.DataFrame,
    formula_col: str = "formula",
    sample_id_col: str = "sample_id",
    base_formulas: list[str] | pd.Series | None = None,
    fill_missing: bool = True,
) -> pd.DataFrame:
    """Build the original notebook tabular representation.

    The original notebook calls this a "mendeleev" encoding, but it is simply
    a raw element-count matrix from ``chemparse.parse_formula``. Columns are
    aligned to the element vocabulary from the provided base formula pool.
    """

    required = {sample_id_col, formula_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing descriptor input columns: {sorted(missing)}")

    formula_series = pd.Series(frame[formula_col].astype(str).to_list())
    base_series = (
        pd.Series(base_formulas).astype(str)
        if base_formulas is not None
        else formula_series
    )
    base = _formula_count_frame(base_series)
    descriptors = _formula_count_frame(formula_series)
    descriptors = descriptors.reindex(columns=base.columns, fill_value=0)
    descriptors.insert(0, "sample_id", frame[sample_id_col].astype(str).to_list())
    descriptors = descriptors.replace([np.inf, -np.inf], np.nan)
    if fill_missing:
        feature_cols = [column for column in descriptors.columns if column != "sample_id"]
        descriptors[feature_cols] = descriptors[feature_cols].fillna(0.0)
    return descriptors


def classical_model_factories(
    random_state: int,
    include_xgboost: bool = True,
    n_estimators: int = 500,
    n_jobs: int = -1,
) -> dict[str, Callable[[], Pipeline]]:
    factories: dict[str, Callable[[], Pipeline]] = {
        "random_forest": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=n_estimators,
                        random_state=random_state,
                        n_jobs=n_jobs,
                        min_samples_leaf=1,
                    ),
                ),
            ]
        ),
    }
    if include_xgboost:
        try:
            from xgboost import XGBRegressor
        except Exception:
            LOGGER.warning("xgboost is not installed; skipping xgboost baseline")
        else:
            factories["xgboost"] = lambda: Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=n_estimators,
                            learning_rate=0.05,
                            max_depth=6,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective="reg:squarederror",
                            random_state=random_state,
                            n_jobs=n_jobs,
                        ),
                    ),
                ]
            )
    return factories


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


def _feature_matrix(frame: pd.DataFrame, base_formulas: pd.Series) -> tuple[np.ndarray, list[str]]:
    descriptors = build_composition_descriptor_frame(
        frame,
        base_formulas=base_formulas,
        fill_missing=False,
    )
    feature_cols = [column for column in descriptors.columns if column != "sample_id"]
    return descriptors[feature_cols].to_numpy(dtype=float), feature_cols


def _prediction_frame(
    split_frame: pd.DataFrame,
    y_pred: np.ndarray,
    split: str,
    model_name: str,
    seed: int,
    target_col: str,
) -> pd.DataFrame:
    output = split_frame[["sample_id", "formula", "working_ion", "anion_family", "target"]].copy()
    output = output.rename(columns={"target": "y_true"})
    output["y_pred"] = y_pred
    output["split"] = split
    output["model_name"] = model_name
    output["modality_set"] = "composition_counts"
    output["target_col"] = target_col
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
            "target_col",
            "seed",
        ]
    ]


def _load_split_frames(split_dir: Path, seed: int) -> dict[str, pd.DataFrame]:
    seed_dir = Path(split_dir) / f"seed_{seed}"
    frames = {split: pd.read_csv(seed_dir / f"{split}.csv") for split in ["train", "val", "test"]}
    for split, frame in frames.items():
        required = {"sample_id", "formula", "working_ion", "anion_family", "target"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{seed_dir / f'{split}.csv'} missing columns: {sorted(missing)}")
    return frames


def run_classical_baselines(
    split_dir: Path,
    output_dir: Path,
    target_col: str,
    seeds: list[int],
    models: list[str] | None = None,
    predictions_root: Path = Path("results/predictions"),
    experiment_name: str = "publication_classical_random",
    include_xgboost: bool = True,
    n_estimators: int = 500,
    n_jobs: int = -1,
    vocabulary_csv: Path | None = None,
    vocabulary_formula_col: str = "formula_discharge",
    overwrite: bool = False,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "classical_baseline_metrics.csv"
    if metrics_path.exists() and not overwrite:
        raise FileExistsError(f"{metrics_path} exists; pass --overwrite to replace it")
    if metrics_path.exists():
        metrics_path.unlink()

    rows: list[dict[str, object]] = []
    selected = set(models) if models else None
    feature_columns: list[str] | None = None
    fixed_base_formulas = (
        load_formula_vocabulary(vocabulary_csv, vocabulary_formula_col)
        if vocabulary_csv is not None
        else None
    )
    for seed in seeds:
        frames = _load_split_frames(split_dir, seed)
        base_formulas = (
            fixed_base_formulas
            if fixed_base_formulas is not None
            else pd.concat(
                [frames[split]["formula"].astype(str) for split in ["train", "val", "test"]],
                ignore_index=True,
            )
        )
        x_by_split: dict[str, np.ndarray] = {}
        for split, frame in frames.items():
            x_by_split[split], feature_columns = _feature_matrix(frame, base_formulas=base_formulas)
        y_by_split = {
            split: frame["target"].to_numpy(dtype=float)
            for split, frame in frames.items()
        }
        factories = classical_model_factories(
            random_state=seed,
            include_xgboost=include_xgboost,
            n_estimators=n_estimators,
            n_jobs=n_jobs,
        )
        for model_name, factory in factories.items():
            if selected is not None and model_name not in selected:
                continue
            model = factory()
            model.fit(x_by_split["train"], y_by_split["train"])
            model_dir = output_dir / "models" / model_name / f"seed_{seed}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pkl"
            if model_path.exists() and not overwrite:
                raise FileExistsError(f"{model_path} exists; pass --overwrite to replace it")
            with open(model_path, "wb") as handle:
                pickle.dump(model, handle)

            for split in ["train", "val", "test"]:
                predictions = model.predict(x_by_split[split])
                metrics = regression_metrics_np(y_by_split[split], predictions)
                pred_frame = _prediction_frame(
                    frames[split],
                    predictions,
                    split=split,
                    model_name=model_name,
                    seed=seed,
                    target_col=target_col,
                )
                prediction_path = (
                    Path(predictions_root)
                    / experiment_name
                    / target_col
                    / model_name
                    / f"seed_{seed}_{split}_predictions.csv"
                )
                if prediction_path.exists() and not overwrite:
                    raise FileExistsError(f"{prediction_path} exists; pass --overwrite to replace it")
                prediction_path.parent.mkdir(parents=True, exist_ok=True)
                pred_frame.to_csv(prediction_path, index=False)
                rows.append(
                    {
                        "experiment_name": experiment_name,
                        "target_col": target_col,
                        "model_name": model_name,
                        "modality_set": "composition_counts",
                        "seed": seed,
                        "split": split,
                        **metrics,
                        "n_samples": int(len(frames[split])),
                        "n_train": int(len(frames["train"])),
                        "n_val": int(len(frames["val"])),
                        "n_test": int(len(frames["test"])),
                        "model_path": str(model_path),
                        "prediction_path": str(prediction_path),
                    }
                )
                LOGGER.info(
                    "%s seed %s %s MAE=%.4f R2=%.4f",
                    model_name,
                    seed,
                    split,
                    metrics["MAE"],
                    metrics["R2"],
                )

    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(metrics_path, index=False)
    (output_dir / "classical_baseline_config.json").write_text(
        json.dumps(
            {
                "split_dir": str(split_dir),
                "output_dir": str(output_dir),
                "target_col": target_col,
                "seeds": seeds,
                "models": models,
                "include_xgboost": include_xgboost,
                "n_estimators": n_estimators,
                "n_jobs": n_jobs,
                "vocabulary_csv": str(vocabulary_csv) if vocabulary_csv is not None else None,
                "vocabulary_formula_col": vocabulary_formula_col,
                "predictions_root": str(predictions_root),
                "experiment_name": experiment_name,
                "feature_count": len(feature_columns or []),
                "feature_columns": feature_columns or [],
                "descriptor_source": "single_modal/tabular.ipynb chemparse element-count encoding",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return metrics_frame


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical composition baselines.")
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--predictions_root", type=Path, default=Path("results/predictions"))
    parser.add_argument("--experiment_name", default="publication_classical_random")
    parser.add_argument("--include_xgboost", action="store_true")
    parser.add_argument("--n_estimators", type=int, default=500)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument(
        "--vocabulary_csv",
        type=Path,
        default=None,
        help="Optional CSV whose formula column defines the fixed element-count vocabulary.",
    )
    parser.add_argument("--vocabulary_formula_col", default="formula_discharge")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    run_classical_baselines(
        split_dir=args.split_dir,
        output_dir=args.output_dir,
        target_col=args.target_col,
        seeds=args.seeds,
        models=args.models,
        predictions_root=args.predictions_root,
        experiment_name=args.experiment_name,
        include_xgboost=args.include_xgboost,
        n_estimators=args.n_estimators,
        n_jobs=args.n_jobs,
        vocabulary_csv=args.vocabulary_csv,
        vocabulary_formula_col=args.vocabulary_formula_col,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
