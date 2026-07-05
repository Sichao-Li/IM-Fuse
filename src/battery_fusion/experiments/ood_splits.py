from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from battery_fusion.features.tabular import formula_vector, vocabulary_from_formulas
from battery_fusion.utils.chemistry_groups import assign_chemistry_groups, normalize_working_ion


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompositionClusterSplitResult:
    created: list[dict[str, Path]]
    skipped: list[dict[str, int]]


def _validate_split_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"sample_id", "formula", "working_ion", "anion_family", "target"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Split frame is missing required columns: {sorted(missing)}")
    output = frame.copy()
    output["sample_id"] = output["sample_id"].astype(str)
    output["formula"] = output["formula"].astype(str)
    output["working_ion"] = output["working_ion"].map(normalize_working_ion)
    return output


def _load_split_frame(
    input_data: Path,
    sample_id_col: str,
    formula_col: str,
    target_col: str,
    working_ion_col: str | None,
) -> pd.DataFrame:
    frame = pd.read_csv(input_data)
    if frame[sample_id_col].duplicated().any():
        LOGGER.info("Duplicate sample ids found; keeping the last row per sample id")
        frame = frame.groupby(sample_id_col, as_index=False).tail(1)
    return assign_chemistry_groups(
        frame,
        sample_id_col=sample_id_col,
        formula_col=formula_col,
        target_col=target_col,
        working_ion_col=working_ion_col,
    )


def _split_train_val(
    train_side: pd.DataFrame,
    seed: int,
    val_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_side.empty:
        raise ValueError("No training-side samples remain after holdout")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    rng = np.random.default_rng(seed)
    shuffled = train_side.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    n_val = int(round(len(shuffled) * val_ratio))
    if len(shuffled) > 1:
        n_val = min(max(n_val, 1), len(shuffled) - 1)
    val_rows = shuffled.iloc[:n_val].sort_values("sample_id")
    train_rows = shuffled.iloc[n_val:].sort_values("sample_id")
    return train_rows, val_rows


def _write_split(
    seed_dir: Path,
    train_rows: pd.DataFrame,
    val_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    config: dict,
    overwrite: bool,
) -> dict[str, Path]:
    seed_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": seed_dir / "train.csv",
        "val": seed_dir / "val.csv",
        "test": seed_dir / "test.csv",
    }
    for split_name, split_frame in [
        ("train", train_rows),
        ("val", val_rows),
        ("test", test_rows),
    ]:
        path = paths[split_name]
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        split_frame.to_csv(path, index=False)

    config_path = seed_dir / "split_config.json"
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"{config_path} exists; pass --overwrite to replace it")
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return paths


def assign_composition_clusters(
    frame: pd.DataFrame,
    n_clusters: int = 3,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign KMeans clusters from formula-derived composition counts.

    Clustering uses only composition vectors derived from the formula column.
    The target column is intentionally ignored so the resulting holdout is a
    data-driven chemistry split, not a target-stratified split.
    """

    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    frame = _validate_split_frame(frame)
    if len(frame) < n_clusters:
        raise ValueError(f"Cannot fit {n_clusters} clusters with only {len(frame)} samples")

    vocabulary = vocabulary_from_formulas(frame["formula"].tolist())
    if not vocabulary:
        raise ValueError("No elements could be parsed from the formula column")
    features = np.vstack(
        [formula_vector(formula, vocabulary) for formula in frame["formula"].tolist()]
    )
    scaled = StandardScaler().fit_transform(features)
    labels = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit_predict(scaled)

    output = frame.copy()
    output["composition_cluster"] = labels.astype(int)
    return output


def create_composition_cluster_holdout_splits(
    frame: pd.DataFrame,
    output_dir: Path,
    seeds: list[int],
    n_clusters: int = 3,
    cluster_seed: int = 0,
    min_test_size: int = 50,
    val_ratio: float = 0.1,
    overwrite: bool = False,
) -> CompositionClusterSplitResult:
    frame = _validate_split_frame(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[dict[str, Path]] = []
    skipped: list[dict[str, int]] = []
    clustered = assign_composition_clusters(frame, n_clusters=n_clusters, seed=cluster_seed)
    assignment_path = output_dir / "composition_cluster_assignments.csv"
    if assignment_path.exists() and not overwrite:
        raise FileExistsError(f"{assignment_path} exists; pass --overwrite to replace it")
    clustered.sort_values("sample_id").to_csv(assignment_path, index=False)
    cluster_counts = clustered["composition_cluster"].value_counts().sort_index()

    for seed in seeds:
        for cluster_id, cluster_size in cluster_counts.items():
            cluster_id = int(cluster_id)
            cluster_size = int(cluster_size)
            if cluster_size < min_test_size:
                LOGGER.warning(
                    "Skipping composition cluster %s for seed %s: %s samples < min_test_size %s",
                    cluster_id,
                    seed,
                    cluster_size,
                    min_test_size,
                )
                skipped.append(
                    {
                        "seed": int(seed),
                        "cluster_seed": int(cluster_seed),
                        "composition_cluster": cluster_id,
                        "n_test": cluster_size,
                    }
                )
                continue

            test_rows = clustered[clustered["composition_cluster"] == cluster_id].sort_values(
                "sample_id"
            )
            train_side = clustered[clustered["composition_cluster"] != cluster_id]
            train_rows, val_rows = _split_train_val(train_side, seed=seed, val_ratio=val_ratio)
            seed_dir = output_dir / f"cluster_{cluster_id}" / f"seed_{seed}"
            paths = _write_split(
                seed_dir,
                train_rows=train_rows,
                val_rows=val_rows,
                test_rows=test_rows,
                config={
                    "ood_protocol": "composition_cluster_holdout",
                    "seed": int(seed),
                    "cluster_seed": int(cluster_seed),
                    "n_clusters": int(n_clusters),
                    "heldout_cluster": cluster_id,
                    "min_test_size": int(min_test_size),
                    "val_ratio": float(val_ratio),
                    "n_train": int(len(train_rows)),
                    "n_val": int(len(val_rows)),
                    "n_test": int(len(test_rows)),
                    "assignment_path": str(assignment_path),
                },
                overwrite=overwrite,
            )
            created.append(paths)
            LOGGER.info(
                "Created composition-cluster split seed=%s cluster=%s train=%s val=%s test=%s",
                seed,
                cluster_id,
                len(train_rows),
                len(val_rows),
                len(test_rows),
            )

    summary = pd.DataFrame(
        [
            {
                "seed": int(path["test"].parent.name.replace("seed_", "")),
                "composition_cluster": int(path["test"].parent.parent.name.replace("cluster_", "")),
                "split_dir": str(path["test"].parent.parent),
            }
            for path in created
        ]
    )
    if not summary.empty:
        summary_path = output_dir / "created_splits.csv"
        if summary_path.exists() and not overwrite:
            raise FileExistsError(f"{summary_path} exists; pass --overwrite to replace it")
        summary.to_csv(summary_path, index=False)
    if skipped:
        skipped_path = output_dir / "skipped_clusters.csv"
        if skipped_path.exists() and not overwrite:
            raise FileExistsError(f"{skipped_path} exists; pass --overwrite to replace it")
        pd.DataFrame(skipped).to_csv(skipped_path, index=False)

    config_path = output_dir / "composition_cluster_holdout_config.json"
    if config_path.exists() and not overwrite:
        raise FileExistsError(f"{config_path} exists; pass --overwrite to replace it")
    config_path.write_text(
        json.dumps(
            {
                "ood_protocol": "composition_cluster_holdout",
                "seeds": [int(seed) for seed in seeds],
                "cluster_seed": int(cluster_seed),
                "n_clusters": int(n_clusters),
                "min_test_size": int(min_test_size),
                "val_ratio": float(val_ratio),
                "n_created": len(created),
                "n_skipped": len(skipped),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return CompositionClusterSplitResult(created=created, skipped=skipped)


def create_working_ion_holdout_splits(
    frame: pd.DataFrame,
    heldout_ions: Iterable[str],
    output_dir: Path,
    seeds: list[int],
    min_test_size: int = 50,
    val_ratio: float = 0.1,
    overwrite: bool = False,
) -> list[dict[str, Path]]:
    frame = _validate_split_frame(frame)
    heldout = tuple(normalize_working_ion(ion) for ion in heldout_ions)
    heldout = tuple(ion for ion in heldout if ion != "other")
    if not heldout:
        raise ValueError("heldout_ions must include at least one explicit ion")

    test_mask = frame["working_ion"].isin(heldout)
    test_rows = frame[test_mask].sort_values("sample_id")
    train_side = frame[~test_mask]
    if len(test_rows) < min_test_size:
        raise ValueError(
            f"Working-ion holdout {heldout} has {len(test_rows)} samples, "
            f"below min_test_size {min_test_size}"
        )
    if train_side.empty:
        raise ValueError("No training-side samples remain after working-ion holdout")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = output_dir / "working_ion_holdout_assignments.csv"
    if assignment_path.exists() and not overwrite:
        raise FileExistsError(f"{assignment_path} exists; pass --overwrite to replace it")
    assignments = frame.copy()
    assignments["working_ion_holdout"] = np.where(test_mask, "test", "train_val")
    assignments.sort_values("sample_id").to_csv(assignment_path, index=False)

    created: list[dict[str, Path]] = []
    for seed in seeds:
        train_rows, val_rows = _split_train_val(train_side, seed=seed, val_ratio=val_ratio)
        seed_dir = output_dir / f"seed_{seed}"
        paths = _write_split(
            seed_dir,
            train_rows=train_rows,
            val_rows=val_rows,
            test_rows=test_rows,
            config={
                "ood_protocol": "working_ion_holdout",
                "seed": int(seed),
                "heldout_ions": list(heldout),
                "min_test_size": int(min_test_size),
                "val_ratio": float(val_ratio),
                "n_train": int(len(train_rows)),
                "n_val": int(len(val_rows)),
                "n_test": int(len(test_rows)),
                "assignment_path": str(assignment_path),
            },
            overwrite=overwrite,
        )
        created.append(paths)
        LOGGER.info(
            "Created working-ion split heldout=%s seed=%s train=%s val=%s test=%s",
            ",".join(heldout),
            seed,
            len(train_rows),
            len(val_rows),
            len(test_rows),
        )
    return created


def _base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input_data", type=Path, required=True)
    parser.add_argument("--sample_id_col", default="id_discharge")
    parser.add_argument("--formula_col", default="formula_discharge")
    parser.add_argument("--target_col", choices=["average_voltage", "capacity_vol"], required=True)
    parser.add_argument("--working_ion_col", default="working_ion")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--min_test_size", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main_composition_cluster(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = _base_parser("Create composition-cluster OOD holdout splits.")
    parser.add_argument("--n_clusters", type=int, default=3)
    parser.add_argument("--cluster_seed", type=int, default=0)
    args = parser.parse_args(argv)
    frame = _load_split_frame(
        input_data=args.input_data,
        sample_id_col=args.sample_id_col,
        formula_col=args.formula_col,
        target_col=args.target_col,
        working_ion_col=args.working_ion_col,
    )
    result = create_composition_cluster_holdout_splits(
        frame,
        output_dir=args.output_dir,
        seeds=args.seeds,
        n_clusters=args.n_clusters,
        cluster_seed=args.cluster_seed,
        min_test_size=args.min_test_size,
        val_ratio=args.val_ratio,
        overwrite=args.overwrite,
    )
    LOGGER.info("Created %s splits; skipped %s clusters", len(result.created), len(result.skipped))


def main_working_ion_holdout(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    parser = _base_parser("Create leave-one-working-ion-out OOD holdout splits.")
    parser.add_argument("--heldout_ions", nargs="+", required=True)
    args = parser.parse_args(argv)
    frame = _load_split_frame(
        input_data=args.input_data,
        sample_id_col=args.sample_id_col,
        formula_col=args.formula_col,
        target_col=args.target_col,
        working_ion_col=args.working_ion_col,
    )
    create_working_ion_holdout_splits(
        frame,
        heldout_ions=args.heldout_ions,
        output_dir=args.output_dir,
        seeds=args.seeds,
        min_test_size=args.min_test_size,
        val_ratio=args.val_ratio,
        overwrite=args.overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create manuscript OOD splits: composition-cluster or working-ion holdout."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("composition-cluster", help="Hold out one KMeans composition cluster at a time.")
    subparsers.add_parser("working-ion", help="Hold out one or more working-ion groups.")
    args, rest = parser.parse_known_args()
    if args.command == "composition-cluster":
        main_composition_cluster(rest)
    elif args.command == "working-ion":
        main_working_ion_holdout(rest)


if __name__ == "__main__":
    main()
