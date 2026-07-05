from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from battery_fusion.utils.chemistry_groups import (
    ANION_FAMILIES,
    assign_chemistry_groups,
)

LOGGER = logging.getLogger(__name__)


def choose_default_heldout_family(
    assignments: pd.DataFrame,
    min_test_samples: int = 100,
) -> str:
    counts = assignments["anion_family"].value_counts()
    if counts.get("halide", 0) >= min_test_samples:
        return "halide"
    if counts.get("phosphate_or_polyanion", 0) >= min_test_samples:
        return "phosphate_or_polyanion"
    return str(counts.index[0])


def create_anion_holdout_splits(
    assignments: pd.DataFrame,
    heldout_family: str,
    output_dir: Path,
    seeds: list[int],
    min_test_samples: int = 100,
    val_ratio: float = 0.1,
    allow_small_holdout: bool = False,
    overwrite: bool = False,
) -> list[dict[str, Path]]:
    if heldout_family not in ANION_FAMILIES:
        raise ValueError(f"Unknown anion family {heldout_family!r}")
    required = {"sample_id", "anion_family"}
    missing = required.difference(assignments.columns)
    if missing:
        raise ValueError(f"Assignments are missing required columns: {sorted(missing)}")

    assignments = assignments.copy()
    assignments["sample_id"] = assignments["sample_id"].astype(str)
    test_rows = assignments[assignments["anion_family"] == heldout_family].copy()
    train_side = assignments[assignments["anion_family"] != heldout_family].copy()
    if len(test_rows) < min_test_samples and not allow_small_holdout:
        raise ValueError(
            f"Held-out family {heldout_family!r} has {len(test_rows)} samples, "
            f"below --min_test_samples {min_test_samples}. Pass "
            "--allow_small_holdout to create the split anyway."
        )
    if train_side.empty:
        raise ValueError("No training-side samples remain after holdout")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Path]] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        shuffled = train_side.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
        n_val = int(round(len(shuffled) * val_ratio))
        if len(shuffled) > 1:
            n_val = min(max(n_val, 1), len(shuffled) - 1)
        val_rows = shuffled.iloc[:n_val].sort_values("sample_id")
        train_rows = shuffled.iloc[n_val:].sort_values("sample_id")
        test_out = test_rows.sort_values("sample_id")

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "train": seed_dir / "train.csv",
            "val": seed_dir / "val.csv",
            "test": seed_dir / "test.csv",
        }
        for split_name, split_frame in [
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_out),
        ]:
            path = paths[split_name]
            if path.exists() and not overwrite:
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
            split_frame.to_csv(path, index=False)

        config = {
            "heldout_family": heldout_family,
            "seed": seed,
            "val_ratio": val_ratio,
            "min_test_samples": min_test_samples,
            "allow_small_holdout": allow_small_holdout,
            "n_train": int(len(train_rows)),
            "n_val": int(len(val_rows)),
            "n_test": int(len(test_out)),
        }
        config_path = seed_dir / "split_config.json"
        if config_path.exists() and not overwrite:
            raise FileExistsError(f"{config_path} exists; pass --overwrite to replace it")
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        LOGGER.info(
            "Created %s seed %s split: train=%s val=%s test=%s",
            heldout_family,
            seed,
            len(train_rows),
            len(val_rows),
            len(test_out),
        )
        results.append(paths)
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create anion-family holdout splits.")
    parser.add_argument("--input_data", type=Path, required=True)
    parser.add_argument("--sample_id_col", default="id_discharge")
    parser.add_argument("--formula_col", default="formula_discharge")
    parser.add_argument("--target_col", default="target")
    parser.add_argument("--working_ion_col", default="working_ion")
    parser.add_argument("--grouping", choices=["anion_family"], default="anion_family")
    parser.add_argument("--heldout_family", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--assignment_output", type=Path, default=Path("results/chemistry_groups/anion_family_assignments.csv"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--min_test_samples", type=int, default=100)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--allow_small_holdout", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = _parse_args()
    frame = pd.read_csv(args.input_data)
    if frame[args.sample_id_col].duplicated().any():
        LOGGER.info("Duplicate sample ids found; keeping the last row per sample id")
        frame = frame.groupby(args.sample_id_col, as_index=False).tail(1)
    assignments = assign_chemistry_groups(
        frame,
        sample_id_col=args.sample_id_col,
        formula_col=args.formula_col,
        target_col=args.target_col,
        working_ion_col=args.working_ion_col,
    )
    args.assignment_output.parent.mkdir(parents=True, exist_ok=True)
    if args.assignment_output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.assignment_output} exists; pass --overwrite to replace it"
        )
    assignments.to_csv(args.assignment_output, index=False)

    family_counts = assignments["anion_family"].value_counts().to_dict()
    working_ion_counts = assignments["working_ion"].value_counts().to_dict()
    LOGGER.info("Anion-family counts: %s", family_counts)
    LOGGER.info("Working-ion counts: %s", working_ion_counts)

    heldout_family = args.heldout_family or choose_default_heldout_family(
        assignments, min_test_samples=args.min_test_samples
    )
    LOGGER.info("Held-out family: %s", heldout_family)
    create_anion_holdout_splits(
        assignments,
        heldout_family=heldout_family,
        output_dir=args.output_dir,
        seeds=args.seeds,
        min_test_samples=args.min_test_samples,
        val_ratio=args.val_ratio,
        allow_small_holdout=args.allow_small_holdout,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
