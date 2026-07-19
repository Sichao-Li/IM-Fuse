from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from battery_fusion import __version__


PUBLICATION_TARGETS = ("average_voltage", "capacity_vol")
PUBLICATION_SEEDS = tuple(range(5))
PUBLICATION_SOURCE_ROW_COUNT = 10114
PUBLICATION_SAMPLE_COUNT = 8088
REQUIRED_SPLIT_COLUMNS = {"sample_id", "formula", "working_ion", "anion_family", "target"}
CORE_PACKAGES = (
    "chemparse",
    "matplotlib",
    "numpy",
    "pandas",
    "pymatgen",
    "rdfpy",
    "scikit-learn",
    "torch",
)
OPTIONAL_PACKAGES = (
    "xgboost",
    "generalized-rashomon-set",
    "mp-api",
    "alignn",
    "dgl",
)


@dataclass
class ReleaseCheckReport:
    package_version: str
    python_version: str
    dependencies: dict[str, str]
    devices: dict[str, bool]
    split_summary: dict[str, dict[str, int]] = field(default_factory=dict)
    artifact_summary: dict[str, int | bool] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_data_paths(root: Path) -> list[Path]:
    paths = [root / "data" / "sample_order" / "sample_order_keep_last.csv"]
    paths.extend(sorted((root / "data" / "splits" / "publication").glob("**/*.csv")))
    paths.extend(sorted((root / "data" / "splits" / "publication").glob("**/*.json")))
    return sorted(paths)


def validate_checksums(root: Path, report: ReleaseCheckReport) -> None:
    manifest_path = root / "data" / "checksums.sha256"
    if not manifest_path.exists():
        report.errors.append("Missing data/checksums.sha256")
        return

    entries: dict[str, str] = {}
    for line_number, raw_line in enumerate(manifest_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            report.errors.append(f"Malformed checksum line {line_number}: {raw_line!r}")
            continue
        expected, relative = parts
        entries[relative.lstrip("*")] = expected

    expected_paths = {
        path.relative_to(root).as_posix()
        for path in _release_data_paths(root)
        if path.exists()
    }
    if set(entries) != expected_paths:
        missing = sorted(expected_paths.difference(entries))
        extra = sorted(set(entries).difference(expected_paths))
        if missing:
            report.errors.append(f"Checksum manifest is missing: {', '.join(missing)}")
        if extra:
            report.errors.append(f"Checksum manifest has unexpected entries: {', '.join(extra)}")

    for relative, expected in entries.items():
        path = root / relative
        if not path.exists():
            report.errors.append(f"Checksum target is missing: {relative}")
        elif _sha256(path) != expected:
            report.errors.append(f"Checksum mismatch: {relative}")


def validate_publication_splits(root: Path, report: ReleaseCheckReport) -> None:
    order_path = root / "data" / "sample_order" / "sample_order_keep_last.csv"
    if not order_path.exists():
        report.errors.append(f"Missing {order_path.relative_to(root)}")
        return
    order_frame = pd.read_csv(order_path)
    if "id_discharge" not in order_frame.columns:
        report.errors.append("Sample-order table is missing id_discharge")
        return
    sample_order = order_frame["id_discharge"].astype(str).tolist()
    if len(sample_order) != PUBLICATION_SAMPLE_COUNT:
        report.errors.append(
            f"Expected {PUBLICATION_SAMPLE_COUNT} model-ready IDs, found {len(sample_order)}"
        )
    if len(sample_order) != len(set(sample_order)):
        report.errors.append("Sample-order table contains duplicate IDs")

    reference_membership: dict[tuple[int, str], list[str]] = {}
    reference_metadata: pd.DataFrame | None = None
    for target in PUBLICATION_TARGETS:
        target_root = root / "data" / "splits" / "publication" / target
        seed_pools: list[set[str]] = []
        for seed in PUBLICATION_SEEDS:
            seed_dir = target_root / f"seed_{seed}"
            frames: dict[str, pd.DataFrame] = {}
            for split in ("train", "val", "test"):
                path = seed_dir / f"{split}.csv"
                if not path.exists():
                    report.errors.append(f"Missing {path.relative_to(root)}")
                    continue
                frame = pd.read_csv(path)
                missing_columns = REQUIRED_SPLIT_COLUMNS.difference(frame.columns)
                if missing_columns:
                    report.errors.append(
                        f"{path.relative_to(root)} is missing columns {sorted(missing_columns)}"
                    )
                    continue
                if frame["sample_id"].astype(str).duplicated().any():
                    report.errors.append(f"Duplicate sample IDs in {path.relative_to(root)}")
                if not np.isfinite(pd.to_numeric(frame["target"], errors="coerce")).all():
                    report.errors.append(f"Non-finite targets in {path.relative_to(root)}")
                frames[split] = frame
            if len(frames) != 3:
                continue

            ids = {
                split: frame["sample_id"].astype(str).tolist()
                for split, frame in frames.items()
            }
            id_sets = {split: set(values) for split, values in ids.items()}
            if id_sets["train"] & id_sets["val"] or id_sets["train"] & id_sets["test"] or id_sets["val"] & id_sets["test"]:
                report.errors.append(f"Split overlap for {target}, seed {seed}")
            pool = set().union(*id_sets.values())
            seed_pools.append(pool)
            if pool != set(sample_order):
                report.errors.append(f"Split pool differs from sample order for {target}, seed {seed}")

            shuffled = sample_order[:]
            random.Random(seed).shuffle(shuffled)
            train_end = int(PUBLICATION_SAMPLE_COUNT * 0.8)
            val_end = train_end + int(PUBLICATION_SAMPLE_COUNT * 0.1)
            expected_ids = {
                "train": shuffled[:train_end],
                "val": shuffled[train_end:val_end],
                "test": shuffled[val_end:],
            }
            for split in ("train", "val", "test"):
                if ids[split] != expected_ids[split]:
                    report.errors.append(
                        f"Non-deterministic split order for {target}, seed {seed}, {split}"
                    )
                key = (seed, split)
                if target == PUBLICATION_TARGETS[0]:
                    reference_membership[key] = ids[split]
                elif reference_membership.get(key) != ids[split]:
                    report.errors.append(
                        f"Target split membership differs for seed {seed}, {split}"
                    )

            config_path = seed_dir / "split_config.json"
            if not config_path.exists():
                report.errors.append(f"Missing {config_path.relative_to(root)}")
            else:
                config = json.loads(config_path.read_text())
                expected_config = {
                    "seed": seed,
                    "target_col": target,
                    "n_samples": PUBLICATION_SAMPLE_COUNT,
                    "n_train": len(ids["train"]),
                    "n_val": len(ids["val"]),
                    "n_test": len(ids["test"]),
                }
                for key, expected in expected_config.items():
                    if config.get(key) != expected:
                        report.errors.append(
                            f"Incorrect {key} in {config_path.relative_to(root)}: "
                            f"expected {expected!r}, found {config.get(key)!r}"
                        )

            combined = pd.concat(frames.values(), ignore_index=True)
            metadata = combined.set_index(combined["sample_id"].astype(str))[
                ["formula", "working_ion", "anion_family"]
            ].sort_index()
            if seed == 0 and target == PUBLICATION_TARGETS[0]:
                reference_metadata = metadata
            elif seed == 0 and reference_metadata is not None and not metadata.equals(reference_metadata):
                report.errors.append("Formula or chemistry metadata differs between targets")

        if seed_pools and any(pool != seed_pools[0] for pool in seed_pools[1:]):
            report.errors.append(f"Sample pool differs across seeds for {target}")
        if seed_pools:
            report.split_summary[target] = {
                "n_samples": len(seed_pools[0]),
                "n_seeds": len(seed_pools),
                "n_train": int(PUBLICATION_SAMPLE_COUNT * 0.8),
                "n_val": int(PUBLICATION_SAMPLE_COUNT * 0.1),
                "n_test": PUBLICATION_SAMPLE_COUNT
                - int(PUBLICATION_SAMPLE_COUNT * 0.8)
                - int(PUBLICATION_SAMPLE_COUNT * 0.1),
            }


def inspect_artifacts(root: Path, report: ReleaseCheckReport, strict: bool) -> None:
    raw_data = root / "data" / "raw" / "mp_total.csv"
    atom_init = root / "data" / "raw" / "atom_init.json"
    cif_count = len(list((root / "data" / "raw" / "cifs").glob("*.cif")))
    processed_root = root / "data" / "processed" / "publication"
    modality_counts = {
        modality: len({path.stem for path in (processed_root / modality).glob("*/*.pt")})
        for modality in ("tabular", "rdf", "structure")
    }
    source_row_count: int | None = None
    source_unique_id_count: int | None = None
    source_problem: str | None = None
    if raw_data.exists():
        try:
            source_ids = pd.read_csv(raw_data, usecols=["id_discharge"])["id_discharge"]
            source_row_count = int(len(source_ids))
            source_unique_id_count = int(source_ids.astype(str).nunique())
            if source_row_count != PUBLICATION_SOURCE_ROW_COUNT:
                source_problem = (
                    f"data/raw/mp_total.csv has {source_row_count} rows; "
                    f"the cleaned publication source table has {PUBLICATION_SOURCE_ROW_COUNT}"
                )
            elif source_unique_id_count != PUBLICATION_SAMPLE_COUNT:
                source_problem = (
                    f"data/raw/mp_total.csv has {source_unique_id_count} unique discharge IDs; "
                    f"expected {PUBLICATION_SAMPLE_COUNT}"
                )
        except (ValueError, KeyError) as exc:
            source_problem = f"Could not validate data/raw/mp_total.csv: {exc}"

    report.artifact_summary = {
        "raw_mp_total": raw_data.exists(),
        "source_row_count": source_row_count if source_row_count is not None else 0,
        "source_unique_id_count": (
            source_unique_id_count if source_unique_id_count is not None else 0
        ),
        "atom_init": atom_init.exists(),
        "cif_count": cif_count,
        **{f"{modality}_count": count for modality, count in modality_counts.items()},
    }
    missing = []
    if not raw_data.exists():
        missing.append("data/raw/mp_total.csv")
    if not atom_init.exists():
        missing.append("data/raw/atom_init.json")
    if cif_count != PUBLICATION_SAMPLE_COUNT:
        missing.append(f"{PUBLICATION_SAMPLE_COUNT} CIFs (found {cif_count})")
    for modality, count in modality_counts.items():
        if count != PUBLICATION_SAMPLE_COUNT:
            missing.append(f"{PUBLICATION_SAMPLE_COUNT} {modality} caches (found {count})")
    if missing:
        message = "External full-rerun artifacts are incomplete: " + "; ".join(missing)
        (report.errors if strict else report.warnings).append(message)
    if source_problem:
        (report.errors if strict else report.warnings).append(source_problem)


def run_release_check(root: Path, strict_artifacts: bool = False) -> ReleaseCheckReport:
    root = Path(root).resolve()
    dependencies = {
        name: _distribution_version(name)
        for name in (*CORE_PACKAGES, *OPTIONAL_PACKAGES)
    }
    devices = {"mps": False, "cuda": False}
    try:
        import torch

        devices["mps"] = bool(torch.backends.mps.is_available())
        devices["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        pass
    report = ReleaseCheckReport(
        package_version=__version__,
        python_version=platform.python_version(),
        dependencies=dependencies,
        devices=devices,
    )
    if __version__ != "1.0.0":
        report.errors.append(f"Expected package version 1.0.0, found {__version__}")
    validate_publication_splits(root, report)
    validate_checksums(root, report)
    inspect_artifacts(root, report, strict=strict_artifacts)
    return report


def _print_report(report: ReleaseCheckReport) -> None:
    print(f"IM-Fuse {report.package_version}")
    print(f"Python {report.python_version}")
    print(
        "Devices: "
        + ", ".join(f"{name}={'available' if value else 'unavailable'}" for name, value in report.devices.items())
    )
    print("Dependencies:")
    for name, version in report.dependencies.items():
        print(f"  {name}: {version}")
    print("Publication splits:")
    for target, summary in report.split_summary.items():
        print(
            f"  {target}: {summary['n_samples']} samples, {summary['n_seeds']} seeds, "
            f"{summary['n_train']}/{summary['n_val']}/{summary['n_test']} train/val/test"
        )
    print("Artifacts:")
    for name, value in report.artifact_summary.items():
        print(f"  {name}: {value}")
    for warning in report.warnings:
        print(f"WARNING: {warning}")
    for error in report.errors:
        print(f"ERROR: {error}")
    print(f"Release check: {'PASS' if report.ok else 'FAIL'}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the IM-Fuse environment and tracked publication data contract."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--strict-artifacts",
        action="store_true",
        help="Fail when externally hosted raw data or processed caches are absent.",
    )
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    report = run_release_check(args.root, strict_artifacts=args.strict_artifacts)
    _print_report(report)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps({**asdict(report), "ok": report.ok}, indent=2) + "\n")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
