from __future__ import annotations

import json
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from battery_fusion.data.preprocess import preprocess_modalities
from battery_fusion.data.splits import create_split_manifest
from battery_fusion.paths import ProjectPaths
from battery_fusion.utils.hash import file_sha256


@dataclass(frozen=True)
class DataFoundationResult:
    """Paths and counts produced by the public data-preparation command."""

    mp_total_path: str
    labels_path: str
    cif_coverage_path: str
    manifest_path: str
    split_paths: list[str]
    processed_roots: list[str]
    n_source_rows: int
    n_labels: int
    n_duplicate_rows_removed: int
    n_missing_target_rows_removed: int
    n_cifs_found: int
    n_cifs_missing: int


def download_file(url: str, output_path: Path, overwrite: bool = False) -> Path:
    """Download a file from an explicit URL.

    This helper is intentionally small: public users can point it at a GitHub
    release asset, Zenodo archive, or institutional mirror without hardcoding a
    data host into the package.
    """

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, output_path)
    return output_path


def build_label_table_from_mp_total(
    mp_total_path: Path,
    output_path: Path,
    target_col: str,
    id_col: str = "id_discharge",
    formula_col: str = "formula_discharge",
    working_ion_col: str | None = "working_ion",
    extra_metadata_cols: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build the canonical IM-Fuse label table from ``mp_total.csv``.

    The source MP table can contain multiple rows per discharge ID. For
    reproducibility and consistency with the manuscript pipeline, the public
    preparation path keeps the last row for each ID and writes standardized
    columns consumed by the rest of the package.
    """

    mp_total_path = Path(mp_total_path)
    output_path = Path(output_path)
    frame = pd.read_csv(mp_total_path)
    required = {id_col, target_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in {mp_total_path}: {sorted(missing)}")
    if formula_col not in frame.columns:
        raise ValueError(
            f"Missing formula column {formula_col!r} in {mp_total_path}. "
            "Composition features require discharge formulas."
        )

    n_source_rows = len(frame)
    deduped = frame.drop_duplicates(subset=id_col, keep="last").copy()
    target_values = pd.to_numeric(deduped[target_col], errors="coerce")
    valid_target = target_values.notna()
    n_missing_target = int((~valid_target).sum())
    deduped = deduped.loc[valid_target].copy()
    target_values = target_values.loc[valid_target]

    labels = pd.DataFrame(
        {
            "id_discharge": deduped[id_col].astype(str),
            "target": target_values.astype(float),
            "formula_discharge": deduped[formula_col].astype(str),
        }
    )
    if working_ion_col and working_ion_col in deduped.columns:
        labels["working_ion"] = deduped[working_ion_col].astype(str)
    for column in extra_metadata_cols:
        if column in deduped.columns and column not in labels.columns:
            labels[column] = deduped[column].astype(str)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_path, index=False)
    n_unique_ids = len(frame.drop_duplicates(subset=id_col, keep="last"))
    stats = {
        "n_source_rows": int(n_source_rows),
        "n_labels": int(len(labels)),
        "n_duplicate_rows_removed": int(n_source_rows - n_unique_ids),
        "n_missing_target_rows_removed": n_missing_target,
    }
    return labels, stats


def copy_matching_cifs(
    source_cif_dir: Path,
    sample_ids: list[str],
    target_cif_dir: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Copy CIF files whose stem matches the canonical sample ID."""

    source_cif_dir = Path(source_cif_dir)
    target_cif_dir = Path(target_cif_dir)
    if not source_cif_dir.exists():
        raise FileNotFoundError(source_cif_dir)
    target_cif_dir.mkdir(parents=True, exist_ok=True)

    available = _cif_stem_map(source_cif_dir)
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        source = available.get(str(sample_id).lower())
        target = target_cif_dir / f"{sample_id}.cif"
        if source is not None and (overwrite or not target.exists()):
            if source.resolve() != target.resolve():
                shutil.copy2(source, target)
        rows.append(
            {
                "sample_id": sample_id,
                "source_cif": str(source) if source is not None else "",
                "target_cif": str(target),
                "copied": bool(source is not None),
            }
        )
    return pd.DataFrame(rows)


def extract_cif_archive(
    archive_path: Path,
    target_cif_dir: Path,
    overwrite: bool = False,
) -> int:
    """Extract CIF files from a zip/tar archive into ``data/raw/cifs``."""

    archive_path = Path(archive_path)
    target_cif_dir = Path(target_cif_dir)
    target_cif_dir.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)

    extracted = 0
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or not member.filename.lower().endswith(".cif"):
                    continue
                target = target_cif_dir / Path(member.filename).name
                if target.exists() and not overwrite:
                    continue
                with archive.open(member) as source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                extracted += 1
        return extracted

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.lower().endswith(".cif"):
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                target = target_cif_dir / Path(member.name).name
                if target.exists() and not overwrite:
                    continue
                with source, target.open("wb") as handle:
                    shutil.copyfileobj(source, handle)
                extracted += 1
        return extracted

    raise ValueError(f"Unsupported CIF archive format: {archive_path}")


def download_cifs_from_url_template(
    url_template: str,
    sample_ids: list[str],
    target_cif_dir: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download per-sample CIFs from a template containing ``{sample_id}``."""

    target_cif_dir = Path(target_cif_dir)
    target_cif_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        url = url_template.format(sample_id=sample_id)
        target = target_cif_dir / f"{sample_id}.cif"
        status = "exists"
        error = ""
        if overwrite or not target.exists():
            try:
                download_file(url, target, overwrite=overwrite)
                status = "downloaded"
            except Exception as exc:  # pragma: no cover - network path
                status = "failed"
                error = str(exc)
        rows.append(
            {
                "sample_id": sample_id,
                "url": url,
                "target_cif": str(target),
                "status": status,
                "error": error,
            }
        )
    return pd.DataFrame(rows)


def download_cifs_from_materials_project(
    labels: pd.DataFrame,
    target_cif_dir: Path,
    mp_api_key: str,
    mp_id_col: str,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Download CIFs from Materials Project using ``mp-api`` if requested."""

    try:
        from mp_api.client import MPRester
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Materials Project CIF download requires `pip install -e .[data]` "
            "or a compatible `mp-api` installation."
        ) from exc

    if mp_id_col not in labels.columns:
        raise ValueError(f"Materials Project ID column {mp_id_col!r} is not in the label table")

    target_cif_dir = Path(target_cif_dir)
    target_cif_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with MPRester(mp_api_key) as rester:
        for _index, row in labels.iterrows():
            sample_id = str(row["id_discharge"])
            material_id = str(row[mp_id_col])
            target = target_cif_dir / f"{sample_id}.cif"
            status = "exists"
            error = ""
            if overwrite or not target.exists():
                try:
                    structure = rester.get_structure_by_material_id(material_id)
                    structure.to(filename=str(target))
                    status = "downloaded"
                except Exception as exc:  # pragma: no cover - network path
                    status = "failed"
                    error = str(exc)
            rows.append(
                {
                    "sample_id": sample_id,
                    "material_id": material_id,
                    "target_cif": str(target),
                    "status": status,
                    "error": error,
                }
            )
    return pd.DataFrame(rows)


def audit_cif_coverage(
    sample_ids: list[str],
    cif_dir: Path,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Write a transparent sample-by-sample CIF coverage table."""

    cif_dir = Path(cif_dir)
    rows = []
    for sample_id in sample_ids:
        expected = cif_dir / f"{sample_id}.cif"
        rows.append(
            {
                "sample_id": sample_id,
                "expected_cif": str(expected),
                "has_cif": expected.exists(),
            }
        )
    coverage = pd.DataFrame(rows)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        coverage.to_csv(output_path, index=False)
    return coverage


def prepare_data_foundation(
    root: Path,
    target_col: str,
    mp_total: Path | None = None,
    mp_total_url: str | None = None,
    cif_dir: Path | None = None,
    cif_archive: Path | None = None,
    cif_archive_url: str | None = None,
    cif_url_template: str | None = None,
    mp_api_key: str | None = None,
    mp_id_col: str | None = None,
    atom_init: Path | None = None,
    atom_init_url: str | None = None,
    id_col: str = "id_discharge",
    formula_col: str = "formula_discharge",
    working_ion_col: str | None = "working_ion",
    labels_output: Path | None = None,
    seeds: list[int] | None = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    modalities: list[str] | None = None,
    preprocess: bool = False,
    allow_missing_cifs: bool = False,
    overwrite: bool = False,
) -> DataFoundationResult:
    """Prepare raw inputs, labels, splits, and optional modality caches."""

    root = Path(root).resolve()
    paths = ProjectPaths(root=root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_cif_dir.mkdir(parents=True, exist_ok=True)
    seeds = [42] if seeds is None else list(seeds)
    modalities = ["tabular", "structure", "rdf"] if modalities is None else list(modalities)

    raw_mp_total = _stage_mp_total(
        target_path=paths.raw_mp_total,
        mp_total=mp_total,
        mp_total_url=mp_total_url,
        overwrite=overwrite,
    )
    _stage_atom_init(paths.raw_dir / "atom_init.json", atom_init, atom_init_url, overwrite=overwrite)

    labels_path = labels_output or paths.data_dir / "labels" / f"{target_col}_labels_keep_last.csv"
    labels, label_stats = build_label_table_from_mp_total(
        mp_total_path=raw_mp_total,
        output_path=labels_path,
        target_col=target_col,
        id_col=id_col,
        formula_col=formula_col,
        working_ion_col=working_ion_col,
        extra_metadata_cols=tuple([mp_id_col] if mp_id_col else []),
    )
    sample_ids = labels["id_discharge"].astype(str).tolist()

    if cif_archive_url:
        archive_path = paths.raw_dir / "external" / Path(cif_archive_url).name
        download_file(cif_archive_url, archive_path, overwrite=overwrite)
        extract_cif_archive(archive_path, paths.raw_cif_dir, overwrite=overwrite)
    if cif_archive:
        extract_cif_archive(cif_archive, paths.raw_cif_dir, overwrite=overwrite)
    if cif_dir:
        copy_matching_cifs(cif_dir, sample_ids, paths.raw_cif_dir, overwrite=overwrite)
    if cif_url_template:
        download_cifs_from_url_template(cif_url_template, sample_ids, paths.raw_cif_dir, overwrite=overwrite)
    if mp_api_key and mp_id_col:
        download_cifs_from_materials_project(labels, paths.raw_cif_dir, mp_api_key, mp_id_col, overwrite=overwrite)

    coverage_path = paths.data_dir / "manifests" / f"{target_col}_cif_coverage.csv"
    coverage = audit_cif_coverage(sample_ids, paths.raw_cif_dir, output_path=coverage_path)
    missing = coverage.loc[~coverage["has_cif"], "sample_id"].astype(str).tolist()
    if missing and not allow_missing_cifs:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} CIF files in {paths.raw_cif_dir}. "
            f"Examples: {preview}. Use --allow-missing-cifs only for staging/auditing."
        )

    split_paths: list[str] = []
    processed_roots: list[str] = []
    for seed in seeds:
        split_path = (
            paths.data_dir
            / "splits"
            / "random"
            / target_col
            / f"random_{target_col}_seed_{seed}.json"
        )
        create_split_manifest(
            labels_path=labels_path,
            output_path=split_path,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        split_paths.append(str(split_path))
        if preprocess:
            preprocess_modalities(
                root=root,
                split_path=split_path,
                labels_path=labels_path,
                modalities=modalities,
            )
            processed_roots.append(str(paths.processed_split_dir(split_path.stem)))

    manifest_path = paths.data_dir / "manifests" / f"{target_col}_data_foundation_manifest.json"
    result = DataFoundationResult(
        mp_total_path=str(raw_mp_total),
        labels_path=str(labels_path),
        cif_coverage_path=str(coverage_path),
        manifest_path=str(manifest_path),
        split_paths=split_paths,
        processed_roots=processed_roots,
        n_source_rows=label_stats["n_source_rows"],
        n_labels=label_stats["n_labels"],
        n_duplicate_rows_removed=label_stats["n_duplicate_rows_removed"],
        n_missing_target_rows_removed=label_stats["n_missing_target_rows_removed"],
        n_cifs_found=int(coverage["has_cif"].sum()),
        n_cifs_missing=int((~coverage["has_cif"]).sum()),
    )
    _write_foundation_manifest(result, target_col, modalities, raw_mp_total, labels_path, manifest_path)
    return result


def _stage_mp_total(
    target_path: Path,
    mp_total: Path | None,
    mp_total_url: str | None,
    overwrite: bool,
) -> Path:
    target_path = Path(target_path)
    if mp_total_url:
        return download_file(mp_total_url, target_path, overwrite=overwrite)
    if mp_total is None:
        if target_path.exists():
            return target_path
        raise FileNotFoundError(
            "Provide --mp-total, --mp-total-url, or place mp_total.csv at data/raw/mp_total.csv"
        )
    mp_total = Path(mp_total)
    if not mp_total.exists():
        raise FileNotFoundError(mp_total)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if mp_total.resolve() != target_path.resolve() and (overwrite or not target_path.exists()):
        shutil.copy2(mp_total, target_path)
    return target_path


def _stage_atom_init(
    target_path: Path,
    atom_init: Path | None,
    atom_init_url: str | None,
    overwrite: bool,
) -> None:
    if atom_init_url:
        download_file(atom_init_url, target_path, overwrite=overwrite)
        return
    if atom_init is None:
        return
    atom_init = Path(atom_init)
    if not atom_init.exists():
        raise FileNotFoundError(atom_init)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    if atom_init.resolve() != target_path.resolve() and (overwrite or not target_path.exists()):
        shutil.copy2(atom_init, target_path)


def _cif_stem_map(source_cif_dir: Path) -> dict[str, Path]:
    paths = list(Path(source_cif_dir).rglob("*.cif")) + list(Path(source_cif_dir).rglob("*.CIF"))
    mapping: dict[str, Path] = {}
    for path in paths:
        mapping.setdefault(path.stem.lower(), path)
    return mapping


def _write_foundation_manifest(
    result: DataFoundationResult,
    target_col: str,
    modalities: list[str],
    mp_total_path: Path,
    labels_path: Path,
    manifest_path: Path,
) -> None:
    manifest = asdict(result)
    manifest["target_col"] = target_col
    manifest["modalities"] = modalities
    manifest["source_hashes"] = {
        "mp_total_sha256": file_sha256(mp_total_path),
        "labels_sha256": file_sha256(labels_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prepare IM-Fuse data from mp_total.csv and CIF structures."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--target-col", required=True)
    parser.add_argument("--mp-total", type=Path, default=None)
    parser.add_argument("--mp-total-url", default=None)
    parser.add_argument("--cif-dir", type=Path, default=None)
    parser.add_argument("--cif-archive", type=Path, default=None)
    parser.add_argument("--cif-archive-url", default=None)
    parser.add_argument("--cif-url-template", default=None)
    parser.add_argument("--mp-api-key", default=None)
    parser.add_argument("--mp-id-col", default=None)
    parser.add_argument("--atom-init", type=Path, default=None)
    parser.add_argument("--atom-init-url", default=None)
    parser.add_argument("--id-col", default="id_discharge")
    parser.add_argument("--formula-col", default="formula_discharge")
    parser.add_argument("--working-ion-col", default="working_ion")
    parser.add_argument("--labels-output", type=Path, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--modalities", nargs="+", default=["tabular", "structure", "rdf"])
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--allow-missing-cifs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = prepare_data_foundation(
        root=args.root,
        target_col=args.target_col,
        mp_total=args.mp_total,
        mp_total_url=args.mp_total_url,
        cif_dir=args.cif_dir,
        cif_archive=args.cif_archive,
        cif_archive_url=args.cif_archive_url,
        cif_url_template=args.cif_url_template,
        mp_api_key=args.mp_api_key,
        mp_id_col=args.mp_id_col,
        atom_init=args.atom_init,
        atom_init_url=args.atom_init_url,
        id_col=args.id_col,
        formula_col=args.formula_col,
        working_ion_col=args.working_ion_col,
        labels_output=args.labels_output,
        seeds=args.seeds,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        modalities=args.modalities,
        preprocess=args.preprocess,
        allow_missing_cifs=args.allow_missing_cifs,
        overwrite=args.overwrite,
    )
    print(f"Wrote labels: {result.labels_path}")
    print(f"Wrote CIF coverage: {result.cif_coverage_path}")
    print(f"Wrote data manifest: {result.manifest_path}")
    print(
        "CIF coverage: "
        f"{result.n_cifs_found}/{result.n_labels} found, {result.n_cifs_missing} missing"
    )
    if result.split_paths:
        print(f"Wrote {len(result.split_paths)} split manifest(s)")
    if result.processed_roots:
        print(f"Wrote processed modality cache(s): {', '.join(result.processed_roots)}")


if __name__ == "__main__":
    main()
