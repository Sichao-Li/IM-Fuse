import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

from battery_fusion.utils.hash import file_sha256


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("Split ratios must all be positive")


def create_split_manifest(
    labels_path: Path,
    output_path: Path,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> dict[str, Any]:
    """Create a deterministic split manifest from a deduplicated label table."""

    _validate_ratios(train_ratio, val_ratio, test_ratio)
    labels_path = Path(labels_path)
    output_path = Path(output_path)
    labels = pd.read_csv(labels_path)
    if "id_discharge" not in labels.columns:
        raise ValueError(f"{labels_path} must contain an id_discharge column")

    ids = labels["id_discharge"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError(f"{labels_path} must contain one row per id_discharge")

    shuffled = ids[:]
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)

    manifest: dict[str, Any] = {
        "name": output_path.stem,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "val": val_ratio,
            "test": test_ratio,
        },
        "source": {
            "labels_path": str(labels_path),
            "labels_sha256": file_sha256(labels_path),
            "n_samples": len(shuffled),
        },
        "splits": {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def load_split_manifest(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create a deterministic split manifest.")
    parser.add_argument(
        "--labels", type=Path, default=Path("data/sample_order/sample_order_keep_last.csv")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = args.output or Path("data/splits") / f"split_seed_{args.seed}.json"
    manifest = create_split_manifest(
        labels_path=args.labels,
        output_path=output,
        seed=args.seed,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
    )
    print(
        "Wrote split manifest "
        f"{output} with {len(manifest['splits']['train'])}/"
        f"{len(manifest['splits']['val'])}/{len(manifest['splits']['test'])} samples"
    )


if __name__ == "__main__":
    main()
