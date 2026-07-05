#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_dir", type=Path, required=True)
    parser.add_argument("--split_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for prediction_path in sorted(args.predictions_dir.glob("*/*_predictions.csv")):
        predictions = pd.read_csv(prediction_path)
        if predictions.empty:
            continue
        seed = int(predictions["seed"].iloc[0])
        split_root = args.split_dir / f"seed_{seed}"
        train = pd.read_csv(split_root / "train.csv")
        val = pd.read_csv(split_root / "val.csv")
        test = pd.read_csv(split_root / "test.csv")
        mse = mean_squared_error(predictions["y_true"], predictions["y_pred"])
        rows.append(
            {
                "heldout_family": str(test["anion_family"].dropna().iloc[0]),
                "model_name": str(predictions["model_name"].iloc[0]),
                "modality_set": str(predictions["modality_set"].iloc[0]),
                "seed": seed,
                "MAE": mean_absolute_error(predictions["y_true"], predictions["y_pred"]),
                "MSE": mse,
                "RMSE": mse**0.5,
                "R2": r2_score(predictions["y_true"], predictions["y_pred"]),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["seed", "model_name"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} rows to {args.output}")


if __name__ == "__main__":
    main()
