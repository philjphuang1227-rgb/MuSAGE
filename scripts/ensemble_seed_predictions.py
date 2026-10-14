#!/usr/bin/env python3
"""Average aligned out-of-fold predictions from seeds or fixed model sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from independent_ddg.metrics import regression_metrics


def read_predictions(output_dir: Path) -> dict[str, tuple[int, float, float]]:
    rows: dict[str, tuple[int, float, float]] = {}
    paths = sorted(output_dir.glob("fold_*/predictions.csv"))
    if not paths and (output_dir / "predictions.csv").exists():
        paths = [output_dir / "predictions.csv"]
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rows[row["sample_id"]] = (
                    int(row["fold"]), float(row["ddg_true"]), float(row["ddg_pred"])
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    seed_rows = [read_predictions(path) for path in args.output_dirs]
    sample_ids = sorted(seed_rows[0])
    if any(set(rows) != set(sample_ids) for rows in seed_rows[1:]):
        raise ValueError("seed prediction files do not contain identical sample IDs")

    ensemble_rows = []
    for sample_id in sample_ids:
        fold, truth, _ = seed_rows[0][sample_id]
        predictions = []
        for rows in seed_rows:
            other_fold, other_truth, prediction = rows[sample_id]
            if other_fold != fold or not np.isclose(other_truth, truth):
                raise ValueError(f"inconsistent fold or label for {sample_id}")
            predictions.append(prediction)
        ensemble_rows.append((sample_id, fold, truth, float(np.mean(predictions))))

    fold_metrics = []
    folds = sorted({row[1] for row in ensemble_rows})
    for fold in folds:
        fold_rows = [row for row in ensemble_rows if row[1] == fold]
        truth = np.asarray([row[2] for row in fold_rows])
        predicted = np.asarray([row[3] for row in fold_rows])
        fold_metrics.append({"fold": fold, **regression_metrics(truth, predicted)})

    metric_names = ("pearson", "spearman", "rmse", "mae", "auroc")
    pooled_true = np.asarray([row[2] for row in ensemble_rows])
    pooled_predicted = np.asarray([row[3] for row in ensemble_rows])
    report = {
        "n_sources": len(seed_rows),
        "source_dirs": [str(path) for path in args.output_dirs],
        "fold_metrics": fold_metrics,
        "fold_mean_std": {
            name: {
                "mean": float(np.mean([row[name] for row in fold_metrics])),
                "std": float(np.std([row[name] for row in fold_metrics], ddof=1)),
            }
            for name in metric_names
        },
        "pooled_validation": regression_metrics(pooled_true, pooled_predicted),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "fold", "ddg_true", "ddg_pred"])
        writer.writerows(ensemble_rows)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
