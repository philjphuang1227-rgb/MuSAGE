#!/usr/bin/env python3
"""Merge five independently trained fold outputs into one benchmark report."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from independent_ddg.metrics import regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--num-folds", type=int, default=5)
    args = parser.parse_args()

    if args.num_folds < 2:
        raise ValueError("--num-folds must be at least 2")

    reports = []
    configurations = []
    for fold in range(1, args.num_folds + 1):
        path = args.output_dir / f"metrics_fold_{fold}.json"
        if not path.exists():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if len(payload["fold_metrics"]) != 1:
            raise ValueError(f"{path} does not contain exactly one fold")
        row = payload["fold_metrics"][0]
        if int(row["fold"]) != fold:
            raise ValueError(f"{path} contains fold {row['fold']}")
        reports.append(row)
        configurations.append(payload["configuration"])

    predictions = []
    seen = set()
    for fold in range(1, args.num_folds + 1):
        path = args.output_dir / f"fold_{fold}" / "predictions.csv"
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                sample_id = row["sample_id"]
                if sample_id in seen:
                    raise ValueError(f"duplicate prediction for {sample_id}")
                seen.add(sample_id)
                predictions.append(
                    (
                        sample_id,
                        int(row["fold"]),
                        float(row["ddg_true"]),
                        float(row["ddg_pred"]),
                    )
                )
    predictions.sort(key=lambda row: row[0])
    truth = np.asarray([row[2] for row in predictions])
    predicted = np.asarray([row[3] for row in predictions])
    metric_names = ("pearson", "spearman", "rmse", "mae", "auroc")
    report = {
        "mode": configurations[0]["mode"],
        "configuration": configurations[0],
        "parallel_fold_jobs": True,
        "fold_metrics": reports,
        "fold_mean_std": {
            name: {
                "mean": float(np.mean([row[name] for row in reports])),
                "std": float(np.std([row[name] for row in reports], ddof=1)),
            }
            for name in metric_names
        },
        "pooled_validation": regression_metrics(truth, predicted),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "fold", "ddg_true", "ddg_pred"])
        writer.writerows(predictions)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
