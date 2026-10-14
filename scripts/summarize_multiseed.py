#!/usr/bin/env python3
"""Aggregate completed five-fold metric files across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = ("pearson", "spearman", "rmse", "mae", "auroc")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--benchmark-pcc", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for path in args.metrics:
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("configuration", {})
        row = {"path": str(path), "seed": int(config.get("seed", -1))}
        for metric in METRICS:
            row[metric] = float(payload["fold_mean_std"][metric]["mean"])
        rows.append(row)

    if len(rows) < 2:
        raise SystemExit("at least two completed seed metrics files are required")

    rng = np.random.default_rng(args.seed)
    summary: dict[str, object] = {"n_seeds": len(rows), "seeds": rows, "aggregate": {}}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        draws = rng.choice(values, size=(args.bootstrap_samples, len(values)), replace=True)
        bootstrap_means = draws.mean(axis=1)
        summary["aggregate"][metric] = {
            "mean": float(values.mean()),
            "seed_std": float(values.std(ddof=1)),
            "bootstrap_95_ci": [
                float(np.quantile(bootstrap_means, 0.025)),
                float(np.quantile(bootstrap_means, 0.975)),
            ],
        }

    if args.benchmark_pcc is not None:
        pcc_values = np.asarray([row["pearson"] for row in rows], dtype=np.float64)
        summary["benchmark_pcc"] = args.benchmark_pcc
        summary["all_seeds_exceed_benchmark"] = bool(
            np.all(pcc_values > args.benchmark_pcc)
        )
        summary["mean_pcc_margin"] = float(pcc_values.mean() - args.benchmark_pcc)

    rendered = json.dumps(summary, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
