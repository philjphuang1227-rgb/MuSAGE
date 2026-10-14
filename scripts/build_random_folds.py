#!/usr/bin/env python3
"""Build reproducible random cross-validation folds at row or complex level."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--split-level",
        choices=("mutation-level", "complex-level"),
        default="mutation-level",
        help="Randomize individual samples or keep every complex in one fold.",
    )
    args = parser.parse_args()
    if args.num_folds < 2:
        raise ValueError("--num-folds must be at least 2")

    with args.samples.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "sample_id" not in rows[0]:
        raise ValueError("sample table has no sample_id rows")
    sample_ids = [row["sample_id"] for row in rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("sample IDs are not unique")

    rng = random.Random(args.seed)
    if args.split_level == "mutation-level":
        shuffled = list(sample_ids)
        rng.shuffle(shuffled)
        validation_by_fold = {
            fold: set(shuffled[fold - 1 :: args.num_folds])
            for fold in range(1, args.num_folds + 1)
        }
        report_extra = {
            "group_column": None,
            "n_groups": len(sample_ids),
            "validation_group_sizes": {
                str(fold): len(validation_by_fold[fold])
                for fold in range(1, args.num_folds + 1)
            },
        }
    else:
        if "complex_id" not in rows[0]:
            raise ValueError("complex-level splitting requires a complex_id column")
        groups: dict[str, list[str]] = {}
        for row in rows:
            complex_id = row["complex_id"].strip()
            if not complex_id:
                raise ValueError(f"sample {row['sample_id']} has an empty complex_id")
            groups.setdefault(complex_id, []).append(row["sample_id"])
        if len(groups) < args.num_folds:
            raise ValueError(
                f"only {len(groups)} complexes are available for {args.num_folds} folds"
            )

        # Shuffle first, then greedily balance whole complexes by sample count.
        # The pre-sort shuffle breaks ties reproducibly, while the allocation
        # prevents one very large complex from making a fold unusably small.
        shuffled_groups = list(groups.items())
        rng.shuffle(shuffled_groups)
        shuffled_groups.sort(key=lambda item: len(item[1]), reverse=True)
        fold_samples = {fold: set() for fold in range(1, args.num_folds + 1)}
        fold_groups = {fold: [] for fold in range(1, args.num_folds + 1)}
        for complex_id, members in shuffled_groups:
            fold = min(
                range(1, args.num_folds + 1),
                key=lambda number: (len(fold_samples[number]), number),
            )
            fold_samples[fold].update(members)
            fold_groups[fold].append(complex_id)
        validation_by_fold = fold_samples
        report_extra = {
            "group_column": "complex_id",
            "n_groups": len(groups),
            "validation_group_sizes": {
                str(fold): len(fold_groups[fold])
                for fold in range(1, args.num_folds + 1)
            },
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, args.num_folds + 1):
        path = args.output_dir / f"fold_{fold}.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "split"])
            validation = validation_by_fold[fold]
            for sample_id in sample_ids:
                writer.writerow([sample_id, "val" if sample_id in validation else "train"])

    report = {
        "samples": str(args.samples),
        "n_samples": len(sample_ids),
        "num_folds": args.num_folds,
        "seed": args.seed,
        "split": args.split_level.replace("-", "_") + "_random",
        "validation_sizes": {
            str(fold): len(validation_by_fold[fold])
            for fold in range(1, args.num_folds + 1)
        },
    }
    report.update(report_extra)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
