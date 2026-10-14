#!/usr/bin/env python3
"""Score one or more fold prediction CSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from independent_ddg.metrics import regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.predictions:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        true = np.asarray([float(row["ddg_true"]) for row in rows])
        predicted = np.asarray([float(row["ddg_pred"]) for row in rows])
        reports.append({"path": str(path), **regression_metrics(true, predicted)})
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
