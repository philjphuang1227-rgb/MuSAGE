#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
with args.input.open("r", encoding="utf-8", newline="") as handle:
    rows = list(csv.DictReader(handle))
ready = [row for row in rows if row.get("status") == "ready" and row.get("ddg", "")]
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(ready)
print(f"ready={len(ready)} total={len(rows)}")
