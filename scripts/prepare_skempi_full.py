#!/usr/bin/env python3
"""Prepare the full SKEMPI v2 CSV for the graph-model pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

from prepare_data import (
    OUTPUT_FIELDS,
    apply_mutations,
    first_nonempty,
    make_record,
    normalize_pdb_id,
    sample_identity,
    stable_sample_id,
    write_csv,
)


def read_skempi(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def affinity(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            try:
                parsed = float(value)
                if parsed > 0 and math.isfinite(parsed):
                    return parsed
            except ValueError:
                pass
    return None


def ddg_from_affinity(row: dict[str, str]) -> float:
    mut = affinity(row, "Affinity_mut_parsed", "Affinity_mut")
    wt = affinity(row, "Affinity_wt_parsed", "Affinity_wt")
    if mut is None or wt is None:
        raise ValueError("missing positive mutant/wild-type affinity")
    temperature_text = first_nonempty(row, "Temperature") or "298.15"
    match = re.search(r"[-+]?\d+(?:\.\d+)?", temperature_text)
    temperature = float(match.group(0)) if match else 298.15
    # Dataset target: ddG = RT ln(Kd_wt / Kd_mut), kcal/mol.
    return (1.987204258e-3 * temperature) * math.log(wt / mut)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    pdb_lookup = {path.stem.upper(): path.resolve() for path in args.pdb_dir.glob("*.pdb")}
    structure_cache = {}
    records: dict[str, dict[str, str]] = {}
    occurrences: dict[str, int] = {}
    status_counts: dict[str, int] = {}

    for row in read_skempi(args.csv):
        mutation_raw = first_nonempty(row, "Mutation(s)_cleaned", "Mutation(s)_PDB")
        identity = sample_identity(row, mutation_raw)
        occurrence = occurrences.get(identity, 0)
        occurrences[identity] = occurrence + 1
        sample_id = stable_sample_id("SKEMPI", identity, occurrence)
        row = dict(row)
        try:
            row["ddG"] = f"{ddg_from_affinity(row):.12g}"
        except Exception:
            row["ddG"] = ""
        record = make_record(
            "SKEMPI", row, args.pdb_dir, pdb_lookup, structure_cache, sample_id
        )
        records[sample_id] = record
        status_counts[record["status"]] = status_counts.get(record["status"], 0) + 1

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(records.values(), key=lambda item: item["sample_id"])
    write_csv(output / "samples.csv", list(OUTPUT_FIELDS), rows)
    (output / "prepare_report.json").write_text(
        json.dumps(
            {
                "source_csv": str(args.csv),
                "pdb_dir": str(args.pdb_dir),
                "raw_rows": len(read_skempi(args.csv)),
                "unique_samples": len(rows),
                "status_counts": status_counts,
                "pdb_files": len(pdb_lookup),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(status_counts, sort_keys=True))


if __name__ == "__main__":
    main()
