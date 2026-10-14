#!/usr/bin/env python3
"""Normalize public fold CSVs and map mutations to independently parsed WT PDBs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from independent_ddg.mutations import Mutation, parse_mutations
from independent_ddg.pdb import read_atom_residues, sequence_and_index


DATASETS = ("S4169", "S1131", "M1707")
OUTPUT_FIELDS = (
    "sample_id", "dataset", "pdb_id", "complex_id", "pdb_sub_type", "partners",
    "mutation_raw", "n_mutations", "ddg", "mode", "pdb_path", "status", "error",
    "mutations_json", "mutation_sites_json", "wt_sequences_json", "mut_sequences_json",
)


def first_nonempty(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            return value
    return ""


def normalize_pdb_id(row: dict[str, str]) -> str:
    value = first_nonempty(row, "pdb_id", "protein", "#Pdb")
    return value.split("_")[0].upper()


def sample_identity(row: dict[str, str], mutation_raw: str) -> str:
    return "|".join(
        [
            first_nonempty(row, "pdb_sub_type"),
            first_nonempty(row, "#Pdb", "protein"),
            mutation_raw,
            first_nonempty(row, "mode"),
        ]
    )


def stable_sample_id(dataset: str, identity: str, occurrence: int) -> str:
    identity_with_replicate = identity
    if occurrence:
        identity_with_replicate = f"{identity}|replicate={occurrence}"
    token = hashlib.sha1(
        f"{dataset}|{identity_with_replicate}".encode("utf-8")
    ).hexdigest()[:16]
    return f"{dataset}_{token}"


def apply_mutations(
    pdb_path: Path,
    mutations: list[Mutation],
    structure_cache: dict[Path, tuple[dict[str, str], dict[str, dict[tuple[int, str], int]]]],
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    if pdb_path not in structure_cache:
        chain_residues = read_atom_residues(pdb_path)
        if not chain_residues:
            raise ValueError("PDB has no parseable ATOM residues")
        sequences: dict[str, str] = {}
        indexes: dict[str, dict[tuple[int, str], int]] = {}
        for chain, residues in chain_residues.items():
            sequences[chain], indexes[chain] = sequence_and_index(residues)
        structure_cache[pdb_path] = (sequences, indexes)
    sequences, indexes = structure_cache[pdb_path]

    mutant_lists = {chain: list(sequence) for chain, sequence in sequences.items()}
    sites: list[dict[str, Any]] = []
    for mutation in mutations:
        if mutation.chain not in sequences:
            raise ValueError(
                f"chain {mutation.chain!r} absent; available chains={sorted(sequences)}"
            )
        residue_key = (mutation.resseq, mutation.icode)
        if residue_key not in indexes[mutation.chain]:
            raise ValueError(
                f"residue {mutation.chain}:{mutation.resseq}{mutation.icode} absent from ATOM records"
            )
        sequence_index = indexes[mutation.chain][residue_key]
        observed = sequences[mutation.chain][sequence_index]
        if observed != mutation.wt:
            raise ValueError(
                f"WT mismatch at {mutation.chain}:{mutation.resseq}{mutation.icode}: "
                f"CSV={mutation.wt}, PDB={observed}"
            )
        mutant_lists[mutation.chain][sequence_index] = mutation.mutant
        sites.append(
            {
                **mutation.to_dict(),
                "sequence_index": sequence_index,
                "observed_wt": observed,
            }
        )

    affected_chains = sorted({mutation.chain for mutation in mutations})
    wt_sequences = {chain: sequences[chain] for chain in affected_chains}
    mut_sequences = {chain: "".join(mutant_lists[chain]) for chain in affected_chains}
    return wt_sequences, mut_sequences, sites


def make_record(
    dataset: str,
    row: dict[str, str],
    pdb_dir: Path,
    pdb_lookup: dict[str, Path],
    structure_cache: dict[Path, tuple[dict[str, str], dict[str, dict[tuple[int, str], int]]]],
    sample_id: str,
) -> dict[str, str]:
    mutation_raw = first_nonempty(row, "mutation", "Mutation(s)_cleaned", "Mutation(s)_PDB")
    pdb_id = normalize_pdb_id(row)
    complex_id = first_nonempty(row, "#Pdb", "protein") or pdb_id
    partners = first_nonempty(row, "Partners(A_B)")
    if not partners and "_" in complex_id:
        partners = "_".join(complex_id.split("_")[1:])
    label_text = first_nonempty(row, "ddG", "DDG")
    pdb_path = pdb_lookup.get(pdb_id)

    record = {
        "sample_id": sample_id,
        "dataset": dataset,
        "pdb_id": pdb_id,
        "complex_id": complex_id,
        "pdb_sub_type": first_nonempty(row, "pdb_sub_type"),
        "partners": partners,
        "mutation_raw": mutation_raw,
        "n_mutations": "0",
        "ddg": label_text,
        "mode": first_nonempty(row, "mode"),
        "pdb_path": str(pdb_path or (pdb_dir / f"{pdb_id}.pdb")),
        "status": "error",
        "error": "",
        "mutations_json": "[]",
        "mutation_sites_json": "[]",
        "wt_sequences_json": "{}",
        "mut_sequences_json": "{}",
    }

    try:
        float(label_text)
        if pdb_path is None:
            raise FileNotFoundError(f"PDB file not found for {pdb_id}")
        mutations = parse_mutations(mutation_raw)
        wt_sequences, mut_sequences, sites = apply_mutations(
            pdb_path, mutations, structure_cache
        )
        record.update(
            {
                "n_mutations": str(len(mutations)),
                "status": "ready",
                "mutations_json": json.dumps(
                    [mutation.to_dict() for mutation in mutations], separators=(",", ":")
                ),
                "mutation_sites_json": json.dumps(sites, separators=(",", ":")),
                "wt_sequences_json": json.dumps(wt_sequences, separators=(",", ":")),
                "mut_sequences_json": json.dumps(mut_sequences, separators=(",", ":")),
            }
        )
    except Exception as exc:  # retain every failed sample for an auditable report
        record["error"] = str(exc)
    return record


def read_raw(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def write_csv(path: Path, fieldnames: tuple[str, ...] | list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-fold-dir", type=Path, required=True)
    parser.add_argument("--pdb-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()

    pdb_lookup = {path.stem.upper(): path.resolve() for path in args.pdb_dir.glob("*.pdb")}
    structure_cache: dict[
        Path, tuple[dict[str, str], dict[str, dict[tuple[int, str], int]]]
    ] = {}
    summary: dict[str, Any] = {"pdb_files": len(pdb_lookup), "datasets": {}}

    for dataset in args.datasets:
        dataset_records: dict[str, dict[str, str]] = {}
        fold_rows: dict[int, list[dict[str, str]]] = {fold: [] for fold in range(1, 6)}
        raw_rows = 0
        duplicate_rows = 0

        for fold in range(1, 6):
            fold_occurrences: dict[str, int] = {}
            for split in ("train", "val"):
                raw_path = args.raw_fold_dir / dataset / f"{split}_fold_{fold}.csv"
                if not raw_path.exists():
                    raise FileNotFoundError(raw_path)
                for row_number, row in enumerate(read_raw(raw_path), start=2):
                    raw_rows += 1
                    mutation_raw = first_nonempty(
                        row, "mutation", "Mutation(s)_cleaned", "Mutation(s)_PDB"
                    )
                    identity = sample_identity(row, mutation_raw)
                    occurrence = fold_occurrences.get(identity, 0)
                    fold_occurrences[identity] = occurrence + 1
                    sample_id = stable_sample_id(dataset, identity, occurrence)
                    previous = dataset_records.get(sample_id)
                    if previous is None:
                        record = make_record(
                            dataset,
                            row,
                            args.pdb_dir,
                            pdb_lookup,
                            structure_cache,
                            sample_id,
                        )
                        dataset_records[sample_id] = record
                    else:
                        duplicate_rows += 1
                        record = previous
                        comparable = ("pdb_id", "mutation_raw", "ddg")
                        current_pdb = normalize_pdb_id(row)
                        current_ddg = first_nonempty(row, "ddG", "DDG")
                        current_values = (current_pdb, mutation_raw, current_ddg)
                        if any(previous[key] != value for key, value in zip(comparable, current_values)):
                            raise ValueError(f"sample identity collision for {sample_id}")
                    fold_rows[fold].append(
                        {
                            "sample_id": sample_id,
                            "split": split,
                            "source_file": raw_path.name,
                            "source_row": str(row_number),
                        }
                    )

        records = sorted(dataset_records.values(), key=lambda row: row["sample_id"])
        output_dataset = args.output_dir / dataset
        write_csv(output_dataset / "samples.csv", list(OUTPUT_FIELDS), records)
        for fold, rows in fold_rows.items():
            write_csv(
                output_dataset / f"fold_{fold}.csv",
                ["sample_id", "split", "source_file", "source_row"],
                rows,
            )

        errors: dict[str, int] = {}
        for record in records:
            if record["status"] != "ready":
                key = record["error"].split(":", 1)[0]
                errors[key] = errors.get(key, 0) + 1
        dataset_summary = {
            "raw_rows_across_folds": raw_rows,
            "duplicate_fold_rows": duplicate_rows,
            "unique_samples": len(records),
            "ready_samples": sum(record["status"] == "ready" for record in records),
            "error_samples": sum(record["status"] != "ready" for record in records),
            "single_mutations": sum(record["n_mutations"] == "1" for record in records),
            "multiple_mutations": sum(int(record["n_mutations"] or 0) > 1 for record in records),
            "errors": errors,
            "fold_sizes": {
                str(fold): {
                    split: sum(row["split"] == split for row in rows)
                    for split in ("train", "val")
                }
                for fold, rows in fold_rows.items()
            },
        }
        summary["datasets"][dataset] = dataset_summary
        print(dataset, json.dumps(dataset_summary, indent=2))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
