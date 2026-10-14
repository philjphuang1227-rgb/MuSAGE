#!/usr/bin/env python3
"""Build label-free mutation-centered residue graphs from WT PDB coordinates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from independent_ddg.pdb import CoordinateResidue, read_ca_residues


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWYX"
AA_INDEX = {aa: index for index, aa in enumerate(AMINO_ACIDS)}


def partner_index(chain: str, partners: str) -> int:
    groups = partners.split("_", 1)
    if len(groups) == 2:
        if chain in set(groups[0]):
            return 0
        if chain in set(groups[1]):
            return 1
    return 0


def radial_basis(distances: np.ndarray, bins: int, maximum: float) -> np.ndarray:
    centers = np.linspace(0.0, maximum, bins, dtype=np.float32)
    width = maximum / bins
    return np.exp(-((distances[:, None] - centers[None, :]) / width) ** 2)


def build_graph(
    row: dict[str, str],
    residues: list[CoordinateResidue],
    radius: float,
    edge_radius: float,
) -> dict[str, object]:
    sites = json.loads(row["mutation_sites_json"])
    residue_lookup = {
        (residue.chain, residue.resseq, residue.icode): index
        for index, residue in enumerate(residues)
    }
    site_indexes: list[int] = []
    for site in sites:
        key = (site["chain"], int(site["resseq"]), site["icode"])
        if key not in residue_lookup:
            raise ValueError(f"missing C-alpha for mutation {key}")
        site_indexes.append(residue_lookup[key])

    all_coordinates = np.asarray([residue.ca for residue in residues], dtype=np.float32)
    site_coordinates = all_coordinates[site_indexes]
    distances_to_site = np.linalg.norm(
        all_coordinates[:, None, :] - site_coordinates[None, :, :], axis=-1
    ).min(axis=1)
    selected = np.flatnonzero(distances_to_site <= radius)
    if len(selected) < 2:
        raise ValueError("local graph contains fewer than two residues")

    local_residues = [residues[index] for index in selected]
    coordinates = all_coordinates[selected]
    coordinates = coordinates - coordinates.mean(axis=0, keepdims=True)
    local_lookup = {
        (residue.chain, residue.resseq, residue.icode): index
        for index, residue in enumerate(local_residues)
    }

    node_features = np.zeros((len(local_residues), 45), dtype=np.float32)
    mutation_mask = np.zeros(len(local_residues), dtype=np.float32)
    for index, residue in enumerate(local_residues):
        aa_index = AA_INDEX.get(residue.aa, AA_INDEX["X"])
        node_features[index, aa_index] = 1.0
        node_features[index, 21 + partner_index(residue.chain, row["partners"])] = 1.0
    for site in sites:
        key = (site["chain"], int(site["resseq"]), site["icode"])
        index = local_lookup[key]
        mutation_mask[index] = 1.0
        node_features[index, 23] = 1.0
        wt_index = AA_INDEX.get(site["wt"], AA_INDEX["X"])
        mut_index = AA_INDEX.get(site["mutant"], AA_INDEX["X"])
        node_features[index, 24 + mut_index] += 1.0
        node_features[index, 24 + wt_index] -= 1.0

    pair_distance = np.linalg.norm(
        coordinates[:, None, :] - coordinates[None, :, :], axis=-1
    )
    source, target = np.where((pair_distance <= edge_radius) & (pair_distance > 0.0))
    if len(source) == 0:
        raise ValueError("local graph contains no edges")
    distances = pair_distance[source, target].astype(np.float32)
    extra = np.zeros((len(source), 3), dtype=np.float32)
    for edge, (left, right) in enumerate(zip(source, target)):
        left_residue = local_residues[left]
        right_residue = local_residues[right]
        extra[edge, 0] = left_residue.chain == right_residue.chain
        extra[edge, 1] = partner_index(
            left_residue.chain, row["partners"]
        ) == partner_index(right_residue.chain, row["partners"])
        if left_residue.chain == right_residue.chain:
            extra[edge, 2] = np.exp(
                -abs(left_residue.resseq - right_residue.resseq) / 10.0
            )
    edge_features = np.concatenate(
        [radial_basis(distances, bins=16, maximum=edge_radius), extra], axis=1
    )
    return {
        "sample_id": row["sample_id"],
        "node_features": torch.from_numpy(node_features),
        "coordinates": torch.from_numpy(coordinates.astype(np.float32)),
        "edge_index": torch.from_numpy(np.stack([source, target]).astype(np.int64)),
        "edge_features": torch.from_numpy(edge_features.astype(np.float32)),
        "mutation_mask": torch.from_numpy(mutation_mask),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--edge-radius", type=float, default=10.0)
    args = parser.parse_args()

    with args.samples.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ready"]
    pdb_cache: dict[str, list[CoordinateResidue]] = {}
    graphs: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []
    for number, row in enumerate(rows, start=1):
        pdb_path = row["pdb_path"]
        if pdb_path not in pdb_cache:
            pdb_cache[pdb_path] = read_ca_residues(pdb_path)
        try:
            graphs.append(
                build_graph(row, pdb_cache[pdb_path], args.radius, args.edge_radius)
            )
        except Exception as exc:
            errors.append({"sample_id": row["sample_id"], "error": str(exc)})
        if number % 500 == 0 or number == len(rows):
            print(
                f"processed {number}/{len(rows)} graphs; errors={len(errors)}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "graphs": graphs,
            "errors": errors,
            "radius": args.radius,
            "edge_radius": args.edge_radius,
            "node_feature_dim": 45,
            "edge_feature_dim": 19,
        },
        args.output,
    )
    print(f"saved {len(graphs)} graphs to {args.output}; errors={len(errors)}")
    if errors:
        error_path = args.output.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
