#!/usr/bin/env python3
"""Cache label-free mutation-centered 3D pooling of residue-level ESM deltas."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import esm
import numpy as np
import torch

from cache_esm_features import crop_around, make_batches
from independent_ddg.pdb import read_atom_residues, read_ca_residues, sequence_and_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--layer", type=int, default=33)
    parser.add_argument("--radius", type=float, default=12.0)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    with args.samples.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ready"]

    requests: dict[str, set[int]] = defaultdict(set)
    sample_requests: dict[str, list[tuple[str, int, str, int, float]]] = {}
    structure_cache: dict[str, tuple[list, dict[str, dict[tuple[int, str], int]]]] = {}

    for number, row in enumerate(rows, start=1):
        pdb_path = row["pdb_path"]
        if pdb_path not in structure_cache:
            ca_residues = read_ca_residues(pdb_path)
            atom_chains = read_atom_residues(pdb_path)
            chain_indexes = {
                chain: sequence_and_index(residues)[1]
                for chain, residues in atom_chains.items()
            }
            structure_cache[pdb_path] = (ca_residues, chain_indexes)
        ca_residues, chain_indexes = structure_cache[pdb_path]
        sites = json.loads(row["mutation_sites_json"])
        wt_sequences = json.loads(row["wt_sequences_json"])
        mut_sequences = json.loads(row["mut_sequences_json"])
        site_coordinates = np.asarray(
            [
                next(
                    residue.ca
                    for residue in ca_residues
                    if residue.chain == site["chain"]
                    and residue.resseq == int(site["resseq"])
                    and residue.icode == site["icode"]
                )
                for site in sites
            ],
            dtype=np.float32,
        )

        local: list[tuple[str, int, str, int, float]] = []
        for residue in ca_residues:
            if residue.chain not in wt_sequences:
                continue
            distance = float(
                np.linalg.norm(
                    site_coordinates - np.asarray(residue.ca, dtype=np.float32), axis=1
                ).min()
            )
            if distance > args.radius:
                continue
            sequence_index = chain_indexes[residue.chain][
                (residue.resseq, residue.icode)
            ]
            wt_crop, wt_index = crop_around(
                wt_sequences[residue.chain], sequence_index, args.max_residues
            )
            mut_crop, mut_index = crop_around(
                mut_sequences[residue.chain], sequence_index, args.max_residues
            )
            requests[wt_crop].add(wt_index)
            requests[mut_crop].add(mut_index)
            local.append(
                (wt_crop, wt_index, mut_crop, mut_index, np.exp(-distance / args.sigma))
            )
        if not local:
            raise ValueError(f"empty local neighborhood for {row['sample_id']}")
        sample_requests[row["sample_id"]] = local
        if number % 500 == 0 or number == len(rows):
            print(f"prepared {number}/{len(rows)} samples", flush=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    model, alphabet = getattr(esm.pretrained, args.model)()
    model.eval().to(device)
    converter = alphabet.get_batch_converter()
    sequence_items = [(f"seq_{i}", sequence) for i, sequence in enumerate(requests)]
    batches = make_batches(sequence_items, args.max_tokens)
    extracted: dict[tuple[str, int], np.ndarray] = {}
    print(
        f"encoding {len(sequence_items)} sequences in {len(batches)} batches on {device}",
        flush=True,
    )

    with torch.inference_mode():
        for batch_number, batch in enumerate(batches, start=1):
            _, _, tokens = converter(batch)
            tokens = tokens.to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")
            ):
                output = model(tokens, repr_layers=[args.layer], return_contacts=False)
            representations = output["representations"][args.layer].float().cpu()
            for row_index, (_, sequence) in enumerate(batch):
                for sequence_index in requests[sequence]:
                    extracted[(sequence, sequence_index)] = (
                        representations[row_index, sequence_index + 1].numpy().copy()
                    )
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(batches):
                print(f"completed batch {batch_number}/{len(batches)}", flush=True)

    sample_ids = sorted(sample_requests)
    pooled = []
    for sample_id in sample_ids:
        deltas = []
        weights = []
        for wt_sequence, wt_index, mut_sequence, mut_index, weight in sample_requests[sample_id]:
            deltas.append(
                extracted[(mut_sequence, mut_index)]
                - extracted[(wt_sequence, wt_index)]
            )
            weights.append(weight)
        pooled.append(np.average(np.stack(deltas), axis=0, weights=np.asarray(weights)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(sample_ids),
        x=np.stack(pooled).astype(np.float32),
        model=np.asarray(args.model),
        layer=np.asarray(args.layer),
        radius=np.asarray(args.radius),
        sigma=np.asarray(args.sigma),
    )
    print(f"saved {len(sample_ids)} local delta vectors to {args.output}", flush=True)


if __name__ == "__main__":
    main()
