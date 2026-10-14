#!/usr/bin/env python3
"""Generate label-free frozen ESM-2 mutation-site features independently."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import esm


def crop_around(sequence: str, index: int, max_residues: int) -> tuple[str, int]:
    if not 0 <= index < len(sequence):
        raise IndexError(f"sequence index {index} outside length {len(sequence)}")
    if len(sequence) <= max_residues:
        return sequence, index
    left = max(0, min(index - max_residues // 2, len(sequence) - max_residues))
    return sequence[left : left + max_residues], index - left


def make_batches(items: list[tuple[str, str]], max_tokens: int) -> list[list[tuple[str, str]]]:
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_max = 0
    for item in sorted(items, key=lambda pair: len(pair[1])):
        proposed_max = max(current_max, len(item[1]) + 2)
        if current and proposed_max * (len(current) + 1) > max_tokens:
            batches.append(current)
            current = []
            current_max = 0
        current.append(item)
        current_max = max(current_max, len(item[1]) + 2)
    if current:
        batches.append(current)
    return batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--max-residues", type=int, default=1022)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    with args.samples.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ready"]
    if not rows:
        raise ValueError("no ready samples found")

    requests: dict[str, set[int]] = defaultdict(set)
    sample_sites: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, float] = {}
    for row in rows:
        wt_sequences = json.loads(row["wt_sequences_json"])
        mut_sequences = json.loads(row["mut_sequences_json"])
        sites = json.loads(row["mutation_sites_json"])
        prepared_sites: list[dict[str, Any]] = []
        for site in sites:
            chain = site["chain"]
            index = int(site["sequence_index"])
            wt_crop, wt_index = crop_around(wt_sequences[chain], index, args.max_residues)
            mut_crop, mut_index = crop_around(mut_sequences[chain], index, args.max_residues)
            requests[wt_crop].add(wt_index)
            requests[mut_crop].add(mut_index)
            prepared_sites.append(
                {
                    "wt_sequence": wt_crop,
                    "wt_index": wt_index,
                    "mut_sequence": mut_crop,
                    "mut_index": mut_index,
                }
            )
        sample_sites[row["sample_id"]] = prepared_sites
        labels[row["sample_id"]] = float(row["ddg"])

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    loader = getattr(esm.pretrained, args.model, None)
    if loader is None:
        raise ValueError(f"fair-esm has no pretrained loader named {args.model}")
    model, alphabet = loader()
    model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()

    sequence_items = [(f"seq_{index}", sequence) for index, sequence in enumerate(requests)]
    batches = make_batches(sequence_items, args.max_tokens)
    extracted: dict[tuple[str, int], np.ndarray] = {}
    print(
        f"encoding {len(sequence_items)} unique sequence windows in {len(batches)} batches "
        f"with {args.model} on {device}",
        flush=True,
    )

    with torch.inference_mode():
        for batch_number, batch in enumerate(batches, start=1):
            _, _, tokens = batch_converter(batch)
            tokens = tokens.to(device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")
            ):
                output = model(tokens, repr_layers=[args.layer], return_contacts=False)
            representations = output["representations"][args.layer].float().cpu()
            for row_index, (label, sequence) in enumerate(batch):
                for sequence_index in requests[sequence]:
                    extracted[(sequence, sequence_index)] = (
                        representations[row_index, sequence_index + 1].numpy().copy()
                    )
            if batch_number == 1 or batch_number % 25 == 0 or batch_number == len(batches):
                print(f"completed batch {batch_number}/{len(batches)}", flush=True)

    sample_ids = sorted(sample_sites)
    features: list[np.ndarray] = []
    site_features_flat: list[np.ndarray] = []
    site_offsets = [0]
    y: list[float] = []
    for sample_id in sample_ids:
        site_features = []
        for site in sample_sites[sample_id]:
            h_wt = extracted[(site["wt_sequence"], site["wt_index"])]
            h_mut = extracted[(site["mut_sequence"], site["mut_index"])]
            site_features.append(np.concatenate([h_wt, h_mut, h_mut - h_wt]))
        stacked_sites = np.stack(site_features)
        features.append(np.mean(stacked_sites, axis=0))
        site_features_flat.extend(stacked_sites)
        site_offsets.append(site_offsets[-1] + len(stacked_sites))
        y.append(labels[sample_id])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(sample_ids),
        x=np.stack(features).astype(np.float32),
        site_x=np.stack(site_features_flat).astype(np.float32),
        site_offsets=np.asarray(site_offsets, dtype=np.int64),
        site_counts=np.diff(np.asarray(site_offsets, dtype=np.int64)),
        y=np.asarray(y, dtype=np.float32),
        model=np.asarray(args.model),
        layer=np.asarray(args.layer),
    )
    print(f"saved {len(sample_ids)} samples with shape {np.stack(features).shape} to {args.output}")


if __name__ == "__main__":
    main()
