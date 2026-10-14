#!/usr/bin/env python3
"""Cache mutation-site ESM-IF1 embeddings with local complex context.

Each unique mutation site is encoded once and reused by every multi-mutation
sample containing that site.  The output still has one row for every sample;
its feature is the mean of all of that sample's site embeddings.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from esm import pretrained
from esm.inverse_folding.util import CoordBatchConverter


@dataclass
class BackboneResidue:
    chain: str
    resseq: int
    icode: str
    chain_index: int
    coords: np.ndarray


def read_backbone(path: Path) -> dict[str, list[BackboneResidue]]:
    records: dict[tuple[str, int, str], dict[str, object]] = {}
    order: list[tuple[str, int, str]] = []
    in_first_model = True
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()
            if record == "MODEL":
                model_text = line[10:14].strip()
                in_first_model = not model_text or model_text == "1"
                continue
            if record == "ENDMDL" and in_first_model:
                break
            if record != "ATOM" or not in_first_model or len(line) < 54:
                continue
            atom = line[12:16].strip()
            if atom not in {"N", "CA", "C"} or line[16].strip() not in {"", "A"}:
                continue
            chain = line[21].strip() or "_"
            try:
                resseq = int(line[22:26])
                xyz = np.asarray(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float32,
                )
            except ValueError:
                continue
            key = (chain, resseq, line[26].strip())
            if key not in records:
                records[key] = {"atoms": {}}
                order.append(key)
            records[key]["atoms"][atom] = xyz

    chains: dict[str, list[BackboneResidue]] = {}
    for chain, resseq, icode in order:
        atoms = records[(chain, resseq, icode)]["atoms"]
        coords = np.stack(
            [atoms.get(atom, np.full(3, np.nan, dtype=np.float32)) for atom in ("N", "CA", "C")]
        )
        chain_rows = chains.setdefault(chain, [])
        chain_rows.append(
            BackboneResidue(chain, resseq, icode, len(chain_rows), coords)
        )
    return chains


def centered_window(rows: list[BackboneResidue], center: int, size: int) -> list[BackboneResidue]:
    size = min(size, len(rows))
    left = max(0, min(center - size // 2, len(rows) - size))
    return rows[left : left + size]


def partner_chain_ids(partners: str) -> list[str]:
    # PDB chain identifiers in these benchmarks are single characters.  A side
    # such as "HL" therefore denotes chains H and L.
    return list(dict.fromkeys(character for side in partners.split("_") for character in side))


def site_input(
    chains: dict[str, list[BackboneResidue]],
    partners: str,
    mutation: dict[str, object],
    max_residues: int,
    target_residues: int,
    padding: int,
) -> tuple[np.ndarray, int]:
    target_chain = str(mutation["chain"])
    target_rows = chains[target_chain]
    target_position = next(
        i
        for i, residue in enumerate(target_rows)
        if residue.resseq == int(mutation["resseq"])
        and residue.icode == str(mutation.get("icode", ""))
    )
    target_ca = target_rows[target_position].coords[1]
    if not np.isfinite(target_ca).all():
        raise ValueError("mutation residue has no finite CA coordinate")

    context_chains = [
        chain
        for chain in partner_chain_ids(partners)
        if chain != target_chain and chain in chains and chains[chain]
    ]
    target_size = min(target_residues, max_residues, len(target_rows))
    target_segment = centered_window(target_rows, target_position, target_size)
    usable = max_residues - len(target_segment) - padding * len(context_chains)
    context_size = max(1, usable // max(1, len(context_chains)))

    segments = [target_segment]
    for chain in context_chains:
        rows = chains[chain]
        finite = [i for i, residue in enumerate(rows) if np.isfinite(residue.coords[1]).all()]
        if not finite:
            continue
        closest = min(
            finite,
            key=lambda i: float(np.square(rows[i].coords[1] - target_ca).sum()),
        )
        segments.append(centered_window(rows, closest, context_size))

    concatenated: list[np.ndarray] = []
    target_offset = next(
        i
        for i, residue in enumerate(target_segment)
        if residue.resseq == int(mutation["resseq"])
        and residue.icode == str(mutation.get("icode", ""))
    )
    for index, segment in enumerate(segments):
        if index:
            concatenated.extend(
                [np.full((3, 3), np.nan, dtype=np.float32)] * padding
            )
        concatenated.extend(residue.coords for residue in segment)
    if len(concatenated) > max_residues:
        raise ValueError(f"constructed IF1 input has {len(concatenated)} residues")
    return np.stack(concatenated).astype(np.float32), target_offset


def site_key(row: dict[str, str], mutation: dict[str, object]) -> str:
    identity = json.dumps(
        [
            str(Path(row["pdb_path"]).resolve()),
            row["partners"],
            mutation["chain"],
            int(mutation["resseq"]),
            str(mutation.get("icode", "")),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def encode(model, alphabet, coords: np.ndarray, target_offset: int, device: torch.device) -> np.ndarray:
    converter = CoordBatchConverter(alphabet)
    batch_coords, confidence, _, _, padding_mask = converter([(coords, None, None)])
    batch_coords = batch_coords.to(device)
    confidence = confidence.to(device)
    padding_mask = padding_mask.to(device)
    with torch.inference_mode():
        output = model.encoder.forward(
            batch_coords, padding_mask, confidence, return_all_hiddens=False
        )["encoder_out"][0][1:-1, 0]
    return output[target_offset].float().cpu().numpy().astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-cache", type=Path, required=True)
    parser.add_argument("--max-residues", type=int, default=500)
    parser.add_argument("--target-residues", type=int, default=256)
    parser.add_argument("--padding", type=int, default=10)
    args = parser.parse_args()

    with args.samples.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.site_cache.mkdir(parents=True, exist_ok=True)

    requests: dict[str, tuple[dict[str, str], dict[str, object]]] = {}
    sample_keys: dict[str, list[str]] = {}
    labels: dict[str, float] = {}
    for row in rows:
        keys = []
        for mutation in json.loads(row["mutations_json"]):
            key = site_key(row, mutation)
            requests.setdefault(key, (row, mutation))
            keys.append(key)
        sample_keys[row["sample_id"]] = keys
        labels[row["sample_id"]] = float(row["ddg"])

    missing = [key for key in requests if not (args.site_cache / f"{key}.npy").exists()]
    print(
        f"samples={len(rows)} unique_sites={len(requests)} cached={len(requests)-len(missing)} missing={len(missing)}",
        flush=True,
    )
    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, alphabet = pretrained.load_model_and_alphabet_local(args.checkpoint)
        model = model.eval().to(device)
        structure_cache: dict[str, dict[str, list[BackboneResidue]]] = {}
        for number, key in enumerate(missing, start=1):
            row, mutation = requests[key]
            pdb_path = row["pdb_path"]
            if pdb_path not in structure_cache:
                structure_cache[pdb_path] = read_backbone(Path(pdb_path))
            coords, target_offset = site_input(
                structure_cache[pdb_path], row["partners"], mutation,
                args.max_residues, args.target_residues, args.padding,
            )
            feature = encode(model, alphabet, coords, target_offset, device)
            target = args.site_cache / f"{key}.npy"
            temporary = args.site_cache / f"{key}.tmp.npy"
            np.save(temporary, feature)
            temporary.replace(target)
            if number % 25 == 0 or number == len(missing):
                print(f"encoded {number}/{len(missing)} sites", flush=True)

    sample_ids = sorted(sample_keys)
    features = []
    site_features = []
    site_offsets = [0]
    site_counts = []
    for sample_id in sample_ids:
        sample_site_features = [
            np.load(args.site_cache / f"{key}.npy", allow_pickle=False)
            for key in sample_keys[sample_id]
        ]
        sample_matrix = np.stack(sample_site_features).astype(np.float32)
        features.append(sample_matrix.mean(axis=0))
        site_features.extend(sample_matrix)
        site_counts.append(len(sample_matrix))
        site_offsets.append(len(site_features))
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(sample_ids),
        x=np.stack(features).astype(np.float32),
        site_x=np.stack(site_features).astype(np.float32),
        site_offsets=np.asarray(site_offsets, dtype=np.int64),
        site_counts=np.asarray(site_counts, dtype=np.int64),
        y=np.asarray([labels[sample_id] for sample_id in sample_ids], dtype=np.float32),
        model=np.asarray("esm_if1_gvp4_t16_142M_UR50"),
        pooling=np.asarray("mean_over_mutation_sites"),
    )
    print(f"saved {len(sample_ids)} samples with shape {np.stack(features).shape} to {args.output}", flush=True)


if __name__ == "__main__":
    main()
