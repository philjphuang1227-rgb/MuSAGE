#!/usr/bin/env python3
"""Cache frozen GearNet-Edge mutation-site embeddings for ddG samples.

The official multiview-contrast checkpoint was trained on individual protein
chains.  Each mutation is therefore encoded in a deterministic, contiguous
window of its own chain.  Multiple mutation sites that fit in the same window
share one encoder pass.  Per-site files make an interrupted cluster job fully
resumable.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchdrug import data, layers, models
from torchdrug.layers import geometry

from independent_ddg.pdb import read_ca_residues


def protein_from_ca(residues) -> data.Protein:
    """Build a residue-view TorchDrug protein without RDKit bond parsing."""

    if not residues:
        raise ValueError("no C-alpha residues supplied")
    count = len(residues)
    unknown_residue_id = 20
    residue_type = torch.tensor(
        [data.Protein.residue_symbol2id.get(row.aa, unknown_residue_id) for row in residues],
        dtype=torch.long,
    )
    return data.Protein(
        edge_list=torch.empty((0, 3), dtype=torch.long),
        atom_type=torch.full((count,), 6, dtype=torch.long),
        bond_type=torch.empty((0,), dtype=torch.long),
        residue_type=residue_type,
        view="residue",
        atom_name=torch.full((count,), data.Protein.atom_name2id["CA"], dtype=torch.long),
        atom2residue=torch.arange(count, dtype=torch.long),
        residue_feature=F.one_hot(residue_type, num_classes=21).float(),
        residue_number=torch.tensor([row.resseq for row in residues], dtype=torch.long),
        insertion_code=torch.tensor(
            [data.Protein.alphabet2id.get(row.icode[:1], 0) for row in residues],
            dtype=torch.long,
        ),
        chain_id=torch.tensor(
            [data.Protein.alphabet2id.get(row.chain[:1], 0) for row in residues],
            dtype=torch.long,
        ),
        node_position=torch.tensor([row.ca for row in residues], dtype=torch.float),
        num_node=count,
    )


def site_key(row: dict[str, str], mutation: dict[str, object]) -> str:
    """Return a chain-site identity; partner labels do not change this encoder input."""

    identity = json.dumps(
        [
            str(Path(row["pdb_path"]).resolve()),
            str(mutation["chain"]),
            int(mutation["resseq"]),
            str(mutation.get("icode", "")),
        ],
        separators=(",", ":"),
    )
    return hashlib.sha1(identity.encode("utf-8")).hexdigest()


def make_model(checkpoint: Path, device: torch.device):
    construction = layers.GraphConstruction(
        node_layers=[geometry.IdentityNode()],
        edge_layers=[
            geometry.SequentialEdge(max_distance=2),
            geometry.SpatialEdge(radius=10.0, min_distance=5),
            geometry.KNNEdge(k=10, min_distance=5),
        ],
        edge_feature="gearnet",
    ).to(device)
    model = models.GearNet(
        input_dim=21,
        hidden_dims=[512] * 6,
        batch_norm=True,
        concat_hidden=True,
        short_cut=True,
        readout="sum",
        num_relation=7,
        edge_input_dim=59,
        num_angle_bin=8,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    return construction, model.eval().to(device)


def save_feature(path: Path, feature: np.ndarray) -> None:
    if feature.shape != (3072,) or not np.isfinite(feature).all():
        raise ValueError(f"invalid GearNet feature for {path.name}: {feature.shape}")
    temporary = path.with_suffix(".tmp.npy")
    np.save(temporary, feature.astype(np.float32, copy=False))
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--site-cache", type=Path, required=True)
    parser.add_argument("--max-residues", type=int, default=256)
    args = parser.parse_args()
    if args.max_residues < 16:
        raise ValueError("--max-residues must be at least 16")

    with args.samples.open("r", encoding="utf-8", newline="") as handle:
        all_rows = list(csv.DictReader(handle))
    rows = [row for row in all_rows if not row.get("status") or row["status"] == "ready"]
    if not rows:
        raise ValueError("sample table contains no ready rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.site_cache.mkdir(parents=True, exist_ok=True)

    requests: dict[str, tuple[dict[str, str], dict[str, object]]] = {}
    sample_keys: dict[str, list[str]] = {}
    labels: dict[str, float] = {}
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        keys = []
        mutations = json.loads(row["mutations_json"])
        if not mutations:
            raise ValueError(f"sample {row['sample_id']} has no mutations")
        for mutation in mutations:
            key = site_key(row, mutation)
            requests.setdefault(key, (row, mutation))
            groups[(row["pdb_path"], str(mutation["chain"]))].add(key)
            keys.append(key)
        sample_keys[row["sample_id"]] = keys
        labels[row["sample_id"]] = float(row["ddg"])

    missing = {key for key in requests if not (args.site_cache / f"{key}.npy").exists()}
    print(
        f"samples={len(rows)} unique_sites={len(requests)} "
        f"cached={len(requests) - len(missing)} missing={len(missing)}",
        flush=True,
    )
    started = time.time()
    encoded_windows = 0
    if missing:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("GearNet feature caching requires a CUDA GPU")
        construction, model = make_model(args.checkpoint, device)
        torch.cuda.reset_peak_memory_stats(device)

        active_groups = [item for item in sorted(groups) if groups[item] & missing]
        encoded_sites = 0
        for group_number, (pdb_path, chain) in enumerate(active_groups, start=1):
            # The pretrained encoder has 20 standard residue classes.  M1707
            # contains 17 leading UNK records in one 4NKQ chain; they are not
            # mutation sites and are outside the checkpoint's training domain.
            chain_rows = [
                row
                for row in read_ca_residues(pdb_path)
                if row.chain == chain and row.aa in "ACDEFGHIKLMNPQRSTVWY"
            ]
            if not chain_rows:
                raise ValueError(f"no C-alpha residues for chain {chain} in {pdb_path}")
            index = {(row.resseq, row.icode): i for i, row in enumerate(chain_rows)}
            site_positions: dict[str, int] = {}
            for key in groups[(pdb_path, chain)] & missing:
                mutation = requests[key][1]
                identity = (int(mutation["resseq"]), str(mutation.get("icode", "")))
                if identity not in index:
                    raise KeyError(f"mutation site {chain}{identity[0]}{identity[1]} absent in {pdb_path}")
                site_positions[key] = index[identity]

            pending = set(site_positions)
            while pending:
                anchor_key = min(pending, key=lambda key: site_positions[key])
                anchor = site_positions[anchor_key]
                size = min(args.max_residues, len(chain_rows))
                left = max(0, min(anchor - size // 2, len(chain_rows) - size))
                right = left + size
                covered = sorted(
                    (key for key in pending if left <= site_positions[key] < right),
                    key=lambda key: site_positions[key],
                )
                protein = protein_from_ca(chain_rows[left:right])
                graph = construction(data.Protein.pack([protein]).to(device))
                with torch.inference_mode():
                    output = model(graph, graph.node_feature.float())["node_feature"]
                node_feature = output.float().cpu().numpy()
                for key in covered:
                    feature = node_feature[site_positions[key] - left]
                    save_feature(args.site_cache / f"{key}.npy", feature)
                    pending.remove(key)
                    encoded_sites += 1
                encoded_windows += 1
                del protein, graph, output, node_feature

            if group_number % 10 == 0 or group_number == len(active_groups):
                print(
                    f"groups={group_number}/{len(active_groups)} "
                    f"windows={encoded_windows} sites={encoded_sites}/{len(missing)} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

    sample_ids = sorted(sample_keys)
    pooled_features = []
    site_features = []
    site_offsets = [0]
    site_counts = []
    for sample_id in sample_ids:
        matrix = np.stack(
            [np.load(args.site_cache / f"{key}.npy", allow_pickle=False) for key in sample_keys[sample_id]]
        ).astype(np.float32)
        pooled_features.append(matrix.mean(axis=0))
        site_features.extend(matrix)
        site_counts.append(len(matrix))
        site_offsets.append(len(site_features))

    pooled = np.stack(pooled_features).astype(np.float32)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(sample_ids),
        x=pooled,
        site_x=np.stack(site_features).astype(np.float32),
        site_offsets=np.asarray(site_offsets, dtype=np.int64),
        site_counts=np.asarray(site_counts, dtype=np.int64),
        y=np.asarray([labels[sample_id] for sample_id in sample_ids], dtype=np.float32),
        model=np.asarray("official_mc_gearnet_edge"),
        model_commit=np.asarray("28bf75f5dfb6cd7b9bf4394d153a87f69bddf1a6"),
        pooling=np.asarray("mean_over_mutation_sites"),
        window=np.asarray(f"contiguous_same_chain_{args.max_residues}"),
    )
    print(
        f"saved samples={len(sample_ids)} shape={pooled.shape} windows={encoded_windows} "
        f"elapsed={time.time() - started:.1f}s output={args.output}",
        flush=True,
    )
    if torch.cuda.is_available():
        print(
            f"cuda_peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024**3:.3f} "
            f"cuda_peak_reserved_gib={torch.cuda.max_memory_reserved() / 1024**3:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
