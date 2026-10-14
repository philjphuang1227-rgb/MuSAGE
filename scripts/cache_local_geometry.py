#!/usr/bin/env python3
"""Create explicit label-free mutation-neighborhood geometry descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


DISTANCE_BINS = ((0.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.0), (10.0, 12.01))
AA_COUNT = 21


def descriptor(graph: dict[str, object]) -> tuple[np.ndarray, list[str]]:
    coordinates = graph["coordinates"].numpy()
    node_features = graph["node_features"].numpy()
    mutation_indexes = np.flatnonzero(graph["mutation_mask"].numpy() > 0.5)
    partner = np.argmax(node_features[:, 21:23], axis=1)
    amino_acid = np.argmax(node_features[:, :AA_COUNT], axis=1)

    distances_by_site = np.linalg.norm(
        coordinates[None, :, :] - coordinates[mutation_indexes, None, :], axis=-1
    )
    nearest_site = np.argmin(distances_by_site, axis=0)
    distance = distances_by_site[nearest_site, np.arange(len(coordinates))]
    same_partner = partner == partner[mutation_indexes[nearest_site]]

    values: list[float] = []
    names: list[str] = []
    for relationship, mask in (("same", same_partner), ("opposite", ~same_partner)):
        for left, right in DISTANCE_BINS:
            values.append(float(np.log1p(np.sum(mask & (distance >= left) & (distance < right)))))
            names.append(f"{relationship}_count_{left:g}_{right:g}")
        local_mask = mask & (distance <= 8.0)
        denominator = max(1, int(local_mask.sum()))
        for aa_index in range(AA_COUNT):
            values.append(float(np.sum(local_mask & (amino_acid == aa_index)) / denominator))
            names.append(f"{relationship}_aa_{aa_index}")

    opposite_distance = distance[~same_partner]
    values.extend(
        [
            float(opposite_distance.min()) if len(opposite_distance) else 12.0,
            float(opposite_distance.mean()) if len(opposite_distance) else 12.0,
            float(np.any(opposite_distance <= 8.0)),
            float(len(mutation_indexes)),
            float(np.log1p(len(coordinates))),
        ]
    )
    names.extend(
        ["nearest_opposite", "mean_opposite", "interface_flag", "mutation_count", "log_node_count"]
    )

    same_coordinates = coordinates[same_partner]
    opposite_coordinates = coordinates[~same_partner]
    mutation_coordinates = coordinates[mutation_indexes]
    same_center = same_coordinates.mean(axis=0)
    opposite_center = (
        opposite_coordinates.mean(axis=0) if len(opposite_coordinates) else same_center
    )
    values.extend(
        [
            float(np.linalg.norm(mutation_coordinates - same_center, axis=1).mean()),
            float(np.linalg.norm(mutation_coordinates - opposite_center, axis=1).mean()),
        ]
    )
    names.extend(["distance_to_same_center", "distance_to_opposite_center"])
    return np.asarray(values, dtype=np.float32), names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.graphs.is_dir():
        sources = sorted(args.graphs.glob("*.pt"))
        graph_iterator = (torch.load(path, map_location="cpu") for path in sources)
    else:
        bundle = torch.load(args.graphs, map_location="cpu")
        graph_iterator = iter(bundle["graphs"])

    sample_ids = []
    features = []
    feature_names = None
    for number, graph in enumerate(graph_iterator, start=1):
        values, names = descriptor(graph)
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise RuntimeError("geometry descriptor schema changed within dataset")
        sample_ids.append(graph["sample_id"])
        features.append(values)
        if number % 500 == 0:
            print(f"processed {number} graphs", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=np.asarray(sample_ids),
        x=np.stack(features),
        feature_names=np.asarray(feature_names),
    )
    metadata = {
        "graphs": str(args.graphs),
        "samples": len(sample_ids),
        "features": len(feature_names or []),
        "feature_names": feature_names,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata), flush=True)


if __name__ == "__main__":
    main()
