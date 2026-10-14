#!/usr/bin/env python3
"""Train independent structure, gated-fusion, or cross-attention models."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from independent_ddg.graph_model import DeltaGModel
from independent_ddg.metrics import regression_metrics


def load_dense_feature_bundle(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load a compressed dense feature matrix once and index it by sample ID."""
    with np.load(path, allow_pickle=False) as bundle:
        sample_ids = bundle["sample_ids"].astype(str)
        matrix = bundle["x"].astype(np.float32)
    return {sample_id: matrix[i] for i, sample_id in enumerate(sample_ids)}, matrix


class GraphDataset(Dataset):
    def __init__(
        self,
        sample_ids: list[str],
        graph_by_id: dict[str, object],
        labels: dict[str, float],
        esm_by_id: dict[str, np.ndarray] | None,
        geometry_by_id: dict[str, np.ndarray] | None = None,
        pretrained_structure_by_id: dict[str, np.ndarray] | None = None,
        auxiliary_pretrained_structure_by_id: dict[str, np.ndarray] | None = None,
        site_features_by_id: dict[str, np.ndarray] | None = None,
    ) -> None:
        self.sample_ids = sample_ids
        self.graph_by_id = graph_by_id
        self.labels = labels
        self.esm_by_id = esm_by_id
        self.geometry_by_id = geometry_by_id
        self.pretrained_structure_by_id = pretrained_structure_by_id
        self.auxiliary_pretrained_structure_by_id = auxiliary_pretrained_structure_by_id
        self.site_features_by_id = site_features_by_id

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int):
        sample_id = self.sample_ids[index]
        esm = None if self.esm_by_id is None else self.esm_by_id[sample_id]
        geometry = None if self.geometry_by_id is None else self.geometry_by_id[sample_id]
        pretrained_structure = (
            None
            if self.pretrained_structure_by_id is None
            else self.pretrained_structure_by_id[sample_id]
        )
        auxiliary_pretrained_structure = (
            None
            if self.auxiliary_pretrained_structure_by_id is None
            else self.auxiliary_pretrained_structure_by_id[sample_id]
        )
        site_features = (
            None if self.site_features_by_id is None else self.site_features_by_id[sample_id]
        )
        graph_source = self.graph_by_id[sample_id]
        graph = (
            torch.load(graph_source, map_location="cpu")
            if isinstance(graph_source, Path)
            else graph_source
        )
        return (
            graph,
            self.labels[sample_id],
            esm,
            geometry,
            pretrained_structure,
            auxiliary_pretrained_structure,
            site_features,
            sample_id,
        )


def collate_graphs(items):
    node_features = []
    coordinates = []
    edge_indexes = []
    edge_features = []
    mutation_masks = []
    graph_indexes = []
    esm_features = []
    geometry_features = []
    pretrained_structure_features = []
    auxiliary_pretrained_structure_features = []
    site_features = []
    labels = []
    sample_ids = []
    offset = 0
    for graph_number, item in enumerate(items):
        if len(item) == 4:
            graph, label, esm, sample_id = item
            geometry = None
            pretrained_structure = None
            auxiliary_pretrained_structure = None
            sites = None
        elif len(item) == 5:
            graph, label, esm, geometry, sample_id = item
            pretrained_structure = None
            auxiliary_pretrained_structure = None
            sites = None
        elif len(item) == 6:
            graph, label, esm, geometry, pretrained_structure, sample_id = item
            auxiliary_pretrained_structure = None
            sites = None
        elif len(item) == 7:
            graph, label, esm, geometry, pretrained_structure, sites, sample_id = item
            auxiliary_pretrained_structure = None
        else:
            (
                graph, label, esm, geometry, pretrained_structure,
                auxiliary_pretrained_structure, sites, sample_id,
            ) = item
        nodes = graph["node_features"]
        node_features.append(nodes)
        coordinates.append(graph["coordinates"])
        edge_indexes.append(graph["edge_index"] + offset)
        edge_features.append(graph["edge_features"])
        mutation_masks.append(graph["mutation_mask"])
        graph_indexes.append(torch.full((len(nodes),), graph_number, dtype=torch.long))
        if esm is not None:
            esm_features.append(torch.from_numpy(esm).float())
        if geometry is not None:
            geometry_features.append(torch.from_numpy(geometry).float())
        if pretrained_structure is not None:
            pretrained_structure_features.append(
                torch.from_numpy(pretrained_structure).float()
            )
        if auxiliary_pretrained_structure is not None:
            auxiliary_pretrained_structure_features.append(
                torch.from_numpy(auxiliary_pretrained_structure).float()
            )
        if sites is not None:
            site_features.append(torch.from_numpy(sites).float())
        labels.append(label)
        sample_ids.append(sample_id)
        offset += len(nodes)
    batch = {
        "node_features": torch.cat(node_features),
        "coordinates": torch.cat(coordinates),
        "edge_index": torch.cat(edge_indexes, dim=1),
        "edge_features": torch.cat(edge_features),
        "mutation_mask": torch.cat(mutation_masks),
        "graph_index": torch.cat(graph_indexes),
    }
    if esm_features:
        batch["esm_features"] = torch.stack(esm_features)
    if geometry_features:
        batch["geometry_features"] = torch.stack(geometry_features)
    if pretrained_structure_features:
        batch["pretrained_structure_features"] = torch.stack(
            pretrained_structure_features
        )
    if auxiliary_pretrained_structure_features:
        batch["auxiliary_pretrained_structure_features"] = torch.stack(
            auxiliary_pretrained_structure_features
        )
    if site_features:
        max_sites = max(len(features) for features in site_features)
        feature_dim = site_features[0].shape[-1]
        padded = torch.zeros((len(site_features), max_sites, feature_dim))
        site_mask = torch.zeros((len(site_features), max_sites), dtype=torch.bool)
        for row, features in enumerate(site_features):
            padded[row, : len(features)] = features
            site_mask[row, : len(features)] = True
        batch["site_features"] = padded
        batch["site_mask"] = site_mask
    return batch, torch.tensor(labels, dtype=torch.float32), sample_ids


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def balanced_sampler(
    sample_ids: list[str], groups: dict[str, object], exponent: float = 1.0
) -> WeightedRandomSampler:
    counts = Counter(groups[sample_id] for sample_id in sample_ids)
    weights = torch.tensor(
        [counts[groups[sample_id]] ** (-exponent) for sample_id in sample_ids],
        dtype=torch.double,
    )
    return WeightedRandomSampler(weights, num_samples=len(sample_ids), replacement=True)


def composite_loss(
    prediction: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    sample_ids: list[str],
    groups: dict[str, str],
    huber: nn.Module,
    rank_weight: float,
    classification_weight: float,
    rank_margin: float,
    mse_weight: float,
    antisymmetry_penalty: torch.Tensor | None = None,
    antisymmetry_penalty_weight: float = 0.0,
) -> torch.Tensor:
    if isinstance(prediction, tuple):
        mean_prediction, log_variance = prediction
        log_variance = log_variance.clamp(-6.0, 6.0)
        residual = mean_prediction - labels
        loss = 0.5 * (torch.exp(-log_variance) * residual.square() + log_variance).mean()
        prediction = mean_prediction
    else:
        loss = huber(prediction, labels)
    if mse_weight:
        loss = loss + mse_weight * F.mse_loss(prediction, labels)
    if rank_weight:
        left, right = torch.triu_indices(len(labels), len(labels), offset=1, device=labels.device)
        same_complex = torch.tensor(
            [groups[sample_ids[i]] == groups[sample_ids[j]] for i, j in zip(left.tolist(), right.tolist())],
            dtype=torch.bool,
            device=labels.device,
        )
        sufficiently_separated = (labels[right] - labels[left]).abs() >= rank_margin
        keep = same_complex & sufficiently_separated
        if keep.any():
            target_order = torch.sign(labels[right[keep]] - labels[left[keep]])
            predicted_difference = prediction[right[keep]] - prediction[left[keep]]
            loss = loss + rank_weight * F.softplus(-target_order * predicted_difference).mean()
    if classification_weight:
        favorable = (labels < 0.0).float()
        loss = loss + classification_weight * F.binary_cross_entropy_with_logits(
            -prediction, favorable
        )
    if antisymmetry_penalty_weight:
        if antisymmetry_penalty is None:
            raise ValueError(
                'antisymmetry_penalty is required when antisymmetry_penalty_weight is nonzero'
            )
        loss = loss + antisymmetry_penalty_weight * antisymmetry_penalty
    return loss


def load_sample_metadata(
    path: Path,
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    labels: dict[str, float] = {}
    groups: dict[str, str] = {}
    mutation_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["sample_id"]] = float(row["ddg"])
            groups[row["sample_id"]] = row["complex_id"]
            mutation_counts[row["sample_id"]] = int(row["n_mutations"])
    return labels, groups, mutation_counts


def load_fold(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["sample_id"]: row["split"] for row in csv.DictReader(handle)}


def load_ragged_site_features(path: Path) -> dict[str, np.ndarray]:
    bundle = np.load(path, allow_pickle=False)
    required = {"sample_ids", "site_x", "site_offsets"}
    missing = required - set(bundle.files)
    if missing:
        raise ValueError(f"{path} lacks site-level arrays: {sorted(missing)}")
    sample_ids = bundle["sample_ids"].astype(str)
    site_x = bundle["site_x"].astype(np.float32)
    offsets = bundle["site_offsets"].astype(np.int64)
    if len(offsets) != len(sample_ids) + 1 or offsets[0] != 0 or offsets[-1] != len(site_x):
        raise ValueError(f"invalid ragged site offsets in {path}")
    features = {
        sample_id: site_x[offsets[index] : offsets[index + 1]]
        for index, sample_id in enumerate(sample_ids)
    }
    if any(len(rows) == 0 for rows in features.values()):
        raise ValueError(f"zero-site sample found in {path}")
    return features


def scale_ragged_site_features(
    features: dict[str, np.ndarray], fit_ids: list[str], available: set[str]
) -> dict[str, np.ndarray]:
    scaler = StandardScaler().fit(np.concatenate([features[sample_id] for sample_id in fit_ids]))
    return {
        sample_id: scaler.transform(features[sample_id]).astype(np.float32)
        for sample_id in available
    }


def evaluate(model, loader, device):
    model.eval()
    truths = []
    predictions = []
    log_variances = []
    sample_ids = []
    with torch.inference_mode():
        for batch, labels, ids in loader:
            output = model(move_batch(batch, device))
            if isinstance(output, tuple):
                prediction, log_variance = output
                log_variances.extend(log_variance.cpu().numpy().tolist())
            else:
                prediction = output
            prediction = prediction.cpu().numpy()
            truths.extend(labels.numpy().tolist())
            predictions.extend(prediction.tolist())
            sample_ids.extend(ids)
    uncertainty = np.asarray(log_variances) if log_variances else None
    return np.asarray(truths), np.asarray(predictions), sample_ids, uncertainty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--esm-features", type=Path)
    parser.add_argument("--local-delta-features", type=Path)
    parser.add_argument("--geometry-features", type=Path)
    parser.add_argument("--pretrained-structure-features", type=Path)
    parser.add_argument("--auxiliary-pretrained-structure-features", type=Path)
    parser.add_argument("--site-esm-features", type=Path)
    parser.add_argument("--site-pretrained-structure-features", type=Path)
    parser.add_argument("--site-cross-attention", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("structure", "gated", "cross", "residual_gated", "residual_cross"),
        required=True,
    )
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min-refit-epochs", type=int, default=8)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--classification-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin", type=float, default=0.5)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument("--complex-balanced", action="store_true")
    parser.add_argument("--mutation-count-balanced", action="store_true")
    parser.add_argument("--antisymmetric", action="store_true")
    parser.add_argument(
        "--antisymmetry-penalty-weight",
        type=float,
        default=0.0,
        help=(
            "Soft raw forward/reverse consistency penalty. This is used only "
            "when --antisymmetric is absent; --antisymmetric remains the exact hard projection."
        ),
    )
    parser.add_argument("--pairwise-epistasis", action="store_true")
    parser.add_argument("--higher-order-epistasis", action="store_true")
    parser.add_argument("--distance-aware-higher-order", action="store_true")
    parser.add_argument("--interface-hierarchical-pooling", action="store_true")
    parser.add_argument("--paired-state-delta", action="store_true")
    parser.add_argument("--uncertainty-head", action="store_true")
    parser.add_argument("--backbone", choices=("egnn", "gvp"), default="egnn")
    parser.add_argument("--seed", type=int, default=20260907)
    parser.add_argument("--num-folds", type=int, default=5)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()

    if args.antisymmetry_penalty_weight < 0:
        raise ValueError("--antisymmetry-penalty-weight must be nonnegative")
    if args.antisymmetric and args.antisymmetry_penalty_weight:
        raise ValueError(
            "--antisymmetric is an exact hard projection and cannot be combined "
            "with --antisymmetry-penalty-weight"
        )
    if args.num_folds < 2:
        raise ValueError("--num-folds must be at least 2")
    if args.fold is not None and not 1 <= args.fold <= args.num_folds:
        raise ValueError(f"--fold must be between 1 and {args.num_folds}")

    if args.graphs.is_dir():
        graph_by_id = {path.stem: path for path in args.graphs.glob("*.pt")}
        manifest_path = args.graphs / "manifest.json"
        graph_errors = 0
        if manifest_path.exists():
            graph_errors = len(json.loads(manifest_path.read_text())["errors"])
    else:
        graph_bundle = torch.load(args.graphs, map_location="cpu")
        graph_by_id = {graph["sample_id"]: graph for graph in graph_bundle["graphs"]}
        graph_errors = len(graph_bundle.get("errors", []))
    if args.complex_balanced and args.mutation_count_balanced:
        raise ValueError("choose only one balanced sampler")
    labels, complex_groups, mutation_counts = load_sample_metadata(args.samples)
    geometry_by_id = None
    if args.geometry_features is not None:
        geometry_by_id, _ = load_dense_feature_bundle(args.geometry_features)
    pretrained_structure_by_id = None
    if args.pretrained_structure_features is not None:
        pretrained_structure_by_id, _ = load_dense_feature_bundle(
            args.pretrained_structure_features
        )
    auxiliary_pretrained_structure_by_id = None
    if args.auxiliary_pretrained_structure_features is not None:
        auxiliary_pretrained_structure_by_id, _ = load_dense_feature_bundle(
            args.auxiliary_pretrained_structure_features
        )
    site_features_by_id = None
    if args.site_cross_attention:
        if args.site_esm_features is None or args.site_pretrained_structure_features is None:
            raise ValueError(
                "site cross-attention requires --site-esm-features and "
                "--site-pretrained-structure-features"
            )
        site_esm_by_id = load_ragged_site_features(args.site_esm_features)
        site_structure_by_id = load_ragged_site_features(
            args.site_pretrained_structure_features
        )
        shared_site_ids = set(site_esm_by_id) & set(site_structure_by_id)
        site_features_by_id = {}
        for sample_id in shared_site_ids:
            if len(site_esm_by_id[sample_id]) != len(site_structure_by_id[sample_id]):
                raise ValueError(f"site count mismatch for {sample_id}")
            site_features_by_id[sample_id] = np.concatenate(
                [site_esm_by_id[sample_id], site_structure_by_id[sample_id]], axis=1
            ).astype(np.float32)
    esm_by_id = None
    scaler_source = None
    if args.mode != "structure":
        if args.esm_features is None:
            raise ValueError("--esm-features is required for gated and cross modes")
        esm_by_id, scaler_source = load_dense_feature_bundle(args.esm_features)
        if args.local_delta_features is not None:
            local_by_id, _ = load_dense_feature_bundle(args.local_delta_features)
            esm_by_id = {
                sample_id: np.concatenate([features, local_by_id[sample_id]])
                for sample_id, features in esm_by_id.items()
                if sample_id in local_by_id
            }

    available = set(graph_by_id)
    if esm_by_id is not None:
        available &= set(esm_by_id)
    if geometry_by_id is not None:
        available &= set(geometry_by_id)
    if pretrained_structure_by_id is not None:
        available &= set(pretrained_structure_by_id)
    if auxiliary_pretrained_structure_by_id is not None:
        available &= set(auxiliary_pretrained_structure_by_id)
    if site_features_by_id is not None:
        available &= set(site_features_by_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_reports = []
    pooled_true = []
    pooled_prediction = []

    folds = [args.fold] if args.fold is not None else list(range(1, args.num_folds + 1))
    for fold in folds:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        split_map = load_fold(args.fold_dir / f"fold_{fold}.csv")
        outer_train = [
            sample_id for sample_id, split in split_map.items()
            if split == "train" and sample_id in available
        ]
        outer_val = [
            sample_id for sample_id, split in split_map.items()
            if split == "val" and sample_id in available
        ]
        group_values = np.asarray([complex_groups[sample_id] for sample_id in outer_train])
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.15, random_state=args.seed + fold
        )
        inner_train_rel, inner_val_rel = next(
            splitter.split(np.arange(len(outer_train)), groups=group_values)
        )
        inner_train = [outer_train[i] for i in inner_train_rel]
        inner_val = [outer_train[i] for i in inner_val_rel]

        fold_esm = esm_by_id
        if esm_by_id is not None:
            train_matrix = np.stack([esm_by_id[sample_id] for sample_id in inner_train])
            scaler = StandardScaler().fit(train_matrix)
            fold_esm = {
                sample_id: scaler.transform(esm_by_id[sample_id][None])[0].astype(np.float32)
                for sample_id in available
            }
        fold_geometry = geometry_by_id
        if geometry_by_id is not None:
            geometry_scaler = StandardScaler().fit(
                np.stack([geometry_by_id[sample_id] for sample_id in inner_train])
            )
            fold_geometry = {
                sample_id: geometry_scaler.transform(
                    geometry_by_id[sample_id][None]
                )[0].astype(np.float32)
                for sample_id in available
            }
        fold_pretrained_structure = pretrained_structure_by_id
        if pretrained_structure_by_id is not None:
            pretrained_scaler = StandardScaler().fit(
                np.stack(
                    [pretrained_structure_by_id[sample_id] for sample_id in inner_train]
                )
            )
            fold_pretrained_structure = {
                sample_id: pretrained_scaler.transform(
                    pretrained_structure_by_id[sample_id][None]
                )[0].astype(np.float32)
                for sample_id in available
            }
        fold_auxiliary_pretrained_structure = auxiliary_pretrained_structure_by_id
        if auxiliary_pretrained_structure_by_id is not None:
            auxiliary_scaler = StandardScaler().fit(
                np.stack(
                    [auxiliary_pretrained_structure_by_id[sample_id] for sample_id in inner_train]
                )
            )
            fold_auxiliary_pretrained_structure = {
                sample_id: auxiliary_scaler.transform(
                    auxiliary_pretrained_structure_by_id[sample_id][None]
                )[0].astype(np.float32)
                for sample_id in available
            }
        fold_site_features = site_features_by_id
        if site_features_by_id is not None:
            fold_site_features = scale_ragged_site_features(
                site_features_by_id, inner_train, available
            )
        train_sampler = None
        if args.complex_balanced:
            train_sampler = balanced_sampler(inner_train, complex_groups)
        elif args.mutation_count_balanced:
            train_sampler = balanced_sampler(inner_train, mutation_counts, exponent=0.5)
        train_loader = DataLoader(
            GraphDataset(
                inner_train, graph_by_id, labels, fold_esm, fold_geometry,
                fold_pretrained_structure, fold_auxiliary_pretrained_structure,
                fold_site_features,
            ),
            batch_size=args.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=collate_graphs,
        )
        inner_val_loader = DataLoader(
            GraphDataset(
                inner_val, graph_by_id, labels, fold_esm, fold_geometry,
                fold_pretrained_structure, fold_auxiliary_pretrained_structure,
                fold_site_features,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )
        outer_val_loader = DataLoader(
            GraphDataset(
                outer_val, graph_by_id, labels, fold_esm, fold_geometry,
                fold_pretrained_structure, fold_auxiliary_pretrained_structure,
                fold_site_features,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )

        set_seed(args.seed + fold)
        model = DeltaGModel(
            mode=args.mode,
            hidden=args.hidden,
            layers=args.layers,
            dropout=args.dropout,
            sequence_feature_dim=(
                None if fold_esm is None else len(next(iter(fold_esm.values())))
            ),
            geometry_dim=(
                0 if fold_geometry is None else len(next(iter(fold_geometry.values())))
            ),
            pretrained_structure_dim=(
                0
                if fold_pretrained_structure is None
                else len(next(iter(fold_pretrained_structure.values())))
            ),
            auxiliary_pretrained_structure_dim=(
                0
                if fold_auxiliary_pretrained_structure is None
                else len(next(iter(fold_auxiliary_pretrained_structure.values())))
            ),
            site_feature_dim=(
                0 if fold_site_features is None else
                next(iter(fold_site_features.values())).shape[1]
            ),
            site_cross_attention=args.site_cross_attention,
            antisymmetric=args.antisymmetric,
            pairwise_epistasis=args.pairwise_epistasis,
            higher_order_epistasis=args.higher_order_epistasis,
            distance_aware_higher_order=args.distance_aware_higher_order,
            interface_hierarchical_pooling=args.interface_hierarchical_pooling,
            paired_state_delta=args.paired_state_delta,
            uncertainty_head=args.uncertainty_head,
            backbone=args.backbone,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        loss_function = nn.HuberLoss(delta=1.0)
        best_state = None
        best_loss = float("inf")
        best_epoch = 0
        stale = 0

        for epoch in range(1, args.max_epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for step, (batch, batch_labels, batch_ids) in enumerate(train_loader, start=1):
                batch = move_batch(batch, device)
                batch_labels = batch_labels.to(device)
                prediction = model(batch)
                antisymmetry_penalty = (
                    model.antisymmetry_penalty(batch)
                    if args.antisymmetry_penalty_weight
                    else None
                )
                loss = composite_loss(
                    prediction, batch_labels, batch_ids, complex_groups, loss_function,
                    args.rank_weight, args.classification_weight, args.rank_margin,
                    args.mse_weight, antisymmetry_penalty,
                    args.antisymmetry_penalty_weight,
                )
                (loss / args.accumulation_steps).backward()
                if step % args.accumulation_steps == 0 or step == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            inner_true, inner_prediction, _, _ = evaluate(model, inner_val_loader, device)
            inner_loss = float(np.mean(np.abs(inner_prediction - inner_true)))
            if inner_loss < best_loss - 1e-4:
                best_loss = inner_loss
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
            if stale >= args.patience:
                break
        if best_state is None:
            raise RuntimeError("no best model state was recorded")
        # Refit from scratch on the complete outer-training fold.  The inner split
        # selects the epoch count only; it must not permanently discard 15% of the
        # training complexes.
        refit_epochs = max(best_epoch, args.min_refit_epochs)
        refit_esm = esm_by_id
        if esm_by_id is not None:
            refit_scaler = StandardScaler().fit(
                np.stack([esm_by_id[sample_id] for sample_id in outer_train])
            )
            refit_esm = {
                sample_id: refit_scaler.transform(esm_by_id[sample_id][None])[0].astype(np.float32)
                for sample_id in available
            }
        refit_geometry = geometry_by_id
        if geometry_by_id is not None:
            refit_geometry_scaler = StandardScaler().fit(
                np.stack([geometry_by_id[sample_id] for sample_id in outer_train])
            )
            refit_geometry = {
                sample_id: refit_geometry_scaler.transform(
                    geometry_by_id[sample_id][None]
                )[0].astype(np.float32)
                for sample_id in available
            }
        refit_pretrained_structure = pretrained_structure_by_id
        if pretrained_structure_by_id is not None:
            refit_pretrained_scaler = StandardScaler().fit(
                np.stack(
                    [pretrained_structure_by_id[sample_id] for sample_id in outer_train]
                )
            )
            refit_pretrained_structure = {
                sample_id: refit_pretrained_scaler.transform(
                    pretrained_structure_by_id[sample_id][None]
                )[0].astype(np.float32)
                for sample_id in available
            }
        refit_auxiliary_pretrained_structure = auxiliary_pretrained_structure_by_id
        if auxiliary_pretrained_structure_by_id is not None:
            refit_auxiliary_scaler = StandardScaler().fit(
                np.stack(
                    [auxiliary_pretrained_structure_by_id[sample_id] for sample_id in outer_train]
                )
            )
            refit_auxiliary_pretrained_structure = {
                sample_id: refit_auxiliary_scaler.transform(
                    auxiliary_pretrained_structure_by_id[sample_id][None]
                )[0].astype(np.float32)
                for sample_id in available
            }
        refit_site_features = site_features_by_id
        if site_features_by_id is not None:
            refit_site_features = scale_ragged_site_features(
                site_features_by_id, outer_train, available
            )
        refit_sampler = None
        if args.complex_balanced:
            refit_sampler = balanced_sampler(outer_train, complex_groups)
        elif args.mutation_count_balanced:
            refit_sampler = balanced_sampler(outer_train, mutation_counts, exponent=0.5)
        refit_loader = DataLoader(
            GraphDataset(
                outer_train, graph_by_id, labels, refit_esm, refit_geometry,
                refit_pretrained_structure, refit_auxiliary_pretrained_structure,
                refit_site_features,
            ),
            batch_size=args.batch_size,
            shuffle=refit_sampler is None,
            sampler=refit_sampler,
            collate_fn=collate_graphs,
        )
        outer_val_loader = DataLoader(
            GraphDataset(
                outer_val, graph_by_id, labels, refit_esm, refit_geometry,
                refit_pretrained_structure, refit_auxiliary_pretrained_structure,
                refit_site_features,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_graphs,
        )
        set_seed(args.seed + 1000 + fold)
        model = DeltaGModel(
            mode=args.mode,
            hidden=args.hidden,
            layers=args.layers,
            dropout=args.dropout,
            sequence_feature_dim=(
                None if refit_esm is None else len(next(iter(refit_esm.values())))
            ),
            geometry_dim=(
                0 if refit_geometry is None else len(next(iter(refit_geometry.values())))
            ),
            pretrained_structure_dim=(
                0
                if refit_pretrained_structure is None
                else len(next(iter(refit_pretrained_structure.values())))
            ),
            auxiliary_pretrained_structure_dim=(
                0
                if refit_auxiliary_pretrained_structure is None
                else len(next(iter(refit_auxiliary_pretrained_structure.values())))
            ),
            site_feature_dim=(
                0 if refit_site_features is None else
                next(iter(refit_site_features.values())).shape[1]
            ),
            site_cross_attention=args.site_cross_attention,
            antisymmetric=args.antisymmetric,
            pairwise_epistasis=args.pairwise_epistasis,
            higher_order_epistasis=args.higher_order_epistasis,
            distance_aware_higher_order=args.distance_aware_higher_order,
            interface_hierarchical_pooling=args.interface_hierarchical_pooling,
            paired_state_delta=args.paired_state_delta,
            uncertainty_head=args.uncertainty_head,
            backbone=args.backbone,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
        )
        for _ in range(refit_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for step, (batch, batch_labels, batch_ids) in enumerate(refit_loader, start=1):
                batch = move_batch(batch, device)
                batch_labels = batch_labels.to(device)
                prediction = model(batch)
                antisymmetry_penalty = (
                    model.antisymmetry_penalty(batch)
                    if args.antisymmetry_penalty_weight
                    else None
                )
                loss = composite_loss(
                    prediction, batch_labels, batch_ids, complex_groups, loss_function,
                    args.rank_weight, args.classification_weight, args.rank_margin,
                    args.mse_weight, antisymmetry_penalty,
                    args.antisymmetry_penalty_weight,
                )
                (loss / args.accumulation_steps).backward()
                if step % args.accumulation_steps == 0 or step == len(refit_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
        truth, prediction, prediction_ids, log_variance = evaluate(model, outer_val_loader, device)
        metrics = regression_metrics(truth, prediction)
        metrics.update(
            {
                "fold": fold,
                "outer_train_n": len(outer_train),
                "inner_train_n": len(inner_train),
                "inner_val_n": len(inner_val),
                "best_epoch": best_epoch,
                "refit_epochs": refit_epochs,
            }
        )
        if device.type == "cuda":
            metrics["cuda_peak_allocated_gib"] = float(
                torch.cuda.max_memory_allocated(device) / 1024**3
            )
            metrics["cuda_peak_reserved_gib"] = float(
                torch.cuda.max_memory_reserved(device) / 1024**3
            )
        fold_reports.append(metrics)
        pooled_true.extend(truth.tolist())
        pooled_prediction.extend(prediction.tolist())
        print(json.dumps(metrics), flush=True)

        fold_output = args.output_dir / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, fold_output / "model.pt")
        with (fold_output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sample_id", "fold", "ddg_true", "ddg_pred"])
            for sample_id, y_true, y_pred in zip(prediction_ids, truth, prediction):
                writer.writerow([sample_id, fold, f"{y_true:.8g}", f"{y_pred:.8g}"])
        if log_variance is not None:
            with (fold_output / "uncertainty.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["sample_id", "fold", "log_variance"])
                for sample_id, value in zip(prediction_ids, log_variance):
                    writer.writerow([sample_id, fold, f"{value:.8g}"])

    metric_names = ("pearson", "spearman", "rmse", "mae", "auroc")
    report = {
        "mode": args.mode,
        "configuration": {key: str(value) for key, value in vars(args).items()},
        "graph_errors": graph_errors,
        "fold_metrics": fold_reports,
        "fold_mean_std": {
            name: {
                "mean": float(np.mean([row[name] for row in fold_reports])),
                "std": (
                    float(np.std([row[name] for row in fold_reports], ddof=1))
                    if len(fold_reports) > 1
                    else None
                ),
            }
            for name in metric_names
        },
        "pooled_validation": regression_metrics(
            np.asarray(pooled_true), np.asarray(pooled_prediction)
        ),
    }
    metrics_name = "metrics.json" if args.fold is None else f"metrics_fold_{args.fold}.json"
    (args.output_dir / metrics_name).write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
