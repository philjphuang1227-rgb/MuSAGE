"""Independent EGNN and mutation-centered multimodal fusion models."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def scatter_mean(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    output = values.new_zeros((size, values.shape[-1]))
    count = values.new_zeros((size, 1))
    output.index_add_(0, index, values)
    count.index_add_(0, index, values.new_ones((len(values), 1)))
    return output / count.clamp_min(1.0)


class EGNNLayer(nn.Module):
    def __init__(self, hidden: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim + 1, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.coordinate_gate = nn.Sequential(nn.Linear(hidden, 1), nn.Tanh())
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, hidden)
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        hidden: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        displacement = coordinates[source] - coordinates[target]
        squared_distance = (displacement * displacement).sum(dim=-1, keepdim=True)
        messages = self.message(
            torch.cat([hidden[source], hidden[target], edge_features, squared_distance], dim=-1)
        )
        aggregate = hidden.new_zeros(hidden.shape)
        degree = hidden.new_zeros((len(hidden), 1))
        aggregate.index_add_(0, target, messages)
        degree.index_add_(0, target, hidden.new_ones((len(target), 1)))
        aggregate = aggregate / degree.clamp_min(1.0)
        hidden = self.norm(hidden + self.update(torch.cat([hidden, aggregate], dim=-1)))

        normalized = displacement / squared_distance.sqrt().clamp_min(1.0)
        coordinate_message = normalized * self.coordinate_gate(messages)
        coordinate_update = coordinates.new_zeros(coordinates.shape)
        coordinate_update.index_add_(0, target, coordinate_message)
        coordinates = coordinates + 0.1 * coordinate_update / degree.clamp_min(1.0)
        return hidden, coordinates


class GraphEncoder(nn.Module):
    def __init__(
        self, node_dim: int = 45, edge_dim: int = 19, hidden: int = 128,
        layers: int = 4, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.input = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [EGNNLayer(hidden, edge_dim, dropout) for _ in range(layers)]
        )
        self.pool = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden))

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.input(batch["node_features"])
        coordinates = batch["coordinates"]
        for layer in self.layers:
            hidden, coordinates = layer(
                hidden, coordinates, batch["edge_index"], batch["edge_features"]
            )
        graph_count = int(batch["graph_index"].max().item()) + 1
        graph_mean = scatter_mean(hidden, batch["graph_index"], graph_count)
        mutation_nodes = batch["mutation_mask"] > 0.5
        mutation_mean = scatter_mean(
            hidden[mutation_nodes], batch["graph_index"][mutation_nodes], graph_count
        )
        return self.pool(torch.cat([mutation_mean, graph_mean], dim=-1)), hidden


class GVPStyleLayer(nn.Module):
    """Rotation-equivariant scalar/vector message passing for a GVP ablation."""

    def __init__(
        self, hidden: int, edge_dim: int, vector_dim: int, dropout: float
    ) -> None:
        super().__init__()
        self.scalar_message = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim + 1 + vector_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.direction_gate = nn.Linear(hidden, vector_dim)
        self.vector_message = nn.Linear(vector_dim, vector_dim, bias=False)
        self.scalar_update = nn.Sequential(
            nn.Linear(2 * hidden + vector_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.vector_update = nn.Linear(vector_dim, vector_dim, bias=False)
        self.vector_gate = nn.Sequential(nn.Linear(hidden, vector_dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(hidden)

    def forward(
        self,
        scalar: torch.Tensor,
        vector: torch.Tensor,
        coordinates: torch.Tensor,
        edge_index: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source, target = edge_index
        displacement = coordinates[source] - coordinates[target]
        distance = displacement.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        direction = displacement / distance
        source_vector_norm = vector[source].norm(dim=-1)
        scalar_message = self.scalar_message(
            torch.cat(
                [
                    scalar[source],
                    scalar[target],
                    edge_features,
                    distance.square(),
                    source_vector_norm,
                ],
                dim=-1,
            )
        )
        rotated_source = self.vector_message(vector[source].transpose(1, 2)).transpose(1, 2)
        vector_message = rotated_source + self.direction_gate(scalar_message).unsqueeze(-1) * direction.unsqueeze(1)

        scalar_aggregate = scalar.new_zeros(scalar.shape)
        vector_aggregate = vector.new_zeros(vector.shape)
        degree = scalar.new_zeros((len(scalar), 1))
        scalar_aggregate.index_add_(0, target, scalar_message)
        vector_aggregate.index_add_(0, target, vector_message)
        degree.index_add_(0, target, scalar.new_ones((len(target), 1)))
        scalar_aggregate = scalar_aggregate / degree.clamp_min(1.0)
        vector_aggregate = vector_aggregate / degree.clamp_min(1.0).unsqueeze(-1)

        scalar = self.norm(
            scalar
            + self.scalar_update(
                torch.cat(
                    [scalar, scalar_aggregate, vector_aggregate.norm(dim=-1)], dim=-1
                )
            )
        )
        vector_candidate = self.vector_update(
            vector_aggregate.transpose(1, 2)
        ).transpose(1, 2)
        vector = vector + self.vector_gate(scalar).unsqueeze(-1) * vector_candidate
        return scalar, vector


class GVPStyleGraphEncoder(nn.Module):
    """Compact GVP-style backbone with invariant scalar readout."""

    def __init__(
        self,
        node_dim: int = 45,
        edge_dim: int = 19,
        hidden: int = 128,
        layers: int = 4,
        vector_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vector_dim = vector_dim
        self.input = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [
                GVPStyleLayer(hidden, edge_dim, vector_dim, dropout)
                for _ in range(layers)
            ]
        )
        self.vector_readout = nn.Linear(vector_dim, hidden)
        self.node_norm = nn.LayerNorm(hidden)
        self.pool = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        scalar = self.input(batch["node_features"])
        vector = scalar.new_zeros((len(scalar), self.vector_dim, 3))
        for layer in self.layers:
            scalar, vector = layer(
                scalar,
                vector,
                batch["coordinates"],
                batch["edge_index"],
                batch["edge_features"],
            )
        hidden = self.node_norm(scalar + self.vector_readout(vector.norm(dim=-1)))
        graph_count = int(batch["graph_index"].max().item()) + 1
        graph_mean = scatter_mean(hidden, batch["graph_index"], graph_count)
        mutation_nodes = batch["mutation_mask"] > 0.5
        mutation_mean = scatter_mean(
            hidden[mutation_nodes], batch["graph_index"][mutation_nodes], graph_count
        )
        return self.pool(torch.cat([mutation_mean, graph_mean], dim=-1)), hidden


class DeltaGModel(nn.Module):
    def __init__(
        self,
        mode: str,
        esm_dim: int = 1280,
        hidden: int = 128,
        layers: int = 4,
        dropout: float = 0.1,
        heads: int = 4,
        sequence_feature_dim: int | None = None,
        geometry_dim: int = 0,
        pretrained_structure_dim: int = 0,
        auxiliary_pretrained_structure_dim: int = 0,
        site_feature_dim: int = 0,
        site_cross_attention: bool = False,
        antisymmetric: bool = False,
        pairwise_epistasis: bool = False,
        higher_order_epistasis: bool = False,
        distance_aware_higher_order: bool = False,
        interface_hierarchical_pooling: bool = False,
        paired_state_delta: bool = False,
        uncertainty_head: bool = False,
        backbone: str = "egnn",
    ) -> None:
        super().__init__()
        if mode not in {"structure", "gated", "cross", "residual_gated", "residual_cross"}:
            raise ValueError(f"unsupported mode: {mode}")
        self.mode = mode
        self.antisymmetric = antisymmetric
        self.pairwise_epistasis = pairwise_epistasis
        self.higher_order_epistasis = higher_order_epistasis
        self.distance_aware_higher_order = distance_aware_higher_order
        self.interface_hierarchical_pooling = interface_hierarchical_pooling
        self.paired_state_delta = paired_state_delta
        self.uncertainty_head_enabled = uncertainty_head
        self.site_cross_attention = site_cross_attention
        self.esm_dim = esm_dim
        if site_cross_attention and not site_feature_dim:
            raise ValueError("site_feature_dim is required for site cross-attention")
        if distance_aware_higher_order and not higher_order_epistasis:
            raise ValueError(
                "distance-aware higher-order attention requires higher_order_epistasis"
            )
        if backbone not in {"egnn", "gvp"}:
            raise ValueError(f"unsupported backbone: {backbone}")
        sequence_feature_dim = sequence_feature_dim or 3 * esm_dim
        self.graph_encoder = (
            GraphEncoder(hidden=hidden, layers=layers, dropout=dropout)
            if backbone == "egnn"
            else GVPStyleGraphEncoder(hidden=hidden, layers=layers, dropout=dropout)
        )
        self.geometry_projection = (
            nn.Sequential(nn.Linear(geometry_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden))
            if geometry_dim
            else None
        )
        self.pretrained_structure_projection = (
            nn.Sequential(
                nn.Linear(pretrained_structure_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
            if pretrained_structure_dim
            else None
        )
        self.auxiliary_pretrained_structure_projection = (
            nn.Sequential(
                nn.Linear(auxiliary_pretrained_structure_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
            if auxiliary_pretrained_structure_dim
            else None
        )
        self.auxiliary_structure_logit = (
            nn.Parameter(torch.tensor(-2.0))
            if auxiliary_pretrained_structure_dim
            else None
        )
        self.site_token_projection = (
            nn.Sequential(
                nn.Linear(site_feature_dim, hidden),
                nn.SiLU(),
                nn.LayerNorm(hidden),
            )
            if site_cross_attention
            else None
        )
        self.site_to_nodes = (
            nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            if site_cross_attention
            else None
        )
        self.site_context_projection = (
            nn.Sequential(
                nn.Linear(2 * hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            if site_cross_attention
            else None
        )
        self.site_attention_logit = (
            nn.Parameter(torch.tensor(-2.0)) if site_cross_attention else None
        )
        self.pairwise_projection = (
            nn.Sequential(
                nn.Linear(2 * hidden + 1, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            if pairwise_epistasis
            else None
        )
        self.higher_order_set_attention = (
            nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
            if higher_order_epistasis
            else None
        )
        self.higher_order_set_projection = (
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            if higher_order_epistasis
            else None
        )
        self.higher_order_set_logit = (
            nn.Parameter(torch.tensor(-2.0)) if higher_order_epistasis else None
        )
        self.higher_order_distance_log_scale = (
            nn.Parameter(torch.tensor(math.log(8.0)))
            if distance_aware_higher_order
            else None
        )
        self.interface_projection = (
            nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            if interface_hierarchical_pooling
            else None
        )
        self.interface_logit = (
            nn.Parameter(torch.tensor(-2.0))
            if interface_hierarchical_pooling
            else None
        )
        self.paired_delta_projection = (
            nn.Sequential(
                nn.Linear(21, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            if paired_state_delta
            else None
        )
        self.paired_state_projection = (
            nn.Sequential(
                nn.Linear(3 * hidden, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.LayerNorm(hidden),
            )
            if paired_state_delta
            else None
        )
        self.paired_state_logit = (
            nn.Parameter(torch.tensor(-2.0)) if paired_state_delta else None
        )
        if mode in {"gated", "residual_gated"}:
            self.sequence_projection = nn.Sequential(
                nn.Linear(sequence_feature_dim, hidden), nn.SiLU(), nn.LayerNorm(hidden)
            )
            self.gate = nn.Linear(2 * hidden, hidden)
        elif mode in {"cross", "residual_cross"}:
            self.token_projection = nn.Linear(esm_dim, hidden)
            self.sequence_to_structure = nn.MultiheadAttention(
                hidden, heads, dropout=dropout, batch_first=True
            )
            self.structure_to_sequence = nn.MultiheadAttention(
                hidden, heads, dropout=dropout, batch_first=True
            )
            self.cross_projection = nn.Sequential(
                nn.Linear(3 * hidden, hidden), nn.SiLU(), nn.LayerNorm(hidden)
            )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden // 2, 1)
        )
        self.uncertainty_head = (
            nn.Sequential(
                nn.Linear(hidden, hidden // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, 1),
            )
            if uncertainty_head
            else None
        )
        if mode.startswith("residual_"):
            self.sequence_head = nn.Sequential(
                nn.Linear(sequence_feature_dim, hidden),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.SiLU(),
                nn.Linear(hidden // 2, 1),
            )
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def _forward_raw(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        structure, node_hidden = self.graph_encoder(batch)
        paired_context = None
        if self.paired_delta_projection is not None:
            # Encode the signed WT->MT substitution at mutation nodes and fuse it
            # with the shared EGNN state. This avoids a second full graph tower,
            # which is essential for the 24-GB M1707 GPUs. A true mutant-coordinate
            # tower remains a future extension once MT coordinate caches exist.
            mutation_nodes = batch["mutation_mask"] > 0.5
            graph_count = int(batch["graph_index"].max().item()) + 1
            wt_context = scatter_mean(
                node_hidden[mutation_nodes],
                batch["graph_index"][mutation_nodes],
                graph_count,
            )
            substitution = batch["node_features"][:, 24:45]
            delta_nodes = self.paired_delta_projection(substitution)
            paired_delta = scatter_mean(
                delta_nodes[mutation_nodes],
                batch["graph_index"][mutation_nodes],
                graph_count,
            )
            mt_context = wt_context + paired_delta
            paired_context = self.paired_state_projection(
                torch.cat([wt_context, mt_context, paired_delta], dim=-1)
            )
        if self.geometry_projection is not None:
            structure = structure + self.geometry_projection(batch["geometry_features"])
        if self.pretrained_structure_projection is not None:
            structure = structure + self.pretrained_structure_projection(
                batch["pretrained_structure_features"]
            )
        if self.auxiliary_pretrained_structure_projection is not None:
            auxiliary_structure = self.auxiliary_pretrained_structure_projection(
                batch["auxiliary_pretrained_structure_features"]
            )
            structure = structure + torch.sigmoid(
                self.auxiliary_structure_logit
            ) * auxiliary_structure
        if paired_context is not None:
            structure = structure + torch.sigmoid(self.paired_state_logit) * paired_context
        if self.site_token_projection is not None:
            site_tokens = self.site_token_projection(batch["site_features"])
            site_rows = []
            for graph_number in range(len(structure)):
                valid_sites = batch["site_mask"][graph_number]
                queries = site_tokens[graph_number, valid_sites].unsqueeze(0)
                node_mask = batch["graph_index"] == graph_number
                nodes = node_hidden[node_mask].unsqueeze(0)
                attended_nodes, _ = self.site_to_nodes(queries, nodes, nodes)
                fused_sites = self.site_context_projection(
                    torch.cat([queries, attended_nodes], dim=-1)
                )
                site_rows.append(fused_sites.mean(dim=1).squeeze(0))
            site_context = torch.stack(site_rows)
            structure = structure + torch.sigmoid(self.site_attention_logit) * site_context
        if self.pairwise_projection is not None:
            pairwise_rows = []
            for graph_number in range(len(structure)):
                mutation_indexes = torch.nonzero(
                    (batch["graph_index"] == graph_number)
                    & (batch["mutation_mask"] > 0.5),
                    as_tuple=False,
                ).flatten()
                if len(mutation_indexes) < 2:
                    pairwise_rows.append(structure.new_zeros(structure.shape[-1]))
                    continue
                left, right = torch.triu_indices(
                    len(mutation_indexes), len(mutation_indexes), offset=1,
                    device=node_hidden.device,
                )
                left_indexes = mutation_indexes[left]
                right_indexes = mutation_indexes[right]
                left_hidden = node_hidden[left_indexes]
                right_hidden = node_hidden[right_indexes]
                squared_distance = (
                    batch["coordinates"][left_indexes]
                    - batch["coordinates"][right_indexes]
                ).square().sum(dim=-1, keepdim=True)
                pair_features = torch.cat(
                    [
                        left_hidden + right_hidden,
                        (left_hidden - right_hidden).abs(),
                        squared_distance,
                    ],
                    dim=-1,
                )
                pairwise_rows.append(self.pairwise_projection(pair_features).mean(dim=0))
            structure = structure + torch.stack(pairwise_rows)
        if self.higher_order_set_attention is not None:
            set_rows = []
            for graph_number in range(len(structure)):
                mutation_indexes = torch.nonzero(
                    (batch["graph_index"] == graph_number)
                    & (batch["mutation_mask"] > 0.5),
                    as_tuple=False,
                ).flatten()
                if len(mutation_indexes) < 2:
                    set_rows.append(structure.new_zeros(structure.shape[-1]))
                    continue
                mutation_tokens = node_hidden[mutation_indexes].unsqueeze(0)
                attention_mask = None
                if self.higher_order_distance_log_scale is not None:
                    mutation_coordinates = batch["coordinates"][mutation_indexes]
                    mutation_distances = torch.cdist(
                        mutation_coordinates, mutation_coordinates
                    )
                    distance_scale = self.higher_order_distance_log_scale.exp().clamp(
                        min=1.0, max=50.0
                    )
                    attention_mask = -mutation_distances / distance_scale
                attended, _ = self.higher_order_set_attention(
                    mutation_tokens,
                    mutation_tokens,
                    mutation_tokens,
                    attn_mask=attention_mask,
                )
                set_context = attended.mean(dim=1).squeeze(0)
                set_rows.append(self.higher_order_set_projection(set_context))
            structure = structure + torch.sigmoid(self.higher_order_set_logit) * torch.stack(set_rows)
        if self.interface_projection is not None:
            interface_rows = []
            source, target = batch["edge_index"]
            # Edge feature 17 indicates that two residues belong to the same
            # binding partner. Cross-partner edges therefore identify the
            # local antibody-antigen interface without an additional cache.
            cross_partner = batch["edge_features"][:, 17] < 0.5
            for graph_number in range(len(structure)):
                graph_edges = cross_partner & (
                    batch["graph_index"][source] == graph_number
                )
                if not graph_edges.any():
                    interface_rows.append(structure.new_zeros(structure.shape[-1]))
                    continue
                interface_indexes = torch.unique(
                    torch.cat([source[graph_edges], target[graph_edges]])
                )
                partner_a = interface_indexes[
                    batch["node_features"][interface_indexes, 21] > 0.5
                ]
                partner_b = interface_indexes[
                    batch["node_features"][interface_indexes, 22] > 0.5
                ]
                if len(partner_a) == 0 or len(partner_b) == 0:
                    interface_rows.append(structure.new_zeros(structure.shape[-1]))
                    continue
                context_a = node_hidden[partner_a].mean(dim=0)
                context_b = node_hidden[partner_b].mean(dim=0)
                symmetric_context = torch.cat(
                    [
                        context_a + context_b,
                        (context_a - context_b).abs(),
                        context_a * context_b,
                    ],
                    dim=-1,
                )
                interface_rows.append(self.interface_projection(symmetric_context))
            structure = structure + torch.sigmoid(self.interface_logit) * torch.stack(
                interface_rows
            )
        fused = structure
        if self.mode in {"gated", "residual_gated"}:
            sequence = self.sequence_projection(batch["esm_features"])
            gate = torch.sigmoid(self.gate(torch.cat([sequence, structure], dim=-1)))
            fused = gate * sequence + (1.0 - gate) * structure
        elif self.mode in {"cross", "residual_cross"}:
            sequence_tokens = self.token_projection(
                batch["esm_features"].reshape(len(structure), -1, 1280)
            )
            cross_rows = []
            for graph_number in range(len(structure)):
                node_mask = batch["graph_index"] == graph_number
                nodes = node_hidden[node_mask].unsqueeze(0)
                tokens = sequence_tokens[graph_number].unsqueeze(0)
                sequence_context, _ = self.sequence_to_structure(tokens, nodes, nodes)
                structure_query = structure[graph_number].reshape(1, 1, -1)
                structure_context, _ = self.structure_to_sequence(
                    structure_query, tokens, tokens
                )
                cross_rows.append(
                    torch.cat(
                        [
                            structure[graph_number],
                            sequence_context.mean(dim=1).squeeze(0),
                            structure_context.squeeze(0).squeeze(0),
                        ],
                        dim=-1,
                    )
                )
            fused = self.cross_projection(torch.stack(cross_rows))
        correction = self.head(fused).squeeze(-1)
        if self.mode.startswith("residual_"):
            sequence_baseline = self.sequence_head(batch["esm_features"]).squeeze(-1)
            mean_prediction = sequence_baseline + correction
        else:
            mean_prediction = correction
        if self.uncertainty_head is not None:
            log_variance = self.uncertainty_head(fused).squeeze(-1).clamp(-6.0, 6.0)
            return mean_prediction, log_variance
        return mean_prediction

    def _make_reverse_batch(
        self,
        batch: dict[str, torch.Tensor],
        prediction_batch_size: int,
    ) -> dict[str, torch.Tensor]:
        """Return the WT/MT-swapped representation used for antisymmetry checks."""
        reverse_batch = dict(batch)
        reverse_nodes = batch["node_features"].clone()
        reverse_nodes[:, 24:45] = -reverse_nodes[:, 24:45]
        reverse_batch["node_features"] = reverse_nodes
        if "esm_features" in batch:
            reverse_tokens = batch["esm_features"].reshape(
                prediction_batch_size, -1, 1280
            ).clone()
            wild_type = reverse_tokens[:, 0].clone()
            reverse_tokens[:, 0] = reverse_tokens[:, 1]
            reverse_tokens[:, 1] = wild_type
            reverse_tokens[:, 2:] = -reverse_tokens[:, 2:]
            reverse_batch["esm_features"] = reverse_tokens.flatten(start_dim=1)
        if "site_features" in batch:
            reverse_sites = batch["site_features"].clone()
            wild_type = reverse_sites[..., : self.esm_dim].clone()
            reverse_sites[..., : self.esm_dim] = reverse_sites[
                ..., self.esm_dim : 2 * self.esm_dim
            ]
            reverse_sites[..., self.esm_dim : 2 * self.esm_dim] = wild_type
            reverse_sites[..., 2 * self.esm_dim : 3 * self.esm_dim] *= -1
            reverse_batch["site_features"] = reverse_sites
        return reverse_batch

    @staticmethod
    def _mean_prediction(
        prediction: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        return prediction[0] if isinstance(prediction, tuple) else prediction

    def antisymmetry_penalty(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Squared WT/MT reversal violation for the optional soft-constraint study."""
        if self.antisymmetric:
            raise ValueError(
                "soft antisymmetry penalty is incompatible with exact antisymmetric projection"
            )
        prediction = self._forward_raw(batch)
        mean_prediction = self._mean_prediction(prediction)
        reverse_prediction = self._forward_raw(
            self._make_reverse_batch(batch, len(mean_prediction))
        )
        reverse_mean = self._mean_prediction(reverse_prediction)
        return (mean_prediction + reverse_mean).square().mean()

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        checkpointed = self.training and self.paired_delta_projection is not None
        if checkpointed:
            # The dummy parameter gives checkpoint a grad-requiring input while
            # the batch remains a dictionary of non-grad tensors.
            dummy = next(self.parameters())
            prediction = checkpoint(
                lambda _dummy: self._forward_raw(batch),
                dummy,
            )
        else:
            prediction = self._forward_raw(batch)
        if not self.antisymmetric:
            return prediction
        prediction_batch_size = prediction[0].shape[0] if isinstance(prediction, tuple) else len(prediction)
        reverse_batch = self._make_reverse_batch(batch, prediction_batch_size)
        if checkpointed:
            reverse_prediction = checkpoint(
                lambda _dummy: self._forward_raw(reverse_batch),
                dummy,
            )
        else:
            reverse_prediction = self._forward_raw(reverse_batch)
        if isinstance(prediction, tuple):
            mean_prediction, log_variance = prediction
            reverse_mean, reverse_log_variance = reverse_prediction
            return (
                0.5 * (mean_prediction - reverse_mean),
                0.5 * (log_variance + reverse_log_variance),
            )
        return 0.5 * (prediction - reverse_prediction)
