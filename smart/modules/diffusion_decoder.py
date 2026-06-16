import torch
import torch.nn as nn
from typing import Optional

from smart.layers.attention_layer import AttentionLayer
from smart.layers.fourier_embedding import FourierEmbedding
from smart.modules.smart_edge_builder import (
    build_radius_interaction_edges,
    build_radius_map2agent_edges,
    build_temporal_edges_from_flat,
    empty_edge_index,
)


class DiffusionDecoder(nn.Module):
    """Graph attention decoder that denoises future trajectory tokens.

    Takes noisy future token IDs for all agents across all future chunks
    (packed into per-scene sequences), plus scene and map context.
    Future-token and map-to-future attention use SMART-style explicit graph
    edges with edge-relative geometry.

    Raw world coordinates are used only for radius graph construction and
    are converted to relative edge features before embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        token_size: int,
        num_future_chunks: int,
        num_heads: int,
        head_dim: int,
        dropout: float,
        num_freq_bands: int,
        a2a_radius: float,
        pl2a_radius: float,
        time_span: Optional[int],
        future_chunk_steps: int,
        num_layers: int = 2,
        num_token_types: int = 1,
        use_agent_context: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.token_size = token_size
        self.num_future_chunks = num_future_chunks
        self.use_agent_context = use_agent_context
        self.a2a_radius = a2a_radius
        self.pl2a_radius = pl2a_radius
        self.time_span = time_span
        self.future_chunk_steps = max(1, int(future_chunk_steps))
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.mask_token_id = token_size
        self.token_embedding = nn.Embedding(token_size, hidden_dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.r_temporal_embedding = FourierEmbedding(
            input_dim=4,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_a2a_embedding = FourierEmbedding(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.r_map2token_embedding = FourierEmbedding(
            input_dim=3,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.chunk_embedding = nn.Embedding(num_future_chunks, hidden_dim)
        if num_token_types > 1:
            self.type_embedding = nn.Embedding(num_token_types, hidden_dim)
        else:
            self.type_embedding = None

        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.map_context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.agent_context_projection = (
            nn.Linear(hidden_dim, hidden_dim) if use_agent_context else None
        )
        self.geometry_confidence_projection = nn.Linear(1, hidden_dim)

        self.temporal_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.future_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=False,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.map2future_layers = nn.ModuleList(
            [
                AttentionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    dropout=dropout,
                    bipartite=True,
                    has_pos_emb=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_projection = nn.Linear(hidden_dim, token_size)

        nn.init.normal_(self.chunk_embedding.weight, std=0.02)

    @staticmethod
    def _flat_batch_ids(mask: torch.Tensor) -> torch.Tensor:
        B, L = mask.shape
        return torch.arange(B, device=mask.device).unsqueeze(1).expand(B, L).reshape(-1)

    @staticmethod
    def _empty_edge_index(device: torch.device) -> torch.Tensor:
        return empty_edge_index(device)

    def _build_spatial_token_edges(
        self,
        positions: torch.Tensor,
        headings: torch.Tensor,
        chunk_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ):
        device = positions.device
        flat_pos = positions.reshape(-1, 2)
        flat_head = headings.reshape(-1)
        flat_valid = valid_mask.reshape(-1)
        flat_source = flat_valid if source_mask is None else (source_mask.reshape(-1) & flat_valid)
        flat_target = flat_valid if target_mask is None else (target_mask.reshape(-1) & flat_valid)
        search_mask = flat_source | flat_target
        if not search_mask.any():
            return self._empty_edge_index(device), positions.new_zeros((0, self.hidden_dim))
        flat_batch = self._flat_batch_ids(valid_mask)
        flat_chunk = chunk_ids.reshape(-1)
        head_vector = torch.stack([flat_head.cos(), flat_head.sin()], dim=-1)
        group_ids = flat_batch * self.num_future_chunks + flat_chunk
        search_nodes = torch.nonzero(search_mask, as_tuple=False).squeeze(-1)
        sort_order = torch.argsort(group_ids[search_nodes])
        sorted_nodes = search_nodes[sort_order]
        edge_index, r_raw = build_radius_interaction_edges(
            pos_s=flat_pos[sorted_nodes],
            head_s=flat_head[sorted_nodes],
            head_vector_s=head_vector[sorted_nodes],
            batch_s=group_ids[sorted_nodes],
            mask_s=torch.ones(sorted_nodes.numel(), dtype=torch.bool, device=device),
            radius_m=self.a2a_radius,
        )
        if edge_index.numel() == 0:
            return edge_index, self.r_a2a_embedding(continuous_inputs=r_raw, categorical_embs=None)
        edge_index = sorted_nodes[edge_index]
        src, dst = edge_index
        keep = (
            flat_source[src]
            & flat_target[dst]
            & (flat_batch[src] == flat_batch[dst])
            & (flat_chunk[src] == flat_chunk[dst])
        )
        edge_index = edge_index[:, keep]
        r_raw = r_raw[keep]
        if edge_index.numel() == 0:
            return self._empty_edge_index(device), positions.new_zeros((0, self.hidden_dim))
        return edge_index, self.r_a2a_embedding(continuous_inputs=r_raw, categorical_embs=None)

    def _build_temporal_token_edges(
        self,
        positions: torch.Tensor,
        headings: torch.Tensor,
        chunk_ids: torch.Tensor,
        agent_ids: torch.Tensor,
        valid_mask: torch.Tensor,
        source_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ):
        flat_pos = positions.reshape(-1, 2)
        flat_head = headings.reshape(-1)
        flat_chunk = chunk_ids.reshape(-1)
        flat_agent = agent_ids.reshape(-1)
        flat_valid = valid_mask.reshape(-1) & (flat_agent >= 0)
        flat_source = flat_valid if source_mask is None else (source_mask.reshape(-1) & flat_valid)
        flat_target = flat_valid if target_mask is None else (target_mask.reshape(-1) & flat_valid)
        head_vector = torch.stack([flat_head.cos(), flat_head.sin()], dim=-1)
        max_chunk_delta = None
        if self.time_span is not None:
            max_chunk_delta = max(1, int(self.time_span // self.future_chunk_steps))
        edge_index, r_raw = build_temporal_edges_from_flat(
            pos_s=flat_pos,
            head_s=flat_head,
            head_vector_s=head_vector,
            agent_ids=flat_agent,
            step_ids=flat_chunk,
            valid_mask=flat_source,
            target_mask=flat_target,
            max_step_delta=max_chunk_delta,
            causal=False,
        )
        return edge_index, self.r_temporal_embedding(continuous_inputs=r_raw, categorical_embs=None)

    def _build_map2token_edges(
        self,
        token_positions: torch.Tensor,
        token_headings: torch.Tensor,
        valid_mask: torch.Tensor,
        map_positions: Optional[torch.Tensor],
        map_orientations: Optional[torch.Tensor],
        map_batch: Optional[torch.Tensor],
        map_valid_mask: Optional[torch.Tensor],
    ):
        device = token_positions.device
        if map_positions is None or map_orientations is None:
            return self._empty_edge_index(device), token_positions.new_zeros((0, self.hidden_dim))

        flat_token_pos = token_positions.reshape(-1, 2)
        flat_token_head = token_headings.reshape(-1)
        flat_token_valid = valid_mask.reshape(-1)

        if map_positions.dim() == 3:
            B, M, _ = map_positions.shape
            if M == 0:
                return self._empty_edge_index(device), token_positions.new_zeros((0, self.hidden_dim))
            flat_map_positions = map_positions.reshape(-1, 2)
            flat_map_orientations = map_orientations.reshape(-1)
            flat_map_batch = torch.arange(B, device=device).unsqueeze(1).expand(B, M).reshape(-1)
            flat_map_valid = (
                map_valid_mask.reshape(-1).bool()
                if map_valid_mask is not None
                else torch.ones(B * M, dtype=torch.bool, device=device)
            )
        else:
            if map_positions.shape[0] == 0:
                return self._empty_edge_index(device), token_positions.new_zeros((0, self.hidden_dim))
            if map_batch is None:
                raise ValueError("flat map context requires map_batch.")
            flat_map_positions = map_positions
            flat_map_orientations = map_orientations
            flat_map_batch = map_batch
            flat_map_valid = (
                map_valid_mask.bool()
                if map_valid_mask is not None
                else torch.ones(map_positions.shape[0], dtype=torch.bool, device=device)
            )
        if not flat_map_valid.any():
            return self._empty_edge_index(device), token_positions.new_zeros((0, self.hidden_dim))

        token_batch = self._flat_batch_ids(valid_mask)
        token_head_vector = torch.stack([flat_token_head.cos(), flat_token_head.sin()], dim=-1)
        edge_index, r_raw = build_radius_map2agent_edges(
            pos_s=flat_token_pos,
            head_s=flat_token_head,
            head_vector_s=token_head_vector,
            pos_pl=flat_map_positions,
            orient_pl=flat_map_orientations,
            batch_s=token_batch,
            batch_pl=flat_map_batch,
            mask_s=flat_token_valid,
            radius_m=self.pl2a_radius,
            map_token_visible_mask=flat_map_valid,
        )
        return edge_index, self.r_map2token_embedding(continuous_inputs=r_raw, categorical_embs=None)

    def forward(
        self,
        noisy_token_ids: torch.Tensor,
        token_positions: torch.Tensor,
        token_headings: torch.Tensor,
        token_agent_ids: torch.Tensor,
        noisy_token_chunk_ids: torch.Tensor,
        scene_summary: torch.Tensor,
        t: torch.Tensor,
        valid_mask: torch.Tensor,
        agent_context: Optional[torch.Tensor] = None,
        agent_type_ids: Optional[torch.Tensor] = None,
        agent_shape_embeddings: Optional[torch.Tensor] = None,
        physical_token_embeddings: Optional[torch.Tensor] = None,
        map_context: Optional[torch.Tensor] = None,
        map_positions: Optional[torch.Tensor] = None,
        map_orientations: Optional[torch.Tensor] = None,
        map_batch: Optional[torch.Tensor] = None,
        map_valid_mask: Optional[torch.Tensor] = None,
        geometry_confidence: Optional[torch.Tensor] = None,
        temporal_source_mask: Optional[torch.Tensor] = None,
        spatial_source_mask: Optional[torch.Tensor] = None,
        proposal_token_embeddings: Optional[torch.Tensor] = None,
        proposal_confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = noisy_token_ids.shape

        if physical_token_embeddings is None:
            tok_emb = self.token_embedding(noisy_token_ids.clamp(max=self.token_size - 1))
        else:
            tok_emb = physical_token_embeddings.to(dtype=self.mask_token.dtype)
        mask_indicator = (noisy_token_ids == self.mask_token_id)
        tok_emb = torch.where(
            mask_indicator.unsqueeze(-1),
            self.mask_token.view(1, 1, -1),
            tok_emb,
        )

        time_emb = self.time_mlp(t.view(-1, 1)).unsqueeze(1)
        chunk_emb = self.chunk_embedding(noisy_token_chunk_ids)

        x = tok_emb + time_emb + chunk_emb + self.context_projection(scene_summary).unsqueeze(1)

        if self.agent_context_projection is not None and agent_context is not None:
            x = x + self.agent_context_projection(agent_context.to(dtype=x.dtype))

        if agent_shape_embeddings is not None:
            x = x + agent_shape_embeddings.to(dtype=x.dtype)

        if self.type_embedding is not None and agent_type_ids is not None:
            x = x + self.type_embedding(agent_type_ids.clamp(min=0, max=self.type_embedding.num_embeddings - 1))

        if proposal_token_embeddings is not None and proposal_confidence is not None:
            x = x + (
                proposal_token_embeddings.to(dtype=x.dtype)
                * proposal_confidence.to(dtype=x.dtype).clamp(0.0, 1.0).unsqueeze(-1)
            )

        if geometry_confidence is None:
            geometry_confidence = valid_mask.new_zeros(valid_mask.shape, dtype=x.dtype)
        x = x + self.geometry_confidence_projection(
            geometry_confidence.to(dtype=x.dtype).clamp(0.0, 1.0).unsqueeze(-1)
        )

        x = x * valid_mask.unsqueeze(-1).to(x.dtype)

        temporal_edge_index, r_temporal = self._build_temporal_token_edges(
            token_positions,
            token_headings,
            noisy_token_chunk_ids,
            token_agent_ids,
            valid_mask,
            source_mask=temporal_source_mask,
            target_mask=valid_mask,
        )
        spatial_edge_index, r_spatial = self._build_spatial_token_edges(
            token_positions,
            token_headings,
            noisy_token_chunk_ids,
            valid_mask,
            source_mask=spatial_source_mask,
            target_mask=valid_mask,
        )
        map_edge_index, r_map = self._build_map2token_edges(
            token_positions,
            token_headings,
            valid_mask,
            map_positions,
            map_orientations,
            map_batch,
            map_valid_mask,
        )
        flat_x = x.reshape(B * L, self.hidden_dim)
        if map_context is not None and map_context.numel() > 0:
            flat_map_x = self.map_context_projection(map_context.to(dtype=x.dtype)).reshape(-1, self.hidden_dim)
            if map_valid_mask is not None:
                flat_map_x = flat_map_x * map_valid_mask.reshape(-1, 1).to(flat_map_x.dtype)
        else:
            flat_map_x = None

        flat_valid = valid_mask.reshape(-1, 1).to(flat_x.dtype)
        for layer_idx, future_layer in enumerate(self.future_layers):
            if temporal_edge_index.numel() > 0:
                flat_x = self.temporal_layers[layer_idx](flat_x, r_temporal, temporal_edge_index)
                flat_x = flat_x * flat_valid
            if flat_map_x is not None and map_edge_index.numel() > 0:
                flat_x = self.map2future_layers[layer_idx]((flat_map_x, flat_x), r_map, map_edge_index)
                flat_x = flat_x * flat_valid
            if spatial_edge_index.numel() > 0:
                flat_x = future_layer(flat_x, r_spatial, spatial_edge_index)
                flat_x = flat_x * flat_valid

        logits = self.output_projection(flat_x.reshape(B, L, self.hidden_dim))
        return logits
