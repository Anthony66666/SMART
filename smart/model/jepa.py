from copy import deepcopy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FutureBlockEncoder(nn.Module):
    def __init__(self, hidden_dim: int, future_steps: int, future_chunk_steps: int) -> None:
        super().__init__()
        if future_steps % future_chunk_steps != 0:
            raise ValueError(
                f"num_future_steps ({future_steps}) must be divisible by future_chunk_steps ({future_chunk_steps})."
            )
        self.future_steps = future_steps
        self.future_chunk_steps = future_chunk_steps
        self.num_chunks = future_steps // future_chunk_steps
        self.chunk_projection = nn.Linear(future_chunk_steps * 4, hidden_dim)
        self.chunk_embedding = nn.Parameter(torch.randn(self.num_chunks, hidden_dim) * 0.02)

    def forward(
        self,
        future_states: torch.Tensor,
        valid_agents: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_agents, future_steps, channels = future_states.shape
        if future_steps != self.future_steps:
            raise ValueError(f"Expected {self.future_steps} future steps, got {future_steps}.")
        if channels != 4:
            raise ValueError(f"Expected 4 future-state channels, got {channels}.")

        reshaped = future_states.view(num_agents, self.num_chunks, self.future_chunk_steps * channels)
        tokens = self.chunk_projection(reshaped) + self.chunk_embedding.unsqueeze(0)
        chunk_mask = future_mask.view(num_agents, self.num_chunks, self.future_chunk_steps).all(dim=-1)
        block_mask = valid_agents.unsqueeze(-1) & chunk_mask
        return tokens, block_mask


class JointEmbeddingPredictiveModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        future_steps: int,
        future_chunk_steps: int,
        num_heads: int,
        dropout: float,
        mask_ratio: float,
    ) -> None:
        super().__init__()
        self.mask_ratio = mask_ratio
        self.online_future_encoder = FutureBlockEncoder(hidden_dim, future_steps, future_chunk_steps)
        self.target_future_encoder = deepcopy(self.online_future_encoder)
        for parameter in self.target_future_encoder.parameters():
            parameter.requires_grad = False

        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=2)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        scene_summary: torch.Tensor,
        future_states: torch.Tensor,
        valid_agents: torch.Tensor,
        future_mask: torch.Tensor,
        graph_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        online_tokens, block_mask = self.online_future_encoder(future_states, valid_agents, future_mask)
        with torch.no_grad():
            target_tokens, _ = self.target_future_encoder(future_states, valid_agents, future_mask)
            target_tokens = F.normalize(target_tokens, dim=-1)

        packed_online, packed_target, packed_mask = self._pack_tokens_by_graph(
            online_tokens,
            target_tokens,
            block_mask,
            graph_index,
        )
        if packed_mask.numel() == 0 or not packed_mask.any():
            zero = scene_summary.new_zeros(())
            return zero, {"jepa_cosine": zero, "masked_fraction": zero}

        prediction_mask = (torch.rand_like(packed_mask.float()) < self.mask_ratio) & packed_mask
        empty_rows = (prediction_mask.sum(dim=-1) == 0) & (packed_mask.sum(dim=-1) > 0)
        if empty_rows.any():
            first_valid = packed_mask.float().argmax(dim=-1)
            prediction_mask[empty_rows, first_valid[empty_rows]] = True
        if not prediction_mask.any():
            zero = scene_summary.new_zeros(())
            return zero, {"jepa_cosine": zero, "masked_fraction": zero}

        predictor_tokens = torch.where(
            prediction_mask.unsqueeze(-1),
            self.mask_token.view(1, 1, -1),
            packed_online,
        )
        context_token = self.context_projection(scene_summary).unsqueeze(1)
        predictor_input = torch.cat([context_token, predictor_tokens], dim=1)
        predictor_mask = torch.cat(
            [torch.ones(scene_summary.shape[0], 1, dtype=torch.bool, device=scene_summary.device), packed_mask],
            dim=1,
        )
        predicted = self.predictor(
            predictor_input,
            src_key_padding_mask=~predictor_mask,
        )[:, 1:, :]
        predicted = F.normalize(self.output_projection(predicted), dim=-1)

        cosine = (predicted * packed_target).sum(dim=-1)
        loss = (1.0 - cosine)[prediction_mask].mean()
        stats = {
            "jepa_cosine": cosine[prediction_mask].mean(),
            "masked_fraction": prediction_mask.float().mean(),
        }
        return loss, stats

    def _pack_tokens_by_graph(
        self,
        online_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        block_mask: torch.Tensor,
        graph_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if graph_index.numel() == 0:
            hidden_dim = online_tokens.shape[-1]
            empty_tokens = online_tokens.new_zeros((0, 0, hidden_dim))
            empty_mask = block_mask.new_zeros((0, 0))
            return empty_tokens, empty_tokens, empty_mask

        num_graphs = int(graph_index.max().item()) + 1
        sequences_online = []
        sequences_target = []
        sequences_mask = []
        max_blocks = 0
        for graph_id in range(num_graphs):
            graph_agents = graph_index == graph_id
            graph_online = online_tokens[graph_agents].reshape(-1, online_tokens.shape[-1])
            graph_target = target_tokens[graph_agents].reshape(-1, target_tokens.shape[-1])
            graph_mask = block_mask[graph_agents].reshape(-1)
            valid_online = graph_online[graph_mask]
            valid_target = graph_target[graph_mask]
            sequences_online.append(valid_online)
            sequences_target.append(valid_target)
            sequences_mask.append(torch.ones(valid_online.shape[0], dtype=torch.bool, device=graph_mask.device))
            max_blocks = max(max_blocks, valid_online.shape[0])

        if max_blocks == 0:
            empty_tokens = online_tokens.new_zeros((num_graphs, 0, online_tokens.shape[-1]))
            empty_mask = block_mask.new_zeros((num_graphs, 0))
            return empty_tokens, empty_tokens, empty_mask

        packed_online = online_tokens.new_zeros((num_graphs, max_blocks, online_tokens.shape[-1]))
        packed_target = target_tokens.new_zeros((num_graphs, max_blocks, target_tokens.shape[-1]))
        packed_mask = block_mask.new_zeros((num_graphs, max_blocks))
        for graph_id, (graph_online, graph_target, graph_mask) in enumerate(
            zip(sequences_online, sequences_target, sequences_mask)
        ):
            if graph_online.shape[0] == 0:
                continue
            packed_online[graph_id, : graph_online.shape[0]] = graph_online
            packed_target[graph_id, : graph_target.shape[0]] = graph_target
            packed_mask[graph_id, : graph_mask.shape[0]] = graph_mask
        return packed_online, packed_target, packed_mask

    @torch.no_grad()
    def update_target_encoder(self, ema_decay: float) -> None:
        for target_param, online_param in zip(
            self.target_future_encoder.parameters(), self.online_future_encoder.parameters()
        ):
            target_param.data.mul_(ema_decay).add_(online_param.data, alpha=1.0 - ema_decay)
