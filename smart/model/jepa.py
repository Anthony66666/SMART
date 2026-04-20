from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from smart.layers.fourier_embedding import FourierEmbedding


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


class MapBlockEncoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        map_features: torch.Tensor,
        map_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if map_features.numel() == 0:
            return map_features, map_valid_mask
        return self.projection(map_features), map_valid_mask.bool()


class JointEmbeddingPredictiveModule(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        future_steps: int,
        future_chunk_steps: int,
        num_heads: int,
        dropout: float,
        num_freq_bands: int,
        mask_ratio: float,
        agent_loss_weight: float = 0.5,
        map_loss_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_future_chunks = future_steps // future_chunk_steps
        self.mask_ratio = mask_ratio
        self.agent_loss_weight = agent_loss_weight
        self.map_loss_weight = map_loss_weight

        self.online_future_encoder = FutureBlockEncoder(hidden_dim, future_steps, future_chunk_steps)
        self.target_future_encoder = deepcopy(self.online_future_encoder)
        self.online_map_encoder = MapBlockEncoder(hidden_dim)
        self.target_map_encoder = deepcopy(self.online_map_encoder)
        for parameter in self.target_future_encoder.parameters():
            parameter.requires_grad = False
        for parameter in self.target_map_encoder.parameters():
            parameter.requires_grad = False

        self.context_projection = nn.Linear(hidden_dim, hidden_dim)
        self.agent_mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.map_mask_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.agent_chunk_embedding = nn.Embedding(self.num_future_chunks, hidden_dim)
        self.agent_position_embedding = FourierEmbedding(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.map_position_embedding = FourierEmbedding(
            input_dim=2,
            hidden_dim=hidden_dim,
            num_freq_bands=num_freq_bands,
        )
        self.token_type_embedding = nn.Embedding(2, hidden_dim)
        predictor_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.predictor = nn.TransformerEncoder(predictor_layer, num_layers=2)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        nn.init.normal_(self.agent_chunk_embedding.weight, std=0.02)

    def forward(
        self,
        scene_summary: torch.Tensor,
        future_states: torch.Tensor,
        valid_agents: torch.Tensor,
        future_mask: torch.Tensor,
        agent_graph_index: torch.Tensor,
        agent_prediction_mask: torch.Tensor,
        agent_token_positions: Optional[torch.Tensor] = None,
        map_online_features: Optional[torch.Tensor] = None,
        map_target_features: Optional[torch.Tensor] = None,
        map_valid_mask: Optional[torch.Tensor] = None,
        map_graph_index: Optional[torch.Tensor] = None,
        map_prediction_mask: Optional[torch.Tensor] = None,
        map_token_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        online_agent_tokens, agent_block_mask = self.online_future_encoder(future_states, valid_agents, future_mask)
        with torch.no_grad():
            target_agent_tokens, _ = self.target_future_encoder(future_states, valid_agents, future_mask)
            target_agent_tokens = F.normalize(target_agent_tokens, dim=-1)

        if agent_token_positions is None:
            agent_token_positions = future_states.new_zeros((future_states.shape[0], 2))
        else:
            agent_token_positions = agent_token_positions.to(device=scene_summary.device, dtype=scene_summary.dtype)
        agent_chunk_ids = torch.arange(
            self.num_future_chunks,
            device=scene_summary.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(online_agent_tokens.shape[0], -1)

        if map_online_features is None:
            map_online_features = scene_summary.new_zeros((0, self.hidden_dim))
        if map_target_features is None:
            map_target_features = map_online_features
        if map_valid_mask is None:
            map_valid_mask = torch.zeros(map_online_features.shape[0], dtype=torch.bool, device=scene_summary.device)
        else:
            map_valid_mask = map_valid_mask.to(device=scene_summary.device, dtype=torch.bool)
        if map_graph_index is None:
            map_graph_index = torch.zeros(map_online_features.shape[0], dtype=torch.long, device=scene_summary.device)
        if map_prediction_mask is None:
            map_prediction_mask = torch.zeros(map_online_features.shape[0], dtype=torch.bool, device=scene_summary.device)
        if map_token_positions is None:
            map_token_positions = scene_summary.new_zeros((map_online_features.shape[0], 2))
        else:
            map_token_positions = map_token_positions.to(device=scene_summary.device, dtype=scene_summary.dtype)

        online_map_tokens, map_block_mask = self.online_map_encoder(map_online_features, map_valid_mask)
        with torch.no_grad():
            target_map_tokens, _ = self.target_map_encoder(map_target_features, map_valid_mask)
            target_map_tokens = F.normalize(target_map_tokens, dim=-1) if target_map_tokens.numel() > 0 else target_map_tokens

        (
            packed_online,
            packed_target,
            packed_valid_mask,
            packed_prediction_mask,
            token_type_ids,
            packed_chunk_ids,
            packed_positions,
        ) = (
            self._pack_tokens_by_graph(
                num_graphs=scene_summary.shape[0],
                agent_online_tokens=online_agent_tokens,
                agent_target_tokens=target_agent_tokens,
                agent_block_mask=agent_block_mask,
                agent_prediction_mask=agent_prediction_mask,
                agent_graph_index=agent_graph_index,
                agent_chunk_ids=agent_chunk_ids,
                agent_token_positions=agent_token_positions,
                map_online_tokens=online_map_tokens,
                map_target_tokens=target_map_tokens,
                map_block_mask=map_block_mask,
                map_prediction_mask=map_prediction_mask,
                map_graph_index=map_graph_index,
                map_token_positions=map_token_positions,
            )
        )
        if packed_valid_mask.numel() == 0 or not packed_valid_mask.any():
            zero = scene_summary.new_zeros(())
            return self._zero_output(zero)
        if not packed_prediction_mask.any():
            zero = scene_summary.new_zeros(())
            return self._zero_output(zero)

        predictor_tokens = self._build_predictor_tokens(
            packed_online,
            packed_prediction_mask,
            token_type_ids,
        )
        predictor_tokens = predictor_tokens + self._build_token_metadata_embeddings(
            packed_valid_mask=packed_valid_mask,
            token_type_ids=token_type_ids,
            agent_chunk_ids=packed_chunk_ids,
            token_positions=packed_positions,
        )
        predictor_tokens = predictor_tokens + self.token_type_embedding(token_type_ids.clamp(min=0))
        context_token = self.context_projection(scene_summary).unsqueeze(1)
        predictor_input = torch.cat([context_token, predictor_tokens], dim=1)
        predictor_mask = torch.cat(
            [torch.ones(scene_summary.shape[0], 1, dtype=torch.bool, device=scene_summary.device), packed_valid_mask],
            dim=1,
        )
        predicted = self.predictor(
            predictor_input,
            src_key_padding_mask=~predictor_mask,
        )[:, 1:, :]
        predicted = F.normalize(self.output_projection(predicted), dim=-1)

        cosine = (predicted * packed_target).sum(dim=-1)
        agent_mask = packed_prediction_mask & (token_type_ids == 0)
        map_mask = packed_prediction_mask & (token_type_ids == 1)

        zero = scene_summary.new_zeros(())
        agent_loss = (1.0 - cosine)[agent_mask].mean() if agent_mask.any() else zero
        map_loss = (1.0 - cosine)[map_mask].mean() if map_mask.any() else zero

        total_loss = zero
        total_weight = 0.0
        if agent_mask.any():
            total_loss = total_loss + self.agent_loss_weight * agent_loss
            total_weight += self.agent_loss_weight
        if map_mask.any():
            total_loss = total_loss + self.map_loss_weight * map_loss
            total_weight += self.map_loss_weight
        if total_weight > 0.0:
            total_loss = total_loss / total_weight

        stats = {
            "jepa_cosine": cosine[packed_prediction_mask].mean(),
            "masked_fraction": packed_prediction_mask.float().sum() / packed_valid_mask.float().sum().clamp_min(1.0),
            "agent_jepa_loss": agent_loss,
            "map_jepa_loss": map_loss,
            "masked_agent_count": agent_mask.float().sum(),
            "masked_map_count": map_mask.float().sum(),
        }
        return total_loss, stats

    def _build_predictor_tokens(
        self,
        packed_online: torch.Tensor,
        packed_prediction_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        if packed_online.numel() == 0:
            return packed_online

        agent_mask_token = self.agent_mask_token.view(1, 1, -1)
        map_mask_token = self.map_mask_token.view(1, 1, -1)
        mask_tokens = torch.where(
            (token_type_ids == 0).unsqueeze(-1),
            agent_mask_token,
            map_mask_token,
        )
        return torch.where(packed_prediction_mask.unsqueeze(-1), mask_tokens, packed_online)

    def _build_token_metadata_embeddings(
        self,
        packed_valid_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        agent_chunk_ids: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        metadata = token_positions.new_zeros((*token_type_ids.shape, self.hidden_dim))
        flat_metadata = metadata.reshape(-1, self.hidden_dim)
        flat_valid_mask = packed_valid_mask.reshape(-1)
        flat_token_type_ids = token_type_ids.reshape(-1)
        flat_agent_chunk_ids = agent_chunk_ids.reshape(-1)
        flat_token_positions = token_positions.reshape(-1, token_positions.shape[-1])

        agent_mask = flat_valid_mask & (flat_token_type_ids == 0)
        if agent_mask.any():
            clamped_chunk_ids = flat_agent_chunk_ids[agent_mask].clamp(min=0, max=self.num_future_chunks - 1)
            flat_metadata[agent_mask] = flat_metadata[agent_mask] + self.agent_chunk_embedding(clamped_chunk_ids)
            flat_metadata[agent_mask] = flat_metadata[agent_mask] + self.agent_position_embedding(
                continuous_inputs=flat_token_positions[agent_mask],
            )

        map_mask = flat_valid_mask & (flat_token_type_ids == 1)
        if map_mask.any():
            flat_metadata[map_mask] = flat_metadata[map_mask] + self.map_position_embedding(
                continuous_inputs=flat_token_positions[map_mask],
            )

        return metadata

    def _pack_tokens_by_graph(
        self,
        num_graphs: int,
        agent_online_tokens: torch.Tensor,
        agent_target_tokens: torch.Tensor,
        agent_block_mask: torch.Tensor,
        agent_prediction_mask: torch.Tensor,
        agent_graph_index: torch.Tensor,
        agent_chunk_ids: torch.Tensor,
        agent_token_positions: torch.Tensor,
        map_online_tokens: torch.Tensor,
        map_target_tokens: torch.Tensor,
        map_block_mask: torch.Tensor,
        map_prediction_mask: torch.Tensor,
        map_graph_index: torch.Tensor,
        map_token_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sequences_online = []
        sequences_target = []
        sequences_valid = []
        sequences_prediction = []
        sequences_type = []
        sequences_chunk_ids = []
        sequences_positions = []
        max_tokens = 0

        for graph_id in range(num_graphs):
            graph_online_parts = []
            graph_target_parts = []
            graph_prediction_parts = []
            graph_type_parts = []
            graph_chunk_id_parts = []
            graph_position_parts = []

            graph_agents = agent_graph_index == graph_id
            if graph_agents.any():
                graph_online = agent_online_tokens[graph_agents].reshape(-1, self.hidden_dim)
                graph_target = agent_target_tokens[graph_agents].reshape(-1, self.hidden_dim)
                graph_valid = agent_block_mask[graph_agents].reshape(-1)
                graph_prediction = agent_prediction_mask[graph_agents].reshape(-1)
                graph_chunk_id = agent_chunk_ids[graph_agents].reshape(-1)
                graph_position = agent_token_positions[graph_agents].unsqueeze(1).expand(
                    -1,
                    agent_online_tokens.shape[1],
                    -1,
                ).reshape(-1, agent_token_positions.shape[-1])
                if graph_valid.any():
                    graph_online_parts.append(graph_online[graph_valid])
                    graph_target_parts.append(graph_target[graph_valid])
                    graph_prediction_parts.append(graph_prediction[graph_valid])
                    graph_type_parts.append(torch.zeros(int(graph_valid.sum().item()), dtype=torch.long, device=graph_valid.device))
                    graph_chunk_id_parts.append(graph_chunk_id[graph_valid])
                    graph_position_parts.append(graph_position[graph_valid])

            graph_maps = map_graph_index == graph_id
            if graph_maps.any():
                graph_online = map_online_tokens[graph_maps]
                graph_target = map_target_tokens[graph_maps]
                graph_valid = map_block_mask[graph_maps]
                graph_prediction = map_prediction_mask[graph_maps]
                graph_position = map_token_positions[graph_maps]
                if graph_valid.any():
                    graph_online_parts.append(graph_online[graph_valid])
                    graph_target_parts.append(graph_target[graph_valid])
                    graph_prediction_parts.append(graph_prediction[graph_valid])
                    graph_type_parts.append(torch.ones(int(graph_valid.sum().item()), dtype=torch.long, device=graph_valid.device))
                    graph_chunk_id_parts.append(torch.zeros(int(graph_valid.sum().item()), dtype=torch.long, device=graph_valid.device))
                    graph_position_parts.append(graph_position[graph_valid])

            if graph_online_parts:
                graph_online = torch.cat(graph_online_parts, dim=0)
                graph_target = torch.cat(graph_target_parts, dim=0)
                graph_prediction = torch.cat(graph_prediction_parts, dim=0)
                graph_type = torch.cat(graph_type_parts, dim=0)
                graph_chunk_ids = torch.cat(graph_chunk_id_parts, dim=0)
                graph_positions = torch.cat(graph_position_parts, dim=0)
            else:
                graph_online = agent_online_tokens.new_zeros((0, self.hidden_dim))
                graph_target = agent_target_tokens.new_zeros((0, self.hidden_dim))
                graph_prediction = agent_block_mask.new_zeros((0,))
                graph_type = torch.zeros((0,), dtype=torch.long, device=agent_online_tokens.device)
                graph_chunk_ids = torch.zeros((0,), dtype=torch.long, device=agent_online_tokens.device)
                graph_positions = agent_token_positions.new_zeros((0, agent_token_positions.shape[-1]))

            sequences_online.append(graph_online)
            sequences_target.append(graph_target)
            sequences_prediction.append(graph_prediction)
            sequences_type.append(graph_type)
            sequences_chunk_ids.append(graph_chunk_ids)
            sequences_positions.append(graph_positions)
            sequences_valid.append(torch.ones(graph_online.shape[0], dtype=torch.bool, device=graph_online.device))
            max_tokens = max(max_tokens, graph_online.shape[0])

        if max_tokens == 0:
            empty_tokens = agent_online_tokens.new_zeros((num_graphs, 0, self.hidden_dim))
            empty_mask = agent_block_mask.new_zeros((num_graphs, 0))
            empty_types = torch.zeros((num_graphs, 0), dtype=torch.long, device=agent_online_tokens.device)
            empty_chunk_ids = torch.zeros((num_graphs, 0), dtype=torch.long, device=agent_online_tokens.device)
            empty_positions = agent_token_positions.new_zeros((num_graphs, 0, agent_token_positions.shape[-1]))
            return empty_tokens, empty_tokens, empty_mask, empty_mask, empty_types, empty_chunk_ids, empty_positions

        packed_online = agent_online_tokens.new_zeros((num_graphs, max_tokens, self.hidden_dim))
        packed_target = agent_target_tokens.new_zeros((num_graphs, max_tokens, self.hidden_dim))
        packed_valid = agent_block_mask.new_zeros((num_graphs, max_tokens))
        packed_prediction = agent_block_mask.new_zeros((num_graphs, max_tokens))
        token_type_ids = torch.zeros((num_graphs, max_tokens), dtype=torch.long, device=agent_online_tokens.device)
        packed_chunk_ids = torch.zeros((num_graphs, max_tokens), dtype=torch.long, device=agent_online_tokens.device)
        packed_positions = agent_token_positions.new_zeros((num_graphs, max_tokens, agent_token_positions.shape[-1]))

        for graph_id, (
            graph_online,
            graph_target,
            graph_valid,
            graph_prediction,
            graph_type,
            graph_chunk_ids,
            graph_positions,
        ) in enumerate(
            zip(
                sequences_online,
                sequences_target,
                sequences_valid,
                sequences_prediction,
                sequences_type,
                sequences_chunk_ids,
                sequences_positions,
            )
        ):
            if graph_online.shape[0] == 0:
                continue
            length = graph_online.shape[0]
            packed_online[graph_id, :length] = graph_online
            packed_target[graph_id, :length] = graph_target
            packed_valid[graph_id, :length] = graph_valid
            packed_prediction[graph_id, :length] = graph_prediction
            token_type_ids[graph_id, :length] = graph_type
            packed_chunk_ids[graph_id, :length] = graph_chunk_ids
            packed_positions[graph_id, :length] = graph_positions

        return packed_online, packed_target, packed_valid, packed_prediction, token_type_ids, packed_chunk_ids, packed_positions

    def _zero_output(self, zero: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return zero, {
            "jepa_cosine": zero,
            "masked_fraction": zero,
            "agent_jepa_loss": zero,
            "map_jepa_loss": zero,
            "masked_agent_count": zero,
            "masked_map_count": zero,
        }

    @torch.no_grad()
    def update_target_encoder(self, ema_decay: float) -> None:
        for target_param, online_param in zip(
            self.target_future_encoder.parameters(), self.online_future_encoder.parameters()
        ):
            target_param.data.mul_(ema_decay).add_(online_param.data, alpha=1.0 - ema_decay)
        for target_param, online_param in zip(
            self.target_map_encoder.parameters(), self.online_map_encoder.parameters()
        ):
            target_param.data.mul_(ema_decay).add_(online_param.data, alpha=1.0 - ema_decay)
