from typing import Optional

import torch
import torch.nn as nn

from smart.modules.diffusion_decoder import DiffusionDecoder


class EmbeddedLanguageFlowDecoder(DiffusionDecoder):
    """Non-causal graph decoder with separate ELF velocity and token heads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.flow_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L = noisy_token_ids.shape

        if physical_token_embeddings is None:
            tok_emb = self.token_embedding(noisy_token_ids.clamp(max=self.token_size - 1))
        else:
            tok_emb = physical_token_embeddings.to(dtype=self.mask_token.dtype)
        mask_indicator = noisy_token_ids == self.mask_token_id
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
            x = x + self.type_embedding(
                agent_type_ids.clamp(min=0, max=self.type_embedding.num_embeddings - 1)
            )
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
            flat_map_x = self.map_context_projection(
                map_context.to(dtype=x.dtype)
            ).reshape(-1, self.hidden_dim)
            if map_valid_mask is not None:
                flat_map_x = flat_map_x * map_valid_mask.reshape(-1, 1).to(flat_map_x.dtype)
        else:
            flat_map_x = None

        flat_valid = valid_mask.reshape(-1, 1).to(flat_x.dtype)
        for layer_idx, future_layer in enumerate(self.future_layers):
            if temporal_edge_index.numel() > 0:
                flat_x = self.temporal_layers[layer_idx](
                    flat_x,
                    r_temporal,
                    temporal_edge_index,
                )
                flat_x = flat_x * flat_valid
            if flat_map_x is not None and map_edge_index.numel() > 0:
                flat_x = self.map2future_layers[layer_idx](
                    (flat_map_x, flat_x),
                    r_map,
                    map_edge_index,
                )
                flat_x = flat_x * flat_valid
            if spatial_edge_index.numel() > 0:
                flat_x = future_layer(flat_x, r_spatial, spatial_edge_index)
                flat_x = flat_x * flat_valid

        hidden = flat_x.reshape(B, L, self.hidden_dim)
        velocity = self.flow_projection(hidden)
        decoder_logits = self.output_projection(hidden)
        return velocity, decoder_logits
