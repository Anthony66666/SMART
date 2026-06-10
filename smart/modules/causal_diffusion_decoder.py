from typing import Optional

import torch

from smart.modules.diffusion_decoder import DiffusionDecoder
from smart.modules.smart_edge_builder import build_temporal_edges_from_flat


class CausalDiffusionDecoder(DiffusionDecoder):
    """Diffusion decoder whose temporal token attention is strictly causal."""

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
            causal=True,
        )
        return edge_index, self.r_temporal_embedding(
            continuous_inputs=r_raw,
            categorical_embs=None,
        )
