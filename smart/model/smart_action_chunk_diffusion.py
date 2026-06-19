import torch

from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion


class SMARTActionChunkDiffusion(SMARTAutoregressiveDiffusion):
    """AR diffusion variant with ACT-style overlapping action chunk ensembling."""

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        diffusion_cfg = getattr(model_config, 'diffusion', None)
        self.action_chunk_temporal_ensemble_enabled = bool(
            getattr(diffusion_cfg, 'temporal_ensemble_enabled', True)
        )
        self.action_chunk_temporal_ensemble_decay = min(
            1.0,
            max(0.0, float(getattr(diffusion_cfg, 'temporal_ensemble_decay', 0.8))),
        )
        self.action_chunk_temporal_ensemble_confidence_floor = max(
            0.0,
            float(getattr(diffusion_cfg, 'temporal_ensemble_confidence_floor', 1.0e-4)),
        )
        if self.action_chunk_temporal_ensemble_enabled and self.ar_commit_tokens != 1:
            raise ValueError(
                "SMARTActionChunkDiffusion temporal ensembling requires "
                "diffusion.commit_tokens: 1."
            )
        self._reset_action_chunk_ensemble()

    def _reset_action_chunk_ensemble(self):
        self._action_chunk_ensemble_ids = None
        self._action_chunk_ensemble_confidence = None
        self._action_chunk_ensemble_valid = None

    @torch.no_grad()
    def inference(self, data):
        self._reset_action_chunk_ensemble()
        try:
            return super().inference(data)
        finally:
            self._reset_action_chunk_ensemble()

    def _ensure_action_chunk_ensemble(self, num_agents, total_tokens, device, dtype):
        shape = (num_agents, total_tokens, self.ar_prediction_tokens)
        if (
            self._action_chunk_ensemble_ids is not None
            and tuple(self._action_chunk_ensemble_ids.shape) == shape
            and self._action_chunk_ensemble_ids.device == device
        ):
            return
        self._action_chunk_ensemble_ids = torch.full(
            shape,
            -1,
            dtype=torch.long,
            device=device,
        )
        self._action_chunk_ensemble_confidence = torch.zeros(
            shape,
            dtype=dtype,
            device=device,
        )
        self._action_chunk_ensemble_valid = torch.zeros(
            shape,
            dtype=torch.bool,
            device=device,
        )

    def _action_chunk_temporal_weights(self, device, dtype):
        decay = self.action_chunk_temporal_ensemble_decay
        offsets = torch.arange(self.ar_prediction_tokens, device=device, dtype=dtype)
        return torch.pow(torch.full_like(offsets, decay), offsets)

    def _update_action_chunk_ensemble(
        self,
        per_agent_tokens,
        per_agent_confidence,
        window_valid,
        generation_agents,
        round_idx,
        total_tokens,
    ):
        generation_agents = generation_agents.to(
            device=per_agent_tokens.device,
            dtype=torch.bool,
        )
        window_valid = window_valid.to(
            device=per_agent_tokens.device,
            dtype=torch.bool,
        )
        for chunk_offset in range(self.ar_prediction_tokens):
            target_idx = round_idx * self.ar_commit_tokens + chunk_offset
            if target_idx >= total_tokens:
                continue
            valid = window_valid[:, chunk_offset] & generation_agents
            ids = per_agent_tokens[:, chunk_offset].clone().masked_fill(~valid, -1)
            confidence = per_agent_confidence[:, chunk_offset].clone().masked_fill(
                ~valid,
                0.0,
            )
            self._action_chunk_ensemble_ids[:, target_idx, chunk_offset] = ids
            self._action_chunk_ensemble_confidence[:, target_idx, chunk_offset] = confidence
            self._action_chunk_ensemble_valid[:, target_idx, chunk_offset] = valid

    def _weighted_vote_action_chunk_tokens(
        self,
        candidate_ids,
        candidate_confidence,
        candidate_valid,
        fallback_tokens,
        fallback_confidence,
    ):
        out_tokens = fallback_tokens.clone()
        out_confidence = fallback_confidence.clone()
        temporal_weights = self._action_chunk_temporal_weights(
            candidate_confidence.device,
            candidate_confidence.dtype,
        )
        scores = (
            candidate_confidence.clamp_min(
                self.action_chunk_temporal_ensemble_confidence_floor
            )
            * temporal_weights[None, :]
        )
        scores = scores.masked_fill(~candidate_valid, 0.0)
        for agent_idx in range(candidate_ids.shape[0]):
            valid = candidate_valid[agent_idx]
            if not bool(valid.any()):
                continue
            ids = candidate_ids[agent_idx, valid]
            local_scores = scores[agent_idx, valid]
            unique_ids, inverse = torch.unique(ids, return_inverse=True)
            totals = torch.zeros(
                unique_ids.shape[0],
                device=local_scores.device,
                dtype=local_scores.dtype,
            )
            totals.scatter_add_(0, inverse, local_scores)
            best = int(torch.argmax(totals).item())
            out_tokens[agent_idx] = unique_ids[best]
            out_confidence[agent_idx] = totals[best].clamp(max=1.0)
        return out_tokens, out_confidence

    def _select_ar_committed_tokens(
        self,
        per_agent_tokens,
        per_agent_confidence,
        window_valid,
        generation_agents,
        round_idx,
        rounds,
    ):
        fallback_tokens, fallback_confidence = super()._select_ar_committed_tokens(
            per_agent_tokens,
            per_agent_confidence,
            window_valid,
            generation_agents,
            round_idx,
            rounds,
        )
        if not self.action_chunk_temporal_ensemble_enabled:
            return fallback_tokens, fallback_confidence
        if self.ar_commit_tokens != 1:
            return fallback_tokens, fallback_confidence

        total_tokens = int(rounds) * int(self.ar_commit_tokens)
        self._ensure_action_chunk_ensemble(
            int(per_agent_tokens.shape[0]),
            total_tokens,
            per_agent_tokens.device,
            per_agent_confidence.dtype,
        )
        self._update_action_chunk_ensemble(
            per_agent_tokens,
            per_agent_confidence,
            window_valid,
            generation_agents,
            round_idx,
            total_tokens,
        )

        commit_idx = round_idx * self.ar_commit_tokens
        commit_valid = (
            window_valid[:, :self.ar_commit_tokens].to(
                device=per_agent_tokens.device,
                dtype=torch.bool,
            )
            & generation_agents[:, None].to(device=per_agent_tokens.device, dtype=torch.bool)
        )
        candidate_ids = self._action_chunk_ensemble_ids[:, commit_idx, :]
        candidate_confidence = self._action_chunk_ensemble_confidence[:, commit_idx, :]
        candidate_valid = (
            self._action_chunk_ensemble_valid[:, commit_idx, :]
            & commit_valid[:, :1]
            & (candidate_ids >= 0)
        )
        selected_tokens, selected_confidence = self._weighted_vote_action_chunk_tokens(
            candidate_ids,
            candidate_confidence,
            candidate_valid,
            fallback_tokens[:, 0],
            fallback_confidence[:, 0],
        )
        return selected_tokens[:, None], selected_confidence[:, None]
