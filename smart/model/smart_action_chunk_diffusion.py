import torch
import torch.nn.functional as F

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
        self.action_chunk_shift_consistency_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'action_chunk_shift_consistency_loss_weight', 0.0)),
        )
        self.action_chunk_shift_consistency_interval = max(
            1,
            int(getattr(diffusion_cfg, 'action_chunk_shift_consistency_interval', 1)),
        )
        self.action_chunk_shift_consistency_temperature = max(
            1.0e-3,
            float(getattr(diffusion_cfg, 'action_chunk_shift_consistency_temperature', 1.0)),
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

    def _action_chunk_shift_consistency_active_for_step(self):
        return (
            self.action_chunk_shift_consistency_loss_weight > 0.0
            and self.ar_commit_tokens == 1
            and self.ar_prediction_tokens > 1
            and self._should_run_training_interval(
                self.action_chunk_shift_consistency_interval
            )
        )

    def _cadf_lite_auxiliary_requires_details(self):
        return self._action_chunk_shift_consistency_active_for_step()

    def _action_chunk_shift_consistency_loss_from_logits(
        self,
        current_logits,
        current_valid,
        current_agent_ids,
        next_logits,
        next_valid,
        next_agent_ids,
        current_chunk_ids=None,
        next_chunk_ids=None,
        current_scene_ids=None,
        next_scene_ids=None,
    ):
        if self.ar_commit_tokens != 1 or self.ar_prediction_tokens <= 1:
            return current_logits.sum() * 0.0
        if current_logits.numel() == 0 or next_logits.numel() == 0:
            return current_logits.sum() * 0.0 + next_logits.sum() * 0.0

        losses = []
        temperature = self.action_chunk_shift_consistency_temperature
        if current_agent_ids.dim() == 2:
            if current_chunk_ids is None or next_chunk_ids is None:
                return current_logits.sum() * 0.0 + next_logits.sum() * 0.0
            batch_count = min(current_logits.shape[0], next_logits.shape[0])
            for batch_idx in range(batch_count):
                next_batch_idx = batch_idx
                if current_scene_ids is not None and next_scene_ids is not None:
                    scene_id = int(current_scene_ids[batch_idx].item())
                    scene_matches = (next_scene_ids == scene_id).nonzero(as_tuple=False)
                    if scene_matches.numel() == 0:
                        continue
                    next_batch_idx = int(scene_matches[0, 0].item())
                for token_idx in range(current_logits.shape[1]):
                    if not bool(current_valid[batch_idx, token_idx]):
                        continue
                    agent_id = int(current_agent_ids[batch_idx, token_idx].item())
                    chunk_id = int(current_chunk_ids[batch_idx, token_idx].item())
                    if agent_id < 0 or chunk_id < self.ar_commit_tokens:
                        continue
                    shifted_chunk = chunk_id - self.ar_commit_tokens
                    next_mask = (
                        next_valid[next_batch_idx].bool()
                        & (next_agent_ids[next_batch_idx] == agent_id)
                        & (next_chunk_ids[next_batch_idx] == shifted_chunk)
                    )
                    matches = next_mask.nonzero(as_tuple=False)
                    if matches.numel() == 0:
                        continue
                    next_idx = int(matches[0, 0].item())
                    current_log_prob = F.log_softmax(
                        current_logits[batch_idx, token_idx] / temperature,
                        dim=-1,
                    )
                    next_log_prob = F.log_softmax(
                        next_logits[next_batch_idx, next_idx] / temperature,
                        dim=-1,
                    )
                    current_prob = current_log_prob.exp()
                    next_prob = next_log_prob.exp()
                    forward = F.kl_div(
                        current_log_prob,
                        next_prob.detach(),
                        reduction='sum',
                    )
                    backward = F.kl_div(
                        next_log_prob,
                        current_prob.detach(),
                        reduction='sum',
                    )
                    losses.append((0.5 * (forward + backward)).view(1))
            if not losses:
                return current_logits.sum() * 0.0 + next_logits.sum() * 0.0
            return torch.cat(losses, dim=0).mean()

        for current_row, agent_id in enumerate(current_agent_ids.tolist()):
            matches = (next_agent_ids == int(agent_id)).nonzero(as_tuple=False)
            if matches.numel() == 0:
                continue
            next_row = int(matches[0, 0].item())
            compare_len = min(
                current_logits.shape[1] - self.ar_commit_tokens,
                next_logits.shape[1],
            )
            if compare_len <= 0:
                continue
            current_slice = current_logits[
                current_row,
                self.ar_commit_tokens:self.ar_commit_tokens + compare_len,
            ] / temperature
            next_slice = next_logits[next_row, :compare_len] / temperature
            valid = (
                current_valid[
                    current_row,
                    self.ar_commit_tokens:self.ar_commit_tokens + compare_len,
                ].bool()
                & next_valid[next_row, :compare_len].bool()
            )
            if not bool(valid.any()):
                continue
            current_log_prob = F.log_softmax(current_slice[valid], dim=-1)
            next_log_prob = F.log_softmax(next_slice[valid], dim=-1)
            current_prob = current_log_prob.exp()
            next_prob = next_log_prob.exp()
            forward = F.kl_div(
                current_log_prob,
                next_prob.detach(),
                reduction='none',
            ).sum(dim=-1)
            backward = F.kl_div(
                next_log_prob,
                current_prob.detach(),
                reduction='none',
            ).sum(dim=-1)
            losses.append(0.5 * (forward + backward))
        if not losses:
            return current_logits.sum() * 0.0 + next_logits.sum() * 0.0
        return torch.cat(losses, dim=0).mean()

    def _compute_cadf_lite_auxiliary_loss(
        self,
        data,
        anchor,
        packed,
        summary,
        diffusion_details,
        ref_tensor,
    ):
        del summary
        zero = ref_tensor.new_zeros(())
        active = self._action_chunk_shift_consistency_active_for_step()
        if not active or diffusion_details is None:
            return {
                'auxiliary_loss': zero,
                'action_chunk_shift_consistency_loss': zero,
                'action_chunk_shift_consistency_active': False,
            }

        next_anchor = int(anchor) + int(self.ar_commit_tokens)
        if next_anchor >= self.num_future_chunks:
            return {
                'auxiliary_loss': zero,
                'action_chunk_shift_consistency_loss': zero,
                'action_chunk_shift_consistency_active': False,
            }

        next_view, _target_tokens, _target_valid, _anchor = self._build_ar_training_view(
            data,
            anchor_token=next_anchor,
            perturb=None,
            allow_incomplete_window=True,
        )
        built = self._build_diffusion_inputs(next_view, return_context=False)
        (
            next_packed,
            next_summary,
            _ft,
            _fv,
            _generation_agents,
            _supervision_agents,
            _agent_batch,
        ) = built
        if next_packed is None:
            return {
                'auxiliary_loss': zero,
                'action_chunk_shift_consistency_loss': zero,
                'action_chunk_shift_consistency_active': False,
            }

        next_result = self._compute_diffusion_loss(
            next_packed,
            next_summary,
            forced_mask=next_packed['valid_mask'],
            return_details=True,
            loss_normalization='supervision_weight',
        )
        _next_loss, _next_acc, next_details = next_result
        current_scene_ids = torch.tensor(
            [int(agent_map[0]) for agent_map in packed['agent_maps']],
            dtype=torch.long,
            device=packed['valid_mask'].device,
        )
        next_scene_ids = torch.tensor(
            [int(agent_map[0]) for agent_map in next_packed['agent_maps']],
            dtype=torch.long,
            device=next_packed['valid_mask'].device,
        )
        shift_loss = self._action_chunk_shift_consistency_loss_from_logits(
            diffusion_details['logits'],
            packed['valid_mask'],
            packed['token_agent_ids'],
            next_details['logits'],
            next_packed['valid_mask'],
            next_packed['token_agent_ids'],
            packed['chunk_ids'],
            next_packed['chunk_ids'],
            current_scene_ids,
            next_scene_ids,
        )
        return {
            'auxiliary_loss': self.action_chunk_shift_consistency_loss_weight * shift_loss,
            'action_chunk_shift_consistency_loss': shift_loss,
            'action_chunk_shift_consistency_active': True,
        }

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
