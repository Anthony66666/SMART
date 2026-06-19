import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion


class SMARTContinuousActionDiffusion(SMARTAutoregressiveDiffusion):
    """Diffusion-policy style continuous action chunks on top of SMART context."""

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        diffusion_cfg = getattr(model_config, 'diffusion', None)
        objective = str(
            getattr(diffusion_cfg, 'continuous_action_objective', 'diffusion_policy_v1')
        ).lower()
        if objective != 'diffusion_policy_v1':
            raise ValueError(
                "diffusion.continuous_action_objective must be diffusion_policy_v1."
            )
        if self.ar_commit_tokens != 1:
            raise ValueError(
                "SMARTContinuousActionDiffusion requires diffusion.commit_tokens: 1."
            )

        self.continuous_action_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'continuous_action_loss_weight', 1.0)),
        )
        self.continuous_action_sigma_min = max(
            1.0e-4,
            float(getattr(diffusion_cfg, 'continuous_action_sigma_min', 0.01)),
        )
        self.continuous_action_sigma_max = max(
            self.continuous_action_sigma_min,
            float(getattr(diffusion_cfg, 'continuous_action_sigma_max', 2.0)),
        )
        self.continuous_action_sample_steps = max(
            1,
            int(getattr(diffusion_cfg, 'continuous_action_sample_steps', 8)),
        )
        self.continuous_action_temporal_ensemble_enabled = bool(
            getattr(diffusion_cfg, 'continuous_action_temporal_ensemble_enabled', True)
        )
        self.continuous_action_temporal_ensemble_decay = min(
            1.0,
            max(
                0.0,
                float(getattr(diffusion_cfg, 'continuous_action_temporal_ensemble_decay', 0.8)),
            ),
        )
        self.continuous_action_tokenize_temperature = max(
            1.0e-4,
            float(getattr(diffusion_cfg, 'continuous_action_tokenize_temperature', 4.0)),
        )

        action_dim = int(self.ar_token_steps) * 2
        hidden_dim = int(self.hidden_dim)
        self.continuous_action_chunk_emb = nn.Embedding(
            int(self.ar_prediction_tokens),
            hidden_dim,
        )
        self.continuous_action_time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.continuous_action_denoiser = nn.Sequential(
            nn.Linear(hidden_dim * 4 + action_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, action_dim),
        )
        self._continuous_token_center_cache = None
        self._reset_continuous_action_ensemble()

    def _reset_continuous_action_ensemble(self):
        self._continuous_action_ensemble_world = None
        self._continuous_action_ensemble_valid = None

    @staticmethod
    def _world_action_to_local(world_action, start_pos, start_heading):
        while start_pos.dim() < world_action.dim():
            start_pos = start_pos.unsqueeze(1)
        heading = start_heading
        while heading.dim() < world_action.dim() - 1:
            heading = heading.unsqueeze(1)
        delta = world_action - start_pos
        cos = heading.cos()
        sin = heading.sin()
        local_x = delta[..., 0] * cos + delta[..., 1] * sin
        local_y = -delta[..., 0] * sin + delta[..., 1] * cos
        return torch.stack((local_x, local_y), dim=-1)

    @staticmethod
    def _local_action_to_world(local_action, start_pos, start_heading):
        while start_pos.dim() < local_action.dim():
            start_pos = start_pos.unsqueeze(1)
        heading = start_heading
        while heading.dim() < local_action.dim() - 1:
            heading = heading.unsqueeze(1)
        cos = heading.cos()
        sin = heading.sin()
        world_x = local_action[..., 0] * cos - local_action[..., 1] * sin
        world_y = local_action[..., 0] * sin + local_action[..., 1] * cos
        return torch.stack((world_x, world_y), dim=-1) + start_pos

    def _continuous_noise_sigma(self, t):
        return (
            self.continuous_action_sigma_min
            + t * (self.continuous_action_sigma_max - self.continuous_action_sigma_min)
        )

    def _continuous_action_targets_from_view(self, view):
        agent = view['agent']
        total_frames = int(self.ar_prediction_tokens) * int(self.ar_token_steps)
        future_slice = slice(
            self.num_historical_steps,
            self.num_historical_steps + total_frames,
        )
        future_pos = agent['position'][:, future_slice, :2].float()
        future_valid = agent['valid_mask'][:, future_slice].bool()
        target = future_pos.reshape(
            future_pos.shape[0],
            self.ar_prediction_tokens,
            self.ar_token_steps,
            2,
        )
        frame_valid = future_valid.reshape(
            future_valid.shape[0],
            self.ar_prediction_tokens,
            self.ar_token_steps,
        )
        start_pos = agent['position'][:, self.num_historical_steps - 1, :2].float()
        start_heading = agent['heading'][:, self.num_historical_steps - 1].float()
        return (
            self._world_action_to_local(target, start_pos, start_heading),
            frame_valid,
        )

    def _pack_continuous_action_targets(self, view, packed):
        local_targets, frame_valid = self._continuous_action_targets_from_view(view)
        packed_targets = self._pack_agent_window_values(
            local_targets,
            packed,
            fill_value=0.0,
        )
        packed_frame_valid = self._pack_agent_window_values(
            frame_valid,
            packed,
            fill_value=False,
        ).bool()
        return packed_targets, packed_frame_valid

    def _continuous_action_denoise(self, noisy_action, packed, summary, t):
        B, L = noisy_action.shape[:2]
        scene = summary[:, None, :].expand(B, L, -1)
        chunk_ids = packed['chunk_ids'].clamp(
            min=0,
            max=self.ar_prediction_tokens - 1,
        )
        chunk_emb = self.continuous_action_chunk_emb(chunk_ids)
        time_emb = self.continuous_action_time_mlp(t.unsqueeze(-1))
        action_flat = noisy_action.reshape(B, L, self.ar_token_steps * 2)
        features = torch.cat(
            (
                packed['agent_context'],
                scene,
                chunk_emb,
                time_emb,
                action_flat,
            ),
            dim=-1,
        )
        return self.continuous_action_denoiser(features).reshape(
            B,
            L,
            self.ar_token_steps,
            2,
        )

    def _compute_continuous_action_loss(
        self,
        data,
        anchor_token=None,
        perturb=None,
        allow_incomplete_window=True,
    ):
        view, _target_tokens, _target_valid, anchor = self._build_ar_training_view(
            data,
            anchor_token=anchor_token,
            perturb=perturb,
            allow_incomplete_window=allow_incomplete_window,
        )
        built = self._build_diffusion_inputs(view, return_context=False)
        packed, summary, _ft, _fv, _generation_agents, _supervision_agents, _agent_batch = built
        if packed is None:
            zero = data['agent']['position'].new_zeros(())
            return {
                'continuous_loss': zero,
                'window_count': 0,
                'anchor': int(anchor),
            }

        target, frame_valid = self._pack_continuous_action_targets(view, packed)
        token_mask = (
            packed['valid_mask'].bool()
            & packed.get('loss_mask_base', packed['valid_mask']).bool()
            & frame_valid.any(dim=-1)
        )
        frame_mask = frame_valid & token_mask.unsqueeze(-1)
        if not bool(frame_mask.any()):
            zero = target.sum() * 0.0
            return {
                'continuous_loss': zero,
                'window_count': 0,
                'anchor': int(anchor),
            }

        t = torch.rand(
            packed['valid_mask'].shape,
            device=target.device,
            dtype=target.dtype,
        )
        sigma = self._continuous_noise_sigma(t)
        noisy = target + sigma[..., None, None] * torch.randn_like(target)
        pred = self._continuous_action_denoise(noisy, packed, summary, t)
        sq_error = (pred - target).pow(2).sum(dim=-1)
        loss = (
            sq_error * frame_mask.to(dtype=sq_error.dtype)
        ).sum() / frame_mask.to(dtype=sq_error.dtype).sum().clamp_min(1.0)
        return {
            'continuous_loss': loss,
            'window_count': 1,
            'anchor': int(anchor),
        }

    def _sample_continuous_actions(self, packed, summary):
        B, L = packed['valid_mask'].shape
        device = summary.device
        dtype = summary.dtype
        action = torch.randn(
            B,
            L,
            self.ar_token_steps,
            2,
            device=device,
            dtype=dtype,
        ) * self.continuous_action_sigma_max
        steps = int(self.continuous_action_sample_steps)
        for step_idx in range(steps, 0, -1):
            t_value = float(step_idx) / float(steps)
            t = torch.full((B, L), t_value, device=device, dtype=dtype)
            sigma = self._continuous_noise_sigma(t).clamp_min(1.0e-4)
            clean = self._continuous_action_denoise(action, packed, summary, t)
            if step_idx == 1:
                action = clean
                continue
            next_t = float(step_idx - 1) / float(steps)
            next_sigma = self._continuous_noise_sigma(
                torch.full((B, L), next_t, device=device, dtype=dtype)
            )
            estimated_noise = (action - clean) / sigma[..., None, None]
            action = clean + next_sigma[..., None, None] * estimated_noise
        return action.masked_fill(~packed['valid_mask'][..., None, None], 0.0)

    def _unpack_continuous_actions(self, packed_actions, packed_frame_valid, packed, num_agents):
        device = packed_actions.device
        actions = torch.zeros(
            num_agents,
            self.ar_prediction_tokens,
            self.ar_token_steps,
            2,
            device=device,
            dtype=packed_actions.dtype,
        )
        valid = torch.zeros(
            num_agents,
            self.ar_prediction_tokens,
            self.ar_token_steps,
            dtype=torch.bool,
            device=device,
        )
        for _scene_idx, seq_idx, agent_indices in packed['agent_maps']:
            seq = packed_actions[seq_idx]
            seq_valid = packed_frame_valid[seq_idx]
            for local_idx, agent_idx in enumerate(agent_indices.tolist()):
                start = local_idx * self.ar_prediction_tokens
                end = start + self.ar_prediction_tokens
                actions[agent_idx] = seq[start:end]
                valid[agent_idx] = seq_valid[start:end]
        return actions, valid

    def _continuous_action_temporal_weights(self, device, dtype):
        offsets = torch.arange(
            self.ar_prediction_tokens,
            device=device,
            dtype=dtype,
        )
        decay = torch.full_like(offsets, self.continuous_action_temporal_ensemble_decay)
        return torch.pow(decay, offsets)

    def _ensure_continuous_action_ensemble(self, num_agents, total_chunks, device, dtype):
        shape = (
            num_agents,
            total_chunks,
            self.ar_prediction_tokens,
            self.ar_token_steps,
            2,
        )
        if (
            self._continuous_action_ensemble_world is not None
            and tuple(self._continuous_action_ensemble_world.shape) == shape
            and self._continuous_action_ensemble_world.device == device
        ):
            return
        self._continuous_action_ensemble_world = torch.zeros(
            shape,
            device=device,
            dtype=dtype,
        )
        self._continuous_action_ensemble_valid = torch.zeros(
            shape[:-1],
            device=device,
            dtype=torch.bool,
        )

    def _update_continuous_action_ensemble(
        self,
        world_actions,
        action_valid,
        generation_agents,
        round_idx,
        total_chunks,
    ):
        for chunk_offset in range(self.ar_prediction_tokens):
            target_idx = round_idx * self.ar_commit_tokens + chunk_offset
            if target_idx >= total_chunks:
                continue
            valid = (
                action_valid[:, chunk_offset].bool()
                & generation_agents[:, None].to(
                    device=action_valid.device,
                    dtype=torch.bool,
                )
            )
            self._continuous_action_ensemble_world[
                :,
                target_idx,
                chunk_offset,
            ] = world_actions[:, chunk_offset]
            self._continuous_action_ensemble_valid[
                :,
                target_idx,
                chunk_offset,
            ] = valid

    def _select_continuous_action_commit(
        self,
        world_actions,
        action_valid,
        generation_agents,
        round_idx,
        rounds,
    ):
        fallback = world_actions[:, 0].clone()
        fallback_valid = (
            action_valid[:, 0].bool()
            & generation_agents[:, None].to(device=action_valid.device, dtype=torch.bool)
        )
        if not self.continuous_action_temporal_ensemble_enabled:
            return fallback, fallback_valid

        total_chunks = int(rounds) * int(self.ar_commit_tokens)
        self._ensure_continuous_action_ensemble(
            int(world_actions.shape[0]),
            total_chunks,
            world_actions.device,
            world_actions.dtype,
        )
        self._update_continuous_action_ensemble(
            world_actions,
            action_valid,
            generation_agents,
            round_idx,
            total_chunks,
        )
        commit_idx = round_idx * self.ar_commit_tokens
        candidates = self._continuous_action_ensemble_world[:, commit_idx]
        candidate_valid = self._continuous_action_ensemble_valid[:, commit_idx]
        weights = self._continuous_action_temporal_weights(
            world_actions.device,
            world_actions.dtype,
        )
        frame_weights = (
            candidate_valid.to(dtype=world_actions.dtype)
            * weights[None, :, None]
        )
        denom = frame_weights.sum(dim=1).clamp_min(1.0e-6)
        averaged = (
            candidates * frame_weights[..., None]
        ).sum(dim=1) / denom[..., None]
        commit_valid = candidate_valid.any(dim=1)
        averaged = torch.where(commit_valid[..., None], averaged, fallback)
        commit_valid = commit_valid | fallback_valid
        return averaged, commit_valid

    def _continuous_heading_from_world(self, world_action, valid, start_pos, start_heading):
        num_agents, steps = world_action.shape[:2]
        heading = torch.zeros(
            num_agents,
            steps,
            device=world_action.device,
            dtype=world_action.dtype,
        )
        previous_pos = start_pos
        previous_heading = start_heading
        for step_idx in range(steps):
            delta = world_action[:, step_idx] - previous_pos
            moved = torch.norm(delta, dim=-1) > 1.0e-4
            step_heading = torch.atan2(delta[:, 1], delta[:, 0])
            step_heading = torch.where(moved & valid[:, step_idx], step_heading, previous_heading)
            heading[:, step_idx] = step_heading
            previous_heading = step_heading
            previous_pos = torch.where(
                valid[:, step_idx, None],
                world_action[:, step_idx],
                previous_pos,
            )
        return heading

    def _continuous_token_center_vocab(self, token_name, device, dtype):
        cache = getattr(self, '_continuous_token_center_cache', None)
        if cache is None:
            cache = {}
            self._continuous_token_center_cache = cache
        key = (token_name, device, dtype)
        if key in cache:
            return cache[key]
        traj = self.token_vocab[token_name].to(device=device, dtype=dtype)
        endpoint = self.token_endpoint_vocab[token_name].to(device=device, dtype=dtype)
        smart_traj = torch.cat(
            [traj[:, :self.future_chunk_steps], endpoint[:, None]],
            dim=1,
        )
        centers = smart_traj[:, 1:1 + self.future_chunk_steps].mean(dim=2)
        cache[key] = centers
        return centers

    def _nearest_token_ids_from_local_action(self, local_action, agent_types, valid):
        token_ids = torch.full(
            (local_action.shape[0],),
            -1,
            dtype=torch.long,
            device=local_action.device,
        )
        confidence = torch.zeros(
            local_action.shape[0],
            device=local_action.device,
            dtype=local_action.dtype,
        )
        specs = (('veh', 0), ('ped', 1), ('cyc', 2))
        for token_name, type_id in specs:
            type_mask = valid & (agent_types == type_id)
            if not bool(type_mask.any()):
                continue
            centers = self._continuous_token_center_vocab(
                token_name,
                local_action.device,
                local_action.dtype,
            )
            query = local_action[type_mask]
            dist = (query[:, None] - centers[None]).pow(2).mean(dim=(2, 3))
            min_dist, nearest = dist.min(dim=1)
            token_ids[type_mask] = nearest
            confidence[type_mask] = torch.exp(
                -min_dist / self.continuous_action_tokenize_temperature
            )
        return token_ids, confidence

    def training_step(self, data, batch_idx):
        del batch_idx
        data = self._prepare_batch(data)
        anchor = self._cadf_lite_anchor(data)
        if anchor is None:
            return self._zero_connected_loss()
        continuous = self._compute_continuous_action_loss(
            data,
            anchor_token=anchor,
            allow_incomplete_window=True,
        )
        continuous_loss = continuous['continuous_loss']
        dense_smart_ce_active = self._dense_smart_ce_active_for_step()
        dense_smart_ce_loss = self._compute_dense_smart_ce_loss(data, continuous_loss)
        loss = (
            self.continuous_action_loss_weight * continuous_loss
            + float(getattr(self, 'dense_smart_ce_loss_weight', 0.0)) * dense_smart_ce_loss
        )
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('continuous_action_loss', continuous_loss, prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=1)
        self.log('dense_smart_ce_loss', dense_smart_ce_loss, prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=1)
        self.log('train_dense_smart_ce_active',
                 loss.new_tensor(float(dense_smart_ce_active)),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_continuous_action_anchor',
                 loss.new_tensor(float(continuous['anchor'])),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        val_start = time.perf_counter()
        data = self._prepare_batch(data)
        try:
            continuous = self._compute_continuous_action_loss(
                data,
                perturb=False,
                allow_incomplete_window=False,
            )
        except ValueError:
            self.log('val_ar_window_empty_diffusion_batch',
                     data['agent']['position'].new_ones(()),
                     prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)
            return
        continuous_loss = continuous['continuous_loss']
        self.log('val_ar_window_empty_diffusion_batch', continuous_loss.new_zeros(()),
                 prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_ar_window_loss', continuous_loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_continuous_action_loss', continuous_loss, prog_bar=True,
                 on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_ar_window_mask_acc', continuous_loss.new_zeros(()),
                 prog_bar=True, on_step=False, on_epoch=True, batch_size=1,
                 sync_dist=True)

        if self._should_run_validation_inference(batch_idx):
            pred_out = self.inference(data)
            if pred_out is not None:
                em = self._metric_agent_mask(data)
                if em.any():
                    eval_valid = self._validation_eval_valid_mask(data, pred_out)
                    self.minADE.update(pred=pred_out['pred_traj'][em],
                                       target=pred_out['gt'][em],
                                       valid_mask=eval_valid[em])
                    self.minFDE.update(pred=pred_out['pred_traj'][em],
                                       target=pred_out['gt'][em],
                                       valid_mask=eval_valid[em])
                    self.log('val_minADE', self.minADE, prog_bar=True,
                             on_step=False, on_epoch=True, batch_size=1)
                    self.log('val_minFDE', self.minFDE, prog_bar=True,
                             on_step=False, on_epoch=True, batch_size=1)
        self._debug_log(
            f"continuous_val_step_done batch_idx={batch_idx} "
            f"elapsed={time.perf_counter() - val_start:.2f}s"
        )

    @torch.no_grad()
    def inference(self, data):
        data = self._prepare_batch(data)
        self._reset_continuous_action_ensemble()
        num_agents = int(data['agent']['position'].shape[0])
        device = data['agent']['position'].device
        rounds = self._num_ar_rollout_rounds()
        generation_agents = self._generation_agent_mask(data)

        history_token_ids = data['agent']['token_idx'][:, :self.ar_history_tokens].long().clone()
        history_token_valid = data['agent']['agent_valid_mask'][:, :self.ar_history_tokens].bool().clone()
        history_token_pos = data['agent']['token_pos'][:, :self.ar_history_tokens, :2].float().clone()
        history_token_heading = data['agent']['token_heading'][:, :self.ar_history_tokens].float().clone()
        history_frame_pos = data['agent']['position'][:, :self.num_historical_steps, :2].float().clone()
        history_frame_heading = data['agent']['heading'][:, :self.num_historical_steps].float().clone()
        history_frame_valid = data['agent']['valid_mask'][:, :self.num_historical_steps].bool().clone()
        current_pos = history_frame_pos[:, -1].clone()
        current_heading = history_frame_heading[:, -1].clone()

        pred_traj = torch.zeros(num_agents, self.ar_total_rollout_steps, 2, device=device)
        pred_head = torch.zeros(num_agents, self.ar_total_rollout_steps, device=device)
        pred_valid_mask = torch.zeros(
            num_agents,
            self.ar_total_rollout_steps,
            dtype=torch.bool,
            device=device,
        )
        pred_token_ids = torch.full(
            (num_agents, rounds * self.ar_commit_tokens),
            -1,
            dtype=torch.long,
            device=device,
        )
        pred_prob = torch.zeros_like(pred_token_ids, dtype=torch.float)

        try:
            for round_idx in range(rounds):
                rollout_view = self._build_ar_rollout_view(
                    data,
                    history_token_ids,
                    history_token_pos,
                    history_token_heading,
                    history_frame_pos,
                    history_frame_heading,
                    history_frame_valid,
                    generation_agents,
                    history_token_valid=history_token_valid,
                )
                packed, summary, _ft, fv, _generation_agents, _supervision_agents, _agent_batch = (
                    self._build_diffusion_inputs(rollout_view)
                )
                if packed is None:
                    break
                local_actions = self._sample_continuous_actions(packed, summary)
                frame_valid = packed['valid_mask'][..., None].expand(
                    -1,
                    -1,
                    self.ar_token_steps,
                )
                per_agent_local, per_agent_valid = self._unpack_continuous_actions(
                    local_actions,
                    frame_valid,
                    packed,
                    num_agents,
                )
                per_agent_world = self._local_action_to_world(
                    per_agent_local,
                    current_pos,
                    current_heading,
                )
                commit_traj, commit_valid_frames = self._select_continuous_action_commit(
                    per_agent_world,
                    per_agent_valid,
                    generation_agents,
                    round_idx,
                    rounds,
                )
                committed_valid = (
                    fv[:, :self.ar_commit_tokens].bool()
                    & generation_agents[:, None]
                )
                commit_valid_frames = commit_valid_frames & committed_valid[:, 0:1]
                commit_head = self._continuous_heading_from_world(
                    commit_traj,
                    commit_valid_frames,
                    current_pos,
                    current_heading,
                )
                frame_start = round_idx * self.ar_token_steps
                frame_end = frame_start + self.ar_token_steps
                pred_traj[:, frame_start:frame_end] = commit_traj
                pred_head[:, frame_start:frame_end] = commit_head
                pred_valid_mask[:, frame_start:frame_end] = commit_valid_frames

                local_commit = self._world_action_to_local(
                    commit_traj[:, None],
                    current_pos,
                    current_heading,
                )[:, 0]
                commit_token_ids, commit_confidence = self._nearest_token_ids_from_local_action(
                    local_commit,
                    data['agent']['type'],
                    commit_valid_frames.any(dim=-1),
                )
                pred_token_ids[:, round_idx] = commit_token_ids
                pred_prob[:, round_idx] = commit_confidence

                commit_token_pos = commit_traj[:, -1:, :]
                commit_token_heading = commit_head[:, -1:]
                current_pos = torch.where(
                    commit_valid_frames[:, -1, None],
                    commit_traj[:, -1],
                    current_pos,
                )
                current_heading = torch.where(
                    commit_valid_frames[:, -1],
                    commit_head[:, -1],
                    current_heading,
                )
                history_token_ids = self._roll_history_token_ids(
                    history_token_ids,
                    commit_token_ids[:, None],
                )
                history_token_valid = self._roll_history_token_valid(
                    history_token_valid,
                    committed_valid,
                )
                history_token_pos = self._roll_history_token_state(
                    history_token_pos,
                    commit_token_pos,
                )
                history_token_heading = self._roll_history_token_state(
                    history_token_heading,
                    commit_token_heading,
                )
                history_frame_pos = torch.cat(
                    [history_frame_pos, commit_traj],
                    dim=1,
                )[:, -self.num_historical_steps:]
                history_frame_heading = torch.cat(
                    [history_frame_heading, commit_head],
                    dim=1,
                )[:, -self.num_historical_steps:]
                history_frame_valid = torch.cat(
                    [history_frame_valid, commit_valid_frames],
                    dim=1,
                )[:, -self.num_historical_steps:]
        finally:
            self._reset_continuous_action_ensemble()

        gt_pos = data['agent']['position'][
            :,
            self.num_historical_steps:self.num_historical_steps + self.ar_total_rollout_steps,
            :2,
        ]
        official_valid = data['agent']['valid_mask'][
            :,
            self.num_historical_steps:self.num_historical_steps + self.ar_total_rollout_steps,
        ].bool().clone()
        gt_val = official_valid.clone()
        try:
            gt_val[data['agent']['category'].long() != 3] = False
        except Exception:
            pass
        return {
            'pos_a': torch.cat(
                [
                    data['agent']['position'][
                        :,
                        self.num_historical_steps - 1:self.num_historical_steps,
                        :2,
                    ],
                    pred_traj,
                ],
                dim=1,
            ),
            'head_a': torch.cat(
                [
                    data['agent']['heading'][
                        :,
                        self.num_historical_steps - 1:self.num_historical_steps,
                    ],
                    pred_head,
                ],
                dim=1,
            ),
            'gt': gt_pos,
            'valid_mask': gt_val,
            'official_valid_mask': official_valid,
            'pred_valid_mask': pred_valid_mask,
            'pred_traj': pred_traj,
            'pred_head': pred_head,
            'next_token_idx': pred_token_ids,
            'next_token_idx_gt': data['agent']['token_idx'][
                :,
                self.history_token_steps:self.history_token_steps + pred_token_ids.shape[1],
            ],
            'next_token_eval_mask': data['agent']['agent_valid_mask'][
                :,
                self.history_token_steps:self.history_token_steps + pred_token_ids.shape[1],
            ],
            'pred_prob': pred_prob,
        }
