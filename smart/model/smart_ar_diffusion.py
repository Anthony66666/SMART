
import math
import time

import torch
from torch_geometric.data import Batch

from smart.model.smart_diffusion import SMARTDiffusion


class SMARTAutoregressiveDiffusion(SMARTDiffusion):
    """Discrete SMART-token diffusion with an autoregressive outer rollout loop.

    Each denoising call predicts a short joint future in SMART trajectory-token
    space. The outer controller commits only the first configured tokens,
    refreshes current agent state/local map context, and repeats.
    """

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        diffusion_cfg = getattr(model_config, 'diffusion', None)
        self.ar_history_tokens = int(getattr(diffusion_cfg, 'history_tokens', 2))
        self.ar_prediction_tokens = int(getattr(diffusion_cfg, 'prediction_tokens', 4))
        self.ar_commit_tokens = int(getattr(diffusion_cfg, 'commit_tokens', 2))
        self.ar_token_steps = int(getattr(diffusion_cfg, 'token_steps', self.future_chunk_steps))
        self.ar_total_rollout_steps = int(getattr(diffusion_cfg, 'total_rollout_steps', self.num_future_steps))
        self.ar_local_map_refresh = str(getattr(diffusion_cfg, 'local_map_refresh', 'rescreen')).lower()
        self.ar_rolling_anchor_training = bool(getattr(diffusion_cfg, 'rolling_anchor_training', True))
        self.ar_state_perturb_prob = float(getattr(diffusion_cfg, 'state_perturb_prob', 0.5))
        self.ar_state_perturb_prob = min(max(self.ar_state_perturb_prob, 0.0), 1.0)
        self.ar_state_perturb_pos_sigma_m = float(getattr(diffusion_cfg, 'state_perturb_pos_sigma_m', 0.3))
        self.ar_state_perturb_heading_sigma_rad = float(getattr(diffusion_cfg, 'state_perturb_heading_sigma_rad', 0.05))
        self.ar_local_map_radius = float(
            getattr(diffusion_cfg, 'local_map_radius', getattr(model_config.decoder, 'pl2a_radius', 30.0))
        )
        if self.ar_token_steps != self.future_chunk_steps:
            raise ValueError(
                f"AR diffusion token_steps ({self.ar_token_steps}) must match future_chunk_steps "
                f"({self.future_chunk_steps})."
            )
        if self.ar_history_tokens != self.history_token_steps:
            raise ValueError(
                f"AR diffusion history_tokens ({self.ar_history_tokens}) must match current SMART "
                f"history token count ({self.history_token_steps})."
            )
        if self.ar_prediction_tokens <= 0 or self.ar_commit_tokens <= 0:
            raise ValueError("AR diffusion prediction_tokens and commit_tokens must be positive.")
        if self.ar_commit_tokens > self.ar_prediction_tokens:
            raise ValueError("AR diffusion commit_tokens cannot exceed prediction_tokens.")
        if self.ar_total_rollout_steps % (self.ar_commit_tokens * self.ar_token_steps) != 0:
            raise ValueError("AR diffusion total_rollout_steps must be divisible by commit_tokens * token_steps.")

        # The diffusion decoder was constructed with the full SMART chunk count;
        # AR training/sampling packs only the short prediction window and uses
        # the first prediction-token chunk ids.
        self.full_num_future_chunks = self.num_future_chunks
        self.num_future_chunks = self.ar_prediction_tokens

    def _num_ar_rollout_rounds(self):
        return self.ar_total_rollout_steps // (self.ar_commit_tokens * self.ar_token_steps)

    def _select_anchor_token(self, data, anchor_token=None):
        if anchor_token is not None:
            return int(anchor_token)
        token_count = int(data['agent']['token_idx'].shape[1])
        min_anchor = self.ar_history_tokens
        max_anchor = token_count - self.ar_prediction_tokens
        frame_max_anchor = (int(data['agent']['position'].shape[1]) - 1 - self.ar_prediction_tokens * self.ar_token_steps) // self.ar_token_steps
        max_anchor = min(max_anchor, frame_max_anchor)
        if max_anchor < min_anchor:
            raise ValueError(
                f"AR diffusion needs at least {self.ar_history_tokens + self.ar_prediction_tokens} "
                f"tokens and matching frames; got token_count={token_count}."
            )
        if self.training and self.ar_rolling_anchor_training and max_anchor > min_anchor:
            sample = torch.randint(
                low=min_anchor,
                high=max_anchor + 1,
                size=(1,),
                device=data['agent']['token_idx'].device,
            )
            return int(sample.item())
        return min_anchor

    def _build_ar_training_view(self, data, anchor_token=None, perturb=None):
        anchor = self._select_anchor_token(data, anchor_token=anchor_token)
        token_start = anchor - self.ar_history_tokens
        token_end = anchor + self.ar_prediction_tokens
        frame_anchor = anchor * self.ar_token_steps
        frame_start = frame_anchor - (self.num_historical_steps - 1)
        frame_end = frame_anchor + self.ar_prediction_tokens * self.ar_token_steps
        if frame_start < 0 or frame_end >= int(data['agent']['position'].shape[1]):
            raise ValueError(
                f"AR diffusion anchor {anchor} maps to invalid frame window "
                f"[{frame_start}, {frame_end}] for {data['agent']['position'].shape[1]} frames."
            )

        view = data.clone()
        agent = view['agent']
        source_agent = data['agent']
        agent['token_idx'] = source_agent['token_idx'][:, token_start:token_end].clone()
        agent['agent_valid_mask'] = source_agent['agent_valid_mask'][:, token_start:token_end].bool().clone()
        if 'token_pos' in source_agent:
            agent['token_pos'] = source_agent['token_pos'][:, token_start:token_end].clone()
        if 'token_heading' in source_agent:
            agent['token_heading'] = source_agent['token_heading'][:, token_start:token_end].clone()
        for key in ('position', 'heading', 'valid_mask'):
            if key in source_agent:
                agent[key] = source_agent[key][:, frame_start:frame_end + 1].clone()
        if 'shape' in source_agent and source_agent['shape'].dim() == 3:
            agent['shape'] = source_agent['shape'][:, frame_start:frame_end + 1].clone()

        target_tokens = source_agent['token_idx'][:, anchor:token_end].long().clone()
        target_valid = source_agent['agent_valid_mask'][:, anchor:token_end].bool().clone()
        do_perturb = bool(perturb) if perturb is not None else (
            self.training
            and self.ar_state_perturb_prob > 0.0
            and torch.rand((), device=target_tokens.device) < self.ar_state_perturb_prob
        )
        if do_perturb:
            self._perturb_ar_history_state(view)
        return view, target_tokens, target_valid, anchor

    def _perturb_ar_history_state(self, data):
        agent = data['agent']
        num_agents = int(agent['position'].shape[0])
        device = agent['position'].device
        if self.ar_state_perturb_pos_sigma_m > 0.0:
            offset = torch.randn(num_agents, 2, device=device) * self.ar_state_perturb_pos_sigma_m
            agent['position'][:, :self.num_historical_steps, :2] += offset[:, None, :]
            if 'token_pos' in agent:
                agent['token_pos'][:, :self.ar_history_tokens, :2] += offset[:, None, :]
        if self.ar_state_perturb_heading_sigma_rad > 0.0:
            delta = torch.randn(num_agents, device=device) * self.ar_state_perturb_heading_sigma_rad
            agent['heading'][:, :self.num_historical_steps] += delta[:, None]
            if 'token_heading' in agent:
                agent['token_heading'][:, :self.ar_history_tokens] += delta[:, None]

    def _roll_history_token_ids(self, history_token_ids, committed_token_ids):
        combined = torch.cat([history_token_ids, committed_token_ids], dim=1)
        return combined[:, -self.ar_history_tokens:].clone()

    def _roll_history_token_valid(self, history_token_valid, committed_token_valid):
        combined = torch.cat([history_token_valid, committed_token_valid], dim=1)
        return combined[:, -self.ar_history_tokens:].clone()

    def _select_local_map_indices(self, map_positions, map_batch, scene_idx, agent_positions, map_visible=None):
        scene_mask = map_batch == int(scene_idx)
        if map_visible is not None:
            scene_mask = scene_mask & map_visible.bool()
        candidates = torch.nonzero(scene_mask, as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            return candidates
        if agent_positions.numel() == 0:
            return candidates[:0]
        dist = torch.cdist(map_positions[candidates, :2], agent_positions[:, :2])
        nearest = dist.min(dim=1).values
        local = candidates[nearest <= self.ar_local_map_radius]
        if local.numel() == 0:
            local = candidates[nearest.argmin().view(1)]
        if self.max_map_tokens > 0 and local.numel() > self.max_map_tokens:
            local_dist = torch.cdist(map_positions[local, :2], agent_positions[:, :2]).min(dim=1).values
            local = local[torch.argsort(local_dist)[:self.max_map_tokens]]
        return local

    def _pack_map_context(self, data, ctx, packed, agent_positions):
        if self.ar_local_map_refresh != 'rescreen':
            return super()._pack_map_context(data, ctx, packed, agent_positions)
        node_types = getattr(data, 'node_types', [])
        if (
            not self.use_map_context
            or 'x_pt' not in ctx
            or 'pt_token' not in node_types
        ):
            return None, None, None, None, None

        map_features = ctx['x_pt']
        map_positions = data['pt_token']['position'][:, :2].float()
        map_orientation = data['pt_token']['orientation'].float()
        map_batch = self._get_map_batch(data)
        map_visible = ctx.get('pt_visibility_mask', None)
        if map_visible is None:
            map_visible = torch.ones(map_features.shape[0], dtype=torch.bool, device=map_features.device)
        else:
            map_visible = map_visible.to(device=map_features.device, dtype=torch.bool)

        kept_context = []
        kept_positions = []
        kept_orientations = []
        kept_batch = []
        for seq_idx, (scene_idx, _packed_seq_idx, agent_indices) in enumerate(packed['agent_maps']):
            candidates = self._select_local_map_indices(
                map_positions=map_positions,
                map_batch=map_batch,
                scene_idx=scene_idx,
                agent_positions=agent_positions[agent_indices],
                map_visible=map_visible,
            )
            if candidates.numel() == 0:
                continue
            kept_context.append(map_features[candidates])
            kept_positions.append(map_positions[candidates])
            kept_orientations.append(map_orientation[candidates])
            kept_batch.append(torch.full(
                (int(candidates.numel()),),
                seq_idx,
                dtype=torch.long,
                device=map_features.device,
            ))

        if not kept_context:
            return None, None, None, None, None

        return (
            torch.cat(kept_context, dim=0),
            torch.cat(kept_positions, dim=0),
            torch.cat(kept_orientations, dim=0),
            torch.cat(kept_batch, dim=0),
            torch.ones(sum(x.shape[0] for x in kept_context), dtype=torch.bool, device=map_features.device),
        )

    def training_step(self, data, batch_idx):
        data = self._prepare_batch(data)
        if self.ar_rolling_anchor_training:
            data, _target_tokens, _target_valid, _anchor = self._build_ar_training_view(data)
        packed, summary, _ft, _fv, _generation_agents, _supervision_agents, _agent_batch = self._build_diffusion_inputs(data)
        if packed is None:
            zero_loss = self._zero_connected_loss()
            self.log('train_empty_diffusion_batch', zero_loss.detach().new_ones(()),
                     prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            return zero_loss

        diffusion_loss, mask_acc = self._compute_diffusion_loss(packed, summary)
        ntp_loss = self._compute_optional_ntp_loss(data, diffusion_loss)
        loss = diffusion_loss + self.ntp_aux_loss_weight * ntp_loss

        self.log('train_empty_diffusion_batch', loss.new_zeros(()),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=1)
        self.log('diffusion_loss', diffusion_loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=1)
        self.log('ntp_loss', ntp_loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=1)
        self.log('train_mask_acc', mask_acc, on_step=True, on_epoch=True, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        val_start = time.perf_counter()
        data = self._prepare_batch(data)
        num_agents = int(data['agent']['position'].shape[0])
        self._debug_log(
            f"val_step_start batch_idx={batch_idx} agents={num_agents} "
            f"inference_token={bool(self.inference_token)}"
        )
        build_start = time.perf_counter()
        loss_data, _target_tokens, _target_valid, _anchor = self._build_ar_training_view(
            data,
            perturb=False,
        )
        packed, summary, _ft, _fv, _generation_agents, _supervision_agents, _agent_batch = self._build_diffusion_inputs(loss_data)
        if packed is None:
            self._debug_log(
                f"val_step_empty batch_idx={batch_idx} elapsed={time.perf_counter() - val_start:.2f}s"
            )
            self.log('val_ar_window_empty_diffusion_batch', data['agent']['position'].new_ones(()),
                     prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)
            return
        packed_tokens = int(packed['valid_mask'].sum().item())
        map_tokens = 0 if packed.get('map_context') is None else int(packed['map_context'].shape[0])
        self._debug_log(
            f"val_step_window_inputs batch_idx={batch_idx} packed_tokens={packed_tokens} "
            f"map_tokens={map_tokens} build_elapsed={time.perf_counter() - build_start:.2f}s"
        )

        loss_start = time.perf_counter()
        diffusion_loss, mask_acc = self._compute_diffusion_loss(packed, summary)
        ntp_loss = self._compute_optional_ntp_loss(loss_data, diffusion_loss)
        total_loss = diffusion_loss + self.ntp_aux_loss_weight * ntp_loss
        self._debug_log(
            f"val_step_window_loss_done batch_idx={batch_idx} "
            f"loss={self._debug_scalar(total_loss):.4f} mask_acc={self._debug_scalar(mask_acc):.4f} "
            f"loss_elapsed={time.perf_counter() - loss_start:.2f}s"
        )

        self.log('val_ar_window_empty_diffusion_batch', total_loss.new_zeros(()),
                 prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_ar_window_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_ar_window_total_loss', total_loss, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_ar_window_diffusion_loss', diffusion_loss, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_ar_window_ntp_loss', ntp_loss, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_ar_window_mask_acc', mask_acc, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)

        if self._should_run_validation_inference(batch_idx):
            inference_start = time.perf_counter()
            self._debug_log(f"val_step_inference_start batch_idx={batch_idx}")
            pred_out = self.inference(data)
            self._debug_log(
                f"val_step_inference_done batch_idx={batch_idx} "
                f"elapsed={time.perf_counter() - inference_start:.2f}s "
                f"pred_valid_frames={0 if pred_out is None else int(pred_out['pred_valid_mask'].sum().item())}"
            )
            if pred_out is not None:
                em = self._metric_agent_mask(data)
                if not em.any():
                    self._debug_log(
                        f"val_step_no_metric_agents batch_idx={batch_idx} total_elapsed={time.perf_counter() - val_start:.2f}s"
                    )
                    return
                eval_valid = self._validation_eval_valid_mask(data, pred_out)
                self.minADE.update(pred=pred_out['pred_traj'][em],
                                   target=pred_out['gt'][em],
                                   valid_mask=eval_valid[em])
                self.minFDE.update(pred=pred_out['pred_traj'][em],
                                   target=pred_out['gt'][em],
                                   valid_mask=eval_valid[em])
                self.log('val_minADE', self.minADE, prog_bar=True, on_step=False,
                         on_epoch=True, batch_size=1)
                self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False,
                         on_epoch=True, batch_size=1)
                shapes = data['agent']['shape']
                if shapes.dim() == 3:
                    shapes = shapes[:, self.num_historical_steps - 1, :]
                self.conflict_rate.update(
                    pred_out['pred_traj'][em],
                    pred_out['pred_head'][em],
                    shapes[em],
                    eval_valid[em],
                )
                self.interaction_consistency.update(
                    pred_out['pred_traj'][em],
                    pred_out['gt'][em],
                    eval_valid[em],
                )
                self.log('val_conflict_rate', self.conflict_rate, prog_bar=False,
                         on_step=False, on_epoch=True, batch_size=1)
                self.log('val_interaction_consistency', self.interaction_consistency,
                         prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
        self._debug_log(
            f"val_step_done batch_idx={batch_idx} total_elapsed={time.perf_counter() - val_start:.2f}s"
        )

    def _build_ar_rollout_view(
        self,
        data,
        history_token_ids,
        history_token_pos,
        history_token_heading,
        history_frame_pos,
        history_frame_heading,
        history_frame_valid,
        generation_agents,
        history_token_valid=None,
    ):
        view = data.clone()
        agent = view['agent']
        num_agents = int(history_token_ids.shape[0])
        device = history_token_ids.device
        total_tokens = self.ar_history_tokens + self.ar_prediction_tokens
        total_frames = self.num_historical_steps + self.ar_prediction_tokens * self.ar_token_steps

        agent['token_idx'] = torch.zeros(num_agents, total_tokens, dtype=torch.long, device=device)
        agent['token_idx'][:, :self.ar_history_tokens] = history_token_ids
        agent['agent_valid_mask'] = torch.zeros(num_agents, total_tokens, dtype=torch.bool, device=device)
        if history_token_valid is None:
            if 'agent_valid_mask' in data['agent']:
                history_token_valid = data['agent']['agent_valid_mask'][:, :self.ar_history_tokens]
            else:
                history_token_valid = torch.ones(num_agents, self.ar_history_tokens, dtype=torch.bool, device=device)
        history_token_valid = history_token_valid.to(device=device, dtype=torch.bool)
        history_token_valid = history_token_valid & generation_agents[:, None].to(device=device, dtype=torch.bool)
        agent['agent_valid_mask'][:, :self.ar_history_tokens] = history_token_valid
        agent['agent_valid_mask'][:, self.ar_history_tokens:] = generation_agents[:, None]
        agent['token_pos'] = torch.zeros(num_agents, total_tokens, 2, dtype=history_token_pos.dtype, device=history_token_pos.device)
        agent['token_pos'][:, :self.ar_history_tokens] = history_token_pos
        agent['token_heading'] = torch.zeros(num_agents, total_tokens, dtype=history_token_heading.dtype, device=history_token_heading.device)
        agent['token_heading'][:, :self.ar_history_tokens] = history_token_heading
        agent['position'] = torch.zeros(num_agents, total_frames, 2, dtype=history_frame_pos.dtype, device=history_frame_pos.device)
        agent['position'][:, :self.num_historical_steps] = history_frame_pos
        agent['heading'] = torch.zeros(num_agents, total_frames, dtype=history_frame_heading.dtype, device=history_frame_heading.device)
        agent['heading'][:, :self.num_historical_steps] = history_frame_heading
        agent['valid_mask'] = torch.zeros(num_agents, total_frames, dtype=torch.bool, device=history_frame_valid.device)
        agent['valid_mask'][:, :self.num_historical_steps] = history_frame_valid
        agent['valid_mask'][:, self.num_historical_steps:] = generation_agents[:, None]
        if 'shape' in data['agent'] and data['agent']['shape'].dim() == 3:
            current_shape = data['agent']['shape'][:, min(self.num_historical_steps - 1, data['agent']['shape'].shape[1] - 1)]
            agent['shape'] = current_shape[:, None, :].expand(-1, total_frames, -1).clone()
        return view

    def _unpack_sampled_tokens(self, sampled_ids, sampled_confidence, packed, num_agents):
        device = sampled_ids.device
        tokens = torch.full((num_agents, self.ar_prediction_tokens), -1, dtype=torch.long, device=device)
        confidence = torch.zeros(num_agents, self.ar_prediction_tokens, device=device)
        for _scene_idx, seq_idx, agent_indices in packed['agent_maps']:
            seq = sampled_ids[seq_idx]
            seq_conf = None if sampled_confidence is None else sampled_confidence[seq_idx]
            for local_idx, agent_idx in enumerate(agent_indices.tolist()):
                start = local_idx * self.ar_prediction_tokens
                end = start + self.ar_prediction_tokens
                tokens[agent_idx] = seq[start:end]
                if seq_conf is not None:
                    confidence[agent_idx] = seq_conf[start:end]
        return tokens, confidence

    def _decode_token_sequence(self, token_ids, token_valid, agent_types, start_pos, start_heading):
        num_agents, num_tokens = token_ids.shape
        device = token_ids.device
        traj = start_pos.new_zeros(num_agents, num_tokens * self.ar_token_steps, 2)
        head = start_heading.new_zeros(num_agents, num_tokens * self.ar_token_steps)
        valid = torch.zeros(num_agents, num_tokens * self.ar_token_steps, dtype=torch.bool, device=device)
        token_pos = start_pos.new_zeros(num_agents, num_tokens, 2)
        token_heading = start_heading.new_zeros(num_agents, num_tokens)
        pos = start_pos.clone()
        heading = start_heading.clone()
        for token_idx in range(num_tokens):
            active = (
                token_valid[:, token_idx]
                & (token_ids[:, token_idx] >= 0)
                & (token_ids[:, token_idx] < self.token_size)
            )
            if active.any():
                world, world_heading = self._token_chunk_world(
                    token_ids[active, token_idx],
                    agent_types[active],
                    pos[active],
                    heading[active],
                )
                frame_start = token_idx * self.ar_token_steps
                frame_end = frame_start + self.ar_token_steps
                traj[active, frame_start:frame_end] = world
                head[active, frame_start:frame_end] = world_heading
                valid[active, frame_start:frame_end] = True
                pos[active] = world[:, -1]
                heading[active] = world_heading[:, -1]
            token_pos[:, token_idx] = pos
            token_heading[:, token_idx] = heading
        return traj, head, valid, token_pos, token_heading, pos, heading

    @torch.no_grad()
    def inference(self, data):
        inference_start = time.perf_counter()
        data = self._prepare_batch(data)
        num_agents = int(data['agent']['position'].shape[0])
        device = data['agent']['position'].device
        rounds = self._num_ar_rollout_rounds()
        self._debug_log(
            f"ar_inference_start agents={num_agents} rounds={rounds} "
            f"diffusion_steps={self.diffusion_num_steps} commit_tokens={self.ar_commit_tokens}"
        )
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
        pred_valid_mask = torch.zeros(num_agents, self.ar_total_rollout_steps, dtype=torch.bool, device=device)
        pred_token_ids = torch.full(
            (num_agents, rounds * self.ar_commit_tokens),
            -1,
            dtype=torch.long,
            device=device,
        )
        pred_prob = torch.zeros_like(pred_token_ids, dtype=torch.float)

        for round_idx in range(rounds):
            round_start = time.perf_counter()
            self._debug_log(f"ar_inference_round_start round={round_idx + 1}/{rounds}")
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
            packed, summary, ft, fv, _generation_agents, _supervision_agents, agent_batch = self._build_diffusion_inputs(rollout_view)
            if packed is None:
                self._debug_log(
                    f"ar_inference_round_empty round={round_idx + 1}/{rounds} elapsed={time.perf_counter() - round_start:.2f}s"
                )
                break
            packed_tokens = int(packed['valid_mask'].sum().item())
            map_tokens = 0 if packed.get('map_context') is None else int(packed['map_context'].shape[0])
            self._debug_log(
                f"ar_inference_round_packed round={round_idx + 1}/{rounds} "
                f"packed_tokens={packed_tokens} map_tokens={map_tokens}"
            )
            sample_start = time.perf_counter()
            sampled_ids, sampled_confidence = self._diffusion_sample(
                summary=summary,
                token_positions=packed['token_positions'],
                token_headings=packed['token_headings'],
                token_agent_ids=packed['token_agent_ids'],
                chunk_ids=packed['chunk_ids'],
                valid_mask=packed['valid_mask'],
                agent_context=packed['agent_context'],
                agent_type_ids=packed['agent_type_ids'],
                agent_shape_embeddings=packed['agent_shape_embeddings'],
                map_context=packed.get('map_context'),
                map_positions=packed.get('map_positions'),
                map_orientations=packed.get('map_orientations'),
                map_batch=packed.get('map_batch'),
                map_valid_mask=packed.get('map_valid_mask'),
                packed=packed,
            )
            self._debug_log(
                f"ar_inference_round_sample_done round={round_idx + 1}/{rounds} "
                f"sample_elapsed={time.perf_counter() - sample_start:.2f}s"
            )
            per_agent_tokens, per_agent_confidence = self._unpack_sampled_tokens(
                sampled_ids,
                sampled_confidence,
                packed,
                num_agents,
            )
            committed_tokens = per_agent_tokens[:, :self.ar_commit_tokens]
            committed_confidence = per_agent_confidence[:, :self.ar_commit_tokens]
            committed_valid = fv[:, :self.ar_commit_tokens].bool() & generation_agents[:, None]
            commit_traj, commit_head, commit_valid_frames, commit_token_pos, commit_token_heading, current_pos, current_heading = self._decode_token_sequence(
                committed_tokens,
                committed_valid,
                data['agent']['type'],
                current_pos,
                current_heading,
            )
            frame_start = round_idx * self.ar_commit_tokens * self.ar_token_steps
            frame_end = frame_start + self.ar_commit_tokens * self.ar_token_steps
            pred_traj[:, frame_start:frame_end] = commit_traj
            pred_head[:, frame_start:frame_end] = commit_head
            pred_valid_mask[:, frame_start:frame_end] = commit_valid_frames
            token_start = round_idx * self.ar_commit_tokens
            token_end = token_start + self.ar_commit_tokens
            pred_token_ids[:, token_start:token_end] = committed_tokens
            pred_prob[:, token_start:token_end] = committed_confidence

            history_token_ids = self._roll_history_token_ids(history_token_ids, committed_tokens)
            history_token_valid = self._roll_history_token_valid(history_token_valid, committed_valid)
            history_token_pos = commit_token_pos[:, -self.ar_history_tokens:].clone()
            history_token_heading = commit_token_heading[:, -self.ar_history_tokens:].clone()
            history_frame_pos = torch.cat([history_frame_pos, commit_traj], dim=1)[:, -self.num_historical_steps:]
            history_frame_heading = torch.cat([history_frame_heading, commit_head], dim=1)[:, -self.num_historical_steps:]
            history_frame_valid = torch.cat([history_frame_valid, commit_valid_frames], dim=1)[:, -self.num_historical_steps:]
            self._debug_log(
                f"ar_inference_round_done round={round_idx + 1}/{rounds} "
                f"committed_valid_frames={int(commit_valid_frames.sum().item())} "
                f"round_elapsed={time.perf_counter() - round_start:.2f}s"
            )

        self._debug_log(
            f"ar_inference_done elapsed={time.perf_counter() - inference_start:.2f}s "
            f"pred_valid_frames={int(pred_valid_mask.sum().item())}"
        )

        gt_pos = data['agent']['position'][:, self.num_historical_steps:self.num_historical_steps + self.ar_total_rollout_steps, :2]
        official_valid = data['agent']['valid_mask'][:, self.num_historical_steps:self.num_historical_steps + self.ar_total_rollout_steps].bool().clone()
        gt_val = official_valid.clone()
        try:
            gt_val[data['agent']['category'].long() != 3] = False
        except Exception:
            pass

        return {
            'pos_a': torch.cat([data['agent']['position'][:, self.num_historical_steps - 1:self.num_historical_steps, :2], pred_traj], dim=1),
            'head_a': torch.cat([data['agent']['heading'][:, self.num_historical_steps - 1:self.num_historical_steps], pred_head], dim=1),
            'gt': gt_pos,
            'valid_mask': gt_val,
            'official_valid_mask': official_valid,
            'pred_valid_mask': pred_valid_mask,
            'pred_traj': pred_traj,
            'pred_head': pred_head,
            'next_token_idx': pred_token_ids,
            'next_token_idx_gt': data['agent']['token_idx'][:, self.history_token_steps:self.history_token_steps + pred_token_ids.shape[1]],
            'next_token_eval_mask': data['agent']['agent_valid_mask'][:, self.history_token_steps:self.history_token_steps + pred_token_ids.shape[1]],
            'pred_prob': pred_prob,
        }
