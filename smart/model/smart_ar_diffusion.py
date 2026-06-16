
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from smart.model.smart_diffusion import SMARTDiffusion
from smart.modules.causal_diffusion_decoder import CausalDiffusionDecoder
from smart.modules.trajectory_energy import TrajectoryEnergy
from smart.utils import wrap_angle


class SMARTAutoregressiveDiffusion(SMARTDiffusion):
    """Discrete SMART-token diffusion with an autoregressive outer rollout loop.

    Each denoising call predicts a short joint future in SMART trajectory-token
    space. The outer controller commits only the first configured tokens,
    refreshes current agent state/local map context, and repeats.
    """

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        diffusion_cfg = getattr(model_config, 'diffusion', None)
        self.ar_objective = str(getattr(diffusion_cfg, 'ar_objective', 'maskgit')).lower()
        if self.ar_objective not in ('maskgit', 'causal_frontier_v1'):
            raise ValueError(f"Unsupported diffusion.ar_objective: {self.ar_objective}")
        self.ar_history_tokens = int(getattr(diffusion_cfg, 'history_tokens', 2))
        self.ar_prediction_tokens = int(getattr(diffusion_cfg, 'prediction_tokens', 4))
        self.ar_commit_tokens = int(getattr(diffusion_cfg, 'commit_tokens', 2))
        self.ar_token_steps = int(getattr(diffusion_cfg, 'token_steps', self.future_chunk_steps))
        self.ar_total_rollout_steps = int(getattr(diffusion_cfg, 'total_rollout_steps', self.num_future_steps))
        self.ar_local_map_refresh = str(getattr(diffusion_cfg, 'local_map_refresh', 'rescreen')).lower()
        self.ar_carry_tail_proposal = bool(getattr(diffusion_cfg, 'carry_tail_proposal', False))
        self.ar_causal_temporal_edges = bool(getattr(diffusion_cfg, 'causal_temporal_edges', False))
        self.ar_rolling_anchor_training = bool(getattr(diffusion_cfg, 'rolling_anchor_training', True))
        self.current_state_enabled = bool(getattr(diffusion_cfg, 'current_state_enabled', False))
        self.ar_frontier_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'frontier_loss_weight', 1.0)),
        )
        self.ar_continuous_recovery_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'continuous_recovery_loss_weight', 0.0)),
        )
        thresholds = tuple(
            float(value)
            for value in getattr(
                diffusion_cfg,
                'retokenization_error_thresholds',
                (0.65, 0.78, 0.62),
            )
        )
        if len(thresholds) != 3:
            raise ValueError(
                "diffusion.retokenization_error_thresholds must contain veh/ped/cyc values."
            )
        self.retokenization_error_thresholds = thresholds
        self.ar_closed_loop_batch_ratio_max = min(
            1.0,
            max(
                0.0,
                float(getattr(diffusion_cfg, 'closed_loop_batch_ratio_max', 0.0)),
            ),
        )
        self.ar_closed_loop_max_depth = max(
            1,
            min(4, int(getattr(diffusion_cfg, 'closed_loop_max_depth', 4))),
        )
        self.ar_state_perturb_prob = float(getattr(diffusion_cfg, 'state_perturb_prob', 0.5))
        self.ar_state_perturb_prob = min(max(self.ar_state_perturb_prob, 0.0), 1.0)
        self.ar_state_perturb_pos_sigma_m = float(getattr(diffusion_cfg, 'state_perturb_pos_sigma_m', 0.3))
        self.ar_state_perturb_heading_sigma_rad = float(getattr(diffusion_cfg, 'state_perturb_heading_sigma_rad', 0.05))
        sampling_guidance_cfg = getattr(diffusion_cfg, 'sampling_guidance', None)
        self.ar_sampling_guidance_enabled = bool(
            getattr(
                sampling_guidance_cfg,
                'enabled',
                getattr(diffusion_cfg, 'sampling_guidance_enabled', False),
            )
        )
        self.ar_sampling_guidance_mode = str(
            getattr(
                sampling_guidance_cfg,
                'mode',
                getattr(diffusion_cfg, 'sampling_guidance_mode', 'none'),
            )
        ).lower()
        self.ar_sampling_guidance_topk = max(
            1,
            int(
                getattr(
                    sampling_guidance_cfg,
                    'safe_topk',
                    getattr(diffusion_cfg, 'sampling_guidance_topk', 16),
                )
            ),
        )
        self.lane_distance_energy_weight = float(
            getattr(diffusion_cfg, 'lane_distance_energy_weight', 1.0)
        )
        self.lane_heading_energy_weight = float(
            getattr(diffusion_cfg, 'lane_heading_energy_weight', 0.5)
        )
        self.dynamics_energy_weight = float(
            getattr(diffusion_cfg, 'dynamics_energy_weight', 0.25)
        )
        self.collision_energy_weight = float(
            getattr(diffusion_cfg, 'collision_energy_weight', 2.0)
        )
        self.commit_speed_energy_weight = float(
            getattr(diffusion_cfg, 'commit_speed_energy_weight', 2.0)
        )
        self.commit_min_speed_ratio = float(
            getattr(diffusion_cfg, 'commit_min_speed_ratio', 0.75)
        )
        self.commit_max_speed_ratio = float(
            getattr(diffusion_cfg, 'commit_max_speed_ratio', 1.25)
        )
        self.commit_speed_threshold = float(
            getattr(diffusion_cfg, 'commit_speed_threshold', 1.0)
        )
        self.commit_speed_reference_decay = float(
            getattr(diffusion_cfg, 'commit_speed_reference_decay', 1.0)
        )
        map_token_noise_cfg = getattr(diffusion_cfg, 'map_token_noise', None)
        self.map_token_noise_enabled = bool(
            getattr(
                map_token_noise_cfg,
                'enabled',
                getattr(diffusion_cfg, 'map_token_noise_enabled', getattr(self, 'noise', True)),
            )
        )
        self.noise = self.map_token_noise_enabled
        history_dropout_cfg = getattr(diffusion_cfg, 'history_context_dropout', None)
        self.ar_history_context_dropout_enabled = bool(
            getattr(
                history_dropout_cfg,
                'enabled',
                getattr(diffusion_cfg, 'history_context_dropout_enabled', False),
            )
        )
        self.ar_history_context_dropout_prob = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        history_dropout_cfg,
                        'prob',
                        getattr(diffusion_cfg, 'history_context_dropout_prob', 0.0),
                    )
                ),
            ),
        )
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
        if self._use_causal_temporal_decoder():
            token_size = int(getattr(model_config.decoder, 'token_size', 2048))
            self.diffusion_decoder = CausalDiffusionDecoder(
                hidden_dim=self.hidden_dim,
                token_size=token_size,
                num_future_chunks=self.num_future_chunks,
                num_heads=self.model_config.num_heads,
                head_dim=self.model_config.head_dim,
                dropout=self.model_config.dropout,
                num_freq_bands=self.model_config.num_freq_bands,
                a2a_radius=float(self.model_config.decoder.a2a_radius),
                pl2a_radius=float(self.model_config.decoder.pl2a_radius),
                time_span=getattr(self.model_config.decoder, 'time_span', None),
                future_chunk_steps=self.future_chunk_steps,
                num_layers=self.diffusion_num_layers,
                num_token_types=4 if self.use_type_embedding else 1,
                use_agent_context=self.use_agent_context,
            )
        if self.current_state_enabled:
            self.current_state_projection = nn.Sequential(
                nn.Linear(5, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        if self.ar_sampling_guidance_enabled:
            self.trajectory_energy = TrajectoryEnergy(dt=0.1)
        self._token_center_vocab_cache = None

    def _num_ar_rollout_rounds(self):
        return self.ar_total_rollout_steps // (self.ar_commit_tokens * self.ar_token_steps)

    def _use_causal_temporal_decoder(self):
        return bool(getattr(self, 'ar_causal_temporal_edges', False))

    def _ar_sampling_guidance_active(self):
        return (
            bool(getattr(self, 'ar_sampling_guidance_enabled', False))
            and str(getattr(self, 'ar_sampling_guidance_mode', 'none')).lower()
            in ('safe_speed', 'safe')
        )

    def _history_context_mask(self, data):
        return self._ar_history_context_mask(data)

    def _ar_history_context_mask(self, data):
        if (
            not bool(getattr(self, 'ar_history_context_dropout_enabled', False))
            or not bool(getattr(self, 'training', False))
        ):
            return None
        drop_prob = float(getattr(self, 'ar_history_context_dropout_prob', 0.0))
        if drop_prob <= 0.0:
            return None
        base_mask = data['agent']['agent_valid_mask'].bool()
        mask = base_mask.clone()
        history_steps = min(int(getattr(self, 'ar_history_tokens', 0)), mask.shape[1])
        if history_steps <= 0:
            return mask
        drop = torch.rand(
            mask[:, :history_steps].shape,
            device=mask.device,
        ) < drop_prob
        mask[:, :history_steps] = mask[:, :history_steps] & ~drop
        return mask

    def _attach_ar_sampling_guidance_context(
        self,
        packed,
        current_velocities,
        current_headings,
        reference_speeds,
    ):
        if packed is None:
            return
        current_velocities = current_velocities.to(
            device=packed['valid_mask'].device,
            dtype=packed['token_positions'].dtype,
        )
        current_headings = current_headings.to(
            device=packed['valid_mask'].device,
            dtype=packed['token_headings'].dtype,
        )
        reference_speeds = reference_speeds.to(
            device=packed['valid_mask'].device,
            dtype=packed['token_positions'].dtype,
        )
        velocity_window = current_velocities[:, None, :].expand(
            -1,
            self.ar_prediction_tokens,
            -1,
        )
        heading_window = current_headings[:, None].expand(
            -1,
            self.ar_prediction_tokens,
        )
        speed_window = reference_speeds[:, None].expand(
            -1,
            self.ar_prediction_tokens,
        )
        packed['current_velocities'] = self._pack_agent_window_values(
            velocity_window,
            packed,
            fill_value=0.0,
        )
        packed['current_headings'] = self._pack_agent_window_values(
            heading_window,
            packed,
            fill_value=0.0,
        )
        packed['commit_speed_reference'] = self._pack_agent_window_values(
            speed_window,
            packed,
            fill_value=0.0,
        )

    def _ar_commit_speed_energy(
        self,
        candidate_positions,
        anchor_positions,
        current_velocities,
        commit_chunk_ids,
        reference_speeds=None,
    ):
        energy = candidate_positions.new_zeros(candidate_positions.shape[:2])
        if current_velocities is None or candidate_positions.numel() == 0:
            return energy
        commit_mask = commit_chunk_ids.to(device=candidate_positions.device) < int(
            getattr(self, 'ar_commit_tokens', 1)
        )
        if not commit_mask.any():
            return energy

        current_velocities = current_velocities.to(
            device=candidate_positions.device,
            dtype=candidate_positions.dtype,
        )
        anchor_positions = anchor_positions.to(
            device=candidate_positions.device,
            dtype=candidate_positions.dtype,
        )
        current_speed = torch.norm(current_velocities, dim=-1)
        if reference_speeds is not None:
            reference_speeds = reference_speeds.to(
                device=candidate_positions.device,
                dtype=candidate_positions.dtype,
            )
            current_speed = torch.maximum(current_speed, reference_speeds.clamp_min(0.0))

        moving_mask = commit_mask & (
            current_speed >= float(getattr(self, 'commit_speed_threshold', 1.0))
        )
        if not moving_mask.any():
            return energy

        dt = float(getattr(getattr(self, 'trajectory_energy', None), 'dt', 0.1))
        step_dt = max(dt, 1e-3)
        anchor = anchor_positions[:, None, None, :].expand(
            -1,
            candidate_positions.shape[1],
            1,
            -1,
        )
        position_chain = torch.cat([anchor, candidate_positions], dim=-2)
        step_speed = torch.norm(
            position_chain[:, :, 1:] - position_chain[:, :, :-1],
            dim=-1,
        ) / step_dt
        candidate_speed = step_speed.median(dim=-1).values
        minimum_speed = (
            current_speed * float(getattr(self, 'commit_min_speed_ratio', 0.75))
        ).unsqueeze(-1)
        maximum_speed = (
            current_speed * float(getattr(self, 'commit_max_speed_ratio', 1.25))
        ).unsqueeze(-1)
        speed_deficit = (minimum_speed - candidate_speed).clamp_min(0.0)
        speed_excess = (candidate_speed - maximum_speed).clamp_min(0.0)
        scale = max(float(getattr(self, 'commit_speed_threshold', 1.0)), 1.0)
        energy[moving_mask] = (
            (speed_deficit[moving_mask] / scale) ** 2
            + (speed_excess[moving_mask] / scale) ** 2
        )
        return energy

    def _ar_rerank_commit_tokens(
        self,
        sampled_ids,
        sampled_confidence,
        packed,
        summary,
        initial_proposal_token_ids=None,
        initial_proposal_confidence=None,
    ):
        if (
            not self._ar_sampling_guidance_active()
            or packed is None
            or sampled_ids.numel() == 0
        ):
            return sampled_ids, sampled_confidence
        commit_mask = (
            packed['valid_mask']
            & (packed['chunk_ids'] < int(getattr(self, 'ar_commit_tokens', 1)))
        )
        if not commit_mask.any():
            return sampled_ids, sampled_confidence

        rerank_noisy = sampled_ids.clone()
        rerank_noisy[commit_mask] = self.mask_token_id
        geometry_known = packed['valid_mask'] & (rerank_noisy != self.mask_token_id)
        t_batch = torch.full(
            (sampled_ids.shape[0],),
            float(getattr(self, 'min_t', 1.0e-3)),
            device=sampled_ids.device,
        )
        logits = self._decode_diffusion_logits(
            rerank_noisy,
            packed,
            summary,
            t_batch,
            geometry_known,
            proposal_token_ids=initial_proposal_token_ids,
            proposal_confidence=initial_proposal_confidence,
        )
        flat_commit = torch.nonzero(commit_mask.reshape(-1), as_tuple=False).squeeze(-1)
        commit_logits = logits.reshape(-1, logits.shape[-1])[flat_commit]
        if commit_logits.numel() == 0:
            return sampled_ids, sampled_confidence
        topk = min(
            int(getattr(self, 'ar_sampling_guidance_topk', 16)),
            int(commit_logits.shape[-1]),
        )
        topk_log_probabilities, topk_ids = F.log_softmax(
            commit_logits,
            dim=-1,
        ).topk(topk, dim=-1)

        anchor_positions, anchor_headings, _geometry_confidence = self._refresh_token_geometry(
            rerank_noisy,
            packed,
            geometry_known_mask=geometry_known,
            proposal_token_ids=initial_proposal_token_ids,
            proposal_confidence=initial_proposal_confidence,
        )
        flat_positions = anchor_positions.reshape(-1, 2)[flat_commit]
        flat_headings = anchor_headings.reshape(-1)[flat_commit]
        flat_agent_types = packed['agent_type_ids'].reshape(-1)[flat_commit]
        commit_chunk_ids = packed['chunk_ids'].reshape(-1)[flat_commit]
        num_commit = int(flat_commit.numel())
        repeated_positions = flat_positions[:, None].expand(
            -1,
            topk,
            -1,
        ).reshape(-1, 2)
        repeated_headings = flat_headings[:, None].expand(
            -1,
            topk,
        ).reshape(-1)
        repeated_types = flat_agent_types[:, None].expand(
            -1,
            topk,
        ).reshape(-1)
        candidate_positions, candidate_headings = self._token_chunk_world(
            topk_ids.reshape(-1),
            repeated_types,
            repeated_positions,
            repeated_headings,
        )
        candidate_positions = candidate_positions.reshape(
            num_commit,
            topk,
            self.ar_token_steps,
            2,
        )
        candidate_headings = candidate_headings.reshape(
            num_commit,
            topk,
            self.ar_token_steps,
        )

        sequence_length = packed['valid_mask'].shape[1]
        candidate_batch = torch.div(
            flat_commit,
            sequence_length,
            rounding_mode='floor',
        )
        trajectory_energy = getattr(self, 'trajectory_energy', None)
        if trajectory_energy is None:
            trajectory_energy = TrajectoryEnergy(dt=0.1).to(candidate_positions.device)
        lane_distance, lane_heading = trajectory_energy.lane_energy(
            candidate_positions,
            candidate_headings,
            packed.get('map_positions'),
            packed.get('map_orientations'),
            candidate_batch=candidate_batch,
            map_batch=packed.get('map_batch'),
            map_valid_mask=packed.get('map_valid_mask'),
        )
        dynamics = trajectory_energy.dynamics_energy(
            candidate_positions,
            candidate_headings,
        )
        commit_speed_energy = candidate_positions.new_zeros(candidate_positions.shape[:2])
        if packed.get('current_velocities') is not None:
            current_velocities = packed['current_velocities'].reshape(-1, 2)[flat_commit]
            current_headings = packed.get(
                'current_headings',
                anchor_headings,
            ).reshape(-1)[flat_commit]
            reference_speeds = None
            if packed.get('commit_speed_reference') is not None:
                reference_speeds = packed['commit_speed_reference'].reshape(-1)[flat_commit]
            transition_dynamics = trajectory_energy.dynamics_energy(
                candidate_positions,
                candidate_headings,
                current_positions=flat_positions,
                current_velocities=current_velocities,
                current_headings=current_headings,
            )
            dynamics = transition_dynamics
            commit_speed_energy = self._ar_commit_speed_energy(
                candidate_positions,
                flat_positions,
                current_velocities,
                commit_chunk_ids,
                reference_speeds=reference_speeds,
            )

        preliminary_energy = (
            self.lane_distance_energy_weight * lane_distance
            + self.lane_heading_energy_weight * lane_heading
            + self.dynamics_energy_weight * dynamics
            + self.commit_speed_energy_weight * commit_speed_energy
        )
        preliminary_selection = (
            topk_log_probabilities - preliminary_energy
        ).argmax(dim=-1)
        row = torch.arange(num_commit, device=topk_ids.device)
        nominal_other = candidate_positions[row, preliminary_selection]
        commit_agent_ids = packed['token_agent_ids'].reshape(-1)[flat_commit]
        collision = trajectory_energy.collision_energy(
            candidate_positions,
            nominal_other,
            candidate_batch=candidate_batch,
            other_batch=candidate_batch,
            candidate_agent_ids=commit_agent_ids,
            other_agent_ids=commit_agent_ids,
        )
        total_energy = (
            self.lane_distance_energy_weight * lane_distance
            + self.lane_heading_energy_weight * lane_heading
            + self.dynamics_energy_weight * dynamics
            + self.commit_speed_energy_weight * commit_speed_energy
            + self.collision_energy_weight * collision
        )
        selected = (topk_log_probabilities - total_energy).argmax(dim=-1)
        reranked_ids = sampled_ids.clone()
        reranked_ids.reshape(-1)[flat_commit] = topk_ids[row, selected]
        if sampled_confidence is None:
            reranked_confidence = sampled_confidence
        else:
            reranked_confidence = sampled_confidence.clone()
            reranked_confidence.reshape(-1)[flat_commit] = topk_log_probabilities[
                row,
                selected,
            ].exp()
        return reranked_ids, reranked_confidence

    def _ar_closed_loop_curriculum(self, epoch):
        epoch = max(0, int(epoch))
        maximum = float(getattr(self, 'ar_closed_loop_batch_ratio_max', 0.0))
        if epoch <= 3:
            return 0.0, maximum
        return 0.25, maximum

    def _current_state_motion(self, data):
        agent = data['agent']
        current_index = self.num_historical_steps - 1
        previous_index = max(current_index - 1, 0)
        current_position = agent['position'][:, current_index, :2].float()
        previous_position = agent['position'][:, previous_index, :2].float()
        current_heading = agent['heading'][:, current_index].float()
        previous_heading = agent['heading'][:, previous_index].float()
        current_valid = agent['valid_mask'][:, current_index].bool()
        previous_valid = agent['valid_mask'][:, previous_index].bool()
        finite_difference_valid = current_valid & previous_valid

        velocity = (current_position - previous_position) / 0.1
        if 'velocity' in agent:
            observed_velocity = agent['velocity'][
                :,
                min(current_index, agent['velocity'].shape[1] - 1),
                :2,
            ].float()
            velocity = torch.where(
                finite_difference_valid.unsqueeze(-1),
                velocity,
                observed_velocity,
            )
        velocity = velocity.masked_fill(~current_valid.unsqueeze(-1), 0.0)

        cos_heading = current_heading.cos()
        sin_heading = current_heading.sin()
        longitudinal = velocity[:, 0] * cos_heading + velocity[:, 1] * sin_heading
        lateral = -velocity[:, 0] * sin_heading + velocity[:, 1] * cos_heading
        speed = torch.norm(velocity, dim=-1)
        yaw_rate = wrap_angle(current_heading - previous_heading) / 0.1
        yaw_rate = yaw_rate.masked_fill(~finite_difference_valid, 0.0)
        features = torch.stack(
            [
                longitudinal / 20.0,
                lateral / 20.0,
                speed / 20.0,
                yaw_rate / 2.0,
                current_valid.to(dtype=velocity.dtype),
            ],
            dim=-1,
        )
        return velocity, current_heading, features

    def _apply_current_state_context(self, data, packed):
        if not bool(getattr(self, 'current_state_enabled', False)):
            return
        _velocity, _current_heading, state_features = self._current_state_motion(data)
        state_embeddings = self.current_state_projection(state_features)
        for _scene_idx, sequence_idx, agent_indices in packed['agent_maps']:
            for local_idx, agent_idx in enumerate(agent_indices.tolist()):
                start = local_idx * self.ar_prediction_tokens
                end = start + self.ar_prediction_tokens
                packed['agent_context'][sequence_idx, start:end] += (
                    state_embeddings[agent_idx]
                )

    def _build_diffusion_inputs(self, data, rollout_valid=False):
        result = super()._build_diffusion_inputs(
            data,
            rollout_valid=rollout_valid,
        )
        packed = result[0]
        if (
            packed is not None
            and getattr(self, 'ar_objective', 'maskgit') == 'causal_frontier_v1'
        ):
            self._apply_current_state_context(data, packed)
        return result

    def _sample_frontier_ids(self, loss_mask_base, chunk_ids):
        frontier_ids = torch.full(
            (loss_mask_base.shape[0],),
            -1,
            dtype=torch.long,
            device=chunk_ids.device,
        )
        for batch_idx in range(loss_mask_base.shape[0]):
            available = torch.unique(
                chunk_ids[batch_idx][loss_mask_base[batch_idx]]
            )
            available = available[
                (available >= 0) & (available < self.num_future_chunks)
            ]
            if available.numel() == 0:
                continue
            selected = torch.randint(
                available.numel(),
                (1,),
                device=chunk_ids.device,
            )
            frontier_ids[batch_idx] = available[selected]
        return frontier_ids

    def _compute_diffusion_loss(self, packed, summary):
        if getattr(self, 'ar_objective', 'maskgit') == 'causal_frontier_v1':
            return self._compute_frontier_diffusion_loss(packed, summary)
        return super()._compute_diffusion_loss(packed, summary)

    def _compute_frontier_diffusion_loss(self, packed, summary):
        gt = packed['token_ids']
        valid_mask = packed['valid_mask']
        raw_loss_mask_base = packed.get('loss_mask_base', valid_mask) & valid_mask
        retokenization_valid = packed.get(
            'retokenization_valid',
            torch.ones_like(valid_mask),
        ).bool()
        loss_mask_base = raw_loss_mask_base & retokenization_valid
        frontier_ids = self._sample_frontier_ids(
            raw_loss_mask_base,
            packed['chunk_ids'],
        )
        selected_frontier = frontier_ids.unsqueeze(-1)
        has_frontier = selected_frontier >= 0
        mask = (
            valid_mask
            & has_frontier
            & (packed['chunk_ids'] >= selected_frontier)
        )
        frontier_raw_mask = (
            mask
            & (packed['chunk_ids'] == selected_frontier)
            & raw_loss_mask_base
        )
        frontier_mask = frontier_raw_mask & loss_mask_base

        noisy = gt.clone()
        noisy[mask] = self.mask_token_id
        geometry_known_mask = (~mask) & valid_mask
        t = (
            1.0
            - frontier_ids.clamp_min(0).to(dtype=summary.dtype)
            / float(self.num_future_chunks)
        ).clamp_min(float(getattr(self, 'min_t', 1e-3)))
        logits = self._decode_diffusion_logits(
            noisy,
            packed,
            summary,
            t,
            geometry_known_mask,
        )

        if hasattr(self, '_continuous_recovery_loss'):
            recovery_loss = self._continuous_recovery_loss(
                logits,
                packed,
                masked_supervision=frontier_raw_mask,
            )
        else:
            recovery_loss = logits.sum() * 0.0

        if frontier_mask.any():
            log_p = F.log_softmax(logits, dim=-1)
            nll = -log_p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
            loss = nll[frontier_mask].mean() * float(
                getattr(self, 'ar_frontier_loss_weight', 1.0)
            )
            acc = (
                logits[frontier_mask].argmax(-1) == gt[frontier_mask]
            ).float().mean()
        else:
            loss = logits.sum() * 0.0
            acc = logits.new_zeros(())
        loss = loss + float(
            getattr(self, 'ar_continuous_recovery_loss_weight', 0.0)
        ) * recovery_loss

        if getattr(self, 'training', False):
            valid_count = valid_mask.float().sum().clamp_min(1.0)
            self.log(
                'train_ar_frontier_mask_frac',
                mask.float().sum() / valid_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_ar_frontier_frac',
                frontier_mask.float().sum() / valid_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_ar_continuous_recovery_loss',
                recovery_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
        return loss, acc

    def _token_center_vocabs(self):
        if getattr(self, '_token_center_vocab_cache', None) is None:
            center_vocabs = {}
            for type_name in ('veh', 'ped', 'cyc'):
                trajectories = self.token_vocab[type_name]
                endpoints = self.token_endpoint_vocab[type_name]
                smart_trajectories = torch.cat(
                    [
                        trajectories[:, :self.ar_token_steps],
                        endpoints[:, None],
                    ],
                    dim=1,
                )
                center_vocabs[type_name] = smart_trajectories[
                    :, 1:1 + self.ar_token_steps
                ].mean(dim=2)
            self._token_center_vocab_cache = center_vocabs
        return self._token_center_vocab_cache

    def _retokenize_future(
        self,
        future_positions,
        future_valid,
        start_positions,
        start_headings,
        agent_types,
        token_center_vocabs=None,
    ):
        if future_positions.dim() != 4:
            raise ValueError("future_positions must have shape [agent, chunk, step, 2].")
        num_agents, num_chunks, num_steps, _ = future_positions.shape
        if num_steps != self.ar_token_steps:
            raise ValueError(
                f"Expected {self.ar_token_steps} frames per token, got {num_steps}."
            )
        use_physical_decode = token_center_vocabs is None
        center_vocabs = (
            self._token_center_vocabs()
            if use_physical_decode
            else token_center_vocabs
        )
        device = future_positions.device
        token_ids = torch.zeros(num_agents, num_chunks, dtype=torch.long, device=device)
        errors = future_positions.new_full((num_agents, num_chunks), float('inf'))
        target_valid = torch.zeros(
            num_agents,
            num_chunks,
            dtype=torch.bool,
            device=device,
        )
        local_endpoints = future_positions.new_zeros(num_agents, num_chunks, 2)
        current_positions = start_positions.clone()
        current_headings = start_headings.clone()
        type_names = ('veh', 'ped', 'cyc')

        for chunk_idx in range(num_chunks):
            for agent_idx in range(num_agents):
                agent_type = int(agent_types[agent_idx].item())
                if agent_type < 0 or agent_type >= len(type_names):
                    continue
                frame_valid = future_valid[agent_idx, chunk_idx].bool()
                if not frame_valid.any():
                    continue
                world_delta = (
                    future_positions[agent_idx, chunk_idx]
                    - current_positions[agent_idx]
                )
                heading = current_headings[agent_idx]
                cos_heading = heading.cos()
                sin_heading = heading.sin()
                world_to_local = torch.stack([
                    torch.stack([cos_heading, -sin_heading]),
                    torch.stack([sin_heading, cos_heading]),
                ])
                target_local = world_delta @ world_to_local
                valid_indices = torch.nonzero(frame_valid, as_tuple=False).squeeze(-1)
                local_endpoints[agent_idx, chunk_idx] = target_local[valid_indices[-1]]

                vocab = center_vocabs[type_names[agent_type]].to(
                    device=device,
                    dtype=future_positions.dtype,
                )
                distances = torch.norm(
                    vocab[:, frame_valid] - target_local[frame_valid].unsqueeze(0),
                    dim=-1,
                ).mean(dim=-1)
                best_error, best_token = distances.min(dim=0)
                token_ids[agent_idx, chunk_idx] = best_token
                errors[agent_idx, chunk_idx] = best_error
                threshold = self.retokenization_error_thresholds[agent_type]
                target_valid[agent_idx, chunk_idx] = best_error <= threshold

                if use_physical_decode:
                    world, world_heading = self._token_chunk_world(
                        best_token.view(1),
                        agent_types[agent_idx].view(1),
                        current_positions[agent_idx].view(1, 2),
                        current_headings[agent_idx].view(1),
                    )
                    current_positions[agent_idx] = world[0, valid_indices[-1]]
                    current_headings[agent_idx] = world_heading[
                        0,
                        valid_indices[-1],
                    ]
                else:
                    selected = vocab[best_token]
                    endpoint_local = selected[valid_indices[-1]]
                    local_to_world = torch.stack([
                        torch.stack([cos_heading, sin_heading]),
                        torch.stack([-sin_heading, cos_heading]),
                    ])
                    current_positions[agent_idx] = (
                        endpoint_local @ local_to_world
                        + current_positions[agent_idx]
                    )
                    if valid_indices.numel() >= 2:
                        last_delta = (
                            selected[valid_indices[-1]]
                            - selected[valid_indices[-2]]
                        )
                    else:
                        last_delta = selected[valid_indices[-1]]
                    if torch.norm(last_delta) > 1e-6:
                        current_headings[agent_idx] = (
                            current_headings[agent_idx]
                            + torch.atan2(last_delta[1], last_delta[0])
                        )
        return token_ids, errors, target_valid, local_endpoints

    def _continuous_recovery_loss(self, logits, packed, masked_supervision):
        retokenization_valid = packed.get('retokenization_valid')
        target_endpoints = packed.get('recovery_target_local_endpoint')
        if retokenization_valid is None or target_endpoints is None:
            return logits.sum() * 0.0
        recovery_mask = (
            masked_supervision
            & packed.get('loss_mask_base', masked_supervision)
            & ~retokenization_valid.bool()
        )
        if not recovery_mask.any():
            return logits.sum() * 0.0

        probabilities = F.softmax(logits, dim=-1)
        predicted_endpoints = target_endpoints.new_zeros(target_endpoints.shape)
        center_vocabs = self._token_center_vocabs()
        for type_name, type_id in (('veh', 0), ('ped', 1), ('cyc', 2)):
            type_mask = recovery_mask & (packed['agent_type_ids'] == type_id)
            if not type_mask.any():
                continue
            endpoints = center_vocabs[type_name][:, -1].to(
                device=logits.device,
                dtype=logits.dtype,
            )
            predicted_endpoints[type_mask] = probabilities[type_mask] @ endpoints
        return F.smooth_l1_loss(
            predicted_endpoints[recovery_mask],
            target_endpoints[recovery_mask].to(dtype=logits.dtype),
            reduction='mean',
        )

    def _retokenize_training_view(self, view):
        agent = view['agent']
        future_start = self.num_historical_steps
        future_end = future_start + self.ar_prediction_tokens * self.ar_token_steps
        future_positions = agent['position'][
            :,
            future_start:future_end,
            :2,
        ].reshape(
            -1,
            self.ar_prediction_tokens,
            self.ar_token_steps,
            2,
        )
        future_valid = agent['valid_mask'][:, future_start:future_end].reshape(
            -1,
            self.ar_prediction_tokens,
            self.ar_token_steps,
        )
        start_positions = agent['position'][:, self.num_historical_steps - 1, :2]
        start_headings = agent['heading'][:, self.num_historical_steps - 1]
        token_ids, errors, retokenization_valid, local_endpoints = (
            self._retokenize_future(
                future_positions=future_positions,
                future_valid=future_valid,
                start_positions=start_positions,
                start_headings=start_headings,
                agent_types=agent['type'],
            )
        )

        target_slice = slice(
            self.ar_history_tokens,
            self.ar_history_tokens + self.ar_prediction_tokens,
        )
        target_token_valid = agent['agent_valid_mask'][:, target_slice].bool()
        agent['token_idx'][:, target_slice] = token_ids
        (
            _target_traj,
            _target_head,
            _target_frame_valid,
            token_positions,
            token_headings,
            _end_positions,
            _end_headings,
        ) = self._decode_token_sequence(
            token_ids,
            target_token_valid,
            agent['type'],
            start_positions,
            start_headings,
        )
        if 'token_pos' in agent:
            agent['token_pos'][:, target_slice] = token_positions
        if 'token_heading' in agent:
            agent['token_heading'][:, target_slice] = token_headings
        return {
            'retokenization_error': errors,
            'retokenization_valid': retokenization_valid & target_token_valid,
            'recovery_target_local_endpoint': local_endpoints,
        }

    def _select_closed_loop_anchor(self, data, rollout_depth):
        token_count = int(data['agent']['token_idx'].shape[1])
        frame_count = int(data['agent']['position'].shape[1])
        minimum_anchor = self.ar_history_tokens
        required_future_tokens = self.ar_prediction_tokens + rollout_depth
        maximum_anchor = token_count - required_future_tokens
        maximum_frame_anchor = (
            frame_count - 1 - required_future_tokens * self.ar_token_steps
        ) // self.ar_token_steps
        maximum_anchor = min(maximum_anchor, maximum_frame_anchor)
        if maximum_anchor < minimum_anchor:
            return None
        if maximum_anchor == minimum_anchor:
            return minimum_anchor
        return int(torch.randint(
            minimum_anchor,
            maximum_anchor + 1,
            (1,),
            device=data['agent']['token_idx'].device,
        ).item())

    @torch.no_grad()
    def _build_model_rollout_training_view(self, data, rollout_depth):
        start_anchor = self._select_closed_loop_anchor(data, rollout_depth)
        if start_anchor is None:
            return None
        start_view, _tokens, _valid, _anchor = self._build_ar_training_view(
            data,
            anchor_token=start_anchor,
            perturb=False,
        )
        (
            packed,
            summary,
            _ft,
            future_valid,
            generation_agents,
            _supervision_agents,
            _agent_batch,
        ) = self._build_diffusion_inputs(start_view, rollout_valid=True)
        if packed is None:
            return None
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
        num_agents = int(data['agent']['position'].shape[0])
        per_agent_tokens, _confidence = self._unpack_sampled_tokens(
            sampled_ids,
            sampled_confidence,
            packed,
            num_agents,
        )
        committed_tokens = per_agent_tokens[:, :rollout_depth]
        committed_valid = (
            future_valid[:, :rollout_depth].bool()
            & generation_agents[:, None]
        )
        current_index = self.num_historical_steps - 1
        start_positions = start_view['agent']['position'][:, current_index, :2]
        start_headings = start_view['agent']['heading'][:, current_index]
        (
            committed_traj,
            committed_head,
            committed_frame_valid,
            committed_token_pos,
            committed_token_heading,
            _end_positions,
            _end_headings,
        ) = self._decode_token_sequence(
            committed_tokens,
            committed_valid,
            start_view['agent']['type'],
            start_positions,
            start_headings,
        )

        history_token_ids = self._roll_history_token_ids(
            start_view['agent']['token_idx'][:, :self.ar_history_tokens],
            committed_tokens,
        )
        history_token_valid = self._roll_history_token_valid(
            start_view['agent']['agent_valid_mask'][:, :self.ar_history_tokens],
            committed_valid,
        )
        history_token_pos = self._roll_history_token_state(
            start_view['agent']['token_pos'][:, :self.ar_history_tokens],
            committed_token_pos,
        )
        history_token_heading = self._roll_history_token_state(
            start_view['agent']['token_heading'][:, :self.ar_history_tokens],
            committed_token_heading,
        )
        history_frame_pos = torch.cat(
            [
                start_view['agent']['position'][
                    :,
                    :self.num_historical_steps,
                    :2,
                ],
                committed_traj,
            ],
            dim=1,
        )[:, -self.num_historical_steps:]
        history_frame_heading = torch.cat(
            [
                start_view['agent']['heading'][:, :self.num_historical_steps],
                committed_head,
            ],
            dim=1,
        )[:, -self.num_historical_steps:]
        history_frame_valid = torch.cat(
            [
                start_view['agent']['valid_mask'][:, :self.num_historical_steps],
                committed_frame_valid,
            ],
            dim=1,
        )[:, -self.num_historical_steps:]
        final_view = self._build_ar_rollout_view(
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

        final_anchor = start_anchor + rollout_depth
        source_frame_start = final_anchor * self.ar_token_steps + 1
        source_frame_end = (
            source_frame_start
            + self.ar_prediction_tokens * self.ar_token_steps
        )
        target_frame_slice = slice(
            self.num_historical_steps,
            self.num_historical_steps
            + self.ar_prediction_tokens * self.ar_token_steps,
        )
        final_view['agent']['position'][:, target_frame_slice] = data['agent'][
            'position'
        ][:, source_frame_start:source_frame_end, :2]
        for key in ('heading', 'valid_mask'):
            final_view['agent'][key][:, target_frame_slice] = data['agent'][key][
                :, source_frame_start:source_frame_end
            ]
        target_token_slice = slice(
            self.ar_history_tokens,
            self.ar_history_tokens + self.ar_prediction_tokens,
        )
        final_view['agent']['agent_valid_mask'][:, target_token_slice] = data[
            'agent'
        ]['agent_valid_mask'][
            :, final_anchor:final_anchor + self.ar_prediction_tokens
        ]
        return final_view

    def _clean_retokenization_metadata(self, view):
        agent = view['agent']
        target_slice = slice(
            self.ar_history_tokens,
            self.ar_history_tokens + self.ar_prediction_tokens,
        )
        target_valid = agent['agent_valid_mask'][:, target_slice].bool()
        num_agents = int(target_valid.shape[0])
        return {
            'retokenization_error': torch.zeros(
                num_agents,
                self.ar_prediction_tokens,
                device=target_valid.device,
            ),
            'retokenization_valid': target_valid.clone(),
            'recovery_target_local_endpoint': torch.zeros(
                num_agents,
                self.ar_prediction_tokens,
                2,
                device=target_valid.device,
            ),
        }

    def _build_ar_frontier_training_view(self, data):
        epoch = int(getattr(self, 'current_epoch', 0))
        perturb_prob, rollout_prob = self._ar_closed_loop_curriculum(epoch)
        rollout_draw = torch.rand((), device=data['agent']['token_idx'].device)
        if rollout_prob > 0.0 and rollout_draw < rollout_prob:
            rollout_depth = int(torch.randint(
                1,
                int(getattr(self, 'ar_closed_loop_max_depth', 1)) + 1,
                (1,),
                device=data['agent']['token_idx'].device,
            ).item())
            rollout_view = self._build_model_rollout_training_view(
                data,
                rollout_depth,
            )
            if rollout_view is not None:
                metadata = self._retokenize_training_view(rollout_view)
                return rollout_view, metadata, 'rollout', rollout_depth

        view, _tokens, _valid, _anchor = self._build_ar_training_view(
            data,
            perturb=False,
        )
        if perturb_prob > 0.0 and torch.rand(
            (),
            device=data['agent']['token_idx'].device,
        ) < perturb_prob:
            self._perturb_ar_history_state(view)
            metadata = self._retokenize_training_view(view)
            return view, metadata, 'perturb', 0
        return view, self._clean_retokenization_metadata(view), 'clean', 0

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

    def _roll_history_token_state(self, history_values, committed_values):
        combined = torch.cat([history_values, committed_values], dim=1)
        return combined[:, -self.ar_history_tokens:].clone()

    def _pack_agent_window_values(self, agent_values, packed, fill_value=0):
        B, L = packed['valid_mask'].shape
        extra_shape = tuple(agent_values.shape[2:])
        result_shape = (B, L) + extra_shape
        if agent_values.dtype == torch.bool:
            result = torch.full(
                result_shape,
                bool(fill_value),
                dtype=agent_values.dtype,
                device=agent_values.device,
            )
        else:
            result = torch.full(
                result_shape,
                fill_value,
                dtype=agent_values.dtype,
                device=agent_values.device,
            )
        for _scene_idx, seq_idx, agent_indices in packed['agent_maps']:
            for local_idx, agent_idx in enumerate(agent_indices.tolist()):
                start = local_idx * self.ar_prediction_tokens
                end = start + self.ar_prediction_tokens
                result[seq_idx, start:end] = agent_values[agent_idx]
        return result

    def _next_tail_proposal(self, per_agent_tokens, per_agent_confidence, future_valid, generation_agents):
        tail_len = self.ar_prediction_tokens - self.ar_commit_tokens
        if tail_len <= 0:
            return None, None
        proposal_ids = torch.zeros_like(per_agent_tokens)
        proposal_confidence = torch.zeros_like(per_agent_confidence)
        proposal_ids[:, :tail_len] = per_agent_tokens[:, self.ar_commit_tokens:]
        proposal_confidence[:, :tail_len] = per_agent_confidence[:, self.ar_commit_tokens:]
        tail_valid = future_valid[:, self.ar_commit_tokens:].bool() & generation_agents[:, None].bool()
        proposal_confidence[:, :tail_len] = proposal_confidence[:, :tail_len].masked_fill(~tail_valid, 0.0)
        proposal_ids[:, :tail_len] = proposal_ids[:, :tail_len].masked_fill(~tail_valid, 0)
        return proposal_ids, proposal_confidence

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
        for seq_idx, (scene_idx, _packed_seq_idx, _agent_indices) in enumerate(packed['agent_maps']):
            # Match SMART inference: keep the scene map feature set available,
            # then let map-to-token radius edges rescreen by the current pose.
            scene_mask = (map_batch == int(scene_idx)) & map_visible
            candidates = torch.nonzero(scene_mask, as_tuple=False).squeeze(-1)
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
        del batch_idx
        data = self._prepare_batch(data)
        retokenization = None
        state_mode = 'clean'
        rollout_depth = 0
        if self.ar_rolling_anchor_training:
            if getattr(self, 'ar_objective', 'maskgit') == 'causal_frontier_v1':
                data, retokenization, state_mode, rollout_depth = (
                    self._build_ar_frontier_training_view(data)
                )
            else:
                data, _target_tokens, _target_valid, _anchor = self._build_ar_training_view(data)
        packed, summary, _ft, _fv, _generation_agents, _supervision_agents, _agent_batch = self._build_diffusion_inputs(data)
        if packed is None:
            zero_loss = self._zero_connected_loss()
            self.log('train_empty_diffusion_batch', zero_loss.detach().new_ones(()),
                     prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            return zero_loss
        if retokenization is not None:
            for key, values in retokenization.items():
                packed[key] = self._pack_agent_window_values(values, packed)

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
        if getattr(self, 'ar_objective', 'maskgit') == 'causal_frontier_v1':
            for name in ('clean', 'perturb', 'rollout'):
                value = loss.new_tensor(1.0 if state_mode == name else 0.0)
                self.log(
                    f'train_ar_state_mode_{name}',
                    value,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    batch_size=1,
                )
            self.log(
                'train_ar_rollout_depth',
                loss.new_tensor(float(rollout_depth)),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
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
                self._log_additional_rollout_metrics(
                    data,
                    pred_out,
                    em,
                    eval_valid,
                )
        self._debug_log(
            f"val_step_done batch_idx={batch_idx} total_elapsed={time.perf_counter() - val_start:.2f}s"
        )

    def _log_additional_rollout_metrics(self, data, pred_out, eval_mask, eval_valid):
        del data, pred_out, eval_mask, eval_valid

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
        if history_frame_pos.shape[1] >= 2:
            reference_velocity = (
                history_frame_pos[:, -1] - history_frame_pos[:, -2]
            ) / 0.1
            reference_speed = torch.norm(reference_velocity, dim=-1)
            reference_valid = history_frame_valid[:, -1] & history_frame_valid[:, -2]
            reference_speed = reference_speed.masked_fill(~reference_valid, 0.0)
        else:
            reference_speed = torch.zeros(num_agents, device=device)
        reference_decay = float(
            getattr(self, 'commit_speed_reference_decay', 1.0)
        )

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
        carried_proposal_ids = None
        carried_proposal_confidence = None

        for round_idx in range(rounds):
            round_start = time.perf_counter()
            self._debug_log(f"ar_inference_round_start round={round_idx + 1}/{rounds}")
            data['agent']['commit_speed_reference'] = reference_speed
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
            if self._ar_sampling_guidance_active():
                if history_frame_pos.shape[1] >= 2:
                    guidance_velocity = (
                        history_frame_pos[:, -1] - history_frame_pos[:, -2]
                    ) / 0.1
                    guidance_valid = history_frame_valid[:, -1] & history_frame_valid[:, -2]
                    guidance_velocity = guidance_velocity.masked_fill(
                        ~guidance_valid.unsqueeze(-1),
                        0.0,
                    )
                else:
                    guidance_velocity = torch.zeros(
                        num_agents,
                        2,
                        device=device,
                        dtype=history_frame_pos.dtype,
                    )
                self._attach_ar_sampling_guidance_context(
                    packed,
                    guidance_velocity,
                    current_heading,
                    reference_speed,
                )
            initial_proposal_ids = None
            initial_proposal_confidence = None
            if (
                getattr(self, 'ar_carry_tail_proposal', False)
                and carried_proposal_ids is not None
                and carried_proposal_confidence is not None
            ):
                initial_proposal_ids = self._pack_agent_window_values(
                    carried_proposal_ids,
                    packed,
                    fill_value=0,
                )
                initial_proposal_confidence = self._pack_agent_window_values(
                    carried_proposal_confidence,
                    packed,
                    fill_value=0.0,
                )
            seed_token_ids = None
            editable_mask = None
            seed_trajs = None
            edit_window_mask = None
            if (
                str(getattr(self, 'guidance_mode', 'none')).lower()
                in ('ego_stress', 'ego_edit')
                and hasattr(self, '_build_guidance_edit_controls')
            ):
                seed_window_tokens, edit_window_mask = self._build_guidance_edit_controls(
                    data,
                    round_idx,
                    fv,
                    generation_agents,
                )
                seed_token_ids = self._pack_agent_window_values(
                    seed_window_tokens,
                    packed,
                    fill_value=0,
                )
                editable_mask = self._pack_agent_window_values(
                    edit_window_mask,
                    packed,
                    fill_value=False,
                )
                if (
                    'seed_trajs' in data['agent']
                    and hasattr(self, '_window_from_seed_trajs')
                ):
                    seed_window_trajs = self._window_from_seed_trajs(
                        data['agent']['seed_trajs'],
                        round_idx,
                    )
                    seed_trajs = self._pack_agent_window_values(
                        seed_window_trajs,
                        packed,
                        fill_value=0.0,
                    )

            sample_start = time.perf_counter()
            sample_kwargs = {
                'summary': summary,
                'token_positions': packed['token_positions'],
                'token_headings': packed['token_headings'],
                'token_agent_ids': packed['token_agent_ids'],
                'chunk_ids': packed['chunk_ids'],
                'valid_mask': packed['valid_mask'],
                'agent_context': packed['agent_context'],
                'agent_type_ids': packed['agent_type_ids'],
                'agent_shape_embeddings': packed['agent_shape_embeddings'],
                'map_context': packed.get('map_context'),
                'map_positions': packed.get('map_positions'),
                'map_orientations': packed.get('map_orientations'),
                'map_batch': packed.get('map_batch'),
                'map_valid_mask': packed.get('map_valid_mask'),
                'packed': packed,
                'initial_proposal_token_ids': initial_proposal_ids,
                'initial_proposal_confidence': initial_proposal_confidence,
            }
            if seed_token_ids is not None or editable_mask is not None:
                sample_kwargs['seed_token_ids'] = seed_token_ids
                sample_kwargs['editable_mask'] = editable_mask
                sample_kwargs['seed_trajs'] = seed_trajs
            sampled_ids, sampled_confidence = self._diffusion_sample(**sample_kwargs)
            sampled_ids, sampled_confidence = self._ar_rerank_commit_tokens(
                sampled_ids,
                sampled_confidence,
                packed,
                summary,
                initial_proposal_token_ids=initial_proposal_ids,
                initial_proposal_confidence=initial_proposal_confidence,
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
            if (
                edit_window_mask is not None
                and hasattr(self, '_apply_guidance_seed_commit_overrides')
            ):
                (
                    commit_traj,
                    commit_head,
                    commit_valid_frames,
                    commit_token_pos,
                    commit_token_heading,
                    current_pos,
                    current_heading,
                ) = self._apply_guidance_seed_commit_overrides(
                    data=data,
                    round_idx=round_idx,
                    edit_window_mask=edit_window_mask,
                    commit_traj=commit_traj,
                    commit_head=commit_head,
                    commit_valid_frames=commit_valid_frames,
                    commit_token_pos=commit_token_pos,
                    commit_token_heading=commit_token_heading,
                    current_pos=current_pos,
                    current_heading=current_heading,
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

            if getattr(self, 'ar_carry_tail_proposal', False):
                carried_proposal_ids, carried_proposal_confidence = self._next_tail_proposal(
                    per_agent_tokens,
                    per_agent_confidence,
                    fv,
                    generation_agents,
                )

            history_token_ids = self._roll_history_token_ids(history_token_ids, committed_tokens)
            history_token_valid = self._roll_history_token_valid(history_token_valid, committed_valid)
            history_token_pos = self._roll_history_token_state(history_token_pos, commit_token_pos)
            history_token_heading = self._roll_history_token_state(history_token_heading, commit_token_heading)
            history_frame_pos = torch.cat([history_frame_pos, commit_traj], dim=1)[:, -self.num_historical_steps:]
            history_frame_heading = torch.cat([history_frame_heading, commit_head], dim=1)[:, -self.num_historical_steps:]
            history_frame_valid = torch.cat([history_frame_valid, commit_valid_frames], dim=1)[:, -self.num_historical_steps:]
            if history_frame_pos.shape[1] >= 2:
                rolled_velocity = (
                    history_frame_pos[:, -1] - history_frame_pos[:, -2]
                ) / 0.1
                rolled_speed = torch.norm(rolled_velocity, dim=-1)
                rolled_valid = history_frame_valid[:, -1] & history_frame_valid[:, -2]
                rolled_speed = rolled_speed.masked_fill(~rolled_valid, 0.0)
                reference_speed = torch.maximum(
                    rolled_speed,
                    reference_speed * reference_decay,
                )
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
