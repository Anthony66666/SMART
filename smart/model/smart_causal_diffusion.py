import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.modules.causal_diffusion_decoder import CausalDiffusionDecoder
from smart.modules.trajectory_energy import TrajectoryEnergy
from smart.utils import wrap_angle


class SMARTCausalDiffusion(SMARTAutoregressiveDiffusion):
    """Causal discrete-frontier planning over short SMART token windows."""

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        if self.ar_prediction_tokens != 4:
            raise ValueError(
                "SMART causal diffusion requires diffusion.prediction_tokens == 4."
            )
        if self.ar_commit_tokens != 1:
            raise ValueError(
                "SMART causal diffusion requires diffusion.commit_tokens == 1."
            )
        if self.diffusion_num_steps != self.num_future_chunks:
            raise ValueError(
                "SMART causal diffusion requires num_steps == prediction_tokens "
                "so every sampling step releases exactly one causal frontier."
            )

        diffusion_cfg = model_config.diffusion
        self.causal_objective = str(
            getattr(diffusion_cfg, 'causal_objective', 'discrete_frontier_v2')
        ).lower()
        if self.causal_objective != 'discrete_frontier_v2':
            raise ValueError(
                "SMART causal diffusion requires "
                "diffusion.causal_objective == 'discrete_frontier_v2'."
            )
        self.causal_frontier_loss_weight = float(
            getattr(diffusion_cfg, 'frontier_loss_weight', 1.0)
        )
        self.causal_frontier_loss_weight = max(self.causal_frontier_loss_weight, 0.0)
        self.closed_loop_batch_ratio_max = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        diffusion_cfg,
                        'closed_loop_batch_ratio_max',
                        0.5,
                    )
                ),
            ),
        )
        self.closed_loop_max_depth = max(
            1,
            min(4, int(getattr(diffusion_cfg, 'closed_loop_max_depth', 4))),
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
        self.continuous_recovery_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'continuous_recovery_loss_weight', 1.0)),
        )
        self.encoder_lr_scale = max(
            0.0,
            float(getattr(diffusion_cfg, 'encoder_lr_scale', 0.5)),
        )
        self.safety_energy_enabled = bool(
            getattr(diffusion_cfg, 'safety_energy_enabled', True)
        )
        guidance_cfg = getattr(diffusion_cfg, 'guidance', None)
        configured_guidance_mode = getattr(guidance_cfg, 'mode', None)
        configured_guidance_mode = getattr(
            diffusion_cfg,
            'guidance_mode',
            configured_guidance_mode,
        )
        if configured_guidance_mode is None:
            configured_guidance_mode = 'safe' if self.safety_energy_enabled else 'none'
        self.guidance_mode = str(configured_guidance_mode).lower()
        if self.guidance_mode not in ('none', 'safe', 'ego_stress', 'ego_edit'):
            raise ValueError(
                "diffusion.guidance.mode must be one of none/safe/ego_stress/ego_edit."
            )
        self.safety_topk = max(
            1,
            int(getattr(diffusion_cfg, 'safety_topk', 16)),
        )
        self.guidance_safe_topk = max(
            1,
            int(getattr(guidance_cfg, 'safe_topk', self.safety_topk)),
        )
        self.guidance_ego_stress_topk = max(
            1,
            int(getattr(
                guidance_cfg,
                'ego_stress_topk',
                getattr(diffusion_cfg, 'ego_stress_topk', 64),
            )),
        )
        self.guidance_ego_edit_topk = max(
            1,
            int(getattr(
                guidance_cfg,
                'ego_edit_topk',
                getattr(diffusion_cfg, 'ego_edit_topk', 64),
            )),
        )
        self.guidance_ego_interaction_alpha = max(
            0.0,
            float(getattr(
                guidance_cfg,
                'ego_interaction_alpha',
                getattr(diffusion_cfg, 'ego_interaction_alpha', 1.0),
            )),
        )
        self.guidance_target_event_eta = max(
            0.0,
            float(getattr(
                guidance_cfg,
                'target_event_eta',
                getattr(diffusion_cfg, 'target_event_eta', 1.0),
            )),
        )
        self.guidance_invalid_beta = max(
            0.0,
            float(getattr(guidance_cfg, 'invalid_beta', getattr(diffusion_cfg, 'invalid_beta', 1.0))),
        )
        self.guidance_edit_gamma = max(
            0.0,
            float(getattr(guidance_cfg, 'edit_gamma', getattr(diffusion_cfg, 'edit_gamma', 1.0))),
        )
        self.guidance_near_miss_distance = max(
            0.0,
            float(getattr(guidance_cfg, 'near_miss_distance', getattr(diffusion_cfg, 'near_miss_distance', 2.0))),
        )
        self.guidance_ttc_threshold = max(
            1e-3,
            float(getattr(guidance_cfg, 'ttc_threshold', getattr(diffusion_cfg, 'ttc_threshold', 3.0))),
        )
        self.guidance_offroad_distance = max(
            0.0,
            float(getattr(guidance_cfg, 'offroad_distance', getattr(diffusion_cfg, 'offroad_distance', 4.0))),
        )
        self.guidance_hard_collision_weight = max(
            0.0,
            float(getattr(guidance_cfg, 'hard_collision_weight', getattr(diffusion_cfg, 'hard_collision_weight', 10.0))),
        )
        self.guidance_path_corridor_width = max(
            1e-3,
            float(getattr(guidance_cfg, 'path_corridor_width', getattr(diffusion_cfg, 'path_corridor_width', 2.0))),
        )
        self.guidance_conflict_tta_threshold = max(
            1e-3,
            float(getattr(guidance_cfg, 'conflict_tta_threshold', getattr(diffusion_cfg, 'conflict_tta_threshold', 1.0))),
        )
        ego_agent_id = getattr(
            guidance_cfg,
            'ego_agent_id',
            getattr(diffusion_cfg, 'ego_agent_id', None),
        )
        self.guidance_ego_agent_id = (
            None
            if ego_agent_id is None
            else int(ego_agent_id)
        )
        target_agents = getattr(
            guidance_cfg,
            'target_agents',
            getattr(diffusion_cfg, 'target_agents', ()),
        )
        if target_agents is None:
            target_agents = ()
        if isinstance(target_agents, int):
            target_agents = (target_agents,)
        self.guidance_target_agents = tuple(int(agent) for agent in target_agents)
        target_time_window = getattr(
            guidance_cfg,
            'target_time_window',
            getattr(diffusion_cfg, 'target_time_window', None),
        )
        self.guidance_target_time_window = (
            None
            if target_time_window is None
            else tuple(int(value) for value in target_time_window)
        )
        self.guidance_target_spec = getattr(
            guidance_cfg,
            'target_spec',
            getattr(diffusion_cfg, 'target_spec', 'ego_risk'),
        )
        self.safety_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'safety_energy_weight', 1.0)),
        )
        self.commit_safety_weight = max(
            0.0,
            float(
                getattr(
                    diffusion_cfg,
                    'commit_safety_weight',
                    self.safety_energy_weight,
                )
            ),
        )
        self.proposal_conditioning_enabled = bool(
            getattr(diffusion_cfg, 'proposal_conditioning_enabled', True)
        )
        self.current_state_enabled = bool(
            getattr(diffusion_cfg, 'current_state_enabled', True)
        )
        self.causal_current_state_edges = bool(
            getattr(diffusion_cfg, 'current_state_edges', True)
        )
        self.history_recency_decay = min(
            1.0,
            max(
                1e-3,
                float(getattr(diffusion_cfg, 'history_recency_decay', 0.5)),
            ),
        )
        self.lane_distance_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'lane_distance_energy_weight', 1.0)),
        )
        self.lane_heading_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'lane_heading_energy_weight', 0.5)),
        )
        self.dynamics_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'dynamics_energy_weight', 0.25)),
        )
        self.commit_speed_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'commit_speed_energy_weight', 2.0)),
        )
        self.commit_min_speed_ratio = min(
            1.0,
            max(
                0.0,
                float(getattr(diffusion_cfg, 'commit_min_speed_ratio', 0.55)),
            ),
        )
        self.commit_speed_threshold = max(
            0.0,
            float(getattr(diffusion_cfg, 'commit_speed_threshold', 1.0)),
        )
        self.commit_speed_reference_decay = min(
            1.0,
            max(
                0.0,
                float(getattr(diffusion_cfg, 'commit_speed_reference_decay', 0.92)),
            ),
        )
        self.collision_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'collision_energy_weight', 2.0)),
        )
        self.rollout_score_weights = {
            'ade_8s': float(getattr(diffusion_cfg, 'rollout_score_ade_8s', 1.0)),
            'late_ade': float(getattr(diffusion_cfg, 'rollout_score_late_ade', 1.0)),
            'lane_distance': float(
                getattr(diffusion_cfg, 'rollout_score_lane_distance', 2.0)
            ),
            'lane_heading': float(
                getattr(diffusion_cfg, 'rollout_score_lane_heading', 0.5)
            ),
            'dynamics': float(
                getattr(diffusion_cfg, 'rollout_score_dynamics', 0.25)
            ),
            'collision': float(
                getattr(diffusion_cfg, 'rollout_score_collision', 4.0)
            ),
        }
        self._token_center_vocab_cache = None
        self.current_state_projection = nn.Sequential(
            nn.Linear(5, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.trajectory_energy = TrajectoryEnergy(
            dt=float(getattr(diffusion_cfg, 'trajectory_dt', 0.1)),
            max_acceleration=float(
                getattr(diffusion_cfg, 'max_acceleration', 6.0)
            ),
            max_yaw_rate=float(getattr(diffusion_cfg, 'max_yaw_rate', 1.2)),
            collision_distance=float(
                getattr(diffusion_cfg, 'collision_distance', 2.0)
            ),
        )

        # Executed tokens remain monotonic. Uncommitted tail tokens are carried
        # only as revisable proposal conditioning for the next rollout window.
        self.remask_sampling = False
        self.prefix_constrained_sampling = True
        self.prefix_constrained_training = True
        self.causal_noise_schedule = False
        self.visible_token_corruption_prob = 0.0
        self.visible_token_corruption_probs = ()
        self.self_condition_prob = 0.0
        self.use_proposal_geometry = self.proposal_conditioning_enabled
        self.ar_carry_tail_proposal = bool(
            getattr(diffusion_cfg, 'carry_tail_proposal', True)
        )

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

    def configure_optimizers(self):
        encoder_parameters = [
            parameter
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        ]
        encoder_parameter_ids = {
            id(parameter)
            for parameter in encoder_parameters
        }
        other_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad and id(parameter) not in encoder_parameter_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {
                    'params': encoder_parameters,
                    'lr': self.lr * self.encoder_lr_scale,
                },
                {
                    'params': other_parameters,
                    'lr': self.lr,
                },
            ],
            lr=self.lr,
        )

        def lr_lambda(current_step):
            if current_step + 1 < self.warmup_steps:
                return float(current_step + 1) / float(max(1, self.warmup_steps))
            if current_step >= self.total_steps:
                return 0.0
            return max(
                0.0,
                0.5 * (
                    1.0
                    + math.cos(
                        math.pi * (current_step - self.warmup_steps)
                        / float(max(1, self.total_steps - self.warmup_steps))
                    )
                ),
            )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )
        return [optimizer], [scheduler]

    @staticmethod
    def _causal_reveal_count(step, num_steps, num_chunks):
        if num_steps <= 0 or num_chunks <= 0:
            return 0
        if int(num_steps) != int(num_chunks):
            raise ValueError(
                "Causal diffusion num_steps must equal the number of future chunks."
            )
        step = min(max(int(step), 0), int(num_steps) - 1)
        return step + 1

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

    def _closed_loop_curriculum(self, epoch):
        epoch = max(0, int(epoch))
        maximum = float(
            getattr(self, 'closed_loop_batch_ratio_max', 0.5)
        )
        if epoch <= 3:
            return 0.0, maximum
        return 0.25, maximum

    def _pool_agent_context(self, hist_tokens, hist_mask):
        """Pool ordered history with greater weight on the latest token."""
        num_history = int(hist_tokens.shape[1])
        recency = torch.arange(
            num_history - 1,
            -1,
            -1,
            device=hist_tokens.device,
            dtype=hist_tokens.dtype,
        )
        recency = self.history_recency_decay ** recency
        weights = (
            hist_mask.to(dtype=hist_tokens.dtype)
            * recency.unsqueeze(0)
        ).unsqueeze(-1)
        return (hist_tokens * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

    def _current_state_motion(self, data):
        agent = data['agent']
        current_index = self.num_historical_steps - 1
        previous_index = max(0, current_index - 1)
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

    def _scene_ego_indices(self, data, agent_batch):
        agent = data['agent']
        device = agent_batch.device
        num_agents = int(agent_batch.shape[0])
        num_scenes = (
            int(agent_batch.max().item()) + 1
            if agent_batch.numel() > 0
            else 1
        )
        ego_indices = torch.full(
            (num_scenes,),
            -1,
            dtype=torch.long,
            device=device,
        )
        configured = getattr(self, 'guidance_ego_agent_id', None)
        if configured is not None:
            ego_indices[:] = int(configured)
        elif 'ego_agent_id' in agent:
            value = agent['ego_agent_id']
            if not torch.is_tensor(value):
                value = torch.tensor([int(value)], device=device)
            value = value.to(device=device, dtype=torch.long).reshape(-1)
            if value.numel() == 1:
                ego_indices[:] = value[0]
            elif value.numel() >= num_scenes:
                ego_indices[:] = value[:num_scenes]
        elif 'av_index' in agent:
            value = agent['av_index']
            if not torch.is_tensor(value):
                value = torch.tensor([int(value)], device=device)
            value = value.to(device=device, dtype=torch.long).reshape(-1)
            if value.numel() == 1:
                ego_indices[:] = value[0]
            elif value.numel() >= num_scenes:
                ego_indices[:] = value[:num_scenes]

        for scene_idx in range(num_scenes):
            scene_agents = torch.nonzero(
                agent_batch == scene_idx,
                as_tuple=False,
            ).squeeze(-1)
            if scene_agents.numel() == 0:
                continue
            ego_idx = int(ego_indices[scene_idx].item())
            if ego_idx < 0 or ego_idx >= num_agents or not bool(
                (agent_batch[ego_idx] == scene_idx).item()
            ):
                ego_indices[scene_idx] = scene_agents[0]
        return ego_indices

    def _select_ego_reference_value(
        self,
        value,
        scene_idx,
        ego_idx,
        num_scenes,
        num_agents,
        flatten_chunks=False,
    ):
        if not torch.is_tensor(value):
            return None
        if flatten_chunks and value.dim() == 4:
            if value.shape[0] == num_agents:
                return value[ego_idx].reshape(-1, value.shape[-1])
            if value.shape[0] == num_scenes:
                return value[scene_idx].reshape(-1, value.shape[-1])
            if value.shape[0] == 1:
                return value[0].reshape(-1, value.shape[-1])
        if value.dim() == 1:
            return value
        if value.dim() == 2:
            if value.shape[-1] != 2:
                if value.shape[0] == num_agents:
                    return value[ego_idx]
                if value.shape[0] == num_scenes:
                    return value[scene_idx]
                if value.shape[0] == 1:
                    return value[0]
            return value
        if value.dim() == 3:
            if value.shape[0] == num_agents:
                return value[ego_idx]
            if value.shape[0] == num_scenes:
                return value[scene_idx]
            if value.shape[0] == 1:
                return value[0]
        return None

    @staticmethod
    def _pad_reference_sequence(reference, length, fallback):
        output = fallback.new_zeros((length,) + tuple(fallback.shape[1:]))
        if reference is not None and reference.numel() > 0:
            reference = reference.to(device=fallback.device, dtype=fallback.dtype)
            count = min(length, int(reference.shape[0]))
            if count > 0:
                output[:count] = reference[:count]
                if count < length:
                    output[count:] = output[count - 1]
                return output
        output[:] = fallback[0]
        return output

    def _attach_ego_reference_to_packed(self, data, packed, agent_batch):
        agent = data['agent']
        device = packed['token_positions'].device
        dtype = packed['token_positions'].dtype
        num_agents = int(agent['position'].shape[0])
        num_scenes = (
            int(agent_batch.max().item()) + 1
            if agent_batch.numel() > 0
            else 1
        )
        ego_indices = self._scene_ego_indices(data, agent_batch)
        frame_count = int(self.ar_prediction_tokens) * int(self.ar_token_steps)
        future_start = int(self.num_historical_steps)
        future_end = future_start + frame_count
        ref_positions = agent['position'].new_zeros(num_agents, frame_count, 2).to(
            device=device,
            dtype=dtype,
        )
        ref_headings = agent['heading'].new_zeros(num_agents, frame_count).to(
            device=device,
            dtype=dtype,
        )
        route_source = agent.get('ego_route_corridor', None)
        route_values = None
        route_length = 0
        if torch.is_tensor(route_source):
            if route_source.dim() == 4:
                route_length = int(route_source.shape[1] * route_source.shape[2])
            elif route_source.dim() >= 2:
                route_length = int(route_source.shape[-2])
            if route_length > 0:
                route_values = agent['position'].new_zeros(
                    num_agents,
                    route_length,
                    2,
                ).to(device=device, dtype=dtype)

        for agent_idx in range(num_agents):
            scene_idx = int(agent_batch[agent_idx].item())
            ego_idx = int(ego_indices[scene_idx].item())
            explicit_traj = (
                self._select_ego_reference_value(
                    agent['ego_ref_traj'],
                    scene_idx,
                    ego_idx,
                    num_scenes,
                    num_agents,
                    flatten_chunks=True,
                )
                if 'ego_ref_traj' in agent
                else None
            )
            if explicit_traj is None and 'seed_trajs' in agent:
                explicit_traj = self._select_ego_reference_value(
                    agent['seed_trajs'],
                    scene_idx,
                    ego_idx,
                    num_scenes,
                    num_agents,
                    flatten_chunks=True,
                )
            if explicit_traj is None:
                explicit_traj = agent['position'][
                    ego_idx,
                    future_start:min(future_end, agent['position'].shape[1]),
                    :2,
                ]
            fallback_pos = agent['position'][
                ego_idx,
                min(future_start, agent['position'].shape[1] - 1):min(future_start + 1, agent['position'].shape[1]),
                :2,
            ].to(device=device, dtype=dtype)
            ref_positions[agent_idx] = self._pad_reference_sequence(
                explicit_traj,
                frame_count,
                fallback_pos,
            )

            explicit_heading = (
                self._select_ego_reference_value(
                    agent['ego_ref_heading'],
                    scene_idx,
                    ego_idx,
                    num_scenes,
                    num_agents,
                    flatten_chunks=True,
                )
                if 'ego_ref_heading' in agent
                else None
            )
            if explicit_heading is None:
                explicit_heading = agent['heading'][
                    ego_idx,
                    future_start:min(future_end, agent['heading'].shape[1]),
                ]
            fallback_heading = agent['heading'][
                ego_idx,
                min(future_start, agent['heading'].shape[1] - 1):min(future_start + 1, agent['heading'].shape[1]),
            ].to(device=device, dtype=dtype)
            ref_headings[agent_idx] = self._pad_reference_sequence(
                explicit_heading,
                frame_count,
                fallback_heading,
            )

            if route_values is not None:
                route = self._select_ego_reference_value(
                    route_source,
                    scene_idx,
                    ego_idx,
                    num_scenes,
                    num_agents,
                    flatten_chunks=True,
                )
                route_values[agent_idx] = self._pad_reference_sequence(
                    route,
                    route_length,
                    ref_positions[agent_idx, :1],
                )

        ego_chunks = ref_positions.reshape(
            num_agents,
            int(self.ar_prediction_tokens),
            int(self.ar_token_steps),
            2,
        )
        ego_heading_chunks = ref_headings.reshape(
            num_agents,
            int(self.ar_prediction_tokens),
            int(self.ar_token_steps),
        )
        packed['ego_ref_positions'] = self._pack_agent_window_values(
            ego_chunks,
            packed,
            fill_value=0.0,
        )
        packed['ego_ref_headings'] = self._pack_agent_window_values(
            ego_heading_chunks,
            packed,
            fill_value=0.0,
        )
        if route_values is not None:
            expanded_route = route_values[:, None].expand(
                -1,
                int(self.ar_prediction_tokens),
                -1,
                -1,
            )
            packed['ego_route_corridor'] = self._pack_agent_window_values(
                expanded_route,
                packed,
                fill_value=0.0,
            )

    def _build_diffusion_inputs(self, data, rollout_valid=False):
        result = super()._build_diffusion_inputs(
            data,
            rollout_valid=rollout_valid,
        )
        packed = result[0]
        if packed is None:
            return result

        velocity, current_heading, state_features = self._current_state_motion(data)
        state_embeddings = self.current_state_projection(state_features)
        if self.current_state_enabled:
            for _scene_idx, sequence_idx, agent_indices in packed['agent_maps']:
                for local_idx, agent_idx in enumerate(agent_indices.tolist()):
                    start = local_idx * self.ar_prediction_tokens
                    end = start + self.ar_prediction_tokens
                    packed['agent_context'][sequence_idx, start:end] += (
                        state_embeddings[agent_idx]
                    )

        expanded_velocity = velocity[:, None, :].expand(
            -1,
            self.ar_prediction_tokens,
            -1,
        )
        expanded_heading = current_heading[:, None].expand(
            -1,
            self.ar_prediction_tokens,
        )
        packed['current_velocities'] = self._pack_agent_chunk_values(
            expanded_velocity,
            packed,
        )
        packed['current_headings'] = self._pack_agent_chunk_values(
            expanded_heading,
            packed,
        )
        if 'commit_speed_reference' in data['agent']:
            reference_speed = data['agent']['commit_speed_reference'].to(
                device=velocity.device,
                dtype=velocity.dtype,
            )
            expanded_reference_speed = reference_speed[:, None].expand(
                -1,
                self.ar_prediction_tokens,
            )
            packed['commit_speed_reference'] = self._pack_agent_chunk_values(
                expanded_reference_speed,
                packed,
            )
        agent_batch = result[6]
        self._attach_ego_reference_to_packed(data, packed, agent_batch)
        return result

    def _token_center_vocabs(self):
        if self._token_center_vocab_cache is None:
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
        """Retokenize a world-frame continuation from an arbitrary anchor."""
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

    def _select_topk_by_energy(
        self,
        topk_log_probabilities,
        energies,
        t_value,
        frontier_chunk_ids=None,
    ):
        guidance_scale = float(getattr(
            self,
            'safety_energy_weight',
            0.0,
        )) * (1.0 - float(t_value)) ** 2
        if frontier_chunk_ids is None:
            scale = guidance_scale
        else:
            scale = energies.new_full(
                (energies.shape[0], 1),
                guidance_scale,
            )
            commit_mask = frontier_chunk_ids.to(
                device=energies.device,
            ) == 0
            scale[commit_mask] = float(
                getattr(self, 'commit_safety_weight', guidance_scale)
            )
        adjusted_score = topk_log_probabilities - scale * energies
        return adjusted_score.argmax(dim=-1)

    def _commit_speed_energy(
        self,
        candidate_positions,
        anchor_positions,
        current_velocities,
        frontier_chunk_ids,
        reference_speeds=None,
    ):
        energy = candidate_positions.new_zeros(candidate_positions.shape[:2])
        if current_velocities is None or candidate_positions.numel() == 0:
            return energy
        commit_mask = frontier_chunk_ids.to(
            device=candidate_positions.device,
        ) == 0
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
            current_speed = torch.maximum(
                current_speed,
                reference_speeds.clamp_min(0.0),
            )
        moving_mask = (
            commit_mask
            & (current_speed >= float(getattr(self, 'commit_speed_threshold', 1.0)))
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
            current_speed
            * float(getattr(self, 'commit_min_speed_ratio', 0.45))
        ).unsqueeze(-1)
        speed_deficit = (minimum_speed - candidate_speed).clamp_min(0.0)
        scale = max(float(getattr(self, 'commit_speed_threshold', 1.0)), 1.0)
        energy[moving_mask] = (speed_deficit[moving_mask] / scale) ** 2
        return energy

    def _guidance_topk(self):
        mode = str(getattr(self, 'guidance_mode', 'safe')).lower()
        if mode == 'ego_stress':
            return int(getattr(self, 'guidance_ego_stress_topk', 64))
        if mode == 'ego_edit':
            return int(getattr(self, 'guidance_ego_edit_topk', 64))
        return int(getattr(self, 'guidance_safe_topk', getattr(self, 'safety_topk', 16)))

    def _select_topk_by_guidance(
        self,
        topk_log_probabilities,
        guidance,
        t_value,
        frontier_chunk_ids=None,
    ):
        mode = str(getattr(self, 'guidance_mode', 'safe')).lower()
        if mode == 'safe':
            return self._select_topk_by_energy(
                topk_log_probabilities,
                guidance['safe_energy'],
                t_value,
                frontier_chunk_ids=frontier_chunk_ids,
            )
        if mode in ('ego_stress', 'ego_edit'):
            ego_reward = guidance.get(
                'ego_risk_reward',
                guidance.get('ego_interaction_reward', 0.0),
            )
            target_reward = guidance.get('target_event_reward', 0.0)
            adjusted_score = (
                topk_log_probabilities
                + float(getattr(self, 'guidance_ego_interaction_alpha', 1.0))
                * ego_reward
                + float(getattr(self, 'guidance_target_event_eta', 1.0))
                * target_reward
                - float(getattr(self, 'guidance_invalid_beta', 1.0))
                * guidance.get('invalid_energy', guidance['safe_energy'])
                - float(getattr(self, 'guidance_edit_gamma', 1.0))
                * guidance.get('edit_distance', 0.0)
            )
            return adjusted_score.argmax(dim=-1)
        return topk_log_probabilities.argmax(dim=-1)

    def _zero_guidance_diagnostics(self, reference):
        zero = reference.sum() * 0.0
        return {
            'lane_distance': zero.clone(),
            'lane_heading': zero.clone(),
            'dynamics': zero.clone(),
            'dynamics_energy': zero.clone(),
            'commit_speed_energy': zero.clone(),
            'collision': zero.clone(),
            'min_ttc': zero.clone(),
            'ego_min_ttc': zero.clone(),
            'ego_risk_min_ttc': zero.clone(),
            'ego_min_distance': zero.clone(),
            'ego_required_decel': zero.clone(),
            'ego_path_intrusion_rate': zero.clone(),
            'ego_conflict_tta_error': zero.clone(),
            'ego_risk_reward': zero.clone(),
            'ego_risk_success_rate': zero.clone(),
            'ego_near_miss_success_rate': zero.clone(),
            'near_miss_rate': zero.clone(),
            'hard_collision_rate': zero.clone(),
            'offroad_rate': zero.clone(),
            'edit_distance': zero.clone(),
            'target_success_rate': zero.clone(),
            'target_event_success_rate': zero.clone(),
        }

    def _candidate_edit_distance(
        self,
        candidate_positions,
        topk_ids,
        flat_frontier,
        seed_token_ids=None,
        seed_trajs=None,
    ):
        if seed_trajs is not None:
            seed_trajs = seed_trajs.to(
                device=candidate_positions.device,
                dtype=candidate_positions.dtype,
            )
            if seed_trajs.dim() != 4:
                raise ValueError("seed_trajs must have shape [B,L,T,2].")
            seed_flat = seed_trajs.reshape(
                -1,
                seed_trajs.shape[-2],
                2,
            )[flat_frontier]
            step_count = min(
                int(candidate_positions.shape[-2]),
                int(seed_flat.shape[-2]),
            )
            if step_count <= 0:
                return candidate_positions.new_zeros(topk_ids.shape)
            return torch.norm(
                candidate_positions[:, :, :step_count]
                - seed_flat[:, None, :step_count],
                dim=-1,
            ).mean(dim=-1)
        if seed_token_ids is None:
            return candidate_positions.new_zeros(topk_ids.shape)
        seed_frontier_ids = seed_token_ids.reshape(-1)[flat_frontier]
        return (
            topk_ids != seed_frontier_ids.unsqueeze(-1)
        ).to(dtype=candidate_positions.dtype)

    def _criticality_against_nominal_other_agents(
        self,
        candidate_positions,
        nominal_other_positions,
        candidate_batch,
        candidate_agent_ids,
    ):
        criticality = self.trajectory_energy.criticality_metrics(
            candidate_positions,
            None,
            near_miss_distance=getattr(self, 'guidance_near_miss_distance', 2.0),
            ttc_threshold=getattr(self, 'guidance_ttc_threshold', 3.0),
        )
        for candidate_idx in range(int(candidate_positions.shape[0])):
            other_mask = (
                candidate_batch == candidate_batch[candidate_idx]
            ) & (
                candidate_agent_ids != candidate_agent_ids[candidate_idx]
            )
            if not other_mask.any():
                continue
            row_metrics = self.trajectory_energy.criticality_metrics(
                candidate_positions[candidate_idx:candidate_idx + 1],
                nominal_other_positions[other_mask],
                near_miss_distance=getattr(self, 'guidance_near_miss_distance', 2.0),
                ttc_threshold=getattr(self, 'guidance_ttc_threshold', 3.0),
            )
            for key, value in row_metrics.items():
                criticality[key][candidate_idx] = value[0]
        return criticality

    @staticmethod
    def _horizon_displacement_metrics(
        prediction,
        target,
        valid_mask,
        horizon_steps,
    ):
        horizon_steps = min(
            int(horizon_steps),
            int(prediction.shape[1]),
            int(target.shape[1]),
            int(valid_mask.shape[1]),
        )
        if horizon_steps <= 0:
            zero = prediction.sum() * 0.0
            return zero, zero
        prediction = prediction[:, :horizon_steps]
        target = target[:, :horizon_steps]
        valid_mask = valid_mask[:, :horizon_steps].bool()
        distance = torch.norm(prediction - target, dim=-1)
        if valid_mask.any():
            ade = distance[valid_mask].mean()
        else:
            ade = distance.sum() * 0.0

        step_ids = torch.arange(
            horizon_steps,
            device=prediction.device,
        ).unsqueeze(0).expand_as(valid_mask)
        last_valid = step_ids.masked_fill(~valid_mask, -1).max(dim=1).values
        has_valid = last_valid >= 0
        if has_valid.any():
            agent_ids = torch.nonzero(has_valid, as_tuple=False).squeeze(-1)
            fde = distance[agent_ids, last_valid[has_valid]].mean()
        else:
            fde = distance.sum() * 0.0
        return ade, fde

    def _rollout_score(self, ade_8s, late_ade, energies):
        weights = self.rollout_score_weights
        return (
            weights['ade_8s'] * ade_8s
            + weights['late_ade'] * late_ade
            + weights['lane_distance'] * energies['lane_distance']
            + weights['lane_heading'] * energies['lane_heading']
            + weights['dynamics'] * energies['dynamics']
            + weights['collision'] * energies['collision']
        )

    def _guided_frontier_tokens(
        self,
        logits,
        probabilities,
        frontier,
        packed,
        sampled,
        t_value,
        seed_token_ids=None,
        seed_trajs=None,
    ):
        frontier_logits = logits[frontier]
        topk = min(self._guidance_topk(), int(frontier_logits.shape[-1]))
        topk_log_probabilities, topk_ids = F.log_softmax(
            frontier_logits,
            dim=-1,
        ).topk(topk, dim=-1)
        flat_frontier = torch.nonzero(
            frontier.reshape(-1),
            as_tuple=False,
        ).squeeze(-1)
        geometry_known = packed['valid_mask'] & (sampled != self.mask_token_id)
        anchor_positions, anchor_headings, _geometry_confidence = (
            self._refresh_token_geometry(
                sampled,
                packed,
                geometry_known_mask=geometry_known,
            )
        )
        flat_positions = anchor_positions.reshape(-1, 2)[flat_frontier]
        flat_headings = anchor_headings.reshape(-1)[flat_frontier]
        flat_agent_types = packed['agent_type_ids'].reshape(-1)[flat_frontier]
        frontier_chunk_ids = packed['chunk_ids'].reshape(-1)[flat_frontier]
        num_frontier = int(flat_frontier.numel())
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
            num_frontier,
            topk,
            self.ar_token_steps,
            2,
        )
        candidate_headings = candidate_headings.reshape(
            num_frontier,
            topk,
            self.ar_token_steps,
        )
        sequence_length = packed['valid_mask'].shape[1]
        candidate_batch = torch.div(
            flat_frontier,
            sequence_length,
            rounding_mode='floor',
        )
        lane_distance, lane_heading = self.trajectory_energy.lane_energy(
            candidate_positions,
            candidate_headings,
            packed.get('map_positions'),
            packed.get('map_orientations'),
            candidate_batch=candidate_batch,
            map_batch=packed.get('map_batch'),
            map_valid_mask=packed.get('map_valid_mask'),
        )
        dynamics = self.trajectory_energy.dynamics_energy(
            candidate_positions,
            candidate_headings,
        )
        commit_mask = frontier_chunk_ids == 0
        commit_speed_energy = candidate_positions.new_zeros(
            candidate_positions.shape[:2]
        )
        if commit_mask.any() and packed.get('current_velocities') is not None:
            current_velocities = packed['current_velocities'].reshape(
                -1,
                2,
            )[flat_frontier]
            reference_speeds = None
            if packed.get('commit_speed_reference') is not None:
                reference_speeds = packed['commit_speed_reference'].reshape(
                    -1,
                )[flat_frontier]
            current_headings = packed.get(
                'current_headings',
                anchor_headings,
            ).reshape(-1)[flat_frontier]
            transition_dynamics = self.trajectory_energy.dynamics_energy(
                candidate_positions,
                candidate_headings,
                current_positions=flat_positions,
                current_velocities=current_velocities,
                current_headings=current_headings,
            )
            dynamics[commit_mask] = transition_dynamics[commit_mask]
            commit_speed_energy = self._commit_speed_energy(
                candidate_positions,
                flat_positions,
                current_velocities,
                frontier_chunk_ids,
                reference_speeds=reference_speeds,
            )

        preliminary_energy = (
            self.lane_distance_energy_weight * lane_distance
            + self.lane_heading_energy_weight * lane_heading
            + self.dynamics_energy_weight * dynamics
            + self.commit_speed_energy_weight * commit_speed_energy
        )
        preliminary_selection = self._select_topk_by_energy(
            topk_log_probabilities,
            preliminary_energy,
            t_value,
            frontier_chunk_ids=frontier_chunk_ids,
        )
        row = torch.arange(
            num_frontier,
            device=topk_ids.device,
        )
        nominal_other = candidate_positions[row, preliminary_selection]
        frontier_agent_ids = packed['token_agent_ids'].reshape(-1)[flat_frontier]
        collision = self.trajectory_energy.collision_energy(
            candidate_positions,
            nominal_other,
            candidate_batch=candidate_batch,
            other_batch=candidate_batch,
            candidate_agent_ids=frontier_agent_ids,
            other_agent_ids=frontier_agent_ids,
        )
        criticality = self._criticality_against_nominal_other_agents(
            candidate_positions,
            nominal_other,
            candidate_batch,
            frontier_agent_ids,
        )
        ego_ref_positions = packed.get('ego_ref_positions')
        if ego_ref_positions is not None:
            ego_positions = ego_ref_positions.reshape(
                -1,
                ego_ref_positions.shape[-2],
                2,
            )[flat_frontier]
        else:
            ego_positions = None
        ego_ref_headings = packed.get('ego_ref_headings')
        if ego_ref_headings is not None:
            ego_headings = ego_ref_headings.reshape(
                -1,
                ego_ref_headings.shape[-1],
            )[flat_frontier]
        else:
            ego_headings = None
        ego_route_corridor = packed.get('ego_route_corridor')
        if ego_route_corridor is not None:
            ego_route = ego_route_corridor.reshape(
                -1,
                ego_route_corridor.shape[-2],
                2,
            )[flat_frontier]
        else:
            ego_route = None
        guidance_target_spec = getattr(self, 'guidance_target_spec', 'ego_risk')
        ego_metrics = self.trajectory_energy.ego_interaction_metrics(
            candidate_positions,
            ego_positions,
            ego_headings=ego_headings,
            ego_route_corridor=ego_route,
            path_corridor_width=getattr(self, 'guidance_path_corridor_width', 2.0),
            near_miss_distance=getattr(self, 'guidance_near_miss_distance', 2.0),
            ttc_threshold=getattr(self, 'guidance_ttc_threshold', 3.0),
            conflict_tta_threshold=getattr(self, 'guidance_conflict_tta_threshold', 1.0),
            target_spec=guidance_target_spec,
        )
        total_energy = (
            self.lane_distance_energy_weight * lane_distance
            + self.lane_heading_energy_weight * lane_heading
            + self.dynamics_energy_weight * dynamics
            + self.commit_speed_energy_weight * commit_speed_energy
            + self.collision_energy_weight * collision
        )
        offroad = lane_distance > float(getattr(self, 'guidance_offroad_distance', 4.0))
        hard_collision = criticality['hard_collision'] | ego_metrics['hard_collision']
        invalid_energy = (
            total_energy
            + offroad.to(dtype=total_energy.dtype) * self.lane_distance_energy_weight
            + hard_collision.to(dtype=total_energy.dtype)
            * float(getattr(self, 'guidance_hard_collision_weight', 10.0))
        )
        edit_distance = self._candidate_edit_distance(
            candidate_positions,
            topk_ids,
            flat_frontier,
            seed_token_ids=seed_token_ids,
            seed_trajs=seed_trajs,
        )
        risk_spec = str(guidance_target_spec or '').lower().replace('-', '_') in (
            'ego_risk',
            'risk',
            'low_ttc',
            'ttc',
        )
        target_event_reward = ego_metrics['target_event_reward']
        if risk_spec:
            target_event_reward = torch.zeros_like(target_event_reward)
        guidance_terms = {
            'safe_energy': total_energy,
            'invalid_energy': invalid_energy,
            'critical_reward': criticality['critical_reward'],
            'ego_interaction_reward': ego_metrics['ego_interaction_reward'],
            'ego_risk_reward': ego_metrics['ego_risk_reward'],
            'target_event_reward': target_event_reward,
            'edit_distance': edit_distance,
        }
        selected_topk = self._select_topk_by_guidance(
            topk_log_probabilities,
            guidance_terms,
            t_value,
            frontier_chunk_ids=frontier_chunk_ids,
        )
        selected_ids = topk_ids[row, selected_topk]
        selected_confidence = probabilities[frontier].gather(
            -1,
            selected_ids.unsqueeze(-1),
        ).squeeze(-1)
        selected_ego_ttc = ego_metrics['ego_min_ttc'][row, selected_topk]
        selected_ego_ttc_for_min = torch.nan_to_num(
            selected_ego_ttc,
            nan=float('inf'),
            posinf=float('inf'),
            neginf=float('inf'),
        )
        finite_selected_ego_ttc = selected_ego_ttc_for_min[
            torch.isfinite(selected_ego_ttc_for_min)
        ]
        if finite_selected_ego_ttc.numel() > 0:
            ego_risk_min_ttc = finite_selected_ego_ttc.amin()
        else:
            ego_risk_min_ttc = torch.full_like(
                selected_ego_ttc.sum(),
                float('inf'),
            )
        diagnostics = {
            'lane_distance': lane_distance[row, selected_topk].mean(),
            'lane_heading': lane_heading[row, selected_topk].mean(),
            'dynamics': dynamics[row, selected_topk].mean(),
            'dynamics_energy': dynamics[row, selected_topk].mean(),
            'commit_speed_energy': commit_speed_energy[row, selected_topk].mean(),
            'collision': collision[row, selected_topk].mean(),
            'min_ttc': (
                finite_selected_ego_ttc.mean()
                if finite_selected_ego_ttc.numel() > 0
                else torch.full_like(selected_ego_ttc.sum(), float('inf'))
            ),
            'ego_min_ttc': (
                finite_selected_ego_ttc.mean()
                if finite_selected_ego_ttc.numel() > 0
                else torch.full_like(selected_ego_ttc.sum(), float('inf'))
            ),
            'ego_risk_min_ttc': ego_risk_min_ttc,
            'ego_min_distance': ego_metrics['ego_min_distance'][
                row,
                selected_topk,
            ].mean(),
            'ego_required_decel': ego_metrics['ego_required_decel'][
                row,
                selected_topk,
            ].mean(),
            'ego_path_intrusion_rate': ego_metrics['ego_path_intrusion_rate'][
                row,
                selected_topk,
            ].mean(),
            'ego_conflict_tta_error': ego_metrics['ego_conflict_tta_error'][
                row,
                selected_topk,
            ].mean(),
            'ego_risk_reward': ego_metrics['ego_risk_reward'][
                row,
                selected_topk,
            ].mean(),
            'ego_risk_success_rate': ego_metrics['ego_risk_success'][
                row,
                selected_topk,
            ].to(dtype=total_energy.dtype).mean(),
            'near_miss_rate': ego_metrics['ego_near_miss_success'][
                row,
                selected_topk,
            ].to(dtype=total_energy.dtype).mean(),
            'ego_near_miss_success_rate': ego_metrics['ego_near_miss_success'][
                row,
                selected_topk,
            ].to(dtype=total_energy.dtype).mean(),
            'hard_collision_rate': hard_collision[
                row,
                selected_topk,
            ].to(dtype=total_energy.dtype).mean(),
            'offroad_rate': offroad[row, selected_topk].to(
                dtype=total_energy.dtype,
            ).mean(),
            'edit_distance': edit_distance[row, selected_topk].mean(),
            'target_success_rate': ego_metrics['target_event_success'][
                row,
                selected_topk,
            ].to(dtype=total_energy.dtype).mean(),
            'target_event_success_rate': ego_metrics['target_event_success'][
                row,
                selected_topk,
            ].to(dtype=total_energy.dtype).mean(),
        }
        return selected_ids, selected_confidence, diagnostics

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

    def _pack_agent_chunk_values(self, agent_values, packed, fill_value=0):
        batch_size, sequence_length = packed['valid_mask'].shape
        output_shape = (
            batch_size,
            sequence_length,
            *agent_values.shape[2:],
        )
        result = torch.full(
            output_shape,
            fill_value,
            dtype=agent_values.dtype,
            device=agent_values.device,
        )
        for _scene_idx, sequence_idx, agent_indices in packed['agent_maps']:
            for local_idx, agent_idx in enumerate(agent_indices.tolist()):
                start = local_idx * self.ar_prediction_tokens
                end = start + self.ar_prediction_tokens
                result[sequence_idx, start:end] = agent_values[agent_idx]
        return result

    def _window_from_token_source(self, token_source, round_idx, fill_value=0):
        num_agents = int(token_source.shape[0])
        start = int(getattr(self, 'ar_history_tokens', 0)) + int(round_idx)
        if token_source.shape[1] <= start:
            start = int(round_idx)
        end = start + int(self.ar_prediction_tokens)
        window = torch.full(
            (num_agents, int(self.ar_prediction_tokens)),
            fill_value,
            dtype=token_source.dtype,
            device=token_source.device,
        )
        available = token_source[:, start:min(end, token_source.shape[1])]
        if available.numel() > 0:
            window[:, :available.shape[1]] = available
        return window

    def _window_from_seed_trajs(self, seed_trajs, round_idx):
        num_agents = int(seed_trajs.shape[0])
        if seed_trajs.dim() == 4:
            start = int(getattr(self, 'ar_history_tokens', 0)) + int(round_idx)
            if seed_trajs.shape[1] <= start:
                start = int(round_idx)
            end = start + int(self.ar_prediction_tokens)
            window = seed_trajs.new_zeros(
                num_agents,
                int(self.ar_prediction_tokens),
                int(self.ar_token_steps),
                2,
            )
            available = seed_trajs[:, start:min(end, seed_trajs.shape[1])]
            if available.numel() > 0:
                step_count = min(int(self.ar_token_steps), int(available.shape[2]))
                window[:, :available.shape[1], :step_count] = available[:, :, :step_count, :2]
            return window
        if seed_trajs.dim() != 3:
            raise ValueError("seed_trajs must have shape [A,T,2] or [A,K,S,2].")
        start = int(round_idx) * int(self.ar_token_steps)
        frame_count = int(self.ar_prediction_tokens) * int(self.ar_token_steps)
        end = start + frame_count
        window_frames = seed_trajs.new_zeros(num_agents, frame_count, 2)
        available = seed_trajs[:, start:min(end, seed_trajs.shape[1]), :2]
        if available.numel() > 0:
            window_frames[:, :available.shape[1]] = available
        return window_frames.reshape(
            num_agents,
            int(self.ar_prediction_tokens),
            int(self.ar_token_steps),
            2,
        )

    def _window_from_seed_headings(self, seed_headings, round_idx):
        num_agents = int(seed_headings.shape[0])
        if seed_headings.dim() == 3:
            start = int(getattr(self, 'ar_history_tokens', 0)) + int(round_idx)
            if seed_headings.shape[1] <= start:
                start = int(round_idx)
            end = start + int(self.ar_prediction_tokens)
            window = seed_headings.new_zeros(
                num_agents,
                int(self.ar_prediction_tokens),
                int(self.ar_token_steps),
            )
            available = seed_headings[:, start:min(end, seed_headings.shape[1])]
            if available.numel() > 0:
                step_count = min(int(self.ar_token_steps), int(available.shape[2]))
                window[:, :available.shape[1], :step_count] = available[:, :, :step_count]
            return window
        if seed_headings.dim() != 2:
            raise ValueError("seed headings must have shape [A,T] or [A,K,S].")
        start = int(round_idx) * int(self.ar_token_steps)
        frame_count = int(self.ar_prediction_tokens) * int(self.ar_token_steps)
        end = start + frame_count
        window_frames = seed_headings.new_zeros(num_agents, frame_count)
        available = seed_headings[:, start:min(end, seed_headings.shape[1])]
        if available.numel() > 0:
            window_frames[:, :available.shape[1]] = available
        return window_frames.reshape(
            num_agents,
            int(self.ar_prediction_tokens),
            int(self.ar_token_steps),
        )

    def _apply_guidance_seed_commit_overrides(
        self,
        data,
        round_idx,
        edit_window_mask,
        commit_traj,
        commit_head,
        commit_valid_frames,
        commit_token_pos,
        commit_token_heading,
        current_pos,
        current_heading,
    ):
        if edit_window_mask is None or 'seed_trajs' not in data['agent']:
            return (
                commit_traj,
                commit_head,
                commit_valid_frames,
                commit_token_pos,
                commit_token_heading,
                current_pos,
                current_heading,
            )
        commit_tokens = int(getattr(self, 'ar_commit_tokens', 1))
        locked = ~edit_window_mask[:, :commit_tokens].bool()
        if not locked.any():
            return (
                commit_traj,
                commit_head,
                commit_valid_frames,
                commit_token_pos,
                commit_token_heading,
                current_pos,
                current_heading,
            )
        seed_window = self._window_from_seed_trajs(
            data['agent']['seed_trajs'],
            round_idx,
        ).to(device=commit_traj.device, dtype=commit_traj.dtype)
        seed_commit = seed_window[:, :commit_tokens].reshape(
            commit_traj.shape[0],
            commit_tokens * int(self.ar_token_steps),
            2,
        )
        frame_locked = locked.unsqueeze(-1).expand(
            -1,
            -1,
            int(self.ar_token_steps),
        ).reshape(commit_traj.shape[0], -1)
        commit_traj = torch.where(
            frame_locked.unsqueeze(-1),
            seed_commit,
            commit_traj,
        )
        if 'seed_headings' in data['agent']:
            seed_heading_window = self._window_from_seed_headings(
                data['agent']['seed_headings'],
                round_idx,
            )
        else:
            future_start = int(self.num_historical_steps)
            future_heading = data['agent']['heading'][:, future_start:]
            seed_heading_window = self._window_from_seed_headings(
                future_heading,
                round_idx,
            )
        seed_heading_window = seed_heading_window.to(
            device=commit_head.device,
            dtype=commit_head.dtype,
        )
        seed_commit_head = seed_heading_window[:, :commit_tokens].reshape(
            commit_head.shape[0],
            commit_tokens * int(self.ar_token_steps),
        )
        commit_head = torch.where(
            frame_locked,
            seed_commit_head,
            commit_head,
        )
        seed_valid = torch.ones_like(commit_valid_frames)
        if 'valid_mask' in data['agent']:
            future_start = int(self.num_historical_steps)
            future_valid = data['agent']['valid_mask'][:, future_start:]
            seed_valid_window = self._window_from_seed_headings(
                future_valid.to(dtype=commit_valid_frames.dtype),
                round_idx,
            ).bool()
            seed_valid = seed_valid_window[:, :commit_tokens].reshape_as(
                commit_valid_frames,
            ).to(device=commit_valid_frames.device)
        commit_valid_frames = torch.where(
            frame_locked,
            seed_valid,
            commit_valid_frames,
        )
        seed_token_pos = seed_window[:, :commit_tokens, -1]
        commit_token_pos = torch.where(
            locked.unsqueeze(-1),
            seed_token_pos,
            commit_token_pos,
        )
        seed_token_heading = seed_heading_window[:, :commit_tokens, -1]
        commit_token_heading = torch.where(
            locked,
            seed_token_heading,
            commit_token_heading,
        )
        latest_locked = locked[:, -1]
        current_pos = torch.where(
            latest_locked.unsqueeze(-1),
            seed_token_pos[:, -1],
            current_pos,
        )
        current_heading = torch.where(
            latest_locked,
            seed_token_heading[:, -1],
            current_heading,
        )
        return (
            commit_traj,
            commit_head,
            commit_valid_frames,
            commit_token_pos,
            commit_token_heading,
            current_pos,
            current_heading,
        )

    def _build_guidance_edit_controls(
        self,
        data,
        round_idx,
        future_valid,
        generation_agents,
    ):
        agent = data['agent']
        token_source = agent.get('seed_token_ids', agent['token_idx']).long()
        seed_tokens = self._window_from_token_source(
            token_source,
            round_idx,
            fill_value=0,
        )
        editable = future_valid.bool() & generation_agents[:, None].bool()
        target_mask = self._guidance_target_agent_mask(
            data,
            generation_agents,
        )
        editable = editable & target_mask[:, None]
        ego_mask = self._guidance_ego_agent_mask(
            data,
            generation_agents,
        )
        editable = editable & ~ego_mask[:, None]

        target_time_window = getattr(self, 'guidance_target_time_window', None)
        if target_time_window is not None:
            start, end = target_time_window
            chunk_ids = (
                torch.arange(
                    int(self.ar_prediction_tokens),
                    device=future_valid.device,
                )
                + int(round_idx)
            )
            time_mask = (chunk_ids >= int(start)) & (chunk_ids < int(end))
            editable = editable & time_mask.unsqueeze(0)

        if 'edit_mask' in agent:
            explicit = self._window_from_token_source(
                agent['edit_mask'].bool(),
                round_idx,
                fill_value=False,
            ).bool()
            editable = editable & explicit
        return seed_tokens, editable

    def _guidance_target_agent_mask(self, data, fallback_mask):
        target_agents = tuple(getattr(self, 'guidance_target_agents', ()))
        if target_agents:
            target_mask = torch.zeros_like(fallback_mask, dtype=torch.bool)
            for agent_idx in target_agents:
                if 0 <= int(agent_idx) < int(target_mask.shape[0]):
                    target_mask[int(agent_idx)] = True
            return target_mask
        agent = data['agent']
        if 'target_agents' in agent:
            value = agent['target_agents']
            if value.dtype == torch.bool and value.shape == fallback_mask.shape:
                return value.to(device=fallback_mask.device, dtype=torch.bool)
            target_mask = torch.zeros_like(fallback_mask, dtype=torch.bool)
            for agent_idx in value.reshape(-1).tolist():
                if 0 <= int(agent_idx) < int(target_mask.shape[0]):
                    target_mask[int(agent_idx)] = True
            return target_mask
        return fallback_mask.bool()

    def _guidance_ego_agent_mask(self, data, fallback_mask):
        agent = data['agent']
        if 'batch' in agent:
            agent_batch = agent['batch']
        else:
            agent_batch = torch.zeros(
                int(fallback_mask.shape[0]),
                dtype=torch.long,
                device=fallback_mask.device,
            )
        ego_indices = self._scene_ego_indices(data, agent_batch)
        ego_mask = torch.zeros_like(fallback_mask, dtype=torch.bool)
        for ego_idx in ego_indices.reshape(-1).tolist():
            if 0 <= int(ego_idx) < int(ego_mask.shape[0]):
                ego_mask[int(ego_idx)] = True
        return ego_mask.to(device=fallback_mask.device) & fallback_mask.bool()

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

    def _build_causal_training_view(self, data):
        epoch = int(getattr(self, 'current_epoch', 0))
        perturb_prob, rollout_prob = self._closed_loop_curriculum(epoch)
        rollout_draw = torch.rand((), device=data['agent']['token_idx'].device)
        if rollout_prob > 0.0 and rollout_draw < rollout_prob:
            rollout_depth = int(torch.randint(
                1,
                self.closed_loop_max_depth + 1,
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

    def training_step(self, data, batch_idx):
        del batch_idx
        data = self._prepare_batch(data)
        if self.ar_rolling_anchor_training:
            data, retokenization, state_mode, rollout_depth = (
                self._build_causal_training_view(data)
            )
        else:
            retokenization = self._clean_retokenization_metadata(data)
            state_mode = 'clean'
            rollout_depth = 0
        (
            packed,
            summary,
            _ft,
            _fv,
            _generation_agents,
            _supervision_agents,
            _agent_batch,
        ) = self._build_diffusion_inputs(data)
        if packed is None:
            zero_loss = self._zero_connected_loss()
            self.log(
                'train_empty_diffusion_batch',
                zero_loss.detach().new_ones(()),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            return zero_loss
        for key, values in retokenization.items():
            packed[key] = self._pack_agent_chunk_values(values, packed)

        diffusion_loss, mask_acc = self._compute_diffusion_loss(packed, summary)
        ntp_loss = self._compute_optional_ntp_loss(data, diffusion_loss)
        loss = diffusion_loss + self.ntp_aux_loss_weight * ntp_loss
        mode_values = {'clean': 0.0, 'perturb': 1.0, 'rollout': 2.0}
        self.log(
            'train_state_mode',
            loss.new_tensor(mode_values[state_mode]),
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            'train_rollout_depth',
            loss.new_tensor(float(rollout_depth)),
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            'train_empty_diffusion_batch',
            loss.new_zeros(()),
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            'train_loss',
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            'diffusion_loss',
            diffusion_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            'ntp_loss',
            ntp_loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            'train_mask_acc',
            mask_acc,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def _compute_diffusion_loss(self, packed, summary):
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

        recovery_loss = self._continuous_recovery_loss(
            logits,
            packed,
            masked_supervision=frontier_raw_mask,
        )
        if frontier_mask.any():
            log_p = F.log_softmax(logits, dim=-1)
            nll = -log_p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
            loss = (
                nll[frontier_mask].mean()
                * self.causal_frontier_loss_weight
            )
            acc = (
                logits[frontier_mask].argmax(-1) == gt[frontier_mask]
            ).float().mean()
        else:
            loss = logits.sum() * 0.0
            acc = logits.new_zeros(())
        loss = loss + getattr(
            self,
            'continuous_recovery_loss_weight',
            0.0,
        ) * recovery_loss

        if self.training:
            valid_count = valid_mask.float().sum().clamp_min(1.0)
            self.log(
                'train_causal_mask_frac',
                mask.float().sum() / valid_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_causal_frontier_frac',
                frontier_mask.float().sum() / valid_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            sampled_frontiers = frontier_ids >= 0
            frontier_denominator = sampled_frontiers.float().sum().clamp_min(1.0)
            for chunk_idx in range(self.num_future_chunks):
                self.log(
                    f'train_frontier_chunk_{chunk_idx}_frac',
                    (
                        (frontier_ids == chunk_idx).float().sum()
                        / frontier_denominator
                    ),
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    batch_size=1,
                )
            invalid_count = (
                raw_loss_mask_base & ~retokenization_valid
            ).float().sum()
            supervision_count = raw_loss_mask_base.float().sum().clamp_min(1.0)
            self.log(
                'train_retokenization_invalid_frac',
                invalid_count / supervision_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_continuous_recovery_loss',
                recovery_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
        return loss, acc

    @torch.no_grad()
    def _diffusion_sample(
        self,
        summary,
        token_positions,
        token_headings,
        token_agent_ids,
        chunk_ids,
        valid_mask,
        agent_context,
        agent_type_ids,
        map_context=None,
        map_positions=None,
        map_orientations=None,
        map_batch=None,
        map_valid_mask=None,
        agent_shape_embeddings=None,
        packed=None,
        return_trace=False,
        initial_proposal_token_ids=None,
        initial_proposal_confidence=None,
        seed_token_ids=None,
        seed_trajs=None,
        editable_mask=None,
    ):
        batch_size, sequence_length = valid_mask.shape
        device = summary.device
        confidence = summary.new_zeros((batch_size, sequence_length))
        if seed_token_ids is not None:
            seed_token_ids = seed_token_ids.to(
                device=device,
                dtype=torch.long,
            )
            if seed_token_ids.shape != valid_mask.shape:
                raise ValueError("seed_token_ids must match valid_mask shape.")
            if editable_mask is None:
                editable_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
            editable_mask = editable_mask.to(device=device, dtype=torch.bool) & valid_mask
            sampled = seed_token_ids.clone().masked_fill(~valid_mask, 0)
            sampled[editable_mask] = self.mask_token_id
            confidence = confidence.masked_fill(valid_mask & ~editable_mask, 1.0)
        else:
            editable_mask = valid_mask
            sampled = torch.full(
                (batch_size, sequence_length),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
        trace = []
        energy_totals = self._zero_guidance_diagnostics(summary)
        energy_min_keys = {'ego_risk_min_ttc'}
        energy_minimums = {
            key: torch.full_like(summary.sum(), float('inf'))
            for key in energy_min_keys
            if key in energy_totals
        }
        energy_events = 0

        for step in range(self.diffusion_num_steps):
            reveal_count = self._causal_reveal_count(
                step,
                self.diffusion_num_steps,
                self.num_future_chunks,
            )
            newly_revealed = 0
            mask = valid_mask & editable_mask & (sampled == self.mask_token_id)
            prefix_frontier = self._prefix_frontier_mask(
                mask,
                valid_mask,
                token_agent_ids,
                chunk_ids,
            )
            frontier = prefix_frontier & (chunk_ids == step)
            if frontier.any():
                t_value = max(1.0 - step / self.diffusion_num_steps, self.min_t)
                t_batch = torch.full(
                    (batch_size,),
                    t_value,
                    device=device,
                    dtype=summary.dtype,
                )
                if packed is not None:
                    logits = self._decode_diffusion_logits(
                        sampled,
                        packed,
                        summary,
                        t_batch,
                        geometry_known_mask=(~mask) & valid_mask,
                        proposal_token_ids=initial_proposal_token_ids,
                        proposal_confidence=initial_proposal_confidence,
                    )
                else:
                    logits = self.diffusion_decoder(
                        noisy_token_ids=sampled,
                        token_positions=token_positions,
                        token_headings=token_headings,
                        token_agent_ids=token_agent_ids,
                        noisy_token_chunk_ids=chunk_ids,
                        scene_summary=summary,
                        t=t_batch,
                        valid_mask=valid_mask,
                        agent_context=agent_context,
                        agent_type_ids=agent_type_ids,
                        agent_shape_embeddings=agent_shape_embeddings,
                        physical_token_embeddings=self._physical_token_embeddings(
                            sampled,
                            agent_type_ids,
                        ),
                        map_context=map_context,
                        map_positions=map_positions,
                        map_orientations=map_orientations,
                        map_batch=map_batch,
                        map_valid_mask=map_valid_mask,
                    )
                probabilities = F.softmax(
                    logits / self.remask_confidence_temperature,
                    dim=-1,
                )
                if (
                    str(
                        getattr(
                            self,
                            'guidance_mode',
                            'safe' if getattr(self, 'safety_energy_enabled', False) else 'none',
                        )
                    ).lower() != 'none'
                    and packed is not None
                    and 'valid_mask' in packed
                ):
                    (
                        frontier_ids,
                        frontier_confidence,
                        energy_diagnostics,
                    ) = self._guided_frontier_tokens(
                        logits,
                        probabilities,
                        frontier,
                        packed,
                        sampled,
                        t_value,
                        seed_token_ids=seed_token_ids,
                        seed_trajs=seed_trajs,
                    )
                    for key in energy_totals:
                        if key in energy_minimums:
                            value = torch.nan_to_num(
                                energy_diagnostics[key],
                                nan=float('inf'),
                                posinf=float('inf'),
                                neginf=float('inf'),
                            )
                            energy_minimums[key] = torch.minimum(
                                energy_minimums[key],
                                value,
                            )
                        else:
                            energy_totals[key] = (
                                energy_totals[key] + energy_diagnostics[key]
                            )
                    energy_events += 1
                else:
                    frontier_probabilities = probabilities[frontier].clamp_min(
                        1e-10
                    )
                    frontier_ids = torch.multinomial(
                        frontier_probabilities,
                        1,
                    ).squeeze(-1)
                    frontier_confidence = frontier_probabilities.gather(
                        -1,
                        frontier_ids.unsqueeze(-1),
                    ).squeeze(-1)
                sampled[frontier] = frontier_ids
                confidence[frontier] = frontier_confidence
                newly_revealed = int(frontier.sum().item())

            if return_trace:
                remaining = valid_mask & (sampled == self.mask_token_id)
                trace.append({
                    'step': step,
                    'reveal_count': reveal_count,
                    'newly_revealed': newly_revealed,
                    'masked_after': int(remaining.sum().item()),
                    'remasked': 0,
                })

        remaining = valid_mask & editable_mask & (sampled == self.mask_token_id)
        if remaining.any():
            raise RuntimeError("Causal diffusion sampling ended with masked valid tokens.")
        denominator = max(energy_events, 1)
        self._last_sampling_energy = {
            key: (value / denominator).detach()
            for key, value in energy_totals.items()
        }
        for key, value in energy_minimums.items():
            self._last_sampling_energy[key] = value.detach()
        if hasattr(self, '_sampling_energy_accumulator'):
            self._sampling_energy_accumulator.append(self._last_sampling_energy)
        sampled = sampled.masked_fill(~valid_mask, 0)
        confidence = confidence.masked_fill(~valid_mask, 0.0)
        if return_trace:
            return sampled, confidence, trace
        return sampled, confidence

    @torch.no_grad()
    def inference(self, data):
        self._sampling_energy_accumulator = []
        output = super().inference(data)
        if output is None:
            return None
        if self._sampling_energy_accumulator:
            guidance_metrics = {}
            for key in self._sampling_energy_accumulator[0]:
                values = torch.stack([
                    entry[key]
                    for entry in self._sampling_energy_accumulator
                ])
                if key == 'ego_risk_min_ttc':
                    values = torch.nan_to_num(
                        values,
                        nan=float('inf'),
                        posinf=float('inf'),
                        neginf=float('inf'),
                    )
                    finite_values = values[torch.isfinite(values)]
                    if finite_values.numel() > 0:
                        guidance_metrics[key] = finite_values.amin()
                    else:
                        guidance_metrics[key] = torch.full_like(
                            values.sum(),
                            float('inf'),
                        )
                else:
                    guidance_metrics[key] = values.mean()
        else:
            zero = output['pred_traj'].sum() * 0.0
            guidance_metrics = self._zero_guidance_diagnostics(zero)
        output['guidance_metrics'] = guidance_metrics
        output['safety_energy'] = {
            key: guidance_metrics[key]
            for key in ('lane_distance', 'lane_heading', 'dynamics', 'collision')
        }
        return output

    @torch.no_grad()
    def _validation_retokenization_invalid_rate(self, data, agent_mask):
        future_start = self.num_historical_steps
        future_end = future_start + self.num_future_steps
        selected_positions = data['agent']['position'][
            agent_mask,
            future_start:future_end,
            :2,
        ]
        selected_valid = data['agent']['valid_mask'][
            agent_mask,
            future_start:future_end,
        ]
        if selected_positions.numel() == 0:
            return data['agent']['position'].new_zeros(())
        future_positions = selected_positions.reshape(
            -1,
            self.num_future_steps // self.ar_token_steps,
            self.ar_token_steps,
            2,
        )
        future_valid = selected_valid.reshape(
            -1,
            self.num_future_steps // self.ar_token_steps,
            self.ar_token_steps,
        )
        _ids, _errors, retokenization_valid, _endpoints = self._retokenize_future(
            future_positions=future_positions,
            future_valid=future_valid,
            start_positions=data['agent']['position'][
                agent_mask,
                self.num_historical_steps - 1,
                :2,
            ],
            start_headings=data['agent']['heading'][
                agent_mask,
                self.num_historical_steps - 1,
            ],
            agent_types=data['agent']['type'][agent_mask],
        )
        eligible = future_valid.any(dim=-1)
        if not eligible.any():
            return selected_positions.new_zeros(())
        return (~retokenization_valid & eligible).float().sum() / eligible.float().sum()

    def _log_additional_rollout_metrics(self, data, pred_out, eval_mask, eval_valid):
        prediction = pred_out['pred_traj'][eval_mask]
        target = pred_out['gt'][eval_mask]
        valid = eval_valid[eval_mask]
        horizon_metrics = {}
        for seconds, steps in ((2, 20), (4, 40), (6, 60), (8, 80)):
            ade, fde = self._horizon_displacement_metrics(
                prediction,
                target,
                valid,
                steps,
            )
            horizon_metrics[seconds] = (ade, fde)
            self.log(
                f'val_ADE_{seconds}s',
                ade,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f'val_FDE_{seconds}s',
                fde,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        late_prediction = prediction[:, 40:80]
        late_target = target[:, 40:80]
        late_valid = valid[:, 40:80]
        if late_valid.any():
            late_ade = torch.norm(
                late_prediction - late_target,
                dim=-1,
            )[late_valid].mean()
        else:
            late_ade = prediction.sum() * 0.0
        self.log(
            'val_late_ADE_4s',
            late_ade,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        energies = pred_out['safety_energy']
        for key, value in energies.items():
            self.log(
                f'val_energy_{key}',
                value,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        guidance_metrics = pred_out.get('guidance_metrics', {})
        for key in (
            'min_ttc',
            'ego_min_ttc',
            'ego_risk_min_ttc',
            'ego_min_distance',
            'ego_required_decel',
            'ego_path_intrusion_rate',
            'ego_conflict_tta_error',
            'ego_risk_reward',
            'ego_risk_success_rate',
            'ego_near_miss_success_rate',
            'near_miss_rate',
            'hard_collision_rate',
            'offroad_rate',
            'dynamics_energy',
            'commit_speed_energy',
            'edit_distance',
            'target_success_rate',
            'target_event_success_rate',
        ):
            if key in guidance_metrics:
                self.log(
                    f'val_{key}',
                    guidance_metrics[key],
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=1,
                )
        coverage = pred_out['pred_valid_mask'][eval_mask].float().mean()
        invalid_rate = self._validation_retokenization_invalid_rate(
            data,
            eval_mask,
        )
        self.log(
            'val_prediction_coverage',
            coverage,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            'val_retokenization_invalid_rate',
            invalid_rate,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        if str(getattr(self, 'guidance_mode', 'safe')).lower() in (
            'ego_stress',
            'ego_edit',
        ):
            target_mask = self._guidance_target_agent_mask(
                data,
                eval_mask,
            ).to(device=eval_mask.device)
            non_target = eval_mask & ~target_mask
            if non_target.any():
                non_target_valid = eval_valid[non_target]
                if non_target_valid.any():
                    non_target_distance = torch.norm(
                        pred_out['pred_traj'][non_target]
                        - pred_out['gt'][non_target],
                        dim=-1,
                    )
                    non_target_preservation_ade = (
                        non_target_distance[non_target_valid].mean()
                    )
                else:
                    non_target_preservation_ade = prediction.sum() * 0.0
            else:
                non_target_preservation_ade = prediction.sum() * 0.0
            self.log(
                'val_non_target_preservation_ADE',
                non_target_preservation_ade,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=1,
            )
        rollout_score = self._rollout_score(
            ade_8s=horizon_metrics[8][0],
            late_ade=late_ade,
            energies=energies,
        )
        self.log(
            'val_rollout_score',
            rollout_score,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
