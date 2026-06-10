import math

import torch
import torch.nn.functional as F

from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.modules.causal_diffusion_decoder import CausalDiffusionDecoder
from smart.modules.trajectory_energy import TrajectoryEnergy


class SMARTCausalDiffusion(SMARTAutoregressiveDiffusion):
    """Causal absorbing diffusion over short SMART trajectory-token windows."""

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
        if self.diffusion_num_steps < self.num_future_chunks:
            raise ValueError(
                "SMART causal diffusion requires num_steps >= prediction_tokens."
            )

        diffusion_cfg = model_config.diffusion
        self.causal_frontier_loss_weight = float(
            getattr(diffusion_cfg, 'frontier_loss_weight', 1.0)
        )
        self.causal_frontier_loss_weight = max(self.causal_frontier_loss_weight, 0.0)
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
        self.safety_topk = max(
            1,
            int(getattr(diffusion_cfg, 'safety_topk', 16)),
        )
        self.safety_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'safety_energy_weight', 1.0)),
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

        # Causal diffusion has one monotonic state path. Legacy remasking,
        # proposal carry, and independently corrupted visible tokens would
        # violate that state definition.
        self.remask_sampling = False
        self.prefix_constrained_sampling = True
        self.prefix_constrained_training = True
        self.causal_noise_schedule = False
        self.visible_token_corruption_prob = 0.0
        self.visible_token_corruption_probs = ()
        self.self_condition_prob = 0.0
        self.use_proposal_geometry = False
        self.ar_carry_tail_proposal = False

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

    def _sample_absorbing_prefix_mask(
        self,
        valid_mask,
        survival_prob,
        token_agent_ids,
        chunk_ids,
        random_values=None,
    ):
        """Sample a visible prefix and absorbing masked suffix for each agent."""
        if random_values is None:
            random_values = torch.rand_like(valid_mask, dtype=torch.float)
        survival_prob = survival_prob.to(
            device=valid_mask.device,
            dtype=random_values.dtype,
        )
        if survival_prob.dim() == 0:
            survival_prob = survival_prob.expand(valid_mask.shape[0])
        if survival_prob.dim() != 1 or survival_prob.shape[0] != valid_mask.shape[0]:
            raise ValueError("survival_prob must be scalar or have shape [batch].")

        mask = torch.zeros_like(valid_mask)
        for batch_idx in range(valid_mask.shape[0]):
            agent_ids = torch.unique(token_agent_ids[batch_idx][valid_mask[batch_idx]])
            for agent_id in agent_ids.tolist():
                if agent_id < 0:
                    continue
                agent_nodes = torch.nonzero(
                    valid_mask[batch_idx]
                    & (token_agent_ids[batch_idx] == agent_id),
                    as_tuple=False,
                ).squeeze(-1)
                order = torch.argsort(chunk_ids[batch_idx, agent_nodes])
                agent_nodes = agent_nodes[order]
                survives = random_values[batch_idx, agent_nodes] < survival_prob[batch_idx]
                failure = torch.nonzero(~survives, as_tuple=False)
                if failure.numel() > 0:
                    first_masked = int(failure[0, 0].item())
                    mask[batch_idx, agent_nodes[first_masked:]] = True
        return mask & valid_mask

    @staticmethod
    def _causal_reveal_count(step, num_steps, num_chunks):
        if num_steps <= 0 or num_chunks <= 0:
            return 0
        step = min(max(int(step), 0), int(num_steps) - 1)
        first_reveal_step = max(0, int(num_steps) - int(num_chunks))
        return min(int(num_chunks), max(0, step - first_reveal_step + 1))

    @staticmethod
    def _causal_diffusion_weight(sigma_t, dsigma_t, chunk_ids):
        chunk_order = chunk_ids.to(dtype=sigma_t.dtype) + 1.0
        scaled_sigma = sigma_t.unsqueeze(-1) * chunk_order
        numerator = dsigma_t.unsqueeze(-1) * chunk_order
        return numerator / torch.expm1(scaled_sigma).clamp_min(1e-12)

    @staticmethod
    def _closed_loop_curriculum(epoch):
        epoch = max(0, int(epoch))
        if epoch <= 3:
            return 0.0, 0.0
        if epoch <= 7:
            return 0.25, 0.0
        if epoch <= 15:
            rollout_prob = 0.10 + (epoch - 8) * (0.20 / 7.0)
            return 0.25, rollout_prob
        return 0.25, 0.50

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

    def _select_topk_by_energy(self, topk_log_probabilities, energies, t_value):
        guidance_scale = getattr(
            self,
            'safety_energy_weight',
            0.0,
        ) * (1.0 - float(t_value)) ** 2
        adjusted_score = topk_log_probabilities - guidance_scale * energies
        return adjusted_score.argmax(dim=-1)

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
    ):
        frontier_logits = logits[frontier]
        topk = min(self.safety_topk, int(frontier_logits.shape[-1]))
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
        nominal_other = candidate_positions[:, 0]
        frontier_agent_ids = packed['token_agent_ids'].reshape(-1)[flat_frontier]
        collision = self.trajectory_energy.collision_energy(
            candidate_positions,
            nominal_other,
            candidate_batch=candidate_batch,
            other_batch=candidate_batch,
            candidate_agent_ids=frontier_agent_ids,
            other_agent_ids=frontier_agent_ids,
        )
        total_energy = (
            self.lane_distance_energy_weight * lane_distance
            + self.lane_heading_energy_weight * lane_heading
            + self.dynamics_energy_weight * dynamics
            + self.collision_energy_weight * collision
        )
        selected_topk = self._select_topk_by_energy(
            topk_log_probabilities,
            total_energy,
            t_value,
        )
        row = torch.arange(
            num_frontier,
            device=topk_ids.device,
        )
        selected_ids = topk_ids[row, selected_topk]
        selected_confidence = probabilities[frontier].gather(
            -1,
            selected_ids.unsqueeze(-1),
        ).squeeze(-1)
        diagnostics = {
            'lane_distance': lane_distance[row, selected_topk].mean(),
            'lane_heading': lane_heading[row, selected_topk].mean(),
            'dynamics': dynamics[row, selected_topk].mean(),
            'collision': collision[row, selected_topk].mean(),
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
        batch_size = summary.shape[0]
        t = self._sample_diffusion_timesteps(batch_size, summary.device)
        sigma_t, mask_prob, dsigma_t = self.noise_schedule(t)

        gt = packed['token_ids']
        valid_mask = packed['valid_mask']
        raw_loss_mask_base = packed.get('loss_mask_base', valid_mask) & valid_mask
        retokenization_valid = packed.get(
            'retokenization_valid',
            torch.ones_like(valid_mask),
        ).bool()
        loss_mask_base = raw_loss_mask_base & retokenization_valid
        mask = self._sample_absorbing_prefix_mask(
            valid_mask=valid_mask,
            survival_prob=1.0 - mask_prob,
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
        )
        for batch_idx in range(mask.shape[0]):
            if raw_loss_mask_base[batch_idx].any() and not (
                mask[batch_idx] & raw_loss_mask_base[batch_idx]
            ).any():
                eligible = torch.nonzero(
                    raw_loss_mask_base[batch_idx],
                    as_tuple=False,
                ).squeeze(-1)
                farthest = eligible[
                    torch.argmax(packed['chunk_ids'][batch_idx, eligible])
                ]
                agent_id = packed['token_agent_ids'][batch_idx, farthest]
                chunk_id = packed['chunk_ids'][batch_idx, farthest]
                mask[batch_idx] |= (
                    valid_mask[batch_idx]
                    & (packed['token_agent_ids'][batch_idx] == agent_id)
                    & (packed['chunk_ids'][batch_idx] >= chunk_id)
                )

        frontier_mask = self._prefix_frontier_mask(
            mask,
            valid_mask,
            packed['token_agent_ids'],
            packed['chunk_ids'],
        ) & loss_mask_base
        suffix_loss_mask = mask & loss_mask_base
        noisy = gt.clone()
        noisy[mask] = self.mask_token_id
        geometry_known_mask = (~mask) & valid_mask
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
            masked_supervision=mask,
        )
        if suffix_loss_mask.any():
            log_p = F.log_softmax(logits, dim=-1)
            nll = -log_p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
            diffusion_weight = self._causal_diffusion_weight(
                sigma_t,
                dsigma_t,
                packed['chunk_ids'],
            )
            supervision_weight = suffix_loss_mask.to(dtype=nll.dtype)
            supervision_weight = supervision_weight + (
                self.causal_frontier_loss_weight
                * frontier_mask.to(dtype=nll.dtype)
            )
            loss = (diffusion_weight * nll * supervision_weight).sum()
            loss = loss / loss_mask_base.to(dtype=nll.dtype).sum().clamp_min(1.0)
            acc = (
                logits[suffix_loss_mask].argmax(-1) == gt[suffix_loss_mask]
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
    ):
        del initial_proposal_token_ids, initial_proposal_confidence
        batch_size, sequence_length = valid_mask.shape
        device = summary.device
        sampled = torch.full(
            (batch_size, sequence_length),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        confidence = summary.new_zeros((batch_size, sequence_length))
        trace = []
        previous_reveal_count = 0
        energy_totals = {
            'lane_distance': summary.new_zeros(()),
            'lane_heading': summary.new_zeros(()),
            'dynamics': summary.new_zeros(()),
            'collision': summary.new_zeros(()),
        }
        energy_events = 0

        for step in range(self.diffusion_num_steps):
            reveal_count = self._causal_reveal_count(
                step,
                self.diffusion_num_steps,
                self.num_future_chunks,
            )
            newly_revealed = 0
            while previous_reveal_count < reveal_count:
                mask = valid_mask & (sampled == self.mask_token_id)
                frontier = self._prefix_frontier_mask(
                    mask,
                    valid_mask,
                    token_agent_ids,
                    chunk_ids,
                )
                if not frontier.any():
                    break
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
                    getattr(self, 'safety_energy_enabled', False)
                    and packed is not None
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
                    )
                    for key in energy_totals:
                        energy_totals[key] += energy_diagnostics[key]
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
                newly_revealed += int(frontier.sum().item())
                previous_reveal_count += 1

            if return_trace:
                remaining = valid_mask & (sampled == self.mask_token_id)
                trace.append({
                    'step': step,
                    'reveal_count': reveal_count,
                    'newly_revealed': newly_revealed,
                    'masked_after': int(remaining.sum().item()),
                    'remasked': 0,
                })

        remaining = valid_mask & (sampled == self.mask_token_id)
        if remaining.any():
            raise RuntimeError("Causal diffusion sampling ended with masked valid tokens.")
        denominator = max(energy_events, 1)
        self._last_sampling_energy = {
            key: (value / denominator).detach()
            for key, value in energy_totals.items()
        }
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
            output['safety_energy'] = {
                key: torch.stack([
                    entry[key]
                    for entry in self._sampling_energy_accumulator
                ]).mean()
                for key in self._sampling_energy_accumulator[0]
            }
        else:
            zero = output['pred_traj'].sum() * 0.0
            output['safety_energy'] = {
                'lane_distance': zero,
                'lane_heading': zero,
                'dynamics': zero,
                'collision': zero,
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
