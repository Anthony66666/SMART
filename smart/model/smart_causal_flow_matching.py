import torch
import torch.nn.functional as F

from smart.model.smart_causal_diffusion import SMARTCausalDiffusion


class SMARTCausalFlowMatching(SMARTCausalDiffusion):
    """Causal SMART-token flow matching over short receding-horizon windows."""

    def __init__(self, model_config) -> None:
        diffusion_cfg = model_config.diffusion
        requested_objective = getattr(
            diffusion_cfg,
            'causal_objective',
            'flow_matching_v1',
        )
        if str(requested_objective).lower() in ('flow_matching', 'flow_matching_v1'):
            setattr(diffusion_cfg, 'causal_objective', 'discrete_frontier_v2')
        try:
            super().__init__(model_config)
        finally:
            setattr(diffusion_cfg, 'causal_objective', requested_objective)

        self.causal_objective = 'flow_matching_v1'
        self.flow_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'flow_loss_weight', 1.0)),
        )
        self.flow_integration_steps = max(
            1,
            int(getattr(diffusion_cfg, 'flow_integration_steps', 4)),
        )
        self.flow_sampling_strategy = str(
            getattr(diffusion_cfg, 'flow_sampling_strategy', 'argmax')
        ).lower()
        if self.flow_sampling_strategy not in ('argmax', 'multinomial'):
            raise ValueError(
                "diffusion.flow_sampling_strategy must be argmax or multinomial."
            )

    def _one_hot_tokens(self, token_ids, valid_mask=None):
        one_hot = F.one_hot(
            token_ids.clamp(min=0, max=self.token_size - 1),
            num_classes=self.token_size,
        ).to(dtype=torch.float)
        if valid_mask is not None:
            one_hot = one_hot * valid_mask.unsqueeze(-1).to(dtype=one_hot.dtype)
        return one_hot

    def _flow_source_distribution(
        self,
        token_ids,
        valid_mask,
        proposal_token_ids=None,
        proposal_confidence=None,
    ):
        del token_ids
        source = torch.full(
            (*valid_mask.shape, self.token_size),
            1.0 / float(self.token_size),
            device=valid_mask.device,
            dtype=torch.float,
        )
        if proposal_token_ids is not None and proposal_confidence is not None:
            proposal = self._one_hot_tokens(
                proposal_token_ids.to(device=valid_mask.device, dtype=torch.long),
                valid_mask,
            ).to(dtype=source.dtype)
            confidence = proposal_confidence.to(
                device=valid_mask.device,
                dtype=source.dtype,
            ).clamp(0.0, 1.0).unsqueeze(-1)
            source = source * (1.0 - confidence) + proposal * confidence
        return source * valid_mask.unsqueeze(-1).to(dtype=source.dtype)

    def _flow_interpolate(self, source_probs, target_ids, t):
        target = self._one_hot_tokens(target_ids).to(
            device=source_probs.device,
            dtype=source_probs.dtype,
        )
        t = t.to(device=source_probs.device, dtype=source_probs.dtype)
        while t.dim() < source_probs.dim():
            t = t.unsqueeze(-1)
        velocity = target - source_probs
        state = source_probs + t * velocity
        return state, velocity

    def _sample_flow_times(self, frontier_ids, summary):
        if self.training:
            return torch.rand(
                frontier_ids.shape,
                device=summary.device,
                dtype=summary.dtype,
            )
        return torch.full(
            frontier_ids.shape,
            0.5,
            device=summary.device,
            dtype=summary.dtype,
        )

    def _flow_proxy_token_ids(self, flow_probs, valid_mask):
        proxy = flow_probs.argmax(dim=-1).to(dtype=torch.long)
        return proxy.masked_fill(~valid_mask, 0)

    def _expected_physical_token_embeddings(self, flow_probs, agent_type_ids):
        agent_encoder = self.encoder.agent_encoder
        device = flow_probs.device
        dtype = flow_probs.dtype
        embeddings = agent_encoder.type_a_emb.weight.new_zeros(
            (*flow_probs.shape[:2], self.hidden_dim),
            device=device,
            dtype=dtype,
        )
        token_specs = (
            ('veh', 0, agent_encoder.token_emb_veh),
            ('ped', 1, agent_encoder.token_emb_ped),
            ('cyc', 2, agent_encoder.token_emb_cyc),
        )
        for token_name, type_id, token_embedder in token_specs:
            type_mask = agent_type_ids == type_id
            if not type_mask.any():
                continue
            token_template = torch.from_numpy(
                agent_encoder.trajectory_token[token_name]
            ).to(device=device, dtype=torch.float)
            token_table = token_embedder(
                token_template.reshape(token_template.shape[0], -1)
            ).to(dtype=dtype)
            embeddings[type_mask] = flow_probs[type_mask].matmul(token_table)
        return embeddings

    def _decode_flow_velocity(
        self,
        flow_probs,
        proxy_token_ids,
        packed,
        summary,
        t,
        geometry_known_mask,
        proposal_token_ids=None,
        proposal_confidence=None,
    ):
        token_positions, token_headings, geometry_confidence = self._refresh_token_geometry(
            proxy_token_ids,
            packed,
            geometry_known_mask=geometry_known_mask,
            proposal_token_ids=proposal_token_ids if self.use_proposal_geometry else None,
            proposal_confidence=proposal_confidence if self.use_proposal_geometry else None,
        )
        source_mask = packed['valid_mask'] & (
            geometry_confidence >= self.geometry_confidence_source_threshold
        )
        current_state_source = (
            packed['valid_mask']
            & (packed['chunk_ids'] == 0)
            & bool(getattr(self, 'causal_current_state_edges', False))
        )
        source_mask = source_mask | current_state_source

        proposal_embeddings = None
        proposal_conditioning_confidence = None
        if (
            bool(getattr(self, 'proposal_conditioning_enabled', False))
            and proposal_token_ids is not None
            and proposal_confidence is not None
        ):
            proposal_conditioning_confidence = proposal_confidence.to(
                device=flow_probs.device,
                dtype=summary.dtype,
            ).clamp(0.0, 1.0)
            proposal_conditioning_confidence = (
                proposal_conditioning_confidence
                * packed['valid_mask'].to(dtype=summary.dtype)
            )
            proposal_embeddings = self._physical_token_embeddings(
                proposal_token_ids,
                packed['agent_type_ids'],
            )

        return self.diffusion_decoder(
            noisy_token_ids=proxy_token_ids,
            token_positions=token_positions,
            token_headings=token_headings,
            token_agent_ids=packed['token_agent_ids'],
            noisy_token_chunk_ids=packed['chunk_ids'],
            scene_summary=summary,
            t=t,
            valid_mask=packed['valid_mask'],
            agent_context=packed['agent_context'],
            agent_type_ids=packed['agent_type_ids'],
            agent_shape_embeddings=packed['agent_shape_embeddings'],
            physical_token_embeddings=self._expected_physical_token_embeddings(
                flow_probs,
                packed['agent_type_ids'],
            ),
            map_context=packed.get('map_context'),
            map_positions=packed.get('map_positions'),
            map_orientations=packed.get('map_orientations'),
            map_batch=packed.get('map_batch'),
            map_valid_mask=packed.get('map_valid_mask'),
            geometry_confidence=geometry_confidence,
            temporal_source_mask=source_mask,
            spatial_source_mask=source_mask,
            proposal_token_embeddings=proposal_embeddings,
            proposal_confidence=proposal_conditioning_confidence,
        )

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

        source = self._flow_source_distribution(gt, valid_mask).to(
            device=summary.device,
            dtype=summary.dtype,
        )
        t = self._sample_flow_times(frontier_ids, summary).clamp(0.0, 1.0)
        flow_state, target_velocity = self._flow_interpolate(source, gt, t)
        target = self._one_hot_tokens(gt, valid_mask).to(
            device=summary.device,
            dtype=summary.dtype,
        )
        flow_probs = target.clone()
        flow_probs[mask] = source[mask]
        flow_probs[frontier_raw_mask] = flow_state[frontier_raw_mask]

        proxy_ids = gt.clone()
        proxy_ids[mask] = self._flow_proxy_token_ids(flow_probs, valid_mask)[mask]
        velocity = self._decode_flow_velocity(
            flow_probs,
            proxy_ids,
            packed,
            summary,
            t,
            geometry_known_mask=(~mask) & valid_mask,
        )

        if frontier_mask.any():
            loss = F.mse_loss(
                velocity[frontier_mask],
                target_velocity[frontier_mask],
            ) * self.flow_loss_weight
            predicted_target = (flow_probs + velocity).argmax(dim=-1)
            acc = (
                predicted_target[frontier_mask] == gt[frontier_mask]
            ).float().mean()
        else:
            loss = velocity.sum() * 0.0
            acc = velocity.new_zeros(())

        if self.training:
            valid_count = valid_mask.float().sum().clamp_min(1.0)
            self.log(
                'train_flow_mask_frac',
                mask.float().sum() / valid_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_flow_frontier_frac',
                frontier_mask.float().sum() / valid_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_flow_loss',
                loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
        return loss, acc

    def _integrate_frontier_flow(
        self,
        flow_probs,
        sampled,
        frontier,
        mask,
        valid_mask,
        packed,
        summary,
        initial_proposal_token_ids=None,
        initial_proposal_confidence=None,
    ):
        steps = max(1, int(getattr(self, 'flow_integration_steps', 1)))
        dt = 1.0 / float(steps)
        for flow_step in range(steps):
            t_value = max(flow_step / float(steps), float(getattr(self, 'min_t', 1e-3)))
            t_batch = torch.full(
                (flow_probs.shape[0],),
                t_value,
                device=summary.device,
                dtype=summary.dtype,
            )
            proxy_ids = sampled.clone()
            proxy_ids[mask] = self._flow_proxy_token_ids(flow_probs, valid_mask)[mask]
            velocity = self._decode_flow_velocity(
                flow_probs,
                proxy_ids,
                packed,
                summary,
                t_batch,
                geometry_known_mask=(~mask) & valid_mask,
                proposal_token_ids=initial_proposal_token_ids,
                proposal_confidence=initial_proposal_confidence,
            )
            updated = flow_probs[frontier] + dt * velocity[frontier]
            updated = updated.clamp_min(1e-8)
            updated = updated / updated.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            flow_probs[frontier] = updated
        return flow_probs

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
        del (
            token_positions,
            token_headings,
            agent_context,
            agent_type_ids,
            map_context,
            map_positions,
            map_orientations,
            map_batch,
            map_valid_mask,
            agent_shape_embeddings,
        )
        if packed is None:
            raise ValueError("SMARTCausalFlowMatching sampling requires packed inputs.")
        batch_size, sequence_length = valid_mask.shape
        device = summary.device
        confidence = summary.new_zeros((batch_size, sequence_length))
        if seed_token_ids is not None:
            seed_token_ids = seed_token_ids.to(device=device, dtype=torch.long)
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

        flow_probs = self._flow_source_distribution(
            sampled.masked_fill(sampled == self.mask_token_id, 0),
            valid_mask,
            proposal_token_ids=initial_proposal_token_ids,
            proposal_confidence=initial_proposal_confidence,
        ).to(device=device, dtype=summary.dtype)
        locked = valid_mask & (sampled != self.mask_token_id)
        if locked.any():
            locked_probs = self._one_hot_tokens(sampled, valid_mask).to(
                device=device,
                dtype=summary.dtype,
            )
            flow_probs[locked] = locked_probs[locked]

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
                flow_probs = self._integrate_frontier_flow(
                    flow_probs,
                    sampled,
                    frontier,
                    mask,
                    valid_mask,
                    packed,
                    summary,
                    initial_proposal_token_ids=initial_proposal_token_ids,
                    initial_proposal_confidence=initial_proposal_confidence,
                )
                probabilities = flow_probs.clamp_min(1e-10)
                probabilities = probabilities / probabilities.sum(
                    dim=-1,
                    keepdim=True,
                ).clamp_min(1e-10)
                logits = probabilities.clamp_min(1e-10).log()
                if (
                    str(getattr(self, 'guidance_mode', 'none')).lower() != 'none'
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
                        1.0,
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
                    frontier_probabilities = probabilities[frontier]
                    if self.flow_sampling_strategy == 'multinomial':
                        frontier_ids = torch.multinomial(
                            frontier_probabilities,
                            1,
                        ).squeeze(-1)
                    else:
                        frontier_ids = frontier_probabilities.argmax(dim=-1)
                    frontier_confidence = frontier_probabilities.gather(
                        -1,
                        frontier_ids.unsqueeze(-1),
                    ).squeeze(-1)
                sampled[frontier] = frontier_ids
                confidence[frontier] = frontier_confidence
                flow_probs[frontier] = self._one_hot_tokens(
                    sampled,
                    valid_mask,
                ).to(device=device, dtype=summary.dtype)[frontier]
                newly_revealed = int(frontier.sum().item())

            if return_trace:
                remaining = valid_mask & (sampled == self.mask_token_id)
                trace.append({
                    'step': step,
                    'reveal_count': reveal_count,
                    'newly_revealed': newly_revealed,
                    'masked_after': int(remaining.sum().item()),
                    'remasked': 0,
                    'flow_steps': int(getattr(self, 'flow_integration_steps', 1)),
                })

        remaining = valid_mask & editable_mask & (sampled == self.mask_token_id)
        if remaining.any():
            raise RuntimeError("Causal flow matching sampling ended with masked valid tokens.")
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
