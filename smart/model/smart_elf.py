import torch
import torch.nn.functional as F

from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.modules.elf_decoder import EmbeddedLanguageFlowDecoder


class SMARTEmbeddedLanguageFlow(SMARTAutoregressiveDiffusion):
    """AR rollout wrapper with non-causal embedded-language-flow windows."""

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        diffusion_cfg = model_config.diffusion

        self.elf_objective = str(
            getattr(
                diffusion_cfg,
                'elf_objective',
                getattr(diffusion_cfg, 'causal_objective', 'embedded_language_flow_v1'),
            )
        ).lower()
        if self.elf_objective not in (
            'elf',
            'embedded_language_flow',
            'embedded_language_flow_v1',
        ):
            raise ValueError(
                "diffusion.elf_objective must be embedded_language_flow_v1."
            )
        self.elf_objective = 'embedded_language_flow_v1'

        self.elf_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'elf_loss_weight', 1.0)),
        )
        self.elf_tail_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'elf_tail_loss_weight', 0.25)),
        )
        self.elf_decoder_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'elf_decoder_loss_weight', 0.1)),
        )
        self.elf_integration_steps = max(
            1,
            int(getattr(diffusion_cfg, 'elf_integration_steps', 4)),
        )
        self.elf_noise_scale = max(
            0.0,
            float(getattr(diffusion_cfg, 'elf_noise_scale', 1.0)),
        )
        self.elf_detach_targets = bool(
            getattr(diffusion_cfg, 'elf_detach_targets', True)
        )
        self.elf_sampling_strategy = str(
            getattr(diffusion_cfg, 'elf_sampling_strategy', 'argmax')
        ).lower()
        if self.elf_sampling_strategy not in ('argmax', 'multinomial'):
            raise ValueError(
                "diffusion.elf_sampling_strategy must be argmax or multinomial."
            )
        self.proposal_conditioning_enabled = bool(
            getattr(diffusion_cfg, 'proposal_conditioning_enabled', True)
        )

        token_size = int(getattr(model_config.decoder, 'token_size', 2048))
        self.diffusion_decoder = EmbeddedLanguageFlowDecoder(
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

    def _elf_target_embeddings(self, token_ids, agent_type_ids, valid_mask=None):
        target = self._physical_token_embeddings(token_ids, agent_type_ids)
        if self.elf_detach_targets:
            target = target.detach()
        if valid_mask is not None:
            target = target * valid_mask.unsqueeze(-1).to(dtype=target.dtype)
        return target

    def _elf_source_embeddings(
        self,
        target_embeddings,
        valid_mask,
        proposal_token_ids=None,
        proposal_confidence=None,
        agent_type_ids=None,
    ):
        if self.elf_noise_scale > 0.0:
            source = torch.randn_like(target_embeddings) * self.elf_noise_scale
        else:
            source = torch.zeros_like(target_embeddings)
        if (
            proposal_token_ids is not None
            and proposal_confidence is not None
            and agent_type_ids is not None
        ):
            proposal = self._elf_target_embeddings(
                proposal_token_ids.to(device=target_embeddings.device, dtype=torch.long),
                agent_type_ids,
                valid_mask,
            ).to(dtype=target_embeddings.dtype)
            confidence = proposal_confidence.to(
                device=target_embeddings.device,
                dtype=target_embeddings.dtype,
            ).clamp(0.0, 1.0).unsqueeze(-1)
            source = source * (1.0 - confidence) + proposal * confidence
        return source * valid_mask.unsqueeze(-1).to(dtype=source.dtype)

    def _elf_interpolate(self, source_embeddings, target_embeddings, t):
        t = t.to(device=source_embeddings.device, dtype=source_embeddings.dtype)
        while t.dim() < source_embeddings.dim():
            t = t.unsqueeze(-1)
        velocity = target_embeddings - source_embeddings
        state = source_embeddings + t * velocity
        return state, velocity

    def _sample_elf_times(self, valid_mask, summary):
        batch_size = int(valid_mask.shape[0])
        if self.training:
            times = torch.rand(
                batch_size,
                device=summary.device,
                dtype=summary.dtype,
            )
            return times.clamp(float(getattr(self, 'min_t', 1e-3)), 1.0)
        return torch.full(
            (batch_size,),
            0.5,
            device=summary.device,
            dtype=summary.dtype,
        )

    def _token_embedding_tables(self, device, dtype):
        agent_encoder = self.encoder.agent_encoder
        specs = (
            ('veh', 0, agent_encoder.token_emb_veh),
            ('ped', 1, agent_encoder.token_emb_ped),
            ('cyc', 2, agent_encoder.token_emb_cyc),
        )
        tables = []
        for token_name, type_id, token_embedder in specs:
            token_template = torch.from_numpy(
                agent_encoder.trajectory_token[token_name]
            ).to(device=device, dtype=torch.float)
            table = token_embedder(
                token_template.reshape(token_template.shape[0], -1)
            ).to(dtype=dtype)
            tables.append((type_id, table))
        return tables

    def _elf_proxy_token_ids(self, elf_embeddings, agent_type_ids, valid_mask):
        proxy = torch.zeros(
            valid_mask.shape,
            dtype=torch.long,
            device=valid_mask.device,
        )
        if not valid_mask.any():
            return proxy
        for type_id, table in self._token_embedding_tables(
            elf_embeddings.device,
            elf_embeddings.dtype,
        ):
            type_mask = valid_mask & (agent_type_ids == type_id)
            if not type_mask.any():
                continue
            emb = elf_embeddings[type_mask]
            logits = 2.0 * emb.matmul(table.t()) - table.pow(2).sum(dim=-1).unsqueeze(0)
            proxy[type_mask] = logits.argmax(dim=-1)
        return proxy.masked_fill(~valid_mask, 0)

    def _decode_elf_velocity(
        self,
        elf_embeddings,
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

        proposal_embeddings = None
        proposal_conditioning_confidence = None
        if (
            bool(getattr(self, 'proposal_conditioning_enabled', False))
            and proposal_token_ids is not None
            and proposal_confidence is not None
        ):
            proposal_conditioning_confidence = proposal_confidence.to(
                device=elf_embeddings.device,
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
            physical_token_embeddings=elf_embeddings.to(dtype=summary.dtype),
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
        loss_mask = raw_loss_mask_base & retokenization_valid

        target = self._elf_target_embeddings(
            gt,
            packed['agent_type_ids'],
            valid_mask,
        ).to(device=summary.device, dtype=summary.dtype)
        source = self._elf_source_embeddings(
            target,
            valid_mask,
        ).to(device=summary.device, dtype=summary.dtype)
        t = self._sample_elf_times(valid_mask, summary).clamp(
            float(getattr(self, 'min_t', 1e-3)),
            1.0,
        )
        elf_embeddings, target_velocity = self._elf_interpolate(source, target, t)
        proxy_ids = self._elf_proxy_token_ids(
            elf_embeddings,
            packed['agent_type_ids'],
            valid_mask,
        )
        velocity, decoder_logits = self._decode_elf_velocity(
            elf_embeddings,
            proxy_ids,
            packed,
            summary,
            t,
            geometry_known_mask=valid_mask,
        )

        if loss_mask.any():
            commit_chunks = int(getattr(self, 'ar_commit_tokens', 1))
            commit_mask = loss_mask & (packed['chunk_ids'] < commit_chunks)
            tail_mask = loss_mask & ~commit_mask

            def masked_mean(values, mask):
                if mask.any():
                    return values[mask].mean()
                return values.sum() * 0.0

            token_flow_loss = (velocity - target_velocity).pow(2).mean(dim=-1)
            commit_flow_loss = masked_mean(token_flow_loss, commit_mask)
            tail_flow_loss = masked_mean(token_flow_loss, tail_mask)
            flow_loss = (
                commit_flow_loss
                + self.elf_tail_loss_weight * tail_flow_loss
            ) * self.elf_loss_weight

            token_decoder_loss = decoder_logits.new_zeros(gt.shape)
            token_decoder_loss[loss_mask] = F.cross_entropy(
                decoder_logits[loss_mask].to(torch.float32),
                gt[loss_mask],
                reduction='none',
            ).to(dtype=decoder_logits.dtype)
            commit_decoder_loss = masked_mean(token_decoder_loss, commit_mask)
            tail_decoder_loss = masked_mean(token_decoder_loss, tail_mask)
            decoder_loss = (
                commit_decoder_loss
                + self.elf_tail_loss_weight * tail_decoder_loss
            )
            loss = flow_loss + self.elf_decoder_loss_weight * decoder_loss
            acc_mask = commit_mask if commit_mask.any() else loss_mask
            acc = (
                decoder_logits[acc_mask].argmax(-1) == gt[acc_mask]
            ).float().mean()
        else:
            flow_loss = velocity.sum() * 0.0
            decoder_loss = decoder_logits.sum() * 0.0
            loss = flow_loss + decoder_loss
            acc = velocity.new_zeros(())

        if self.training:
            valid_count = valid_mask.float().sum().clamp_min(1.0)
            supervision_count = raw_loss_mask_base.float().sum().clamp_min(1.0)
            invalid_count = (raw_loss_mask_base & ~retokenization_valid).float().sum()
            self.log(
                'train_elf_loss_frac',
                loss_mask.float().sum() / valid_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_elf_retokenization_invalid_frac',
                invalid_count / supervision_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_elf_flow_loss',
                flow_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                'train_elf_decoder_loss',
                decoder_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
        return loss, acc

    def _integrate_window_elf(
        self,
        elf_embeddings,
        sampled,
        editable_mask,
        valid_mask,
        packed,
        summary,
        initial_proposal_token_ids=None,
        initial_proposal_confidence=None,
    ):
        steps = max(1, int(getattr(self, 'elf_integration_steps', 1)))
        dt = 1.0 / float(steps)
        decoder_logits = None
        for flow_step in range(steps):
            t_value = max(
                flow_step / float(steps),
                float(getattr(self, 'min_t', 1e-3)),
            )
            t_batch = torch.full(
                (elf_embeddings.shape[0],),
                t_value,
                device=summary.device,
                dtype=summary.dtype,
            )
            proxy_ids = sampled.clone()
            proxy = self._elf_proxy_token_ids(
                elf_embeddings,
                packed['agent_type_ids'],
                valid_mask,
            )
            proxy_ids[editable_mask] = proxy[editable_mask]
            velocity, decoder_logits = self._decode_elf_velocity(
                elf_embeddings,
                proxy_ids,
                packed,
                summary,
                t_batch,
                geometry_known_mask=valid_mask,
                proposal_token_ids=initial_proposal_token_ids,
                proposal_confidence=initial_proposal_confidence,
            )
            elf_embeddings[editable_mask] = (
                elf_embeddings[editable_mask] + dt * velocity[editable_mask]
            )

        t_batch = torch.ones(
            (elf_embeddings.shape[0],),
            device=summary.device,
            dtype=summary.dtype,
        )
        proxy_ids = sampled.clone()
        proxy = self._elf_proxy_token_ids(
            elf_embeddings,
            packed['agent_type_ids'],
            valid_mask,
        )
        proxy_ids[editable_mask] = proxy[editable_mask]
        _velocity, decoder_logits = self._decode_elf_velocity(
            elf_embeddings,
            proxy_ids,
            packed,
            summary,
            t_batch,
            geometry_known_mask=valid_mask,
            proposal_token_ids=initial_proposal_token_ids,
            proposal_confidence=initial_proposal_confidence,
        )
        return elf_embeddings, decoder_logits

    def _build_diffusion_inputs(self, data, rollout_valid=False):
        result = super()._build_diffusion_inputs(
            data,
            rollout_valid=rollout_valid,
        )
        packed = result[0]
        if (
            packed is not None
            and bool(getattr(self, 'current_state_enabled', False))
            and getattr(self, 'ar_objective', 'maskgit') != 'causal_frontier_v1'
        ):
            self._apply_current_state_context(data, packed)
        return result

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
            token_agent_ids,
            chunk_ids,
            agent_context,
            agent_type_ids,
            map_context,
            map_positions,
            map_orientations,
            map_batch,
            map_valid_mask,
            agent_shape_embeddings,
            seed_trajs,
        )
        if packed is None:
            raise ValueError("SMARTEmbeddedLanguageFlow sampling requires packed inputs.")
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
            editable_mask = valid_mask.clone()
            sampled = torch.full(
                (batch_size, sequence_length),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            )
            sampled = sampled.masked_fill(~valid_mask, 0)

        seed_for_source = sampled.masked_fill(sampled == self.mask_token_id, 0)
        target_for_source = self._elf_target_embeddings(
            seed_for_source,
            packed['agent_type_ids'],
            valid_mask,
        ).to(device=device, dtype=summary.dtype)
        elf_embeddings = self._elf_source_embeddings(
            target_for_source,
            valid_mask,
            proposal_token_ids=initial_proposal_token_ids,
            proposal_confidence=initial_proposal_confidence,
            agent_type_ids=packed['agent_type_ids'],
        ).to(device=device, dtype=summary.dtype)

        locked = valid_mask & (sampled != self.mask_token_id)
        if locked.any():
            locked_embeddings = self._elf_target_embeddings(
                sampled,
                packed['agent_type_ids'],
                valid_mask,
            ).to(device=device, dtype=summary.dtype)
            elf_embeddings[locked] = locked_embeddings[locked]

        if editable_mask.any():
            elf_embeddings, decoder_logits = self._integrate_window_elf(
                elf_embeddings,
                sampled,
                editable_mask,
                valid_mask,
                packed,
                summary,
                initial_proposal_token_ids=initial_proposal_token_ids,
                initial_proposal_confidence=initial_proposal_confidence,
            )
            sampled_ids = self._elf_proxy_token_ids(
                elf_embeddings,
                packed['agent_type_ids'],
                valid_mask,
            )[editable_mask]
            probabilities = F.softmax(
                decoder_logits / self.remask_confidence_temperature,
                dim=-1,
            )
            editable_probabilities = probabilities[editable_mask].clamp_min(1e-10)
            sampled_confidence = editable_probabilities.gather(
                -1,
                sampled_ids.unsqueeze(-1),
            ).squeeze(-1)
            sampled[editable_mask] = sampled_ids
            confidence[editable_mask] = sampled_confidence

        remaining = valid_mask & editable_mask & (sampled == self.mask_token_id)
        if remaining.any():
            raise RuntimeError("Embedded language flow sampling ended with masked valid tokens.")
        sampled = sampled.masked_fill(~valid_mask, 0)
        confidence = confidence.masked_fill(~valid_mask, 0.0)
        if return_trace:
            trace = [{
                'step': 0,
                'mode': 'full_window',
                'sampled': int((valid_mask & editable_mask).sum().item()),
                'masked_after': int(remaining.sum().item()),
                'remasked': 0,
                'elf_steps': int(getattr(self, 'elf_integration_steps', 1)),
            }]
            return sampled, confidence, trace
        return sampled, confidence

    def _build_elf_training_view(self, data):
        return self._build_ar_frontier_training_view(data)

    def training_step(self, data, batch_idx):
        del batch_idx
        data = self._prepare_batch(data)
        retokenization = None
        state_mode = 'clean'
        rollout_depth = 0
        if self.ar_rolling_anchor_training:
            data, retokenization, state_mode, rollout_depth = self._build_elf_training_view(
                data
            )
        packed, summary, _ft, _fv, _generation_agents, _supervision_agents, _agent_batch = (
            self._build_diffusion_inputs(data)
        )
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
        if retokenization is not None:
            for key, values in retokenization.items():
                packed[key] = self._pack_agent_window_values(values, packed)

        diffusion_loss, mask_acc = self._compute_diffusion_loss(packed, summary)
        ntp_loss = self._compute_optional_ntp_loss(data, diffusion_loss)
        loss = diffusion_loss + self.ntp_aux_loss_weight * ntp_loss

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
        for name in ('clean', 'perturb', 'rollout'):
            self.log(
                f'train_elf_state_mode_{name}',
                loss.new_tensor(1.0 if state_mode == name else 0.0),
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
        self.log(
            'train_elf_rollout_depth',
            loss.new_tensor(float(rollout_depth)),
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        return loss
