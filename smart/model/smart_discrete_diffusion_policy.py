import torch
import torch.nn.functional as F

from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.modules.trajectory_energy import TrajectoryEnergy


class SMARTDiscreteDiffusionPolicy(SMARTAutoregressiveDiffusion):
    """Discrete diffusion-policy ablation with window rerank and one-token commit."""

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        diffusion_cfg = getattr(model_config, 'diffusion', None)
        objective = str(
            getattr(diffusion_cfg, 'discrete_policy_objective', 'chunk_rerank_v1')
        ).lower()
        if objective != 'chunk_rerank_v1':
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy supports only "
                "diffusion.discrete_policy_objective: chunk_rerank_v1."
            )
        if self.ar_commit_tokens != 1:
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy requires diffusion.commit_tokens: 1."
            )
        if bool(getattr(diffusion_cfg, 'carry_tail_proposal', False)):
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy discards tail chunks at inference; "
                "set diffusion.carry_tail_proposal: false."
            )
        if self.ntp_aux_loss_weight <= 0.0:
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy requires ntp_aux_loss_weight > 0 "
                "for the dense SMART next-token CE prior."
            )

        self.discrete_policy_span_tokens = int(
            getattr(diffusion_cfg, 'discrete_policy_span_tokens', 4)
        )
        chunk_weights = tuple(
            float(value)
            for value in getattr(
                diffusion_cfg,
                'discrete_policy_chunk_loss_weights',
                getattr(diffusion_cfg, 'causal_loss_weights', (1.0, 0.3, 0.15, 0.075)),
            )
        )
        if len(chunk_weights) != self.ar_prediction_tokens:
            raise ValueError(
                "diffusion.discrete_policy_chunk_loss_weights must have one "
                "entry per prediction token."
            )
        self.discrete_policy_chunk_loss_weights = chunk_weights
        self.causal_loss_weighting_enabled = True
        self.causal_loss_weights = chunk_weights
        self.discrete_policy_overlap_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'discrete_policy_overlap_loss_weight', 0.05)),
        )
        self.discrete_policy_candidate_count = max(
            1,
            int(getattr(diffusion_cfg, 'discrete_policy_candidate_count', 1)),
        )
        self.discrete_policy_candidate_score_enabled = bool(
            getattr(diffusion_cfg, 'discrete_policy_candidate_score_enabled', True)
        )
        self.discrete_policy_candidate_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, 'discrete_policy_candidate_energy_weight', 1.0)),
        )
        candidate_chunk_weights = tuple(
            float(value)
            for value in getattr(
                diffusion_cfg,
                'discrete_policy_candidate_chunk_weights',
                chunk_weights,
            )
        )
        if len(candidate_chunk_weights) != self.ar_prediction_tokens:
            raise ValueError(
                "diffusion.discrete_policy_candidate_chunk_weights must have one "
                "entry per prediction token."
            )
        self.discrete_policy_candidate_chunk_weights = candidate_chunk_weights
        if (
            self.discrete_policy_candidate_score_enabled
            and not hasattr(self, 'trajectory_energy')
        ):
            self.trajectory_energy = TrajectoryEnergy(dt=0.1)

    def _discrete_policy_training_anchors(self, data):
        anchors = self._commitment_training_anchors(data)
        if not anchors:
            return []
        span = int(getattr(self, 'discrete_policy_span_tokens', 0))
        if span <= 0 or span >= len(anchors):
            return anchors
        start_limit = max(1, len(anchors) - span + 1)
        start = self._training_step_index() % start_limit
        return anchors[start:start + span]

    def _compute_discrete_policy_smart_ntp_loss(self, data, ref_tensor):
        if self.ntp_aux_loss_weight <= 0.0:
            return ref_tensor.new_zeros(())
        return self._compute_ntp_loss(self(data))

    def _discrete_policy_overlap_loss(
        self,
        source_logits,
        target_logits,
        source_mask,
        target_mask,
        source_chunk,
    ):
        if source_logits.numel() == 0 or target_logits.numel() == 0:
            return source_logits.sum() * 0.0 + target_logits.sum() * 0.0
        source_chunk = int(source_chunk)
        if source_chunk <= 0 or source_chunk >= source_logits.shape[1]:
            return source_logits.sum() * 0.0 + target_logits.sum() * 0.0
        valid = (
            source_mask[:, source_chunk].bool()
            & target_mask[:, 0].bool()
        )
        if not bool(valid.any()):
            return source_logits.sum() * 0.0 + target_logits.sum() * 0.0

        teacher_log_prob = F.log_softmax(source_logits[:, source_chunk], dim=-1)
        student_log_prob = F.log_softmax(target_logits[:, 0], dim=-1)
        teacher_prob = teacher_log_prob.exp().detach()
        kl = F.kl_div(
            student_log_prob[valid],
            teacher_prob[valid],
            reduction='none',
        ).sum(dim=-1)
        return kl.mean()

    def _compute_discrete_policy_training_loss(self, data, ref_tensor):
        anchors = self._discrete_policy_training_anchors(data)
        if not anchors:
            zero = ref_tensor.new_zeros(())
            return {
                'chunk_loss': zero,
                'mask_acc': zero,
                'overlap_loss': zero,
                'window_count': 0,
            }

        num_agents = int(data['agent']['token_idx'].shape[0])
        losses = []
        accuracies = []
        window_records = {}

        for anchor in anchors:
            view, _target_tokens, _target_valid, _anchor = self._build_ar_training_view(
                data,
                anchor_token=anchor,
                perturb=None,
                allow_incomplete_window=True,
            )
            (
                packed,
                summary,
                _ft,
                _fv,
                _generation_agents,
                _supervision_agents,
                _agent_batch,
            ) = self._build_diffusion_inputs(view)
            if packed is None:
                continue

            loss, acc, details = self._compute_diffusion_loss(
                packed,
                summary,
                forced_mask=packed['valid_mask'],
                return_details=True,
                loss_normalization='supervision_weight',
            )
            losses.append(loss)
            accuracies.append(acc)
            window_records[int(anchor)] = {
                'logits': self._unpack_packed_window_values(
                    details['logits'],
                    packed,
                    num_agents,
                    fill_value=0.0,
                ),
                'mask': self._unpack_packed_window_values(
                    details.get('loss_mask', packed['loss_mask_base']),
                    packed,
                    num_agents,
                    fill_value=False,
                ),
            }

        if not losses:
            zero = ref_tensor.new_zeros(())
            return {
                'chunk_loss': zero,
                'mask_acc': zero,
                'overlap_loss': zero,
                'window_count': 0,
            }

        overlap_losses = []
        for anchor in anchors:
            source = window_records.get(int(anchor))
            if source is None:
                continue
            for source_chunk in range(1, self.ar_prediction_tokens):
                target = window_records.get(int(anchor) + source_chunk)
                if target is None:
                    continue
                overlap_losses.append(
                    self._discrete_policy_overlap_loss(
                        source['logits'],
                        target['logits'],
                        source['mask'],
                        target['mask'],
                        source_chunk,
                    )
                )

        chunk_loss = torch.stack(losses).mean()
        mask_acc = torch.stack(accuracies).mean()
        if overlap_losses:
            overlap_loss = torch.stack(overlap_losses).mean()
        else:
            overlap_loss = chunk_loss.new_zeros(())
        return {
            'chunk_loss': chunk_loss,
            'mask_acc': mask_acc,
            'overlap_loss': overlap_loss,
            'window_count': len(losses),
        }

    def training_step(self, data, batch_idx):
        del batch_idx
        data = self._prepare_batch(data)
        ref_tensor = data['agent']['token_idx'].new_zeros((), dtype=torch.float)
        policy = self._compute_discrete_policy_training_loss(data, ref_tensor)
        if int(policy['window_count']) <= 0:
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

        smart_ntp_loss = self._compute_discrete_policy_smart_ntp_loss(
            data,
            policy['chunk_loss'],
        )
        loss = (
            self.ntp_aux_loss_weight * smart_ntp_loss
            + self.diffusion_loss_weight * policy['chunk_loss']
            + self.discrete_policy_overlap_loss_weight * policy['overlap_loss']
        )

        self.log('train_empty_diffusion_batch', loss.new_zeros(()),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=1)
        self.log('smart_ntp_loss', smart_ntp_loss, prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=1)
        self.log('diffusion_loss', policy['chunk_loss'], prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=1)
        self.log('discrete_policy_chunk_loss', policy['chunk_loss'],
                 prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('discrete_policy_overlap_loss', policy['overlap_loss'],
                 prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_mask_acc', policy['mask_acc'],
                 on_step=True, on_epoch=True, batch_size=1)
        self.log(
            'train_discrete_policy_window_count',
            loss.new_tensor(float(policy['window_count'])),
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def _score_discrete_policy_candidate_windows(
        self,
        base_logp,
        energy_by_chunk,
        chunk_valid,
    ):
        weights = torch.tensor(
            self.discrete_policy_candidate_chunk_weights,
            device=energy_by_chunk.device,
            dtype=energy_by_chunk.dtype,
        )
        while weights.numel() < energy_by_chunk.shape[-1]:
            weights = torch.cat([weights, weights[-1:]], dim=0)
        weights = weights[:energy_by_chunk.shape[-1]]
        valid = chunk_valid.to(device=energy_by_chunk.device, dtype=energy_by_chunk.dtype)
        penalty = (
            energy_by_chunk
            * weights.unsqueeze(0)
            * valid
        ).sum(dim=-1)
        return base_logp - self.discrete_policy_candidate_energy_weight * penalty

    def _candidate_base_logp_by_window(self, sampled_ids, sampled_confidence, valid_mask):
        if sampled_confidence is None:
            return sampled_ids.new_zeros((), dtype=torch.float)
        valid = valid_mask.to(device=sampled_confidence.device, dtype=torch.bool)
        if not bool(valid.any()):
            return sampled_confidence.new_zeros(())
        return torch.log(sampled_confidence[valid].clamp_min(1.0e-8)).sum()

    def _candidate_energy_by_chunk(self, sampled_ids, sample_kwargs):
        packed = sample_kwargs.get('packed')
        if packed is None:
            valid_mask = sample_kwargs['valid_mask']
            return (
                sampled_ids.new_zeros(self.ar_prediction_tokens, dtype=torch.float),
                torch.zeros(
                    self.ar_prediction_tokens,
                    dtype=torch.bool,
                    device=sampled_ids.device,
                ),
            )

        valid_mask = sample_kwargs['valid_mask'].to(
            device=sampled_ids.device,
            dtype=torch.bool,
        )
        chunk_ids = sample_kwargs['chunk_ids'].to(
            device=sampled_ids.device,
            dtype=torch.long,
        )
        token_valid = (
            valid_mask
            & (sampled_ids >= 0)
            & (sampled_ids < self.token_size)
        )
        if not bool(token_valid.any()):
            return (
                sampled_ids.new_zeros(self.ar_prediction_tokens, dtype=torch.float),
                torch.zeros(
                    self.ar_prediction_tokens,
                    dtype=torch.bool,
                    device=sampled_ids.device,
                ),
            )

        geometry_known = valid_mask & (sampled_ids != self.mask_token_id)
        anchor_positions, anchor_headings, _geometry_confidence = (
            self._refresh_token_geometry(
                sampled_ids,
                packed,
                geometry_known_mask=geometry_known,
                proposal_token_ids=sample_kwargs.get('initial_proposal_token_ids'),
                proposal_confidence=sample_kwargs.get('initial_proposal_confidence'),
            )
        )
        flat_valid = token_valid.reshape(-1)
        flat_ids = sampled_ids.reshape(-1)[flat_valid]
        flat_types = sample_kwargs['agent_type_ids'].reshape(-1)[flat_valid]
        flat_positions = anchor_positions.reshape(-1, 2)[flat_valid]
        flat_headings = anchor_headings.reshape(-1)[flat_valid]
        flat_chunks = chunk_ids.reshape(-1)[flat_valid]
        flat_agent_ids = sample_kwargs['token_agent_ids'].reshape(-1)[flat_valid]
        sequence_length = valid_mask.shape[1]
        flat_indices = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        flat_batch = torch.div(flat_indices, sequence_length, rounding_mode='floor')

        candidate_positions, candidate_headings = self._token_chunk_world(
            flat_ids,
            flat_types,
            flat_positions,
            flat_headings,
        )
        candidate_positions = candidate_positions.unsqueeze(1)
        candidate_headings = candidate_headings.unsqueeze(1)

        trajectory_energy = getattr(self, 'trajectory_energy', None)
        if trajectory_energy is None:
            trajectory_energy = TrajectoryEnergy(dt=0.1).to(candidate_positions.device)
        lane_distance, lane_heading = trajectory_energy.lane_energy(
            candidate_positions,
            candidate_headings,
            sample_kwargs.get('map_positions'),
            sample_kwargs.get('map_orientations'),
            candidate_batch=flat_batch,
            map_batch=sample_kwargs.get('map_batch'),
            map_valid_mask=sample_kwargs.get('map_valid_mask'),
        )
        dynamics = trajectory_energy.dynamics_energy(
            candidate_positions,
            candidate_headings,
            current_positions=flat_positions,
        )
        collision_batch = flat_batch * int(self.ar_prediction_tokens) + flat_chunks
        collision = trajectory_energy.collision_energy(
            candidate_positions,
            candidate_positions[:, 0],
            candidate_batch=collision_batch,
            other_batch=collision_batch,
            candidate_agent_ids=flat_agent_ids,
            other_agent_ids=flat_agent_ids,
        )
        token_energy = (
            self.lane_distance_energy_weight * lane_distance[:, 0]
            + self.lane_heading_energy_weight * lane_heading[:, 0]
            + self.dynamics_energy_weight * dynamics[:, 0]
            + self.collision_energy_weight * collision[:, 0]
        )

        energy_by_chunk = token_energy.new_zeros(self.ar_prediction_tokens)
        chunk_valid = torch.zeros(
            self.ar_prediction_tokens,
            dtype=torch.bool,
            device=token_energy.device,
        )
        for chunk_idx in range(self.ar_prediction_tokens):
            mask = flat_chunks == chunk_idx
            if not bool(mask.any()):
                continue
            energy_by_chunk[chunk_idx] = token_energy[mask].mean()
            chunk_valid[chunk_idx] = True
        return energy_by_chunk, chunk_valid

    def _diffusion_sample(self, *args, **kwargs):
        if (
            not bool(getattr(self, 'discrete_policy_candidate_score_enabled', True))
            or int(getattr(self, 'discrete_policy_candidate_count', 1)) <= 1
            or bool(kwargs.get('return_trace', False))
        ):
            return super()._diffusion_sample(*args, **kwargs)

        candidate_ids = []
        candidate_confidence = []
        base_logp = []
        energy_by_chunk = []
        chunk_valid = []
        for _candidate_idx in range(int(self.discrete_policy_candidate_count)):
            sampled_ids, sampled_confidence = super()._diffusion_sample(
                *args,
                **kwargs,
            )
            candidate_ids.append(sampled_ids)
            candidate_confidence.append(sampled_confidence)
            valid_mask = kwargs['valid_mask']
            base_logp.append(
                self._candidate_base_logp_by_window(
                    sampled_ids,
                    sampled_confidence,
                    valid_mask,
                )
            )
            energy, valid = self._candidate_energy_by_chunk(sampled_ids, kwargs)
            energy_by_chunk.append(energy)
            chunk_valid.append(valid)

        scores = self._score_discrete_policy_candidate_windows(
            torch.stack(base_logp, dim=0),
            torch.stack(energy_by_chunk, dim=0),
            torch.stack(chunk_valid, dim=0),
        )
        best = int(torch.argmax(scores).item())
        return candidate_ids[best], candidate_confidence[best]
