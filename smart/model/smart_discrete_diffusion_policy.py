import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.modules.trajectory_energy import TrajectoryEnergy


class SMARTDiscreteDiffusionPolicy(SMARTAutoregressiveDiffusion):
    """Discrete diffusion-policy ablation with window rerank and one-token commit."""

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        diffusion_cfg = getattr(model_config, 'diffusion', None)
        objective = str(
            getattr(diffusion_cfg, 'discrete_policy_objective', 'pure_chunk_v1')
        ).lower()
        if objective not in ('pure_chunk_v1', 'chunk_rerank_v1'):
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy supports only "
                "diffusion.discrete_policy_objective: pure_chunk_v1."
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
        prediction_horizon = int(
            getattr(diffusion_cfg, 'prediction_horizon', self.ar_prediction_tokens)
        )
        execution_horizon = int(
            getattr(diffusion_cfg, 'execution_horizon', self.ar_commit_tokens)
        )
        if prediction_horizon != self.ar_prediction_tokens:
            raise ValueError(
                "diffusion.prediction_horizon must match diffusion.prediction_tokens "
                "for SMARTDiscreteDiffusionPolicy."
            )
        if execution_horizon != self.ar_commit_tokens:
            raise ValueError(
                "diffusion.execution_horizon must match diffusion.commit_tokens "
                "for SMARTDiscreteDiffusionPolicy."
            )
        if bool(getattr(diffusion_cfg, 'use_smart_ntp_head', False)):
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy is a pure diffusion-policy objective; "
                "set diffusion.use_smart_ntp_head: false."
            )
        if bool(getattr(diffusion_cfg, 'use_smart_prior_fusion', False)):
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy does not fuse the original SMART "
                "prior; set diffusion.use_smart_prior_fusion: false."
            )
        if self.ntp_aux_loss_weight > 0.0:
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy no longer runs dense SMART NTP CE; "
                "set diffusion.ntp_aux_loss_weight: 0.0."
            )
        proposal_memory_cfg = getattr(diffusion_cfg, 'proposal_memory', None)
        if bool(getattr(proposal_memory_cfg, 'enabled', False)):
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy discards tail chunks; set "
                "diffusion.proposal_memory.enabled: false."
            )
        temporal_ensemble_cfg = getattr(diffusion_cfg, 'temporal_ensemble', None)
        if bool(getattr(temporal_ensemble_cfg, 'enabled', False)):
            raise ValueError(
                "SMARTDiscreteDiffusionPolicy does not ensemble overlapping tails; "
                "set diffusion.temporal_ensemble.enabled: false."
            )

        self.discrete_policy_span_tokens = int(
            getattr(diffusion_cfg, 'discrete_policy_span_tokens', 4)
        )
        self.discrete_policy_batched_multi_anchor = bool(
            getattr(diffusion_cfg, 'discrete_policy_batched_multi_anchor', True)
        )
        chunk_weights = tuple(
            float(value)
            for value in getattr(
                diffusion_cfg,
                'chunk_loss_weights',
                getattr(
                    diffusion_cfg,
                    'discrete_policy_chunk_loss_weights',
                    getattr(diffusion_cfg, 'causal_loss_weights', (1.0, 0.3, 0.15, 0.075)),
                ),
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
            float(
                getattr(
                    diffusion_cfg,
                    'overlap_loss_weight',
                    getattr(diffusion_cfg, 'discrete_policy_overlap_loss_weight', 0.05),
                )
            ),
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

    def _empty_discrete_policy_training_result(self, ref_tensor):
        zero = ref_tensor.new_zeros(())
        zero_chunks = zero.repeat(int(getattr(self, 'ar_prediction_tokens', 4)))
        return {
            'chunk_loss': zero,
            'mask_acc': zero,
            'overlap_loss': zero,
            'chunk_x0_losses': zero_chunks,
            'chunk_acc': zero_chunks,
            'valid_tokens': zero,
            'supervised_tokens': zero,
            'window_count': 0,
        }

    def _discrete_policy_chunk_metrics(self, packed, details):
        logits = details['logits']
        target = packed['token_ids']
        if logits.shape[:-1] != target.shape:
            raise AssertionError(
                "Discrete policy logits must have shape "
                "[batch, sequence, vocab] matching packed token ids."
            )
        chunk_ids = packed['chunk_ids'].to(device=target.device, dtype=torch.long)
        if chunk_ids.shape != target.shape:
            raise AssertionError(
                "Discrete policy chunk ids must match packed token id shape."
            )
        valid_mask = packed['valid_mask'].to(device=target.device, dtype=torch.bool)
        loss_mask_base = packed.get('loss_mask_base', valid_mask).to(
            device=target.device,
            dtype=torch.bool,
        ) & valid_mask
        loss_mask = details.get('loss_mask', loss_mask_base).to(
            device=target.device,
            dtype=torch.bool,
        )
        expected_loss_mask = loss_mask_base
        if not torch.equal(loss_mask, expected_loss_mask):
            raise AssertionError(
                "Discrete policy training must supervise every valid target in "
                "the current window; expected forced full-window loss mask."
            )

        log_prob = F.log_softmax(logits, dim=-1)
        nll = -log_prob.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        pred = logits.argmax(dim=-1)
        chunk_losses = []
        chunk_acc = []
        chunk_valid_counts = []
        chunk_supervised_counts = []
        for chunk_idx in range(self.ar_prediction_tokens):
            chunk_mask = chunk_ids == chunk_idx
            valid = loss_mask & chunk_mask
            valid_count = (valid_mask & chunk_mask).to(dtype=logits.dtype).sum()
            supervised_count = valid.to(dtype=logits.dtype).sum()
            chunk_valid_counts.append(valid_count)
            chunk_supervised_counts.append(supervised_count)
            if bool(valid.any()):
                chunk_losses.append(nll[valid].mean())
                chunk_acc.append((pred[valid] == target[valid]).float().mean())
            else:
                chunk_losses.append(logits.sum() * 0.0)
                chunk_acc.append(logits.new_zeros(()))
        return {
            'chunk_x0_losses': torch.stack(chunk_losses),
            'chunk_acc': torch.stack(chunk_acc),
            'chunk_valid_counts': torch.stack(chunk_valid_counts),
            'chunk_supervised_counts': torch.stack(chunk_supervised_counts),
            'valid_tokens': torch.stack(chunk_valid_counts).sum(),
            'supervised_tokens': torch.stack(chunk_supervised_counts).sum(),
        }

    def _split_anchor_source_scenes(self, data):
        to_data_list = getattr(data, 'to_data_list', None)
        if callable(to_data_list):
            return to_data_list()
        return [data]

    def _has_pt_token_store(self, data):
        return 'pt_token' in getattr(data, 'node_types', ())

    def _prepare_discrete_policy_anchor_view(self, data):
        if not self._has_pt_token_store(data):
            return data
        setattr(data, '_smart_diffusion_prepared', False)
        return self._prepare_batch(data)

    def _build_discrete_policy_batched_anchor_view(self, data, anchors):
        scenes = self._split_anchor_source_scenes(data)
        views = []
        for anchor in anchors:
            perturb = (
                getattr(self, 'training', False)
                and self.ar_state_perturb_prob > 0.0
                and torch.rand((), device=data['agent']['token_idx'].device) < self.ar_state_perturb_prob
            )
            perturb = bool(perturb)
            for scene_idx, scene in enumerate(scenes):
                view, _target_tokens, _target_valid, _anchor = self._build_ar_training_view(
                    scene,
                    anchor_token=int(anchor),
                    perturb=perturb,
                    allow_incomplete_window=True,
                )
                num_agents = int(view['agent']['token_idx'].shape[0])
                device = view['agent']['token_idx'].device
                view['agent']['source_scene_id'] = torch.full(
                    (num_agents,),
                    int(scene_idx),
                    dtype=torch.long,
                    device=device,
                )
                view['agent']['source_agent_id'] = torch.arange(
                    num_agents,
                    dtype=torch.long,
                    device=device,
                )
                view['agent']['source_anchor_token'] = torch.full(
                    (num_agents,),
                    int(anchor),
                    dtype=torch.long,
                    device=device,
                )
                views.append(view)
        if not views:
            return None, 0
        if len(views) == 1:
            return self._prepare_discrete_policy_anchor_view(views[0]), 1
        batched_view = Batch.from_data_list(views)
        return self._prepare_discrete_policy_anchor_view(batched_view), len(views)

    def _attach_discrete_policy_packed_metadata(self, packed, batched_view):
        if packed is None:
            return
        agent_ids = packed['token_agent_ids'].to(dtype=torch.long)
        valid_agent = agent_ids >= 0
        safe_agent_ids = agent_ids.clamp_min(0)
        for source_key, packed_key in (
            ('source_scene_id', 'source_scene_ids'),
            ('source_agent_id', 'source_agent_ids'),
            ('source_anchor_token', 'anchor_tokens'),
        ):
            if source_key not in batched_view['agent']:
                continue
            source_values = batched_view['agent'][source_key].to(
                device=agent_ids.device,
                dtype=torch.long,
            )
            packed_values = torch.full_like(agent_ids, -1)
            if bool(valid_agent.any()):
                packed_values[valid_agent] = source_values[safe_agent_ids[valid_agent]]
            packed[packed_key] = packed_values

    def _discrete_policy_batched_overlap_loss(self, packed, details):
        logits = details['logits']
        required = ('source_scene_ids', 'source_agent_ids', 'anchor_tokens')
        if any(key not in packed for key in required):
            return logits.sum() * 0.0
        loss_mask = details.get('loss_mask', packed['loss_mask_base']).to(
            device=logits.device,
            dtype=torch.bool,
        )
        valid = loss_mask & packed['valid_mask'].to(device=logits.device, dtype=torch.bool)
        scene_ids = packed['source_scene_ids'].to(device=logits.device, dtype=torch.long)
        agent_ids = packed['source_agent_ids'].to(device=logits.device, dtype=torch.long)
        anchor_tokens = packed['anchor_tokens'].to(device=logits.device, dtype=torch.long)
        chunk_ids = packed['chunk_ids'].to(device=logits.device, dtype=torch.long)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_valid = valid.reshape(-1)
        flat_scene = scene_ids.reshape(-1)
        flat_agent = agent_ids.reshape(-1)
        flat_anchor = anchor_tokens.reshape(-1)
        flat_chunk = chunk_ids.reshape(-1)

        target_map = {}
        target_mask = (
            flat_valid
            & (flat_chunk == 0)
            & (flat_scene >= 0)
            & (flat_agent >= 0)
            & (flat_anchor >= 0)
        )
        target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
        for index in target_indices.tolist():
            key = (
                int(flat_scene[index].item()),
                int(flat_agent[index].item()),
                int(flat_anchor[index].item()),
            )
            target_map[key] = int(index)

        source_indices = []
        matched_target_indices = []
        for source_chunk in range(1, self.ar_prediction_tokens):
            source_mask = (
                flat_valid
                & (flat_chunk == source_chunk)
                & (flat_scene >= 0)
                & (flat_agent >= 0)
                & (flat_anchor >= 0)
            )
            for index in torch.nonzero(source_mask, as_tuple=False).squeeze(-1).tolist():
                key = (
                    int(flat_scene[index].item()),
                    int(flat_agent[index].item()),
                    int(flat_anchor[index].item()) + int(source_chunk),
                )
                target_index = target_map.get(key)
                if target_index is None:
                    continue
                source_indices.append(int(index))
                matched_target_indices.append(target_index)

        if not source_indices:
            return logits.sum() * 0.0
        source = torch.tensor(source_indices, dtype=torch.long, device=logits.device)
        target = torch.tensor(matched_target_indices, dtype=torch.long, device=logits.device)
        teacher_log_prob = F.log_softmax(flat_logits[source], dim=-1)
        student_log_prob = F.log_softmax(flat_logits[target], dim=-1)
        teacher_prob = teacher_log_prob.exp().detach()
        kl = F.kl_div(
            student_log_prob,
            teacher_prob,
            reduction='none',
        ).sum(dim=-1)
        return kl.mean()

    def _compute_discrete_policy_batched_training_loss(self, data, anchors, ref_tensor):
        batched_view, window_count = self._build_discrete_policy_batched_anchor_view(
            data,
            anchors,
        )
        if batched_view is None or int(window_count) <= 0:
            return self._empty_discrete_policy_training_result(ref_tensor)
        (
            packed,
            summary,
            _ft,
            _fv,
            _generation_agents,
            _supervision_agents,
            _agent_batch,
        ) = self._build_diffusion_inputs(batched_view)
        if packed is None:
            return self._empty_discrete_policy_training_result(ref_tensor)
        self._attach_discrete_policy_packed_metadata(packed, batched_view)
        loss, acc, details = self._compute_diffusion_loss(
            packed,
            summary,
            forced_mask=packed['valid_mask'],
            return_details=True,
            loss_normalization='supervision_weight',
        )
        chunk_metrics = self._discrete_policy_chunk_metrics(packed, details)
        return {
            'chunk_loss': loss,
            'mask_acc': acc,
            'overlap_loss': self._discrete_policy_batched_overlap_loss(packed, details),
            'chunk_x0_losses': chunk_metrics['chunk_x0_losses'],
            'chunk_acc': chunk_metrics['chunk_acc'],
            'valid_tokens': chunk_metrics['valid_tokens'],
            'supervised_tokens': chunk_metrics['supervised_tokens'],
            'window_count': int(window_count),
        }

    def _compute_discrete_policy_training_loss(self, data, ref_tensor):
        anchors = self._discrete_policy_training_anchors(data)
        if not anchors:
            return self._empty_discrete_policy_training_result(ref_tensor)
        if bool(getattr(self, 'discrete_policy_batched_multi_anchor', True)):
            return self._compute_discrete_policy_batched_training_loss(
                data,
                anchors,
                ref_tensor,
            )

        num_agents = int(data['agent']['token_idx'].shape[0])
        losses = []
        accuracies = []
        chunk_losses = []
        chunk_accuracies = []
        chunk_supervised_counts = []
        valid_token_counts = []
        supervised_token_counts = []
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
            chunk_metrics = self._discrete_policy_chunk_metrics(packed, details)
            chunk_losses.append(chunk_metrics['chunk_x0_losses'])
            chunk_accuracies.append(chunk_metrics['chunk_acc'])
            chunk_supervised_counts.append(chunk_metrics['chunk_supervised_counts'])
            valid_token_counts.append(chunk_metrics['valid_tokens'])
            supervised_token_counts.append(chunk_metrics['supervised_tokens'])
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
            return self._empty_discrete_policy_training_result(ref_tensor)

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
        chunk_loss_stack = torch.stack(chunk_losses, dim=0)
        chunk_acc_stack = torch.stack(chunk_accuracies, dim=0)
        chunk_count_stack = torch.stack(chunk_supervised_counts, dim=0)
        chunk_count_sum = chunk_count_stack.sum(dim=0)
        chunk_count_denominator = chunk_count_sum.clamp_min(1.0)
        return {
            'chunk_loss': chunk_loss,
            'mask_acc': mask_acc,
            'overlap_loss': overlap_loss,
            'chunk_x0_losses': (
                chunk_loss_stack * chunk_count_stack
            ).sum(dim=0) / chunk_count_denominator,
            'chunk_acc': (
                chunk_acc_stack * chunk_count_stack
            ).sum(dim=0) / chunk_count_denominator,
            'valid_tokens': torch.stack(valid_token_counts).sum(),
            'supervised_tokens': torch.stack(supervised_token_counts).sum(),
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

        loss = (
            self.diffusion_loss_weight * policy['chunk_loss']
            + self.discrete_policy_overlap_loss_weight * policy['overlap_loss']
        )

        self.log('train_empty_diffusion_batch', loss.new_zeros(()),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=1)
        self.log('diffusion_loss', policy['chunk_loss'], prog_bar=True,
                 on_step=True, on_epoch=True, batch_size=1)
        self.log('discrete_policy_chunk_loss', policy['chunk_loss'],
                 prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('discrete_policy_overlap_loss', policy['overlap_loss'],
                 prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('loss_overlap', policy['overlap_loss'],
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('train_mask_acc', policy['mask_acc'],
                 on_step=True, on_epoch=True, batch_size=1)
        for chunk_idx in range(int(getattr(self, 'ar_prediction_tokens', 4))):
            self.log(
                f'loss_x0_chunk{chunk_idx}',
                policy['chunk_x0_losses'][chunk_idx],
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                f'chunk{chunk_idx}_acc',
                policy['chunk_acc'][chunk_idx],
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
        self.log(
            'train_discrete_policy_valid_tokens',
            policy['valid_tokens'],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        self.log(
            'train_discrete_policy_supervised_tokens',
            policy['supervised_tokens'],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        coverage = policy['supervised_tokens'] / policy['valid_tokens'].clamp_min(1.0)
        self.log(
            'train_discrete_policy_supervision_coverage',
            coverage,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
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
