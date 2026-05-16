import copy
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

from smart.model.distillation import (
    aggregate_rollout_weights,
    catk_recovery_targets,
    masked_entropy,
    masked_kl_div,
    rollout_quality_weights,
)
from smart.model.smart import SMART


def _cfg_get(config: Optional[Mapping], name: str, default=None):
    if config is None:
        return default
    if hasattr(config, name):
        return getattr(config, name)
    if isinstance(config, Mapping) and name in config:
        return config[name]
    return default


class SMARTSelfDistill(SMART):
    """SMART training variant with quality-diverse closed-loop distillation."""

    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        self.distill_config = _cfg_get(model_config, 'distill', None)
        self.distill_enabled = bool(_cfg_get(self.distill_config, 'enabled', False))
        self.teacher_ema_decay = float(_cfg_get(self.distill_config, 'teacher_ema_decay', 0.999))
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self._set_teacher_frozen()

    def _set_teacher_frozen(self):
        self.teacher_encoder.eval()
        for param in self.teacher_encoder.parameters():
            param.requires_grad_(False)

    def _sync_teacher(self, hard: bool = False):
        self.teacher_encoder.eval()
        with torch.no_grad():
            if hard:
                self.teacher_encoder.load_state_dict(self.encoder.state_dict())
                return
            decay = self.teacher_ema_decay
            for teacher_param, student_param in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
                teacher_param.data.mul_(decay).add_(student_param.data, alpha=1.0 - decay)
            for teacher_buffer, student_buffer in zip(self.teacher_encoder.buffers(), self.encoder.buffers()):
                teacher_buffer.copy_(student_buffer)

    def load_params_from_file(self, filename, logger, to_cpu=False):
        result = super().load_params_from_file(filename=filename, logger=logger, to_cpu=to_cpu)
        self._sync_teacher(hard=True)
        return result

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.distill_enabled:
            self._sync_teacher(hard=False)

    def _loss_weight(self, name: str, default: float) -> float:
        return float(_cfg_get(_cfg_get(self.distill_config, 'loss_weights', None), name, default))

    def _distill_value(self, name: str, default):
        return _cfg_get(self.distill_config, name, default)

    def _masked_weighted_mean(self, values: torch.Tensor, mask: torch.Tensor, weights: Optional[torch.Tensor] = None):
        if values.numel() == 0 or not mask.any():
            return values.new_zeros(())
        mask_f = mask.to(values.dtype)
        if weights is not None:
            mask_f = mask_f * weights.to(values.dtype)
        return (values * mask_f).sum() / mask_f.sum().clamp_min(1e-6)

    def _rollout_nll(self, logits: torch.Tensor, token_idx: torch.Tensor,
                     mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(logits, dim=-1)
        nll = -log_prob.gather(dim=-1, index=token_idx.unsqueeze(-1)).squeeze(-1)
        return self._masked_weighted_mean(nll, mask, weights)

    def _catk_loss(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0 or not mask.any():
            return logits.new_zeros(())
        return F.cross_entropy(logits[mask], targets[mask])

    def _compute_distill_losses(self, data, pred):
        num_steps = int(self._distill_value('rollout_steps', 4))
        num_rollouts = int(self._distill_value('num_rollouts', 2))
        topk = int(self._distill_value('rollout_topk', 5))
        temperature = float(self._distill_value('temperature', 2.0))
        weight_temperature = float(self._distill_value('weight_temperature', 1.0))

        map_enc = {'x_pt': pred['x_pt']}
        rollouts = self.encoder.rollout_tokens(
            data, map_enc=map_enc, num_steps=num_steps, num_rollouts=num_rollouts, topk=topk)
        if rollouts['token_idx'].numel() == 0:
            zero = pred['next_token_prob'].new_zeros(())
            return {
                'catk_loss': zero,
                'rollout_kl': zero,
                'rollout_nll': zero,
                'entropy_loss': zero,
                'entropy': zero,
                'rollout_reward': zero,
                'valid_diversity': zero,
            }

        student_scores = self.encoder.score_token_prefix(
            data, forced_token_idx=rollouts['token_idx'], num_steps=num_steps, map_enc=map_enc)
        self.teacher_encoder.eval()
        with torch.no_grad():
            teacher_scores = self.teacher_encoder.score_token_prefix(
                data, forced_token_idx=rollouts['token_idx'], num_steps=num_steps)

        score_mask = student_scores['mask'] & rollouts['mask']
        gt = data['agent']['position'][:, self.num_historical_steps:, :self.input_dim].contiguous()
        valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps:]
        agent_batch = data['agent']['batch'] if isinstance(data, Batch) else None
        rewards, reward_components = rollout_quality_weights(
            rollouts['pred_traj'],
            rollouts['pred_head'],
            gt,
            valid_mask,
            _cfg_get(self.distill_config, 'reward_weights', None),
            agent_batch=agent_batch,
        )
        weights = aggregate_rollout_weights(rewards, score_mask, temperature=weight_temperature)

        rollout_kl = masked_kl_div(
            student_scores['logits'], teacher_scores['logits'], score_mask, temperature, weights=weights)
        rollout_nll = self._rollout_nll(student_scores['logits'], rollouts['token_idx'], score_mask, weights)
        entropy = masked_entropy(student_scores['logits'], score_mask, weights=weights)
        entropy_loss = -entropy

        catk_targets, catk_mask, _ = catk_recovery_targets(
            rollouts['candidate_idx'], rollouts['candidate_trajs'], gt, valid_mask)
        catk_mask = catk_mask & score_mask
        catk_loss = self._catk_loss(student_scores['logits'], catk_targets, catk_mask)

        return {
            'catk_loss': catk_loss,
            'rollout_kl': rollout_kl,
            'rollout_nll': rollout_nll,
            'entropy_loss': entropy_loss,
            'entropy': entropy.detach(),
            'rollout_reward': reward_components['reward'],
            'valid_diversity': reward_components['diversity'],
        }

    def training_step(self,
                      data,
                      batch_idx):
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]

        pred = self(data)
        next_token_prob = pred['next_token_prob']
        next_token_idx_gt = pred['next_token_idx_gt']
        next_token_eval_mask = pred['next_token_eval_mask']
        cls_loss = self.cls_loss(next_token_prob[next_token_eval_mask], next_token_idx_gt[next_token_eval_mask])

        if not self.distill_enabled:
            loss = cls_loss
            self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            self.log('cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            return loss

        distill_losses = self._compute_distill_losses(data, pred)
        loss = self._loss_weight('gt_ce', 1.0) * cls_loss
        loss = loss + self._loss_weight('catk_ce', 0.5) * distill_losses['catk_loss']
        loss = loss + self._loss_weight('rollout_kl', 0.2) * distill_losses['rollout_kl']
        loss = loss + self._loss_weight('rollout_nll', 0.1) * distill_losses['rollout_nll']
        loss = loss + self._loss_weight('entropy', 0.01) * distill_losses['entropy_loss']

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('catk_loss', distill_losses['catk_loss'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('rollout_kl', distill_losses['rollout_kl'], prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('rollout_nll', distill_losses['rollout_nll'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('rollout_entropy', distill_losses['entropy'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('rollout_reward', distill_losses['rollout_reward'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log('valid_diversity', distill_losses['valid_diversity'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        return loss
