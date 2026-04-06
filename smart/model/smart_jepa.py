from typing import Dict, Tuple

import torch
from torch_geometric.data import Batch, HeteroData

from smart.model.jepa import JointEmbeddingPredictiveModule
from smart.model.smart import SMART
from smart.utils import wrap_angle


class SMARTJEPA(SMART):
    def __init__(self, model_config) -> None:
        super().__init__(model_config)
        if not hasattr(model_config, "jepa") or not getattr(model_config.jepa, "enabled", False):
            raise ValueError("SMARTJEPA requires Model.jepa.enabled = true in the config.")

        future_chunk_steps = int(model_config.jepa.future_chunk_steps)
        if self.num_future_steps % future_chunk_steps != 0:
            raise ValueError(
                f"decoder.num_future_steps ({self.num_future_steps}) must be divisible by "
                f"jepa.future_chunk_steps ({future_chunk_steps})."
            )

        self.inference_token = bool(getattr(model_config, "inference_token", False))
        self.rollout_num = int(getattr(model_config, "rollout_num", 1))
        self.jepa_aux_loss_weight = float(model_config.jepa.aux_loss_weight)
        self.jepa_ema_decay = float(model_config.jepa.ema_decay)
        self.jepa = JointEmbeddingPredictiveModule(
            hidden_dim=self.hidden_dim,
            future_steps=self.num_future_steps,
            future_chunk_steps=future_chunk_steps,
            num_heads=self.model_config.num_heads,
            dropout=self.model_config.dropout,
            mask_ratio=float(model_config.jepa.mask_ratio),
        )

    def training_step(self, data, batch_idx):
        data = self._prepare_batch(data)
        pred = self(data)
        cls_loss = self._compute_cls_loss(pred)
        jepa_loss, jepa_stats = self.compute_jepa_loss(data)
        loss = cls_loss + self.jepa_aux_loss_weight * jepa_loss

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('jepa_loss', jepa_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('jepa_cosine', jepa_stats['jepa_cosine'], prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        self.log(
            'jepa_masked_fraction',
            jepa_stats['masked_fraction'],
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=1,
        )
        return loss

    def validation_step(self, data, batch_idx):
        data = self._prepare_batch(data)
        pred = self(data)
        cls_loss = self._compute_cls_loss(pred)
        jepa_loss, jepa_stats = self.compute_jepa_loss(data)
        total_loss = cls_loss + self.jepa_aux_loss_weight * jepa_loss

        next_token_idx = pred['next_token_idx']
        next_token_idx_gt = pred['next_token_idx_gt']
        next_token_eval_mask = pred['next_token_eval_mask']

        self.TokenCls.update(
            pred=next_token_idx[next_token_eval_mask],
            target=next_token_idx_gt[next_token_eval_mask],
            valid_mask=next_token_eval_mask[next_token_eval_mask],
        )
        self.log('val_cls_acc', self.TokenCls, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_loss', cls_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_jepa_loss', jepa_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log('val_total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log(
            'val_jepa_cosine',
            jepa_stats['jepa_cosine'],
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )

        if self.inference_token:
            pred_inference = self.inference(data)
            gt = pred_inference['gt']
            valid_mask = data['agent']['valid_mask'][:, self.num_historical_steps:]
            pred_traj = pred_inference['pred_traj']
            eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
            self.minADE.update(pred=pred_traj[eval_mask], target=gt[eval_mask], valid_mask=valid_mask[eval_mask])
            self.minFDE.update(pred=pred_traj[eval_mask], target=gt[eval_mask], valid_mask=valid_mask[eval_mask])
            self.log('val_minADE', self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)

    def compute_jepa_loss(self, data: HeteroData) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        context = self.encoder.encode_history_context(data)
        history_tokens = context['x_a_history']
        history_mask = context['history_token_mask']
        agent_batch = context['agent_batch']
        scene_summary = self._pool_scene_summary(
            history_tokens,
            history_mask,
            agent_batch,
            data['agent']['type'],
        )
        future_states, future_mask, valid_agents = self._build_future_targets(data)
        return self.jepa(scene_summary, future_states, valid_agents, future_mask, agent_batch)

    def _prepare_batch(self, data: HeteroData) -> HeteroData:
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        return data

    def _compute_cls_loss(self, pred: Dict[str, torch.Tensor]) -> torch.Tensor:
        next_token_prob = pred['next_token_prob']
        next_token_idx_gt = pred['next_token_idx_gt']
        next_token_eval_mask = pred['next_token_eval_mask']
        return self.cls_loss(next_token_prob[next_token_eval_mask], next_token_idx_gt[next_token_eval_mask])

    def _pool_scene_summary(
        self,
        history_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        agent_batch: torch.Tensor,
        agent_type: torch.Tensor,
    ) -> torch.Tensor:
        valid_mask = history_mask & (agent_type != 3).unsqueeze(-1)
        hidden_dim = history_tokens.shape[-1]
        if agent_batch.numel() == 0:
            return history_tokens.new_zeros((1, hidden_dim))

        num_graphs = int(agent_batch.max().item()) + 1
        flat_tokens = history_tokens.reshape(-1, hidden_dim)
        flat_mask = valid_mask.reshape(-1)
        flat_batch = agent_batch.unsqueeze(-1).expand(-1, history_tokens.shape[1]).reshape(-1)
        pooled = history_tokens.new_zeros((num_graphs, hidden_dim))
        counts = history_tokens.new_zeros((num_graphs,))
        if flat_mask.any():
            valid_batch = flat_batch[flat_mask]
            pooled.index_add_(0, valid_batch, flat_tokens[flat_mask])
            counts.index_add_(0, valid_batch, torch.ones(valid_batch.shape[0], device=counts.device))
        return pooled / counts.clamp_min(1).unsqueeze(-1)

    def _build_future_targets(self, data: HeteroData) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hist = self.num_historical_steps
        future_slice = slice(hist, hist + self.num_future_steps)
        position = data['agent']['position'][:, :, :2].float()
        heading = data['agent']['heading'].float()
        velocity = data['agent']['velocity'].float()
        valid_mask = data['agent']['valid_mask'].bool()

        if position.shape[1] < hist + self.num_future_steps:
            raise ValueError(
                f"Expected at least {hist + self.num_future_steps} agent steps, got {position.shape[1]}."
            )

        origin = position[:, hist - 1]
        theta = heading[:, hist - 1]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = theta.new_zeros(position.shape[0], 2, 2)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = -sin
        rot_mat[:, 1, 0] = sin
        rot_mat[:, 1, 1] = cos

        future_position = torch.bmm(position[:, future_slice] - origin[:, None, :], rot_mat)
        future_heading = wrap_angle(heading[:, future_slice] - theta.unsqueeze(-1))
        speed = torch.norm(velocity[:, future_slice, :2], dim=-1)
        future_states = torch.stack(
            [future_position[..., 0], future_position[..., 1], future_heading, speed],
            dim=-1,
        )
        future_valid = valid_mask[:, future_slice]
        future_states[~future_valid] = 0

        valid_agents = (
            valid_mask[:, hist - 1]
            & future_valid.any(dim=-1)
            & (data['agent']['type'] != 3)
        )
        return future_states, future_valid, valid_agents

    def on_before_zero_grad(self, optimizer) -> None:
        self.jepa.update_target_encoder(self.jepa_ema_decay)
