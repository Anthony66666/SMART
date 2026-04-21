from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
from torch_geometric.data import Batch, HeteroData

from smart.model.jepa import JointEmbeddingPredictiveModule
from smart.model.smart import SMART
from smart.utils.torch_compat import torch_load_compat
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

        self.future_chunk_steps = future_chunk_steps
        self.num_future_chunks = self.num_future_steps // self.future_chunk_steps
        self.inference_token = bool(getattr(model_config, "inference_token", False))
        self.rollout_num = int(getattr(model_config, "rollout_num", 1))
        self.jepa_aux_loss_weight = float(model_config.jepa.aux_loss_weight)
        self.jepa_ema_decay = float(model_config.jepa.ema_decay)
        self.pretrain_objective = str(getattr(model_config.jepa, "pretrain_objective", "legacy"))
        self.mask_strategy = str(getattr(model_config.jepa, "mask_strategy", "random_chunk"))
        self.agent_mask_radius = float(getattr(model_config.jepa, "agent_mask_radius", model_config.decoder.a2a_radius))
        self.map_mask_radius = float(getattr(model_config.jepa, "map_mask_radius", model_config.decoder.pl2a_radius))
        self.num_masked_agent_regions_per_scene = int(getattr(model_config.jepa, "num_masked_agent_regions_per_scene", 1))
        self.num_masked_time_windows = int(getattr(model_config.jepa, "num_masked_time_windows", 1))
        self.map_block_unit = str(getattr(model_config.jepa, "map_block_unit", "polygon"))
        self.joint_predictor = bool(getattr(model_config.jepa, "joint_predictor", True))
        self.predict_map_latent = bool(getattr(model_config.jepa, "predict_map_latent", False))
        self.agent_loss_weight = float(getattr(model_config.jepa, "agent_loss_weight", 0.5))
        self.map_loss_weight = float(getattr(model_config.jepa, "map_loss_weight", 0.5))
        self.disable_map_mae_aux_when_jepa = bool(getattr(model_config.jepa, "disable_map_mae_aux_when_jepa", True))
        self.mask_ratio = float(getattr(model_config.jepa, "mask_ratio", 0.5))
        self.training_stage = str(getattr(model_config.jepa, "training_stage", "joint"))
        self.masked_agent_history_mode = str(getattr(model_config.jepa, "masked_agent_history_mode", "visible"))
        self.masked_agent_history_dropout_ratio = float(
            getattr(model_config.jepa, "masked_agent_history_dropout_ratio", 0.5)
        )
        self.masked_agent_history_min_visible_tokens = int(
            getattr(model_config.jepa, "masked_agent_history_min_visible_tokens", 1)
        )
        self.forecast_aligned_agent_region_radius = 30.0
        self.forecast_aligned_map_region_radius = 30.0

        if self.map_block_unit != "polygon":
            raise ValueError(f"Unsupported map_block_unit: {self.map_block_unit}")
        if self.training_stage not in {"joint", "pretrain"}:
            raise ValueError(
                f"Unsupported jepa.training_stage: {self.training_stage}"
            )
        if self.pretrain_objective not in {"legacy", "forecast_aligned_ego30_v1"}:
            raise ValueError(f"Unsupported jepa.pretrain_objective: {self.pretrain_objective}")
        if self.masked_agent_history_mode not in {"visible", "partial_dropout", "hidden"}:
            raise ValueError(
                f"Unsupported jepa.masked_agent_history_mode: {self.masked_agent_history_mode}"
            )
        if self.pretrain_objective == "forecast_aligned_ego30_v1":
            if self.training_stage != "pretrain":
                raise ValueError("forecast_aligned_ego30_v1 is only supported for jepa.training_stage = pretrain.")
            if not self.joint_predictor or not self.predict_map_latent:
                raise ValueError("forecast_aligned_ego30_v1 requires joint_predictor = true and predict_map_latent = true.")

        self.jepa = JointEmbeddingPredictiveModule(
            hidden_dim=self.hidden_dim,
            future_steps=self.num_future_steps,
            future_chunk_steps=future_chunk_steps,
            num_heads=self.model_config.num_heads,
            dropout=self.model_config.dropout,
            num_freq_bands=self.model_config.num_freq_bands,
            mask_ratio=self.mask_ratio,
            agent_loss_weight=self.agent_loss_weight,
            map_loss_weight=self.map_loss_weight,
        )
        # Keep a separate EMA teacher for map features. Using the online map encoder
        # directly as the target makes the JEPA target drift with the student.
        self.target_map_backbone = deepcopy(self.encoder.map_encoder)
        for parameter in self.target_map_backbone.parameters():
            parameter.requires_grad = False

    def training_step(self, data, batch_idx):
        data = self._prepare_batch(data)
        if self.training_stage == "pretrain":
            jepa_loss, jepa_stats = self.compute_jepa_loss(data)
            self.log('train_loss', jepa_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            self.log('jepa_loss', jepa_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
            self._log_jepa_stats(jepa_stats, prefix="", on_step=True, on_epoch=True, sync_dist=False)
            return jepa_loss

        pred = self(data)
        cls_loss = self._compute_cls_loss(pred)
        jepa_loss, jepa_stats = self.compute_jepa_loss(data)
        loss = cls_loss + self.jepa_aux_loss_weight * jepa_loss

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('cls_loss', cls_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log('jepa_loss', jepa_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self._log_jepa_stats(jepa_stats, prefix="", on_step=True, on_epoch=True, sync_dist=False)
        return loss

    def load_params_from_file(self, filename, logger, to_cpu=False):
        it, epoch = super().load_params_from_file(filename=filename, logger=logger, to_cpu=to_cpu)
        loc_type = torch.device('cpu') if to_cpu else None
        checkpoint = torch_load_compat(filename, map_location=loc_type, weights_only=False)
        state_dict = checkpoint['state_dict']
        if not any(key.startswith('target_map_backbone.') for key in state_dict.keys()):
            logger.info('Checkpoint missing target_map_backbone; syncing EMA map teacher from online map encoder.')
            self.target_map_backbone.load_state_dict(self.encoder.map_encoder.state_dict())
        return it, epoch

    def validation_step(self, data, batch_idx):
        data = self._prepare_batch(data)
        if self.training_stage == "pretrain":
            jepa_loss, jepa_stats = self.compute_jepa_loss(data)
            self.log('val_loss', jepa_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self.log('val_jepa_loss', jepa_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
            self._log_jepa_stats(jepa_stats, prefix="val_", on_step=False, on_epoch=True, sync_dist=True)
            return

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
        self._log_jepa_stats(jepa_stats, prefix="val_", on_step=False, on_epoch=True, sync_dist=True)

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
        future_states, future_mask, valid_agents = self._build_future_targets(data)
        agent_chunk_mask = self._build_agent_chunk_mask(valid_agents, future_mask)
        agent_graph_index = self._get_node_batch(data, 'agent')
        history_pos = data['agent']['position'][:, self.num_historical_steps - 1, :2].float()
        graph_centers = self._compute_graph_centers(
            positions=history_pos,
            graph_index=agent_graph_index,
            valid_mask=valid_agents,
        )
        agent_token_positions = history_pos - graph_centers[agent_graph_index]
        map_token_positions = None
        if self.predict_map_latent and self.joint_predictor:
            polygon_centers, polygon_valid_mask, polygon_graph_index = self._compute_polygon_centers(data)
            map_token_positions = polygon_centers - graph_centers[polygon_graph_index]
            map_token_positions = map_token_positions.masked_fill(~polygon_valid_mask.unsqueeze(-1), 0.0)
        jepa_masks = self._build_jepa_masks(
            data,
            agent_chunk_mask,
            agent_graph_index,
        )

        agent_include_mask = jepa_masks['agent_include_mask'] & agent_chunk_mask
        agent_input_hidden_mask = jepa_masks['agent_input_hidden_mask'] & agent_include_mask
        agent_loss_mask = jepa_masks['agent_loss_mask'] & agent_include_mask
        map_visible_mask = jepa_masks['map_visible_mask']

        agent_history_mask, history_stats = self._build_masked_agent_history_context_mask(
            data,
            jepa_masks['target_agent_mask'],
            history_mode=self._get_effective_masked_agent_history_mode(),
        )
        context = self.encoder.encode_history_context(
            data,
            map_visible_mask=map_visible_mask,
            agent_history_mask=agent_history_mask,
        )
        history_tokens = context['x_a_history']
        history_mask = context['history_token_mask']
        scene_summary = self._pool_scene_summary(
            history_tokens,
            history_mask,
            agent_graph_index,
            data['agent']['type'],
        )

        map_online_features = None
        map_target_features = None
        map_valid_mask = None
        map_graph_index = None
        map_include_mask = jepa_masks['map_include_mask']
        map_input_hidden_mask = jepa_masks['map_input_hidden_mask']
        map_loss_mask = jepa_masks['map_loss_mask']
        if self.predict_map_latent and self.joint_predictor:
            map_online_features, online_map_valid_mask, map_graph_index = self._pool_map_polygon_features(
                context['x_pt'],
                data,
                token_visible_mask=context.get('pt_visibility_mask'),
            )
            with torch.no_grad():
                was_training = self.target_map_backbone.training
                self.target_map_backbone.eval()
                target_map_enc = self.target_map_backbone(
                    data,
                    disable_prediction=self.disable_map_mae_aux_when_jepa,
                )
                if was_training and self.training:
                    self.target_map_backbone.train()
            map_target_features, target_map_valid_mask, _ = self._pool_map_polygon_features(
                target_map_enc['x_pt'],
                data,
            )
            map_valid_mask = online_map_valid_mask & target_map_valid_mask
            map_include_mask = map_include_mask & map_valid_mask
            map_input_hidden_mask = map_input_hidden_mask & map_include_mask
            map_loss_mask = map_loss_mask & map_include_mask

        jepa_loss, jepa_stats = self.jepa(
            scene_summary=scene_summary,
            future_states=future_states,
            valid_agents=valid_agents,
            future_mask=future_mask,
            agent_graph_index=agent_graph_index,
            agent_include_mask=agent_include_mask,
            agent_input_hidden_mask=agent_input_hidden_mask,
            agent_loss_mask=agent_loss_mask,
            agent_token_positions=agent_token_positions,
            map_online_features=map_online_features,
            map_target_features=map_target_features,
            map_valid_mask=map_valid_mask,
            map_graph_index=map_graph_index,
            map_include_mask=map_include_mask,
            map_input_hidden_mask=map_input_hidden_mask,
            map_loss_mask=map_loss_mask,
            map_token_positions=map_token_positions,
            pack_hidden_map_tokens_last=self._uses_forecast_aligned_pretrain_objective(),
        )
        jepa_stats.update(history_stats)
        agent_visible_future_tokens = (agent_include_mask & ~agent_input_hidden_mask).sum().to(torch.float32)
        total_included_agent_future_tokens = agent_include_mask.sum().clamp_min(1).to(torch.float32)
        jepa_stats.update(
            {
                'target_agent_count': jepa_masks['target_agent_mask'].sum().to(torch.float32),
                'target_polygon_count': jepa_masks['target_polygon_mask'].sum().to(torch.float32),
                'student_agent_future_visible_fraction': agent_visible_future_tokens / total_included_agent_future_tokens,
            }
        )
        return jepa_loss, jepa_stats

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

    def _uses_forecast_aligned_pretrain_objective(self) -> bool:
        return self.pretrain_objective == 'forecast_aligned_ego30_v1'

    def _get_effective_masked_agent_history_mode(self) -> str:
        if self._uses_forecast_aligned_pretrain_objective():
            return 'visible'
        return self.masked_agent_history_mode

    def _get_effective_agent_region_radius(self) -> float:
        if self._uses_forecast_aligned_pretrain_objective():
            return self.forecast_aligned_agent_region_radius
        return self.agent_mask_radius

    def _get_effective_map_region_radius(self) -> float:
        if self._uses_forecast_aligned_pretrain_objective():
            return self.forecast_aligned_map_region_radius
        return self.map_mask_radius

    def _log_jepa_stats(
        self,
        jepa_stats: Dict[str, torch.Tensor],
        prefix: str,
        on_step: bool,
        on_epoch: bool,
        sync_dist: bool,
    ) -> None:
        name = lambda metric: f"{prefix}{metric}"
        self.log(name('jepa_cosine'), jepa_stats['jepa_cosine'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        if prefix == "":
            self.log(name('jepa_masked_fraction'), jepa_stats['masked_fraction'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('agent_jepa_loss'), jepa_stats['agent_jepa_loss'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('map_jepa_loss'), jepa_stats['map_jepa_loss'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('masked_agent_count'), jepa_stats['masked_agent_count'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('masked_map_count'), jepa_stats['masked_map_count'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('target_agent_count'), jepa_stats['target_agent_count'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('target_polygon_count'), jepa_stats['target_polygon_count'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('student_agent_future_visible_fraction'), jepa_stats['student_agent_future_visible_fraction'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('masked_agent_history_visible_fraction'), jepa_stats['masked_agent_history_visible_fraction'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('masked_agent_history_visible_tokens'), jepa_stats['masked_agent_history_visible_tokens'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)
        self.log(name('masked_agent_history_total_tokens'), jepa_stats['masked_agent_history_total_tokens'], prog_bar=False, on_step=on_step, on_epoch=on_epoch, batch_size=1, sync_dist=sync_dist)

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

    def _build_agent_chunk_mask(
        self,
        valid_agents: torch.Tensor,
        future_valid: torch.Tensor,
    ) -> torch.Tensor:
        return valid_agents.unsqueeze(-1) & future_valid.view(
            future_valid.shape[0],
            self.num_future_chunks,
            self.future_chunk_steps,
        ).all(dim=-1)

    def _build_jepa_masks(
        self,
        data: HeteroData,
        agent_chunk_mask: torch.Tensor,
        agent_graph_index: torch.Tensor,
    ) -> Dict[str, Optional[torch.Tensor]]:
        device = agent_chunk_mask.device
        num_polygons = int(data['map_polygon']['num_nodes'])
        zero_map_mask = torch.zeros(num_polygons, dtype=torch.bool, device=device)
        map_enabled = self.predict_map_latent and self.joint_predictor

        if self._uses_forecast_aligned_pretrain_objective():
            (
                target_agent_mask,
                agent_loss_mask,
                target_polygon_mask,
                map_visible_mask,
            ) = self._build_forecast_aligned_ego30_masks(
                data=data,
                agent_chunk_mask=agent_chunk_mask,
                agent_graph_index=agent_graph_index,
            )
            return {
                'target_agent_mask': target_agent_mask,
                'agent_include_mask': agent_loss_mask.clone(),
                'agent_input_hidden_mask': agent_loss_mask.clone(),
                'agent_loss_mask': agent_loss_mask,
                'target_polygon_mask': target_polygon_mask,
                'map_include_mask': torch.ones(num_polygons, dtype=torch.bool, device=device) if map_enabled else zero_map_mask,
                'map_input_hidden_mask': target_polygon_mask.clone() if map_enabled else zero_map_mask,
                'map_loss_mask': target_polygon_mask.clone() if map_enabled else zero_map_mask,
                'map_visible_mask': map_visible_mask if map_enabled else None,
            }

        if self.mask_strategy == 'interaction_multiblock':
            agent_loss_mask, target_polygon_mask, map_visible_mask = self._build_interaction_joint_masks(
                data,
                agent_chunk_mask,
                agent_graph_index,
            )
        elif self.mask_strategy == 'random_chunk':
            agent_loss_mask = self._build_random_chunk_mask(agent_chunk_mask, agent_graph_index)
            target_polygon_mask = zero_map_mask
            map_visible_mask = None
        else:
            raise ValueError(f"Unsupported jepa.mask_strategy: {self.mask_strategy}")

        if not map_enabled:
            target_polygon_mask = zero_map_mask
            map_visible_mask = None

        return {
            'target_agent_mask': agent_loss_mask.any(dim=-1),
            'agent_include_mask': agent_chunk_mask.clone(),
            'agent_input_hidden_mask': agent_loss_mask.clone(),
            'agent_loss_mask': agent_loss_mask,
            'target_polygon_mask': target_polygon_mask,
            'map_include_mask': torch.ones(num_polygons, dtype=torch.bool, device=device) if map_enabled else zero_map_mask,
            'map_input_hidden_mask': target_polygon_mask.clone() if map_enabled else zero_map_mask,
            'map_loss_mask': target_polygon_mask.clone() if map_enabled else zero_map_mask,
            'map_visible_mask': map_visible_mask if map_enabled else None,
        }

    def _build_forecast_aligned_ego30_masks(
        self,
        data: HeteroData,
        agent_chunk_mask: torch.Tensor,
        agent_graph_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = agent_chunk_mask.device
        num_agents = agent_chunk_mask.shape[0]
        num_polygons = int(data['map_polygon']['num_nodes'])
        history_step = self.num_historical_steps - 1
        history_pos = data['agent']['position'][:, history_step, :2].float()
        history_valid = data['agent']['valid_mask'][:, history_step].bool()
        non_background = data['agent']['type'] != 3

        target_agent_mask = torch.zeros(num_agents, dtype=torch.bool, device=device)
        agent_loss_mask = torch.zeros_like(agent_chunk_mask)
        target_polygon_mask = torch.zeros(num_polygons, dtype=torch.bool, device=device)
        polygon_centers, polygon_valid_mask, polygon_graph_index = self._compute_polygon_centers(data)
        map_visible_mask = torch.ones(num_polygons, dtype=torch.bool, device=device)

        num_graphs = int(agent_graph_index.max().item()) + 1 if agent_graph_index.numel() > 0 else 1
        ego_indices = self._get_ego_indices(data, num_graphs)
        for graph_id in range(num_graphs):
            graph_agent_indices = torch.nonzero(agent_graph_index == graph_id, as_tuple=False).squeeze(-1)
            if graph_agent_indices.numel() == 0:
                continue

            ego_index = int(ego_indices[graph_id].item())
            if ego_index < 0 or ego_index >= num_agents:
                continue
            ego_position = history_pos[ego_index]
            candidate_mask = (
                (agent_graph_index == graph_id)
                & history_valid
                & non_background
            )
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
            if candidate_indices.numel() > 0:
                candidate_distance = torch.norm(history_pos[candidate_indices] - ego_position.unsqueeze(0), dim=-1)
                region_agent_indices = candidate_indices[candidate_distance <= self.forecast_aligned_agent_region_radius]
            else:
                region_agent_indices = candidate_indices

            target_agent_mask[region_agent_indices] = True
            target_agent_mask[ego_index] = True
            if region_agent_indices.numel() == 0:
                region_agent_indices = ego_indices[graph_id].view(1)

            agent_loss_mask[region_agent_indices] = agent_chunk_mask[region_agent_indices]

            graph_polygon_indices = torch.nonzero(
                (polygon_graph_index == graph_id) & polygon_valid_mask,
                as_tuple=False,
            ).squeeze(-1)
            if graph_polygon_indices.numel() == 0:
                continue

            polygon_distance = torch.norm(
                polygon_centers[graph_polygon_indices] - ego_position.unsqueeze(0),
                dim=-1,
            )
            region_polygon_indices = graph_polygon_indices[
                polygon_distance <= self.forecast_aligned_map_region_radius
            ]
            if region_polygon_indices.numel() == 0:
                region_polygon_indices = graph_polygon_indices[polygon_distance.argmin()].view(1)
            target_polygon_mask[region_polygon_indices] = True

        map_visible_mask[target_polygon_mask] = False
        return target_agent_mask, agent_loss_mask, target_polygon_mask, map_visible_mask

    def _get_ego_indices(
        self,
        data: HeteroData,
        num_graphs: int,
    ) -> torch.Tensor:
        av_index = data['agent']['av_index']
        device = data['agent']['position'].device
        if torch.is_tensor(av_index):
            ego_indices = av_index.to(device=device, dtype=torch.long).reshape(-1)
            if ego_indices.numel() == 1 and num_graphs > 1:
                ego_indices = ego_indices.expand(num_graphs)
            return ego_indices
        return torch.full((num_graphs,), int(av_index), dtype=torch.long, device=device)

    def _build_random_chunk_mask(
        self,
        agent_chunk_mask: torch.Tensor,
        agent_graph_index: torch.Tensor,
    ) -> torch.Tensor:
        prediction_mask = (torch.rand_like(agent_chunk_mask.float()) < self.mask_ratio) & agent_chunk_mask
        if agent_graph_index.numel() == 0:
            return prediction_mask

        num_graphs = int(agent_graph_index.max().item()) + 1
        for graph_id in range(num_graphs):
            graph_agents = torch.nonzero(agent_graph_index == graph_id, as_tuple=False).squeeze(-1)
            if graph_agents.numel() == 0:
                continue
            graph_valid = agent_chunk_mask[graph_agents]
            if not graph_valid.any() or prediction_mask[graph_agents].any():
                continue
            first_valid = torch.nonzero(graph_valid, as_tuple=False)[0]
            prediction_mask[graph_agents[first_valid[0]], first_valid[1]] = True
        return prediction_mask

    def _build_interaction_joint_masks(
        self,
        data: HeteroData,
        agent_chunk_mask: torch.Tensor,
        agent_graph_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        device = agent_chunk_mask.device
        num_agents, num_chunks = agent_chunk_mask.shape
        num_polygons = int(data['map_polygon']['num_nodes'])
        agent_prediction_mask = torch.zeros_like(agent_chunk_mask)
        map_prediction_mask = torch.zeros(num_polygons, dtype=torch.bool, device=device)
        history_pos = data['agent']['position'][:, self.num_historical_steps - 1, :2].float()
        polygon_centers, polygon_valid_mask, polygon_graph_index = self._compute_polygon_centers(data)
        window_length = max(1, min(num_chunks, int(round(self.mask_ratio * num_chunks))))

        num_graphs = int(agent_graph_index.max().item()) + 1 if agent_graph_index.numel() > 0 else 1
        for graph_id in range(num_graphs):
            graph_agent_indices = torch.nonzero(
                (agent_graph_index == graph_id) & agent_chunk_mask.any(dim=-1),
                as_tuple=False,
            ).squeeze(-1)
            if graph_agent_indices.numel() == 0:
                continue

            for _ in range(self.num_masked_agent_regions_per_scene):
                anchor_index = graph_agent_indices[torch.randint(graph_agent_indices.numel(), (1,), device=device)].item()
                anchor_position = history_pos[anchor_index]
                neighbor_dist = torch.norm(history_pos[graph_agent_indices] - anchor_position.unsqueeze(0), dim=-1)
                region_agent_indices = graph_agent_indices[neighbor_dist <= self.agent_mask_radius]
                if region_agent_indices.numel() == 0:
                    region_agent_indices = torch.tensor([anchor_index], device=device)

                valid_chunks = torch.nonzero(agent_chunk_mask[region_agent_indices].any(dim=0), as_tuple=False).squeeze(-1)
                if valid_chunks.numel() == 0:
                    continue

                for _ in range(self.num_masked_time_windows):
                    chunk_start = self._sample_time_window_start(valid_chunks, num_chunks, window_length, device)
                    time_window = torch.zeros(num_chunks, dtype=torch.bool, device=device)
                    time_window[chunk_start: chunk_start + window_length] = True
                    agent_prediction_mask[region_agent_indices] |= agent_chunk_mask[region_agent_indices] & time_window

                if self.predict_map_latent and self.joint_predictor and polygon_centers.numel() > 0:
                    region_center = history_pos[region_agent_indices].mean(dim=0)
                    graph_polygon_indices = torch.nonzero(
                        (polygon_graph_index == graph_id) & polygon_valid_mask,
                        as_tuple=False,
                    ).squeeze(-1)
                    if graph_polygon_indices.numel() == 0:
                        continue
                    polygon_dist = torch.norm(polygon_centers[graph_polygon_indices] - region_center.unsqueeze(0), dim=-1)
                    region_polygon_indices = graph_polygon_indices[polygon_dist <= self.map_mask_radius]
                    if region_polygon_indices.numel() == 0:
                        nearest_polygon = graph_polygon_indices[polygon_dist.argmin()]
                        region_polygon_indices = nearest_polygon.view(1)
                    map_prediction_mask[region_polygon_indices] = True

            if not agent_prediction_mask[graph_agent_indices].any():
                first_valid = torch.nonzero(agent_chunk_mask[graph_agent_indices], as_tuple=False)[0]
                agent_prediction_mask[graph_agent_indices[first_valid[0]], first_valid[1]] = True

        if self.predict_map_latent and self.joint_predictor:
            map_visible_mask = torch.ones(num_polygons, dtype=torch.bool, device=device)
            map_visible_mask[map_prediction_mask] = False
        else:
            map_visible_mask = None
        return agent_prediction_mask, map_prediction_mask, map_visible_mask

    def _sample_time_window_start(
        self,
        valid_chunks: torch.Tensor,
        num_chunks: int,
        window_length: int,
        device: torch.device,
    ) -> int:
        max_start = max(0, num_chunks - window_length)
        candidate_starts = torch.arange(max_start + 1, device=device)
        if candidate_starts.numel() == 0:
            return 0

        overlap_mask = (
            (valid_chunks.unsqueeze(0) >= candidate_starts.unsqueeze(1))
            & (valid_chunks.unsqueeze(0) < (candidate_starts + window_length).unsqueeze(1))
        )
        valid_starts = candidate_starts[overlap_mask.any(dim=1)]
        if valid_starts.numel() == 0:
            return min(int(valid_chunks.min().item()), max_start)
        chosen_start = valid_starts[torch.randint(valid_starts.numel(), (1,), device=device)]
        return int(chosen_start.item())

    def _build_masked_agent_history_context_mask(
        self,
        data: HeteroData,
        masked_agent_selector: torch.Tensor,
        history_mode: Optional[str] = None,
    ) -> Tuple[Optional[torch.Tensor], Dict[str, torch.Tensor]]:
        base_history_mask = data['agent']['agent_valid_mask'].bool().clone()
        history_token_steps = max(1, (self.num_historical_steps - 1) // self.encoder.agent_encoder.shift)
        base_history_mask[:, history_token_steps:] = False
        masked_agents = (
            masked_agent_selector.any(dim=-1)
            if masked_agent_selector.dim() > 1
            else masked_agent_selector.bool()
        )
        history_mode = self.masked_agent_history_mode if history_mode is None else history_mode

        masked_agent_history_mask = base_history_mask & masked_agents.unsqueeze(-1)
        total_history_tokens = masked_agent_history_mask.sum()
        stats = {
            'masked_agent_history_visible_fraction': base_history_mask.new_tensor(0.0, dtype=torch.float32),
            'masked_agent_history_visible_tokens': total_history_tokens.to(torch.float32),
            'masked_agent_history_total_tokens': total_history_tokens.to(torch.float32),
        }

        if total_history_tokens.item() == 0:
            return None, stats

        if history_mode == 'visible':
            stats['masked_agent_history_visible_fraction'] = base_history_mask.new_tensor(1.0, dtype=torch.float32)
            return None, stats

        history_context_mask = base_history_mask.clone()
        masked_agent_indices = torch.nonzero(masked_agents, as_tuple=False).squeeze(-1)

        if history_mode == 'hidden':
            history_context_mask[masked_agent_indices] = False
        elif history_mode == 'partial_dropout':
            keep_ratio = max(0.0, min(1.0, 1.0 - self.masked_agent_history_dropout_ratio))
            for agent_index in masked_agent_indices.tolist():
                valid_steps = torch.nonzero(base_history_mask[agent_index], as_tuple=False).squeeze(-1)
                if valid_steps.numel() == 0:
                    continue
                keep_count = int(round(valid_steps.numel() * keep_ratio))
                keep_count = max(self.masked_agent_history_min_visible_tokens, keep_count)
                keep_count = min(valid_steps.numel(), keep_count)
                if keep_count >= valid_steps.numel():
                    continue
                keep_steps = valid_steps[torch.randperm(valid_steps.numel(), device=valid_steps.device)[:keep_count]]
                history_context_mask[agent_index] = False
                history_context_mask[agent_index, keep_steps] = True

        visible_history_tokens = (history_context_mask & masked_agents.unsqueeze(-1)).sum().to(torch.float32)
        stats['masked_agent_history_visible_tokens'] = visible_history_tokens
        stats['masked_agent_history_visible_fraction'] = visible_history_tokens / total_history_tokens.clamp_min(1).to(torch.float32)
        return history_context_mask, stats

    def _pool_map_polygon_features(
        self,
        pt_features: torch.Tensor,
        data: HeteroData,
        token_visible_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_polygons = int(data['map_polygon']['num_nodes'])
        token2pl = data[('pt_token', 'to', 'map_polygon')]['edge_index']
        token_indices = token2pl[0].long()
        polygon_indices = token2pl[1].long()
        polygon_features = pt_features.new_zeros((num_polygons, pt_features.shape[-1]))
        polygon_counts = pt_features.new_zeros((num_polygons,))
        visible_counts = pt_features.new_zeros((num_polygons,))
        if token_visible_mask is None:
            token_weights = torch.ones(token_indices.shape[0], device=pt_features.device, dtype=pt_features.dtype)
        else:
            token_weights = token_visible_mask[token_indices].to(device=pt_features.device, dtype=pt_features.dtype)
        polygon_features.index_add_(0, polygon_indices, pt_features[token_indices] * token_weights.unsqueeze(-1))
        polygon_counts.index_add_(0, polygon_indices, torch.ones(polygon_indices.shape[0], device=pt_features.device))
        visible_counts.index_add_(0, polygon_indices, token_weights)
        polygon_valid_mask = polygon_counts > 0
        polygon_features = polygon_features / visible_counts.clamp_min(1).unsqueeze(-1)
        polygon_graph_index = self._get_node_batch(data, 'map_polygon', num_nodes=num_polygons)
        return polygon_features, polygon_valid_mask, polygon_graph_index

    def _compute_polygon_centers(
        self,
        data: HeteroData,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_polygons = int(data['map_polygon']['num_nodes'])
        token2pl = data[('pt_token', 'to', 'map_polygon')]['edge_index']
        token_indices = token2pl[0].long()
        polygon_indices = token2pl[1].long()
        pt_position = data['pt_token']['position'][:, :2].float()
        polygon_centers = pt_position.new_zeros((num_polygons, 2))
        polygon_counts = pt_position.new_zeros((num_polygons,))
        polygon_centers.index_add_(0, polygon_indices, pt_position[token_indices])
        polygon_counts.index_add_(0, polygon_indices, torch.ones(polygon_indices.shape[0], device=pt_position.device))
        polygon_valid_mask = polygon_counts > 0
        polygon_centers = polygon_centers / polygon_counts.clamp_min(1).unsqueeze(-1)
        polygon_graph_index = self._get_node_batch(data, 'map_polygon', num_nodes=num_polygons)
        return polygon_centers, polygon_valid_mask, polygon_graph_index

    def _compute_graph_centers(
        self,
        positions: torch.Tensor,
        graph_index: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_graphs = int(graph_index.max().item()) + 1 if graph_index.numel() > 0 else 1
        graph_centers = positions.new_zeros((num_graphs, positions.shape[-1]))
        graph_counts = positions.new_zeros((num_graphs,))

        if graph_index.numel() == 0:
            return graph_centers

        if valid_mask.any():
            valid_graph_index = graph_index[valid_mask]
            graph_centers.index_add_(0, valid_graph_index, positions[valid_mask])
            graph_counts.index_add_(0, valid_graph_index, torch.ones(valid_graph_index.shape[0], device=positions.device))

        missing_graphs = graph_counts == 0
        if missing_graphs.any():
            fallback_centers = positions.new_zeros((num_graphs, positions.shape[-1]))
            fallback_counts = positions.new_zeros((num_graphs,))
            fallback_centers.index_add_(0, graph_index, positions)
            fallback_counts.index_add_(0, graph_index, torch.ones(graph_index.shape[0], device=positions.device))
            graph_centers[missing_graphs] = fallback_centers[missing_graphs] / fallback_counts[missing_graphs].clamp_min(1).unsqueeze(-1)
            graph_counts[missing_graphs] = fallback_counts[missing_graphs]

        return graph_centers / graph_counts.clamp_min(1).unsqueeze(-1)

    def _get_node_batch(
        self,
        data: HeteroData,
        node_type: str,
        num_nodes: Optional[int] = None,
    ) -> torch.Tensor:
        if isinstance(data, Batch) and 'batch' in data[node_type]:
            return data[node_type]['batch']
        if num_nodes is None:
            num_nodes = int(data[node_type]['num_nodes'])
        return torch.zeros(num_nodes, dtype=torch.long, device=data['agent']['position'].device)

    def on_before_zero_grad(self, optimizer) -> None:
        for target_param, online_param in zip(
            self.target_map_backbone.parameters(),
            self.encoder.map_encoder.parameters(),
        ):
            target_param.data.mul_(self.jepa_ema_decay).add_(online_param.data, alpha=1.0 - self.jepa_ema_decay)
        self.jepa.update_target_encoder(self.jepa_ema_decay)
