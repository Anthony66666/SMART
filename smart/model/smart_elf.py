import contextlib
import math
import os
import pickle
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.data import Batch, HeteroData

from smart.metrics import TokenCls, minADE, minFDE
from smart.metrics.joint_consistency import ConflictRate, InteractionConsistency
from smart.modules import SMARTDecoder
from smart.modules.elf_decoder import EmbeddedLanguageFlowDecoder
from smart.utils.torch_compat import torch_load_compat


class SMARTEmbeddedLanguageFlow(pl.LightningModule):
    """Standalone SMART-token embedded language flow predictor.

    This class intentionally does not inherit SMART, SMARTDiffusion,
    SMARTAutoregressiveDiffusion, or SMARTCausalDiffusion. It keeps the SMART
    data/token interface and composes the original map/history encoder, then
    applies an official-ELF-style embedding flow model over short receding
    future-token windows. Each inference round commits a small prefix, rolls the
    committed anchor into the history context, and re-encodes map/history
    features before sampling the next window.
    """

    def __init__(self, model_config) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model_config = model_config
        self.warmup_steps = int(model_config.warmup_steps)
        self.lr = float(model_config.lr)
        self.total_steps = int(model_config.total_steps)
        self.dataset = model_config.dataset
        self.input_dim = int(model_config.input_dim)
        self.hidden_dim = int(model_config.hidden_dim)
        self.output_dim = int(model_config.output_dim)
        self.output_head = model_config.output_head
        self.num_historical_steps = int(model_config.num_historical_steps)
        self.num_future_steps = int(model_config.decoder.num_future_steps)
        self.num_freq_bands = int(model_config.num_freq_bands)
        self.inference_token = bool(getattr(model_config, "inference_token", True))
        self.rollout_num = int(getattr(model_config, "rollout_num", 1))
        self.vis_map = False
        self.noise = True

        diffusion_cfg = getattr(model_config, "diffusion", None)
        if diffusion_cfg is None:
            raise ValueError("SMARTEmbeddedLanguageFlow requires Model.diffusion config.")
        self.elf_objective = str(
            getattr(diffusion_cfg, "elf_objective", "embedded_language_flow_v1")
        ).lower()
        if self.elf_objective not in (
            "elf",
            "embedded_language_flow",
            "embedded_language_flow_v1",
        ):
            raise ValueError("diffusion.elf_objective must be embedded_language_flow_v1.")
        self.elf_objective = "embedded_language_flow_v1"

        self.future_chunk_steps = int(
            getattr(diffusion_cfg, "future_chunk_steps", getattr(diffusion_cfg, "token_steps", 5))
        )
        self.history_tokens = int(
            getattr(
                diffusion_cfg,
                "history_tokens",
                max(1, (self.num_historical_steps - 1) // self.future_chunk_steps),
            )
        )
        default_sequence_tokens = max(1, self.num_future_steps // self.future_chunk_steps)
        self.elf_sequence_tokens = int(
            getattr(diffusion_cfg, "elf_sequence_tokens", default_sequence_tokens)
        )
        self.elf_window_tokens = int(
            getattr(
                diffusion_cfg,
                "elf_window_tokens",
                min(4, self.elf_sequence_tokens),
            )
        )
        self.elf_commit_tokens = int(getattr(diffusion_cfg, "elf_commit_tokens", 1))
        self.elf_receding_horizon = bool(
            getattr(diffusion_cfg, "elf_receding_horizon", True)
        )
        if self.elf_sequence_tokens <= 0:
            raise ValueError("diffusion.elf_sequence_tokens must be positive.")
        if self.elf_window_tokens <= 0:
            raise ValueError("diffusion.elf_window_tokens must be positive.")
        if self.elf_commit_tokens <= 0:
            raise ValueError("diffusion.elf_commit_tokens must be positive.")
        self.elf_window_tokens = min(self.elf_window_tokens, self.elf_sequence_tokens)
        self.elf_commit_tokens = min(self.elf_commit_tokens, self.elf_window_tokens)
        self.elf_loss_weight = max(0.0, float(getattr(diffusion_cfg, "elf_loss_weight", 1.0)))
        self.elf_tail_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, "elf_tail_loss_weight", 0.25)),
        )
        self.elf_decoder_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, "elf_decoder_loss_weight", 0.1)),
        )
        self.elf_decoder_prob = min(
            1.0,
            max(0.0, float(getattr(diffusion_cfg, "elf_decoder_prob", 0.25))),
        )
        self.elf_integration_steps = max(
            1,
            int(getattr(diffusion_cfg, "elf_integration_steps", 4)),
        )
        self.elf_noise_scale = max(
            0.0,
            float(getattr(diffusion_cfg, "elf_noise_scale", 1.0)),
        )
        self.elf_detach_targets = bool(getattr(diffusion_cfg, "elf_detach_targets", True))
        self.elf_sampling_strategy = str(
            getattr(diffusion_cfg, "elf_sampling_strategy", "argmax")
        ).lower()
        if self.elf_sampling_strategy not in ("argmax", "multinomial"):
            raise ValueError("diffusion.elf_sampling_strategy must be argmax or multinomial.")
        self.elf_map_commit_score_weight = max(
            0.0,
            float(getattr(diffusion_cfg, "elf_map_commit_score_weight", 0.0)),
        )
        self.elf_map_commit_loss_weight = max(
            0.0,
            float(getattr(diffusion_cfg, "elf_map_commit_loss_weight", 0.0)),
        )
        self.elf_geometry_energy_weight = max(
            0.0,
            float(getattr(diffusion_cfg, "elf_geometry_energy_weight", 0.0)),
        )
        self.elf_geometry_topk = max(0, int(getattr(diffusion_cfg, "elf_geometry_topk", 16)))
        self.elf_map_score_temperature = max(
            1e-4,
            float(getattr(diffusion_cfg, "elf_map_score_temperature", 1.0)),
        )
        self.elf_map_score_normalize = bool(getattr(diffusion_cfg, "elf_map_score_normalize", True))
        self.elf_time_schedule = str(getattr(diffusion_cfg, "elf_time_schedule", "uniform"))
        self.elf_denoiser_p_mean = float(getattr(diffusion_cfg, "elf_denoiser_p_mean", -0.8))
        self.elf_denoiser_p_std = float(getattr(diffusion_cfg, "elf_denoiser_p_std", 0.8))
        self.min_t = float(getattr(diffusion_cfg, "min_t", getattr(diffusion_cfg, "eps", 1e-3)))
        self.t_eps = float(getattr(diffusion_cfg, "t_eps", self.min_t))
        self.remask_confidence_temperature = max(
            1e-4,
            float(getattr(diffusion_cfg, "remask_confidence_temperature", 1.0)),
        )
        self.target_category_only = bool(getattr(diffusion_cfg, "target_category_only", False))
        self.supervision_mode = str(getattr(diffusion_cfg, "supervision_mode", "all_agents"))
        self.metric_mode = str(getattr(diffusion_cfg, "metric_mode", "smart_val_compatible"))
        self.encoder_lr_scale = max(0.0, float(getattr(diffusion_cfg, "encoder_lr_scale", 1.0)))
        self.ntp_aux_loss_weight = 0.0

        self.token_size = int(getattr(model_config.decoder, "token_size", 2048))
        self.mask_token_id = self.token_size
        module_dir = os.path.dirname(os.path.dirname(__file__))
        self.map_token_traj_path = os.path.join(module_dir, "tokens/map_traj_token5.pkl")
        self.token_path = os.path.join(module_dir, "tokens/cluster_frame_5_2048.pkl")
        self.init_map_token()
        token_data = self.get_trajectory_token()
        self.encoder = SMARTDecoder(
            dataset=model_config.dataset,
            input_dim=model_config.input_dim,
            hidden_dim=model_config.hidden_dim,
            num_historical_steps=model_config.num_historical_steps,
            num_freq_bands=model_config.num_freq_bands,
            num_heads=model_config.num_heads,
            head_dim=model_config.head_dim,
            dropout=model_config.dropout,
            num_map_layers=model_config.decoder.num_map_layers,
            num_agent_layers=model_config.decoder.num_agent_layers,
            pl2pl_radius=model_config.decoder.pl2pl_radius,
            pl2a_radius=model_config.decoder.pl2a_radius,
            a2a_radius=model_config.decoder.a2a_radius,
            time_span=model_config.decoder.time_span,
            map_token={"traj_src": self.map_token["traj_src"]},
            token_data=token_data,
            token_size=self.token_size,
        )

        self.diffusion_decoder = EmbeddedLanguageFlowDecoder(
            text_encoder_dim=self.hidden_dim,
            max_length=self.elf_sequence_tokens,
            hidden_size=self.hidden_dim,
            depth=max(1, int(getattr(diffusion_cfg, "elf_num_layers", getattr(diffusion_cfg, "num_layers", 6)))),
            num_heads=int(model_config.num_heads),
            vocab_size=self.token_size,
            attn_drop=float(getattr(model_config, "dropout", 0.0)),
            proj_drop=float(getattr(model_config, "dropout", 0.0)),
            bottleneck_dim=int(getattr(diffusion_cfg, "elf_bottleneck_dim", self.hidden_dim)),
            num_time_tokens=int(getattr(diffusion_cfg, "elf_num_time_tokens", 4)),
            num_model_mode_tokens=1,
            num_token_types=4,
        )
        self.elf_map_score_proj = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.minADE = minADE(max_guesses=1)
        self.minFDE = minFDE(max_guesses=1)
        self.TokenCls = TokenCls(max_guesses=1)
        self.conflict_rate = ConflictRate()
        self.interaction_consistency = InteractionConsistency()
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.test_predictions = {}

    def get_trajectory_token(self):
        token_data = pickle.load(open(self.token_path, "rb"))
        self.trajectory_token = token_data["token"]
        self.trajectory_token_traj = token_data["traj"]
        self.trajectory_token_all = token_data["token_all"]
        return token_data

    def init_map_token(self):
        self.argmin_sample_len = 3
        map_token_traj = pickle.load(open(self.map_token_traj_path, "rb"))
        self.map_token = {"traj_src": map_token_traj["traj_src"]}
        traj_end_theta = np.arctan2(
            self.map_token["traj_src"][:, -1, 1] - self.map_token["traj_src"][:, -2, 1],
            self.map_token["traj_src"][:, -1, 0] - self.map_token["traj_src"][:, -2, 0],
        )
        indices = torch.linspace(
            0,
            self.map_token["traj_src"].shape[1] - 1,
            steps=self.argmin_sample_len,
        ).long()
        self.map_token["sample_pt"] = torch.from_numpy(
            self.map_token["traj_src"][:, indices]
        ).to(torch.float)
        self.map_token["traj_end_theta"] = torch.from_numpy(traj_end_theta).to(torch.float)
        self.map_token["traj_src"] = torch.from_numpy(self.map_token["traj_src"]).to(torch.float)

    def maybe_autocast(self, dtype=torch.float16):
        if self.device != torch.device("cpu"):
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def _prepared(self, data) -> bool:
        return bool(getattr(data, "_smart_elf_prepared", False))

    def _prepare_batch(self, data):
        if self._prepared(data):
            return data
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
        setattr(data, "_smart_elf_prepared", True)
        return data

    def match_token_map(self, data):
        traj_pos = data["map_save"]["traj_pos"].to(torch.float)
        traj_theta = data["map_save"]["traj_theta"].to(torch.float)
        pl_idx_list = data["map_save"]["pl_idx_list"]
        token_sample_pt = self.map_token["sample_pt"].to(traj_pos.device)
        token_src = self.map_token["traj_src"].to(traj_pos.device)
        max_traj_len = token_src.shape[1]
        pl_num = traj_pos.shape[0]

        pt_token_pos = traj_pos[:, 0, :].clone()
        pt_token_orientation = traj_theta.clone()
        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = -sin
        rot_mat[..., 1, 0] = sin
        rot_mat[..., 1, 1] = cos
        traj_pos_local = torch.bmm(
            traj_pos - traj_pos[:, 0:1],
            rot_mat.view(-1, 2, 2),
        )
        distance = torch.sum(
            (token_sample_pt[None] - traj_pos_local.unsqueeze(1)) ** 2,
            dim=(-2, -1),
        )
        pt_token_id = torch.argmin(distance, dim=1)

        if self.noise and self.training:
            topk_indices = torch.argsort(distance, dim=1)[:, :8]
            sample_topk = torch.randint(
                0,
                topk_indices.shape[-1],
                size=(topk_indices.shape[0], 1),
                device=topk_indices.device,
            )
            pt_token_id = torch.gather(topk_indices, 1, sample_topk).squeeze(-1)

        cos, sin = traj_theta.cos(), traj_theta.sin()
        rot_mat = traj_theta.new_zeros(pl_num, 2, 2)
        rot_mat[..., 0, 0] = cos
        rot_mat[..., 0, 1] = sin
        rot_mat[..., 1, 0] = -sin
        rot_mat[..., 1, 1] = cos
        token_src_world = torch.bmm(
            token_src[None, ...].repeat(pl_num, 1, 1, 1).reshape(pl_num, -1, 2),
            rot_mat.view(-1, 2, 2),
        ).reshape(pl_num, token_src.shape[0], max_traj_len, 2)
        token_src_world = token_src_world + traj_pos[:, None, [0], :]
        token_src_world_select = token_src_world.view(
            -1,
            token_src.shape[0],
            max_traj_len,
            2,
        )[torch.arange(pt_token_id.view(-1).shape[0], device=traj_pos.device), pt_token_id.view(-1)]

        pl_idx_full = pl_idx_list.clone()
        token2pl = torch.stack(
            [torch.arange(len(pl_idx_list), device=traj_pos.device), pl_idx_full.long()]
        )
        del token_src_world_select
        count_nums = []
        for pl in pl_idx_full.unique():
            pt = token2pl[0, token2pl[1, :] == pl]
            left_side = (data["pt_token"]["side"][pt] == 0).sum()
            right_side = (data["pt_token"]["side"][pt] == 1).sum()
            center_side = (data["pt_token"]["side"][pt] == 2).sum()
            count_nums.append(torch.stack([left_side, right_side, center_side]))
        count_nums = torch.stack(count_nums, dim=0)
        num_polyline = int(count_nums.max().item())
        traj_mask = torch.zeros(
            (int(len(pl_idx_full.unique())), 3, num_polyline),
            dtype=torch.bool,
            device=traj_pos.device,
        )
        idx_matrix = torch.arange(num_polyline, device=traj_pos.device).view(1, 1, -1)
        mask_update = idx_matrix.expand_as(traj_mask) < count_nums.unsqueeze(-1)
        traj_mask[mask_update] = True

        data["pt_token"]["traj_mask"] = traj_mask
        data["pt_token"]["position"] = torch.cat(
            [
                pt_token_pos,
                torch.zeros(
                    (data["pt_token"]["num_nodes"], 1),
                    device=traj_pos.device,
                    dtype=torch.float,
                ),
            ],
            dim=-1,
        )
        data["pt_token"]["orientation"] = pt_token_orientation
        data["pt_token"]["height"] = data["pt_token"]["position"][:, -1]
        data[("pt_token", "to", "map_polygon")] = {}
        data[("pt_token", "to", "map_polygon")]["edge_index"] = token2pl
        data["pt_token"]["token_idx"] = pt_token_id
        return data

    def sample_pt_pred(self, data):
        traj_mask = data["pt_token"]["traj_mask"]
        device = traj_mask.device
        raw_pt_index = torch.arange(1, traj_mask.shape[2], device=device).repeat(
            traj_mask.shape[0],
            traj_mask.shape[1],
            1,
        )
        keep_count = max(1, (traj_mask.shape[2] - 1) // 3)
        masked_pt_index = raw_pt_index.view(-1)[
            torch.randperm(raw_pt_index.numel(), device=device)[
                : traj_mask.shape[0] * traj_mask.shape[1] * keep_count
            ]
        ].reshape(traj_mask.shape[0], traj_mask.shape[1], keep_count)
        masked_pt_index = torch.sort(masked_pt_index, -1)[0]
        pt_valid_mask = traj_mask.clone()
        pt_valid_mask.scatter_(2, masked_pt_index, False)
        pt_pred_mask = traj_mask.clone()
        pt_pred_mask.scatter_(2, masked_pt_index, False)
        tmp_mask = pt_pred_mask.clone()
        tmp_mask[:, :, :] = True
        tmp_mask.scatter_(2, masked_pt_index - 1, False)
        pt_pred_mask.masked_fill_(tmp_mask, False)
        pt_pred_mask = pt_pred_mask * torch.roll(traj_mask, shifts=-1, dims=2)
        pt_target_mask = torch.roll(pt_pred_mask, shifts=1, dims=2)

        data["pt_token"]["pt_valid_mask"] = pt_valid_mask[traj_mask]
        data["pt_token"]["pt_pred_mask"] = pt_pred_mask[traj_mask]
        data["pt_token"]["pt_target_mask"] = pt_target_mask[traj_mask]
        return data

    def _sample_elf_times(self, valid_mask, summary):
        batch_size = int(valid_mask.shape[0])
        if not self.training:
            return torch.full(
                (batch_size,),
                0.5,
                dtype=summary.dtype,
                device=summary.device,
            )
        if self.elf_time_schedule == "logit_normal":
            z = torch.randn(
                (batch_size,),
                dtype=summary.dtype,
                device=summary.device,
            )
            z = z * self.elf_denoiser_p_std + self.elf_denoiser_p_mean
            return torch.sigmoid(z).clamp(self.min_t, 1.0)
        if self.elf_time_schedule == "uniform":
            return torch.rand(
                batch_size,
                dtype=summary.dtype,
                device=summary.device,
            ).clamp(self.min_t, 1.0)
        raise ValueError(f"Unsupported diffusion.elf_time_schedule: {self.elf_time_schedule}")

    def _token_embedding_tables(self, device, dtype):
        agent_encoder = self.encoder.agent_encoder
        specs = (
            ("veh", 0, agent_encoder.token_emb_veh),
            ("ped", 1, agent_encoder.token_emb_ped),
            ("cyc", 2, agent_encoder.token_emb_cyc),
        )
        tables = []
        for token_name, type_id, token_embedder in specs:
            token_template = torch.from_numpy(
                agent_encoder.trajectory_token[token_name]
            ).to(device=device, dtype=torch.float)
            table = token_embedder(token_template.reshape(token_template.shape[0], -1))
            tables.append((type_id, table.to(dtype=dtype)))
        return tables

    def _physical_token_embeddings(self, token_ids, agent_type_ids, valid_mask=None):
        token_ids = token_ids.to(dtype=torch.long).clamp(min=0, max=self.token_size - 1)
        out = torch.zeros(
            *token_ids.shape,
            self.hidden_dim,
            device=token_ids.device,
            dtype=self.diffusion_decoder.t_emb_tokens.dtype,
        )
        for type_id, table in self._token_embedding_tables(token_ids.device, out.dtype):
            type_mask = agent_type_ids.to(device=token_ids.device) == type_id
            if valid_mask is not None:
                type_mask = type_mask & valid_mask.bool()
            if type_mask.any():
                out[type_mask] = table[token_ids[type_mask]]
        if valid_mask is not None:
            out = out * valid_mask.unsqueeze(-1).to(dtype=out.dtype)
        return out

    def _elf_target_embeddings(self, token_ids, agent_type_ids, valid_mask=None):
        target = self._physical_token_embeddings(token_ids, agent_type_ids, valid_mask)
        if self.elf_detach_targets:
            target = target.detach()
        return target

    def _elf_source_embeddings(
        self,
        target_embeddings,
        valid_mask,
        proposal_token_ids=None,
        proposal_confidence=None,
        agent_type_ids=None,
    ):
        del proposal_token_ids, proposal_confidence, agent_type_ids
        if self.elf_noise_scale > 0.0:
            source = torch.randn_like(target_embeddings) * self.elf_noise_scale
        else:
            source = torch.zeros_like(target_embeddings)
        return source * valid_mask.unsqueeze(-1).to(dtype=source.dtype)

    def _elf_interpolate(self, source_embeddings, target_embeddings, t):
        t = t.to(device=source_embeddings.device, dtype=source_embeddings.dtype)
        while t.dim() < source_embeddings.dim():
            t = t.unsqueeze(-1)
        velocity = target_embeddings - source_embeddings
        state = source_embeddings + t * velocity
        return state, velocity

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

    def _elf_token_similarity_logits(self, elf_embeddings, agent_type_ids, valid_mask):
        fill_value = torch.finfo(elf_embeddings.dtype).min
        logits = elf_embeddings.new_full(
            (*valid_mask.shape, self.token_size),
            fill_value,
        )
        if not valid_mask.any():
            return logits
        for type_id, table in self._token_embedding_tables(
            elf_embeddings.device,
            elf_embeddings.dtype,
        ):
            type_mask = valid_mask & (agent_type_ids.to(device=valid_mask.device) == type_id)
            if not type_mask.any():
                continue
            emb = elf_embeddings[type_mask]
            type_logits = 2.0 * emb.matmul(table.t()) - table.pow(2).sum(dim=-1).unsqueeze(0)
            logits[type_mask] = type_logits.to(dtype=logits.dtype)
        return logits

    def _map_conditioned_token_logits(self, packed, valid_mask):
        context = packed.get("context")
        if context is None:
            return torch.zeros(
                *valid_mask.shape,
                self.token_size,
                dtype=torch.float32,
                device=valid_mask.device,
            )
        query = context.to(device=valid_mask.device, dtype=torch.float32)
        if hasattr(self, "elf_map_score_proj"):
            query = self.elf_map_score_proj(query)
        logits = query.new_zeros((*valid_mask.shape, self.token_size))
        if not valid_mask.any():
            return logits
        agent_type_ids = packed["agent_type_ids"].to(device=valid_mask.device)
        for type_id, table in self._token_embedding_tables(valid_mask.device, query.dtype):
            type_mask = valid_mask & (agent_type_ids == type_id)
            if not type_mask.any():
                continue
            token_table = table.to(device=query.device, dtype=query.dtype)
            type_query = query[type_mask]
            if getattr(self, "elf_map_score_normalize", True):
                type_query = F.normalize(type_query, dim=-1)
                token_table = F.normalize(token_table, dim=-1)
                scale = self.elf_map_score_temperature
            else:
                scale = math.sqrt(float(max(type_query.shape[-1], 1))) * self.elf_map_score_temperature
            logits[type_mask] = type_query.matmul(token_table.t()) / scale
        return logits

    def _combined_elf_token_logits(self, elf_embeddings, packed, valid_mask):
        logits = self._elf_token_similarity_logits(
            elf_embeddings,
            packed["agent_type_ids"],
            valid_mask,
        )
        score_weight = float(getattr(self, "elf_map_commit_score_weight", 0.0))
        if score_weight > 0.0:
            map_logits = self._map_conditioned_token_logits(packed, valid_mask).to(
                device=logits.device,
                dtype=logits.dtype,
            )
            logits = logits + score_weight * map_logits
        return self._apply_map_geometry_energy(logits, packed, valid_mask)

    def _apply_map_geometry_energy(self, logits, packed, valid_mask):
        energy_weight = float(getattr(self, "elf_geometry_energy_weight", 0.0))
        topk = int(getattr(self, "elf_geometry_topk", 0))
        if energy_weight <= 0.0 or topk <= 0 or not valid_mask.any():
            return logits
        required = ("anchor_pos", "anchor_heading", "map_positions", "map_batch", "slot_batch")
        if any(key not in packed for key in required):
            return logits
        k = min(topk, int(logits.shape[-1]))
        top_scores, top_ids = torch.topk(logits, k=k, dim=-1)
        energy = self._token_map_geometry_energy(top_ids, packed, valid_mask).to(
            device=logits.device,
            dtype=logits.dtype,
        )
        adjusted = logits.new_full(logits.shape, torch.finfo(logits.dtype).min)
        adjusted.scatter_(-1, top_ids, top_scores - energy_weight * energy)
        return adjusted

    def _token_map_geometry_energy(self, token_ids, packed, valid_mask):
        energy = token_ids.new_zeros(token_ids.shape, dtype=torch.float32)
        map_positions = packed["map_positions"].to(device=token_ids.device, dtype=torch.float32)
        map_batch = packed["map_batch"].to(device=token_ids.device, dtype=torch.long)
        map_valid_mask = packed.get("map_valid_mask")
        if map_valid_mask is not None:
            map_valid_mask = map_valid_mask.to(device=token_ids.device, dtype=torch.bool)
        batch_indices = torch.nonzero(valid_mask, as_tuple=False)
        if batch_indices.numel() == 0:
            return energy
        for batch_idx, slot_idx in batch_indices.tolist():
            scene_idx = int(packed["slot_batch"][batch_idx, slot_idx].item())
            scene_map = map_batch == scene_idx
            if map_valid_mask is not None:
                scene_map = scene_map & map_valid_mask
            if not scene_map.any():
                continue
            ids = token_ids[batch_idx, slot_idx].view(1, -1)
            agent_type = packed["agent_type_ids"][batch_idx, slot_idx].view(1)
            local = self._select_token_traj_all(ids, agent_type)[0]
            theta = packed["anchor_heading"][batch_idx, slot_idx].to(
                device=token_ids.device,
                dtype=torch.float32,
            )
            cos, sin = theta.cos(), theta.sin()
            rot = torch.stack(
                [
                    torch.stack([cos, sin]),
                    torch.stack([-sin, cos]),
                ],
                dim=0,
            )
            world = torch.matmul(local, rot)
            world = world + packed["anchor_pos"][batch_idx, slot_idx].to(
                device=token_ids.device,
                dtype=torch.float32,
            ).view(1, 1, 1, 2)
            centers = world[:, 1:1 + self.future_chunk_steps].mean(dim=2)
            dist = torch.cdist(
                centers.reshape(-1, 2),
                map_positions[scene_map],
            )
            nearest = dist.min(dim=-1).values.view(centers.shape[0], -1)
            energy[batch_idx, slot_idx] = nearest.mean(dim=-1)
        return energy

    def _future_token_slices(self, data, token_start=None, sequence_tokens=None):
        token_start = self.history_tokens if token_start is None else int(token_start)
        sequence_tokens = (
            self.elf_sequence_tokens if sequence_tokens is None else int(sequence_tokens)
        )
        token_end = token_start + sequence_tokens
        token_ids = data["agent"]["token_idx"][:, token_start:token_end].long()
        token_valid = data["agent"]["agent_valid_mask"][:, token_start:token_end].bool()
        if token_ids.shape[1] < sequence_tokens:
            pad_len = sequence_tokens - token_ids.shape[1]
            token_ids = F.pad(token_ids, (0, pad_len), value=0)
            token_valid = F.pad(token_valid, (0, pad_len), value=False)
        generation_agents = self._elf_generation_agent_mask(data).to(device=token_valid.device)
        token_valid = token_valid & generation_agents.unsqueeze(-1)
        return token_ids, token_valid

    def _agent_batch(self, data, num_agents, device):
        if isinstance(data, Batch):
            return data["agent"]["batch"].to(device=device, dtype=torch.long)
        return torch.zeros(num_agents, dtype=torch.long, device=device)

    def _pack_future_window(self, data, context, *, token_start=None, sequence_tokens=None):
        token_ids, token_valid = self._future_token_slices(
            data,
            token_start=token_start,
            sequence_tokens=sequence_tokens,
        )
        token_start = self.history_tokens if token_start is None else int(token_start)
        device = token_ids.device
        num_agents, sequence_tokens = token_ids.shape
        agent_batch = self._agent_batch(data, num_agents, device)
        batch_size = int(agent_batch.max().item()) + 1 if num_agents > 0 else 1
        agent_type = data["agent"]["type"].long().to(device)
        chunk_ids_agent = torch.arange(sequence_tokens, device=device).view(1, -1).expand(
            num_agents,
            -1,
        )
        history_index = min(
            max(self.history_tokens - 1, 0),
            context["x_a_history"].shape[1] - 1,
        )
        agent_context = context["x_a_history"][:, history_index].to(device=device)
        token_context = agent_context[:, None, :].expand(-1, sequence_tokens, -1)
        loss_agent_mask = self._elf_supervision_agent_mask(data).to(device=device)
        loss_mask_agent = token_valid & loss_agent_mask.unsqueeze(-1)

        lengths = []
        for batch_idx in range(batch_size):
            lengths.append(int((agent_batch == batch_idx).sum().item()) * sequence_tokens)
        max_len = max(max(lengths), 1)

        def zeros(shape, dtype):
            return torch.zeros(shape, dtype=dtype, device=device)

        packed_ids = zeros((batch_size, max_len), torch.long)
        packed_valid = zeros((batch_size, max_len), torch.bool)
        packed_loss = zeros((batch_size, max_len), torch.bool)
        packed_chunk_ids = zeros((batch_size, max_len), torch.long)
        packed_type_ids = zeros((batch_size, max_len), torch.long)
        packed_context = zeros((batch_size, max_len, self.hidden_dim), agent_context.dtype)
        packed_anchor_pos = zeros((batch_size, max_len, 2), torch.float32)
        packed_anchor_heading = zeros((batch_size, max_len), torch.float32)
        packed_slot_batch = zeros((batch_size, max_len), torch.long)
        packed_agent_indices = torch.full(
            (batch_size, max_len),
            -1,
            dtype=torch.long,
            device=device,
        )
        anchor_pos = data["agent"]["token_pos"][:, history_index, :2].to(
            device=device,
            dtype=torch.float32,
        )
        anchor_heading = data["agent"]["token_heading"][:, history_index].to(
            device=device,
            dtype=torch.float32,
        )
        for batch_idx in range(batch_size):
            agents = torch.nonzero(agent_batch == batch_idx, as_tuple=False).squeeze(-1)
            if agents.numel() == 0:
                continue
            length = agents.numel() * sequence_tokens
            sl = slice(0, length)
            packed_ids[batch_idx, sl] = token_ids[agents].reshape(-1)
            packed_valid[batch_idx, sl] = token_valid[agents].reshape(-1)
            packed_loss[batch_idx, sl] = loss_mask_agent[agents].reshape(-1)
            packed_chunk_ids[batch_idx, sl] = chunk_ids_agent[agents].reshape(-1)
            packed_type_ids[batch_idx, sl] = agent_type[agents, None].expand(-1, sequence_tokens).reshape(-1)
            packed_context[batch_idx, sl] = token_context[agents].reshape(-1, self.hidden_dim)
            packed_anchor_pos[batch_idx, sl] = anchor_pos[agents, None].expand(-1, sequence_tokens, -1).reshape(-1, 2)
            packed_anchor_heading[batch_idx, sl] = anchor_heading[agents, None].expand(-1, sequence_tokens).reshape(-1)
            packed_slot_batch[batch_idx, sl] = batch_idx
            packed_agent_indices[batch_idx, sl] = agents[:, None].expand(-1, sequence_tokens).reshape(-1)

        summary = packed_context.new_zeros(batch_size, self.hidden_dim)
        counts = packed_valid.float().sum(dim=1).clamp_min(1.0)
        summary = (packed_context * packed_valid.unsqueeze(-1).to(packed_context.dtype)).sum(dim=1)
        summary = summary / counts.unsqueeze(-1)
        packed = {
            "token_ids": packed_ids,
            "valid_mask": packed_valid,
            "loss_mask_base": packed_loss,
            "chunk_ids": packed_chunk_ids,
            "agent_type_ids": packed_type_ids,
            "context": packed_context,
            "anchor_pos": packed_anchor_pos,
            "anchor_heading": packed_anchor_heading,
            "slot_batch": packed_slot_batch,
            "agent_indices": packed_agent_indices,
            "sequence_tokens": sequence_tokens,
            "token_start": token_start,
            "num_agents": num_agents,
            "agent_batch": agent_batch,
            "token_ids_by_agent": token_ids,
            "token_valid_by_agent": token_valid,
        }
        if "pt_token" in data and "position" in data["pt_token"]:
            packed["map_positions"] = data["pt_token"]["position"][:, :2].to(
                device=device,
                dtype=torch.float32,
            )
            if isinstance(data, Batch) and "batch" in data["pt_token"]:
                packed["map_batch"] = data["pt_token"]["batch"].to(device=device, dtype=torch.long)
            else:
                packed["map_batch"] = torch.zeros(
                    int(data["pt_token"]["position"].shape[0]),
                    dtype=torch.long,
                    device=device,
                )
            if "pt_visibility_mask" in context:
                packed["map_valid_mask"] = context["pt_visibility_mask"].to(
                    device=device,
                    dtype=torch.bool,
                )
        return packed, summary

    def _build_diffusion_inputs(
        self,
        data,
        rollout_valid=False,
        *,
        token_start=None,
        sequence_tokens=None,
    ):
        del rollout_valid
        context = self.encoder.encode_history_context(
            data,
            map_agent_mask=self._elf_generation_agent_mask(data),
        )
        if token_start is None and sequence_tokens is None:
            packed, summary = self._pack_future_window(data, context)
        else:
            packed, summary = self._pack_future_window(
                data,
                context,
                token_start=token_start,
                sequence_tokens=sequence_tokens,
            )
        return packed, summary, None, None, None, None, packed["agent_batch"]

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
        del proxy_token_ids, summary, geometry_known_mask, proposal_token_ids, proposal_confidence
        x_pred, decoder_logits = self.diffusion_decoder(
            elf_embeddings,
            t,
            attention_mask=packed["valid_mask"],
            deterministic=not self.training,
            decoder_step_active=packed.get("decoder_step_active", True),
            context=packed.get("context"),
            chunk_ids=packed.get("chunk_ids"),
            agent_type_ids=packed.get("agent_type_ids"),
        )
        t_expand = t.to(dtype=elf_embeddings.dtype, device=elf_embeddings.device)
        while t_expand.dim() < elf_embeddings.dim():
            t_expand = t_expand.unsqueeze(-1)
        velocity = (x_pred.to(dtype=elf_embeddings.dtype) - elf_embeddings) / torch.clamp(
            1.0 - t_expand,
            min=self.t_eps,
        )
        return velocity, decoder_logits

    @staticmethod
    def _masked_branch_loss(values, mask, chunk_ids, tail_weight):
        if not mask.any():
            return values.sum() * 0.0
        commit_mask = mask & (chunk_ids == 0)
        tail_mask = mask & ~commit_mask
        loss = values.sum() * 0.0
        if commit_mask.any():
            loss = loss + values[commit_mask].mean()
        if tail_mask.any():
            loss = loss + tail_weight * values[tail_mask].mean()
        return loss

    def _compute_diffusion_loss(self, packed, summary):
        gt = packed["token_ids"]
        valid_mask = packed["valid_mask"]
        loss_mask = packed.get("loss_mask_base", valid_mask) & valid_mask
        target = self._elf_target_embeddings(gt, packed["agent_type_ids"], valid_mask).to(
            device=summary.device,
            dtype=summary.dtype,
        )
        source = self._elf_source_embeddings(target, valid_mask).to(
            device=summary.device,
            dtype=summary.dtype,
        )
        t = self._sample_elf_times(valid_mask, summary).clamp(self.min_t, 1.0)
        elf_embeddings, target_velocity = self._elf_interpolate(source, target, t)

        decoder_active = None
        if self.training and 0.0 < self.elf_decoder_prob < 1.0:
            decoder_active = torch.bernoulli(
                torch.full(
                    (valid_mask.shape[0],),
                    self.elf_decoder_prob,
                    dtype=summary.dtype,
                    device=summary.device,
                )
            )
        elif self.training:
            decoder_active = torch.full(
                (valid_mask.shape[0],),
                1.0 if self.elf_decoder_prob >= 1.0 else 0.0,
                dtype=summary.dtype,
                device=summary.device,
            )
        packed["decoder_step_active"] = (
            torch.ones((valid_mask.shape[0],), dtype=summary.dtype, device=summary.device)
            if decoder_active is None
            else decoder_active
        )
        proxy_ids = self._elf_proxy_token_ids(
            elf_embeddings,
            packed["agent_type_ids"],
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
        if decoder_logits is None:
            decoder_logits = velocity.new_zeros(*gt.shape, self.token_size)

        if loss_mask.any():
            token_flow_loss = (velocity - target_velocity).pow(2).mean(dim=-1)
            token_decoder_loss = decoder_logits.new_zeros(gt.shape)
            token_decoder_loss[loss_mask] = F.cross_entropy(
                decoder_logits[loss_mask].to(torch.float32),
                gt[loss_mask],
                reduction="none",
            ).to(dtype=decoder_logits.dtype)
            if decoder_active is not None:
                decoder_rows = decoder_active.bool().view(-1, 1)
                flow_mask = loss_mask & ~decoder_rows
                decoder_mask = loss_mask & decoder_rows
            else:
                flow_mask = loss_mask
                decoder_mask = loss_mask
            flow_loss = self._masked_branch_loss(
                token_flow_loss,
                flow_mask,
                packed["chunk_ids"],
                self.elf_tail_loss_weight,
            )
            decoder_loss = self._masked_branch_loss(
                token_decoder_loss,
                decoder_mask,
                packed["chunk_ids"],
                self.elf_tail_loss_weight,
            )
            map_commit_loss = decoder_logits.sum() * 0.0
            if float(getattr(self, "elf_map_commit_loss_weight", 0.0)) > 0.0:
                map_logits = self._map_conditioned_token_logits(packed, valid_mask).to(
                    device=decoder_logits.device,
                    dtype=decoder_logits.dtype,
                )
                token_map_loss = decoder_logits.new_zeros(gt.shape)
                token_map_loss[loss_mask] = F.cross_entropy(
                    map_logits[loss_mask].to(torch.float32),
                    gt[loss_mask],
                    reduction="none",
                ).to(dtype=decoder_logits.dtype)
                map_commit_loss = self._masked_branch_loss(
                    token_map_loss,
                    loss_mask,
                    packed["chunk_ids"],
                    self.elf_tail_loss_weight,
                )
            loss = (
                self.elf_loss_weight * flow_loss
                + self.elf_decoder_loss_weight * decoder_loss
                + float(getattr(self, "elf_map_commit_loss_weight", 0.0)) * map_commit_loss
            )
            acc = (decoder_logits[loss_mask].argmax(-1) == gt[loss_mask]).float().mean()
        else:
            flow_loss = velocity.sum() * 0.0
            decoder_loss = decoder_logits.sum() * 0.0
            map_commit_loss = decoder_logits.sum() * 0.0
            loss = flow_loss + decoder_loss + map_commit_loss
            acc = velocity.new_zeros(())

        if self.training:
            valid_count = valid_mask.float().sum().clamp_min(1.0)
            self.log(
                "train_elf_loss_frac",
                loss_mask.float().sum() / valid_count,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                "train_elf_flow_loss",
                flow_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            self.log(
                "train_elf_decoder_loss",
                decoder_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=1,
            )
            if float(getattr(self, "elf_map_commit_loss_weight", 0.0)) > 0.0:
                self.log(
                    "train_elf_map_commit_loss",
                    map_commit_loss,
                    prog_bar=False,
                    on_step=True,
                    on_epoch=True,
                    batch_size=1,
                )
            if decoder_active is not None:
                self.log(
                    "train_elf_decoder_branch_frac",
                    decoder_active.float().mean(),
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
        del initial_proposal_token_ids, initial_proposal_confidence
        steps = max(1, int(self.elf_integration_steps))
        decoder_logits = None
        packed["decoder_step_active"] = torch.ones(
            (elf_embeddings.shape[0],),
            dtype=summary.dtype,
            device=summary.device,
        )
        for flow_step in range(steps):
            t_value = max(flow_step / float(steps), self.min_t)
            t_next = (flow_step + 1) / float(steps)
            t_batch = torch.full(
                (elf_embeddings.shape[0],),
                t_value,
                dtype=summary.dtype,
                device=summary.device,
            )
            proxy_ids = sampled.clone()
            proxy = self._elf_proxy_token_ids(
                elf_embeddings,
                packed["agent_type_ids"],
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
            )
            dt = t_next - t_value
            elf_embeddings[editable_mask] = (
                elf_embeddings[editable_mask] + dt * velocity[editable_mask]
            )

        t_batch = torch.ones(
            (elf_embeddings.shape[0],),
            dtype=summary.dtype,
            device=summary.device,
        )
        proxy_ids = sampled.clone()
        proxy = self._elf_proxy_token_ids(
            elf_embeddings,
            packed["agent_type_ids"],
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
        )
        return elf_embeddings, decoder_logits

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
            initial_proposal_token_ids,
            initial_proposal_confidence,
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
            editable_mask = valid_mask.clone() if editable_mask is None else editable_mask & valid_mask
            sampled = torch.full(
                (batch_size, sequence_length),
                self.mask_token_id,
                dtype=torch.long,
                device=device,
            ).masked_fill(~valid_mask, 0)

        zero_target = torch.zeros(
            batch_size,
            sequence_length,
            self.hidden_dim,
            dtype=summary.dtype,
            device=device,
        )
        elf_embeddings = self._elf_source_embeddings(zero_target, valid_mask).to(
            device=device,
            dtype=summary.dtype,
        )
        locked = valid_mask & (sampled != self.mask_token_id)
        if locked.any():
            locked_embeddings = self._elf_target_embeddings(
                sampled.masked_fill(sampled == self.mask_token_id, 0),
                packed["agent_type_ids"],
                valid_mask,
            ).to(device=device, dtype=summary.dtype)
            elf_embeddings[locked] = locked_embeddings[locked]

        decoder_logits = None
        if editable_mask.any():
            elf_embeddings, decoder_logits = self._integrate_window_elf(
                elf_embeddings,
                sampled,
                editable_mask,
                valid_mask,
                packed,
                summary,
            )
            use_combined_scores = (
                float(getattr(self, "elf_map_commit_score_weight", 0.0)) > 0.0
                or float(getattr(self, "elf_geometry_energy_weight", 0.0)) > 0.0
            )
            if use_combined_scores:
                selection_logits = self._combined_elf_token_logits(
                    elf_embeddings,
                    packed,
                    valid_mask,
                )
                probabilities = F.softmax(
                    selection_logits / self.remask_confidence_temperature,
                    dim=-1,
                )
                editable_probabilities = probabilities[editable_mask].clamp_min(1e-10)
                if self.elf_sampling_strategy == "argmax":
                    sampled_ids = selection_logits[editable_mask].argmax(dim=-1)
                else:
                    sampled_ids = torch.multinomial(
                        editable_probabilities,
                        1,
                    ).squeeze(-1)
            else:
                if self.elf_sampling_strategy == "argmax":
                    sampled_ids = self._elf_proxy_token_ids(
                        elf_embeddings,
                        packed["agent_type_ids"],
                        valid_mask,
                    )[editable_mask]
                else:
                    probabilities = F.softmax(
                        decoder_logits / self.remask_confidence_temperature,
                        dim=-1,
                    )
                    sampled_ids = torch.multinomial(
                        probabilities[editable_mask].clamp_min(1e-10),
                        1,
                    ).squeeze(-1)
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
                "step": 0,
                "mode": "full_window",
                "sampled": int((valid_mask & editable_mask).sum().item()),
                "masked_after": int(remaining.sum().item()),
                "remasked": 0,
                "elf_steps": int(self.elf_integration_steps),
            }]
            return sampled, confidence, trace
        return sampled, confidence

    def _unpack_agent_tokens(self, packed, sampled, confidence):
        num_agents = int(packed["num_agents"])
        sequence_tokens = int(packed["sequence_tokens"])
        token_ids = torch.zeros(
            num_agents,
            sequence_tokens,
            dtype=torch.long,
            device=sampled.device,
        )
        token_confidence = torch.zeros(
            num_agents,
            sequence_tokens,
            dtype=confidence.dtype,
            device=confidence.device,
        )
        for batch_idx in range(sampled.shape[0]):
            agent_indices = packed["agent_indices"][batch_idx]
            valid_slots = agent_indices >= 0
            if not valid_slots.any():
                continue
            flat_agents = agent_indices[valid_slots]
            flat_chunks = packed["chunk_ids"][batch_idx, valid_slots]
            token_ids[flat_agents, flat_chunks] = sampled[batch_idx, valid_slots]
            token_confidence[flat_agents, flat_chunks] = confidence[batch_idx, valid_slots]
        return token_ids, token_confidence

    def _trajectory_tables(self, device, dtype):
        return {
            0: torch.from_numpy(self.trajectory_token_all["veh"]).to(device=device, dtype=dtype),
            1: torch.from_numpy(self.trajectory_token_all["ped"]).to(device=device, dtype=dtype),
            2: torch.from_numpy(self.trajectory_token_all["cyc"]).to(device=device, dtype=dtype),
        }

    def _select_token_traj_all(self, token_ids, agent_type):
        device = token_ids.device
        tables = self._trajectory_tables(device, torch.float32)
        num_agents, num_tokens = token_ids.shape
        selected = torch.zeros(
            num_agents,
            num_tokens,
            self.future_chunk_steps,
            4,
            2,
            dtype=torch.float32,
            device=device,
        )
        for type_id, table in tables.items():
            mask = agent_type == type_id
            if not mask.any():
                continue
            ids = token_ids[mask].clamp(min=0, max=table.shape[0] - 1)
            selected[mask] = table[ids, : self.future_chunk_steps]
        current_token = self._trajectory_token_current(token_ids, agent_type, device)
        return torch.cat([selected, current_token.unsqueeze(2)], dim=2)

    def _trajectory_token_current(self, token_ids, agent_type, device):
        tables = {
            0: torch.from_numpy(self.trajectory_token["veh"]).to(device=device, dtype=torch.float32),
            1: torch.from_numpy(self.trajectory_token["ped"]).to(device=device, dtype=torch.float32),
            2: torch.from_numpy(self.trajectory_token["cyc"]).to(device=device, dtype=torch.float32),
        }
        out = torch.zeros(
            *token_ids.shape,
            4,
            2,
            dtype=torch.float32,
            device=device,
        )
        for type_id, table in tables.items():
            mask = agent_type == type_id
            if not mask.any():
                continue
            ids = token_ids[mask].clamp(min=0, max=table.shape[0] - 1)
            out[mask] = table[ids]
        return out

    def _clone_elf_value(self, value):
        if torch.is_tensor(value):
            return value.clone()
        if isinstance(value, dict):
            return {key: self._clone_elf_value(child) for key, child in value.items()}
        if hasattr(value, "clone"):
            return value.clone()
        if hasattr(value, "__dict__"):
            clone = value.__class__.__new__(value.__class__)
            for key, child in value.__dict__.items():
                setattr(clone, key, self._clone_elf_value(child))
            return clone
        return value

    def _clone_elf_data(self, data):
        if hasattr(data, "clone"):
            return data.clone()
        return self._clone_elf_value(data)

    def _history_anchor_mask(self, data, slot):
        agent = data["agent"]
        slot = min(max(int(slot), 0), agent["agent_valid_mask"].shape[1] - 1)
        current_index = self.num_historical_steps - 1
        current_index = min(max(current_index, 0), agent["valid_mask"].shape[1] - 1)
        valid = agent["agent_valid_mask"][:, slot].bool()
        valid = valid & agent["valid_mask"][:, current_index].bool()
        valid = valid & (agent["type"] != 3)
        return valid

    def _elf_generation_agent_mask(self, data):
        agent = data["agent"]
        current_index = min(
            max(self.num_historical_steps - 1, 0),
            agent["valid_mask"].shape[1] - 1,
        )
        return agent["valid_mask"][:, current_index].bool() & (agent["type"] != 3)

    def _elf_supervision_agent_mask(self, data):
        generation_agents = self._elf_generation_agent_mask(data)
        if self.supervision_mode in ("smart_category3", "target", "category3"):
            category = data["agent"]["category"].long().to(device=generation_agents.device)
            return generation_agents & (category == 3)
        return generation_agents

    def _roll_elf_history_state(self, history, committed, generation_agents):
        history_steps = int(history.shape[1])
        if history_steps <= 0:
            return history
        committed = committed.to(device=history.device, dtype=history.dtype)
        rolled = torch.cat([history, committed], dim=1)[:, -history_steps:]
        result = history.clone()
        update = generation_agents.to(device=history.device, dtype=torch.bool)
        result[update] = rolled[update]
        return result

    def _write_elf_history_tokens(
        self,
        data,
        history_token_ids,
        history_token_pos,
        history_token_heading,
        history_token_valid,
        generation_agents,
    ):
        agent = data["agent"]
        history_steps = min(int(self.history_tokens), int(history_token_pos.shape[1]))
        if history_steps <= 0:
            return data
        update = generation_agents.to(device=history_token_pos.device, dtype=torch.bool)
        if "token_idx" in agent and history_token_ids is not None:
            target = agent["token_idx"][:, :history_steps]
            target_update = update.to(device=target.device)
            target[target_update] = history_token_ids[:, :history_steps].to(
                device=target.device,
                dtype=target.dtype,
            )[target_update]
        if "token_pos" in agent:
            target = agent["token_pos"][:, :history_steps]
            target_update = update.to(device=target.device)
            target[target_update] = history_token_pos[:, :history_steps].to(
                device=target.device,
                dtype=target.dtype,
            )[target_update]
        if "token_heading" in agent:
            target = agent["token_heading"][:, :history_steps]
            target_update = update.to(device=target.device)
            target[target_update] = history_token_heading[:, :history_steps].to(
                device=target.device,
                dtype=target.dtype,
            )[target_update]
        if "agent_valid_mask" in agent:
            target = agent["agent_valid_mask"][:, :history_steps]
            target_update = update.to(device=target.device)
            target[target_update] = history_token_valid[:, :history_steps].to(
                device=target.device,
                dtype=target.dtype,
            )[target_update]
        return data

    def _write_elf_history_frames(
        self,
        data,
        history_frame_pos,
        history_frame_heading,
        history_frame_valid,
        generation_agents,
    ):
        agent = data["agent"]
        frame_steps = min(int(self.num_historical_steps), int(history_frame_pos.shape[1]))
        if frame_steps <= 0:
            return data
        update = generation_agents.to(device=history_frame_pos.device, dtype=torch.bool)
        if "position" in agent:
            target = agent["position"][:, :frame_steps]
            target_update = update.to(device=target.device)
            pos_dim = min(target.shape[-1], history_frame_pos.shape[-1])
            source = history_frame_pos[:, :frame_steps, :pos_dim].to(
                device=target.device,
                dtype=target.dtype,
            )
            target[target_update, :, :pos_dim] = source[target_update]
        if "heading" in agent:
            target = agent["heading"][:, :frame_steps]
            target_update = update.to(device=target.device)
            target[target_update] = history_frame_heading[:, :frame_steps].to(
                device=target.device,
                dtype=target.dtype,
            )[target_update]
        if "valid_mask" in agent:
            target = agent["valid_mask"][:, :frame_steps]
            target_update = update.to(device=target.device)
            target[target_update] = history_frame_valid[:, :frame_steps].to(
                device=target.device,
                dtype=target.dtype,
            )[target_update]
        return data

    def _write_elf_history_anchor(
        self,
        data,
        anchor_pos,
        anchor_heading,
        active_mask,
        anchor_token_ids=None,
    ):
        agent = data["agent"]
        active_mask = active_mask.to(device=anchor_pos.device, dtype=torch.bool)
        if "position" in agent:
            current_index = min(
                max(self.num_historical_steps - 1, 0),
                agent["position"].shape[1] - 1,
            )
            pos_target = agent["position"][:, current_index]
            pos_dim = min(pos_target.shape[-1], anchor_pos.shape[-1])
            pos_target[active_mask, :pos_dim] = anchor_pos[active_mask, :pos_dim].to(
                device=pos_target.device,
                dtype=pos_target.dtype,
            )
        if "heading" in agent:
            current_index = min(
                max(self.num_historical_steps - 1, 0),
                agent["heading"].shape[1] - 1,
            )
            agent["heading"][active_mask, current_index] = anchor_heading[active_mask].to(
                device=agent["heading"].device,
                dtype=agent["heading"].dtype,
            )
        if "valid_mask" in agent:
            current_index = min(
                max(self.num_historical_steps - 1, 0),
                agent["valid_mask"].shape[1] - 1,
            )
            agent["valid_mask"][:, current_index] = (
                agent["valid_mask"][:, current_index].bool()
                & active_mask.to(device=agent["valid_mask"].device)
            )
        if "token_pos" in agent:
            history_index = min(
                max(self.history_tokens - 1, 0),
                agent["token_pos"].shape[1] - 1,
            )
            token_pos_target = agent["token_pos"][:, history_index]
            pos_dim = min(token_pos_target.shape[-1], anchor_pos.shape[-1])
            token_pos_target[active_mask, :pos_dim] = anchor_pos[
                active_mask,
                :pos_dim,
            ].to(device=token_pos_target.device, dtype=token_pos_target.dtype)
        if "token_heading" in agent:
            history_index = min(
                max(self.history_tokens - 1, 0),
                agent["token_heading"].shape[1] - 1,
            )
            agent["token_heading"][active_mask, history_index] = anchor_heading[
                active_mask
            ].to(
                device=agent["token_heading"].device,
                dtype=agent["token_heading"].dtype,
            )
        if "agent_valid_mask" in agent:
            history_index = min(
                max(self.history_tokens - 1, 0),
                agent["agent_valid_mask"].shape[1] - 1,
            )
            agent["agent_valid_mask"][:, history_index] = (
                agent["agent_valid_mask"][:, history_index].bool()
                & active_mask.to(device=agent["agent_valid_mask"].device)
            )
        if anchor_token_ids is not None and "token_idx" in agent:
            history_index = min(
                max(self.history_tokens - 1, 0),
                agent["token_idx"].shape[1] - 1,
            )
            agent["token_idx"][active_mask, history_index] = anchor_token_ids[
                active_mask
            ].to(device=agent["token_idx"].device, dtype=agent["token_idx"].dtype)
        return data

    def _build_elf_training_view(self, data, window_offset):
        window_offset = int(window_offset)
        if window_offset <= 0:
            return data
        view = self._clone_elf_data(data)
        agent = data["agent"]
        token_start = self.history_tokens + window_offset
        history_start = max(0, token_start - self.history_tokens)
        history_end = min(token_start, agent["token_pos"].shape[1])
        generation_agents = self._elf_generation_agent_mask(data)
        history_token_pos = agent["token_pos"][:, history_start:history_end]
        history_token_heading = agent["token_heading"][:, history_start:history_end]
        history_token_valid = agent["agent_valid_mask"][:, history_start:history_end].bool()
        history_token_valid = history_token_valid & generation_agents[:, None].to(
            device=history_token_valid.device,
            dtype=torch.bool,
        )
        history_token_ids = agent["token_idx"][:, history_start:history_end] if "token_idx" in agent else None
        self._write_elf_history_tokens(
            view,
            history_token_ids,
            history_token_pos,
            history_token_heading,
            history_token_valid,
            generation_agents,
        )
        frame_end = min(
            self.num_historical_steps + window_offset * self.future_chunk_steps,
            agent["position"].shape[1],
        )
        frame_start = max(0, frame_end - self.num_historical_steps)
        history_frame_pos = agent["position"][:, frame_start:frame_end, :2]
        history_frame_heading = agent["heading"][:, frame_start:frame_end]
        history_frame_valid = agent["valid_mask"][:, frame_start:frame_end].bool()
        history_frame_valid = history_frame_valid & generation_agents[:, None].to(
            device=history_frame_valid.device,
            dtype=torch.bool,
        )
        self._write_elf_history_frames(
            view,
            history_frame_pos,
            history_frame_heading,
            history_frame_valid,
            generation_agents,
        )
        return view

    def _sample_elf_training_window_offset(self, data):
        max_offset = max(0, self.elf_sequence_tokens - self.elf_window_tokens)
        if max_offset <= 0 or not self.training:
            return 0
        device = data["agent"]["token_idx"].device
        return int(torch.randint(max_offset + 1, (1,), device=device).item())

    def _decode_elf_token_sequence(self, data, token_ids, token_valid, start_pos, start_heading):
        num_agents, num_tokens = token_ids.shape
        device = token_ids.device
        traj = start_pos.new_zeros(num_agents, num_tokens * self.future_chunk_steps, 2)
        head = start_heading.new_zeros(num_agents, num_tokens * self.future_chunk_steps)
        valid = torch.zeros(
            num_agents,
            num_tokens * self.future_chunk_steps,
            dtype=torch.bool,
            device=device,
        )
        token_pos = start_pos.new_zeros(num_agents, num_tokens, 2)
        token_heading = start_heading.new_zeros(num_agents, num_tokens)
        agent_type = data["agent"]["type"].long().to(device)
        current_pos = start_pos.to(device=device).clone()
        current_heading = start_heading.to(device=device).clone()
        token_traj_all = self._select_token_traj_all(token_ids, agent_type)

        for token_idx in range(num_tokens):
            theta = current_heading
            cos, sin = theta.cos(), theta.sin()
            rot_mat = torch.zeros(num_agents, 2, 2, device=device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            local = token_traj_all[:, token_idx]
            world = torch.bmm(
                local.reshape(-1, 4, 2),
                rot_mat[:, None].repeat(1, local.shape[1], 1, 1).reshape(-1, 2, 2),
            )
            world = world.reshape(num_agents, local.shape[1], 4, 2)
            world = world + current_pos[:, None, None, :]
            active = (
                token_valid[:, token_idx].bool()
                & (token_ids[:, token_idx] >= 0)
                & (token_ids[:, token_idx] < self.token_size)
            )
            frame_start = token_idx * self.future_chunk_steps
            frame_end = frame_start + self.future_chunk_steps
            if active.any():
                traj[active, frame_start:frame_end] = world[
                    active,
                    1:1 + self.future_chunk_steps,
                ].mean(dim=2)
                diff_xy = (
                    world[active, 1:1 + self.future_chunk_steps, 0, :]
                    - world[active, 1:1 + self.future_chunk_steps, 3, :]
                )
                head[active, frame_start:frame_end] = torch.atan2(
                    diff_xy[..., 1],
                    diff_xy[..., 0],
                )
                valid[active, frame_start:frame_end] = True
                current_pos[active] = world[active, -1].mean(dim=1)
                end_diff = world[active, -1, 0, :] - world[active, -1, 3, :]
                current_heading[active] = torch.atan2(end_diff[:, 1], end_diff[:, 0])
            token_pos[:, token_idx] = current_pos
            token_heading[:, token_idx] = current_heading
        return traj, head, valid, token_pos, token_heading, current_pos, current_heading

    def _commit_elf_tokens_to_rollout_data(
        self,
        data,
        token_ids,
        token_confidence,
        token_valid,
        *,
        token_start,
        commit_tokens,
    ):
        del token_confidence
        commit_tokens = min(int(commit_tokens), int(token_ids.shape[1]))
        if commit_tokens <= 0:
            return data
        agent = data["agent"]
        commit_ids = token_ids[:, :commit_tokens].long()
        commit_valid = token_valid[:, :commit_tokens].bool()
        current_index = min(
            max(self.num_historical_steps - 1, 0),
            agent["position"].shape[1] - 1,
        )
        start_pos = agent["position"][:, current_index, :2].to(
            device=commit_ids.device,
            dtype=torch.float32,
        )
        start_heading = agent["heading"][:, current_index].to(
            device=commit_ids.device,
            dtype=torch.float32,
        )
        (
            commit_traj,
            commit_head,
            commit_frame_valid,
            commit_token_pos,
            commit_token_heading,
            _current_pos,
            _current_heading,
        ) = self._decode_elf_token_sequence(
            data,
            commit_ids,
            commit_valid,
            start_pos,
            start_heading,
        )

        if "token_idx" in agent:
            token_end = min(token_start + commit_tokens, agent["token_idx"].shape[1])
            count = max(0, token_end - token_start)
            if count > 0:
                sl = slice(token_start, token_end)
                agent["token_idx"][:, sl] = commit_ids[:, :count].to(
                    device=agent["token_idx"].device,
                    dtype=agent["token_idx"].dtype,
                )
        if "agent_valid_mask" in agent:
            token_end = min(token_start + commit_tokens, agent["agent_valid_mask"].shape[1])
            count = max(0, token_end - token_start)
            if count > 0:
                agent["agent_valid_mask"][:, token_start:token_end] = commit_valid[
                    :,
                    :count,
                ].to(device=agent["agent_valid_mask"].device)
        if "token_pos" in agent:
            token_end = min(token_start + commit_tokens, agent["token_pos"].shape[1])
            count = max(0, token_end - token_start)
            if count > 0:
                agent["token_pos"][:, token_start:token_end, :2] = commit_token_pos[
                    :,
                    :count,
                ].to(device=agent["token_pos"].device, dtype=agent["token_pos"].dtype)
        if "token_heading" in agent:
            token_end = min(token_start + commit_tokens, agent["token_heading"].shape[1])
            count = max(0, token_end - token_start)
            if count > 0:
                agent["token_heading"][:, token_start:token_end] = commit_token_heading[
                    :,
                    :count,
                ].to(device=agent["token_heading"].device, dtype=agent["token_heading"].dtype)

        frame_offset = max(0, int(token_start) - self.history_tokens)
        frame_start = self.num_historical_steps + frame_offset * self.future_chunk_steps
        frame_count = min(commit_traj.shape[1], agent["position"].shape[1] - frame_start)
        if frame_count > 0:
            frame_end = frame_start + frame_count
            agent["position"][:, frame_start:frame_end, :2] = commit_traj[
                :,
                :frame_count,
            ].to(device=agent["position"].device, dtype=agent["position"].dtype)
            agent["heading"][:, frame_start:frame_end] = commit_head[:, :frame_count].to(
                device=agent["heading"].device,
                dtype=agent["heading"].dtype,
            )
            agent["valid_mask"][:, frame_start:frame_end] = commit_frame_valid[
                :,
                :frame_count,
            ].to(device=agent["valid_mask"].device)

        generation_agents = self._elf_generation_agent_mask(data).to(device=commit_ids.device)
        history_steps = min(int(self.history_tokens), int(agent["token_pos"].shape[1]))
        if history_steps > 0:
            committed_valid = commit_valid & generation_agents[:, None].to(
                device=commit_valid.device,
                dtype=torch.bool,
            )
            history_token_ids = self._roll_elf_history_state(
                agent["token_idx"][:, :history_steps],
                commit_ids,
                generation_agents,
            ) if "token_idx" in agent else None
            history_token_pos = self._roll_elf_history_state(
                agent["token_pos"][:, :history_steps],
                commit_token_pos,
                generation_agents,
            )
            history_token_heading = self._roll_elf_history_state(
                agent["token_heading"][:, :history_steps],
                commit_token_heading,
                generation_agents,
            )
            history_token_valid = self._roll_elf_history_state(
                agent["agent_valid_mask"][:, :history_steps].bool(),
                committed_valid,
                generation_agents,
            )
            self._write_elf_history_tokens(
                data,
                history_token_ids,
                history_token_pos,
                history_token_heading,
                history_token_valid,
                generation_agents,
            )

        frame_steps = min(int(self.num_historical_steps), int(agent["position"].shape[1]))
        if frame_steps > 0:
            frame_valid = commit_frame_valid & generation_agents[:, None].to(
                device=commit_frame_valid.device,
                dtype=torch.bool,
            )
            history_frame_pos = self._roll_elf_history_state(
                agent["position"][:, :frame_steps, :2],
                commit_traj,
                generation_agents,
            )
            history_frame_heading = self._roll_elf_history_state(
                agent["heading"][:, :frame_steps],
                commit_head,
                generation_agents,
            )
            history_frame_valid = self._roll_elf_history_state(
                agent["valid_mask"][:, :frame_steps].bool(),
                frame_valid,
                generation_agents,
            )
            self._write_elf_history_frames(
                data,
                history_frame_pos,
                history_frame_heading,
                history_frame_valid,
                generation_agents,
            )
        return data

    def _decode_token_rollout(self, data, token_ids, token_confidence, token_valid):
        device = token_ids.device
        num_agents, num_tokens = token_ids.shape
        pred_len = min(num_tokens * self.future_chunk_steps, self.num_future_steps)
        pred_traj = torch.zeros(num_agents, pred_len, 2, device=device)
        pred_head = torch.zeros(num_agents, pred_len, device=device)
        pred_valid = torch.zeros(num_agents, pred_len, dtype=torch.bool, device=device)
        agent_type = data["agent"]["type"].long().to(device)
        current_pos = data["agent"]["position"][:, self.num_historical_steps - 1, :2].to(device=device).clone()
        current_heading = data["agent"]["heading"][:, self.num_historical_steps - 1].to(device=device).clone()
        token_traj_all = self._select_token_traj_all(token_ids, agent_type)

        for chunk_idx in range(num_tokens):
            theta = current_heading
            cos, sin = theta.cos(), theta.sin()
            rot_mat = torch.zeros(num_agents, 2, 2, device=device)
            rot_mat[:, 0, 0] = cos
            rot_mat[:, 0, 1] = sin
            rot_mat[:, 1, 0] = -sin
            rot_mat[:, 1, 1] = cos
            local = token_traj_all[:, chunk_idx]
            world = torch.bmm(local.reshape(-1, 4, 2), rot_mat[:, None].repeat(1, local.shape[1], 1, 1).reshape(-1, 2, 2))
            world = world.reshape(num_agents, local.shape[1], 4, 2)
            world = world + current_pos[:, None, None, :]
            frame_start = chunk_idx * self.future_chunk_steps
            frame_end = min(frame_start + self.future_chunk_steps, pred_len)
            frame_count = frame_end - frame_start
            if frame_count > 0:
                pred_traj[:, frame_start:frame_end] = world[:, 1:1 + frame_count].mean(dim=2)
                diff_xy = world[:, 1:1 + frame_count, 0, :] - world[:, 1:1 + frame_count, 3, :]
                pred_head[:, frame_start:frame_end] = torch.atan2(diff_xy[..., 1], diff_xy[..., 0])
                pred_valid[:, frame_start:frame_end] = token_valid[:, chunk_idx:chunk_idx + 1]
            current_pos = world[:, -1].mean(dim=1)
            diff_xy = world[:, -1, 0, :] - world[:, -1, 3, :]
            current_heading = torch.atan2(diff_xy[:, 1], diff_xy[:, 0])

        gt = data["agent"]["position"][
            :,
            self.num_historical_steps:self.num_historical_steps + pred_len,
            :2,
        ].contiguous()
        official_valid = data["agent"]["valid_mask"][
            :,
            self.num_historical_steps:self.num_historical_steps + pred_len,
        ].bool().clone()
        pred_valid = pred_valid & official_valid
        velocity = torch.zeros_like(pred_traj)
        if pred_len > 1:
            velocity[:, 1:] = pred_traj[:, 1:] - pred_traj[:, :-1]
        return {
            "pos_a": pred_traj,
            "head_a": pred_head,
            "gt": gt,
            "valid_mask": pred_valid,
            "pred_traj": pred_traj,
            "pred_head": pred_head,
            "next_token_idx": token_ids,
            "next_token_idx_gt": packed_token_gt(data, self.history_tokens, self.elf_sequence_tokens),
            "next_token_eval_mask": token_valid,
            "pred_prob": token_confidence,
            "pred_valid_mask": pred_valid,
            "official_valid_mask": official_valid,
            "vel": velocity,
        }

    def forward(self, data: HeteroData):
        return self.inference(data)

    def training_step(self, data, batch_idx):
        del batch_idx
        data = self._prepare_batch(data)
        window_offset = self._sample_elf_training_window_offset(data)
        window_view = self._build_elf_training_view(data, window_offset)
        sequence_tokens = min(
            self.elf_window_tokens,
            self.elf_sequence_tokens - window_offset,
        )
        packed, summary, *_ = self._build_diffusion_inputs(
            window_view,
            token_start=self.history_tokens + window_offset,
            sequence_tokens=sequence_tokens,
        )
        diffusion_loss, mask_acc = self._compute_diffusion_loss(packed, summary)
        loss = diffusion_loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("diffusion_loss", diffusion_loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_mask_acc", mask_acc, on_step=True, on_epoch=True, batch_size=1)
        self.log("train_elf_window_offset", float(window_offset), on_step=True, on_epoch=False, batch_size=1)
        return loss

    def validation_step(self, data, batch_idx):
        del batch_idx
        data = self._prepare_batch(data)
        packed, summary, *_ = self._build_diffusion_inputs(
            data,
            token_start=self.history_tokens,
            sequence_tokens=self.elf_window_tokens,
        )
        diffusion_loss, mask_acc = self._compute_diffusion_loss(packed, summary)
        self.log("val_loss", diffusion_loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val_ar_window_loss", diffusion_loss, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)
        self.log("val_ar_window_mask_acc", mask_acc, prog_bar=False, on_step=False, on_epoch=True, batch_size=1, sync_dist=True)

        pred_out = self.inference(data)
        em = self._metric_agent_mask(data)
        eval_valid = self._validation_eval_valid_mask(data, pred_out)
        if em.any() and eval_valid[em].any():
            self.minADE.update(
                pred=pred_out["pred_traj"][em],
                target=pred_out["gt"][em],
                valid_mask=eval_valid[em],
            )
            self.minFDE.update(
                pred=pred_out["pred_traj"][em],
                target=pred_out["gt"][em],
                valid_mask=eval_valid[em],
            )
            self.log("val_minADE", self.minADE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            self.log("val_minFDE", self.minFDE, prog_bar=True, on_step=False, on_epoch=True, batch_size=1)
            shapes = data["agent"]["shape"]
            if shapes.dim() == 3:
                shapes = shapes[:, self.num_historical_steps - 1, :]
            self.conflict_rate.update(
                pred_out["pred_traj"][em],
                pred_out["pred_head"][em],
                shapes[em],
                eval_valid[em],
            )
            self.interaction_consistency.update(
                pred_out["pred_traj"][em],
                pred_out["gt"][em],
                eval_valid[em],
            )
            self.log("val_conflict_rate", self.conflict_rate, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)
            self.log("val_interaction_consistency", self.interaction_consistency, prog_bar=False, on_step=False, on_epoch=True, batch_size=1)

    @torch.no_grad()
    def _one_shot_elf_inference(self, data):
        data = self._prepare_batch(data)
        packed, summary, *_ = self._build_diffusion_inputs(data)
        sampled, confidence = self._diffusion_sample(
            summary=summary,
            token_positions=None,
            token_headings=None,
            token_agent_ids=None,
            chunk_ids=packed["chunk_ids"],
            valid_mask=packed["valid_mask"],
            agent_context=packed["context"],
            agent_type_ids=packed["agent_type_ids"],
            packed=packed,
        )
        token_ids, token_confidence = self._unpack_agent_tokens(packed, sampled, confidence)
        token_valid = packed["token_valid_by_agent"]
        return self._decode_token_rollout(data, token_ids, token_confidence, token_valid)

    @torch.no_grad()
    def _receding_horizon_elf_inference(self, data):
        data = self._prepare_batch(data)
        rollout_data = self._clone_elf_data(data)
        total_tokens = int(self.elf_sequence_tokens)
        offset = 0
        pred_token_ids = None
        pred_token_confidence = None
        pred_token_valid = None
        while offset < total_tokens:
            window_tokens = min(self.elf_window_tokens, total_tokens - offset)
            token_start = self.history_tokens + offset
            packed, summary, *_ = self._build_diffusion_inputs(
                rollout_data,
                token_start=token_start,
                sequence_tokens=window_tokens,
            )
            sampled, confidence = self._diffusion_sample(
                summary=summary,
                token_positions=None,
                token_headings=None,
                token_agent_ids=None,
                chunk_ids=packed["chunk_ids"],
                valid_mask=packed["valid_mask"],
                agent_context=packed["context"],
                agent_type_ids=packed["agent_type_ids"],
                packed=packed,
            )
            token_ids, token_confidence = self._unpack_agent_tokens(
                packed,
                sampled,
                confidence,
            )
            token_valid = packed["token_valid_by_agent"]
            if pred_token_ids is None:
                num_agents = int(token_ids.shape[0])
                pred_token_ids = torch.zeros(
                    num_agents,
                    total_tokens,
                    dtype=token_ids.dtype,
                    device=token_ids.device,
                )
                pred_token_confidence = torch.zeros(
                    num_agents,
                    total_tokens,
                    dtype=token_confidence.dtype,
                    device=token_confidence.device,
                )
                pred_token_valid = torch.zeros(
                    num_agents,
                    total_tokens,
                    dtype=torch.bool,
                    device=token_valid.device,
                )
            commit_tokens = min(self.elf_commit_tokens, window_tokens, total_tokens - offset)
            commit_slice = slice(offset, offset + commit_tokens)
            pred_token_ids[:, commit_slice] = token_ids[:, :commit_tokens]
            pred_token_confidence[:, commit_slice] = token_confidence[:, :commit_tokens]
            pred_token_valid[:, commit_slice] = token_valid[:, :commit_tokens]
            rollout_data = self._commit_elf_tokens_to_rollout_data(
                rollout_data,
                token_ids,
                token_confidence,
                token_valid,
                token_start=token_start,
                commit_tokens=commit_tokens,
            )
            offset += commit_tokens

        return self._decode_token_rollout(
            data,
            pred_token_ids,
            pred_token_confidence,
            pred_token_valid,
        )

    @torch.no_grad()
    def inference(self, data):
        if not getattr(self, "elf_receding_horizon", True):
            return self._one_shot_elf_inference(data)
        return self._receding_horizon_elf_inference(data)

    def _metric_agent_mask(self, data):
        current_step = self.num_historical_steps - 1
        valid = data["agent"]["valid_mask"][:, current_step].bool() & (data["agent"]["type"] != 3)
        if self.metric_mode in ("smart_category3", "category3", "target"):
            valid = valid & (data["agent"]["category"].long() == 3)
        return valid

    def _validation_eval_valid_mask(self, data, prediction):
        pred_len = int(prediction["pred_traj"].shape[1])
        valid = data["agent"]["valid_mask"][
            :,
            self.num_historical_steps:self.num_historical_steps + pred_len,
        ].bool()
        if valid.shape[1] < pred_len:
            valid = F.pad(valid, (0, pred_len - valid.shape[1]), value=False)
        pred_valid = prediction.get("pred_valid_mask")
        if pred_valid is not None:
            valid = valid & pred_valid[:, :pred_len].bool()
        return valid

    def on_validation_start(self):
        self.gt = []
        self.pred = []
        self.scenario_rollouts = []
        self.batch_metric = defaultdict(list)

    def configure_optimizers(self):
        encoder_param_ids = {id(param) for param in self.encoder.parameters()}
        encoder_params = [param for param in self.encoder.parameters() if param.requires_grad]
        other_params = [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in encoder_param_ids
        ]
        param_groups = []
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": self.lr * self.encoder_lr_scale})
        if other_params:
            param_groups.append({"params": other_params, "lr": self.lr})
        optimizer = torch.optim.AdamW(param_groups, lr=self.lr)

        def lr_lambda(current_step):
            if current_step + 1 < self.warmup_steps:
                return float(current_step + 1) / float(max(1, self.warmup_steps))
            if current_step >= self.total_steps:
                return 0.0
            return max(
                0.0,
                0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (current_step - self.warmup_steps)
                        / float(max(1, self.total_steps - self.warmup_steps))
                    )
                ),
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return [optimizer], [lr_scheduler]

    def load_params_from_file(self, filename, logger, to_cpu=False):
        if not os.path.isfile(filename):
            raise FileNotFoundError
        logger.info(
            "==> Loading parameters from checkpoint %s to %s"
            % (filename, "CPU" if to_cpu else "GPU")
        )
        loc_type = torch.device("cpu") if to_cpu else None
        checkpoint = torch_load_compat(filename, map_location=loc_type, weights_only=False)
        model_state_disk = checkpoint["state_dict"]
        logger.info(f"The number of disk ckpt keys: {len(model_state_disk)}")
        model_state = self.state_dict()
        filtered = {}
        for key, value in model_state_disk.items():
            load_key = key
            if key.startswith("core."):
                load_key = key[len("core."):]
            if load_key in model_state and model_state[load_key].shape == value.shape:
                filtered[load_key] = value
            else:
                if load_key not in model_state:
                    print(f"Ignore key in disk (not found in model): {key}, shape={value.shape}")
                else:
                    print(
                        "Ignore key in disk (shape does not match): "
                        f"{key}, load_shape={value.shape}, model_shape={model_state[load_key].shape}"
                    )
        missing_keys, unexpected_keys = self.load_state_dict(filtered, strict=False)
        logger.info(f"Missing keys: {missing_keys}")
        logger.info(f"The number of missing keys: {len(missing_keys)}")
        logger.info(f"The number of unexpected keys: {len(unexpected_keys)}")
        logger.info("==> Done (total keys %d)" % len(model_state))
        return checkpoint.get("it", 0.0), checkpoint.get("epoch", -1)


def packed_token_gt(data, history_tokens, sequence_tokens):
    token_start = history_tokens
    token_end = token_start + sequence_tokens
    token_ids = data["agent"]["token_idx"][:, token_start:token_end].long()
    if token_ids.shape[1] < sequence_tokens:
        token_ids = F.pad(token_ids, (0, sequence_tokens - token_ids.shape[1]), value=0)
    return token_ids
