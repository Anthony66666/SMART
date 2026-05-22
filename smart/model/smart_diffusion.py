import torch
import torch.nn.functional as F
from torch_geometric.data import Batch, HeteroData

from smart.metrics.joint_consistency import ConflictRate, InteractionConsistency
from smart.model.smart import SMART
from smart.modules.diffusion_decoder import DiffusionDecoder
from smart.utils.diffusion_noise import LogLinearNoise


def _lookup_token_trajectories(token_ids, token_vocab, agent_types):
    """Map token indices to trajectory waypoints from the vocabulary.

    Args:
        token_ids: [N] long in [0, token_size-1]
        token_vocab: {'veh','ped','cyc'} each [token_size, 6, 4, 2] float
        agent_types: [N] long (0=veh, 1=ped, 2=cyc)

    Returns:
        trajectories: [N, 6, 4, 2] float
    """
    device = token_ids.device
    traj = torch.zeros(token_ids.shape[0], 6, 4, 2, device=device)

    for type_name, type_id in [('veh', 0), ('ped', 1), ('cyc', 2)]:
        mask = (agent_types == type_id)
        if mask.any():
            vocab = token_vocab[type_name].to(device=device, dtype=torch.float)
            traj[mask] = vocab[token_ids[mask]]
    return traj


def _lookup_token_endpoints(token_ids, token_vocab, agent_types):
    """Map token indices to endpoint waypoint polygons from the vocabulary."""
    device = token_ids.device
    sample = next(iter(token_vocab.values()))
    endpoints = torch.zeros(
        token_ids.shape[0],
        *sample.shape[1:],
        device=device,
        dtype=torch.float,
    )

    for type_name, type_id in [('veh', 0), ('ped', 1), ('cyc', 2)]:
        mask = (agent_types == type_id)
        if mask.any():
            vocab = token_vocab[type_name].to(device=device, dtype=torch.float)
            endpoints[mask] = vocab[token_ids[mask]]
    return endpoints


class SMARTDiffusion(SMART):
    """SMART with discrete mask diffusion for multi-agent trajectory generation.

    Replaces the autoregressive token-by-token rollout with joint iterative
    denoising: all agents' future tokens are refined simultaneously through
    a bidirectional Transformer. This addresses the same-step joint
    consistency limitation of AR generation.
    """

    def __init__(self, model_config) -> None:
        super().__init__(model_config)

        diffusion_cfg = getattr(model_config, 'diffusion', None)
        if diffusion_cfg is None:
            raise ValueError("SMARTDiffusion requires Model.diffusion config.")

        fc_steps = int(getattr(diffusion_cfg, 'future_chunk_steps', 5))
        if self.num_future_steps % fc_steps != 0:
            raise ValueError(
                f"num_future_steps ({self.num_future_steps}) "
                f"must be divisible by future_chunk_steps ({fc_steps})."
            )
        self.future_chunk_steps = fc_steps
        self.num_future_chunks = self.num_future_steps // fc_steps
        agent_shift = int(self.encoder.agent_encoder.shift)
        if self.future_chunk_steps != agent_shift:
            raise ValueError(
                "SMARTDiffusion currently uses SMART's trajectory token shift; "
                f"diffusion.future_chunk_steps ({self.future_chunk_steps}) must equal "
                f"agent_encoder.shift ({agent_shift})."
            )
        self.diffusion_num_steps = int(getattr(diffusion_cfg, 'num_steps', 32))
        self.diffusion_num_layers = int(getattr(diffusion_cfg, 'num_layers', 6))
        self.diffusion_eps = float(getattr(diffusion_cfg, 'eps', 1e-3))
        self.min_t = float(getattr(diffusion_cfg, 'min_t', 1e-3))
        self.freeze_encoder = bool(getattr(diffusion_cfg, 'freeze_encoder', False))
        self.ntp_aux_loss_weight = float(getattr(diffusion_cfg, 'ntp_aux_loss_weight', 0.2))
        self.use_agent_context = bool(getattr(diffusion_cfg, 'use_agent_context', True))
        self.use_type_embedding = bool(getattr(diffusion_cfg, 'use_type_embedding', True))
        self.use_map_context = bool(getattr(diffusion_cfg, 'use_map_context', True))
        self.max_map_tokens = int(getattr(diffusion_cfg, 'max_map_tokens', 128))
        self.geometry_dropout_prob = float(getattr(diffusion_cfg, 'geometry_dropout_prob', 0.0))
        self.geometry_dropout_prob = min(max(self.geometry_dropout_prob, 0.0), 1.0)
        self.target_category_only = bool(getattr(diffusion_cfg, 'target_category_only', True))
        self.agent_selection_mode = str(
            getattr(diffusion_cfg, 'agent_selection_mode', 'smart_inference')
        ).lower()
        default_supervision_mode = 'smart_category3' if self.target_category_only else 'smart_generation'
        self.supervision_mode = str(
            getattr(diffusion_cfg, 'supervision_mode', default_supervision_mode)
        ).lower()
        self.metric_mode = str(
            getattr(diffusion_cfg, 'metric_mode', 'smart_val_compatible')
        ).lower()
        self.low_variance_masking = bool(getattr(diffusion_cfg, 'low_variance_masking', True))
        self.mask_count_mode = str(getattr(diffusion_cfg, 'mask_count_mode', 'exact')).lower()
        self.antithetic_mask_ranking = bool(getattr(diffusion_cfg, 'antithetic_mask_ranking', True))
        self.remask_sampling = bool(getattr(diffusion_cfg, 'remask_sampling', True))
        self.remask_confidence_temperature = float(getattr(diffusion_cfg, 'remask_confidence_temperature', 1.0))
        self.remask_confidence_temperature = max(self.remask_confidence_temperature, 1e-6)
        self.prefix_constrained_sampling = bool(getattr(diffusion_cfg, 'prefix_constrained_sampling', False))
        self.prefix_constrained_training = bool(getattr(diffusion_cfg, 'prefix_constrained_training', False))
        self.use_proposal_geometry = bool(getattr(diffusion_cfg, 'use_proposal_geometry', True))
        self.geometry_confidence_source_threshold = float(
            getattr(diffusion_cfg, 'geometry_confidence_source_threshold', 0.35)
        )
        self.self_condition_prob = float(getattr(diffusion_cfg, 'self_condition_prob', 1.0))
        self.self_condition_prob = min(max(self.self_condition_prob, 0.0), 1.0)
        self.self_condition_visible_prob = float(getattr(diffusion_cfg, 'self_condition_visible_prob', 0.1))
        self.self_condition_visible_prob = min(max(self.self_condition_visible_prob, 0.0), 1.0)
        self.self_condition_mode = str(getattr(diffusion_cfg, 'self_condition_mode', 'argmax')).lower()
        if self.self_condition_mode != 'argmax':
            raise ValueError(f"Unsupported diffusion.self_condition_mode: {self.self_condition_mode}")
        self.self_condition_loss_weight = float(getattr(diffusion_cfg, 'self_condition_loss_weight', 1.0))
        self.self_condition_loss_weight = max(self.self_condition_loss_weight, 0.0)
        self.diffusion_eval_batches = int(getattr(diffusion_cfg, 'eval_inference_batches', 2))
        token_size = int(getattr(model_config.decoder, 'token_size', 2048))

        self.noise_schedule = LogLinearNoise(eps=self.diffusion_eps)

        self.diffusion_decoder = DiffusionDecoder(
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

        if self.freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.conflict_rate = ConflictRate()
        self.interaction_consistency = InteractionConsistency()
        self._token_vocab_cache = None
        self._token_endpoint_vocab_cache = None

    # -- properties --------------------------------------------------------

    @property
    def mask_token_id(self):
        return self.diffusion_decoder.mask_token_id

    @property
    def token_size(self):
        return self.model_config.decoder.token_size

    @property
    def history_token_steps(self):
        shift = self.encoder.agent_encoder.shift
        return max(1, (self.num_historical_steps - 1) // shift)

    @property
    def token_vocab(self):
        if getattr(self, '_token_vocab_cache', None) is None:
            dev = next(self.parameters()).device
            self._token_vocab_cache = {
                k: torch.from_numpy(v).clone().to(device=dev, dtype=torch.float)
                for k, v in self.encoder.agent_encoder.trajectory_token_all.items()
            }
        return self._token_vocab_cache

    @property
    def token_endpoint_vocab(self):
        if getattr(self, '_token_endpoint_vocab_cache', None) is None:
            dev = next(self.parameters()).device
            self._token_endpoint_vocab_cache = {
                k: torch.from_numpy(v).clone().to(device=dev, dtype=torch.float)
                for k, v in self.encoder.agent_encoder.trajectory_token.items()
            }
        return self._token_endpoint_vocab_cache

    # -- helpers -----------------------------------------------------------

    def _get_agent_batch(self, data):
        if isinstance(data, Batch) and 'batch' in data['agent']:
            return data['agent']['batch']
        return torch.zeros(int(data['agent']['num_nodes']), dtype=torch.long,
                           device=data['agent']['position'].device)

    def _pool_scene_summary(self, hist_tokens, hist_mask, agent_batch, agent_type):
        """Mean-pool agent history tokens per scene graph."""
        valid = hist_mask & (agent_type != 3).unsqueeze(-1)
        D = hist_tokens.shape[-1]
        if agent_batch.numel() == 0:
            return hist_tokens.new_zeros((1, D))
        G = int(agent_batch.max().item()) + 1
        flat = hist_tokens.reshape(-1, D)
        fmask = valid.reshape(-1)
        fbatch = agent_batch.unsqueeze(-1).expand(-1, hist_tokens.shape[1]).reshape(-1)
        pooled = hist_tokens.new_zeros((G, D))
        counts = hist_tokens.new_zeros((G,))
        if fmask.any():
            vb = fbatch[fmask]
            pooled.index_add_(0, vb, flat[fmask])
            counts.index_add_(0, vb, torch.ones(vb.shape[0], device=counts.device))
        return pooled / counts.clamp_min(1).unsqueeze(-1)

    def _pool_agent_context(self, hist_tokens, hist_mask):
        """Mean-pool each agent's visible history tokens."""
        weights = hist_mask.unsqueeze(-1).to(dtype=hist_tokens.dtype)
        summed = (hist_tokens * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def _physical_token_embeddings(self, token_ids, agent_type_ids):
        """Embed trajectory token ids with SMART's type-specific physical token MLPs."""
        agent_encoder = self.encoder.agent_encoder
        device = token_ids.device
        non_mask = token_ids != self.mask_token_id
        token_ids = token_ids.clamp(min=0, max=self.token_size - 1)
        embeddings = agent_encoder.type_a_emb.weight.new_zeros(
            (*token_ids.shape, self.hidden_dim),
            device=device,
        )

        token_specs = [
            ('veh', 0, agent_encoder.token_emb_veh),
            ('ped', 1, agent_encoder.token_emb_ped),
            ('cyc', 2, agent_encoder.token_emb_cyc),
        ]
        for token_name, type_id, token_embedder in token_specs:
            type_mask = non_mask & (agent_type_ids == type_id)
            if not type_mask.any():
                continue
            token_template = torch.from_numpy(agent_encoder.trajectory_token[token_name]).to(
                device=device,
                dtype=torch.float,
            )
            token_table = token_embedder(token_template.reshape(token_template.shape[0], -1))
            embeddings[type_mask] = token_table[token_ids[type_mask]]
        return embeddings

    def _zero_connected_loss(self):
        """Return a zero scalar still connected to trainable parameters."""
        zero = None
        for parameter in self.parameters():
            if not parameter.requires_grad:
                continue
            term = parameter.sum() * 0.0
            zero = term if zero is None else zero + term
        if zero is not None:
            return zero
        return next(self.parameters()).sum() * 0.0

    def _current_step_rank(self):
        step = 0
        rank = 0
        try:
            step = int(self.global_step)
        except Exception:
            step = int(getattr(self, '_manual_global_step', 0))
        try:
            trainer = self.trainer
            rank = int(getattr(trainer, 'global_rank', 0))
        except Exception:
            try:
                rank = int(getattr(self, 'global_rank', 0))
            except Exception:
                rank = 0
        return step, rank

    def _sample_diffusion_timesteps(self, batch_size, device, step=None, rank=None):
        """Low-discrepancy samples in [min_t, 1], including batch_size=1."""
        if batch_size <= 0:
            return torch.empty(0, device=device)
        if step is None or rank is None:
            current_step, current_rank = self._current_step_rank()
            step = current_step if step is None else step
            rank = current_rank if rank is None else rank

        if self.low_variance_masking:
            golden_ratio_inv = 0.6180339887498949
            offsets = torch.arange(batch_size, device=device, dtype=torch.float)
            base = float(step * max(1, batch_size) + rank * 104729)
            unit_t = torch.frac((offsets + base + 0.5) * golden_ratio_inv)
        elif batch_size <= 1:
            unit_t = torch.rand(batch_size, device=device)
        else:
            strata = torch.arange(batch_size, device=device, dtype=torch.float)
            unit_t = (strata + torch.rand(batch_size, device=device)) / float(batch_size)
            unit_t = unit_t[torch.randperm(batch_size, device=device)]
        return self.min_t + (1.0 - self.min_t) * unit_t

    def _mask_ranking_scores(self, shape, device, step=None, rank=None):
        if step is None or rank is None:
            current_step, current_rank = self._current_step_rank()
            step = current_step if step is None else step
            rank = current_rank if rank is None else rank
        B, L = shape
        golden_ratio_inv = 0.6180339887498949
        batch_axis = torch.arange(B, device=device, dtype=torch.float).unsqueeze(1)
        token_axis = torch.arange(L, device=device, dtype=torch.float).unsqueeze(0)
        pair_step = int(step) // 2 if self.antithetic_mask_ranking else int(step)
        scores = torch.frac(
            (
                token_axis
                + 0.37 * batch_axis
                + 0.754877666 * float(pair_step)
                + 0.56984029 * float(rank)
            )
            * golden_ratio_inv
        )
        if self.antithetic_mask_ranking and int(step) % 2 == 1:
            scores = 1.0 - scores
        return scores

    def _sample_training_mask(self, valid_mask, mask_prob, step=None, rank=None):
        if (
            not self.low_variance_masking
            or self.mask_count_mode != 'exact'
        ):
            return (torch.rand_like(valid_mask.float()) < mask_prob.unsqueeze(-1)) & valid_mask

        mask = torch.zeros_like(valid_mask)
        scores = self._mask_ranking_scores(valid_mask.shape, valid_mask.device, step=step, rank=rank)
        scores = scores.masked_fill(~valid_mask, float('inf'))
        for batch_idx in range(valid_mask.shape[0]):
            valid_count = int(valid_mask[batch_idx].sum().item())
            if valid_count <= 0:
                continue
            prob = float(mask_prob[batch_idx].detach().clamp(0.0, 1.0).item())
            mask_count = int(round(prob * valid_count))
            mask_count = min(max(mask_count, 0), valid_count)
            if mask_count <= 0:
                continue
            selected = torch.topk(scores[batch_idx], k=mask_count, largest=False).indices
            mask[batch_idx, selected] = True
        return mask

    def _apply_prefix_mask_closure(self, mask, valid_mask, token_agent_ids, chunk_ids):
        """If a prefix chunk is masked, all later valid chunks for that agent stay masked."""
        closed = (mask & valid_mask).clone()
        B = valid_mask.shape[0]
        for batch_idx in range(B):
            valid_agents = torch.unique(
                token_agent_ids[batch_idx][
                    valid_mask[batch_idx] & (token_agent_ids[batch_idx] >= 0)
                ]
            )
            for agent_id in valid_agents.tolist():
                agent_valid = (
                    valid_mask[batch_idx]
                    & (token_agent_ids[batch_idx] == int(agent_id))
                )
                masked_chunks = chunk_ids[batch_idx][agent_valid & closed[batch_idx]]
                if masked_chunks.numel() == 0:
                    continue
                first_masked_chunk = masked_chunks.min()
                closed[batch_idx] = (
                    closed[batch_idx]
                    | (
                        agent_valid
                        & (chunk_ids[batch_idx] >= first_masked_chunk)
                    )
                )
        return closed & valid_mask

    def _prefix_frontier_mask(self, mask, valid_mask, token_agent_ids, chunk_ids):
        """Return the first currently masked valid chunk for each agent."""
        candidates = mask & valid_mask
        frontier = torch.zeros_like(candidates)
        B = valid_mask.shape[0]
        for batch_idx in range(B):
            valid_agents = torch.unique(
                token_agent_ids[batch_idx][
                    valid_mask[batch_idx] & (token_agent_ids[batch_idx] >= 0)
                ]
            )
            for agent_id in valid_agents.tolist():
                agent_masked = (
                    candidates[batch_idx]
                    & (token_agent_ids[batch_idx] == int(agent_id))
                )
                if not agent_masked.any():
                    continue
                first_masked_chunk = chunk_ids[batch_idx][agent_masked].min()
                frontier[batch_idx] = (
                    frontier[batch_idx]
                    | (
                        agent_masked
                        & (chunk_ids[batch_idx] == first_masked_chunk)
                    )
                )
        return frontier

    def _generation_agent_mask(self, data):
        """Agents that participate in diffusion rollout."""
        hist_valid = data['agent']['valid_mask'][:, self.num_historical_steps - 1].bool()
        mode = getattr(self, 'agent_selection_mode', 'smart_inference')
        if mode in ('smart_inference', 'history_valid'):
            return hist_valid
        if mode in ('non_background', 'non_bg'):
            return hist_valid & (data['agent']['type'] != 3)
        if mode in ('smart_category3', 'category3', 'target_category3'):
            try:
                return hist_valid & (data['agent']['category'].long() == 3)
            except Exception:
                return hist_valid
        raise ValueError(f"Unsupported diffusion.agent_selection_mode: {mode}")

    def _supervision_agent_mask(self, data):
        """Agents that receive diffusion-token supervision/loss."""
        generation = self._generation_agent_mask(data)
        mode = getattr(self, 'supervision_mode', 'smart_category3')
        if mode in ('smart_category3', 'category3', 'target_category3'):
            try:
                return generation & (data['agent']['category'].long() == 3)
            except Exception:
                return generation
        if mode in ('smart_generation', 'generation', 'all_generation'):
            return generation
        if mode in ('legacy_target_category_only',):
            if getattr(self, 'target_category_only', True):
                try:
                    return generation & (data['agent']['category'].long() == 3)
                except Exception:
                    return generation
            return generation
        raise ValueError(f"Unsupported diffusion.supervision_mode: {mode}")

    def _metric_agent_mask(self, data, mode=None):
        """Agents used by validation metrics/export checks."""
        metric_mode = getattr(self, 'metric_mode', 'smart_val_compatible') if mode is None else str(mode).lower()
        if metric_mode in ('smart_val_compatible', 'smart_category3', 'category3', 'target_category3'):
            generation = self._generation_agent_mask(data)
            try:
                return generation & (data['agent']['category'].long() == 3)
            except Exception:
                return generation
        if metric_mode in ('smart_inference', 'generation'):
            return self._generation_agent_mask(data)
        if metric_mode in ('supervision',):
            return self._supervision_agent_mask(data)
        if metric_mode in ('non_background', 'non_bg'):
            return self._generation_agent_mask(data) & (data['agent']['type'] != 3)
        raise ValueError(f"Unsupported diffusion.metric_mode: {metric_mode}")

    def _target_agent_mask(self, data):
        return self._supervision_agent_mask(data)

    def _get_map_batch(self, data):
        if isinstance(data, Batch) and 'batch' in data['pt_token']:
            return data['pt_token']['batch']
        return torch.zeros(int(data['pt_token']['num_nodes']), dtype=torch.long,
                           device=data['pt_token']['position'].device)

    def _pack_map_context(self, data, ctx, packed, agent_positions):
        node_types = getattr(data, 'node_types', [])
        if (
            not self.use_map_context
            or 'x_pt' not in ctx
            or 'pt_token' not in node_types
        ):
            return None, None, None, None, None

        map_features = ctx['x_pt']
        map_positions = data['pt_token']['position'][:, :2].float()
        map_orientation = data['pt_token']['orientation'].float()
        map_batch = self._get_map_batch(data)
        map_visible = ctx.get('pt_visibility_mask', None)
        if map_visible is None:
            map_visible = torch.ones(map_features.shape[0], dtype=torch.bool, device=map_features.device)
        else:
            map_visible = map_visible.to(device=map_features.device, dtype=torch.bool)

        kept_context = []
        kept_positions = []
        kept_orientations = []
        kept_batch = []
        for seq_idx, (scene_idx, _packed_seq_idx, agent_indices) in enumerate(packed['agent_maps']):
            candidates = torch.nonzero(
                (map_batch == scene_idx) & map_visible,
                as_tuple=False,
            ).squeeze(-1)
            if candidates.numel() > 0 and self.max_map_tokens > 0:
                scene_agent_pos = agent_positions[agent_indices]
                dist = torch.cdist(map_positions[candidates], scene_agent_pos)
                nearest_dist = dist.min(dim=1).values
                order = torch.argsort(nearest_dist)
                candidates = candidates[order]
                candidates = candidates[:self.max_map_tokens]
            if candidates.numel() == 0:
                continue

            kept_context.append(map_features[candidates])
            kept_positions.append(map_positions[candidates])
            kept_orientations.append(map_orientation[candidates])
            kept_batch.append(torch.full(
                (int(candidates.numel()),),
                seq_idx,
                dtype=torch.long,
                device=map_features.device,
            ))

        if not kept_context:
            return None, None, None, None, None

        map_context = torch.cat(kept_context, dim=0)
        packed_map_positions = torch.cat(kept_positions, dim=0)
        packed_map_orientations = torch.cat(kept_orientations, dim=0)
        packed_map_batch = torch.cat(kept_batch, dim=0)
        map_valid_mask = torch.ones(
            map_context.shape[0],
            dtype=torch.bool,
            device=map_features.device,
        )
        return map_context, packed_map_positions, packed_map_orientations, packed_map_batch, map_valid_mask

    def _token_chunk_world(self, token_ids, agent_types, positions, headings):
        traj = _lookup_token_trajectories(token_ids, self.token_vocab, agent_types)
        endpoint = _lookup_token_endpoints(token_ids, self.token_endpoint_vocab, agent_types)
        smart_traj = torch.cat(
            [traj[:, :self.future_chunk_steps, :, :], endpoint[:, None, :, :]],
            dim=1,
        )
        local_corners = smart_traj[:, 1:1 + self.future_chunk_steps, :, :]
        cos, sin = headings.cos(), headings.sin()
        num_tokens = int(token_ids.shape[0])
        rot = torch.zeros(num_tokens, 2, 2, device=positions.device)
        rot[:, 0, 0] = cos
        rot[:, 0, 1] = sin
        rot[:, 1, 0] = -sin
        rot[:, 1, 1] = cos
        world_corners = torch.bmm(local_corners.reshape(num_tokens, -1, 2), rot)
        world_corners = (
            world_corners.reshape(num_tokens, self.future_chunk_steps, 4, 2)
            + positions[:, None, None, :]
        )
        world = world_corners.mean(dim=2)
        diff_xy = world_corners[:, :, 0, :] - world_corners[:, :, 3, :]
        chunk_heading = torch.atan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
        return world, chunk_heading

    def _refresh_token_geometry(
        self,
        token_ids,
        packed,
        geometry_known_mask=None,
        proposal_token_ids=None,
        proposal_confidence=None,
    ):
        """Estimate each future-token node pose from known tokens or proposals."""
        positions = packed['token_positions'].clone()
        headings = packed['token_headings'].clone()
        geometry_confidence = token_ids.new_zeros(token_ids.shape, dtype=torch.float)
        C = self.num_future_chunks
        agent_start_positions = packed['agent_start_positions']
        agent_start_headings = packed['agent_start_headings']
        agent_types = packed['agent_types_global']

        for _scene_idx, seq_idx, agent_indices in packed['agent_maps']:
            num_scene_agents = int(agent_indices.numel())
            if num_scene_agents == 0:
                continue
            offsets = (
                torch.arange(num_scene_agents, device=token_ids.device).unsqueeze(1) * C
                + torch.arange(C, device=token_ids.device).unsqueeze(0)
            )
            cur_pos = agent_start_positions[agent_indices].clone()
            cur_heading = agent_start_headings[agent_indices].clone()
            cur_types = agent_types[agent_indices]

            for chunk_idx in range(C):
                node_idx = offsets[:, chunk_idx]
                node_has_gt = packed['valid_mask'][seq_idx, node_idx]
                if node_has_gt.any():
                    positions[seq_idx, node_idx[node_has_gt]] = cur_pos[node_has_gt]
                    headings[seq_idx, node_idx[node_has_gt]] = cur_heading[node_has_gt]

                chunk_tokens = token_ids[seq_idx, node_idx]
                known = (
                    node_has_gt
                    & (chunk_tokens >= 0)
                    & (chunk_tokens < self.token_size)
                    & (chunk_tokens != self.mask_token_id)
                )
                if geometry_known_mask is not None:
                    known = known & geometry_known_mask[seq_idx, node_idx].bool()

                proposal_known = torch.zeros_like(known)
                proposal_tokens = None
                if proposal_token_ids is not None and proposal_confidence is not None:
                    proposal_tokens = proposal_token_ids[seq_idx, node_idx]
                    proposal_conf = proposal_confidence[seq_idx, node_idx].to(dtype=torch.float)
                    proposal_known = (
                        node_has_gt
                        & ~known
                        & (proposal_tokens >= 0)
                        & (proposal_tokens < self.token_size)
                        & (proposal_conf > 0.0)
                    )
                    if proposal_known.any():
                        geometry_confidence[seq_idx, node_idx[proposal_known]] = proposal_conf[proposal_known]

                if known.any():
                    geometry_confidence[seq_idx, node_idx[known]] = 1.0

                advance = known | proposal_known
                if not advance.any():
                    continue

                advance_tokens = chunk_tokens.clone()
                if proposal_tokens is not None:
                    advance_tokens[proposal_known] = proposal_tokens[proposal_known]
                world, chunk_heading = self._token_chunk_world(
                    advance_tokens[advance],
                    cur_types[advance],
                    cur_pos[advance],
                    cur_heading[advance],
                )
                cur_pos[advance] = world[:, -1]
                cur_heading[advance] = chunk_heading[:, -1]

        return positions, headings, geometry_confidence

    # -- future token targets ----------------------------------------------

    def _build_future_token_targets(self, data):
        """Extract GT token indices for all agents' future chunks.

        Returns:
            tokens:   [num_agents, num_future_chunks] long
            valid:    [num_agents, num_future_chunks] bool
            generation_agents:   [num_agents] bool, agents to roll out
            supervision_agents:  [num_agents] bool, agents to supervise
        """
        h = self.history_token_steps
        future_slice = slice(h, h + self.num_future_chunks)
        tokens = data['agent']['token_idx'][:, future_slice].long()
        valid = data['agent']['agent_valid_mask'][:, future_slice].bool()
        has_future = valid.any(dim=-1)
        generation_agents = self._generation_agent_mask(data) & has_future
        supervision_agents = self._supervision_agent_mask(data) & generation_agents
        return tokens, valid, generation_agents, supervision_agents

    # -- per-scene packing -------------------------------------------------

    def _pack_diffusion_sequence(self, tokens, valid, agents_ok, loss_agents_ok,
                                 agent_batch, agent_positions, agent_headings,
                                 agent_context, agent_types, agent_shape_embeddings):
        """Pack per-agent future tokens into per-scene padded sequences.

        Returns dict with keys: token_ids [B,L], token_positions [B,L,2],
        token_headings [B,L], chunk_ids [B,L], valid_mask [B,L], agent_context [B,L,D],
        agent_type_ids [B,L], agent_map [(scene,pos)->agent_idx].
        Returns None if no valid agents in batch.
        """
        device = tokens.device
        C = self.num_future_chunks
        G = int(agent_batch.max().item()) + 1 if agent_batch.numel() > 0 else 1

        seq_tokens = []
        seq_geometry = []
        seq_chunk = []
        seq_valid = []
        seq_context = []
        seq_shape = []
        seq_type = []
        seq_loss = []
        agent_maps = []  # list of (scene_id, start_pos, agent_indices_tensor)

        for g in range(G):
            ga = torch.nonzero((agent_batch == g) & agents_ok, as_tuple=False).squeeze(-1)
            if ga.numel() == 0:
                continue

            gt = tokens[ga]        # [n_agents, C]
            gv = valid[ga]         # [n_agents, C]
            gl = loss_agents_ok[ga].bool()
            gp = agent_positions[ga]  # [n_agents, 2]
            gh = agent_headings[ga]  # [n_agents]
            gc = agent_context[ga]  # [n_agents, D]
            gs = agent_shape_embeddings[ga]  # [n_agents, D]
            gtype = agent_types[ga].long().clamp(min=0, max=3)

            flat_t = gt.reshape(-1)                        # [n_agents*C]
            flat_v = gv.reshape(-1)
            flat_l = (gv & gl.unsqueeze(1)).reshape(-1)
            flat_pos = gp.unsqueeze(1).expand(-1, C, -1).reshape(-1, 2)
            flat_head = gh.unsqueeze(1).expand(-1, C).reshape(-1)
            flat_context = gc.unsqueeze(1).expand(-1, C, -1).reshape(-1, gc.shape[-1])
            flat_shape = gs.unsqueeze(1).expand(-1, C, -1).reshape(-1, gs.shape[-1])
            flat_type = gtype.unsqueeze(1).expand(-1, C).reshape(-1)
            flat_agent = ga.unsqueeze(1).expand(-1, C).reshape(-1)
            cids = torch.arange(C, device=device).unsqueeze(0).expand(
                ga.numel(), -1).reshape(-1)

            seq_tokens.append(flat_t)
            seq_geometry.append((flat_pos, flat_head, flat_agent))
            seq_chunk.append(cids)
            seq_valid.append(flat_v)
            seq_context.append(flat_context)
            seq_shape.append(flat_shape)
            seq_type.append(flat_type)
            seq_loss.append(flat_l)
            agent_maps.append((g, len(seq_tokens) - 1, ga))

        if not seq_tokens:
            return None

        max_len = max(s.shape[0] for s in seq_tokens)
        B = len(seq_tokens)

        packed = {
            'token_ids': torch.zeros(B, max_len, dtype=torch.long, device=device),
            'token_positions': torch.zeros(B, max_len, 2, device=device, dtype=torch.float),
            'token_headings': torch.zeros(B, max_len, device=device, dtype=torch.float),
            'token_agent_ids': torch.full((B, max_len), -1, dtype=torch.long, device=device),
            'chunk_ids': torch.zeros(B, max_len, dtype=torch.long, device=device),
            'valid_mask': torch.zeros(B, max_len, dtype=torch.bool, device=device),
            'loss_mask_base': torch.zeros(B, max_len, dtype=torch.bool, device=device),
            'agent_context': torch.zeros(
                B, max_len, agent_context.shape[-1],
                dtype=agent_context.dtype,
                device=device,
            ),
            'agent_shape_embeddings': torch.zeros(
                B, max_len, agent_shape_embeddings.shape[-1],
                dtype=agent_shape_embeddings.dtype,
                device=device,
            ),
            'agent_type_ids': torch.zeros(B, max_len, dtype=torch.long, device=device),
        }
        for i in range(B):
            L = seq_tokens[i].shape[0]
            packed['token_ids'][i, :L] = seq_tokens[i]
            packed['token_positions'][i, :L] = seq_geometry[i][0]
            packed['token_headings'][i, :L] = seq_geometry[i][1]
            packed['token_agent_ids'][i, :L] = seq_geometry[i][2]
            packed['chunk_ids'][i, :L] = seq_chunk[i]
            packed['valid_mask'][i, :L] = seq_valid[i]
            packed['loss_mask_base'][i, :L] = seq_loss[i]
            packed['agent_context'][i, :L] = seq_context[i]
            packed['agent_shape_embeddings'][i, :L] = seq_shape[i]
            packed['agent_type_ids'][i, :L] = seq_type[i]

        # agent_maps: list of (scene_idx, packed_seq_idx, agent_indices)
        packed['agent_maps'] = agent_maps
        packed['agent_start_positions'] = agent_positions
        packed['agent_start_headings'] = agent_headings
        packed['agent_types_global'] = agent_types.long().clamp(min=0, max=3)
        return packed

    # -- training / validation ---------------------------------------------

    def _prepare_batch(self, data):
        if bool(getattr(data, '_smart_diffusion_prepared', False)):
            return data
        data = self.match_token_map(data)
        data = self.sample_pt_pred(data)
        if isinstance(data, Batch):
            data['agent']['av_index'] += data['agent']['ptr'][:-1]
        setattr(data, '_smart_diffusion_prepared', True)
        return data

    def _compute_ntp_loss(self, pred):
        next_token_prob = pred['next_token_prob']
        next_token_idx_gt = pred['next_token_idx_gt']
        next_token_eval_mask = pred['next_token_eval_mask']
        if not next_token_eval_mask.any():
            return next_token_prob.new_zeros(())
        return self.cls_loss(
            next_token_prob[next_token_eval_mask],
            next_token_idx_gt[next_token_eval_mask],
        )

    def _compute_optional_ntp_loss(self, data, ref_tensor):
        if self.ntp_aux_loss_weight <= 0.0:
            return ref_tensor.new_zeros(())
        return self._compute_ntp_loss(self(data))

    def _build_diffusion_inputs(self, data):
        ctx = self.encoder.encode_history_context(data)
        ft, fv, generation_agents, supervision_agents = self._build_future_token_targets(data)
        agent_batch = self._get_agent_batch(data)
        agent_pos = data['agent']['position'][:, self.num_historical_steps - 1, :2].float()
        agent_heading = data['agent']['heading'][:, self.num_historical_steps - 1].float()
        agent_shape = data['agent']['shape']
        if agent_shape.dim() == 3:
            agent_shape = agent_shape[:, self.num_historical_steps - 1, :]
        agent_shape_embeddings = self.encoder.agent_encoder.shape_emb(agent_shape.float())
        agent_context = self._pool_agent_context(
            ctx['x_a_history'],
            ctx['history_token_mask'],
        )

        packed = self._pack_diffusion_sequence(
            ft,
            fv,
            generation_agents,
            supervision_agents,
            agent_batch,
            agent_pos,
            agent_heading,
            agent_context,
            data['agent']['type'],
            agent_shape_embeddings,
        )
        if packed is None:
            return None, None, ft, fv, generation_agents, supervision_agents, agent_batch

        summary = self._pool_scene_summary(ctx['x_a_history'], ctx['history_token_mask'],
                                           agent_batch, data['agent']['type'])

        packed_graph_ids = [m[0] for m in packed['agent_maps']]
        summary = summary[packed_graph_ids]
        map_context, map_positions, map_orientations, map_batch, map_valid_mask = self._pack_map_context(
            data,
            ctx,
            packed,
            agent_pos,
        )
        packed['map_context'] = map_context
        packed['map_positions'] = map_positions
        packed['map_orientations'] = map_orientations
        packed['map_batch'] = map_batch
        packed['map_valid_mask'] = map_valid_mask
        return packed, summary, ft, fv, generation_agents, supervision_agents, agent_batch

    def _decode_diffusion_logits(
        self,
        noisy,
        packed,
        summary,
        t,
        geometry_known_mask,
        proposal_token_ids=None,
        proposal_confidence=None,
    ):
        token_positions, token_headings, geometry_confidence = self._refresh_token_geometry(
            noisy,
            packed,
            geometry_known_mask=geometry_known_mask,
            proposal_token_ids=proposal_token_ids if self.use_proposal_geometry else None,
            proposal_confidence=proposal_confidence if self.use_proposal_geometry else None,
        )
        source_mask = packed['valid_mask'] & (
            geometry_confidence >= self.geometry_confidence_source_threshold
        )

        return self.diffusion_decoder(
            noisy_token_ids=noisy,
            token_positions=token_positions,
            token_headings=token_headings,
            token_agent_ids=packed['token_agent_ids'],
            noisy_token_chunk_ids=packed['chunk_ids'], scene_summary=summary,
            t=t, valid_mask=packed['valid_mask'],
            agent_context=packed['agent_context'],
            agent_type_ids=packed['agent_type_ids'],
            agent_shape_embeddings=packed['agent_shape_embeddings'],
            physical_token_embeddings=self._physical_token_embeddings(noisy, packed['agent_type_ids']),
            map_context=packed.get('map_context'),
            map_positions=packed.get('map_positions'),
            map_orientations=packed.get('map_orientations'),
            map_batch=packed.get('map_batch'),
            map_valid_mask=packed.get('map_valid_mask'),
            geometry_confidence=geometry_confidence,
            temporal_source_mask=source_mask,
            spatial_source_mask=source_mask,
        )

    def _maybe_apply_geometry_dropout(self, geometry_known_mask, reference):
        if self.training and self.geometry_dropout_prob > 0.0:
            geometry_keep = torch.rand_like(reference.float()) >= self.geometry_dropout_prob
            geometry_known_mask = geometry_known_mask & geometry_keep
        return geometry_known_mask

    def _compute_diffusion_loss(self, packed, summary):
        B = summary.shape[0]
        t = self._sample_diffusion_timesteps(B, summary.device)
        sigma_t, mask_prob, dsigma_t = self.noise_schedule(t)

        gt = packed['token_ids']
        valid_mask = packed['valid_mask']
        loss_mask_base = packed.get('loss_mask_base', valid_mask) & valid_mask
        mask = self._sample_training_mask(valid_mask, mask_prob)
        if self.prefix_constrained_training:
            mask = self._apply_prefix_mask_closure(
                mask,
                valid_mask,
                packed['token_agent_ids'],
                packed['chunk_ids'],
            )
            loss_mask = self._prefix_frontier_mask(
                mask,
                valid_mask,
                packed['token_agent_ids'],
                packed['chunk_ids'],
            ) & loss_mask_base
        else:
            loss_mask = mask & loss_mask_base

        noisy = gt.clone()
        noisy[mask] = self.mask_token_id

        proposal_ids = None
        proposal_confidence = None
        proposal_mask = torch.zeros_like(mask)
        proposal_input_acc = gt.new_tensor(0.0, dtype=torch.float)
        should_self_condition = (
            self.training
            and self.use_proposal_geometry
            and mask.any()
            and self.self_condition_prob > 0.0
            and torch.rand((), device=gt.device) < self.self_condition_prob
        )
        if should_self_condition:
            first_geometry_known = self._maybe_apply_geometry_dropout(
                (~mask) & valid_mask,
                gt,
            )
            with torch.no_grad():
                first_logits = self._decode_diffusion_logits(
                    noisy,
                    packed,
                    summary,
                    t,
                    first_geometry_known,
                )
                first_prob = F.softmax(first_logits, dim=-1)
                first_conf, first_pred = first_prob.max(dim=-1)
            proposal_mask = mask & valid_mask
            proposal_ids = first_pred.masked_fill(~proposal_mask, 0)
            proposal_confidence = first_conf.masked_fill(~proposal_mask, 0.0)
            proposal_input_acc = (
                first_pred[proposal_mask] == gt[proposal_mask]
            ).float().mean()

        geometry_known_mask = self._maybe_apply_geometry_dropout(
            (~mask) & valid_mask,
            gt,
        )
        logits = self._decode_diffusion_logits(
            noisy,
            packed,
            summary,
            t,
            geometry_known_mask,
            proposal_token_ids=proposal_ids,
            proposal_confidence=proposal_confidence,
        )

        if loss_mask.any():
            log_p = F.log_softmax(logits, dim=-1)
            nll = -log_p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
            weight = (dsigma_t / torch.expm1(sigma_t)).unsqueeze(-1).expand_as(nll)
            supervision_weight = loss_mask.to(dtype=nll.dtype)
            loss = (weight * nll * supervision_weight).sum()
            loss = loss / loss_mask_base.to(dtype=nll.dtype).sum().clamp_min(1.0)
            acc = (logits[loss_mask].argmax(-1) == gt[loss_mask]).float().mean()
        else:
            loss = logits.sum() * 0.0
            acc = logits.new_zeros(())

        if self.training:
            valid_count = valid_mask.to(dtype=torch.float).sum().clamp_min(1.0)
            proposal_frac = proposal_mask.to(dtype=torch.float).sum() / valid_count
            if proposal_mask.any():
                proposal_acc = (
                    logits[proposal_mask].argmax(-1) == gt[proposal_mask]
                ).float().mean()
            else:
                proposal_acc = logits.new_zeros(())
            self.log('train_self_condition_frac', proposal_frac,
                     prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_self_condition_input_acc', proposal_input_acc,
                     prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            self.log('train_self_condition_acc', proposal_acc,
                     prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            for chunk_idx in range(min(self.num_future_chunks, 16)):
                chunk_mask = packed['chunk_ids'] == chunk_idx
                chunk_total = (loss_mask_base & chunk_mask).to(dtype=torch.float).sum().clamp_min(1.0)
                chunk_covered = (loss_mask & chunk_mask).to(dtype=torch.float).sum() / chunk_total
                self.log(f'train_loss_chunk_{chunk_idx:02d}', chunk_covered,
                         prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
        return loss, acc

    def training_step(self, data, batch_idx):
        data = self._prepare_batch(data)
        packed, summary, _ft, _fv, _generation_agents, _supervision_agents, _agent_batch = self._build_diffusion_inputs(data)
        if packed is None:
            zero_loss = self._zero_connected_loss()
            self.log('train_empty_diffusion_batch', zero_loss.detach().new_ones(()),
                     prog_bar=False, on_step=True, on_epoch=True, batch_size=1)
            return zero_loss

        diffusion_loss, mask_acc = self._compute_diffusion_loss(packed, summary)
        ntp_loss = self._compute_optional_ntp_loss(data, diffusion_loss)
        loss = diffusion_loss + self.ntp_aux_loss_weight * ntp_loss

        self.log('train_empty_diffusion_batch', loss.new_zeros(()),
                 prog_bar=False, on_step=True, on_epoch=True, batch_size=1)

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=1)
        self.log('diffusion_loss', diffusion_loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=1)
        self.log('ntp_loss', ntp_loss, prog_bar=True, on_step=True, on_epoch=True,
                 batch_size=1)
        self.log('train_mask_acc', mask_acc, on_step=True, on_epoch=True, batch_size=1)

        return loss

    def validation_step(self, data, batch_idx):
        data = self._prepare_batch(data)
        packed, summary, _ft, _fv, _generation_agents, _supervision_agents, _agent_batch = self._build_diffusion_inputs(data)
        if packed is None:
            self.log('val_empty_diffusion_batch', data['agent']['position'].new_ones(()),
                     prog_bar=False, on_step=False, on_epoch=True,
                     batch_size=1, sync_dist=True)
            return

        diffusion_loss, mask_acc = self._compute_diffusion_loss(packed, summary)
        ntp_loss = self._compute_optional_ntp_loss(data, diffusion_loss)
        total_loss = diffusion_loss + self.ntp_aux_loss_weight * ntp_loss

        self.log('val_empty_diffusion_batch', total_loss.new_zeros(()),
                 prog_bar=False, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)

        self.log('val_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_total_loss', total_loss, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_diffusion_loss', diffusion_loss, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_ntp_loss', ntp_loss, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)
        self.log('val_mask_acc', mask_acc, prog_bar=True, on_step=False, on_epoch=True,
                 batch_size=1, sync_dist=True)

        # Full inference for ADE/FDE (only first 2 batches to limit cost)
        if self.inference_token and batch_idx < self.diffusion_eval_batches:
            pred_out = self.inference(data)
            if pred_out is not None:
                em = self._metric_agent_mask(data)
                if not em.any():
                    return
                pred_valid = pred_out.get('pred_valid_mask', pred_out['valid_mask'])
                eval_valid = pred_out['valid_mask'] & pred_valid
                self.minADE.update(pred=pred_out['pred_traj'][em],
                                   target=pred_out['gt'][em],
                                   valid_mask=eval_valid[em])
                self.minFDE.update(pred=pred_out['pred_traj'][em],
                                   target=pred_out['gt'][em],
                                   valid_mask=eval_valid[em])
                self.log('val_minADE', self.minADE, prog_bar=True, on_step=False,
                         on_epoch=True, batch_size=1)
                self.log('val_minFDE', self.minFDE, prog_bar=True, on_step=False,
                         on_epoch=True, batch_size=1)
                shapes = data['agent']['shape']
                if shapes.dim() == 3:
                    shapes = shapes[:, self.num_historical_steps - 1, :]
                self.conflict_rate.update(
                    pred_out['pred_traj'][em],
                    pred_out['pred_head'][em],
                    shapes[em],
                    eval_valid[em],
                )
                self.interaction_consistency.update(
                    pred_out['pred_traj'][em],
                    pred_out['gt'][em],
                    eval_valid[em],
                )
                self.log('val_conflict_rate', self.conflict_rate, prog_bar=False,
                         on_step=False, on_epoch=True, batch_size=1)
                self.log('val_interaction_consistency', self.interaction_consistency,
                         prog_bar=False, on_step=False, on_epoch=True, batch_size=1)

    # -- diffusion sampling ------------------------------------------------

    @torch.no_grad()
    def _diffusion_sample(self, summary, token_positions, token_headings, token_agent_ids,
                          chunk_ids, valid_mask, agent_context, agent_type_ids,
                          map_context=None, map_positions=None, map_orientations=None,
                          map_batch=None, map_valid_mask=None, agent_shape_embeddings=None,
                          packed=None, return_trace=False):
        """Iteratively denoise masked future tokens with optional remasking.

        Returns: ([B, L] long sampled token IDs, [B, L] token confidence).
        """
        B, L = valid_mask.shape
        device = summary.device
        S = self.diffusion_num_steps

        x = torch.full((B, L), self.mask_token_id, dtype=torch.long, device=device)
        mask = valid_mask.clone()
        confidence_out = summary.new_zeros((B, L))
        proposal_ids = torch.zeros((B, L), dtype=torch.long, device=device)
        proposal_confidence = summary.new_zeros((B, L))
        trace = []

        def decode_probs(t_value):
            nonlocal token_positions, token_headings
            t_batch = torch.full((B,), t_value, device=device)
            if packed is not None:
                geometry_known = valid_mask & (x != self.mask_token_id)
                logits = self._decode_diffusion_logits(
                    x,
                    packed,
                    summary,
                    t_batch,
                    geometry_known,
                    proposal_token_ids=proposal_ids,
                    proposal_confidence=proposal_confidence,
                )
            else:
                logits = self.diffusion_decoder(
                    noisy_token_ids=x,
                    token_positions=token_positions,
                    token_headings=token_headings,
                    token_agent_ids=token_agent_ids,
                    noisy_token_chunk_ids=chunk_ids, scene_summary=summary,
                    t=t_batch, valid_mask=valid_mask,
                    agent_context=agent_context,
                    agent_type_ids=agent_type_ids,
                    agent_shape_embeddings=agent_shape_embeddings,
                    physical_token_embeddings=self._physical_token_embeddings(x, agent_type_ids),
                    map_context=map_context,
                    map_positions=map_positions,
                    map_orientations=map_orientations,
                    map_batch=map_batch,
                    map_valid_mask=map_valid_mask,
                )
            return F.softmax(logits / self.remask_confidence_temperature, dim=-1)

        def store_sample(sample_mask, sampled_ids, sampled_confidence):
            if not sample_mask.any():
                return
            x[sample_mask] = sampled_ids
            confidence_out[sample_mask] = sampled_confidence
            proposal_ids[sample_mask] = sampled_ids
            proposal_confidence[sample_mask] = sampled_confidence

        for step in range(S):
            t_cur = max(1.0 - step / S, self.min_t)
            t_next = max(1.0 - (step + 1) / S, self.min_t)
            probs = decode_probs(t_cur)

            if step == S - 1 or t_next <= self.min_t:
                masked_before = int(mask.sum().item())
                final_rounds = 0
                final_sampled = 0
                if self.prefix_constrained_sampling:
                    while mask.any():
                        sample_mask = self._prefix_frontier_mask(
                            mask,
                            valid_mask,
                            token_agent_ids,
                            chunk_ids,
                        )
                        if not sample_mask.any():
                            sample_mask = mask.clone()
                        final_prob, final_ids = probs[sample_mask].max(dim=-1)
                        store_sample(sample_mask, final_ids, final_prob)
                        final_sampled += int(sample_mask.sum().item())
                        mask = valid_mask & (x == self.mask_token_id)
                        final_rounds += 1
                        if not mask.any():
                            break
                        if final_rounds > self.num_future_chunks + 1:
                            final_prob, final_ids = probs[mask].max(dim=-1)
                            store_sample(mask, final_ids, final_prob)
                            final_sampled += int(mask.sum().item())
                            mask.zero_()
                            break
                        probs = decode_probs(t_cur)
                elif mask.any():
                    final_sampled = int(mask.sum().item())
                    final_rounds = 1
                    final_prob, final_ids = probs[mask].max(dim=-1)
                    store_sample(mask, final_ids, final_prob)
                if return_trace:
                    trace.append({
                        'step': step,
                        'masked_before': masked_before,
                        'masked_after': 0,
                        'remasked': 0,
                        'frontier': final_sampled,
                        'final_rounds': final_rounds,
                    })
                mask.zero_()
            else:
                _, p_next, _ = self.noise_schedule(torch.tensor(t_next, device=device))
                p_next_value = float(p_next.item())
                masked_before = int(mask.sum().item())

                sample_mask = (
                    self._prefix_frontier_mask(
                        mask,
                        valid_mask,
                        token_agent_ids,
                        chunk_ids,
                    )
                    if self.prefix_constrained_sampling
                    else mask
                )

                if sample_mask.any():
                    sampled = torch.multinomial(
                        probs[sample_mask].clamp(min=1e-10),
                        1,
                    ).squeeze(-1)
                    selected_prob = probs[sample_mask].gather(
                        -1,
                        sampled.unsqueeze(-1),
                    ).squeeze(-1)
                    store_sample(sample_mask, sampled, selected_prob)

                token_confidence = confidence_out.clone().masked_fill(~valid_mask, float('inf'))

                if self.prefix_constrained_sampling:
                    next_mask = valid_mask & (x == self.mask_token_id)
                    for batch_idx in range(B):
                        valid_count = int(valid_mask[batch_idx].sum().item())
                        if valid_count == 0:
                            continue
                        target_masked = int(round(p_next_value * valid_count))
                        target_masked = min(max(target_masked, 0), valid_count)
                        already_masked = int(next_mask[batch_idx].sum().item())
                        additional_masked = target_masked - already_masked
                        if additional_masked <= 0 or not self.remask_sampling:
                            continue
                        candidate_mask = (
                            valid_mask[batch_idx]
                            & (x[batch_idx] != self.mask_token_id)
                        )
                        candidate_indices = torch.nonzero(
                            candidate_mask,
                            as_tuple=False,
                        ).squeeze(-1)
                        if candidate_indices.numel() == 0:
                            continue
                        additional_masked = min(additional_masked, int(candidate_indices.numel()))
                        low_conf_local = torch.topk(
                            token_confidence[batch_idx, candidate_indices],
                            k=additional_masked,
                            largest=False,
                        ).indices
                        next_mask[batch_idx, candidate_indices[low_conf_local]] = True
                    next_mask = self._apply_prefix_mask_closure(
                        next_mask,
                        valid_mask,
                        token_agent_ids,
                        chunk_ids,
                    )
                else:
                    next_mask = torch.zeros_like(mask)
                    for batch_idx in range(B):
                        valid_count = int(valid_mask[batch_idx].sum().item())
                        if valid_count == 0:
                            continue
                        candidate_mask = valid_mask[batch_idx] if self.remask_sampling else mask[batch_idx]
                        candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(-1)
                        if candidate_indices.numel() == 0:
                            continue
                        target_masked = int(round(p_next_value * valid_count))
                        target_masked = min(max(target_masked, 0), int(candidate_indices.numel()))
                        if target_masked <= 0:
                            continue
                        low_conf_local = torch.topk(
                            token_confidence[batch_idx, candidate_indices],
                            k=target_masked,
                            largest=False,
                        ).indices
                        next_mask[batch_idx, candidate_indices[low_conf_local]] = True

                remasked = next_mask & valid_mask
                newly_remasked = remasked & (x != self.mask_token_id)
                x[remasked] = self.mask_token_id
                confidence_out[remasked] = 0.0
                if return_trace:
                    trace.append({
                        'step': step,
                        'masked_before': masked_before,
                        'masked_after': int(next_mask.sum().item()),
                        'remasked': int(newly_remasked.sum().item()),
                        'frontier': int(sample_mask.sum().item()),
                    })
                mask = next_mask

        out = x.masked_fill(~valid_mask, 0)
        conf = confidence_out.masked_fill(~valid_mask, 0.0)
        if return_trace:
            return out, conf, trace
        return out, conf

    # -- full inference ----------------------------------------------------

    @torch.no_grad()
    def inference(self, data):
        """Diffusion sampling → chain decoding → predicted trajectories."""
        data = self._prepare_batch(data)
        packed, summary, ft, fv, generation_agents, _supervision_agents, agent_batch = self._build_diffusion_inputs(data)
        if packed is None:
            return None

        sampled_ids, sampled_confidence = self._diffusion_sample(
            summary=summary,
            token_positions=packed['token_positions'],
            token_headings=packed['token_headings'],
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'], valid_mask=packed['valid_mask'],
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

        return self._decode_trajectories(data, sampled_ids, packed, generation_agents,
                                         agent_batch, ft, fv, sampled_confidence)

    @torch.no_grad()
    def _decode_trajectories(self, data, sampled_ids, packed, agents_ok,
                             agent_batch, gt_tokens, gt_valid, sampled_confidence=None):
        """Unpack per-scene tokens → per-agent trajectories via chain decode."""
        device = sampled_ids.device
        C = self.num_future_chunks
        shift = self.encoder.agent_encoder.shift
        num_agents = agents_ok.shape[0]
        agent_types = data['agent']['type']

        # Unpack: for each scene, map sequence positions back to agent indices
        per_agent_tokens = torch.full((num_agents, C), -1, dtype=torch.long, device=device)
        per_agent_confidence = torch.zeros(num_agents, C, device=device)
        for _g, seq_idx, ag_indices in packed['agent_maps']:
            seq = sampled_ids[seq_idx]      # [L]
            seq_confidence = None if sampled_confidence is None else sampled_confidence[seq_idx]
            for ai_local, ag_idx in enumerate(ag_indices.tolist()):
                start = ai_local * C
                end = start + C
                token_slice = seq[start:end]
                per_agent_tokens[ag_idx] = token_slice
                if seq_confidence is not None:
                    per_agent_confidence[ag_idx] = seq_confidence[start:end]

        # Chain decode: for each future chunk, apply token trajectory
        # relative to agent's current pose, then advance pose
        hist_pose = data['agent']['position'][:, self.num_historical_steps - 1, :2]
        hist_heading = data['agent']['heading'][:, self.num_historical_steps - 1]

        pred_traj = torch.zeros(num_agents, self.num_future_steps, 2, device=device)
        pred_head = torch.zeros(num_agents, self.num_future_steps, device=device)
        pred_valid_mask = torch.zeros(num_agents, self.num_future_steps,
                                      dtype=torch.bool, device=device)
        pred_token_ids = torch.full((num_agents, C), -1, dtype=torch.long, device=device)

        pos = hist_pose.clone()
        heading = hist_heading.clone()

        for c in range(C):
            frame_start = c * shift
            frame_end = min(frame_start + shift, self.num_future_steps)
            n_frames = frame_end - frame_start

            chunk_tokens = per_agent_tokens[:, c]
            decode_mask = (
                agents_ok
                & gt_valid[:, c].bool()
                & (chunk_tokens >= 0)
                & (chunk_tokens < self.token_size)
            )
            if not decode_mask.any():
                continue

            active_tokens = chunk_tokens[decode_mask]
            active_types = agent_types[decode_mask]
            world, chunk_heading = self._token_chunk_world(
                active_tokens,
                active_types,
                pos[decode_mask],
                heading[decode_mask],
            )

            pred_traj[decode_mask, frame_start:frame_end] = world[:, :n_frames]
            pred_head[decode_mask, frame_start:frame_end] = chunk_heading[:, :n_frames]
            pred_valid_mask[decode_mask, frame_start:frame_end] = True
            pred_token_ids[decode_mask, c] = active_tokens

            if c < C - 1:
                pos[decode_mask] = world[:, -1]
                heading[decode_mask] = chunk_heading[:, -1]

        gt_pos = data['agent']['position'][:, self.num_historical_steps:, :2]
        gt_val = data['agent']['valid_mask'][:, self.num_historical_steps:].bool().clone()
        try:
            gt_val[data['agent']['category'].long() != 3] = False
        except Exception:
            pass

        return {
            'pos_a': torch.cat([hist_pose.unsqueeze(1), pred_traj], dim=1),
            'head_a': torch.cat([hist_heading.unsqueeze(1), pred_head], dim=1),
            'gt': gt_pos,
            'valid_mask': gt_val,
            'pred_valid_mask': pred_valid_mask,
            'pred_traj': pred_traj,
            'pred_head': pred_head,
            'next_token_idx': pred_token_ids,
            'next_token_idx_gt': gt_tokens,
            'next_token_eval_mask': gt_valid,
            'pred_prob': per_agent_confidence,
        }
