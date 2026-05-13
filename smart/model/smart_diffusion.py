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
        self.diffusion_num_steps = int(getattr(diffusion_cfg, 'num_steps', 16))
        self.diffusion_num_layers = int(getattr(diffusion_cfg, 'num_layers', 2))
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
        if self._token_vocab_cache is None:
            dev = next(self.parameters()).device
            self._token_vocab_cache = {
                k: torch.from_numpy(v).clone().to(device=dev, dtype=torch.float)
                for k, v in self.encoder.agent_encoder.trajectory_token_all.items()
            }
        return self._token_vocab_cache

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

    def _sample_diffusion_timesteps(self, batch_size, device):
        """Low-discrepancy stratified samples in [min_t, 1]."""
        if batch_size <= 1:
            unit_t = torch.rand(batch_size, device=device)
        else:
            strata = torch.arange(batch_size, device=device, dtype=torch.float)
            unit_t = (strata + torch.rand(batch_size, device=device)) / float(batch_size)
            unit_t = unit_t[torch.randperm(batch_size, device=device)]
        return self.min_t + (1.0 - self.min_t) * unit_t

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
        local_corners = traj[:, 1:1 + self.future_chunk_steps, :, :]
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

    def _refresh_token_geometry(self, token_ids, packed, geometry_known_mask=None):
        """Estimate each future-token node pose from currently unmasked tokens."""
        positions = packed['token_positions'].clone()
        headings = packed['token_headings'].clone()
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
                if not known.any():
                    continue

                world, chunk_heading = self._token_chunk_world(
                    chunk_tokens[known],
                    cur_types[known],
                    cur_pos[known],
                    cur_heading[known],
                )
                cur_pos[known] = world[:, -1]
                cur_heading[known] = chunk_heading[:, -1]

        return positions, headings

    # -- future token targets ----------------------------------------------

    def _build_future_token_targets(self, data):
        """Extract GT token indices for all agents' future chunks.

        Returns:
            tokens:   [num_agents, num_future_chunks] long
            valid:    [num_agents, num_future_chunks] bool
            agents:   [num_agents] bool — non-bg agents with valid history
        """
        h = self.history_token_steps
        future_slice = slice(h, h + self.num_future_chunks)
        tokens = data['agent']['token_idx'][:, future_slice].long()
        valid = data['agent']['agent_valid_mask'][:, future_slice].bool()
        agents_ok = (
            data['agent']['valid_mask'][:, self.num_historical_steps - 1].bool()
            & (data['agent']['type'] != 3)
            & valid.any(dim=-1)
        )
        return tokens, valid, agents_ok

    # -- per-scene packing -------------------------------------------------

    def _pack_diffusion_sequence(self, tokens, valid, agents_ok, agent_batch,
                                 agent_positions, agent_headings, agent_context,
                                 agent_types, agent_shape_embeddings):
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
        agent_maps = []  # list of (scene_id, start_pos, agent_indices_tensor)

        for g in range(G):
            ga = torch.nonzero((agent_batch == g) & agents_ok, as_tuple=False).squeeze(-1)
            if ga.numel() == 0:
                continue

            gt = tokens[ga]        # [n_agents, C]
            gv = valid[ga]         # [n_agents, C]
            gp = agent_positions[ga]  # [n_agents, 2]
            gh = agent_headings[ga]  # [n_agents]
            gc = agent_context[ga]  # [n_agents, D]
            gs = agent_shape_embeddings[ga]  # [n_agents, D]
            gtype = agent_types[ga].long().clamp(min=0, max=3)

            flat_t = gt.reshape(-1)                        # [n_agents*C]
            flat_v = gv.reshape(-1)
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
        ft, fv, agents_ok = self._build_future_token_targets(data)
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
            agents_ok,
            agent_batch,
            agent_pos,
            agent_heading,
            agent_context,
            data['agent']['type'],
            agent_shape_embeddings,
        )
        if packed is None:
            return None, None, ft, fv, agents_ok, agent_batch

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
        return packed, summary, ft, fv, agents_ok, agent_batch

    def _compute_diffusion_loss(self, packed, summary):
        B = summary.shape[0]
        t = self._sample_diffusion_timesteps(B, summary.device)
        sigma_t, mask_prob, dsigma_t = self.noise_schedule(t)

        gt = packed['token_ids']
        mask = (torch.rand_like(gt.float()) < mask_prob.unsqueeze(-1)) & packed['valid_mask']
        noisy = gt.clone()
        noisy[mask] = self.mask_token_id
        geometry_known_mask = (~mask) & packed['valid_mask']
        if self.training and self.geometry_dropout_prob > 0.0:
            geometry_keep = torch.rand_like(gt.float()) >= self.geometry_dropout_prob
            geometry_known_mask = geometry_known_mask & geometry_keep
        token_positions, token_headings = self._refresh_token_geometry(
            noisy,
            packed,
            geometry_known_mask=geometry_known_mask,
        )

        logits = self.diffusion_decoder(
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
        )

        if mask.any():
            log_p = F.log_softmax(logits, dim=-1)
            nll = -log_p.gather(-1, gt.unsqueeze(-1)).squeeze(-1)
            weight = (dsigma_t / torch.expm1(sigma_t)).unsqueeze(-1).expand_as(nll)
            loss = (weight * nll * mask.to(dtype=nll.dtype)).sum()
            loss = loss / packed['valid_mask'].to(dtype=nll.dtype).sum().clamp_min(1.0)
            acc = (logits[mask].argmax(-1) == gt[mask]).float().mean()
        else:
            loss = logits.sum() * 0.0
            acc = logits.new_zeros(())
        return loss, acc

    def training_step(self, data, batch_idx):
        data = self._prepare_batch(data)
        packed, summary, _ft, _fv, _agents_ok, _agent_batch = self._build_diffusion_inputs(data)
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
        packed, summary, _ft, _fv, _agents_ok, _agent_batch = self._build_diffusion_inputs(data)
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
                em = (
                    data['agent']['valid_mask'][:, self.num_historical_steps - 1]
                    & (data['agent']['type'] != 3)
                )
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
                          packed=None):
        """Iteratively denoise masked future tokens (MDLM-style).

        Returns: ([B, L] long sampled token IDs, [B, L] token confidence).
        """
        B, L = valid_mask.shape
        device = summary.device
        S = self.diffusion_num_steps

        x = torch.full((B, L), self.mask_token_id, dtype=torch.long, device=device)
        mask = valid_mask.clone()
        confidence_out = summary.new_zeros((B, L))

        for step in range(S):
            t_cur = max(1.0 - step / S, self.min_t)
            t_next = max(1.0 - (step + 1) / S, self.min_t)
            t_batch = torch.full((B,), t_cur, device=device)
            if packed is not None:
                token_positions, token_headings = self._refresh_token_geometry(x, packed)

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
            probs = F.softmax(logits, dim=-1)

            if step == S - 1 or t_next <= self.min_t:
                if mask.any():
                    final_prob, final_ids = probs[mask].max(dim=-1)
                    x[mask] = final_ids
                    confidence_out[mask] = final_prob
                mask.zero_()
            else:
                _, p_next, _ = self.noise_schedule(torch.tensor(t_next, device=device))
                p_next_value = float(p_next.item())
                confidence = probs.max(dim=-1).values
                for batch_idx in range(B):
                    valid_count = int(valid_mask[batch_idx].sum().item())
                    if valid_count == 0:
                        continue
                    current_masked = int(mask[batch_idx].sum().item())
                    target_masked = int(round(p_next_value * valid_count))
                    to_unmask = max(0, current_masked - target_masked)
                    if to_unmask <= 0:
                        continue

                    candidate_indices = torch.nonzero(mask[batch_idx], as_tuple=False).squeeze(-1)
                    if candidate_indices.numel() == 0:
                        continue
                    k = min(to_unmask, int(candidate_indices.numel()))
                    topk_local = torch.topk(confidence[batch_idx, candidate_indices], k=k).indices
                    selected = candidate_indices[topk_local]
                    sampled = torch.multinomial(
                        probs[batch_idx, selected].clamp(min=1e-10),
                        1,
                    ).squeeze(-1)
                    x[batch_idx, selected] = sampled
                    selected_prob = probs[batch_idx, selected].gather(
                        -1,
                        sampled.unsqueeze(-1),
                    ).squeeze(-1)
                    confidence_out[batch_idx, selected] = selected_prob
                    mask[batch_idx, selected] = False

        return x.masked_fill(~valid_mask, 0), confidence_out.masked_fill(~valid_mask, 0.0)

    # -- full inference ----------------------------------------------------

    @torch.no_grad()
    def inference(self, data):
        """Diffusion sampling → chain decoding → predicted trajectories."""
        data = self._prepare_batch(data)
        packed, summary, ft, fv, agents_ok, agent_batch = self._build_diffusion_inputs(data)
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

        return self._decode_trajectories(data, sampled_ids, packed, agents_ok,
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
        gt_val = data['agent']['valid_mask'][:, self.num_historical_steps:]

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
