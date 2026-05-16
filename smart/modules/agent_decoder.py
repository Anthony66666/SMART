import pickle
from typing import Dict, Mapping, Optional
import torch
import torch.nn as nn
from smart.layers import MLPLayer
from smart.layers.attention_layer import AttentionLayer
from smart.layers.fourier_embedding import FourierEmbedding, MLPEmbedding
from torch_cluster import radius, radius_graph
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import dense_to_sparse, subgraph
from smart.utils import angle_between_2d_vectors, weight_init, wrap_angle
import math


def cal_polygon_contour(x, y, theta, width, length):
    left_front_x = x + 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_front_y = y + 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_front = (left_front_x, left_front_y)

    right_front_x = x + 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_front_y = y + 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_front = (right_front_x, right_front_y)

    right_back_x = x - 0.5 * length * math.cos(theta) + 0.5 * width * math.sin(theta)
    right_back_y = y - 0.5 * length * math.sin(theta) - 0.5 * width * math.cos(theta)
    right_back = (right_back_x, right_back_y)

    left_back_x = x - 0.5 * length * math.cos(theta) - 0.5 * width * math.sin(theta)
    left_back_y = y - 0.5 * length * math.sin(theta) + 0.5 * width * math.cos(theta)
    left_back = (left_back_x, left_back_y)
    polygon_contour = [left_front, right_front, right_back, left_back]

    return polygon_contour


class SMARTAgentDecoder(nn.Module):

    def __init__(self,
                 dataset: str,
                 input_dim: int,
                 hidden_dim: int,
                 num_historical_steps: int,
                 time_span: Optional[int],
                 pl2a_radius: float,
                 a2a_radius: float,
                 num_freq_bands: int,
                 num_layers: int,
                 num_heads: int,
                 head_dim: int,
                 dropout: float,
                 token_data: Dict,
                 token_size=512) -> None:
        super(SMARTAgentDecoder, self).__init__()
        self.dataset = dataset
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.time_span = time_span if time_span is not None else num_historical_steps
        self.pl2a_radius = pl2a_radius
        self.a2a_radius = a2a_radius
        self.num_freq_bands = num_freq_bands
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout = dropout

        input_dim_x_a = 2
        input_dim_r_t = 4
        input_dim_r_pt2a = 3
        input_dim_r_a2a = 3
        input_dim_token = 8

        self.type_a_emb = nn.Embedding(4, hidden_dim)
        self.shape_emb = MLPLayer(3, hidden_dim, hidden_dim)

        self.x_a_emb = FourierEmbedding(input_dim=input_dim_x_a, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_t_emb = FourierEmbedding(input_dim=input_dim_r_t, hidden_dim=hidden_dim, num_freq_bands=num_freq_bands)
        self.r_pt2a_emb = FourierEmbedding(input_dim=input_dim_r_pt2a, hidden_dim=hidden_dim,
                                           num_freq_bands=num_freq_bands)
        self.r_a2a_emb = FourierEmbedding(input_dim=input_dim_r_a2a, hidden_dim=hidden_dim,
                                          num_freq_bands=num_freq_bands)
        self.token_emb_veh = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_ped = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.token_emb_cyc = MLPEmbedding(input_dim=input_dim_token, hidden_dim=hidden_dim)
        self.fusion_emb = MLPEmbedding(input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim)

        self.t_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.pt2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=True, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.a2a_attn_layers = nn.ModuleList(
            [AttentionLayer(hidden_dim=hidden_dim, num_heads=num_heads, head_dim=head_dim, dropout=dropout,
                            bipartite=False, has_pos_emb=True) for _ in range(num_layers)]
        )
        self.token_size = token_size
        self.token_predict_head = MLPLayer(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                           output_dim=self.token_size)
        self.trajectory_token = token_data['token']
        self.trajectory_token_traj = token_data['traj']
        self.trajectory_token_all = token_data['token_all']
        self.apply(weight_init)
        self.shift = 5
        self.beam_size = 5
        self.hist_mask = True

    def transform_rel(self, token_traj, prev_pos, prev_heading=None):
        if prev_heading is None:
            diff_xy = prev_pos[:, :, -1, :] - prev_pos[:, :, -2, :]
            prev_heading = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])

        num_agent, num_step, traj_num, traj_dim = token_traj.shape
        cos, sin = prev_heading.cos(), prev_heading.sin()
        rot_mat = torch.zeros((num_agent, num_step, 2, 2), device=prev_heading.device)
        rot_mat[:, :, 0, 0] = cos
        rot_mat[:, :, 0, 1] = -sin
        rot_mat[:, :, 1, 0] = sin
        rot_mat[:, :, 1, 1] = cos
        agent_diff_rel = torch.bmm(token_traj.view(-1, traj_num, 2), rot_mat.view(-1, 2, 2)).view(num_agent, num_step, traj_num, traj_dim)
        agent_pred_rel = agent_diff_rel + prev_pos[:, :, -1:, :]
        return agent_pred_rel

    def agent_token_embedding(self, data, agent_category, agent_token_index, pos_a, head_vector_a, inference=False):
        num_agent, num_step, traj_dim = pos_a.shape
        motion_vector_a = torch.cat([pos_a.new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     pos_a[:, 1:] - pos_a[:, :-1]], dim=1)

        agent_type = data['agent']['type']
        veh_mask = (agent_type == 0)
        cyc_mask = (agent_type == 2)
        ped_mask = (agent_type == 1)
        trajectory_token_veh = torch.from_numpy(self.trajectory_token['veh']).clone().to(pos_a.device).to(torch.float)
        self.agent_token_emb_veh = self.token_emb_veh(trajectory_token_veh.view(trajectory_token_veh.shape[0], -1))
        trajectory_token_ped = torch.from_numpy(self.trajectory_token['ped']).clone().to(pos_a.device).to(torch.float)
        self.agent_token_emb_ped = self.token_emb_ped(trajectory_token_ped.view(trajectory_token_ped.shape[0], -1))
        trajectory_token_cyc = torch.from_numpy(self.trajectory_token['cyc']).clone().to(pos_a.device).to(torch.float)
        self.agent_token_emb_cyc = self.token_emb_cyc(trajectory_token_cyc.view(trajectory_token_cyc.shape[0], -1))

        if inference:
            agent_token_traj_all = torch.zeros((num_agent, self.token_size, self.shift + 1, 4, 2), device=pos_a.device)
            trajectory_token_all_veh = torch.from_numpy(self.trajectory_token_all['veh']).clone().to(pos_a.device).to(
                torch.float)
            trajectory_token_all_ped = torch.from_numpy(self.trajectory_token_all['ped']).clone().to(pos_a.device).to(
                torch.float)
            trajectory_token_all_cyc = torch.from_numpy(self.trajectory_token_all['cyc']).clone().to(pos_a.device).to(
                torch.float)
            agent_token_traj_all[veh_mask] = torch.cat(
                [trajectory_token_all_veh[:, :self.shift], trajectory_token_veh[:, None, ...]], dim=1)
            agent_token_traj_all[ped_mask] = torch.cat(
                [trajectory_token_all_ped[:, :self.shift], trajectory_token_ped[:, None, ...]], dim=1)
            agent_token_traj_all[cyc_mask] = torch.cat(
                [trajectory_token_all_cyc[:, :self.shift], trajectory_token_cyc[:, None, ...]], dim=1)

        agent_token_emb = torch.zeros((num_agent, num_step, self.hidden_dim), device=pos_a.device)
        agent_token_emb[veh_mask] = self.agent_token_emb_veh[agent_token_index[veh_mask]]
        agent_token_emb[ped_mask] = self.agent_token_emb_ped[agent_token_index[ped_mask]]
        agent_token_emb[cyc_mask] = self.agent_token_emb_cyc[agent_token_index[cyc_mask]]

        agent_token_traj = torch.zeros((num_agent, num_step, self.token_size, 4, 2), device=pos_a.device)
        agent_token_traj[veh_mask] = trajectory_token_veh
        agent_token_traj[ped_mask] = trajectory_token_ped
        agent_token_traj[cyc_mask] = trajectory_token_cyc

        vel = data['agent']['token_velocity']

        categorical_embs = [
            self.type_a_emb(data['agent']['type'].long()).repeat_interleave(repeats=num_step,
                                                                            dim=0),

            self.shape_emb(data['agent']['shape'][:, self.num_historical_steps - 1, :]).repeat_interleave(
                repeats=num_step,
                dim=0)
        ]
        feature_a = torch.stack(
            [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2]),
             ], dim=-1)

        x_a = self.x_a_emb(continuous_inputs=feature_a.view(-1, feature_a.size(-1)),
                           categorical_embs=categorical_embs)
        x_a = x_a.view(-1, num_step, self.hidden_dim)

        feat_a = torch.cat((agent_token_emb, x_a), dim=-1)
        feat_a = self.fusion_emb(feat_a)

        if inference:
            return feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs
        else:
            return feat_a, agent_token_traj

    def agent_predict_next(self, data, agent_category, feat_a):
        num_agent, num_step, traj_dim = data['agent']['token_pos'].shape
        agent_type = data['agent']['type']
        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3
        token_res = torch.zeros((num_agent, num_step, self.token_size), device=agent_category.device)
        token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
        token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
        token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])
        return token_res

    def agent_predict_next_inf(self, data, agent_category, feat_a):
        num_agent, traj_dim = feat_a.shape
        agent_type = data['agent']['type']

        veh_mask = (agent_type == 0)  # * agent_category==3
        cyc_mask = (agent_type == 2)  # * agent_category==3
        ped_mask = (agent_type == 1)  # * agent_category==3

        token_res = torch.zeros((num_agent, self.token_size), device=agent_category.device)
        token_res[veh_mask] = self.token_predict_head(feat_a[veh_mask])
        token_res[cyc_mask] = self.token_predict_cyc_head(feat_a[cyc_mask])
        token_res[ped_mask] = self.token_predict_walker_head(feat_a[ped_mask])

        return token_res

    def build_temporal_edge(self, pos_a, head_a, head_vector_a, num_agent, mask, inference_mask=None):
        pos_t = pos_a.reshape(-1, self.input_dim)
        head_t = head_a.reshape(-1)
        head_vector_t = head_vector_a.reshape(-1, 2)
        hist_mask = mask.clone()

        if self.hist_mask and self.training:
            hist_mask[
                torch.arange(mask.shape[0]).unsqueeze(1), torch.randint(0, mask.shape[1], (num_agent, 10))] = False
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)
        elif inference_mask is not None:
            mask_t = hist_mask.unsqueeze(2) & inference_mask.unsqueeze(1)
        else:
            mask_t = hist_mask.unsqueeze(2) & hist_mask.unsqueeze(1)

        edge_index_t = dense_to_sparse(mask_t)[0]
        edge_index_t = edge_index_t[:, edge_index_t[1] > edge_index_t[0]]
        edge_index_t = edge_index_t[:, edge_index_t[1] - edge_index_t[0] <= self.time_span / self.shift]
        rel_pos_t = pos_t[edge_index_t[0]] - pos_t[edge_index_t[1]]
        rel_head_t = wrap_angle(head_t[edge_index_t[0]] - head_t[edge_index_t[1]])
        r_t = torch.stack(
            [torch.norm(rel_pos_t[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_t[edge_index_t[1]], nbr_vector=rel_pos_t[:, :2]),
             rel_head_t,
             edge_index_t[0] - edge_index_t[1]], dim=-1)
        r_t = self.r_t_emb(continuous_inputs=r_t, categorical_embs=None)
        return edge_index_t, r_t

    def build_interaction_edge(self, pos_a, head_a, head_vector_a, batch_s, mask_s):
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        edge_index_a2a = radius_graph(x=pos_s[:, :2], r=self.a2a_radius, batch=batch_s, loop=False,
                                      max_num_neighbors=300)
        edge_index_a2a = subgraph(subset=mask_s, edge_index=edge_index_a2a)[0]
        rel_pos_a2a = pos_s[edge_index_a2a[0]] - pos_s[edge_index_a2a[1]]
        rel_head_a2a = wrap_angle(head_s[edge_index_a2a[0]] - head_s[edge_index_a2a[1]])
        r_a2a = torch.stack(
            [torch.norm(rel_pos_a2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_a2a[1]], nbr_vector=rel_pos_a2a[:, :2]),
             rel_head_a2a], dim=-1)
        r_a2a = self.r_a2a_emb(continuous_inputs=r_a2a, categorical_embs=None)
        return edge_index_a2a, r_a2a

    def build_map2agent_edge(self, data, num_step, agent_category, pos_a, head_a, head_vector_a, mask,
                             batch_s, batch_pl):
        mask_pl2a = mask.clone()
        mask_pl2a = mask_pl2a.transpose(0, 1).reshape(-1)
        pos_s = pos_a.transpose(0, 1).reshape(-1, self.input_dim)
        head_s = head_a.transpose(0, 1).reshape(-1)
        head_vector_s = head_vector_a.transpose(0, 1).reshape(-1, 2)
        pos_pl = data['pt_token']['position'][:, :self.input_dim].contiguous()
        orient_pl = data['pt_token']['orientation'].contiguous()
        pos_pl = pos_pl.repeat(num_step, 1)
        orient_pl = orient_pl.repeat(num_step)
        edge_index_pl2a = radius(x=pos_s[:, :2], y=pos_pl[:, :2], r=self.pl2a_radius,
                                 batch_x=batch_s, batch_y=batch_pl, max_num_neighbors=300)
        edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
        rel_pos_pl2a = pos_pl[edge_index_pl2a[0]] - pos_s[edge_index_pl2a[1]]
        rel_orient_pl2a = wrap_angle(orient_pl[edge_index_pl2a[0]] - head_s[edge_index_pl2a[1]])
        r_pl2a = torch.stack(
            [torch.norm(rel_pos_pl2a[:, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_s[edge_index_pl2a[1]], nbr_vector=rel_pos_pl2a[:, :2]),
             rel_orient_pl2a], dim=-1)
        r_pl2a = self.r_pt2a_emb(continuous_inputs=r_pl2a, categorical_embs=None)
        return edge_index_pl2a, r_pl2a

    def forward(self,
                data: HeteroData,
                map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pos_a = data['agent']['token_pos']
        head_a = data['agent']['token_heading']
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)
        num_agent, num_step, traj_dim = pos_a.shape
        agent_category = data['agent']['category']
        agent_token_index = data['agent']['token_idx']
        feat_a, agent_token_traj = self.agent_token_embedding(data, agent_category, agent_token_index,
                                                              pos_a, head_vector_a)

        agent_valid_mask = data['agent']['agent_valid_mask'].clone()
        # eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
        # agent_valid_mask[~eval_mask] = False
        mask = agent_valid_mask
        edge_index_t, r_t = self.build_temporal_edge(pos_a, head_a, head_vector_a, num_agent, mask)

        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * t
                                 for t in range(num_step)], dim=0)
            batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * t
                                  for t in range(num_step)], dim=0)
        else:
            batch_s = torch.arange(num_step,
                                   device=pos_a.device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(num_step,
                                    device=pos_a.device).repeat_interleave(data['pt_token']['num_nodes'])

        mask_s = mask.transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(pos_a, head_a, head_vector_a, batch_s, mask_s)
        mask[agent_category != 3] = False
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(data, num_step, agent_category, pos_a, head_a,
                                                            head_vector_a, mask, batch_s, batch_pl)

        for i in range(self.num_layers):
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = feat_a.reshape(-1, num_step,
                                    self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            feat_a = self.pt2a_attn_layers[i]((map_enc['x_pt'].repeat_interleave(
                repeats=num_step, dim=0).reshape(-1, num_step, self.hidden_dim).transpose(0, 1).reshape(
                    -1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(num_step, -1, self.hidden_dim).transpose(0, 1)

        num_agent, num_step, hidden_dim, traj_num, traj_dim = agent_token_traj.shape
        next_token_prob = self.token_predict_head(feat_a)
        next_token_prob_softmax = torch.softmax(next_token_prob, dim=-1)
        _, next_token_idx = torch.topk(next_token_prob_softmax, k=10, dim=-1)

        next_token_index_gt = agent_token_index.roll(shifts=-1, dims=1)
        next_token_eval_mask = mask.clone()
        next_token_eval_mask = next_token_eval_mask * next_token_eval_mask.roll(shifts=-1, dims=1) * next_token_eval_mask.roll(shifts=1, dims=1)
        next_token_eval_mask[:, -1] = False

        return {'x_a': feat_a,
                'next_token_idx': next_token_idx,
                'next_token_prob': next_token_prob,
                'next_token_idx_gt': next_token_index_gt,
                'next_token_eval_mask': next_token_eval_mask,
                }

    def _init_rollout_state(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        hist_token_idx = (self.num_historical_steps - 1) // self.shift
        eval_mask = data['agent']['valid_mask'][:, self.num_historical_steps - 1]
        pos_a = data['agent']['token_pos'].clone()
        head_a = data['agent']['token_heading'].clone()
        pos_a[:, hist_token_idx:] = 0
        head_a[:, hist_token_idx:] = 0
        head_vector_a = torch.stack([head_a.cos(), head_a.sin()], dim=-1)

        agent_valid_mask = data['agent']['agent_valid_mask'].clone()
        agent_valid_mask[:, hist_token_idx:] = True
        agent_valid_mask[~eval_mask] = False
        agent_token_index = data['agent']['token_idx']
        agent_category = data['agent']['category']
        feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, categorical_embs = self.agent_token_embedding(
            data,
            agent_category,
            agent_token_index,
            pos_a,
            head_vector_a,
            inference=True)

        agent_type = data["agent"]["type"]
        return {
            'hist_token_idx': hist_token_idx,
            'eval_mask': eval_mask,
            'pos_a': pos_a,
            'head_a': head_a,
            'head_vector_a': head_vector_a,
            'num_agent': pos_a.shape[0],
            'num_step': pos_a.shape[1],
            'agent_valid_mask': agent_valid_mask,
            'agent_token_index': agent_token_index,
            'agent_category': agent_category,
            'agent_token_traj_all': agent_token_traj_all,
            'agent_token_emb': agent_token_emb,
            'categorical_embs': categorical_embs,
            'agent_type': agent_type,
            'veh_mask': agent_type == 0,
            'cyc_mask': agent_type == 2,
            'ped_mask': agent_type == 1,
            'mask': agent_valid_mask.clone(),
            'feat_a': feat_a,
            'feat_a_t_dict': {},
            'vel': torch.zeros_like(pos_a),
        }

    def _token_traj_for_indices(self, state: Dict[str, torch.Tensor], token_idx: torch.Tensor) -> torch.Tensor:
        if token_idx.dim() == 1:
            gather_idx = token_idx[:, None, None, None, None].expand(
                -1, 1, self.shift + 1, 4, 2)
            return torch.gather(state['agent_token_traj_all'], 1, gather_idx)[:, 0]
        gather_idx = token_idx[..., None, None, None].expand(
            -1, -1, self.shift + 1, 4, 2)
        return torch.gather(state['agent_token_traj_all'], 1, gather_idx)

    def _transform_token_traj(self,
                              state: Dict[str, torch.Tensor],
                              token_traj: torch.Tensor,
                              t: int) -> torch.Tensor:
        hist_token_idx = state['hist_token_idx']
        theta = state['head_a'][:, hist_token_idx - 1 + t]
        cos, sin = theta.cos(), theta.sin()
        rot_mat = torch.zeros((state['num_agent'], 2, 2), device=theta.device)
        rot_mat[:, 0, 0] = cos
        rot_mat[:, 0, 1] = sin
        rot_mat[:, 1, 0] = -sin
        rot_mat[:, 1, 1] = cos
        prev_pos = state['pos_a'][:, hist_token_idx - 1 + t]

        if token_traj.dim() == 5:
            beam_size = token_traj.size(1)
            agent_diff_rel = torch.bmm(
                token_traj.reshape(-1, 4, 2),
                rot_mat[:, None, None, ...].repeat(
                    1, beam_size, self.shift + 1, 1, 1).reshape(-1, 2, 2)
            ).reshape(state['num_agent'], beam_size, self.shift + 1, 4, 2)
            return agent_diff_rel + prev_pos[:, None, None, None, :]

        agent_diff_rel = torch.bmm(
            token_traj.reshape(-1, 4, 2),
            rot_mat[:, None, ...].repeat(1, self.shift + 1, 1, 1).reshape(-1, 2, 2)
        ).reshape(state['num_agent'], self.shift + 1, 4, 2)
        return agent_diff_rel + prev_pos[:, None, None, :]

    def _candidate_token_trajs(self,
                               state: Dict[str, torch.Tensor],
                               topk_idx: torch.Tensor,
                               t: int) -> torch.Tensor:
        return self._transform_token_traj(state, self._token_traj_for_indices(state, topk_idx), t)

    def _update_rollout_state(self,
                              data: HeteroData,
                              state: Dict[str, torch.Tensor],
                              selected_token_idx: torch.Tensor,
                              selected_agent_pred_rel: torch.Tensor,
                              t: int) -> torch.Tensor:
        hist_token_idx = state['hist_token_idx']
        pos_write_idx = hist_token_idx + t
        state['pos_a'][:, pos_write_idx] = selected_agent_pred_rel[:, -1].clone().mean(dim=1)
        diff_xy = selected_agent_pred_rel[:, -1, 0, :] - selected_agent_pred_rel[:, -1, 3, :]
        theta = torch.arctan2(diff_xy[:, 1], diff_xy[:, 0])
        state['head_a'][:, pos_write_idx] = theta

        agent_token_emb = state['agent_token_emb'].clone()
        veh_mask = state['veh_mask']
        ped_mask = state['ped_mask']
        cyc_mask = state['cyc_mask']
        agent_token_emb[veh_mask, pos_write_idx] = self.agent_token_emb_veh[selected_token_idx[veh_mask]]
        agent_token_emb[ped_mask, pos_write_idx] = self.agent_token_emb_ped[selected_token_idx[ped_mask]]
        agent_token_emb[cyc_mask, pos_write_idx] = self.agent_token_emb_cyc[selected_token_idx[cyc_mask]]
        state['agent_token_emb'] = agent_token_emb

        motion_vector_a = torch.cat([state['pos_a'].new_zeros(data['agent']['num_nodes'], 1, self.input_dim),
                                     state['pos_a'][:, 1:] - state['pos_a'][:, :-1]], dim=1)
        head_vector_a = torch.stack([state['head_a'].cos(), state['head_a'].sin()], dim=-1)
        vel = motion_vector_a.clone() / (0.1 * self.shift)
        vel[:, hist_token_idx + 1 + t:] = 0
        motion_vector_a[:, hist_token_idx + 1 + t:] = 0
        x_a = torch.stack(
            [torch.norm(motion_vector_a[:, :, :2], p=2, dim=-1),
             angle_between_2d_vectors(ctr_vector=head_vector_a, nbr_vector=motion_vector_a[:, :, :2])], dim=-1)
        x_a = self.x_a_emb(continuous_inputs=x_a.reshape(-1, x_a.size(-1)),
                           categorical_embs=state['categorical_embs'])
        x_a = x_a.reshape(-1, state['num_step'], self.hidden_dim)
        state['feat_a'] = self.fusion_emb(torch.cat((agent_token_emb, x_a), dim=-1))
        state['head_vector_a'] = head_vector_a
        state['vel'] = vel
        return vel

    def _rollout_step(self,
                      data: HeteroData,
                      map_enc: Mapping[str, torch.Tensor],
                      state: Dict[str, torch.Tensor],
                      t: int,
                      sample_policy: str = 'multinomial',
                      forced_token_idx: Optional[torch.Tensor] = None,
                      topk: Optional[int] = None) -> Dict[str, torch.Tensor]:
        hist_token_idx = state['hist_token_idx']
        mask = state['mask']
        if t == 0:
            inference_mask = mask.clone()
            inference_mask[:, hist_token_idx + t:] = False
        else:
            inference_mask = torch.zeros_like(mask)
            inference_mask[:, hist_token_idx + t - 1] = True

        edge_index_t, r_t = self.build_temporal_edge(
            state['pos_a'], state['head_a'], state['head_vector_a'], state['num_agent'], mask, inference_mask)
        if isinstance(data, Batch):
            batch_s = torch.cat([data['agent']['batch'] + data.num_graphs * step
                                 for step in range(state['num_step'])], dim=0)
            batch_pl = torch.cat([data['pt_token']['batch'] + data.num_graphs * step
                                  for step in range(state['num_step'])], dim=0)
        else:
            batch_s = torch.arange(state['num_step'],
                                   device=state['pos_a'].device).repeat_interleave(data['agent']['num_nodes'])
            batch_pl = torch.arange(state['num_step'],
                                    device=state['pos_a'].device).repeat_interleave(data['pt_token']['num_nodes'])
        edge_index_pl2a, r_pl2a = self.build_map2agent_edge(
            data, state['num_step'], state['agent_category'], state['pos_a'], state['head_a'],
            state['head_vector_a'], inference_mask, batch_s, batch_pl)
        mask_s = inference_mask.transpose(0, 1).reshape(-1)
        edge_index_a2a, r_a2a = self.build_interaction_edge(
            state['pos_a'], state['head_a'], state['head_vector_a'], batch_s, mask_s)

        feat_a = state['feat_a']
        feat_a_t_dict = state['feat_a_t_dict']
        for i in range(self.num_layers):
            if i in feat_a_t_dict:
                feat_a = feat_a_t_dict[i]
            feat_a = feat_a.reshape(-1, self.hidden_dim)
            feat_a = self.t_attn_layers[i](feat_a, r_t, edge_index_t)
            feat_a = feat_a.reshape(-1, state['num_step'],
                                    self.hidden_dim).transpose(0, 1).reshape(-1, self.hidden_dim)
            feat_a = self.pt2a_attn_layers[i]((map_enc['x_pt'].repeat_interleave(
                repeats=state['num_step'], dim=0).reshape(-1, state['num_step'], self.hidden_dim).transpose(
                    0, 1).reshape(-1, self.hidden_dim), feat_a), r_pl2a, edge_index_pl2a)
            feat_a = self.a2a_attn_layers[i](feat_a, r_a2a, edge_index_a2a)
            feat_a = feat_a.reshape(state['num_step'], -1, self.hidden_dim).transpose(0, 1)

            update_idx = hist_token_idx - 1 + t
            if i + 1 not in feat_a_t_dict:
                feat_a_t_dict[i + 1] = feat_a
            else:
                cached_feat = feat_a_t_dict[i + 1].clone()
                cached_feat[:, update_idx] = feat_a[:, update_idx]
                feat_a_t_dict[i + 1] = cached_feat

        pred_idx = hist_token_idx - 1 + t
        logits = self.token_predict_head(feat_a[:, pred_idx])
        probs = torch.softmax(logits, dim=-1)
        topk = min(int(topk or self.beam_size), logits.size(-1))
        topk_prob, next_token_idx = torch.topk(probs, k=topk, dim=-1)
        candidate_rel = self._candidate_token_trajs(state, next_token_idx, t)

        if forced_token_idx is None:
            if sample_policy == 'argmax':
                sample_index = torch.zeros((state['num_agent'], 1), dtype=torch.long, device=logits.device)
            else:
                sample_index = torch.multinomial(topk_prob, 1).to(logits.device)
            selected_rel = candidate_rel.gather(
                dim=1,
                index=sample_index[..., None, None, None].expand(
                    -1, -1, self.shift + 1, 4, 2))[:, 0]
            selected_token_idx = next_token_idx.gather(dim=1, index=sample_index).squeeze(-1)
            selected_prob = topk_prob.gather(dim=-1, index=sample_index).squeeze(-1)
        else:
            selected_token_idx = forced_token_idx.long()
            selected_rel = self._transform_token_traj(
                state, self._token_traj_for_indices(state, selected_token_idx), t)
            selected_prob = probs.gather(dim=-1, index=selected_token_idx[:, None]).squeeze(-1)

        selected_traj = selected_rel[:, 1:].clone().mean(dim=2)
        diff_xy = selected_rel[:, 1:, 0, :] - selected_rel[:, 1:, 3, :]
        selected_head = torch.arctan2(diff_xy[:, :, 1], diff_xy[:, :, 0])
        step_mask = state['mask'][:, pred_idx].clone()
        step_mask[state['agent_category'] != 3] = False
        self._update_rollout_state(data, state, selected_token_idx, selected_rel, t)

        return {
            'logits': logits,
            'topk_prob': topk_prob,
            'topk_idx': next_token_idx,
            'candidate_trajs': candidate_rel[:, :, 1:].mean(dim=3),
            'selected_token_idx': selected_token_idx,
            'selected_prob': selected_prob,
            'selected_traj': selected_traj,
            'selected_head': selected_head,
            'mask': step_mask,
            'vel': state['vel'],
        }

    def inference(self,
                  data: HeteroData,
                  map_enc: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        state = self._init_rollout_state(data)
        self.num_recurrent_steps_val = data["agent"]['position'].shape[1]-self.num_historical_steps
        pred_traj = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val, 2,
                                device=state['feat_a'].device)
        pred_head = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val,
                                device=state['feat_a'].device)
        pred_prob = torch.zeros(data["agent"].num_nodes, self.num_recurrent_steps_val // self.shift,
                                device=state['feat_a'].device)
        next_token_idx_list = []
        for t in range(self.num_recurrent_steps_val // self.shift):
            step = self._rollout_step(data, map_enc, state, t, sample_policy='multinomial', topk=self.beam_size)
            pred_prob[:, t] = step['selected_prob']
            pred_traj[:, t * self.shift:(t + 1) * self.shift] = step['selected_traj']
            pred_head[:, t * self.shift:(t + 1) * self.shift] = step['selected_head']
            next_token_idx_list.append(step['selected_token_idx'][:, None])

        state['agent_valid_mask'][state['agent_category'] != 3] = False

        return {
            'pos_a': state['pos_a'][:, state['hist_token_idx']:],
            'head_a': state['head_a'][:, state['hist_token_idx']:],
            'gt': data['agent']['position'][:, self.num_historical_steps:, :self.input_dim].contiguous(),
            'valid_mask': state['agent_valid_mask'][:, self.num_historical_steps:],
            'pred_traj': pred_traj,
            'pred_head': pred_head,
            'next_token_idx': torch.cat(next_token_idx_list, dim=-1),
            'next_token_idx_gt': state['agent_token_index'].roll(shifts=-1, dims=1),
            'next_token_eval_mask': data['agent']['agent_valid_mask'],
            'pred_prob': pred_prob,
            'vel': state['vel']
        }

    def rollout_tokens(self,
                       data: HeteroData,
                       map_enc: Mapping[str, torch.Tensor],
                       num_steps: int,
                       num_rollouts: int,
                       topk: int) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            num_steps = min(int(num_steps), (data["agent"]['position'].shape[1] - self.num_historical_steps) // self.shift)
            token_rollouts = []
            traj_rollouts = []
            head_rollouts = []
            mask_rollouts = []
            candidate_idx_rollouts = []
            candidate_traj_rollouts = []
            candidate_prob_rollouts = []
            for _ in range(int(num_rollouts)):
                state = self._init_rollout_state(data)
                token_steps = []
                traj_steps = []
                head_steps = []
                mask_steps = []
                candidate_idx_steps = []
                candidate_traj_steps = []
                candidate_prob_steps = []
                for t in range(num_steps):
                    step = self._rollout_step(data, map_enc, state, t, sample_policy='multinomial', topk=topk)
                    token_steps.append(step['selected_token_idx'])
                    traj_steps.append(step['selected_traj'])
                    head_steps.append(step['selected_head'])
                    mask_steps.append(step['mask'])
                    candidate_idx_steps.append(step['topk_idx'])
                    candidate_traj_steps.append(step['candidate_trajs'])
                    candidate_prob_steps.append(step['topk_prob'])
                token_rollouts.append(torch.stack(token_steps, dim=-1))
                traj_rollouts.append(torch.cat(traj_steps, dim=1))
                head_rollouts.append(torch.cat(head_steps, dim=1))
                mask_rollouts.append(torch.stack(mask_steps, dim=-1))
                candidate_idx_rollouts.append(torch.stack(candidate_idx_steps, dim=1))
                candidate_traj_rollouts.append(torch.stack(candidate_traj_steps, dim=1))
                candidate_prob_rollouts.append(torch.stack(candidate_prob_steps, dim=1))

            return {
                'token_idx': torch.stack(token_rollouts, dim=0),
                'pred_traj': torch.stack(traj_rollouts, dim=0),
                'pred_head': torch.stack(head_rollouts, dim=0),
                'mask': torch.stack(mask_rollouts, dim=0),
                'candidate_idx': torch.stack(candidate_idx_rollouts, dim=0),
                'candidate_trajs': torch.stack(candidate_traj_rollouts, dim=0),
                'candidate_prob': torch.stack(candidate_prob_rollouts, dim=0),
            }

    def score_token_prefix(self,
                           data: HeteroData,
                           map_enc: Mapping[str, torch.Tensor],
                           forced_token_idx: torch.Tensor,
                           num_steps: int) -> Dict[str, torch.Tensor]:
        if forced_token_idx.dim() == 2:
            forced_token_idx = forced_token_idx.unsqueeze(0)
        rollout_count = forced_token_idx.size(0)
        num_steps = min(int(num_steps), forced_token_idx.size(-1))
        logits_rollouts = []
        mask_rollouts = []
        traj_rollouts = []
        head_rollouts = []
        for rollout_id in range(rollout_count):
            state = self._init_rollout_state(data)
            logits_steps = []
            mask_steps = []
            traj_steps = []
            head_steps = []
            for t in range(num_steps):
                step = self._rollout_step(
                    data, map_enc, state, t, forced_token_idx=forced_token_idx[rollout_id, :, t])
                logits_steps.append(step['logits'])
                mask_steps.append(step['mask'])
                traj_steps.append(step['selected_traj'])
                head_steps.append(step['selected_head'])
            logits_rollouts.append(torch.stack(logits_steps, dim=1))
            mask_rollouts.append(torch.stack(mask_steps, dim=-1))
            traj_rollouts.append(torch.cat(traj_steps, dim=1))
            head_rollouts.append(torch.cat(head_steps, dim=1))

        return {
            'logits': torch.stack(logits_rollouts, dim=0),
            'mask': torch.stack(mask_rollouts, dim=0),
            'pred_traj': torch.stack(traj_rollouts, dim=0),
            'pred_head': torch.stack(head_rollouts, dim=0),
        }
