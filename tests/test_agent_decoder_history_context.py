import unittest
from types import MethodType

import numpy as np
import torch
from torch import nn
from torch_geometric.data import HeteroData

from smart.modules.agent_decoder import SMARTAgentDecoder


class _HalfTokenEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return x.new_zeros(x.shape[0], self.hidden_dim, dtype=torch.float16)


class _HalfFeatureEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return x.new_zeros(x.shape[0], self.hidden_dim, dtype=torch.float16)


class _HalfFourierEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, continuous_inputs, categorical_embs=None):
        del categorical_embs
        return continuous_inputs.new_zeros(continuous_inputs.shape[0], self.hidden_dim, dtype=torch.float16)


class _TakeAgentEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return x[..., :self.hidden_dim]


class SMARTAgentDecoderHistoryContextTest(unittest.TestCase):
    def test_agent_token_embedding_accepts_half_precision_token_embeddings(self):
        decoder = object.__new__(SMARTAgentDecoder)
        nn.Module.__init__(decoder)
        decoder.input_dim = 2
        decoder.hidden_dim = 4
        decoder.num_historical_steps = 2
        decoder.token_size = 2
        decoder.shift = 1
        token = np.zeros((decoder.token_size, 4, 2), dtype=np.float32)
        token_all = np.zeros((decoder.token_size, decoder.shift, 4, 2), dtype=np.float32)
        decoder.trajectory_token = {"veh": token, "ped": token, "cyc": token}
        decoder.trajectory_token_all = {"veh": token_all, "ped": token_all, "cyc": token_all}
        decoder.token_emb_veh = _HalfTokenEmbedding(decoder.hidden_dim)
        decoder.token_emb_ped = _HalfTokenEmbedding(decoder.hidden_dim)
        decoder.token_emb_cyc = _HalfTokenEmbedding(decoder.hidden_dim)
        decoder.type_a_emb = _HalfFeatureEmbedding(decoder.hidden_dim)
        decoder.shape_emb = _HalfFeatureEmbedding(decoder.hidden_dim)
        decoder.x_a_emb = _HalfFourierEmbedding(decoder.hidden_dim)
        decoder.fusion_emb = _TakeAgentEmbedding(decoder.hidden_dim)

        num_agents = 3
        num_steps = 2
        data = {
            "agent": {
                "num_nodes": num_agents,
                "type": torch.tensor([0, 1, 2]),
                "shape": torch.zeros(num_agents, decoder.num_historical_steps, 3),
                "token_velocity": torch.zeros(num_agents, num_steps, 2),
            }
        }
        agent_token_index = torch.tensor([[0, 1], [1, 0], [0, 1]])
        pos_a = torch.zeros(num_agents, num_steps, 2)
        head_vector_a = torch.zeros(num_agents, num_steps, 2)

        try:
            feat_a, agent_token_traj, agent_token_traj_all, agent_token_emb, _ = decoder.agent_token_embedding(
                data,
                agent_category=torch.zeros(num_agents, dtype=torch.long),
                agent_token_index=agent_token_index,
                pos_a=pos_a,
                head_vector_a=head_vector_a,
                inference=True,
            )
        except RuntimeError as exc:
            self.fail(f"agent_token_embedding should accept half precision token embeddings: {exc}")

        self.assertEqual(feat_a.dtype, torch.float16)
        self.assertEqual(agent_token_emb.dtype, torch.float16)
        self.assertEqual(agent_token_traj.dtype, torch.float32)
        self.assertEqual(agent_token_traj_all.dtype, torch.float32)

    def test_history_context_map_mask_matches_official_forward_category_targets(self):
        decoder = object.__new__(SMARTAgentDecoder)
        decoder.num_historical_steps = 11
        decoder.shift = 5
        decoder.hidden_dim = 4
        decoder.num_layers = 0
        captured = {}

        def fake_agent_token_embedding(self, data, agent_category, agent_token_index, pos_a, head_vector_a):
            del self, agent_category, agent_token_index, head_vector_a
            num_agents, num_steps, _ = pos_a.shape
            return pos_a.new_zeros(num_agents, num_steps, decoder.hidden_dim), None

        def fake_build_temporal_edge(self, *args, **kwargs):
            del self, args, kwargs
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, decoder.hidden_dim)

        def fake_build_interaction_edge(self, *args, **kwargs):
            del self, args, kwargs
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, decoder.hidden_dim)

        def fake_build_map2agent_edge(self, data, num_step, agent_category, pos_a, head_a,
                                      head_vector_a, mask, batch_s, batch_pl,
                                      map_token_visible_mask=None):
            del self, data, num_step, agent_category, pos_a, head_a
            del head_vector_a, batch_s, batch_pl, map_token_visible_mask
            captured["mask"] = mask.detach().clone()
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, decoder.hidden_dim)

        decoder.agent_token_embedding = MethodType(fake_agent_token_embedding, decoder)
        decoder.build_temporal_edge = MethodType(fake_build_temporal_edge, decoder)
        decoder.build_interaction_edge = MethodType(fake_build_interaction_edge, decoder)
        decoder.build_map2agent_edge = MethodType(fake_build_map2agent_edge, decoder)
        decoder._build_temporal_batches = MethodType(
            lambda self, data, num_step, pos_a: (
                torch.zeros(pos_a.numel() // pos_a.shape[-1], dtype=torch.long),
                torch.zeros(num_step, dtype=torch.long),
                torch.zeros(pos_a.shape[0], dtype=torch.long),
            ),
            decoder,
        )

        data = HeteroData()
        data["agent"]["token_pos"] = torch.zeros(3, 4, 2)
        data["agent"]["token_heading"] = torch.zeros(3, 4)
        data["agent"]["token_idx"] = torch.zeros(3, 4, dtype=torch.long)
        data["agent"]["agent_valid_mask"] = torch.ones(3, 4, dtype=torch.bool)
        data["agent"]["category"] = torch.tensor([3, 0, 3])
        data["agent"]["type"] = torch.tensor([0, 0, 3])
        map_enc = {
            "x_pt": torch.zeros(1, decoder.hidden_dim),
            "pt_visibility_mask": torch.ones(1, dtype=torch.bool),
        }

        decoder.encode_history_context(data, map_enc)

        expected = torch.tensor([
            [True, True, False, False],
            [False, False, False, False],
            [True, True, False, False],
        ])
        self.assertTrue(torch.equal(captured["mask"], expected))


if __name__ == "__main__":
    unittest.main()
