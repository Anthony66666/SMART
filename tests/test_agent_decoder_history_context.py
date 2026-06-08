import unittest
from types import MethodType

import torch
from torch_geometric.data import HeteroData

from smart.modules.agent_decoder import SMARTAgentDecoder


class SMARTAgentDecoderHistoryContextTest(unittest.TestCase):
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
