import unittest
from types import SimpleNamespace

import torch

import scripts.visualize_ar_map_rollout as viz




class _ProposalGeometryModel(SimpleNamespace):
    def _refresh_token_geometry(
        self,
        _token_ids,
        packed,
        geometry_known_mask=None,
        proposal_token_ids=None,
        proposal_confidence=None,
    ):
        positions = packed["token_positions"].clone()
        headings = packed["token_headings"].clone()
        confidence = torch.zeros_like(packed["valid_mask"], dtype=torch.float32)
        if proposal_token_ids is not None and proposal_confidence is not None:
            proposal_known = packed["valid_mask"] & (proposal_confidence > 0.0)
            positions[..., 0] = positions[..., 0] + proposal_token_ids.float() * proposal_known.float()
            confidence[proposal_known] = proposal_confidence.float()[proposal_known]
        return positions, headings, confidence

class _FakeMapDecoder:
    def _build_map2token_edges(
        self,
        _token_positions,
        _token_headings,
        _valid_mask,
        _map_positions,
        _map_orientations,
        _map_batch,
        _map_valid_mask,
    ):
        return torch.tensor([[0, 2], [0, 4]], dtype=torch.long), None


class VisualizeARMapRolloutTest(unittest.TestCase):
    def test_rollout_input_snapshot_records_actual_rolled_agent_state(self):
        self.assertTrue(hasattr(viz, "_rollout_input_snapshot"))

        model = SimpleNamespace(num_future_chunks=4, diffusion_decoder=_FakeMapDecoder())
        packed = {
            "agent_maps": [(0, 0, torch.tensor([7, 9]))],
            "valid_mask": torch.ones(1, 8, dtype=torch.bool),
            "token_positions": torch.arange(16, dtype=torch.float32).view(1, 8, 2),
            "token_headings": torch.zeros(1, 8),
            "map_positions": torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [2.0, 2.0],
                ]
            ),
            "map_orientations": torch.zeros(3),
            "map_batch": torch.zeros(3, dtype=torch.long),
            "map_valid_mask": torch.ones(3, dtype=torch.bool),
        }
        history_frame_pos = torch.zeros(10, 3, 2)
        history_frame_pos[7, :, 0] = torch.tensor([70.0, 71.0, 72.0])
        history_frame_pos[9, :, 0] = torch.tensor([90.0, 91.0, 92.0])
        history_frame_pos[9, :, 1] = torch.tensor([5.0, 6.0, 7.0])
        history_frame_valid = torch.zeros(10, 3, dtype=torch.bool)
        history_frame_valid[7, :] = True
        history_frame_valid[9, :] = True
        generation_agents = torch.zeros(10, dtype=torch.bool)
        generation_agents[7] = True
        generation_agents[9] = True

        snapshot = viz._rollout_input_snapshot(
            model=model,
            packed=packed,
            agent_index=9,
            history_frame_pos=history_frame_pos,
            history_frame_valid=history_frame_valid,
            generation_agents=generation_agents,
            initial_proposal_ids=None,
            initial_proposal_confidence=None,
        )

        self.assertTrue(torch.equal(snapshot["input_query_positions"], packed["token_positions"][0, 4:8]))
        self.assertTrue(torch.equal(snapshot["input_current_xy"], history_frame_pos[9, -1]))
        self.assertTrue(torch.equal(snapshot["selected_history"], history_frame_pos[9]))
        self.assertEqual(snapshot["num_input_agents"], 2)
        self.assertEqual(snapshot["input_num_connected"], 1)
        self.assertTrue(torch.equal(snapshot["input_connected_map_positions"], packed["map_positions"][2:3]))

    def test_rollout_input_snapshot_records_proposal_refreshed_query_geometry(self):
        model = _ProposalGeometryModel(
            num_future_chunks=4,
            diffusion_decoder=_FakeMapDecoder(),
            mask_token_id=2048,
            use_proposal_geometry=True,
        )
        packed = {
            "agent_maps": [(0, 0, torch.tensor([7, 9]))],
            "valid_mask": torch.ones(1, 8, dtype=torch.bool),
            "token_positions": torch.zeros(1, 8, 2),
            "token_headings": torch.zeros(1, 8),
            "map_positions": torch.tensor([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]]),
            "map_orientations": torch.zeros(3),
            "map_batch": torch.zeros(3, dtype=torch.long),
            "map_valid_mask": torch.ones(3, dtype=torch.bool),
        }
        history_frame_pos = torch.zeros(10, 3, 2)
        history_frame_valid = torch.ones(10, 3, dtype=torch.bool)
        generation_agents = torch.zeros(10, dtype=torch.bool)
        generation_agents[7] = True
        generation_agents[9] = True
        proposal_ids = torch.zeros(1, 8, dtype=torch.long)
        proposal_conf = torch.zeros(1, 8)
        proposal_ids[0, 4:8] = torch.tensor([0, 5, 9, 0])
        proposal_conf[0, 4:8] = torch.tensor([0.0, 0.8, 0.9, 0.0])

        snapshot = viz._rollout_input_snapshot(
            model=model,
            packed=packed,
            agent_index=9,
            history_frame_pos=history_frame_pos,
            history_frame_valid=history_frame_valid,
            generation_agents=generation_agents,
            initial_proposal_ids=proposal_ids,
            initial_proposal_confidence=proposal_conf,
        )

        expected = packed["token_positions"][0, 4:8].clone()
        expected[1, 0] += 5.0
        expected[2, 0] += 9.0
        self.assertTrue(torch.equal(snapshot["proposal_query_positions"], expected))
        self.assertTrue(torch.equal(snapshot["proposal_geometry_confidence"], proposal_conf[0, 4:8]))
        self.assertEqual(snapshot["proposal_num_connected"], 1)
        self.assertTrue(torch.equal(snapshot["proposal_connected_map_positions"], packed["map_positions"][2:3]))


if __name__ == "__main__":
    unittest.main()
