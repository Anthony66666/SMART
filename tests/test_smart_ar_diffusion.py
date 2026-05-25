
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion


def _ar_shell():
    model = object.__new__(SMARTAutoregressiveDiffusion)
    model.num_historical_steps = 11
    model.num_future_steps = 80
    model.future_chunk_steps = 5
    model.num_future_chunks = 4
    model.ar_history_tokens = 2
    model.ar_prediction_tokens = 4
    model.ar_commit_tokens = 2
    model.ar_token_steps = 5
    model.ar_total_rollout_steps = 80
    model.ar_local_map_radius = 2.0
    model.max_map_tokens = 0
    model.model_config = SimpleNamespace(decoder=SimpleNamespace(token_size=2048))
    model.encoder = SimpleNamespace(agent_encoder=SimpleNamespace(shift=5))
    return model


def _toy_sequence(num_agents=2, num_tokens=10, num_frames=61):
    data = HeteroData()
    token_idx = torch.arange(num_tokens).unsqueeze(0).repeat(num_agents, 1)
    data["agent"]["token_idx"] = token_idx
    data["agent"]["agent_valid_mask"] = torch.ones(num_agents, num_tokens, dtype=torch.bool)
    token_pos = torch.zeros(num_agents, num_tokens, 2)
    token_pos[..., 0] = torch.arange(num_tokens).float()
    data["agent"]["token_pos"] = token_pos
    data["agent"]["token_heading"] = torch.zeros(num_agents, num_tokens)
    data["agent"]["position"] = torch.zeros(num_agents, num_frames, 2)
    data["agent"]["position"][..., 0] = torch.arange(num_frames).float()
    data["agent"]["heading"] = torch.zeros(num_agents, num_frames)
    data["agent"]["valid_mask"] = torch.ones(num_agents, num_frames, dtype=torch.bool)
    data["agent"]["shape"] = torch.ones(num_agents, num_frames, 3)
    data["agent"]["type"] = torch.tensor([0, 1])[:num_agents]
    data["agent"]["category"] = torch.tensor([3, 0])[:num_agents]
    data["agent"]["num_nodes"] = num_agents
    return data


class SMARTAutoregressiveDiffusionTest(unittest.TestCase):
    def test_training_view_uses_two_history_tokens_and_four_future_tokens(self):
        model = _ar_shell()
        data = _toy_sequence(num_agents=1)

        view, target_tokens, target_valid, anchor = model._build_ar_training_view(
            data,
            anchor_token=4,
            perturb=False,
        )

        self.assertEqual(anchor, 4)
        self.assertTrue(torch.equal(view["agent"]["token_idx"][0, :6], torch.tensor([2, 3, 4, 5, 6, 7])))
        self.assertTrue(torch.equal(target_tokens[0], torch.tensor([4, 5, 6, 7])))
        self.assertTrue(target_valid[0].all())
        self.assertEqual(view["agent"]["position"].shape[1], 31)
        self.assertAlmostEqual(float(view["agent"]["position"][0, 10, 0]), 20.0, places=5)
        self.assertAlmostEqual(float(view["agent"]["position"][0, 11, 0]), 21.0, places=5)

    def test_commit_updates_history_to_the_two_committed_tokens(self):
        model = _ar_shell()
        history = torch.tensor([[10, 11], [20, 21]])
        committed = torch.tensor([[30, 31], [40, 41]])

        rolled = model._roll_history_token_ids(history, committed)

        self.assertTrue(torch.equal(rolled, committed))

    def test_eighty_step_rollout_uses_eight_rounds(self):
        model = _ar_shell()

        self.assertEqual(model._num_ar_rollout_rounds(), 8)

    def test_local_map_refresh_rescreens_by_current_agent_pose(self):
        model = _ar_shell()
        map_positions = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        map_batch = torch.zeros(3, dtype=torch.long)
        visible = torch.ones(3, dtype=torch.bool)

        near_zero = model._select_local_map_indices(
            map_positions=map_positions,
            map_batch=map_batch,
            scene_idx=0,
            agent_positions=torch.tensor([[0.5, 0.0]]),
            map_visible=visible,
        )
        near_ten = model._select_local_map_indices(
            map_positions=map_positions,
            map_batch=map_batch,
            scene_idx=0,
            agent_positions=torch.tensor([[10.5, 0.0]]),
            map_visible=visible,
        )

        self.assertTrue(torch.equal(near_zero, torch.tensor([0])))
        self.assertTrue(torch.equal(near_ten, torch.tensor([1])))

    def test_rollout_view_preserves_history_token_valid_mask(self):
        model = _ar_shell()
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)
        data["agent"]["agent_valid_mask"][0, 0] = False
        generation = torch.tensor([True])

        view = model._build_ar_rollout_view(
            data,
            data["agent"]["token_idx"][:, :2],
            data["agent"]["token_pos"][:, :2],
            data["agent"]["token_heading"][:, :2],
            data["agent"]["position"][:, :11],
            data["agent"]["heading"][:, :11],
            data["agent"]["valid_mask"][:, :11],
            generation,
        )

        self.assertTrue(torch.equal(
            view["agent"]["agent_valid_mask"][0, :2],
            torch.tensor([False, True]),
        ))
        self.assertTrue(view["agent"]["agent_valid_mask"][0, 2:].all())

    def test_rollout_view_removes_current_invalid_agents_from_history_context(self):
        model = _ar_shell()
        data = _toy_sequence(num_agents=2, num_tokens=18, num_frames=91)
        generation = torch.tensor([True, False])

        view = model._build_ar_rollout_view(
            data,
            data["agent"]["token_idx"][:, :2],
            data["agent"]["token_pos"][:, :2],
            data["agent"]["token_heading"][:, :2],
            data["agent"]["position"][:, :11],
            data["agent"]["heading"][:, :11],
            data["agent"]["valid_mask"][:, :11],
            generation,
        )

        self.assertTrue(view["agent"]["agent_valid_mask"][0, :2].all())
        self.assertFalse(view["agent"]["agent_valid_mask"][1].any())

    def test_inference_commits_two_tokens_per_round_for_full_eighty_steps(self):
        model = _ar_shell()
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)
        call_state = {'count': 0}

        model._prepare_batch = MethodType(lambda self, batch: batch, model)

        def fake_build_inputs(self, rollout_view):
            packed = {
                'agent_maps': [(0, 0, torch.tensor([0]))],
                'valid_mask': torch.ones(1, 4, dtype=torch.bool),
                'token_positions': torch.zeros(1, 4, 2),
                'token_headings': torch.zeros(1, 4),
                'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
                'chunk_ids': torch.arange(4).unsqueeze(0),
                'agent_context': torch.zeros(1, 4, 1),
                'agent_type_ids': torch.zeros(1, 4, dtype=torch.long),
                'agent_shape_embeddings': torch.zeros(1, 4, 1),
            }
            summary = torch.zeros(1, 1)
            ft = torch.zeros(1, 4, dtype=torch.long)
            fv = torch.ones(1, 4, dtype=torch.bool)
            generation = torch.tensor([True])
            return packed, summary, ft, fv, generation, generation, torch.zeros(1, dtype=torch.long)

        def fake_sample(self, **_kwargs):
            base = call_state['count'] * 10
            call_state['count'] += 1
            return torch.tensor([[base, base + 1, base + 2, base + 3]]), torch.ones(1, 4)

        def fake_token_world(self, token_ids, _agent_types, positions, headings):
            step = torch.arange(1, 6, dtype=positions.dtype, device=positions.device).view(1, 5, 1)
            world = positions[:, None, :] + torch.cat([step, torch.zeros_like(step)], dim=-1)
            return world, headings[:, None].expand(-1, 5)

        model._build_diffusion_inputs = MethodType(fake_build_inputs, model)
        model._diffusion_sample = MethodType(fake_sample, model)
        model._token_chunk_world = MethodType(fake_token_world, model)

        out = model.inference(data)

        self.assertEqual(call_state['count'], 8)
        self.assertEqual(out['pred_traj'].shape, (1, 80, 2))
        self.assertTrue(out['pred_valid_mask'].all())
        self.assertTrue(torch.equal(out['next_token_idx'][0, :4], torch.tensor([0, 1, 10, 11])))


    def test_non_target_generation_agent_is_not_in_loss_mask(self):
        model = _ar_shell()
        tokens = torch.arange(8).reshape(2, 4)
        valid = torch.ones(2, 4, dtype=torch.bool)
        generation = torch.tensor([True, True])
        supervision = torch.tensor([True, False])
        packed = model._pack_diffusion_sequence(
            tokens,
            valid,
            generation,
            supervision,
            torch.zeros(2, dtype=torch.long),
            torch.zeros(2, 2),
            torch.zeros(2),
            torch.zeros(2, 4),
            torch.tensor([0, 1]),
            torch.zeros(2, 4),
        )

        self.assertTrue(torch.equal(packed["valid_mask"][0, :8], torch.ones(8, dtype=torch.bool)))
        self.assertTrue(torch.equal(
            packed["loss_mask_base"][0, :8],
            torch.tensor([True, True, True, True, False, False, False, False]),
        ))

    def test_validation_step_uses_ar_window_loss_metrics(self):
        model = _ar_shell()
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)
        model.ntp_aux_loss_weight = 0.0
        calls = {"ar_window": 0}
        logged = []

        model._prepare_batch = MethodType(lambda self, batch: batch, model)

        def fake_ar_view(self, batch, anchor_token=None, perturb=None):
            calls["ar_window"] += 1
            return batch, None, None, 2

        def fake_build_inputs(self, batch):
            packed = {
                "token_ids": torch.zeros(1, 4, dtype=torch.long),
                "valid_mask": torch.ones(1, 4, dtype=torch.bool),
                "loss_mask_base": torch.ones(1, 4, dtype=torch.bool),
                "chunk_ids": torch.arange(4).unsqueeze(0),
                "token_agent_ids": torch.zeros(1, 4, dtype=torch.long),
            }
            summary = torch.zeros(1, 1)
            return packed, summary, None, None, None, None, None

        model._build_ar_training_view = MethodType(fake_ar_view, model)
        model._build_diffusion_inputs = MethodType(fake_build_inputs, model)
        model._compute_diffusion_loss = MethodType(
            lambda self, packed, summary: (torch.tensor(2.0), torch.tensor(0.5)),
            model,
        )
        model._compute_optional_ntp_loss = MethodType(lambda self, batch, ref: torch.tensor(0.0), model)
        model._should_run_validation_inference = MethodType(lambda self, batch_idx: False, model)
        model.log = MethodType(lambda self, name, *args, **kwargs: logged.append(name), model)

        model.validation_step(data, 0)

        self.assertEqual(calls["ar_window"], 1)
        self.assertIn("val_ar_window_loss", logged)
        self.assertIn("val_ar_window_diffusion_loss", logged)
        self.assertNotIn("val_loss", logged)
        self.assertNotIn("val_diffusion_loss", logged)

    def test_ar_train_config_monitors_rollout_metric_not_window_loss(self):
        text = Path("configs/train/train_scalable_ar_diffusion.yaml").read_text()

        self.assertIn('monitor_metric: "val_minADE"', text)
        self.assertIn('monitor_mode: "min"', text)
        self.assertNotIn('monitor_metric: "val_loss"', text)

    def test_unused_self_condition_options_are_removed_from_code_and_configs(self):
        paths = [
            Path("smart/model/smart_diffusion.py"),
            Path("configs/train/train_scalable_diffusion.yaml"),
            Path("configs/train/train_scalable_diffusion_local.yaml"),
            Path("configs/train/train_scalable_ar_diffusion.yaml"),
            Path("configs/train/train_scalable_ar_diffusion_local.yaml"),
        ]
        for path in paths:
            text = path.read_text()
            self.assertNotIn("self_condition_visible_prob", text, str(path))
            self.assertNotIn("self_condition_loss_weight", text, str(path))


if __name__ == "__main__":
    unittest.main()
