import unittest

import torch

from smart.utils.config import load_config_act


class SMARTContinuousActionDiffusionUtilityTest(unittest.TestCase):
    def test_local_world_roundtrip_preserves_action_chunk(self):
        from smart.model.smart_continuous_action_diffusion import (
            SMARTContinuousActionDiffusion,
        )

        model = object.__new__(SMARTContinuousActionDiffusion)
        start_pos = torch.tensor([[10.0, -2.0], [1.0, 3.0]])
        start_heading = torch.tensor([0.0, 1.57079632679])
        local = torch.tensor(
            [
                [[[1.0, 0.0], [2.0, 0.5]]],
                [[[0.0, 1.0], [1.0, 1.0]]],
            ]
        )

        world = model._local_action_to_world(local, start_pos, start_heading)
        recovered = model._world_action_to_local(world, start_pos, start_heading)

        self.assertTrue(torch.allclose(recovered, local, atol=1.0e-5))

    def test_temporal_ensemble_averages_overlapping_world_chunks(self):
        from smart.model.smart_continuous_action_diffusion import (
            SMARTContinuousActionDiffusion,
        )

        model = object.__new__(SMARTContinuousActionDiffusion)
        model.ar_prediction_tokens = 4
        model.ar_commit_tokens = 1
        model.ar_token_steps = 2
        model.continuous_action_temporal_ensemble_enabled = True
        model.continuous_action_temporal_ensemble_decay = 1.0
        model._reset_continuous_action_ensemble()
        generation_agents = torch.tensor([True])
        valid = torch.ones(1, 4, 2, dtype=torch.bool)

        first_window = torch.zeros(1, 4, 2, 2)
        first_window[:, 1] = 2.0
        commit0, valid0 = model._select_continuous_action_commit(
            first_window,
            valid,
            generation_agents,
            round_idx=0,
            rounds=4,
        )
        self.assertTrue(torch.allclose(commit0, torch.zeros_like(commit0)))
        self.assertTrue(valid0.all())

        second_window = torch.zeros(1, 4, 2, 2)
        second_window[:, 0] = 4.0
        commit1, valid1 = model._select_continuous_action_commit(
            second_window,
            valid,
            generation_agents,
            round_idx=1,
            rounds=4,
        )

        self.assertTrue(torch.allclose(commit1, torch.full_like(commit1, 3.0)))
        self.assertTrue(valid1.all())

    def test_sampling_final_step_returns_clean_action(self):
        from smart.model.smart_continuous_action_diffusion import (
            SMARTContinuousActionDiffusion,
        )

        model = object.__new__(SMARTContinuousActionDiffusion)
        model.ar_token_steps = 2
        model.continuous_action_sigma_min = 0.25
        model.continuous_action_sigma_max = 1.0
        model.continuous_action_sample_steps = 1

        clean = torch.full((1, 2, 2, 2), 3.0)

        def denoise(_action, _packed, _summary, _t):
            return clean

        model._continuous_action_denoise = denoise
        packed = {'valid_mask': torch.tensor([[True, False]])}
        summary = torch.zeros(1, 4)

        sampled = model._sample_continuous_actions(packed, summary)

        self.assertTrue(torch.allclose(sampled[:, :1], clean[:, :1]))
        self.assertTrue(torch.equal(sampled[:, 1:], torch.zeros_like(sampled[:, 1:])))


class SMARTContinuousActionDiffusionConfigTest(unittest.TestCase):
    def test_1000_step_config_selects_continuous_action_predictor(self):
        config = load_config_act(
            "configs/train/train_scalable_continuous_action_diffusion_1000.yaml"
        )

        self.assertEqual(config.Model.predictor, "smart_continuous_action_diffusion")
        self.assertEqual(config.Trainer.max_steps, 1000)
        self.assertEqual(config.Trainer.val_check_interval, 1000)
        self.assertEqual(config.Trainer.checkpoint_every_n_train_steps, 1000)
        self.assertEqual(
            config.Model.diffusion.continuous_action_objective,
            "diffusion_policy_v1",
        )
        self.assertTrue(config.Model.diffusion.continuous_action_temporal_ensemble_enabled)
        self.assertEqual(config.Model.diffusion.continuous_action_sample_steps, 8)
        self.assertEqual(config.Model.diffusion.dense_smart_ce_loss_weight, 1.0)

    def test_server_config_uses_full_waymo_paths(self):
        config = load_config_act(
            "configs/train/train_scalable_continuous_action_diffusion.yaml"
        )

        self.assertEqual(config.Model.predictor, "smart_continuous_action_diffusion")
        self.assertEqual(config.Trainer.devices, 14)
        self.assertEqual(
            config.Dataset.train_raw_dir,
            ["/raid/haoq_lab/wangshijie/data/waymo/training"],
        )
        self.assertEqual(
            config.Dataset.val_raw_dir,
            ["/raid/haoq_lab/wangshijie/data/waymo/validation"],
        )


if __name__ == "__main__":
    unittest.main()
