import unittest

import torch

from smart.model.smart_action_chunk_diffusion import SMARTActionChunkDiffusion
from smart.utils.config import load_config_act


def _action_chunk_shell():
    model = object.__new__(SMARTActionChunkDiffusion)
    model.ar_prediction_tokens = 4
    model.ar_commit_tokens = 1
    model.ar_total_rollout_steps = 20
    model.action_chunk_temporal_ensemble_enabled = True
    model.action_chunk_temporal_ensemble_decay = 0.8
    model.action_chunk_temporal_ensemble_confidence_floor = 1.0e-4
    model.action_chunk_shift_consistency_temperature = 1.0
    model._reset_action_chunk_ensemble()
    return model


class SMARTActionChunkDiffusionTemporalEnsembleTest(unittest.TestCase):
    def test_temporal_ensemble_reuses_previous_tail_for_current_commit(self):
        model = _action_chunk_shell()
        valid = torch.ones(1, 4, dtype=torch.bool)
        generation_agents = torch.tensor([True])

        round0_tokens = torch.tensor([[10, 42, 44, 45]])
        round0_confidence = torch.tensor([[0.8, 0.95, 0.7, 0.6]])
        committed0, confidence0 = model._select_ar_committed_tokens(
            round0_tokens,
            round0_confidence,
            valid,
            generation_agents,
            round_idx=0,
            rounds=4,
        )

        self.assertTrue(torch.equal(committed0, torch.tensor([[10]])))
        self.assertTrue(torch.allclose(confidence0, torch.tensor([[0.8]])))

        round1_tokens = torch.tensor([[99, 50, 51, 52]])
        round1_confidence = torch.tensor([[0.1, 0.9, 0.8, 0.7]])
        committed1, confidence1 = model._select_ar_committed_tokens(
            round1_tokens,
            round1_confidence,
            valid,
            generation_agents,
            round_idx=1,
            rounds=4,
        )

        self.assertTrue(torch.equal(committed1, torch.tensor([[42]])))
        self.assertGreater(float(confidence1[0, 0]), float(round1_confidence[0, 0]))

    def test_shift_consistency_penalizes_mismatched_overlap_distributions(self):
        model = _action_chunk_shell()
        current_logits = torch.zeros(2, 4, 5)
        next_logits = torch.zeros(2, 4, 5)
        current_valid = torch.ones(2, 4, dtype=torch.bool)
        next_valid = torch.ones(2, 4, dtype=torch.bool)
        current_agent_ids = torch.tensor([3, 7])
        next_agent_ids = torch.tensor([7, 3])

        current_logits[0, 1, 1] = 8.0
        next_logits[1, 0, 1] = 8.0
        current_logits[1, 1, 2] = 8.0
        next_logits[0, 0, 4] = 8.0

        loss = model._action_chunk_shift_consistency_loss_from_logits(
            current_logits,
            current_valid,
            current_agent_ids,
            next_logits,
            next_valid,
            next_agent_ids,
        )

        self.assertGreater(float(loss), 0.5)

        next_logits[0, 0] = current_logits[1, 1]
        matched_loss = model._action_chunk_shift_consistency_loss_from_logits(
            current_logits,
            current_valid,
            current_agent_ids,
            next_logits,
            next_valid,
            next_agent_ids,
        )

        self.assertLess(float(matched_loss), float(loss))

    def test_shift_consistency_aligns_packed_rows_by_scene_id(self):
        model = _action_chunk_shell()
        current_logits = torch.zeros(2, 2, 5)
        next_logits = torch.zeros(2, 2, 5)
        current_valid = torch.ones(2, 2, dtype=torch.bool)
        next_valid = torch.ones(2, 2, dtype=torch.bool)
        current_agent_ids = torch.tensor([[1, 1], [9, 9]])
        next_agent_ids = torch.tensor([[9, 9], [1, 1]])
        current_chunk_ids = torch.tensor([[0, 1], [0, 1]])
        next_chunk_ids = torch.tensor([[0, 1], [0, 1]])
        current_scene_ids = torch.tensor([10, 20])
        next_scene_ids = torch.tensor([20, 10])

        current_logits[0, 1, 3] = 8.0
        next_logits[1, 0, 3] = 8.0
        current_logits[1, 1, 2] = 8.0
        next_logits[0, 0, 2] = 8.0

        loss = model._action_chunk_shift_consistency_loss_from_logits(
            current_logits,
            current_valid,
            current_agent_ids,
            next_logits,
            next_valid,
            next_agent_ids,
            current_chunk_ids,
            next_chunk_ids,
            current_scene_ids,
            next_scene_ids,
        )

        self.assertLess(float(loss), 1.0e-3)


class SMARTActionChunkDiffusionConfigTest(unittest.TestCase):
    def test_1000_step_config_selects_action_chunk_predictor(self):
        config = load_config_act(
            "configs/train/train_scalable_ar_action_chunk_1000.yaml"
        )

        self.assertEqual(config.Model.predictor, "smart_action_chunk_diffusion")
        self.assertEqual(config.Trainer.max_steps, 1000)
        self.assertEqual(config.Trainer.val_check_interval, 1000)
        self.assertEqual(config.Trainer.checkpoint_every_n_train_steps, 1000)
        self.assertTrue(config.Model.diffusion.temporal_ensemble_enabled)
        self.assertEqual(config.Model.diffusion.temporal_ensemble_decay, 0.8)
        self.assertFalse(config.Model.diffusion.carry_tail_proposal)
        self.assertFalse(config.Model.diffusion.proposal_conditioning_enabled)
        self.assertEqual(config.Model.diffusion.dense_smart_ce_loss_weight, 1.0)
        self.assertEqual(
            config.Model.diffusion.action_chunk_shift_consistency_loss_weight,
            0.1,
        )
        self.assertEqual(config.Model.diffusion.causal_loss_weights, [1.0, 1.0, 1.0, 1.0])
        self.assertEqual(
            config.Visualization.output_dir,
            "./outputs/val_ar_action_chunk_1000",
        )
        self.assertEqual(
            config.Visualization.step_viz.output_dir,
            "./outputs/step_ar_action_chunk_1000",
        )

    def test_server_config_enables_action_chunk_quality_priors(self):
        config = load_config_act(
            "configs/train/train_scalable_ar_action_chunk.yaml"
        )

        self.assertEqual(config.Model.predictor, "smart_action_chunk_diffusion")
        self.assertEqual(config.Trainer.devices, 14)
        self.assertEqual(config.Dataset.train_raw_dir, ["/raid/haoq_lab/wangshijie/data/waymo/training"])
        self.assertTrue(config.Model.diffusion.temporal_ensemble_enabled)
        self.assertFalse(config.Model.diffusion.carry_tail_proposal)
        self.assertFalse(config.Model.diffusion.proposal_conditioning_enabled)
        self.assertEqual(config.Model.diffusion.dense_smart_ce_loss_weight, 1.0)
        self.assertEqual(config.Model.diffusion.dense_smart_ce_interval, 8)
        self.assertEqual(
            config.Model.diffusion.action_chunk_shift_consistency_loss_weight,
            0.1,
        )
        self.assertEqual(
            config.Model.diffusion.action_chunk_shift_consistency_interval,
            2,
        )


if __name__ == "__main__":
    unittest.main()
