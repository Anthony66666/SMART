import unittest

import pytorch_lightning as pl
import torch

from smart.model.smart import SMART
from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.model.smart_causal_diffusion import SMARTCausalDiffusion
from smart.model.smart_diffusion import SMARTDiffusion
from smart.utils.config import load_config_act


class HybridDiffusionInheritanceTest(unittest.TestCase):
    def test_hybrid_predictor_is_lightning_without_existing_predictor_inheritance(self):
        from smart.model.smart_hybrid_diffusion import SMARTHybridDiffusion

        self.assertTrue(issubclass(SMARTHybridDiffusion, pl.LightningModule))
        for parent in (
            SMART,
            SMARTDiffusion,
            SMARTAutoregressiveDiffusion,
            SMARTCausalDiffusion,
        ):
            self.assertFalse(issubclass(SMARTHybridDiffusion, parent))


class HybridDiffusionSpeedEnergyTest(unittest.TestCase):
    def test_balanced_commit_speed_energy_penalizes_overfast_and_slow_tokens(self):
        from smart.model.smart_hybrid_diffusion import SMARTHybridDiffusion

        model = object.__new__(SMARTHybridDiffusion)
        model.ar_token_steps = 5
        model.commit_speed_threshold = 1.0
        model.commit_min_speed_ratio = 0.75
        model.commit_max_speed_ratio = 1.25

        candidate_positions = torch.tensor([[
            [[0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]],
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0]],
        ]])

        energy = model._balanced_commit_speed_energy(
            candidate_positions=candidate_positions,
            anchor_positions=torch.zeros(1, 2),
            current_velocities=torch.tensor([[10.0, 0.0]]),
            frontier_chunk_ids=torch.tensor([0]),
            reference_speeds=torch.tensor([10.0]),
        )

        self.assertGreater(float(energy[0, 0]), 0.0)
        self.assertAlmostEqual(float(energy[0, 1]), 0.0, places=5)
        self.assertGreater(float(energy[0, 2]), 0.0)

    def test_balanced_commit_speed_energy_only_applies_to_executed_frontier(self):
        from smart.model.smart_hybrid_diffusion import SMARTHybridDiffusion

        model = object.__new__(SMARTHybridDiffusion)
        model.ar_token_steps = 5
        model.commit_speed_threshold = 1.0
        model.commit_min_speed_ratio = 0.75
        model.commit_max_speed_ratio = 1.25

        candidate_positions = torch.tensor([[
            [[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0], [10.0, 0.0]],
        ]])

        energy = model._balanced_commit_speed_energy(
            candidate_positions=candidate_positions,
            anchor_positions=torch.zeros(1, 2),
            current_velocities=torch.tensor([[10.0, 0.0]]),
            frontier_chunk_ids=torch.tensor([1]),
            reference_speeds=torch.tensor([10.0]),
        )

        self.assertTrue(torch.equal(energy, torch.zeros_like(energy)))


class HybridDiffusionConfigTest(unittest.TestCase):
    def test_local_config_selects_hybrid_predictor_and_original_smart_input(self):
        config = load_config_act(
            "configs/train/train_scalable_hybrid_diffusion_local.yaml"
        )

        self.assertEqual(config.Model.predictor, "smart_hybrid_diffusion")
        self.assertEqual(config.Model.diffusion.hybrid_objective, "closed_loop_frontier_v1")
        self.assertEqual(config.Model.diffusion.prediction_tokens, 4)
        self.assertEqual(config.Model.diffusion.commit_tokens, 1)
        self.assertTrue(config.Model.diffusion.use_original_smart_input)


if __name__ == "__main__":
    unittest.main()
