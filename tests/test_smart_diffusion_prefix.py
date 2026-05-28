import math
import unittest
from types import MethodType, SimpleNamespace

import torch

from smart.model.smart_diffusion import SMARTDiffusion


def _diffusion_shell():
    return object.__new__(SMARTDiffusion)


class SMARTDiffusionPrefixMaskTest(unittest.TestCase):
    def test_prefix_mask_closure_masks_later_valid_chunks_for_same_agent(self):
        model = _diffusion_shell()
        valid_mask = torch.tensor([[True, True, True, True, True, False, False]])
        token_agent_ids = torch.tensor([[10, 10, 10, 11, 11, 11, -1]])
        chunk_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 0]])
        mask = torch.tensor([[False, True, False, False, False, True, True]])

        closed = model._apply_prefix_mask_closure(
            mask,
            valid_mask,
            token_agent_ids,
            chunk_ids,
        )

        expected = torch.tensor([[False, True, True, False, False, False, False]])
        self.assertTrue(torch.equal(closed, expected))

    def test_prefix_frontier_selects_first_masked_valid_chunk_per_agent(self):
        model = _diffusion_shell()
        valid_mask = torch.tensor([[True, True, True, True, True, False, False]])
        token_agent_ids = torch.tensor([[10, 10, 10, 11, 11, 11, -1]])
        chunk_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 0]])
        mask = torch.tensor([[False, True, True, True, True, False, False]])

        frontier = model._prefix_frontier_mask(
            mask,
            valid_mask,
            token_agent_ids,
            chunk_ids,
        )

        expected = torch.tensor([[False, True, False, True, False, False, False]])
        self.assertTrue(torch.equal(frontier, expected))

    def test_prefix_frontier_ignores_invalid_gaps_before_masked_chunks(self):
        model = _diffusion_shell()
        valid_mask = torch.tensor([[False, True, True]])
        token_agent_ids = torch.tensor([[7, 7, 7]])
        chunk_ids = torch.tensor([[0, 1, 2]])
        mask = torch.tensor([[False, False, True]])

        frontier = model._prefix_frontier_mask(
            mask,
            valid_mask,
            token_agent_ids,
            chunk_ids,
        )

        self.assertTrue(torch.equal(frontier, torch.tensor([[False, False, True]])))


class SMARTDiffusionCausalNoiseScheduleTest(unittest.TestCase):
    def _causal_shell(self):
        model = _diffusion_shell()
        model.training = False
        model.causal_noise_schedule = True
        model.causal_chunk_mask_probs = (0.20, 0.45, 0.70, 0.90)
        model.causal_loss_weights = (1.0, 0.8, 0.4, 0.2)
        model.prefix_constrained_training = False
        model.low_variance_masking = False
        model.mask_count_mode = 'bernoulli'
        model.use_proposal_geometry = False
        model.self_condition_prob = 0.0
        model.diffusion_decoder = SimpleNamespace(mask_token_id=99)
        model.noise_schedule = lambda t: (
            torch.full_like(t, math.log(2.0)),
            torch.full_like(t, 0.5),
            torch.ones_like(t),
        )
        model._sample_diffusion_timesteps = MethodType(
            lambda self, batch_size, device: torch.full((batch_size,), 0.5, device=device),
            model,
        )
        model._maybe_apply_geometry_dropout = MethodType(lambda self, known, gt: known, model)
        return model

    def test_causal_noise_schedule_passes_chunk_mask_probs_to_training_mask(self):
        model = self._causal_shell()
        captured = {}

        def fake_sample_training_mask(self, valid_mask, mask_prob, step=None, rank=None):
            captured['mask_prob'] = mask_prob.detach().clone()
            return torch.ones_like(valid_mask)

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
            return torch.zeros(*noisy.shape, 2, device=noisy.device)

        model._sample_training_mask = MethodType(fake_sample_training_mask, model)
        model._decode_diffusion_logits = MethodType(fake_decode, model)
        packed = {
            'token_ids': torch.zeros(1, 4, dtype=torch.long),
            'valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'loss_mask_base': torch.ones(1, 4, dtype=torch.bool),
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        model._compute_diffusion_loss(packed, torch.zeros(1, 1))

        self.assertTrue(torch.allclose(
            captured['mask_prob'],
            torch.tensor([[0.20, 0.45, 0.70, 0.90]]),
        ))

    def test_causal_loss_weights_downweight_far_chunks(self):
        model = self._causal_shell()
        model.causal_chunk_mask_probs = (1.0, 1.0, 1.0, 1.0)
        model.causal_loss_weights = (1.0, 0.5, 0.25, 0.125)
        model._sample_training_mask = MethodType(
            lambda self, valid_mask, mask_prob, step=None, rank=None: torch.ones_like(valid_mask),
            model,
        )
        model._decode_diffusion_logits = MethodType(
            lambda self, noisy, packed, summary, t, geometry_known_mask, **_kwargs: torch.zeros(*noisy.shape, 2),
            model,
        )
        packed = {
            'token_ids': torch.zeros(1, 4, dtype=torch.long),
            'valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'loss_mask_base': torch.ones(1, 4, dtype=torch.bool),
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        loss, _acc = model._compute_diffusion_loss(packed, torch.zeros(1, 1))

        expected = math.log(2.0) * (1.0 + 0.5 + 0.25 + 0.125) / 4.0
        self.assertAlmostEqual(float(loss.item()), expected, places=6)


if __name__ == '__main__':
    unittest.main()
