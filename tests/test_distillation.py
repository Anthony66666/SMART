import unittest

import torch

from smart.model.distillation import (
    aggregate_rollout_weights,
    catk_recovery_targets,
    masked_entropy,
    masked_kl_div,
    rollout_quality_weights,
)


class DistillationTest(unittest.TestCase):

    def test_masked_kl_ignores_invalid_positions(self):
        student = torch.tensor([
            [[2.0, 0.0], [0.0, 2.0]],
            [[0.0, 2.0], [2.0, 0.0]],
        ])
        teacher = torch.tensor([
            [[0.0, 2.0], [0.0, 2.0]],
            [[0.0, 2.0], [0.0, 2.0]],
        ])
        mask = torch.tensor([[True, False], [False, False]])
        loss = masked_kl_div(student, teacher, mask, temperature=1.0)
        expected = masked_kl_div(student[:1, :1], teacher[:1, :1], torch.ones(1, 1, dtype=torch.bool), 1.0)
        self.assertTrue(torch.allclose(loss, expected))

    def test_catk_recovery_targets_choose_nearest_candidate(self):
        candidate_idx = torch.tensor([[[[7, 9]]]])
        candidate_trajs = torch.tensor([[[[[[10.0, 0.0], [10.0, 0.0]],
                                           [[1.0, 0.0], [1.0, 0.0]]]]]])
        gt = torch.tensor([[[1.2, 0.0], [1.1, 0.0]]])
        valid = torch.tensor([[True, True]])
        targets, mask, _ = catk_recovery_targets(candidate_idx, candidate_trajs, gt, valid)
        self.assertEqual(targets.item(), 9)
        self.assertTrue(mask.item())

    def test_rollout_weights_are_normalized_and_finite(self):
        pred = torch.tensor([
            [[[0.0, 0.0], [1.0, 0.0]]],
            [[[0.0, 0.0], [4.0, 0.0]]],
        ])
        head = torch.zeros(2, 1, 2)
        gt = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]])
        valid = torch.tensor([[True, True]])
        rewards, _ = rollout_quality_weights(
            pred, head, gt, valid, {'gt_ade': 1.0, 'diversity': 0.1}
        )
        mask = torch.ones(2, 1, 1, dtype=torch.bool)
        weights = aggregate_rollout_weights(rewards, mask, temperature=1.0)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.allclose(weights.sum(dim=0), torch.ones(1, 1)))

    def test_masked_entropy_shape(self):
        logits = torch.randn(2, 3, 5)
        mask = torch.tensor([[True, False, True], [False, True, True]])
        entropy = masked_entropy(logits, mask)
        self.assertEqual(entropy.dim(), 0)
        self.assertTrue(torch.isfinite(entropy))


if __name__ == '__main__':
    unittest.main()
