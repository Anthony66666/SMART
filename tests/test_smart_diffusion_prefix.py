import unittest

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


if __name__ == '__main__':
    unittest.main()
