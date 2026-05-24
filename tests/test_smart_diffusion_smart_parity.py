import importlib
import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from smart.callbacks.validation_visualization import (
    _future_gt_valid_mask,
    _prediction_valid_mask,
    _visualized_agent_mask,
)
from smart.model.smart_diffusion import SMARTDiffusion


def _diffusion_shell():
    model = object.__new__(SMARTDiffusion)
    model.model_config = SimpleNamespace(decoder=SimpleNamespace(token_size=1))
    model.diffusion_decoder = SimpleNamespace(mask_token_id=1)
    model.num_historical_steps = 11
    model.num_future_steps = 10
    model.future_chunk_steps = 5
    model.num_future_chunks = 2
    model.agent_selection_mode = 'smart_inference'
    model.supervision_mode = 'smart_category3'
    model.metric_mode = 'smart_val_compatible'
    return model


def _agent_data():
    data = HeteroData()
    valid_mask = torch.zeros(4, 12, dtype=torch.bool)
    valid_mask[:, 10] = torch.tensor([True, True, False, True])
    data['agent']['valid_mask'] = valid_mask
    data['agent']['category'] = torch.tensor([3, 0, 3, 3])
    data['agent']['type'] = torch.tensor([0, 1, 2, 3])
    return data


class SMARTDiffusionSMARTParityTest(unittest.TestCase):
    def test_agent_masks_match_original_smart_roles(self):
        model = _diffusion_shell()
        data = _agent_data()

        generation = model._generation_agent_mask(data)
        supervision = model._supervision_agent_mask(data)
        metric = model._metric_agent_mask(data)
        category_metric = model._metric_agent_mask(data, mode='smart_category3')

        self.assertTrue(torch.equal(
            generation,
            torch.tensor([True, True, False, True]),
        ))
        self.assertTrue(torch.equal(
            supervision,
            torch.tensor([True, False, False, True]),
        ))
        self.assertTrue(torch.equal(
            metric,
            torch.tensor([True, True, False, True]),
        ))
        self.assertTrue(torch.equal(
            category_metric,
            torch.tensor([True, False, False, True]),
        ))

    def test_rollout_future_targets_do_not_require_gt_future_validity(self):
        model = _diffusion_shell()
        object.__setattr__(model, 'encoder', SimpleNamespace(agent_encoder=SimpleNamespace(shift=5)))
        data = _agent_data()
        data['agent']['token_idx'] = torch.zeros(4, 4, dtype=torch.long)
        data['agent']['agent_valid_mask'] = torch.tensor([
            [True, True, True, True],
            [True, True, False, False],
            [True, True, True, True],
            [True, True, True, True],
        ])

        _tokens, valid, generation, supervision = model._build_future_token_targets(
            data,
            rollout_valid=True,
        )

        self.assertTrue(torch.equal(
            generation,
            torch.tensor([True, True, False, True]),
        ))
        self.assertTrue(valid[1].all())
        self.assertFalse(supervision[1])

    def test_pack_keeps_non_target_generation_agents_but_masks_their_loss(self):
        model = _diffusion_shell()
        tokens = torch.tensor([[1, 2], [3, 4], [5, 6]])
        valid = torch.ones(3, 2, dtype=torch.bool)
        generation = torch.tensor([True, True, True])
        supervision = torch.tensor([True, False, True])
        batch = torch.zeros(3, dtype=torch.long)
        positions = torch.zeros(3, 2)
        headings = torch.zeros(3)
        context = torch.zeros(3, 4)
        shape = torch.zeros(3, 4)
        agent_types = torch.tensor([0, 1, 2])

        packed = model._pack_diffusion_sequence(
            tokens,
            valid,
            generation,
            supervision,
            batch,
            positions,
            headings,
            context,
            agent_types,
            shape,
        )

        self.assertIsNotNone(packed)
        self.assertTrue(torch.equal(packed['valid_mask'][0, :6], torch.ones(6, dtype=torch.bool)))
        self.assertTrue(torch.equal(
            packed['loss_mask_base'][0, :6],
            torch.tensor([True, True, False, False, True, True]),
        ))

    def test_token_chunk_world_uses_smart_endpoint_token(self):
        model = _diffusion_shell()
        token_all = torch.zeros(1, 6, 4, 2)
        endpoint = torch.zeros(1, 4, 2)
        for step in range(6):
            token_all[0, step, :, 0] = float(step)
            token_all[0, step, 0, 1] = 1.0
            token_all[0, step, 3, 1] = -1.0
        token_all[0, 5, :, 0] = 99.0
        endpoint[0, :, 0] = 5.0
        endpoint[0, 0, 1] = 1.0
        endpoint[0, 3, 1] = -1.0
        model._token_vocab_cache = {'veh': token_all, 'ped': token_all, 'cyc': token_all}
        model._token_endpoint_vocab_cache = {'veh': endpoint, 'ped': endpoint, 'cyc': endpoint}

        world, _heading = model._token_chunk_world(
            torch.tensor([0]),
            torch.tensor([0]),
            torch.zeros(1, 2),
            torch.zeros(1),
        )

        self.assertAlmostEqual(float(world[0, -1, 0]), 5.0, places=5)

    def test_refresh_geometry_returns_confidence_for_masked_proposals(self):
        model = _diffusion_shell()
        token_all = torch.zeros(1, 6, 4, 2)
        endpoint = torch.zeros(1, 4, 2)
        endpoint[0, :, 0] = 1.0
        model._token_vocab_cache = {'veh': token_all, 'ped': token_all, 'cyc': token_all}
        model._token_endpoint_vocab_cache = {'veh': endpoint, 'ped': endpoint, 'cyc': endpoint}
        packed = {
            'token_positions': torch.zeros(1, 2, 2),
            'token_headings': torch.zeros(1, 2),
            'valid_mask': torch.tensor([[True, True]]),
            'agent_maps': [(0, 0, torch.tensor([0]))],
            'agent_start_positions': torch.zeros(1, 2),
            'agent_start_headings': torch.zeros(1),
            'agent_types_global': torch.tensor([0]),
        }

        _pos, _head, conf = model._refresh_token_geometry(
            torch.tensor([[model.mask_token_id, model.mask_token_id]]),
            packed,
            proposal_token_ids=torch.tensor([[0, 0]]),
            proposal_confidence=torch.tensor([[0.25, 0.75]]),
        )

        self.assertTrue(torch.equal(conf, torch.tensor([[0.25, 0.75]])))


    def test_decode_generates_non_target_agents_but_masks_metric_validity(self):
        model = _diffusion_shell()
        object.__setattr__(model, 'encoder', SimpleNamespace(agent_encoder=SimpleNamespace(shift=5)))
        token_all = torch.zeros(1, 6, 4, 2)
        endpoint = torch.zeros(1, 4, 2)
        endpoint[0, :, 0] = 1.0
        model._token_vocab_cache = {'veh': token_all, 'ped': token_all, 'cyc': token_all}
        model._token_endpoint_vocab_cache = {'veh': endpoint, 'ped': endpoint, 'cyc': endpoint}

        data = HeteroData()
        data['agent']['type'] = torch.tensor([0, 0])
        data['agent']['category'] = torch.tensor([3, 0])
        data['agent']['position'] = torch.zeros(2, 21, 2)
        data['agent']['heading'] = torch.zeros(2, 21)
        data['agent']['valid_mask'] = torch.ones(2, 21, dtype=torch.bool)
        packed = {'agent_maps': [(0, 0, torch.tensor([0, 1]))]}

        out = model._decode_trajectories(
            data=data,
            sampled_ids=torch.zeros(1, 4, dtype=torch.long),
            packed=packed,
            agents_ok=torch.tensor([True, True]),
            agent_batch=torch.zeros(2, dtype=torch.long),
            gt_tokens=torch.zeros(2, 2, dtype=torch.long),
            gt_valid=torch.ones(2, 2, dtype=torch.bool),
        )

        self.assertTrue(out['pred_valid_mask'][1].all())
        self.assertGreater(float(out['pred_traj'][1].abs().sum()), 0.0)
        self.assertFalse(out['valid_mask'][1].any())
        self.assertTrue(out['official_valid_mask'][1].all())
        self.assertTrue(out['valid_mask'][0].all())

    def test_official_future_valid_mask_ignores_category_filtered_prediction_mask(self):
        model = _diffusion_shell()
        data = HeteroData()
        data['agent']['valid_mask'] = torch.ones(2, 21, dtype=torch.bool)
        data['agent']['valid_mask'][1, 12] = False
        pred = {
            'pred_traj': torch.zeros(2, 10, 2),
            'valid_mask': torch.tensor([
                [True] * 10,
                [False] * 10,
            ]),
            'pred_valid_mask': torch.tensor([
                [True] * 10,
                [True, False] + [True] * 8,
            ]),
        }

        official = model._official_future_valid_mask(data, pred)
        eval_valid = official & pred['pred_valid_mask']

        self.assertTrue(official[1, 0])
        self.assertFalse(official[1, 1])
        self.assertFalse(eval_valid[1, 1])


class ValidationVisualizationMaskTest(unittest.TestCase):
    def _toy_data(self, categories):
        data = HeteroData()
        data['agent']['valid_mask'] = torch.ones(3, 21, dtype=torch.bool)
        data['agent']['type'] = torch.tensor([0, 0, 3])
        data['agent']['category'] = torch.tensor(categories)
        return data

    def _toy_prediction(self):
        return {
            'gt': torch.zeros(3, 10, 2),
            'pred_traj': torch.ones(3, 10, 2),
            'pred_valid_mask': torch.ones(3, 10, dtype=torch.bool),
        }

    def test_official_visualization_does_not_depend_on_category(self):
        pred = self._toy_prediction()
        first = _visualized_agent_mask(self._toy_data([3, 0, 3]), pred, 10, 'official')
        second = _visualized_agent_mask(self._toy_data([0, 3, 3]), pred, 10, 'official')

        self.assertTrue(torch.equal(first, torch.tensor([True, True, False])))
        self.assertTrue(torch.equal(second, torch.tensor([True, True, False])))

    def test_supervision_visualization_depends_on_category(self):
        pred = self._toy_prediction()
        target = _visualized_agent_mask(self._toy_data([3, 0, 3]), pred, 10, 'supervision')

        self.assertTrue(torch.equal(target, torch.tensor([True, False, False])))

    def test_non_target_agent_has_gt_and_prediction_masks_in_official_view(self):
        data = self._toy_data([3, 0, 3])
        pred = self._toy_prediction()

        gt_mask = _future_gt_valid_mask(data, pred, 11)[1]
        pred_mask = _prediction_valid_mask(data, pred, 1, 11)

        self.assertTrue(gt_mask.all())
        self.assertTrue(pred_mask.all())


class OfficialEvalCoverageTest(unittest.TestCase):
    def test_sim_agent_prediction_validation_rejects_zero_fallback(self):
        try:
            official_eval = importlib.import_module('eval_waymo_official')
        except ImportError:
            self.skipTest('Waymo official evaluation dependencies are not installed.')

        prediction = {
            'pred_traj': torch.zeros(2, 80, 2),
            'pred_head': torch.zeros(2, 80),
            'pred_valid_mask': torch.ones(2, 80, dtype=torch.bool),
        }
        data = {'agent': {'id': torch.tensor([101, 102])}}

        with self.assertRaisesRegex(ValueError, 'zero fallback'):
            official_eval._validate_sim_agent_prediction(
                prediction=prediction,
                data=data,
                object_id=101,
                agent_index=0,
                future_steps=80,
                scenario_id='scenario-for-test',
            )


if __name__ == '__main__':
    unittest.main()
