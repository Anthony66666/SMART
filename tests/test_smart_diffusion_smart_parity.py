import math
import pickle
import importlib
import unittest
from pathlib import Path
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


def _real_token_diffusion_shell():
    token_path = Path(__file__).resolve().parents[1] / 'smart' / 'tokens' / 'cluster_frame_5_2048.pkl'
    if not token_path.exists():
        raise unittest.SkipTest(f'real SMART token vocab not found: {token_path}')
    with token_path.open('rb') as f:
        token_data = pickle.load(f)

    model = _diffusion_shell()
    token_all = {
        k: torch.from_numpy(v).clone().to(dtype=torch.float)
        for k, v in token_data['token_all'].items()
    }
    endpoint = {
        k: torch.from_numpy(v).clone().to(dtype=torch.float)
        for k, v in token_data['token'].items()
    }
    token_size = int(next(iter(endpoint.values())).shape[0])
    model.model_config = SimpleNamespace(decoder=SimpleNamespace(token_size=token_size))
    model.diffusion_decoder = SimpleNamespace(mask_token_id=token_size)
    model.future_chunk_steps = 5
    model.num_future_chunks = 4
    model.num_future_steps = model.future_chunk_steps * model.num_future_chunks
    model._token_vocab_cache = token_all
    model._token_endpoint_vocab_cache = endpoint
    return model


def _agent_data():
    data = HeteroData()
    valid_mask = torch.zeros(4, 12, dtype=torch.bool)
    valid_mask[:, 10] = torch.tensor([True, True, False, True])
    data['agent']['valid_mask'] = valid_mask
    data['agent']['category'] = torch.tensor([3, 0, 3, 3])
    data['agent']['type'] = torch.tensor([0, 1, 2, 3])
    return data


def _wrap_angle(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _corners_heading(corners):
    diff_xy = corners[..., 0, :] - corners[..., 3, :]
    return torch.atan2(diff_xy[..., 1], diff_xy[..., 0])


def _row_rotate(points, heading):
    cos, sin = heading.cos(), heading.sin()
    rot = torch.stack((
        torch.stack((cos, sin), dim=-1),
        torch.stack((-sin, cos), dim=-1),
    ), dim=-2)
    return torch.matmul(points, rot)


def _local_emitted_corners(model, type_name, token_id):
    traj = model.token_vocab[type_name][token_id]
    endpoint = model.token_endpoint_vocab[type_name][token_id]
    smart_traj = torch.cat([traj[:model.future_chunk_steps], endpoint[None]], dim=0)
    return smart_traj[1:1 + model.future_chunk_steps]


def _select_real_motion_tokens(model, type_name):
    token_all = model.token_vocab[type_name]
    endpoint = model.token_endpoint_vocab[type_name]
    start_center = token_all[:, 0].mean(dim=1)
    endpoint_center = endpoint.mean(dim=1)
    displacement = torch.norm(endpoint_center - start_center, dim=-1)
    fast_id = int(torch.argmax(displacement).item())

    smart_traj = torch.cat(
        [token_all[:, :model.future_chunk_steps], endpoint[:, None]],
        dim=1,
    )
    emitted = smart_traj[:, 1:1 + model.future_chunk_steps]
    heading = _corners_heading(emitted)
    heading_delta = _wrap_angle(heading[:, -1] - heading[:, 0]).abs()
    moving = displacement > torch.quantile(displacement, 0.50)
    turn_id = int(torch.argmax(heading_delta.masked_fill(~moving, -1.0)).item())
    return fast_id, turn_id


def _manual_real_token_sequence(model, type_name, token_seq, start_pos, start_heading):
    pos = start_pos.clone()
    heading = start_heading.clone()
    query_pos = []
    query_heading = []
    frames = []
    frame_headings = []
    for token_id in token_seq:
        query_pos.append(pos.clone())
        query_heading.append(heading.clone())
        local = _local_emitted_corners(model, type_name, token_id)
        local_centers = local.mean(dim=1)
        local_heading = _corners_heading(local)
        world = _row_rotate(local_centers, heading) + pos
        world_heading = _wrap_angle(local_heading + heading)
        frames.append(world)
        frame_headings.append(world_heading)
        pos = world[-1].clone()
        heading = world_heading[-1].clone()
    return {
        'query_pos': torch.stack(query_pos),
        'query_heading': torch.stack(query_heading),
        'frames': torch.cat(frames, dim=0),
        'frame_heading': torch.cat(frame_headings, dim=0),
    }


def _decode_real_token_sequence(model, type_id, token_seq, start_pos, start_heading):
    pos = start_pos[None].clone()
    heading = start_heading[None].clone()
    frames = []
    frame_headings = []
    for token_id in token_seq:
        world, world_heading = model._token_chunk_world(
            torch.tensor([token_id], dtype=torch.long),
            torch.tensor([type_id], dtype=torch.long),
            pos,
            heading,
        )
        frames.append(world[0])
        frame_headings.append(world_heading[0])
        pos = world[:, -1].clone()
        heading = world_heading[:, -1].clone()
    return torch.cat(frames, dim=0), torch.cat(frame_headings, dim=0)


def _refresh_real_token_geometry(model, type_id, token_seq, start_pos, start_heading, mode):
    chunks = len(token_seq)
    packed = {
        'token_positions': torch.zeros(1, chunks, 2),
        'token_headings': torch.zeros(1, chunks),
        'valid_mask': torch.ones(1, chunks, dtype=torch.bool),
        'agent_maps': [(0, 0, torch.tensor([0]))],
        'agent_start_positions': start_pos[None].clone(),
        'agent_start_headings': start_heading[None].clone(),
        'agent_types_global': torch.tensor([type_id]),
    }
    if mode == 'all_known':
        return model._refresh_token_geometry(torch.tensor([token_seq]), packed)
    token_ids = torch.tensor([[token_seq[0]] + [model.mask_token_id] * (chunks - 1)])
    known = torch.tensor([[True] + [False] * (chunks - 1)])
    if mode == 'chunk0_only':
        return model._refresh_token_geometry(
            token_ids,
            packed,
            geometry_known_mask=known,
        )
    if mode == 'tail_proposal':
        return model._refresh_token_geometry(
            token_ids,
            packed,
            geometry_known_mask=known,
            proposal_token_ids=torch.tensor([token_seq]),
            proposal_confidence=torch.tensor([[0.0] + [0.7] * (chunks - 1)]),
        )
    raise ValueError(mode)


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

    def test_refresh_geometry_does_not_advance_after_unknown_chunk_gap(self):
        model = _diffusion_shell()
        model.num_future_chunks = 3
        token_all = torch.zeros(1, 6, 4, 2)
        endpoint = torch.zeros(1, 4, 2)
        endpoint[0, :, 0] = 10.0
        model._token_vocab_cache = {'veh': token_all, 'ped': token_all, 'cyc': token_all}
        model._token_endpoint_vocab_cache = {'veh': endpoint, 'ped': endpoint, 'cyc': endpoint}
        packed = {
            'token_positions': torch.zeros(1, 3, 2),
            'token_headings': torch.zeros(1, 3),
            'valid_mask': torch.tensor([[True, True, True]]),
            'agent_maps': [(0, 0, torch.tensor([0]))],
            'agent_start_positions': torch.zeros(1, 2),
            'agent_start_headings': torch.zeros(1),
            'agent_types_global': torch.tensor([0]),
        }

        positions, _headings, conf = model._refresh_token_geometry(
            torch.tensor([[0, model.mask_token_id, 0]]),
            packed,
            geometry_known_mask=torch.tensor([[True, False, True]]),
        )

        self.assertTrue(torch.equal(conf, torch.tensor([[1.0, 0.0, 0.0]])))
        self.assertAlmostEqual(float(positions[0, 1, 0]), 10.0, places=5)
        self.assertAlmostEqual(float(positions[0, 2, 0]), 10.0, places=5)

    def test_real_token_four_chunk_geometry_and_heading_are_query_relative(self):
        model = _real_token_diffusion_shell()
        start_pos = torch.tensor([3.0, -2.0])
        heading_values = [0.0, math.pi / 6.0, math.pi / 2.0, -math.pi / 2.0, math.pi]

        for type_name, type_id in [('veh', 0), ('ped', 1), ('cyc', 2)]:
            fast_id, turn_id = _select_real_motion_tokens(model, type_name)
            token_seq = [fast_id, turn_id, fast_id, turn_id]
            for heading_value in heading_values:
                with self.subTest(type=type_name, heading=heading_value):
                    start_heading = torch.tensor(heading_value)
                    ref = _manual_real_token_sequence(
                        model,
                        type_name,
                        token_seq,
                        start_pos,
                        start_heading,
                    )
                    frames, frame_heading = _decode_real_token_sequence(
                        model,
                        type_id,
                        token_seq,
                        start_pos,
                        start_heading,
                    )
                    self.assertLessEqual(float((frames - ref['frames']).abs().max()), 5e-5)
                    self.assertLessEqual(
                        float(_wrap_angle(frame_heading - ref['frame_heading']).abs().max()),
                        1e-5,
                    )

                    pos_all, head_all, conf_all = _refresh_real_token_geometry(
                        model,
                        type_id,
                        token_seq,
                        start_pos,
                        start_heading,
                        'all_known',
                    )
                    self.assertLessEqual(float((pos_all[0] - ref['query_pos']).abs().max()), 5e-5)
                    self.assertLessEqual(
                        float(_wrap_angle(head_all[0] - ref['query_heading']).abs().max()),
                        1e-5,
                    )
                    self.assertTrue(torch.equal(conf_all, torch.ones_like(conf_all)))

                    pos_prop, head_prop, conf_prop = _refresh_real_token_geometry(
                        model,
                        type_id,
                        token_seq,
                        start_pos,
                        start_heading,
                        'tail_proposal',
                    )
                    self.assertLessEqual(float((pos_prop[0] - ref['query_pos']).abs().max()), 5e-5)
                    self.assertLessEqual(
                        float(_wrap_angle(head_prop[0] - ref['query_heading']).abs().max()),
                        1e-5,
                    )
                    self.assertTrue(torch.equal(conf_prop[0] > 0.0, torch.ones(4, dtype=torch.bool)))

                    pos_mask, head_mask, conf_mask = _refresh_real_token_geometry(
                        model,
                        type_id,
                        token_seq,
                        start_pos,
                        start_heading,
                        'chunk0_only',
                    )
                    expected_pos = ref['query_pos'].clone()
                    expected_heading = ref['query_heading'].clone()
                    expected_pos[2:] = ref['query_pos'][1]
                    expected_heading[2:] = ref['query_heading'][1]
                    self.assertLessEqual(float((pos_mask[0] - expected_pos).abs().max()), 5e-5)
                    self.assertLessEqual(
                        float(_wrap_angle(head_mask[0] - expected_heading).abs().max()),
                        1e-5,
                    )
                    self.assertTrue(torch.equal(
                        conf_mask,
                        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    ))

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

    def test_validation_eval_valid_mask_matches_official_smart_without_pred_filter(self):
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
                [False, False] + [True] * 8,
            ]),
        }

        eval_valid = model._validation_eval_valid_mask(data, pred)

        self.assertTrue(eval_valid[1, 0])
        self.assertFalse(eval_valid[1, 1])
        self.assertTrue(eval_valid[1, 2])

    def test_validation_inference_is_not_limited_to_first_two_batches(self):
        model = _diffusion_shell()
        model.inference_token = True
        model.diffusion_eval_batches = 0

        self.assertTrue(model._should_run_validation_inference(batch_idx=999))


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
