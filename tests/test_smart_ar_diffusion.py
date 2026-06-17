
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
    model.diffusion_num_steps = 32
    model.debug_validation_logging = False
    model.inference_token = False
    model.ar_local_map_radius = 2.0
    model.max_map_tokens = 0
    model.diffusion_loss_weight = 1.0
    model.causal_loss_weighting_enabled = False
    model.dense_smart_ce_loss_weight = 0.0
    model.proposal_carry_training_enabled = False
    model.proposal_carry_loss_weight = 0.0
    model.proposal_dropout_prob = 0.0
    model.proposal_noise_topk = 5
    model.proposal_confidence = 0.5
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

    def test_rescreen_map_context_keeps_full_scene_like_smart(self):
        model = _ar_shell()
        model.use_map_context = True
        model.ar_local_map_refresh = 'rescreen'
        data = HeteroData()
        data['pt_token']['position'] = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
        data['pt_token']['orientation'] = torch.zeros(3)
        data['pt_token']['num_nodes'] = 3
        ctx = {
            'x_pt': torch.arange(12, dtype=torch.float).view(3, 4),
            'pt_visibility_mask': torch.ones(3, dtype=torch.bool),
        }
        packed = {'agent_maps': [(0, 0, torch.tensor([0]))]}

        map_context, map_positions, map_orientations, map_batch, map_valid_mask = model._pack_map_context(
            data,
            ctx,
            packed,
            agent_positions=torch.tensor([[0.5, 0.0]]),
        )

        self.assertTrue(torch.equal(map_context, ctx['x_pt']))
        self.assertTrue(torch.equal(map_positions, data['pt_token']['position'][:, :2].float()))
        self.assertTrue(torch.equal(map_orientations, data['pt_token']['orientation'].float()))
        self.assertTrue(torch.equal(map_batch, torch.zeros(3, dtype=torch.long)))
        self.assertTrue(torch.equal(map_valid_mask, torch.ones(3, dtype=torch.bool)))

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


    def test_receding_horizon_commit_one_carries_tail_as_next_proposal(self):
        model = _ar_shell()
        model.ar_commit_tokens = 1
        model.ar_total_rollout_steps = 10
        model.ar_carry_tail_proposal = True
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)
        call_state = {'count': 0}
        captured_proposals = []
        test_case = self

        model._prepare_batch = MethodType(lambda self, batch: batch, model)

        def fake_build_inputs(self, rollout_view):
            test_case.assertEqual(rollout_view["agent"]["token_pos"].shape[1], 6)
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

        def fake_sample(self, **kwargs):
            proposal_ids = kwargs.get('initial_proposal_token_ids')
            proposal_confidence = kwargs.get('initial_proposal_confidence')
            captured_proposals.append((
                None if proposal_ids is None else proposal_ids.clone(),
                None if proposal_confidence is None else proposal_confidence.clone(),
            ))
            base = call_state['count'] * 10
            call_state['count'] += 1
            return (
                torch.tensor([[base, base + 1, base + 2, base + 3]]),
                torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
            )

        def fake_token_world(self, token_ids, _agent_types, positions, headings):
            step = torch.arange(1, 6, dtype=positions.dtype, device=positions.device).view(1, 5, 1)
            world = positions[:, None, :] + torch.cat([step, torch.zeros_like(step)], dim=-1)
            return world, headings[:, None].expand(-1, 5)

        model._build_diffusion_inputs = MethodType(fake_build_inputs, model)
        model._diffusion_sample = MethodType(fake_sample, model)
        model._token_chunk_world = MethodType(fake_token_world, model)

        out = model.inference(data)

        self.assertEqual(call_state['count'], 2)
        self.assertIsNone(captured_proposals[0][0])
        self.assertTrue(torch.equal(captured_proposals[1][0], torch.tensor([[1, 2, 3, 0]])))
        self.assertTrue(torch.allclose(
            captured_proposals[1][1],
            torch.tensor([[0.2, 0.3, 0.4, 0.0]]),
        ))
        self.assertTrue(torch.equal(out['next_token_idx'][0], torch.tensor([0, 10])))

    def test_inference_does_not_mutate_input_with_commit_speed_reference(self):
        model = _ar_shell()
        model.ar_commit_tokens = 1
        model.ar_total_rollout_steps = 5
        model.ar_carry_tail_proposal = False
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)

        model._prepare_batch = MethodType(lambda self, batch: batch, model)

        def fake_build_inputs(self, _rollout_view):
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
            return torch.tensor([[0, 1, 2, 3]]), torch.ones(1, 4)

        def fake_token_world(self, token_ids, _agent_types, positions, headings):
            step = torch.arange(1, 6, dtype=positions.dtype, device=positions.device).view(1, 5, 1)
            world = positions[:, None, :] + torch.cat([step, torch.zeros_like(step)], dim=-1)
            return world, headings[:, None].expand(-1, 5)

        model._build_diffusion_inputs = MethodType(fake_build_inputs, model)
        model._diffusion_sample = MethodType(fake_sample, model)
        model._token_chunk_world = MethodType(fake_token_world, model)

        model.inference(data)

        self.assertNotIn("commit_speed_reference", data["agent"])

    def test_retokenize_physical_decode_keeps_heading_for_stationary_token(self):
        model = _ar_shell()
        model.ar_prediction_tokens = 2
        model.ar_token_steps = 5
        model.retokenization_error_thresholds = (100.0, 100.0, 100.0)

        def corners(center, heading):
            forward = torch.tensor([heading.cos(), heading.sin()])
            lateral = torch.tensor([-heading.sin(), heading.cos()])
            return torch.stack([
                center + forward,
                center + lateral,
                center - forward,
                center - lateral,
            ])

        token_all = torch.zeros(3, 6, 4, 2)
        endpoint = torch.zeros(3, 4, 2)
        for step in range(6):
            token_all[0, step] = corners(torch.zeros(2), torch.tensor(torch.pi / 2))
            token_all[1, step] = corners(torch.tensor([float(step), 0.0]), torch.tensor(0.0))
            token_all[2, step] = corners(torch.tensor([0.0, -float(step)]), -torch.tensor(torch.pi / 2))
        endpoint[0] = corners(torch.zeros(2), torch.tensor(torch.pi / 2))
        endpoint[1] = corners(torch.tensor([5.0, 0.0]), torch.tensor(0.0))
        endpoint[2] = corners(torch.tensor([0.0, -5.0]), -torch.tensor(torch.pi / 2))
        model._token_vocab_cache = {'veh': token_all, 'ped': token_all, 'cyc': token_all}
        model._token_endpoint_vocab_cache = {'veh': endpoint, 'ped': endpoint, 'cyc': endpoint}
        model._token_center_vocab_cache = None

        future_positions = torch.zeros(1, 2, 5, 2)
        future_positions[0, 1, :, 0] = torch.arange(1, 6, dtype=torch.float)

        token_ids, _errors, valid, _local_endpoints = model._retokenize_future(
            future_positions=future_positions,
            future_valid=torch.ones(1, 2, 5, dtype=torch.bool),
            start_positions=torch.zeros(1, 2),
            start_headings=torch.zeros(1),
            agent_types=torch.tensor([0]),
        )

        self.assertTrue(valid.all())
        self.assertTrue(torch.equal(token_ids, torch.tensor([[0, 1]])))

    def test_safe_speed_rerank_replaces_commit_token_only(self):
        model = _ar_shell()
        model.ar_commit_tokens = 1
        model.ar_sampling_guidance_enabled = True
        model.ar_sampling_guidance_mode = "safe_speed"
        model.ar_sampling_guidance_topk = 2
        model.model_config.decoder.token_size = 5
        model.diffusion_decoder = SimpleNamespace(mask_token_id=5)
        model.min_t = 1.0e-3
        model.use_proposal_geometry = False
        model.geometry_confidence_source_threshold = 0.2
        model.lane_distance_energy_weight = 0.0
        model.lane_heading_energy_weight = 0.0
        model.dynamics_energy_weight = 0.0
        model.collision_energy_weight = 0.0
        model.commit_speed_energy_weight = 2.0
        model.commit_min_speed_ratio = 0.75
        model.commit_max_speed_ratio = 1.25
        model.commit_speed_threshold = 1.0

        class ZeroEnergy:
            dt = 0.1

            def lane_energy(self, candidate_positions, candidate_headings, *args, **kwargs):
                return (
                    candidate_positions.new_zeros(candidate_positions.shape[:2]),
                    candidate_positions.new_zeros(candidate_positions.shape[:2]),
                )

            def dynamics_energy(self, candidate_positions, candidate_headings, **kwargs):
                return candidate_positions.new_zeros(candidate_positions.shape[:2])

            def collision_energy(self, candidate_positions, nominal_other, **kwargs):
                return candidate_positions.new_zeros(candidate_positions.shape[:2])

        def fake_decode(self, sampled, packed, summary, t, geometry_known_mask, **_kwargs):
            logits = torch.full((1, 4, 5), -20.0)
            logits[0, 0, 0] = 3.0
            logits[0, 0, 1] = 2.0
            logits[0, 1:, 2] = 5.0
            return logits

        def fake_token_world(self, token_ids, _agent_types, positions, headings):
            step = torch.arange(1, 6, dtype=positions.dtype, device=positions.device).view(-1, 1)
            world = positions[:, None, :].expand(-1, 5, -1).clone()
            moving = token_ids == 1
            world[moving, :, 0] = positions[moving, 0:1] + 0.5 * step.squeeze(-1)
            return world, headings[:, None].expand(-1, 5)

        model.trajectory_energy = ZeroEnergy()
        model._decode_diffusion_logits = MethodType(fake_decode, model)
        model._token_chunk_world = MethodType(fake_token_world, model)
        packed = {
            "agent_maps": [(0, 0, torch.tensor([0]))],
            "valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "token_positions": torch.zeros(1, 4, 2),
            "token_headings": torch.zeros(1, 4),
            "token_agent_ids": torch.zeros(1, 4, dtype=torch.long),
            "chunk_ids": torch.arange(4).unsqueeze(0),
            "agent_context": torch.zeros(1, 4, 1),
            "agent_type_ids": torch.zeros(1, 4, dtype=torch.long),
            "agent_shape_embeddings": torch.zeros(1, 4, 1),
            "agent_start_positions": torch.zeros(1, 2),
            "agent_start_headings": torch.zeros(1),
            "agent_types_global": torch.zeros(1, dtype=torch.long),
            "current_velocities": torch.tensor([[[5.0, 0.0]] * 4]),
            "current_headings": torch.zeros(1, 4),
            "commit_speed_reference": torch.full((1, 4), 5.0),
        }
        sampled = torch.tensor([[0, 2, 3, 4]])
        confidence = torch.tensor([[0.9, 0.8, 0.7, 0.6]])

        reranked, reranked_confidence = model._ar_rerank_commit_tokens(
            sampled,
            confidence,
            packed,
            torch.zeros(1, 1),
        )

        self.assertTrue(torch.equal(reranked, torch.tensor([[1, 2, 3, 4]])))
        self.assertLess(float(reranked_confidence[0, 0]), 0.9)
        self.assertTrue(torch.equal(reranked_confidence[0, 1:], confidence[0, 1:]))

    def test_history_context_dropout_masks_conditioning_without_mutating_labels(self):
        model = _ar_shell()
        model.ar_history_context_dropout_enabled = True
        model.ar_history_context_dropout_prob = 1.0
        model.training = True
        data = _toy_sequence(num_agents=1, num_tokens=6, num_frames=31)
        original_valid = data["agent"]["agent_valid_mask"].clone()

        mask = model._ar_history_context_mask(data)

        self.assertFalse(mask[:, :model.ar_history_tokens].any())
        self.assertTrue(mask[:, model.ar_history_tokens:].all())
        self.assertTrue(torch.equal(data["agent"]["agent_valid_mask"], original_valid))


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

    def test_causal_loss_weights_can_apply_without_causal_noise_schedule(self):
        model = _ar_shell()
        model.causal_noise_schedule = False
        model.causal_loss_weighting_enabled = True
        model.causal_loss_weights = (2.0, 1.0, 0.5, 0.25)
        packed = {
            "valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "chunk_ids": torch.arange(4).unsqueeze(0),
        }

        weights = model._training_loss_weights(packed)

        self.assertTrue(torch.allclose(
            weights,
            torch.tensor([[2.0, 1.0, 0.5, 0.25]]),
        ))

    def test_causal_loss_weights_stay_disabled_without_explicit_gate(self):
        model = _ar_shell()
        model.causal_noise_schedule = False
        model.causal_loss_weighting_enabled = False
        model.causal_loss_weights = (2.0, 1.0, 0.5, 0.25)
        packed = {
            "valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "chunk_ids": torch.arange(4).unsqueeze(0),
        }

        weights = model._training_loss_weights(packed)

        self.assertTrue(torch.equal(weights, torch.ones(1, 4)))

    def test_diffusion_loss_accepts_forced_mask_and_initial_proposal(self):
        model = _ar_shell()
        model.training = True
        model.prefix_constrained_training = False
        model.low_variance_masking = False
        model.causal_noise_schedule = False
        model.causal_loss_weights = ()
        model.geometry_dropout_prob = 0.0
        model.use_proposal_geometry = True
        model.self_condition_prob = 1.0
        model.diffusion_decoder = SimpleNamespace(mask_token_id=5)
        model.model_config.decoder.token_size = 5
        model.noise_schedule = lambda t: (
            torch.ones_like(t),
            torch.full_like(t, 0.5),
            torch.ones_like(t),
        )
        model._sample_diffusion_timesteps = MethodType(
            lambda self, batch_size, device: torch.full((batch_size,), 0.5, device=device),
            model,
        )
        model.log = MethodType(lambda self, *args, **kwargs: None, model)
        captured = {}

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **kwargs):
            captured["noisy"] = noisy.clone()
            captured["geometry_known_mask"] = geometry_known_mask.clone()
            captured["proposal_token_ids"] = kwargs.get("proposal_token_ids").clone()
            captured["proposal_confidence"] = kwargs.get("proposal_confidence").clone()
            logits = torch.zeros(1, 4, 6)
            for idx, token_id in enumerate(packed["token_ids"][0].tolist()):
                logits[0, idx, token_id] = 5.0
            return logits

        model._decode_diffusion_logits = MethodType(fake_decode, model)
        packed = {
            "token_ids": torch.tensor([[0, 1, 2, 3]]),
            "valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "loss_mask_base": torch.ones(1, 4, dtype=torch.bool),
            "chunk_ids": torch.arange(4).unsqueeze(0),
            "token_agent_ids": torch.zeros(1, 4, dtype=torch.long),
        }
        forced_mask = torch.tensor([[True, False, True, False]])
        proposal_ids = torch.tensor([[10, 11, 12, 0]])
        proposal_confidence = torch.tensor([[0.5, 0.5, 0.5, 0.0]])

        loss, acc = model._compute_diffusion_loss(
            packed,
            torch.zeros(1, 1),
            forced_mask=forced_mask,
            initial_proposal_token_ids=proposal_ids,
            initial_proposal_confidence=proposal_confidence,
        )

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(acc), 1.0)
        self.assertTrue(torch.equal(captured["noisy"], torch.tensor([[5, 1, 5, 3]])))
        self.assertTrue(torch.equal(
            captured["geometry_known_mask"],
            torch.tensor([[False, True, False, True]]),
        ))
        self.assertTrue(torch.equal(captured["proposal_token_ids"], proposal_ids))
        self.assertTrue(torch.equal(captured["proposal_confidence"], proposal_confidence))

    def test_proposal_carry_training_view_uses_next_window_and_tail_proposal(self):
        model = _ar_shell()
        model.ar_commit_tokens = 1
        model.proposal_confidence = 0.5
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)

        def fake_corrupt(self, token_ids, valid_mask, agent_types):
            del agent_types
            return token_ids + 100, valid_mask.to(dtype=torch.float) * self.proposal_confidence

        model._corrupt_proposal_tokens = MethodType(fake_corrupt, model)

        view, proposal_ids, proposal_confidence, anchor = model._build_proposal_carry_training_view(
            data,
            anchor_token=3,
        )

        self.assertEqual(anchor, 3)
        self.assertTrue(torch.equal(view["agent"]["token_idx"][0, :6], torch.tensor([1, 2, 3, 4, 5, 6])))
        self.assertTrue(torch.equal(proposal_ids[0], torch.tensor([103, 104, 105, 0])))
        self.assertTrue(torch.allclose(
            proposal_confidence[0],
            torch.tensor([0.5, 0.5, 0.5, 0.0]),
        ))

    def test_proposal_carry_training_view_skips_terminal_incomplete_window(self):
        model = _ar_shell()
        model.ar_commit_tokens = 1
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)

        self.assertIsNone(model._select_proposal_carry_anchor(data, anchor_token=15))

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

    def test_ar_configs_keep_causal_schedule_as_disabled_ablation(self):
        disabled_paths = [
            Path("configs/train/train_scalable_ar_diffusion_baseline_1000.yaml"),
            Path("configs/train/train_scalable_ar_diffusion_local.yaml"),
        ]
        for path in disabled_paths:
            text = path.read_text()
            self.assertIn("causal_noise_schedule: false", text, str(path))
            self.assertIn("causal_chunk_mask_multipliers: [0.70, 0.90, 1.10, 1.30]", text, str(path))
            self.assertIn("causal_loss_weights: [1.0, 1.0, 0.75, 0.5]", text, str(path))
            self.assertNotIn("causal_chunk_mask_probs", text, str(path))

        validation_text = Path("configs/validation/validation_scalable_ar_diffusion.yaml").read_text()
        self.assertIn("prediction_tokens: 6", validation_text)
        self.assertIn("causal_noise_schedule: false", validation_text)
        self.assertIn("causal_chunk_mask_multipliers: [1, 1, 1, 1,1,1]", validation_text)
        self.assertIn("causal_loss_weights: [1.0, 1.0, 1.0, 1.0,1.0,1.0]", validation_text)

        server_text = Path("configs/train/train_scalable_ar_diffusion.yaml").read_text()
        self.assertIn("prediction_tokens: 6", server_text)
        self.assertIn("causal_noise_schedule: true", server_text)
        self.assertIn("causal_chunk_mask_multipliers: [0.80, 0.90, 1.00, 1.10, 1.20, 1.30]", server_text)
        self.assertIn("causal_loss_weights: [1.5, 1.25, 1.0, 0.9, 0.8, 0.7]", server_text)


    def test_ar_train_configs_enable_visible_token_neighbor_corruption(self):
        corruption_paths = [
            Path("configs/train/train_scalable_ar_diffusion_baseline_1000.yaml"),
            Path("configs/train/train_scalable_ar_diffusion_local.yaml"),
        ]
        for path in corruption_paths:
            text = path.read_text()
            self.assertIn("visible_token_corruption_prob: 0.15", text, str(path))
            self.assertIn("visible_token_corruption_topk: 5", text, str(path))

        server_text = Path("configs/train/train_scalable_ar_diffusion.yaml").read_text()
        self.assertIn("visible_token_corruption_prob: 0", server_text)
        frontier_text = Path("configs/train/train_scalable_ar_diffusion_frontier_local.yaml").read_text()
        self.assertIn("visible_token_corruption_prob: 0.0", frontier_text)

        validation_text = Path("configs/validation/validation_scalable_ar_diffusion.yaml").read_text()
        self.assertIn("visible_token_corruption_prob: 0.0", validation_text)
        self.assertIn("visible_token_corruption_topk: 5", validation_text)

    def test_ar_rerank_config_keeps_maskgit_and_enables_proposal_training(self):
        text = Path("configs/train/train_scalable_ar_diffusion_rerank_1000.yaml").read_text()

        self.assertIn("predictor: smart_ar_diffusion", text)
        self.assertIn("ar_objective: maskgit", text)
        self.assertIn("causal_temporal_edges: true", text)
        self.assertIn("causal_loss_weighting_enabled: true", text)
        self.assertIn("sampling_guidance:", text)
        self.assertIn("enabled: false", text)
        self.assertIn("mode: safe_speed", text)
        self.assertIn("dense_smart_ce_loss_weight: 1.0", text)
        self.assertIn("diffusion_loss_weight: 0.25", text)
        self.assertIn("proposal_carry_training_enabled: true", text)
        self.assertIn("proposal_carry_loss_weight: 0.5", text)
        self.assertIn("proposal_conditioning_enabled: true", text)
        self.assertIn("commit_min_speed_ratio: 0.75", text)
        self.assertIn("commit_max_speed_ratio: 1.25", text)
        self.assertIn("map_token_noise:", text)
        self.assertIn("history_context_dropout:", text)

    def test_ar_rerank_server_config_uses_full_server_training_paths(self):
        text = Path("configs/train/train_scalable_ar_diffusion_rerank.yaml").read_text()

        self.assertIn('strategy: ddp_find_unused_parameters_true', text)
        self.assertIn("devices: 14", text)
        self.assertIn("/raid/haoq_lab/wangshijie/data/waymo/training", text)
        self.assertIn("predictor: smart_ar_diffusion", text)
        self.assertIn("ar_objective: maskgit", text)
        self.assertIn("causal_temporal_edges: true", text)
        self.assertIn("prediction_tokens: 4", text)
        self.assertIn("commit_tokens: 1", text)
        self.assertIn("causal_noise_schedule: false", text)
        self.assertIn("causal_loss_weighting_enabled: true", text)
        self.assertIn("causal_loss_weights: [2.0, 1.0, 0.5, 0.25]", text)
        self.assertIn("dense_smart_ce_loss_weight: 1.0", text)
        self.assertIn("diffusion_loss_weight: 0.25", text)
        self.assertIn("proposal_carry_training_enabled: true", text)
        self.assertIn("proposal_conditioning_enabled: true", text)
        self.assertIn("sampling_guidance:", text)
        self.assertIn("enabled: false", text)
        self.assertIn("mode: safe_speed", text)
        self.assertIn("map_token_noise:", text)
        self.assertIn("history_context_dropout:", text)

    def test_ar_rerank_validation_config_keeps_guidance_without_training_dropout(self):
        text = Path("configs/validation/validation_scalable_ar_diffusion_rerank.yaml").read_text()

        self.assertIn('mode: "validation"', text)
        self.assertIn("ar_objective: maskgit", text)
        self.assertIn("causal_temporal_edges: true", text)
        self.assertIn("sampling_guidance:", text)
        self.assertIn("mode: safe_speed", text)
        self.assertIn("commit_min_speed_ratio: 0.75", text)
        self.assertIn("commit_max_speed_ratio: 1.25", text)
        self.assertIn("proposal_conditioning_enabled: true", text)
        self.assertIn("history_context_dropout:", text)
        self.assertIn("enabled: false", text)


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

    def test_causal_temporal_decoder_flag_is_config_driven(self):
        model = _ar_shell()
        model.ar_causal_temporal_edges = False

        self.assertFalse(model._use_causal_temporal_decoder())

        model.ar_causal_temporal_edges = True

        self.assertTrue(model._use_causal_temporal_decoder())

    def test_frontier_loss_masks_suffix_and_supervises_selected_frontier(self):
        model = _ar_shell()
        model.ar_objective = "causal_frontier_v1"
        model.ar_frontier_loss_weight = 1.0
        model.ar_continuous_recovery_loss_weight = 0.0
        model.diffusion_decoder = SimpleNamespace(mask_token_id=9)
        model.min_t = 1.0e-3
        captured = {}
        test_case = self

        def fake_frontier_ids(self, loss_mask_base, chunk_ids):
            test_case.assertTrue(loss_mask_base.all())
            test_case.assertTrue(torch.equal(chunk_ids, torch.tensor([[0, 1, 2, 3]])))
            return torch.tensor([1])

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
            captured["noisy"] = noisy.clone()
            captured["geometry_known_mask"] = geometry_known_mask.clone()
            captured["t"] = t.clone()
            logits = torch.zeros(1, 4, 10)
            logits[0, 1, 1] = 4.0
            logits[0, 2, 2] = 4.0
            logits[0, 3, 3] = 4.0
            return logits

        model._sample_frontier_ids = MethodType(fake_frontier_ids, model)
        model._decode_diffusion_logits = MethodType(fake_decode, model)
        packed = {
            "token_ids": torch.tensor([[0, 1, 2, 3]]),
            "valid_mask": torch.ones(1, 4, dtype=torch.bool),
            "loss_mask_base": torch.ones(1, 4, dtype=torch.bool),
            "chunk_ids": torch.tensor([[0, 1, 2, 3]]),
        }

        loss, acc = model._compute_diffusion_loss(packed, torch.zeros(1, 8))

        self.assertGreater(float(loss), 0.0)
        self.assertEqual(float(acc), 1.0)
        self.assertTrue(torch.equal(captured["noisy"], torch.tensor([[0, 9, 9, 9]])))
        self.assertTrue(torch.equal(
            captured["geometry_known_mask"],
            torch.tensor([[True, False, False, False]]),
        ))
        self.assertTrue(torch.allclose(captured["t"], torch.tensor([0.75])))

    def test_ar_closed_loop_curriculum_starts_rollout_at_epoch_zero(self):
        model = _ar_shell()
        model.ar_closed_loop_batch_ratio_max = 0.5

        self.assertEqual(model._ar_closed_loop_curriculum(0), (0.0, 0.5))
        self.assertEqual(model._ar_closed_loop_curriculum(3), (0.0, 0.5))
        self.assertEqual(model._ar_closed_loop_curriculum(4), (0.25, 0.5))

    def test_ar_frontier_training_view_uses_rollout_when_selected(self):
        model = _ar_shell()
        model.ar_closed_loop_max_depth = 1
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)
        metadata = {"retokenization_valid": torch.ones(1, 4, dtype=torch.bool)}
        calls = {"rollout": 0, "retokenize": 0}
        test_case = self

        model._ar_closed_loop_curriculum = MethodType(lambda self, epoch: (0.0, 1.0), model)

        def fake_rollout_view(self, batch, rollout_depth):
            calls["rollout"] += 1
            test_case.assertEqual(rollout_depth, 1)
            return batch

        def fake_retokenize(self, view):
            calls["retokenize"] += 1
            return metadata

        model._build_model_rollout_training_view = MethodType(fake_rollout_view, model)
        model._retokenize_training_view = MethodType(fake_retokenize, model)

        view, retokenization, state_mode, rollout_depth = model._build_ar_frontier_training_view(data)

        self.assertIs(view, data)
        self.assertIs(retokenization, metadata)
        self.assertEqual(state_mode, "rollout")
        self.assertEqual(rollout_depth, 1)
        self.assertEqual(calls, {"rollout": 1, "retokenize": 1})

    def test_frontier_training_step_attaches_retokenization_metadata(self):
        model = _ar_shell()
        model.ar_objective = "causal_frontier_v1"
        model.ar_rolling_anchor_training = True
        model.ntp_aux_loss_weight = 0.0
        data = _toy_sequence(num_agents=2, num_tokens=18, num_frames=91)
        metadata = {
            "retokenization_valid": torch.tensor([
                [True, False, True, True],
                [False, True, True, False],
            ]),
            "recovery_target_local_endpoint": torch.arange(16, dtype=torch.float).view(2, 4, 2),
        }
        calls = {"frontier_view": 0}
        logged = []
        test_case = self

        model._prepare_batch = MethodType(lambda self, batch: batch, model)

        def fake_frontier_view(self, batch):
            calls["frontier_view"] += 1
            return batch, metadata, "clean", 0

        def fake_build_inputs(self, batch):
            packed = {
                "agent_maps": [(0, 0, torch.tensor([0, 1]))],
                "token_ids": torch.zeros(1, 8, dtype=torch.long),
                "valid_mask": torch.ones(1, 8, dtype=torch.bool),
                "loss_mask_base": torch.ones(1, 8, dtype=torch.bool),
                "chunk_ids": torch.arange(4).repeat(2).unsqueeze(0),
            }
            summary = torch.zeros(1, 1)
            return packed, summary, None, None, None, None, None

        def fake_diffusion_loss(self, packed, summary):
            expected_valid = torch.tensor([[True, False, True, True, False, True, True, False]])
            expected_endpoint = torch.arange(16, dtype=torch.float).view(1, 8, 2)
            test_case.assertTrue(torch.equal(
                packed["retokenization_valid"],
                expected_valid,
            ))
            test_case.assertTrue(torch.equal(
                packed["recovery_target_local_endpoint"],
                expected_endpoint,
            ))
            return torch.tensor(2.0), torch.tensor(0.5)

        model._build_ar_frontier_training_view = MethodType(fake_frontier_view, model)
        model._build_diffusion_inputs = MethodType(fake_build_inputs, model)
        model._compute_diffusion_loss = MethodType(fake_diffusion_loss, model)
        model._compute_optional_ntp_loss = MethodType(lambda self, batch, ref: torch.tensor(0.0), model)
        model.log = MethodType(lambda self, name, *args, **kwargs: logged.append(name), model)

        loss = model.training_step(data, 0)

        self.assertEqual(float(loss), 2.0)
        self.assertEqual(calls["frontier_view"], 1)
        self.assertIn("train_ar_state_mode_clean", logged)

    def test_maskgit_training_step_combines_dense_and_proposal_losses(self):
        model = _ar_shell()
        model.ar_objective = "maskgit"
        model.ar_rolling_anchor_training = True
        model.diffusion_loss_weight = 0.25
        model.ntp_aux_loss_weight = 0.5
        model.dense_smart_ce_loss_weight = 1.0
        model.proposal_carry_loss_weight = 0.5
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)
        view = data.clone()
        logged = []
        test_case = self

        model._prepare_batch = MethodType(lambda self, batch: batch, model)
        model._build_ar_training_view = MethodType(
            lambda self, batch: (view, None, None, 2),
            model,
        )
        model._build_diffusion_inputs = MethodType(
            lambda self, batch: (
                {
                    "token_ids": torch.zeros(1, 4, dtype=torch.long),
                    "valid_mask": torch.ones(1, 4, dtype=torch.bool),
                    "chunk_ids": torch.arange(4).unsqueeze(0),
                },
                torch.zeros(1, 1),
                None,
                None,
                None,
                None,
                None,
            ),
            model,
        )
        model._compute_diffusion_loss = MethodType(
            lambda self, packed, summary: (torch.tensor(10.0), torch.tensor(0.1)),
            model,
        )

        def fake_dense(self, batch, ref):
            test_case.assertIs(batch, data)
            return ref.new_tensor(3.0)

        def fake_proposal(self, batch, ref):
            test_case.assertIs(batch, data)
            return ref.new_tensor(5.0), ref.new_tensor(0.2), True

        def fake_ntp(self, batch, ref):
            test_case.assertIs(batch, view)
            return ref.new_tensor(7.0)

        model._compute_dense_smart_ce_loss = MethodType(fake_dense, model)
        model._compute_proposal_carry_training_loss = MethodType(fake_proposal, model)
        model._compute_optional_ntp_loss = MethodType(fake_ntp, model)
        model.log = MethodType(lambda self, name, *args, **kwargs: logged.append(name), model)

        loss = model.training_step(data, 0)

        self.assertAlmostEqual(float(loss), 11.5, places=6)
        self.assertIn("dense_smart_ce_loss", logged)
        self.assertIn("proposal_carry_loss", logged)
        self.assertIn("train_proposal_carry_active", logged)

    def test_select_closed_loop_anchor_reserves_rollout_and_prediction_room(self):
        model = _ar_shell()
        model.ar_prediction_tokens = 4
        model.ar_token_steps = 5
        model.ar_history_tokens = 2
        data = _toy_sequence(num_agents=1, num_tokens=8, num_frames=41)

        anchor = model._select_closed_loop_anchor(data, rollout_depth=2)

        self.assertEqual(anchor, 2)

    def test_retokenize_training_view_updates_future_tokens_and_metadata(self):
        model = _ar_shell()
        model.ar_prediction_tokens = 4
        model.ar_token_steps = 5
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)
        token_ids = torch.tensor([[5, 6, 7, 8]])
        errors = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        retokenization_valid = torch.tensor([[True, False, True, True]])
        local_endpoints = torch.ones(1, 4, 2)

        def fake_retokenize_future(self, **_kwargs):
            return token_ids, errors, retokenization_valid, local_endpoints

        def fake_decode(self, token_ids_arg, token_valid_arg, *_args):
            token_pos = torch.arange(8, dtype=torch.float).view(1, 4, 2)
            token_heading = torch.arange(4, dtype=torch.float).view(1, 4)
            return None, None, None, token_pos, token_heading, None, None

        model._retokenize_future = MethodType(fake_retokenize_future, model)
        model._decode_token_sequence = MethodType(fake_decode, model)

        metadata = model._retokenize_training_view(data)

        self.assertTrue(torch.equal(data["agent"]["token_idx"][0, 2:6], token_ids[0]))
        self.assertTrue(torch.equal(data["agent"]["token_pos"][0, 2:6], torch.arange(8, dtype=torch.float).view(4, 2)))
        self.assertTrue(torch.equal(data["agent"]["token_heading"][0, 2:6], torch.arange(4, dtype=torch.float)))
        self.assertTrue(torch.equal(metadata["retokenization_error"], errors))
        self.assertTrue(torch.equal(metadata["retokenization_valid"], retokenization_valid))
        self.assertTrue(torch.equal(metadata["recovery_target_local_endpoint"], local_endpoints))

    def test_current_state_context_is_added_to_each_agent_chunk(self):
        model = _ar_shell()
        model.ar_prediction_tokens = 2
        model.current_state_enabled = True
        object.__setattr__(model, "current_state_projection", lambda features: features)
        data = _toy_sequence(num_agents=1, num_tokens=18, num_frames=91)
        packed = {
            "agent_maps": [(0, 0, torch.tensor([0]))],
            "agent_context": torch.zeros(1, 2, 5),
            "valid_mask": torch.ones(1, 2, dtype=torch.bool),
        }

        model._apply_current_state_context(data, packed)

        _velocity, _heading, expected_features = model._current_state_motion(data)
        self.assertTrue(torch.allclose(packed["agent_context"][0, 0], expected_features[0]))
        self.assertTrue(torch.allclose(packed["agent_context"][0, 1], expected_features[0]))


if __name__ == "__main__":
    unittest.main()
