import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from smart.model.smart_causal_diffusion import SMARTCausalDiffusion
from smart.modules.causal_diffusion_decoder import CausalDiffusionDecoder
from smart.modules.trajectory_energy import TrajectoryEnergy
from scripts.calibrate_causal_retokenization import compute_type_thresholds
from smart.utils.config import load_config_act


def _causal_shell():
    model = object.__new__(SMARTCausalDiffusion)
    model.num_future_chunks = 4
    return model


class CausalDiffusionDecoderTest(unittest.TestCase):
    def test_temporal_edges_only_flow_from_earlier_to_later_chunks(self):
        decoder = CausalDiffusionDecoder(
            hidden_dim=16,
            token_size=32,
            num_future_chunks=4,
            num_heads=2,
            head_dim=8,
            dropout=0.0,
            num_freq_bands=4,
            a2a_radius=20.0,
            pl2a_radius=20.0,
            time_span=None,
            future_chunk_steps=5,
            num_layers=1,
            num_token_types=4,
        )
        chunk_ids = torch.tensor([[0, 1, 2, 3]])
        edge_index, _ = decoder._build_temporal_token_edges(
            positions=torch.tensor([[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]]),
            headings=torch.zeros(1, 4),
            chunk_ids=chunk_ids,
            agent_ids=torch.zeros(1, 4, dtype=torch.long),
            valid_mask=torch.ones(1, 4, dtype=torch.bool),
        )

        source_chunks = chunk_ids.reshape(-1)[edge_index[0]]
        target_chunks = chunk_ids.reshape(-1)[edge_index[1]]
        self.assertGreater(edge_index.shape[1], 0)
        self.assertTrue(torch.all(source_chunks < target_chunks))

    def test_proposal_embedding_changes_masked_token_logits(self):
        torch.manual_seed(0)
        decoder = CausalDiffusionDecoder(
            hidden_dim=16,
            token_size=32,
            num_future_chunks=4,
            num_heads=2,
            head_dim=8,
            dropout=0.0,
            num_freq_bands=4,
            a2a_radius=20.0,
            pl2a_radius=20.0,
            time_span=None,
            future_chunk_steps=5,
            num_layers=1,
            num_token_types=4,
        )
        common = {
            'noisy_token_ids': torch.tensor([[32]]),
            'token_positions': torch.zeros(1, 1, 2),
            'token_headings': torch.zeros(1, 1),
            'token_agent_ids': torch.zeros(1, 1, dtype=torch.long),
            'noisy_token_chunk_ids': torch.zeros(1, 1, dtype=torch.long),
            'scene_summary': torch.zeros(1, 16),
            't': torch.ones(1),
            'valid_mask': torch.ones(1, 1, dtype=torch.bool),
            'agent_context': torch.zeros(1, 1, 16),
            'agent_type_ids': torch.zeros(1, 1, dtype=torch.long),
            'agent_shape_embeddings': torch.zeros(1, 1, 16),
            'physical_token_embeddings': torch.zeros(1, 1, 16),
        }

        without_proposal = decoder(**common)
        with_proposal = decoder(
            **common,
            proposal_token_embeddings=torch.ones(1, 1, 16),
            proposal_confidence=torch.ones(1, 1),
        )

        self.assertFalse(torch.allclose(without_proposal, with_proposal))


class CausalFrontierPlannerTest(unittest.TestCase):
    def test_reveal_schedule_releases_one_chunk_per_sampling_step(self):
        model = _causal_shell()

        counts = [
            model._causal_reveal_count(step, num_steps=4, num_chunks=4)
            for step in range(4)
        ]

        self.assertEqual(counts, [1, 2, 3, 4])

    def test_reveal_schedule_rejects_idle_sampling_steps(self):
        model = _causal_shell()

        with self.assertRaisesRegex(ValueError, "must equal"):
            model._causal_reveal_count(0, num_steps=16, num_chunks=4)

    def test_sampling_reveals_frontiers_without_remasking(self):
        model = _causal_shell()
        model.diffusion_num_steps = 4
        model.min_t = 1e-3
        model.remask_confidence_temperature = 1.0
        model.safety_energy_enabled = False
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)
        decode_times = []

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
            decode_times.append(float(t[0]))
            logits = torch.full((*noisy.shape, 8), -20.0, device=noisy.device)
            target_ids = packed['chunk_ids'] + 1
            logits.scatter_(-1, target_ids.unsqueeze(-1), 20.0)
            return logits

        model._decode_diffusion_logits = MethodType(fake_decode, model)
        valid_mask = torch.ones(1, 4, dtype=torch.bool)
        chunk_ids = torch.tensor([[0, 1, 2, 3]])
        packed = {
            'chunk_ids': chunk_ids,
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        sampled, _confidence, trace = model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 4, 2),
            token_headings=torch.zeros(1, 4),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=chunk_ids,
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 4, 4),
            agent_type_ids=torch.zeros(1, 4, dtype=torch.long),
            packed=packed,
            return_trace=True,
        )

        masked_counts = [entry['masked_after'] for entry in trace]
        self.assertTrue(torch.equal(sampled, torch.tensor([[1, 2, 3, 4]])))
        self.assertTrue(all(left >= right for left, right in zip(masked_counts, masked_counts[1:])))
        self.assertEqual(masked_counts[-1], 0)
        self.assertTrue(all(entry['remasked'] == 0 for entry in trace))
        self.assertEqual(decode_times, [1.0, 0.75, 0.5, 0.25])

    def test_sampling_uses_carried_tail_as_revisable_conditioning(self):
        model = _causal_shell()
        model.diffusion_num_steps = 4
        model.min_t = 1e-3
        model.remask_confidence_temperature = 1.0
        model.safety_energy_enabled = False
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)
        proposal_ids = torch.tensor([[4, 5, 6, 0]])
        proposal_confidence = torch.tensor([[0.9, 0.8, 0.7, 0.0]])
        decode_proposals = []

        def fake_decode(
            self,
            noisy,
            packed,
            summary,
            t,
            geometry_known_mask,
            proposal_token_ids=None,
            proposal_confidence=None,
        ):
            decode_proposals.append((
                proposal_token_ids.clone(),
                proposal_confidence.clone(),
                noisy.clone(),
            ))
            logits = torch.full((*noisy.shape, 8), -20.0)
            logits[..., 1] = 20.0
            return logits

        model._decode_diffusion_logits = MethodType(fake_decode, model)
        valid_mask = torch.ones(1, 4, dtype=torch.bool)
        packed = {
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        sampled, _confidence = model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 4, 2),
            token_headings=torch.zeros(1, 4),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 4, 4),
            agent_type_ids=torch.zeros(1, 4, dtype=torch.long),
            packed=packed,
            initial_proposal_token_ids=proposal_ids,
            initial_proposal_confidence=proposal_confidence,
        )

        self.assertEqual(len(decode_proposals), 4)
        for seen_ids, seen_confidence, _noisy in decode_proposals:
            self.assertTrue(torch.equal(seen_ids, proposal_ids))
            self.assertTrue(torch.equal(seen_confidence, proposal_confidence))
        self.assertTrue(torch.equal(sampled, torch.ones_like(sampled)))
        self.assertEqual(int(decode_proposals[0][2][0, 0]), model.mask_token_id)

    def test_discrete_frontier_loss_supervises_only_selected_frontier(self):
        model = _causal_shell()
        model.training = False
        model.causal_frontier_loss_weight = 1.0
        model.continuous_recovery_loss_weight = 0.0
        model.diffusion_decoder = SimpleNamespace(mask_token_id=2)
        model._sample_frontier_ids = MethodType(
            lambda self, loss_mask_base, chunk_ids: torch.tensor(
                [2],
                device=chunk_ids.device,
            ),
            model,
        )
        seen_noisy = []

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
            seen_noisy.append(noisy.clone())
            return torch.zeros(*noisy.shape, 2)

        model._decode_diffusion_logits = MethodType(
            fake_decode,
            model,
        )
        packed = {
            'token_ids': torch.zeros(1, 4, dtype=torch.long),
            'valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'loss_mask_base': torch.ones(1, 4, dtype=torch.bool),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
        }

        loss, _acc = model._compute_diffusion_loss(packed, torch.zeros(1, 4))

        self.assertAlmostEqual(float(loss), float(torch.log(torch.tensor(2.0))), places=5)
        self.assertTrue(torch.equal(
            seen_noisy[0],
            torch.tensor([[0, 0, 2, 2]]),
        ))

    def test_all_mask_chunk_zero_remains_an_interaction_source(self):
        model = _causal_shell()
        model.geometry_confidence_source_threshold = 0.2
        model.use_proposal_geometry = True
        model.proposal_conditioning_enabled = True
        model.causal_current_state_edges = True
        captured = {}

        model._refresh_token_geometry = MethodType(
            lambda self, token_ids, packed, **_kwargs: (
                packed['token_positions'],
                packed['token_headings'],
                torch.zeros_like(token_ids, dtype=torch.float),
            ),
            model,
        )
        model._physical_token_embeddings = MethodType(
            lambda self, token_ids, agent_type_ids: torch.zeros(
                *token_ids.shape,
                4,
            ),
            model,
        )

        def fake_decoder(**kwargs):
            captured.update(kwargs)
            return torch.zeros(1, 4, 8)

        model.diffusion_decoder = fake_decoder
        packed = {
            'valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_positions': torch.zeros(1, 4, 2),
            'token_headings': torch.zeros(1, 4),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
            'agent_context': torch.zeros(1, 4, 4),
            'agent_type_ids': torch.zeros(1, 4, dtype=torch.long),
            'agent_shape_embeddings': torch.zeros(1, 4, 4),
        }

        model._decode_diffusion_logits(
            noisy=torch.full((1, 4), 8, dtype=torch.long),
            packed=packed,
            summary=torch.zeros(1, 4),
            t=torch.ones(1),
            geometry_known_mask=torch.zeros(1, 4, dtype=torch.bool),
        )

        expected = torch.tensor([[True, False, False, False]])
        self.assertTrue(torch.equal(captured['spatial_source_mask'], expected))
        self.assertTrue(torch.equal(captured['temporal_source_mask'], expected))

    def test_commit_energy_guidance_is_active_at_the_first_sampling_step(self):
        model = _causal_shell()
        model.safety_energy_weight = 2.0
        model.commit_safety_weight = 2.0
        log_probabilities = torch.log(torch.tensor([[0.6, 0.4]]))
        energies = torch.tensor([[3.0, 0.0]])

        commit = model._select_topk_by_energy(
            log_probabilities,
            energies,
            t_value=1.0,
            frontier_chunk_ids=torch.tensor([0]),
        )
        future = model._select_topk_by_energy(
            log_probabilities,
            energies,
            t_value=1.0,
            frontier_chunk_ids=torch.tensor([1]),
        )

        self.assertEqual(int(commit[0]), 1)
        self.assertEqual(int(future[0]), 0)

    def test_recency_pooling_preserves_latest_motion_state(self):
        model = _causal_shell()
        model.history_recency_decay = 0.5
        history = torch.tensor([[[1.0], [3.0]]])
        valid = torch.tensor([[True, True]])

        pooled = model._pool_agent_context(history, valid)

        self.assertAlmostEqual(float(pooled[0, 0]), 7.0 / 3.0, places=5)

    def test_dynamics_energy_penalizes_observed_to_candidate_velocity_jump(self):
        energy = TrajectoryEnergy(
            dt=0.1,
            max_acceleration=6.0,
            max_yaw_rate=1.2,
        )
        candidate_positions = torch.tensor([[
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        ]])
        candidate_headings = torch.zeros(1, 1, 3)

        without_observation = energy.dynamics_energy(
            candidate_positions,
            candidate_headings,
        )
        with_observation = energy.dynamics_energy(
            candidate_positions,
            candidate_headings,
            current_positions=torch.zeros(1, 2),
            current_velocities=torch.zeros(1, 2),
            current_headings=torch.zeros(1),
        )

        self.assertAlmostEqual(float(without_observation), 0.0, places=5)
        self.assertGreater(float(with_observation), 1000.0)

    def test_commit_speed_energy_penalizes_static_token_for_moving_agent(self):
        model = _causal_shell()
        model.ar_token_steps = 5
        model.commit_min_speed_ratio = 0.5
        model.commit_speed_threshold = 1.0
        object.__setattr__(model, 'trajectory_energy', TrajectoryEnergy(dt=0.1))

        anchor_positions = torch.zeros(2, 2)
        current_velocities = torch.tensor([
            [10.0, 0.0],
            [10.0, 0.0],
        ])
        candidate_positions = torch.tensor([
            [
                [[0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]],
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            ],
            [
                [[0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]],
                [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            ],
        ])

        energy = model._commit_speed_energy(
            candidate_positions,
            anchor_positions,
            current_velocities,
            frontier_chunk_ids=torch.tensor([0, 1]),
        )

        self.assertGreater(float(energy[0, 0]), float(energy[0, 1]) + 1.0)
        self.assertAlmostEqual(float(energy[0, 1]), 0.0, places=5)
        self.assertTrue(torch.equal(energy[1], torch.zeros(2)))

    def test_commit_speed_energy_uses_within_token_speed_not_only_endpoint(self):
        model = _causal_shell()
        model.ar_token_steps = 5
        model.commit_min_speed_ratio = 0.5
        model.commit_speed_threshold = 1.0
        object.__setattr__(model, 'trajectory_energy', TrajectoryEnergy(dt=0.1))

        anchor_positions = torch.zeros(1, 2)
        current_velocities = torch.tensor([[10.0, 0.0]])
        candidate_positions = torch.tensor([[
            [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 0.0]],
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        ]])

        energy = model._commit_speed_energy(
            candidate_positions,
            anchor_positions,
            current_velocities,
            frontier_chunk_ids=torch.tensor([0]),
        )

        self.assertGreater(float(energy[0, 0]), float(energy[0, 1]) + 1.0)
        self.assertAlmostEqual(float(energy[0, 1]), 0.0, places=5)

    def test_commit_speed_energy_uses_reference_speed_after_history_slows(self):
        model = _causal_shell()
        model.ar_token_steps = 5
        model.commit_min_speed_ratio = 0.5
        model.commit_speed_threshold = 1.0
        object.__setattr__(model, 'trajectory_energy', TrajectoryEnergy(dt=0.1))

        candidate_positions = torch.tensor([[
            [[0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0], [0.1, 0.0]],
            [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
        ]])

        energy = model._commit_speed_energy(
            candidate_positions,
            anchor_positions=torch.zeros(1, 2),
            current_velocities=torch.tensor([[0.2, 0.0]]),
            frontier_chunk_ids=torch.tensor([0]),
            reference_speeds=torch.tensor([10.0]),
        )

        self.assertGreater(float(energy[0, 0]), float(energy[0, 1]) + 1.0)
        self.assertAlmostEqual(float(energy[0, 1]), 0.0, places=5)

    def test_safe_guidance_prefers_moving_commit_over_static_high_prob_token(self):
        model = _causal_shell()
        model.ar_token_steps = 5
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)
        model.guidance_mode = 'safe'
        model.guidance_safe_topk = 2
        model.safety_topk = 2
        model.safety_energy_weight = 1.0
        model.commit_safety_weight = 1.0
        model.lane_distance_energy_weight = 0.0
        model.lane_heading_energy_weight = 0.0
        model.dynamics_energy_weight = 0.0
        model.commit_speed_energy_weight = 1.0
        model.collision_energy_weight = 0.0
        model.commit_min_speed_ratio = 0.5
        model.commit_speed_threshold = 1.0
        object.__setattr__(model, 'trajectory_energy', TrajectoryEnergy(dt=0.1))

        def fake_refresh(self, token_ids, packed, **_kwargs):
            return (
                packed['token_positions'],
                packed['token_headings'],
                torch.ones_like(token_ids, dtype=torch.float),
            )

        def fake_token_world(self, token_ids, agent_types, positions, headings):
            del agent_types, headings
            steps = torch.arange(
                1,
                6,
                device=positions.device,
                dtype=positions.dtype,
            )
            world = positions[:, None, :].expand(-1, 5, -1).clone()
            moving = token_ids == 3
            world[moving, :, 0] += steps
            world[~moving, :, 0] += 0.1
            return world, torch.zeros(token_ids.shape[0], 5, device=positions.device)

        model._refresh_token_geometry = MethodType(fake_refresh, model)
        model._token_chunk_world = MethodType(fake_token_world, model)
        logits = torch.full((1, 1, 8), -20.0)
        logits[0, 0, 2] = 4.0
        logits[0, 0, 3] = 3.0
        probabilities = torch.softmax(logits, dim=-1)
        packed = {
            'valid_mask': torch.ones(1, 1, dtype=torch.bool),
            'token_positions': torch.zeros(1, 1, 2),
            'token_headings': torch.zeros(1, 1),
            'token_agent_ids': torch.zeros(1, 1, dtype=torch.long),
            'agent_type_ids': torch.zeros(1, 1, dtype=torch.long),
            'chunk_ids': torch.zeros(1, 1, dtype=torch.long),
            'current_velocities': torch.tensor([[[10.0, 0.0]]]),
            'current_headings': torch.zeros(1, 1),
        }

        selected_ids, _confidence, diagnostics = model._guided_frontier_tokens(
            logits,
            probabilities,
            frontier=torch.ones(1, 1, dtype=torch.bool),
            packed=packed,
            sampled=torch.full((1, 1), model.mask_token_id, dtype=torch.long),
            t_value=1.0,
        )

        self.assertEqual(int(selected_ids[0]), 3)
        self.assertIn('commit_speed_energy', diagnostics)

    def test_guidance_mode_none_uses_plain_sampling_without_safe_rerank(self):
        model = _causal_shell()
        model.diffusion_num_steps = 4
        model.min_t = 1e-3
        model.remask_confidence_temperature = 1.0
        model.safety_energy_enabled = True
        model.guidance_mode = 'none'
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
            logits = torch.full((*noisy.shape, 8), -20.0, device=noisy.device)
            logits[..., 3] = 20.0
            return logits

        def forbidden_guidance(self, *_args, **_kwargs):
            raise AssertionError("none guidance must not call guided rerank")

        model._decode_diffusion_logits = MethodType(fake_decode, model)
        model._guided_frontier_tokens = MethodType(forbidden_guidance, model)
        valid_mask = torch.ones(1, 4, dtype=torch.bool)
        packed = {
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        sampled, _confidence = model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 4, 2),
            token_headings=torch.zeros(1, 4),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 4, 4),
            agent_type_ids=torch.zeros(1, 4, dtype=torch.long),
            packed=packed,
        )

        self.assertTrue(torch.equal(sampled, torch.full((1, 4), 3)))

    def test_ego_stress_guidance_prefers_valid_ego_interaction_candidate(self):
        model = _causal_shell()
        model.guidance_mode = 'ego_stress'
        model.guidance_ego_interaction_alpha = 2.0
        model.guidance_target_event_eta = 1.0
        model.guidance_invalid_beta = 10.0
        model.guidance_edit_gamma = 0.0
        model.safety_energy_weight = 0.0
        log_probabilities = torch.log(torch.tensor([[0.70, 0.20, 0.10]]))
        guidance = {
            'safe_energy': torch.tensor([[0.0, 0.0, 0.0]]),
            'invalid_energy': torch.tensor([[0.0, 3.0, 0.0]]),
            'critical_reward': torch.tensor([[0.5, 10.0, 0.0]]),
            'ego_interaction_reward': torch.tensor([[0.5, 10.0, 0.0]]),
            'target_event_reward': torch.tensor([[0.0, 0.0, 0.0]]),
            'edit_distance': torch.zeros(1, 3),
        }

        selected = model._select_topk_by_guidance(
            log_probabilities,
            guidance,
            t_value=1.0,
            frontier_chunk_ids=torch.tensor([0]),
        )

        self.assertEqual(int(selected[0]), 0)

    def test_ego_edit_guidance_penalizes_distance_from_seed(self):
        model = _causal_shell()
        model.guidance_mode = 'ego_edit'
        model.guidance_ego_interaction_alpha = 0.0
        model.guidance_target_event_eta = 0.0
        model.guidance_invalid_beta = 0.0
        model.guidance_edit_gamma = 2.0
        log_probabilities = torch.log(torch.tensor([[0.5, 0.5]]))
        guidance = {
            'safe_energy': torch.zeros(1, 2),
            'invalid_energy': torch.zeros(1, 2),
            'ego_interaction_reward': torch.zeros(1, 2),
            'target_event_reward': torch.zeros(1, 2),
            'edit_distance': torch.tensor([[0.0, 1.0]]),
        }

        selected = model._select_topk_by_guidance(
            log_probabilities,
            guidance,
            t_value=1.0,
            frontier_chunk_ids=torch.tensor([0]),
        )

        self.assertEqual(int(selected[0]), 0)

    def test_ego_stress_guidance_uses_ego_terms_not_global_criticality(self):
        model = _causal_shell()
        model.guidance_mode = 'ego_stress'
        model.guidance_ego_interaction_alpha = 2.0
        model.guidance_target_event_eta = 1.0
        model.guidance_invalid_beta = 0.0
        model.guidance_edit_gamma = 0.0
        log_probabilities = torch.log(torch.tensor([[0.5, 0.5]]))
        guidance = {
            'safe_energy': torch.zeros(1, 2),
            'invalid_energy': torch.zeros(1, 2),
            'critical_reward': torch.tensor([[100.0, 0.0]]),
            'ego_interaction_reward': torch.tensor([[0.0, 1.0]]),
            'target_event_reward': torch.tensor([[0.0, 0.0]]),
            'edit_distance': torch.zeros(1, 2),
        }

        selected = model._select_topk_by_guidance(
            log_probabilities,
            guidance,
            t_value=1.0,
            frontier_chunk_ids=torch.tensor([0]),
        )

        self.assertEqual(int(selected[0]), 1)

    def test_ego_stress_guidance_prefers_generic_ego_risk_reward(self):
        model = _causal_shell()
        model.guidance_mode = 'ego_stress'
        model.guidance_ego_interaction_alpha = 2.0
        model.guidance_target_event_eta = 0.0
        model.guidance_invalid_beta = 0.0
        model.guidance_edit_gamma = 0.0
        log_probabilities = torch.log(torch.tensor([[0.55, 0.45]]))
        guidance = {
            'safe_energy': torch.zeros(1, 2),
            'invalid_energy': torch.zeros(1, 2),
            'critical_reward': torch.zeros(1, 2),
            'ego_interaction_reward': torch.zeros(1, 2),
            'ego_risk_reward': torch.tensor([[0.0, 1.0]]),
            'target_event_reward': torch.zeros(1, 2),
            'edit_distance': torch.zeros(1, 2),
        }

        selected = model._select_topk_by_guidance(
            log_probabilities,
            guidance,
            t_value=1.0,
            frontier_chunk_ids=torch.tensor([0]),
        )

        self.assertEqual(int(selected[0]), 1)

    def test_ego_edit_guidance_balances_ego_event_and_minimal_edit(self):
        model = _causal_shell()
        model.guidance_mode = 'ego_edit'
        model.guidance_ego_interaction_alpha = 1.0
        model.guidance_target_event_eta = 2.0
        model.guidance_invalid_beta = 0.0
        model.guidance_edit_gamma = 3.0
        log_probabilities = torch.log(torch.tensor([[0.5, 0.5]]))
        guidance = {
            'safe_energy': torch.zeros(1, 2),
            'invalid_energy': torch.zeros(1, 2),
            'ego_interaction_reward': torch.tensor([[0.2, 1.0]]),
            'target_event_reward': torch.tensor([[0.2, 1.0]]),
            'edit_distance': torch.tensor([[0.0, 2.0]]),
        }

        selected = model._select_topk_by_guidance(
            log_probabilities,
            guidance,
            t_value=1.0,
            frontier_chunk_ids=torch.tensor([0]),
        )

        self.assertEqual(int(selected[0]), 0)

    def test_seed_trajs_define_candidate_edit_distance_when_available(self):
        model = _causal_shell()
        candidate_positions = torch.tensor([[
            [[0.0, 0.0], [1.0, 0.0]],
            [[0.0, 2.0], [1.0, 2.0]],
        ]])
        seed_trajs = torch.tensor([[[[0.0, 0.0], [1.0, 0.0]]]])

        edit_distance = model._candidate_edit_distance(
            candidate_positions,
            topk_ids=torch.tensor([[5, 5]]),
            flat_frontier=torch.tensor([0]),
            seed_token_ids=torch.tensor([[5]]),
            seed_trajs=seed_trajs,
        )

        self.assertAlmostEqual(float(edit_distance[0, 0]), 0.0, places=5)
        self.assertGreater(float(edit_distance[0, 1]), 1.0)

    def test_pack_agent_window_values_preserves_seed_traj_dimensions(self):
        model = _causal_shell()
        model.ar_prediction_tokens = 4
        packed = {
            'valid_mask': torch.ones(1, 8, dtype=torch.bool),
            'agent_maps': [(0, 0, torch.tensor([2, 0]))],
        }
        agent_values = torch.arange(
            3 * model.ar_prediction_tokens * 5 * 2,
            dtype=torch.float,
        ).view(3, model.ar_prediction_tokens, 5, 2)

        packed_values = model._pack_agent_window_values(
            agent_values,
            packed,
            fill_value=0.0,
        )

        self.assertEqual(
            tuple(packed_values.shape),
            (1, 8, 5, 2),
        )
        self.assertTrue(torch.equal(
            packed_values[0, :model.ar_prediction_tokens],
            agent_values[2],
        ))
        self.assertTrue(torch.equal(
            packed_values[0, model.ar_prediction_tokens:],
            agent_values[0],
        ))

    def test_zero_guidance_diagnostics_do_not_alias_between_keys(self):
        model = _causal_shell()
        diagnostics = model._zero_guidance_diagnostics(torch.ones(()))

        diagnostics['lane_distance'] += torch.tensor(3.0)

        self.assertEqual(float(diagnostics['lane_distance']), 3.0)
        self.assertEqual(float(diagnostics['collision']), 0.0)
        self.assertIn('dynamics_energy', diagnostics)
        self.assertEqual(float(diagnostics['dynamics_energy']), 0.0)
        self.assertIn('commit_speed_energy', diagnostics)
        self.assertEqual(float(diagnostics['commit_speed_energy']), 0.0)
        self.assertEqual(float(diagnostics['target_success_rate']), 0.0)
        self.assertIn('ego_min_ttc', diagnostics)
        self.assertIn('ego_path_intrusion_rate', diagnostics)
        self.assertIn('ego_conflict_tta_error', diagnostics)
        self.assertIn('ego_risk_min_ttc', diagnostics)
        self.assertIn('ego_risk_success_rate', diagnostics)
        self.assertIn('target_event_success_rate', diagnostics)

    def test_criticality_reference_excludes_same_agent_nominal(self):
        model = _causal_shell()
        object.__setattr__(
            model,
            'trajectory_energy',
            TrajectoryEnergy(dt=0.1, collision_distance=1.0),
        )
        model.guidance_near_miss_distance = 1.5
        model.guidance_ttc_threshold = 3.0
        candidate_positions = torch.tensor([
            [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]],
            [[[4.0, 1.2], [3.0, 1.2], [2.0, 1.2]]],
        ])
        nominal_other = candidate_positions[:, 0]
        candidate_batch = torch.tensor([0, 0])
        agent_ids = torch.tensor([7, 9])

        metrics = model._criticality_against_nominal_other_agents(
            candidate_positions,
            nominal_other,
            candidate_batch,
            agent_ids,
        )

        self.assertTrue(bool(metrics['near_miss'][0, 0]))
        self.assertFalse(bool(metrics['hard_collision'][0, 0]))

    def test_edit_sampling_locks_seed_tokens_outside_edit_mask(self):
        model = _causal_shell()
        model.diffusion_num_steps = 4
        model.min_t = 1e-3
        model.remask_confidence_temperature = 1.0
        model.safety_energy_enabled = False
        model.guidance_mode = 'ego_edit'
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)
        decode_inputs = []

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
            decode_inputs.append(noisy.clone())
            logits = torch.full((*noisy.shape, 8), -20.0, device=noisy.device)
            logits[..., 7] = 20.0
            return logits

        model._decode_diffusion_logits = MethodType(fake_decode, model)
        valid_mask = torch.ones(1, 4, dtype=torch.bool)
        packed = {
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
        }
        seed = torch.tensor([[1, 2, 3, 4]])
        edit_mask = torch.tensor([[False, True, False, False]])

        sampled, _confidence = model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 4, 2),
            token_headings=torch.zeros(1, 4),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 4, 4),
            agent_type_ids=torch.zeros(1, 4, dtype=torch.long),
            packed=packed,
            seed_token_ids=seed,
            editable_mask=edit_mask,
        )

        self.assertTrue(torch.equal(sampled, torch.tensor([[1, 7, 3, 4]])))
        self.assertEqual(len(decode_inputs), 1)
        self.assertTrue(torch.equal(decode_inputs[0], torch.tensor([[1, 8, 3, 4]])))

    def test_edit_sampling_uses_frontier_chunk_timestep_when_seed_locked(self):
        model = _causal_shell()
        model.diffusion_num_steps = 4
        model.min_t = 1e-3
        model.remask_confidence_temperature = 1.0
        model.safety_energy_enabled = False
        model.guidance_mode = 'none'
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)
        decode_times = []

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
            decode_times.append(float(t[0]))
            logits = torch.full((*noisy.shape, 8), -20.0, device=noisy.device)
            logits[..., 7] = 20.0
            return logits

        model._decode_diffusion_logits = MethodType(fake_decode, model)
        valid_mask = torch.ones(1, 4, dtype=torch.bool)
        packed = {
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        sampled, _confidence = model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 4, 2),
            token_headings=torch.zeros(1, 4),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 4, 4),
            agent_type_ids=torch.zeros(1, 4, dtype=torch.long),
            packed=packed,
            seed_token_ids=torch.tensor([[1, 2, 3, 4]]),
            editable_mask=torch.tensor([[False, False, True, False]]),
        )

        self.assertTrue(torch.equal(sampled, torch.tensor([[1, 2, 7, 4]])))
        self.assertEqual(decode_times, [0.5])

    def test_sampling_preserves_infinite_ego_risk_min_ttc_when_no_finite_ttc(self):
        model = _causal_shell()
        model.diffusion_num_steps = 4
        model.min_t = 1e-3
        model.remask_confidence_temperature = 1.0
        model.guidance_mode = 'safe'
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
            logits = torch.full((*noisy.shape, 8), -20.0, device=noisy.device)
            logits[..., 1] = 20.0
            return logits

        def fake_guided(
            self,
            logits,
            probabilities,
            frontier,
            packed,
            sampled,
            t_value,
            **_kwargs,
        ):
            del logits, probabilities, packed, sampled, t_value
            selected_count = int(frontier.sum().item())
            diagnostics = self._zero_guidance_diagnostics(torch.zeros(()))
            diagnostics['ego_risk_min_ttc'] = torch.full((), float('inf'))
            return (
                torch.ones(selected_count, dtype=torch.long),
                torch.ones(selected_count),
                diagnostics,
            )

        model._decode_diffusion_logits = MethodType(fake_decode, model)
        model._guided_frontier_tokens = MethodType(fake_guided, model)
        valid_mask = torch.ones(1, 4, dtype=torch.bool)
        packed = {
            'valid_mask': valid_mask,
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 4, 2),
            token_headings=torch.zeros(1, 4),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 4, 4),
            agent_type_ids=torch.zeros(1, 4, dtype=torch.long),
            packed=packed,
        )

        self.assertTrue(torch.isinf(model._last_sampling_energy['ego_risk_min_ttc']))

    def test_guided_frontier_reports_infinite_ttc_when_no_finite_ttc_exists(self):
        class NoFiniteTtcEnergy:
            def lane_energy(self, candidate_positions, candidate_headings, *args, **kwargs):
                del candidate_headings, args, kwargs
                return (
                    candidate_positions.new_zeros(candidate_positions.shape[:2]),
                    candidate_positions.new_zeros(candidate_positions.shape[:2]),
                )

            def dynamics_energy(self, candidate_positions, candidate_headings, **kwargs):
                del candidate_headings, kwargs
                return candidate_positions.new_zeros(candidate_positions.shape[:2])

            def collision_energy(self, candidate_positions, other_positions, **kwargs):
                del other_positions, kwargs
                return candidate_positions.new_zeros(candidate_positions.shape[:2])

            def criticality_metrics(self, candidate_positions, other_positions, **kwargs):
                del other_positions, kwargs
                shape = candidate_positions.shape[:2]
                return {
                    'critical_reward': candidate_positions.new_zeros(shape),
                    'hard_collision': torch.zeros(shape, dtype=torch.bool),
                }

            def ego_interaction_metrics(self, candidate_positions, ego_positions, **kwargs):
                del ego_positions, kwargs
                shape = candidate_positions.shape[:2]
                inf = torch.full(shape, float('inf'))
                zeros = candidate_positions.new_zeros(shape)
                false = torch.zeros(shape, dtype=torch.bool)
                return {
                    'ego_min_ttc': inf,
                    'ego_min_distance': zeros,
                    'ego_required_decel': zeros,
                    'ego_path_intrusion_rate': zeros,
                    'ego_conflict_tta_error': zeros,
                    'ego_risk_reward': zeros,
                    'ego_interaction_reward': zeros,
                    'target_event_reward': zeros,
                    'ego_risk_success': false,
                    'ego_near_miss_success': false,
                    'target_event_success': false,
                    'hard_collision': false,
                }

        model = _causal_shell()
        model.ar_token_steps = 2
        model.guidance_mode = 'safe'
        model.guidance_safe_topk = 1
        model.safety_topk = 1
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)
        model.safety_energy_weight = 0.0
        model.commit_safety_weight = 0.0
        model.lane_distance_energy_weight = 0.0
        model.lane_heading_energy_weight = 0.0
        model.dynamics_energy_weight = 0.0
        model.commit_speed_energy_weight = 0.0
        model.collision_energy_weight = 0.0
        object.__setattr__(model, 'trajectory_energy', NoFiniteTtcEnergy())

        model._refresh_token_geometry = MethodType(
            lambda self, token_ids, packed, **_kwargs: (
                packed['token_positions'],
                packed['token_headings'],
                torch.ones_like(token_ids, dtype=torch.float),
            ),
            model,
        )
        model._token_chunk_world = MethodType(
            lambda self, token_ids, agent_types, positions, headings: (
                positions[:, None, :].expand(-1, self.ar_token_steps, -1),
                headings[:, None].expand(-1, self.ar_token_steps),
            ),
            model,
        )
        logits = torch.tensor([[[0.0, 1.0]]])
        probabilities = torch.softmax(logits, dim=-1)
        packed = {
            'valid_mask': torch.ones(1, 1, dtype=torch.bool),
            'token_positions': torch.zeros(1, 1, 2),
            'token_headings': torch.zeros(1, 1),
            'token_agent_ids': torch.zeros(1, 1, dtype=torch.long),
            'agent_type_ids': torch.zeros(1, 1, dtype=torch.long),
            'chunk_ids': torch.ones(1, 1, dtype=torch.long),
        }

        _selected_ids, _confidence, diagnostics = model._guided_frontier_tokens(
            logits,
            probabilities,
            frontier=torch.ones(1, 1, dtype=torch.bool),
            packed=packed,
            sampled=torch.full((1, 1), 8, dtype=torch.long),
            t_value=0.75,
        )

        self.assertTrue(torch.isinf(diagnostics['ego_risk_min_ttc']))
        self.assertTrue(torch.isinf(diagnostics['ego_min_ttc']))

    def test_edit_controls_target_agents_and_global_token_window(self):
        model = _causal_shell()
        model.ar_prediction_tokens = 4
        model.ar_history_tokens = 2
        model.guidance_mode = 'ego_edit'
        model.guidance_target_agents = (1,)
        model.guidance_target_time_window = (2, 4)
        data = HeteroData()
        data['agent']['token_idx'] = torch.arange(20).view(2, 10)
        generation_agents = torch.tensor([True, True])
        future_valid = torch.ones(2, 4, dtype=torch.bool)

        seed_tokens, editable = model._build_guidance_edit_controls(
            data,
            round_idx=1,
            future_valid=future_valid,
            generation_agents=generation_agents,
        )

        self.assertTrue(torch.equal(
            seed_tokens,
            torch.tensor([[3, 4, 5, 6], [13, 14, 15, 16]]),
        ))
        self.assertTrue(torch.equal(
            editable,
            torch.tensor([
                [False, False, False, False],
                [False, True, True, False],
            ]),
        ))

    def test_edit_controls_lock_ego_even_when_ego_is_targeted(self):
        model = _causal_shell()
        model.ar_prediction_tokens = 4
        model.ar_history_tokens = 2
        model.num_historical_steps = 3
        model.guidance_mode = 'ego_edit'
        model.guidance_target_agents = (0, 1)
        model.guidance_target_time_window = (0, 4)
        data = HeteroData()
        data['agent']['num_nodes'] = 2
        data['agent']['token_idx'] = torch.arange(12).view(2, 6)
        data['agent']['position'] = torch.zeros(2, 7, 3)
        data['agent']['av_index'] = torch.tensor([0])
        generation_agents = torch.tensor([True, True])
        future_valid = torch.ones(2, 4, dtype=torch.bool)

        _seed_tokens, editable = model._build_guidance_edit_controls(
            data,
            round_idx=0,
            future_valid=future_valid,
            generation_agents=generation_agents,
        )

        self.assertTrue(torch.equal(
            editable,
            torch.tensor([
                [False, False, False, False],
                [True, True, True, True],
            ]),
        ))

    def test_ego_reference_packing_uses_explicit_ego_agent_inputs(self):
        model = _causal_shell()
        model.ar_prediction_tokens = 2
        model.ar_token_steps = 2
        model.num_historical_steps = 3
        data = HeteroData()
        data['agent']['num_nodes'] = 2
        data['agent']['position'] = torch.zeros(2, 7, 3)
        data['agent']['heading'] = torch.zeros(2, 7)
        data['agent']['av_index'] = torch.tensor([0])
        data['agent']['ego_agent_id'] = torch.tensor([1])
        data['agent']['ego_ref_traj'] = torch.stack([
            torch.zeros(4, 2),
            torch.tensor([
                [10.0, 0.0],
                [11.0, 0.0],
                [12.0, 0.0],
                [13.0, 0.0],
            ]),
        ])
        data['agent']['ego_ref_heading'] = torch.stack([
            torch.zeros(4),
            torch.full((4,), 0.5),
        ])
        agent_batch = torch.zeros(2, dtype=torch.long)
        packed = {
            'valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'token_positions': torch.zeros(1, 4, 2),
            'agent_maps': [(0, 0, torch.tensor([0, 1]))],
        }

        model._attach_ego_reference_to_packed(data, packed, agent_batch)

        expected = data['agent']['ego_ref_traj'][1].reshape(2, 2, 2)
        self.assertTrue(torch.equal(
            packed['ego_ref_positions'][0, :2],
            expected,
        ))
        self.assertTrue(torch.equal(
            packed['ego_ref_positions'][0, 2:],
            expected,
        ))
        self.assertTrue(torch.equal(
            packed['ego_ref_headings'][0, :2],
            torch.full((2, 2), 0.5),
        ))

    def test_seed_commit_override_preserves_locked_agent_raw_trajectory(self):
        model = _causal_shell()
        model.ar_commit_tokens = 1
        model.ar_prediction_tokens = 4
        model.ar_token_steps = 2
        model.num_historical_steps = 3
        data = HeteroData()
        data['agent']['position'] = torch.zeros(2, 11, 3)
        data['agent']['heading'] = torch.zeros(2, 11)
        data['agent']['valid_mask'] = torch.ones(2, 11, dtype=torch.bool)
        data['agent']['seed_trajs'] = torch.tensor([
            [[10.0, 0.0], [11.0, 0.0], [12.0, 0.0], [13.0, 0.0]],
            [[20.0, 0.0], [21.0, 0.0], [22.0, 0.0], [23.0, 0.0]],
        ])
        commit_traj = torch.zeros(2, 2, 2)
        commit_head = torch.zeros(2, 2)
        commit_valid = torch.ones(2, 2, dtype=torch.bool)
        token_pos = torch.zeros(2, 1, 2)
        token_heading = torch.zeros(2, 1)
        current_pos = torch.zeros(2, 2)
        current_heading = torch.zeros(2)
        edit_window_mask = torch.tensor([
            [False, False, False, False],
            [True, False, False, False],
        ])

        result = model._apply_guidance_seed_commit_overrides(
            data=data,
            round_idx=0,
            edit_window_mask=edit_window_mask,
            commit_traj=commit_traj,
            commit_head=commit_head,
            commit_valid_frames=commit_valid,
            commit_token_pos=token_pos,
            commit_token_heading=token_heading,
            current_pos=current_pos,
            current_heading=current_heading,
        )

        (
            commit_traj,
            _commit_head,
            _commit_valid,
            token_pos,
            _token_heading,
            current_pos,
            _current_heading,
        ) = result
        self.assertTrue(torch.equal(
            commit_traj[0],
            torch.tensor([[10.0, 0.0], [11.0, 0.0]]),
        ))
        self.assertTrue(torch.equal(commit_traj[1], torch.zeros(2, 2)))
        self.assertTrue(torch.equal(token_pos[0, 0], torch.tensor([11.0, 0.0])))
        self.assertTrue(torch.equal(current_pos[0], torch.tensor([11.0, 0.0])))


class ClosedLoopCurriculumTest(unittest.TestCase):
    def test_curriculum_probabilities_follow_the_four_training_phases(self):
        model = _causal_shell()

        self.assertEqual(model._closed_loop_curriculum(0), (0.0, 0.0))
        self.assertEqual(model._closed_loop_curriculum(4), (0.25, 0.0))
        self.assertAlmostEqual(model._closed_loop_curriculum(8)[1], 0.10)
        self.assertAlmostEqual(model._closed_loop_curriculum(15)[1], 0.30)
        self.assertEqual(model._closed_loop_curriculum(16), (0.25, 0.50))
        self.assertEqual(model._closed_loop_curriculum(31), (0.25, 0.50))

    def test_retokenization_matches_world_future_in_predicted_local_frame(self):
        model = _causal_shell()
        model.ar_token_steps = 5
        model.retokenization_error_thresholds = (0.1, 0.1, 0.1)
        straight = torch.stack(
            [torch.arange(1, 6, dtype=torch.float), torch.zeros(5)],
            dim=-1,
        )
        left = torch.stack(
            [torch.zeros(5), torch.arange(1, 6, dtype=torch.float)],
            dim=-1,
        )
        vocab = {
            'veh': torch.stack([straight, left]),
            'ped': torch.stack([straight, left]),
            'cyc': torch.stack([straight, left]),
        }
        world_future = torch.zeros(1, 1, 5, 2)
        world_future[0, 0, :, 1] = torch.arange(1, 6, dtype=torch.float)

        token_ids, errors, valid, local_endpoints = model._retokenize_future(
            future_positions=world_future,
            future_valid=torch.ones(1, 1, 5, dtype=torch.bool),
            start_positions=torch.zeros(1, 2),
            start_headings=torch.tensor([torch.pi / 2]),
            agent_types=torch.tensor([0]),
            token_center_vocabs=vocab,
        )

        self.assertEqual(int(token_ids[0, 0]), 0)
        self.assertAlmostEqual(float(errors[0, 0]), 0.0, places=5)
        self.assertTrue(bool(valid[0, 0]))
        self.assertTrue(torch.allclose(local_endpoints[0, 0], torch.tensor([5.0, 0.0]), atol=1e-5))

    def test_retokenization_flags_targets_above_type_threshold(self):
        model = _causal_shell()
        model.ar_token_steps = 5
        model.retokenization_error_thresholds = (0.05, 0.05, 0.05)
        straight = torch.stack(
            [torch.arange(1, 6, dtype=torch.float), torch.zeros(5)],
            dim=-1,
        )
        vocab = {
            'veh': straight.unsqueeze(0),
            'ped': straight.unsqueeze(0),
            'cyc': straight.unsqueeze(0),
        }
        world_future = straight.view(1, 1, 5, 2).clone()
        world_future[..., 1] += 0.2

        _token_ids, errors, valid, _local_endpoints = model._retokenize_future(
            future_positions=world_future,
            future_valid=torch.ones(1, 1, 5, dtype=torch.bool),
            start_positions=torch.zeros(1, 2),
            start_headings=torch.zeros(1),
            agent_types=torch.tensor([0]),
            token_center_vocabs=vocab,
        )

        self.assertGreater(float(errors[0, 0]), 0.05)
        self.assertFalse(bool(valid[0, 0]))

    def test_invalid_retokenization_uses_differentiable_endpoint_recovery(self):
        model = _causal_shell()
        straight = torch.stack(
            [torch.arange(1, 6, dtype=torch.float), torch.zeros(5)],
            dim=-1,
        )
        left = torch.stack(
            [torch.zeros(5), torch.arange(1, 6, dtype=torch.float)],
            dim=-1,
        )
        vocab = {
            'veh': torch.stack([straight, left]),
            'ped': torch.stack([straight, left]),
            'cyc': torch.stack([straight, left]),
        }
        model._token_center_vocabs = MethodType(lambda self: vocab, model)
        logits = torch.tensor([[[6.0, -6.0]]], requires_grad=True)
        packed = {
            'agent_type_ids': torch.tensor([[0]]),
            'retokenization_valid': torch.tensor([[False]]),
            'recovery_target_local_endpoint': torch.tensor([[[0.0, 5.0]]]),
            'loss_mask_base': torch.tensor([[True]]),
        }

        recovery_loss = model._continuous_recovery_loss(
            logits,
            packed,
            masked_supervision=torch.tensor([[True]]),
        )
        recovery_loss.backward()

        self.assertGreater(float(recovery_loss), 0.0)
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_training_view_replaces_future_tokens_with_retokenized_targets(self):
        model = _causal_shell()
        model.ar_history_tokens = 2
        model.ar_prediction_tokens = 4
        model.ar_token_steps = 5
        model.num_historical_steps = 11
        view = HeteroData()
        view['agent']['position'] = torch.zeros(1, 31, 3)
        view['agent']['heading'] = torch.zeros(1, 31)
        view['agent']['valid_mask'] = torch.ones(1, 31, dtype=torch.bool)
        view['agent']['token_idx'] = torch.zeros(1, 6, dtype=torch.long)
        view['agent']['agent_valid_mask'] = torch.ones(1, 6, dtype=torch.bool)
        view['agent']['token_pos'] = torch.zeros(1, 6, 2)
        view['agent']['token_heading'] = torch.zeros(1, 6)
        view['agent']['type'] = torch.tensor([0])

        model._retokenize_future = MethodType(
            lambda self, **_kwargs: (
                torch.tensor([[3, 4, 5, 6]]),
                torch.tensor([[0.1, 0.2, 0.3, 0.4]]),
                torch.tensor([[True, False, True, False]]),
                torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]),
            ),
            model,
        )
        model._decode_token_sequence = MethodType(
            lambda self, token_ids, token_valid, agent_types, start_pos, start_heading: (
                torch.zeros(1, 20, 2),
                torch.zeros(1, 20),
                torch.ones(1, 20, dtype=torch.bool),
                torch.ones(1, 4, 2),
                torch.ones(1, 4),
                torch.zeros(1, 2),
                torch.zeros(1),
            ),
            model,
        )

        metadata = model._retokenize_training_view(view)

        self.assertTrue(torch.equal(
            view['agent']['token_idx'][0, 2:],
            torch.tensor([3, 4, 5, 6]),
        ))
        self.assertTrue(torch.equal(
            metadata['retokenization_valid'],
            torch.tensor([[True, False, True, False]]),
        ))
        self.assertTrue(torch.equal(view['agent']['token_pos'][0, 2:], torch.ones(4, 2)))


class CausalDiffusionConfigTest(unittest.TestCase):
    def test_train_and_validation_configs_select_independent_predictor(self):
        root = Path(__file__).resolve().parents[1]
        config_paths = [
            root / 'configs/train/train_scalable_causal_diffusion.yaml',
            root / 'configs/train/train_scalable_causal_diffusion_local.yaml',
            root / 'configs/validation/validation_scalable_causal_diffusion.yaml',
        ]

        for config_path in config_paths:
            config = load_config_act(str(config_path))
            self.assertEqual(config.Model.predictor, 'smart_causal_diffusion')
            self.assertEqual(config.Model.diffusion.prediction_tokens, 4)
            self.assertEqual(config.Model.diffusion.commit_tokens, 1)
            self.assertEqual(
                config.Model.diffusion.num_steps,
                config.Model.diffusion.prediction_tokens,
            )
            self.assertEqual(
                config.Model.diffusion.causal_objective,
                'discrete_frontier_v2',
            )
            self.assertTrue(config.Model.diffusion.carry_tail_proposal)
            self.assertTrue(config.Model.diffusion.proposal_conditioning_enabled)
            self.assertTrue(config.Model.diffusion.current_state_enabled)
            self.assertTrue(config.Model.diffusion.current_state_edges)
            self.assertEqual(
                config.Model.diffusion.closed_loop_batch_ratio_max,
                0.5,
            )
            self.assertEqual(config.Model.diffusion.encoder_lr_scale, 0.5)
            expected_epochs = config.Trainer.max_epochs
            self.assertEqual(config.Model.total_steps, expected_epochs)
            self.assertGreater(config.Model.warmup_steps, 0)
            self.assertLess(config.Model.warmup_steps, expected_epochs)
            self.assertEqual(
                list(config.Model.diffusion.retokenization_error_thresholds),
                [
                    0.7379697561264038,
                    0.8562850952148438,
                    1.2705252170562744,
                ],
            )
            self.assertIn(
                config.Model.diffusion.guidance.mode,
                ('none', 'safe', 'ego_stress', 'ego_edit'),
            )
            self.assertEqual(config.Model.diffusion.guidance.safe_topk, 16)
            self.assertGreaterEqual(config.Model.diffusion.guidance.ego_stress_topk, 64)
            self.assertGreaterEqual(config.Model.diffusion.guidance.ego_edit_topk, 64)
            self.assertEqual(
                config.Model.diffusion.guidance.target_spec,
                'ego_risk',
            )

    def test_horizon_metrics_slice_requested_rollout_prefix(self):
        model = _causal_shell()
        prediction = torch.ones(1, 80, 2)
        target = torch.zeros(1, 80, 2)
        valid = torch.ones(1, 80, dtype=torch.bool)

        ade, fde = model._horizon_displacement_metrics(
            prediction,
            target,
            valid,
            horizon_steps=20,
        )

        expected = 2.0 ** 0.5
        self.assertAlmostEqual(float(ade), expected, places=5)
        self.assertAlmostEqual(float(fde), expected, places=5)

    def test_rollout_score_is_led_by_safety_and_late_horizon_quality(self):
        model = _causal_shell()
        model.rollout_score_weights = {
            'ade_8s': 1.0,
            'late_ade': 1.0,
            'lane_distance': 2.0,
            'lane_heading': 0.5,
            'dynamics': 0.25,
            'collision': 4.0,
        }

        score = model._rollout_score(
            ade_8s=torch.tensor(1.0),
            late_ade=torch.tensor(2.0),
            energies={
                'lane_distance': torch.tensor(0.5),
                'lane_heading': torch.tensor(0.2),
                'dynamics': torch.tensor(0.4),
                'collision': torch.tensor(0.25),
            },
        )

        self.assertAlmostEqual(float(score), 5.2, places=5)

    def test_retokenization_calibration_reports_per_type_quantiles(self):
        thresholds = compute_type_thresholds(
            {
                'veh': [0.1, 0.2, 0.3],
                'ped': [0.4, 0.5],
                'cyc': [0.6],
            },
            quantile=1.0,
        )

        self.assertEqual(
            thresholds,
            {'veh': 0.3, 'ped': 0.5, 'cyc': 0.6},
        )


if __name__ == '__main__':
    unittest.main()
