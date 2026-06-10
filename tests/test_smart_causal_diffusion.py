import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from smart.model.smart_causal_diffusion import SMARTCausalDiffusion
from smart.modules.causal_diffusion_decoder import CausalDiffusionDecoder
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


class CausalAbsorbingProcessTest(unittest.TestCase):
    def test_training_mask_is_a_single_absorbing_suffix_per_agent(self):
        model = _causal_shell()
        valid_mask = torch.tensor([[True, True, True, True, True, True, False]])
        token_agent_ids = torch.tensor([[10, 10, 10, 10, 11, 11, -1]])
        chunk_ids = torch.tensor([[0, 1, 2, 3, 0, 1, 0]])
        random_values = torch.tensor([[0.1, 0.2, 0.9, 0.1, 0.8, 0.1, 0.0]])

        mask = model._sample_absorbing_prefix_mask(
            valid_mask=valid_mask,
            survival_prob=torch.tensor([0.5]),
            token_agent_ids=token_agent_ids,
            chunk_ids=chunk_ids,
            random_values=random_values,
        )

        self.assertTrue(torch.equal(
            mask,
            torch.tensor([[False, False, True, True, True, True, False]]),
        ))

    def test_reveal_schedule_is_monotonic_and_finishes_all_chunks(self):
        model = _causal_shell()

        counts = [
            model._causal_reveal_count(step, num_steps=16, num_chunks=4)
            for step in range(16)
        ]

        self.assertEqual(counts[-1], 4)
        self.assertTrue(all(left <= right for left, right in zip(counts, counts[1:])))
        self.assertEqual(counts[11], 0)
        self.assertEqual(counts[12], 1)
        self.assertEqual(counts[14], 3)

    def test_sampling_reveals_frontiers_without_remasking(self):
        model = _causal_shell()
        model.diffusion_num_steps = 8
        model.min_t = 1e-3
        model.remask_confidence_temperature = 1.0
        model.diffusion_decoder = SimpleNamespace(mask_token_id=8)

        def fake_decode(self, noisy, packed, summary, t, geometry_known_mask, **_kwargs):
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

    def test_frontier_weight_is_added_to_suffix_nll(self):
        model = _causal_shell()
        model.training = False
        model.causal_frontier_loss_weight = 2.0
        model.diffusion_decoder = SimpleNamespace(mask_token_id=2)
        model._sample_diffusion_timesteps = MethodType(
            lambda self, batch_size, device: torch.full((batch_size,), 0.5, device=device),
            model,
        )
        model.noise_schedule = lambda t: (
            torch.full_like(t, torch.log(torch.tensor(2.0))),
            torch.full_like(t, 0.5),
            torch.ones_like(t),
        )
        model._sample_absorbing_prefix_mask = MethodType(
            lambda self, **_kwargs: torch.tensor([[False, True, True, True]]),
            model,
        )
        model._decode_diffusion_logits = MethodType(
            lambda self, noisy, packed, summary, t, geometry_known_mask: torch.zeros(
                *noisy.shape,
                2,
            ),
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

        self.assertAlmostEqual(
            float(loss),
            (
                3.0 * (2.0 / 3.0)
                + (3.0 / 7.0)
                + (4.0 / 15.0)
            ) * float(torch.log(torch.tensor(2.0))) / 4.0,
            places=5,
        )

    def test_chunk_weight_matches_absorbing_prefix_marginal(self):
        model = _causal_shell()
        sigma = torch.tensor([torch.log(torch.tensor(2.0))])
        dsigma = torch.ones(1)
        chunk_ids = torch.tensor([[0, 1, 2, 3]])

        weights = model._causal_diffusion_weight(
            sigma,
            dsigma,
            chunk_ids,
        )

        self.assertTrue(torch.allclose(
            weights,
            torch.tensor([[1.0, 2.0 / 3.0, 3.0 / 7.0, 4.0 / 15.0]]),
            atol=1e-6,
        ))

    def test_late_energy_guidance_can_override_unsafe_top_probability(self):
        model = _causal_shell()
        model.safety_energy_weight = 2.0
        log_probabilities = torch.log(torch.tensor([[0.6, 0.4]]))
        energies = torch.tensor([[3.0, 0.0]])

        early = model._select_topk_by_energy(
            log_probabilities,
            energies,
            t_value=1.0,
        )
        late = model._select_topk_by_energy(
            log_probabilities,
            energies,
            t_value=0.0,
        )

        self.assertEqual(int(early[0]), 0)
        self.assertEqual(int(late[0]), 1)


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
            self.assertEqual(config.Model.diffusion.encoder_lr_scale, 0.5)
            self.assertGreater(config.Model.total_steps, 32)

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
