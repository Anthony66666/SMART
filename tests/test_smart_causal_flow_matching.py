import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

from smart.model import SMARTCausalFlowMatching
from smart.utils.config import load_config_act


def _flow_shell():
    model = object.__new__(SMARTCausalFlowMatching)
    model.model_config = SimpleNamespace(
        decoder=SimpleNamespace(token_size=5),
    )
    model.num_future_chunks = 4
    model.min_t = 1e-3
    model.flow_loss_weight = 1.0
    model.flow_integration_steps = 2
    model.flow_sampling_strategy = 'argmax'
    model.remask_confidence_temperature = 1.0
    model.safety_energy_enabled = False
    model.guidance_mode = 'none'
    model.diffusion_num_steps = 4
    model.diffusion_decoder = SimpleNamespace(mask_token_id=5)
    return model


class CausalFlowMatchingObjectiveTest(unittest.TestCase):
    def test_flow_interpolation_targets_uniform_to_gt_one_hot_velocity(self):
        model = _flow_shell()
        source = torch.full((1, 2, model.token_size), 1.0 / model.token_size)
        target_ids = torch.tensor([[1, 3]])
        t = torch.tensor([0.25])

        state, velocity = model._flow_interpolate(source, target_ids, t)

        expected_target = torch.nn.functional.one_hot(
            target_ids,
            num_classes=model.token_size,
        ).float()
        expected_state = source + 0.25 * (expected_target - source)
        expected_velocity = expected_target - source
        self.assertTrue(torch.allclose(state, expected_state))
        self.assertTrue(torch.allclose(velocity, expected_velocity))
        self.assertTrue(torch.allclose(state.sum(dim=-1), torch.ones(1, 2)))

    def test_flow_loss_supervises_only_selected_frontier_velocity(self):
        model = _flow_shell()
        model.training = False
        model._sample_frontier_ids = MethodType(
            lambda self, loss_mask_base, chunk_ids: torch.tensor(
                [2],
                device=chunk_ids.device,
            ),
            model,
        )
        model._sample_flow_times = MethodType(
            lambda self, frontier_ids, summary: torch.zeros(
                frontier_ids.shape,
                device=summary.device,
                dtype=summary.dtype,
            ),
            model,
        )
        captured = {}

        def fake_decode(
            self,
            flow_probs,
            proxy_token_ids,
            packed,
            summary,
            t,
            geometry_known_mask,
            **_kwargs,
        ):
            del proxy_token_ids, summary, geometry_known_mask, _kwargs
            source = self._flow_source_distribution(
                packed['token_ids'],
                packed['valid_mask'],
            )
            _state, target_velocity = self._flow_interpolate(
                source,
                packed['token_ids'],
                t,
            )
            captured['flow_probs'] = flow_probs.clone()
            return target_velocity

        model._decode_flow_velocity = MethodType(fake_decode, model)
        packed = {
            'token_ids': torch.tensor([[0, 1, 2, 3]]),
            'valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'loss_mask_base': torch.ones(1, 4, dtype=torch.bool),
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
        }

        loss, acc = model._compute_diffusion_loss(packed, torch.zeros(1, 4))

        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(float(acc), 1.0, places=6)
        self.assertTrue(torch.allclose(
            captured['flow_probs'][0, 0],
            torch.nn.functional.one_hot(
                torch.tensor(0),
                num_classes=model.token_size,
            ).float(),
        ))
        self.assertTrue(torch.allclose(
            captured['flow_probs'][0, 2],
            torch.full((model.token_size,), 1.0 / model.token_size),
        ))


class CausalFlowMatchingSamplerTest(unittest.TestCase):
    def test_sampling_integrates_frontier_flow_and_returns_token_confidence(self):
        model = _flow_shell()
        decode_times = []

        def fake_decode(
            self,
            flow_probs,
            proxy_token_ids,
            packed,
            summary,
            t,
            geometry_known_mask,
            **_kwargs,
        ):
            del proxy_token_ids, summary, geometry_known_mask, _kwargs
            decode_times.append(float(t[0]))
            target_ids = packed['chunk_ids'] + 1
            target = torch.nn.functional.one_hot(
                target_ids.clamp(max=self.token_size - 1),
                num_classes=self.token_size,
            ).to(dtype=flow_probs.dtype)
            return target - flow_probs

        model._decode_flow_velocity = MethodType(fake_decode, model)
        valid_mask = torch.ones(1, 4, dtype=torch.bool)
        packed = {
            'valid_mask': valid_mask,
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
            'agent_type_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        sampled, confidence, trace = model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 4, 2),
            token_headings=torch.zeros(1, 4),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 4, 4),
            agent_type_ids=packed['agent_type_ids'],
            packed=packed,
            return_trace=True,
        )

        self.assertTrue(torch.equal(sampled, torch.tensor([[1, 2, 3, 4]])))
        self.assertTrue(torch.all(confidence[valid_mask] > 0.2))
        self.assertEqual([entry['newly_revealed'] for entry in trace], [1, 1, 1, 1])
        self.assertEqual(len(decode_times), model.diffusion_num_steps * model.flow_integration_steps)

    def test_edit_sampling_keeps_seed_locked_tokens(self):
        model = _flow_shell()

        def fake_decode(
            self,
            flow_probs,
            proxy_token_ids,
            packed,
            summary,
            t,
            geometry_known_mask,
            **_kwargs,
        ):
            del proxy_token_ids, summary, t, geometry_known_mask, _kwargs
            target = torch.zeros_like(flow_probs)
            target[..., 4] = 1.0
            return target - flow_probs

        model._decode_flow_velocity = MethodType(fake_decode, model)
        valid_mask = torch.ones(1, 4, dtype=torch.bool)
        packed = {
            'valid_mask': valid_mask,
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'token_agent_ids': torch.zeros(1, 4, dtype=torch.long),
            'agent_type_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        sampled, confidence = model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 4, 2),
            token_headings=torch.zeros(1, 4),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 4, 4),
            agent_type_ids=packed['agent_type_ids'],
            packed=packed,
            seed_token_ids=torch.tensor([[0, 1, 2, 3]]),
            editable_mask=torch.tensor([[False, True, False, False]]),
        )

        self.assertTrue(torch.equal(sampled, torch.tensor([[0, 4, 2, 3]])))
        self.assertTrue(torch.equal(confidence[0, [0, 2, 3]], torch.ones(3)))


class CausalFlowMatchingConfigTest(unittest.TestCase):
    def test_local_config_selects_flow_matching_predictor(self):
        config = load_config_act(
            'configs/train/train_scalable_causal_flow_matching_local.yaml'
        )

        self.assertEqual(config.Model.predictor, 'smart_causal_flow_matching')
        self.assertEqual(config.Model.diffusion.causal_objective, 'flow_matching_v1')

    def test_flow_matching_model_is_exported_without_replacing_causal_diffusion(self):
        from smart.model import SMARTCausalDiffusion

        self.assertIsNot(SMARTCausalFlowMatching, SMARTCausalDiffusion)

        init_path = Path('smart/model/__init__.py')
        text = init_path.read_text(encoding='utf-8')
        self.assertIn('SMARTCausalDiffusion', text)
        self.assertIn('SMARTCausalFlowMatching', text)


if __name__ == '__main__':
    unittest.main()
