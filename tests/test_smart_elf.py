import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from smart.model import SMARTEmbeddedLanguageFlow
from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.model.smart_causal_diffusion import SMARTCausalDiffusion
from smart.utils.config import load_config_act


def _elf_shell():
    model = object.__new__(SMARTEmbeddedLanguageFlow)
    model.model_config = SimpleNamespace(
        decoder=SimpleNamespace(token_size=5),
    )
    model.num_future_chunks = 4
    model.ar_commit_tokens = 1
    model.hidden_dim = 3
    model.min_t = 1e-3
    model.elf_loss_weight = 1.0
    model.elf_tail_loss_weight = 0.25
    model.elf_decoder_loss_weight = 0.25
    model.elf_integration_steps = 2
    model.elf_sampling_strategy = 'argmax'
    model.elf_noise_scale = 0.0
    model.remask_confidence_temperature = 1.0
    model.safety_energy_enabled = False
    model.guidance_mode = 'none'
    model.diffusion_num_steps = 4
    model.diffusion_decoder = SimpleNamespace(mask_token_id=5)
    model.use_proposal_geometry = True
    model.proposal_conditioning_enabled = False
    model.geometry_confidence_source_threshold = 0.0
    return model


class EmbeddedLanguageFlowObjectiveTest(unittest.TestCase):
    def test_embedding_interpolation_targets_noise_to_token_embedding_velocity(self):
        model = _elf_shell()
        source = torch.tensor([[[0.0, 0.5, 1.0], [1.0, 0.0, -1.0]]])
        target = torch.tensor([[[1.0, 1.5, 2.0], [2.0, 1.0, 0.0]]])
        t = torch.tensor([0.25])

        state, velocity = model._elf_interpolate(source, target, t)

        self.assertTrue(torch.allclose(velocity, target - source))
        self.assertTrue(torch.allclose(state, source + 0.25 * (target - source)))

    def test_elf_loss_supervises_full_window_embedding_velocity(self):
        model = _elf_shell()
        model.training = False
        model._sample_frontier_ids = MethodType(
            lambda self, *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("ELF loss must not sample a causal frontier.")
            ),
            model,
        )
        model._sample_elf_times = MethodType(
            lambda self, valid_mask, summary: torch.zeros(
                valid_mask.shape[:1],
                device=summary.device,
                dtype=summary.dtype,
            ),
            model,
        )

        def fake_targets(self, token_ids, agent_type_ids, valid_mask=None):
            del self, agent_type_ids
            base = torch.stack(
                [
                    token_ids.float(),
                    token_ids.float() + 1.0,
                    token_ids.float() + 2.0,
                ],
                dim=-1,
            )
            if valid_mask is not None:
                base = base * valid_mask.unsqueeze(-1).to(base.dtype)
            return base

        def zero_sources(self, target_embeddings, valid_mask, **_kwargs):
            del self, _kwargs
            return torch.zeros_like(target_embeddings) * valid_mask.unsqueeze(-1).to(
                target_embeddings.dtype
            )

        captured = {}

        def fake_decode(
            self,
            elf_embeddings,
            proxy_token_ids,
            packed,
            summary,
            t,
            geometry_known_mask,
            **_kwargs,
        ):
            del proxy_token_ids, summary, geometry_known_mask, _kwargs
            captured['t_shape'] = tuple(t.shape)
            target = self._elf_target_embeddings(
                packed['token_ids'],
                packed['agent_type_ids'],
                packed['valid_mask'],
            )
            source = self._elf_source_embeddings(target, packed['valid_mask'])
            _state, target_velocity = self._elf_interpolate(source, target, t)
            captured['elf_embeddings'] = elf_embeddings.clone()
            logits = torch.full((*packed['token_ids'].shape, self.token_size), -20.0)
            logits.scatter_(-1, packed['token_ids'].unsqueeze(-1), 20.0)
            return target_velocity, logits

        model._elf_target_embeddings = MethodType(fake_targets, model)
        model._elf_source_embeddings = MethodType(zero_sources, model)
        model._elf_proxy_token_ids = MethodType(lambda self, emb, *_args: emb[..., 0].long().clamp(0, 4), model)
        model._decode_elf_velocity = MethodType(fake_decode, model)
        packed = {
            'token_ids': torch.tensor([[0, 1, 2, 3]]),
            'valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'loss_mask_base': torch.ones(1, 4, dtype=torch.bool),
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'agent_type_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        loss, acc = model._compute_diffusion_loss(packed, torch.zeros(1, 4))

        self.assertAlmostEqual(float(loss), 0.0, places=6)
        self.assertAlmostEqual(float(acc), 1.0, places=6)
        self.assertEqual(captured['t_shape'], (1,))
        self.assertTrue(torch.allclose(
            captured['elf_embeddings'][0, 0],
            1e-3 * torch.tensor([0.0, 1.0, 2.0]),
        ))
        self.assertTrue(torch.allclose(
            captured['elf_embeddings'][0, 2],
            1e-3 * torch.tensor([2.0, 3.0, 4.0]),
        ))
        self.assertTrue(torch.allclose(
            captured['elf_embeddings'][0, 3],
            1e-3 * torch.tensor([3.0, 4.0, 5.0]),
        ))

    def test_elf_flow_loss_averages_embedding_dimensions_like_original_elf(self):
        model = _elf_shell()
        model.training = False
        model.elf_decoder_loss_weight = 0.0
        model._sample_elf_times = MethodType(
            lambda self, valid_mask, summary: torch.zeros(
                valid_mask.shape[:1],
                device=summary.device,
                dtype=summary.dtype,
            ),
            model,
        )

        def fake_targets(self, token_ids, agent_type_ids, valid_mask=None):
            del self, agent_type_ids
            target = torch.stack(
                [
                    token_ids.float(),
                    token_ids.float() + 1.0,
                    token_ids.float() + 2.0,
                ],
                dim=-1,
            )
            if valid_mask is not None:
                target = target * valid_mask.unsqueeze(-1).to(target.dtype)
            return target

        def zero_sources(self, target_embeddings, valid_mask, **_kwargs):
            del self, _kwargs
            return torch.zeros_like(target_embeddings) * valid_mask.unsqueeze(-1).to(
                target_embeddings.dtype
            )

        def zero_velocity_decode(
            self,
            elf_embeddings,
            proxy_token_ids,
            packed,
            summary,
            t,
            geometry_known_mask,
            **_kwargs,
        ):
            del elf_embeddings, proxy_token_ids, summary, t, geometry_known_mask, _kwargs
            logits = torch.full((*packed['token_ids'].shape, self.token_size), -20.0)
            logits.scatter_(-1, packed['token_ids'].unsqueeze(-1), 20.0)
            velocity = torch.zeros(*packed['token_ids'].shape, self.hidden_dim)
            return velocity, logits

        model._elf_target_embeddings = MethodType(fake_targets, model)
        model._elf_source_embeddings = MethodType(zero_sources, model)
        model._elf_proxy_token_ids = MethodType(
            lambda self, emb, agent_type_ids, valid_mask: torch.zeros_like(valid_mask, dtype=torch.long),
            model,
        )
        model._decode_elf_velocity = MethodType(zero_velocity_decode, model)
        packed = {
            'token_ids': torch.tensor([[1]]),
            'valid_mask': torch.ones(1, 1, dtype=torch.bool),
            'loss_mask_base': torch.ones(1, 1, dtype=torch.bool),
            'chunk_ids': torch.tensor([[0]]),
            'agent_type_ids': torch.zeros(1, 1, dtype=torch.long),
        }

        loss, _acc = model._compute_diffusion_loss(packed, torch.zeros(1, 4))

        expected = torch.tensor([1.0, 2.0, 3.0]).pow(2).mean()
        self.assertAlmostEqual(float(loss), float(expected), places=6)

    def test_elf_loss_prioritizes_committed_token_over_tail_proposals(self):
        model = _elf_shell()
        model.training = False
        model.hidden_dim = 1
        model.elf_decoder_loss_weight = 0.0
        model.elf_tail_loss_weight = 0.25
        model.ar_commit_tokens = 1
        model._sample_elf_times = MethodType(
            lambda self, valid_mask, summary: torch.zeros(
                valid_mask.shape[:1],
                device=summary.device,
                dtype=summary.dtype,
            ),
            model,
        )

        def fake_targets(self, token_ids, agent_type_ids, valid_mask=None):
            del self, agent_type_ids
            target = token_ids.float().unsqueeze(-1)
            if valid_mask is not None:
                target = target * valid_mask.unsqueeze(-1).to(target.dtype)
            return target

        def zero_sources(self, target_embeddings, valid_mask, **_kwargs):
            del self, _kwargs
            return torch.zeros_like(target_embeddings) * valid_mask.unsqueeze(-1).to(
                target_embeddings.dtype
            )

        def zero_velocity_decode(
            self,
            elf_embeddings,
            proxy_token_ids,
            packed,
            summary,
            t,
            geometry_known_mask,
            **_kwargs,
        ):
            del elf_embeddings, proxy_token_ids, summary, t, geometry_known_mask, _kwargs
            logits = torch.full((*packed['token_ids'].shape, self.token_size), -20.0)
            logits.scatter_(-1, packed['token_ids'].unsqueeze(-1), 20.0)
            velocity = torch.zeros(*packed['token_ids'].shape, self.hidden_dim)
            return velocity, logits

        model._elf_target_embeddings = MethodType(fake_targets, model)
        model._elf_source_embeddings = MethodType(zero_sources, model)
        model._elf_proxy_token_ids = MethodType(
            lambda self, emb, agent_type_ids, valid_mask: torch.zeros_like(valid_mask, dtype=torch.long),
            model,
        )
        model._decode_elf_velocity = MethodType(zero_velocity_decode, model)
        packed = {
            'token_ids': torch.tensor([[1, 2, 3, 4]]),
            'valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'loss_mask_base': torch.ones(1, 4, dtype=torch.bool),
            'chunk_ids': torch.tensor([[0, 1, 2, 3]]),
            'agent_type_ids': torch.zeros(1, 4, dtype=torch.long),
        }

        loss, _acc = model._compute_diffusion_loss(packed, torch.zeros(1, 4))

        commit_loss = torch.tensor(1.0).pow(2)
        tail_loss = torch.tensor([2.0, 3.0, 4.0]).pow(2).mean()
        expected = commit_loss + 0.25 * tail_loss
        self.assertAlmostEqual(float(loss), float(expected), places=6)


class EmbeddedLanguageFlowSamplerTest(unittest.TestCase):
    def test_sampling_integrates_embedding_flow_and_returns_token_confidence(self):
        model = _elf_shell()
        model.elf_integration_steps = 1
        decode_times = []

        def fake_targets(self, token_ids, agent_type_ids, valid_mask=None):
            del self, agent_type_ids
            target = torch.stack(
                [
                    token_ids.float(),
                    torch.zeros_like(token_ids, dtype=torch.float),
                    torch.zeros_like(token_ids, dtype=torch.float),
                ],
                dim=-1,
            )
            if valid_mask is not None:
                target = target * valid_mask.unsqueeze(-1).to(target.dtype)
            return target

        def fake_decode(
            self,
            elf_embeddings,
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
            target = self._elf_target_embeddings(
                target_ids,
                packed['agent_type_ids'],
                packed['valid_mask'],
            )
            logits = torch.full((*target_ids.shape, self.token_size), -20.0)
            logits.scatter_(-1, target_ids.unsqueeze(-1), 20.0)
            return target - elf_embeddings, logits

        model._elf_target_embeddings = MethodType(fake_targets, model)
        model._elf_source_embeddings = MethodType(
            lambda self, target, valid_mask, **_kwargs: torch.zeros_like(target),
            model,
        )
        model._elf_proxy_token_ids = MethodType(
            lambda self, emb, agent_type_ids, valid_mask: emb[..., 0].round().long().clamp(0, 4).masked_fill(~valid_mask, 0),
            model,
        )
        model._decode_elf_velocity = MethodType(fake_decode, model)
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
        self.assertTrue(torch.all(confidence[valid_mask] > 0.99))
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]['sampled'], 4)
        self.assertEqual(trace[0]['mode'], 'full_window')
        self.assertEqual(len(decode_times), model.elf_integration_steps + 1)

    def test_sampling_projects_final_embeddings_instead_of_auxiliary_logits(self):
        model = _elf_shell()
        model.elf_integration_steps = 1

        def fake_targets(self, token_ids, agent_type_ids, valid_mask=None):
            del self, agent_type_ids
            target = torch.nn.functional.one_hot(
                token_ids.clamp(min=0, max=4),
                num_classes=5,
            ).float()[..., :3]
            if valid_mask is not None:
                target = target * valid_mask.unsqueeze(-1).to(target.dtype)
            return target

        def proxy_from_embedding(self, emb, agent_type_ids, valid_mask):
            del self, agent_type_ids
            proxy = emb.argmax(dim=-1).clamp(max=4)
            return proxy.masked_fill(~valid_mask, 0)

        def fake_decode(
            self,
            elf_embeddings,
            proxy_token_ids,
            packed,
            summary,
            t,
            geometry_known_mask,
            **_kwargs,
        ):
            del proxy_token_ids, summary, t, geometry_known_mask, _kwargs
            target_ids = packed['chunk_ids'] + 1
            target = self._elf_target_embeddings(
                target_ids,
                packed['agent_type_ids'],
                packed['valid_mask'],
            )
            logits = torch.full((*target_ids.shape, self.token_size), -20.0)
            logits[..., 0] = 20.0
            return target - elf_embeddings, logits

        model._elf_target_embeddings = MethodType(fake_targets, model)
        model._elf_source_embeddings = MethodType(
            lambda self, target, valid_mask, **_kwargs: torch.zeros_like(target),
            model,
        )
        model._elf_proxy_token_ids = MethodType(proxy_from_embedding, model)
        model._decode_elf_velocity = MethodType(fake_decode, model)
        valid_mask = torch.ones(1, 3, dtype=torch.bool)
        packed = {
            'valid_mask': valid_mask,
            'chunk_ids': torch.tensor([[0, 1, 2]]),
            'token_agent_ids': torch.zeros(1, 3, dtype=torch.long),
            'agent_type_ids': torch.zeros(1, 3, dtype=torch.long),
        }

        sampled, _confidence, trace = model._diffusion_sample(
            summary=torch.zeros(1, 4),
            token_positions=torch.zeros(1, 3, 2),
            token_headings=torch.zeros(1, 3),
            token_agent_ids=packed['token_agent_ids'],
            chunk_ids=packed['chunk_ids'],
            valid_mask=valid_mask,
            agent_context=torch.zeros(1, 3, 4),
            agent_type_ids=packed['agent_type_ids'],
            packed=packed,
            return_trace=True,
        )

        self.assertTrue(torch.equal(sampled, torch.tensor([[1, 2, 0]])))
        self.assertEqual(trace[0]['mode'], 'full_window')


class EmbeddedLanguageFlowCurrentStateTest(unittest.TestCase):
    def test_maskgit_elf_build_inputs_still_applies_current_state_context(self):
        model = _elf_shell()
        model.ar_objective = 'maskgit'
        model.current_state_enabled = True
        packed = {'valid_mask': torch.ones(1, 1, dtype=torch.bool)}
        result = (packed, torch.zeros(1, 3), None, None, None, None, None)
        calls = []

        def fake_apply(self, data, packed_arg):
            calls.append((data, packed_arg))
            packed_arg['current_state_applied'] = True

        model._apply_current_state_context = MethodType(fake_apply, model)

        with patch.object(
            SMARTAutoregressiveDiffusion,
            '_build_diffusion_inputs',
            return_value=result,
        ):
            output = model._build_diffusion_inputs(object())

        self.assertEqual(len(calls), 1)
        self.assertIs(output[0], packed)
        self.assertTrue(output[0]['current_state_applied'])


class EmbeddedLanguageFlowConfigTest(unittest.TestCase):
    def test_1000_step_config_selects_elf_predictor(self):
        config = load_config_act('configs/train/train_scalable_elf_1000.yaml')

        self.assertEqual(config.Model.predictor, 'smart_elf')
        self.assertEqual(config.Model.diffusion.elf_objective, 'embedded_language_flow_v1')
        self.assertEqual(config.Model.diffusion.ar_objective, 'maskgit')
        self.assertEqual(config.Model.diffusion.prediction_tokens, 4)
        self.assertEqual(config.Model.diffusion.commit_tokens, 1)
        self.assertTrue(config.Model.diffusion.carry_tail_proposal)
        self.assertTrue(config.Model.diffusion.rolling_anchor_training)
        self.assertGreater(config.Model.diffusion.closed_loop_batch_ratio_max, 0.0)
        self.assertEqual(config.Model.diffusion.closed_loop_max_depth, 4)
        self.assertAlmostEqual(config.Model.diffusion.elf_tail_loss_weight, 0.25)
        self.assertFalse(hasattr(config.Model.diffusion, 'causal_objective'))
        self.assertEqual(config.Trainer.max_steps, 1000)
        self.assertEqual(config.Trainer.val_check_interval, 1000)
        self.assertEqual(config.Trainer.monitor_metric, 'val_minADE')

    def test_3epoch_local_config_trains_by_epoch(self):
        config = load_config_act('configs/train/train_scalable_elf_3epoch_local.yaml')

        self.assertEqual(config.Model.predictor, 'smart_elf')
        self.assertEqual(config.Model.diffusion.elf_objective, 'embedded_language_flow_v1')
        self.assertEqual(config.Model.diffusion.ar_objective, 'maskgit')
        self.assertEqual(config.Model.diffusion.prediction_tokens, 4)
        self.assertEqual(config.Model.diffusion.commit_tokens, 1)
        self.assertTrue(config.Model.diffusion.carry_tail_proposal)
        self.assertTrue(config.Model.diffusion.rolling_anchor_training)
        self.assertGreater(config.Model.diffusion.closed_loop_batch_ratio_max, 0.0)
        self.assertEqual(config.Model.diffusion.closed_loop_max_depth, 4)
        self.assertAlmostEqual(config.Model.diffusion.elf_tail_loss_weight, 0.25)
        self.assertEqual(config.Trainer.max_epochs, 3)
        self.assertEqual(config.Trainer.max_steps, -1)
        self.assertEqual(config.Trainer.val_check_interval, 1.0)
        self.assertIsNone(config.Trainer.checkpoint_every_n_train_steps)
        self.assertEqual(config.Trainer.monitor_metric, 'val_minADE')
        self.assertEqual(config.Model.total_steps, 3)
        self.assertEqual(config.Visualization.output_dir, './outputs/val_elf_3epoch')

    def test_elf_model_is_exported_without_replacing_existing_paths(self):
        from smart.model import SMARTCausalFlowMatching

        self.assertTrue(issubclass(SMARTEmbeddedLanguageFlow, SMARTAutoregressiveDiffusion))
        self.assertFalse(issubclass(SMARTEmbeddedLanguageFlow, SMARTCausalDiffusion))
        self.assertIsNot(SMARTEmbeddedLanguageFlow, SMARTCausalDiffusion)
        self.assertIsNot(SMARTEmbeddedLanguageFlow, SMARTCausalFlowMatching)

        init_path = Path('smart/model/__init__.py')
        text = init_path.read_text(encoding='utf-8')
        self.assertIn('SMARTEmbeddedLanguageFlow', text)
        self.assertIn('SMARTCausalFlowMatching', text)


if __name__ == '__main__':
    unittest.main()
