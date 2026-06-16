import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from smart.model import SMARTEmbeddedLanguageFlow
from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.model.smart_causal_diffusion import SMARTCausalDiffusion
from smart.model.smart_diffusion import SMARTDiffusion
from smart.modules.agent_decoder import SMARTAgentDecoder
from smart.modules.diffusion_decoder import DiffusionDecoder
from smart.modules.elf_decoder import EmbeddedLanguageFlowDecoder
from smart.utils.config import load_config_act


def _elf_shell():
    model = object.__new__(SMARTEmbeddedLanguageFlow)
    model.model_config = SimpleNamespace(
        decoder=SimpleNamespace(token_size=5),
    )
    model.num_historical_steps = 11
    model.num_future_steps = 20
    model.future_chunk_steps = 5
    model.history_tokens = 2
    model.elf_sequence_tokens = 4
    model.elf_window_tokens = 2
    model.elf_commit_tokens = 1
    model.elf_receding_horizon = True
    model.num_future_chunks = 4
    model.ar_commit_tokens = 1
    model.hidden_dim = 3
    model.token_size = 5
    model.mask_token_id = 5
    model.min_t = 1e-3
    model.t_eps = 1e-3
    model.elf_loss_weight = 1.0
    model.elf_tail_loss_weight = 0.25
    model.elf_decoder_loss_weight = 0.25
    model.elf_decoder_prob = 0.25
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


class EmbeddedLanguageFlowAttentionTest(unittest.TestCase):
    def test_receding_window_attention_blocks_future_chunks_but_keeps_same_chunk_agents_visible(self):
        decoder = EmbeddedLanguageFlowDecoder(
            text_encoder_dim=3,
            max_length=4,
            hidden_size=4,
            depth=1,
            num_heads=1,
            vocab_size=5,
            num_time_tokens=2,
            num_model_mode_tokens=1,
        )
        valid = torch.tensor([[True, True, True, True, True, False]])
        chunk_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])
        prefix_len = 3

        mask = decoder._build_receding_attention_mask(valid, chunk_ids, prefix_len)

        self.assertEqual(tuple(mask.shape), (1, 9, 9))
        self.assertTrue(mask[0, :prefix_len, :prefix_len].all())
        self.assertFalse(mask[0, :prefix_len, prefix_len:].any())

        query_agent0_chunk0 = prefix_len + 0
        query_agent1_chunk0 = prefix_len + 3
        query_agent0_chunk1 = prefix_len + 1
        query_agent1_chunk1 = prefix_len + 4
        query_agent0_chunk2 = prefix_len + 2
        padding_key = prefix_len + 5

        self.assertTrue(mask[0, query_agent0_chunk0, :prefix_len].all())
        self.assertTrue(mask[0, query_agent0_chunk0, query_agent0_chunk0])
        self.assertTrue(mask[0, query_agent0_chunk0, query_agent1_chunk0])
        self.assertFalse(mask[0, query_agent0_chunk0, query_agent0_chunk1])
        self.assertFalse(mask[0, query_agent0_chunk0, query_agent0_chunk2])

        self.assertTrue(mask[0, query_agent1_chunk1, query_agent0_chunk0])
        self.assertTrue(mask[0, query_agent1_chunk1, query_agent0_chunk1])
        self.assertTrue(mask[0, query_agent1_chunk1, query_agent1_chunk1])
        self.assertFalse(mask[0, query_agent1_chunk1, query_agent0_chunk2])
        self.assertFalse(mask[0, query_agent1_chunk1, padding_key])


class EmbeddedLanguageFlowCurrentStateTest(unittest.TestCase):
    def test_agent_history_context_map_mask_can_include_all_generation_agents(self):
        decoder = SimpleNamespace(
            num_historical_steps=11,
            shift=5,
            hidden_dim=3,
            num_layers=0,
        )
        data = {
            'agent': {
                'token_pos': torch.zeros(3, 2, 2),
                'token_heading': torch.zeros(3, 2),
                'category': torch.tensor([1, 3, 2]),
                'token_idx': torch.zeros(3, 2, dtype=torch.long),
                'agent_valid_mask': torch.ones(3, 2, dtype=torch.bool),
            }
        }
        captured = {}

        def fake_agent_token_embedding(data_arg, category, token_index, pos, head_vector):
            del data_arg, category, token_index, pos, head_vector
            return torch.zeros(3, 2, 3), None

        def fake_temporal_edge(*_args, **_kwargs):
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 3)

        def fake_temporal_batches(_data, num_step, pos):
            del _data
            return torch.arange(num_step).repeat_interleave(pos.shape[0]), torch.arange(num_step), torch.zeros(pos.shape[0], dtype=torch.long)

        def fake_interaction_edge(*_args, **_kwargs):
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 3)

        def fake_map2agent_edge(_data, num_step, agent_category, pos_a, head_a, head_vector_a, mask, *_args, **_kwargs):
            del _data, num_step, agent_category, pos_a, head_a, head_vector_a
            captured['map_mask'] = mask.clone()
            return torch.empty(2, 0, dtype=torch.long), torch.empty(0, 3)

        decoder.agent_token_embedding = fake_agent_token_embedding
        decoder.build_temporal_edge = fake_temporal_edge
        decoder._build_temporal_batches = fake_temporal_batches
        decoder.build_interaction_edge = fake_interaction_edge
        decoder.build_map2agent_edge = fake_map2agent_edge
        map_agent_mask = torch.tensor([True, True, False])

        context = SMARTAgentDecoder.encode_history_context(
            decoder,
            data,
            {'x_pt': torch.zeros(1, 3)},
            map_agent_mask=map_agent_mask,
        )

        self.assertTrue(torch.equal(captured['map_mask'], map_agent_mask[:, None].expand(-1, 2)))
        self.assertEqual(tuple(context['x_a_history'].shape), (3, 2, 3))

    def test_build_inputs_uses_standalone_encoder_and_packer(self):
        model = _elf_shell()
        context = {'x_a_history': torch.ones(1, 2, 3)}
        packed = {
            'valid_mask': torch.ones(1, 1, dtype=torch.bool),
            'agent_batch': torch.zeros(1, dtype=torch.long),
        }
        calls = []

        def fake_encode(data, **kwargs):
            calls.append(('encode', data, kwargs))
            return context

        def fake_pack(self, data, context_arg):
            calls.append(('pack', data, context_arg))
            return packed, torch.zeros(1, 3)

        model.encoder = SimpleNamespace(encode_history_context=fake_encode)
        model._pack_future_window = MethodType(fake_pack, model)
        data = {
            'agent': {
                'valid_mask': torch.ones(1, model.num_historical_steps, dtype=torch.bool),
                'type': torch.zeros(1, dtype=torch.long),
            }
        }
        expected_map_agent_mask = torch.ones(1, dtype=torch.bool)

        output = model._build_diffusion_inputs(data)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], 'encode')
        self.assertIs(calls[0][1], data)
        self.assertTrue(torch.equal(calls[0][2]['map_agent_mask'], expected_map_agent_mask))
        self.assertEqual(calls[1], ('pack', data, context))
        self.assertIs(output[0], packed)

    def test_build_inputs_accepts_receding_window_bounds(self):
        model = _elf_shell()
        context = {'x_a_history': torch.ones(1, 2, 3)}
        packed = {
            'valid_mask': torch.ones(1, 1, dtype=torch.bool),
            'agent_batch': torch.zeros(1, dtype=torch.long),
        }
        calls = []

        def fake_encode(data, **kwargs):
            calls.append(('encode', data, kwargs))
            return context

        def fake_pack(self, data, context_arg, *, token_start, sequence_tokens):
            calls.append(('pack', data, context_arg, token_start, sequence_tokens))
            return packed, torch.zeros(1, 3)

        model.encoder = SimpleNamespace(encode_history_context=fake_encode)
        model._pack_future_window = MethodType(fake_pack, model)
        data = {
            'agent': {
                'valid_mask': torch.ones(1, model.num_historical_steps, dtype=torch.bool),
                'type': torch.zeros(1, dtype=torch.long),
            }
        }
        expected_map_agent_mask = torch.ones(1, dtype=torch.bool)

        output = model._build_diffusion_inputs(data, token_start=7, sequence_tokens=2)

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][0], 'encode')
        self.assertIs(calls[0][1], data)
        self.assertTrue(torch.equal(calls[0][2]['map_agent_mask'], expected_map_agent_mask))
        self.assertEqual(calls[1], ('pack', data, context, 7, 2))
        self.assertIs(output[0], packed)

    def test_inference_reencodes_after_each_committed_window(self):
        model = _elf_shell()
        model.elf_sequence_tokens = 4
        model.elf_window_tokens = 2
        model.elf_commit_tokens = 1
        data = SimpleNamespace(tag='seed')
        calls = []

        def fake_prepare(self, batch):
            return batch

        def fake_build(self, batch, *, token_start, sequence_tokens, rollout_valid=False):
            del rollout_valid
            calls.append((batch.tag, token_start, sequence_tokens))
            valid = torch.ones(1, sequence_tokens, dtype=torch.bool)
            packed = {
                'chunk_ids': torch.arange(sequence_tokens).view(1, -1),
                'valid_mask': valid,
                'context': torch.zeros(1, sequence_tokens, self.hidden_dim),
                'agent_type_ids': torch.zeros(1, sequence_tokens, dtype=torch.long),
                'sequence_tokens': sequence_tokens,
                'num_agents': 1,
                'token_valid_by_agent': valid.clone(),
            }
            summary = torch.full((1, self.hidden_dim), float(len(calls)))
            return packed, summary, None, None, None, None, torch.zeros(1, dtype=torch.long)

        def fake_sample(self, summary, *args, packed=None, **kwargs):
            del args, kwargs
            value = int(summary[0, 0].item())
            return (
                torch.full((1, packed['sequence_tokens']), value, dtype=torch.long),
                torch.full((1, packed['sequence_tokens']), value / 10.0),
            )

        def fake_commit(self, batch, token_ids, token_confidence, token_valid, *, token_start, commit_tokens):
            del token_confidence, token_valid
            committed = int(token_ids[0, 0].item())
            self._test_commit_tokens.append(commit_tokens)
            return SimpleNamespace(tag=f'{batch.tag}->{token_start}:{committed}')

        def fake_decode(self, batch, token_ids, token_confidence, token_valid):
            del batch, token_confidence, token_valid
            return {'next_token_idx': token_ids.clone()}

        model._prepare_batch = MethodType(fake_prepare, model)
        model._build_diffusion_inputs = MethodType(fake_build, model)
        model._diffusion_sample = MethodType(fake_sample, model)
        model._test_commit_tokens = []
        model._unpack_agent_tokens = MethodType(
            lambda self, packed, sampled, confidence: (sampled, confidence),
            model,
        )
        model._commit_elf_tokens_to_rollout_data = MethodType(fake_commit, model)
        model._decode_token_rollout = MethodType(fake_decode, model)

        pred = model.inference(data)

        self.assertEqual(
            calls,
            [
                ('seed', 2, 2),
                ('seed->2:1', 3, 2),
                ('seed->2:1->3:2', 4, 2),
                ('seed->2:1->3:2->4:3', 5, 1),
            ],
        )
        self.assertEqual(model._test_commit_tokens, [1, 1, 1, 1])
        self.assertTrue(torch.equal(pred['next_token_idx'], torch.tensor([[1, 2, 3, 4]])))

    def test_training_view_moves_history_anchor_to_future_token_slot(self):
        model = _elf_shell()
        agent = {
            'token_pos': torch.tensor([[[0.0, 0.0], [1.0, 0.0], [6.0, 2.0], [9.0, 3.0]]]),
            'token_heading': torch.tensor([[0.0, 0.1, 0.6, 0.9]]),
            'agent_valid_mask': torch.ones(1, 4, dtype=torch.bool),
            'type': torch.zeros(1, dtype=torch.long),
            'position': torch.zeros(1, model.num_historical_steps + model.num_future_steps, 3),
            'heading': torch.zeros(1, model.num_historical_steps + model.num_future_steps),
            'valid_mask': torch.ones(1, model.num_historical_steps + model.num_future_steps, dtype=torch.bool),
        }
        frame_anchor = model.num_historical_steps + 2 * model.future_chunk_steps - 1
        agent['position'][0, frame_anchor, :2] = torch.tensor([9.0, 3.0])
        agent['heading'][0, frame_anchor] = 0.9
        data = {'agent': agent}

        view = model._build_elf_training_view(data, window_offset=2)

        self.assertTrue(torch.equal(data['agent']['position'][0, model.num_historical_steps - 1, :2], torch.zeros(2)))
        self.assertTrue(torch.allclose(
            view['agent']['position'][0, model.num_historical_steps - 1, :2],
            torch.tensor([9.0, 3.0]),
        ))
        self.assertAlmostEqual(float(view['agent']['heading'][0, model.num_historical_steps - 1]), 0.9)

    def test_training_view_rolls_history_tokens_to_preceding_window_slots(self):
        model = _elf_shell()
        agent = {
            'token_idx': torch.tensor([[10, 11, 12, 13, 14]]),
            'token_pos': torch.tensor([[[0.0, 0.0], [1.0, 0.0], [6.0, 2.0], [9.0, 3.0], [12.0, 4.0]]]),
            'token_heading': torch.tensor([[0.0, 0.1, 0.6, 0.9, 1.2]]),
            'agent_valid_mask': torch.ones(1, 5, dtype=torch.bool),
            'type': torch.zeros(1, dtype=torch.long),
            'position': torch.zeros(1, model.num_historical_steps + model.num_future_steps, 3),
            'heading': torch.zeros(1, model.num_historical_steps + model.num_future_steps),
            'valid_mask': torch.ones(1, model.num_historical_steps + model.num_future_steps, dtype=torch.bool),
        }
        data = {'agent': agent}

        view = model._build_elf_training_view(data, window_offset=2)

        self.assertTrue(torch.equal(view['agent']['token_idx'][0, :2], torch.tensor([12, 13])))
        self.assertTrue(torch.allclose(
            view['agent']['token_pos'][0, :2],
            torch.tensor([[6.0, 2.0], [9.0, 3.0]]),
        ))
        self.assertTrue(torch.allclose(
            view['agent']['token_heading'][0, :2],
            torch.tensor([0.6, 0.9]),
        ))

    def test_commit_rolls_history_tokens_after_receding_step(self):
        model = _elf_shell()
        total_frames = model.num_historical_steps + model.num_future_steps
        agent = {
            'token_idx': torch.tensor([[10, 11, 0, 0]]),
            'token_pos': torch.tensor([[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0]]]),
            'token_heading': torch.tensor([[0.0, 0.1, 0.0, 0.0]]),
            'agent_valid_mask': torch.tensor([[True, True, False, False]]),
            'type': torch.zeros(1, dtype=torch.long),
            'position': torch.zeros(1, total_frames, 3),
            'heading': torch.zeros(1, total_frames),
            'valid_mask': torch.ones(1, total_frames, dtype=torch.bool),
        }
        agent['position'][0, model.num_historical_steps - 1, :2] = torch.tensor([1.0, 0.0])
        agent['heading'][0, model.num_historical_steps - 1] = 0.1
        data = {'agent': agent}

        def fake_decode(self, batch, token_ids, token_valid, start_pos, start_heading):
            del batch, token_ids, token_valid, start_pos, start_heading
            return (
                torch.tensor([[[2.0, 0.0], [2.5, 0.0], [3.0, 0.0], [3.5, 0.0], [4.0, 0.0]]]),
                torch.full((1, 5), 0.4),
                torch.ones(1, 5, dtype=torch.bool),
                torch.tensor([[[4.0, 0.0]]]),
                torch.tensor([[0.4]]),
                torch.tensor([[4.0, 0.0]]),
                torch.tensor([0.4]),
            )

        model._decode_elf_token_sequence = MethodType(fake_decode, model)

        updated = model._commit_elf_tokens_to_rollout_data(
            data,
            torch.tensor([[4]]),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
            token_start=model.history_tokens,
            commit_tokens=1,
        )

        self.assertTrue(torch.equal(updated['agent']['token_idx'][0, :2], torch.tensor([11, 4])))
        self.assertTrue(torch.allclose(
            updated['agent']['token_pos'][0, :2],
            torch.tensor([[1.0, 0.0], [4.0, 0.0]]),
        ))
        self.assertTrue(torch.allclose(updated['agent']['token_heading'][0, :2], torch.tensor([0.1, 0.4])))
        self.assertTrue(torch.equal(updated['agent']['agent_valid_mask'][0, :2], torch.tensor([True, True])))


class EmbeddedLanguageFlowConfigTest(unittest.TestCase):
    def test_1000_step_config_selects_elf_predictor(self):
        config = load_config_act('configs/train/train_scalable_elf_1000.yaml')

        self.assertEqual(config.Model.predictor, 'smart_elf')
        self.assertEqual(config.Model.diffusion.elf_objective, 'embedded_language_flow_v1')
        self.assertFalse(hasattr(config.Model.diffusion, 'ar_objective'))
        self.assertFalse(hasattr(config.Model.diffusion, 'causal_objective'))
        self.assertEqual(config.Model.diffusion.elf_sequence_tokens, 16)
        self.assertEqual(config.Model.diffusion.elf_window_tokens, 4)
        self.assertEqual(config.Model.diffusion.elf_commit_tokens, 1)
        self.assertTrue(config.Model.diffusion.elf_receding_horizon)
        self.assertEqual(config.Model.diffusion.elf_decoder_prob, 0.25)
        self.assertAlmostEqual(config.Model.diffusion.elf_tail_loss_weight, 0.25)
        self.assertFalse(config.Model.diffusion.target_category_only)
        self.assertEqual(config.Model.diffusion.supervision_mode, 'all_agents')
        self.assertEqual(config.Trainer.max_steps, 1000)
        self.assertEqual(config.Trainer.val_check_interval, 1000)
        self.assertEqual(config.Trainer.monitor_metric, 'val_minADE')

    def test_3epoch_local_config_trains_by_epoch(self):
        config = load_config_act('configs/train/train_scalable_elf_3epoch_local.yaml')

        self.assertEqual(config.Model.predictor, 'smart_elf')
        self.assertEqual(config.Model.diffusion.elf_objective, 'embedded_language_flow_v1')
        self.assertFalse(hasattr(config.Model.diffusion, 'ar_objective'))
        self.assertFalse(hasattr(config.Model.diffusion, 'causal_objective'))
        self.assertEqual(config.Model.diffusion.elf_sequence_tokens, 16)
        self.assertEqual(config.Model.diffusion.elf_window_tokens, 4)
        self.assertEqual(config.Model.diffusion.elf_commit_tokens, 1)
        self.assertTrue(config.Model.diffusion.elf_receding_horizon)
        self.assertEqual(config.Model.diffusion.elf_decoder_prob, 0.25)
        self.assertAlmostEqual(config.Model.diffusion.elf_tail_loss_weight, 0.25)
        self.assertFalse(config.Model.diffusion.target_category_only)
        self.assertEqual(config.Model.diffusion.supervision_mode, 'all_agents')
        self.assertEqual(config.Trainer.max_epochs, 3)
        self.assertEqual(config.Trainer.max_steps, -1)
        self.assertEqual(config.Trainer.val_check_interval, 1.0)
        self.assertIsNone(config.Trainer.checkpoint_every_n_train_steps)
        self.assertEqual(config.Trainer.monitor_metric, 'val_minADE')
        self.assertEqual(config.Model.total_steps, 3)
        self.assertEqual(config.Visualization.output_dir, './outputs/val_elf_3epoch')

    def test_elf_model_is_exported_without_replacing_existing_paths(self):
        from smart.model import SMARTCausalFlowMatching

        self.assertFalse(issubclass(SMARTEmbeddedLanguageFlow, SMARTAutoregressiveDiffusion))
        self.assertFalse(issubclass(SMARTEmbeddedLanguageFlow, SMARTCausalDiffusion))
        self.assertFalse(issubclass(SMARTEmbeddedLanguageFlow, SMARTDiffusion))
        self.assertIsNot(SMARTEmbeddedLanguageFlow, SMARTCausalDiffusion)
        self.assertIsNot(SMARTEmbeddedLanguageFlow, SMARTCausalFlowMatching)
        self.assertFalse(issubclass(EmbeddedLanguageFlowDecoder, DiffusionDecoder))

        init_path = Path('smart/model/__init__.py')
        text = init_path.read_text(encoding='utf-8')
        self.assertIn('SMARTEmbeddedLanguageFlow', text)
        self.assertIn('SMARTCausalFlowMatching', text)


if __name__ == '__main__':
    unittest.main()
