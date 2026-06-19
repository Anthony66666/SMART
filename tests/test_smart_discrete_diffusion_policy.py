import unittest
from types import MethodType, SimpleNamespace

import torch
from torch_geometric.data import HeteroData

from smart.model.smart_discrete_diffusion_policy import SMARTDiscreteDiffusionPolicy
from smart.model.smart_ar_diffusion import SMARTAutoregressiveDiffusion
from smart.utils.config import load_config_act


def _policy_shell():
    model = object.__new__(SMARTDiscreteDiffusionPolicy)
    model.num_historical_steps = 11
    model.num_future_steps = 80
    model.future_chunk_steps = 5
    model.num_future_chunks = 4
    model.full_num_future_chunks = 16
    model.ar_history_tokens = 2
    model.ar_prediction_tokens = 4
    model.ar_commit_tokens = 1
    model.ar_token_steps = 5
    model.ar_total_rollout_steps = 80
    model.diffusion_num_steps = 8
    model.diffusion_loss_weight = 0.25
    model.ntp_aux_loss_weight = 0.0
    model.discrete_policy_overlap_loss_weight = 0.05
    model.discrete_policy_span_tokens = 3
    model.discrete_policy_chunk_loss_weights = (1.0, 0.3, 0.15, 0.075)
    model.discrete_policy_candidate_count = 4
    model.discrete_policy_candidate_energy_weight = 2.0
    model.discrete_policy_candidate_chunk_weights = (1.0, 0.3, 0.15, 0.075)
    model.discrete_policy_candidate_score_enabled = True
    model.causal_loss_weighting_enabled = True
    model.causal_loss_weights = model.discrete_policy_chunk_loss_weights
    model.debug_validation_logging = False
    model.inference_token = False
    model.ar_local_map_radius = 2.0
    model.max_map_tokens = 0
    model.dense_smart_ce_loss_weight = 0.0
    model.dense_smart_ce_interval = 1
    model.proposal_carry_training_enabled = False
    model.proposal_carry_loss_weight = 0.0
    model.proposal_carry_interval = 1
    model.proposal_carry_detach_encoder = False
    model.ar_training_mode = "discrete_policy"
    model.cadf_lite_proposal_init_modes = ("all_mask",)
    model.cadf_lite_local_ntp_loss_weight = 0.0
    model.commitment_aware_training = False
    model.proposal_shift_consistency_loss_weight = 0.0
    model.ar_state_perturb_prob = 0.0
    model.proposal_dropout_prob = 0.0
    model.proposal_noise_topk = 0
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


class SMARTDiscreteDiffusionPolicyTest(unittest.TestCase):
    def test_select_commit_discards_tail_without_temporal_voting(self):
        model = _policy_shell()
        tokens = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
        confidence = torch.tensor([[0.9, 0.1, 0.1, 0.1], [0.8, 0.7, 0.6, 0.5]])
        valid = torch.ones(2, 4, dtype=torch.bool)
        generation = torch.ones(2, dtype=torch.bool)

        selected_tokens, selected_confidence = model._select_ar_committed_tokens(
            tokens,
            confidence,
            valid,
            generation,
            round_idx=3,
            rounds=16,
        )

        self.assertTrue(torch.equal(selected_tokens, torch.tensor([[10], [20]])))
        self.assertTrue(torch.equal(selected_confidence, torch.tensor([[0.9], [0.8]])))

    def test_candidate_window_scores_use_decayed_chunk_energy_weights(self):
        model = _policy_shell()
        base_logp = torch.zeros(2)
        energy = torch.tensor(
            [
                [0.0, 0.0, 0.0, 10.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        valid = torch.ones_like(energy, dtype=torch.bool)

        scores = model._score_discrete_policy_candidate_windows(
            base_logp,
            energy,
            valid,
        )

        self.assertGreater(float(scores[0]), float(scores[1]))
        self.assertAlmostEqual(float(scores[0]), -1.5, places=5)
        self.assertAlmostEqual(float(scores[1]), -2.0, places=5)

    def test_dense_span_training_enumerates_contiguous_anchors_without_proposals(self):
        model = _policy_shell()
        model._manual_global_step = 1
        data = _toy_sequence(num_agents=1, num_tokens=8, num_frames=41)
        captured_anchors = []
        loss_calls = []
        overlap_pairs = []
        test_case = self

        def fake_training_view(self, batch, anchor_token=None, perturb=None, allow_incomplete_window=False):
            captured_anchors.append(int(anchor_token))
            return SMARTAutoregressiveDiffusion._build_ar_training_view(
                self,
                batch,
                anchor_token=anchor_token,
                perturb=perturb,
                allow_incomplete_window=allow_incomplete_window,
            )

        def fake_build_inputs(self, batch, rollout_valid=False, return_context=False):
            del rollout_valid, return_context
            future = slice(self.ar_history_tokens, self.ar_history_tokens + self.ar_prediction_tokens)
            token_ids = batch["agent"]["token_idx"][:, future].long()
            valid = batch["agent"]["agent_valid_mask"][:, future].bool()
            packed = {
                "token_ids": token_ids,
                "valid_mask": valid,
                "loss_mask_base": valid.clone(),
                "chunk_ids": torch.arange(self.ar_prediction_tokens).unsqueeze(0),
                "agent_maps": [(0, 0, torch.tensor([0]))],
            }
            return (
                packed,
                torch.zeros(1, 1),
                token_ids,
                valid,
                torch.ones(1, dtype=torch.bool),
                torch.ones(1, dtype=torch.bool),
                torch.zeros(1, dtype=torch.long),
            )

        def fake_diffusion_loss(self, packed, summary, **kwargs):
            del summary
            loss_calls.append(kwargs)
            test_case.assertTrue(torch.equal(kwargs["forced_mask"], packed["valid_mask"]))
            test_case.assertEqual(kwargs["loss_normalization"], "supervision_weight")
            test_case.assertTrue(kwargs["return_details"])
            test_case.assertIsNone(kwargs.get("initial_proposal_token_ids"))
            test_case.assertIsNone(kwargs.get("initial_proposal_confidence"))
            logits = torch.full((1, self.ar_prediction_tokens, 8), -4.0)
            for chunk_idx in range(self.ar_prediction_tokens):
                logits[0, chunk_idx, int(packed["token_ids"][0, chunk_idx])] = 4.0
            details = {
                "logits": logits,
                "loss_mask": packed["loss_mask_base"].clone(),
            }
            return torch.tensor(2.0), torch.tensor(0.25), details

        def fake_overlap(self, source_logits, target_logits, source_mask, target_mask, source_chunk):
            del source_logits, target_logits, source_mask, target_mask
            overlap_pairs.append(int(source_chunk))
            return torch.tensor(0.4)

        model._build_ar_training_view = MethodType(fake_training_view, model)
        model._build_diffusion_inputs = MethodType(fake_build_inputs, model)
        model._compute_diffusion_loss = MethodType(fake_diffusion_loss, model)
        model._discrete_policy_overlap_loss = MethodType(fake_overlap, model)

        result = model._compute_discrete_policy_training_loss(data, torch.tensor(0.0))

        self.assertEqual(captured_anchors, [3, 4, 5])
        self.assertEqual(len(loss_calls), 3)
        self.assertEqual(overlap_pairs, [1, 2, 1])
        self.assertEqual(result["window_count"], 3)
        self.assertAlmostEqual(float(result["chunk_loss"]), 2.0, places=5)
        self.assertAlmostEqual(float(result["overlap_loss"]), 0.4, places=5)
        self.assertEqual(tuple(result["chunk_x0_losses"].shape), (4,))
        self.assertEqual(tuple(result["chunk_acc"].shape), (4,))
        self.assertTrue(torch.all(result["chunk_x0_losses"] < 0.01))
        self.assertTrue(torch.allclose(result["chunk_acc"], torch.ones(4)))
        self.assertEqual(int(result["supervised_tokens"].item()), 11)
        self.assertEqual(int(result["valid_tokens"].item()), 11)

    def test_training_step_uses_only_diffusion_chunk_and_overlap_losses(self):
        model = _policy_shell()
        torch.nn.Module.__init__(model)
        model._prepare_batch = MethodType(lambda self, batch: batch, model)
        logged = {}

        def fake_log(self, name, value, **kwargs):
            del kwargs
            logged[name] = value

        model.log = MethodType(fake_log, model)

        def fake_chunk(self, batch, ref_tensor):
            del batch, ref_tensor
            return {
                "chunk_loss": torch.tensor(2.0),
                "mask_acc": torch.tensor(0.5),
                "overlap_loss": torch.tensor(3.0),
                "chunk_x0_losses": torch.tensor([2.0, 3.0, 4.0, 5.0]),
                "chunk_acc": torch.tensor([0.1, 0.2, 0.3, 0.4]),
                "valid_tokens": torch.tensor(8.0),
                "supervised_tokens": torch.tensor(8.0),
                "window_count": 2,
            }

        def fail_ntp(*args, **kwargs):
            del args, kwargs
            raise AssertionError("SMART NTP CE must not run for pure diffusion policy")

        model._compute_discrete_policy_training_loss = MethodType(fake_chunk, model)
        model._compute_discrete_policy_smart_ntp_loss = MethodType(fail_ntp, model)

        loss = model.training_step(_toy_sequence(num_agents=1), 0)

        self.assertAlmostEqual(float(loss), 0.65, places=5)
        self.assertIn("loss_x0_chunk0", logged)
        self.assertIn("loss_x0_chunk1", logged)
        self.assertIn("loss_x0_chunk2", logged)
        self.assertIn("loss_x0_chunk3", logged)
        self.assertIn("chunk0_acc", logged)
        self.assertIn("chunk1_acc", logged)
        self.assertIn("chunk2_acc", logged)
        self.assertIn("chunk3_acc", logged)
        self.assertNotIn("smart_ntp_loss", logged)


class SMARTDiscreteDiffusionPolicyConfigTest(unittest.TestCase):
    def test_2000_step_config_selects_discrete_policy_predictor(self):
        cfg = load_config_act(
            "configs/train/train_scalable_discrete_diffusion_policy_2000.yaml"
        )

        self.assertEqual(cfg.Model.predictor, "smart_discrete_diffusion_policy")
        self.assertEqual(cfg.Dataset.train_raw_dir, ["data/valid_demo"])
        self.assertEqual(cfg.Dataset.val_raw_dir, ["data/valid_demo"])
        self.assertEqual(cfg.Trainer.max_steps, 2000)
        self.assertEqual(cfg.Trainer.val_check_interval, 2000)
        self.assertIsNone(cfg.Trainer.check_val_every_n_epoch)
        self.assertEqual(cfg.Model.hidden_dim, 64)
        self.assertEqual(cfg.Model.decoder.num_map_layers, 0)
        self.assertEqual(cfg.Model.decoder.num_agent_layers, 1)
        self.assertEqual(cfg.Model.diffusion.discrete_policy_objective, "pure_chunk_v1")
        self.assertEqual(cfg.Model.diffusion.num_layers, 1)
        self.assertEqual(cfg.Model.diffusion.ntp_aux_loss_weight, 0.0)
        self.assertFalse(cfg.Model.diffusion.use_smart_ntp_head)
        self.assertFalse(cfg.Model.diffusion.use_smart_prior_fusion)
        self.assertFalse(cfg.Model.diffusion.proposal_memory.enabled)
        self.assertFalse(cfg.Model.diffusion.temporal_ensemble.enabled)
        self.assertFalse(cfg.Model.diffusion.sampling_guidance.enabled)
        self.assertEqual(cfg.Model.diffusion.prediction_horizon, 4)
        self.assertEqual(cfg.Model.diffusion.execution_horizon, 1)
        self.assertEqual(cfg.Model.diffusion.commit_tokens, 1)
        self.assertFalse(cfg.Model.diffusion.carry_tail_proposal)
        self.assertEqual(cfg.Model.diffusion.discrete_policy_candidate_count, 1)
        self.assertFalse(cfg.Model.diffusion.discrete_policy_candidate_score_enabled)
        self.assertFalse(cfg.Model.diffusion.use_map_context)
        self.assertFalse(cfg.Model.diffusion.use_agent_context)
        self.assertEqual(
            list(cfg.Model.diffusion.chunk_loss_weights),
            [1.0, 0.3, 0.15, 0.075],
        )
        self.assertEqual(
            list(cfg.Model.diffusion.causal_loss_weights),
            [1.0, 0.3, 0.15, 0.075],
        )
        self.assertEqual(
            list(cfg.Model.diffusion.discrete_policy_candidate_chunk_weights),
            [1.0, 0.3, 0.15, 0.075],
        )


if __name__ == "__main__":
    unittest.main()
