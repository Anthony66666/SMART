import unittest
from types import SimpleNamespace

import torch
from torch_geometric.data import HeteroData

import scripts.smoke_causal_guidance_modes as smoke
from scripts.smoke_causal_guidance_modes import (
    _aggregate_results,
    _apply_guidance_overrides,
    _flatten_record,
    _pareto_modes,
    _pareto_objectives,
    _parse_indices,
    _select_target_agents,
    _trajectory_metrics,
    _zero_guidance_metrics,
)


class CausalGuidanceSmokeScriptTest(unittest.TestCase):
    def test_parse_indices_accepts_explicit_list_and_filters_dataset_bounds(self):
        self.assertEqual(
            _parse_indices(index=0, indices="2,0,9", num_scenes=1, dataset_len=3),
            [2, 0],
        )

    def test_parse_indices_uses_start_index_and_scene_count(self):
        self.assertEqual(
            _parse_indices(index=1, indices="", num_scenes=3, dataset_len=4),
            [1, 2, 3],
        )

    def test_select_target_agents_excludes_ego_av_index_by_default(self):
        batch = HeteroData()
        batch["agent"]["valid_mask"] = torch.ones(3, 4, dtype=torch.bool)
        batch["agent"]["type"] = torch.zeros(3, dtype=torch.long)
        batch["agent"]["category"] = torch.tensor([3, 3, 3])
        batch["agent"]["av_index"] = torch.tensor([0])
        model = SimpleNamespace(num_historical_steps=4)

        selected = _select_target_agents(
            batch,
            model,
            explicit_agents=[],
            max_targets=1,
        )

        self.assertEqual(selected, [1])

    def test_apply_guidance_overrides_updates_optional_runtime_fields(self):
        model = SimpleNamespace(
            guidance_target_spec="cut_in",
            guidance_ego_interaction_alpha=1.0,
            guidance_target_event_eta=1.0,
            guidance_edit_gamma=1.0,
        )
        args = SimpleNamespace(
            target_spec="lead_hard_brake",
            ego_interaction_alpha=3.0,
            target_event_eta=2.0,
            edit_gamma=0.25,
            invalid_beta=None,
            path_corridor_width=None,
            conflict_tta_threshold=None,
            near_miss_distance=None,
            ttc_threshold=None,
        )

        _apply_guidance_overrides(model, args)

        self.assertEqual(model.guidance_target_spec, "lead_hard_brake")
        self.assertEqual(model.guidance_ego_interaction_alpha, 3.0)
        self.assertEqual(model.guidance_target_event_eta, 2.0)
        self.assertEqual(model.guidance_edit_gamma, 0.25)

    def test_aggregate_results_flattens_guidance_metrics_by_mode(self):
        records = [
            {
                "mode": "safe",
                "ade": 4.0,
                "fde": 8.0,
                "token_change_rate_vs_gt": 0.5,
                "guidance_metrics": {
                    "ego_risk_success_rate": 0.0,
                    "ego_near_miss_success_rate": 0.0,
                    "hard_collision_rate": 0.1,
                    "offroad_rate": 0.2,
                    "dynamics_energy": 0.3,
                    "edit_distance": 0.0,
                    "target_event_success_rate": 0.0,
                },
            },
            {
                "mode": "safe",
                "ade": 2.0,
                "fde": 4.0,
                "token_change_rate_vs_gt": 0.25,
                "guidance_metrics": {
                    "ego_risk_success_rate": 0.2,
                    "ego_near_miss_success_rate": 0.2,
                    "hard_collision_rate": 0.0,
                    "offroad_rate": 0.0,
                    "dynamics_energy": 0.1,
                    "edit_distance": 0.0,
                    "target_event_success_rate": 0.2,
                },
            },
        ]

        summary = _aggregate_results(records)

        safe = summary["by_mode"]["safe"]
        self.assertEqual(safe["count"], 2)
        self.assertAlmostEqual(safe["ade_mean"], 3.0)
        self.assertAlmostEqual(safe["guidance_ego_risk_success_rate_mean"], 0.1)
        self.assertAlmostEqual(safe["guidance_ego_near_miss_success_rate_mean"], 0.1)
        self.assertAlmostEqual(safe["guidance_dynamics_energy_mean"], 0.2)

    def test_zero_guidance_metrics_contains_generic_ego_risk_fields(self):
        metrics = _zero_guidance_metrics()

        self.assertIn("ego_risk_reward", metrics)
        self.assertIn("ego_risk_min_ttc", metrics)
        self.assertIn("ego_risk_success_rate", metrics)
        self.assertEqual(metrics["ego_risk_success_rate"], 0.0)

    def test_flatten_record_prefixes_guidance_metrics(self):
        flat = _flatten_record({
            "mode": "ego_edit",
            "ade": 1.0,
            "guidance_metrics": {
                "edit_distance": 0.3,
            },
        })

        self.assertEqual(flat["mode"], "ego_edit")
        self.assertEqual(flat["guidance_edit_distance"], 0.3)

    def test_trajectory_metrics_report_non_target_preservation_ade(self):
        output = {
            "pred_traj": torch.tensor([
                [[10.0, 0.0], [10.0, 0.0]],
                [[2.0, 0.0], [2.0, 0.0]],
            ]),
            "gt": torch.tensor([
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ]),
            "valid_mask": torch.ones(2, 2, dtype=torch.bool),
            "pred_valid_mask": torch.ones(2, 2, dtype=torch.bool),
        }

        metrics = _trajectory_metrics(output, target_agents=[0])

        self.assertAlmostEqual(
            metrics["non_target_preservation_ADE"],
            2.0,
        )

    def test_trajectory_metrics_fde_uses_last_true_valid_frame(self):
        output = {
            "pred_traj": torch.tensor([
                [[1.0, 0.0], [2.0, 0.0], [1000.0, 0.0], [4.0, 0.0]],
            ]),
            "gt": torch.tensor([
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
            ]),
            "valid_mask": torch.tensor([[True, True, False, True]]),
            "pred_valid_mask": torch.ones(1, 4, dtype=torch.bool),
        }

        metrics = _trajectory_metrics(output)

        self.assertAlmostEqual(metrics["fde"], 3.0)

    def test_trajectory_metrics_report_moving_speed_ratio(self):
        output = {
            "pred_traj": torch.tensor([
                [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            ]),
            "gt": torch.tensor([
                [[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ]),
            "valid_mask": torch.ones(2, 3, dtype=torch.bool),
            "pred_valid_mask": torch.ones(2, 3, dtype=torch.bool),
        }

        metrics = _trajectory_metrics(output)

        self.assertEqual(metrics["moving_pair_count"], 2)
        self.assertAlmostEqual(metrics["moving_speed_ratio"], 0.5)
        self.assertAlmostEqual(metrics["moving_speed_ratio_p50"], 0.5)

    def test_token_change_rate_uses_valid_token_denominator(self):
        helper = getattr(smoke, "_token_change_rate", None)
        self.assertTrue(callable(helper))

        pred_tokens = torch.tensor([[1, 2, 9, 9]])
        gt_tokens = torch.tensor([[1, 3, 0, 0]])
        token_valid = torch.tensor([[True, True, False, False]])

        self.assertAlmostEqual(
            helper(pred_tokens, gt_tokens, token_valid),
            0.5,
        )

    def test_non_target_preservation_uses_official_validity_not_eval_category_mask(self):
        output = {
            "pred_traj": torch.tensor([
                [[10.0, 0.0]],
                [[3.0, 0.0]],
            ]),
            "gt": torch.tensor([
                [[0.0, 0.0]],
                [[0.0, 0.0]],
            ]),
            "valid_mask": torch.tensor([[True], [False]]),
            "official_valid_mask": torch.tensor([[True], [True]]),
            "pred_valid_mask": torch.tensor([[True], [True]]),
        }

        metrics = _trajectory_metrics(output, target_agents=[0])

        self.assertAlmostEqual(
            metrics["non_target_preservation_ADE"],
            3.0,
        )

    def test_pareto_modes_uses_realism_criticality_minimality(self):
        summary = {
            "by_mode": {
                "safe": {
                    "ade_mean": 2.0,
                    "non_target_preservation_ADE_mean": 0.2,
                    "token_change_rate_vs_gt_mean": 0.2,
                    "guidance_ego_risk_success_rate_mean": 0.0,
                    "guidance_ego_near_miss_success_rate_mean": 0.0,
                    "guidance_target_event_success_rate_mean": 0.0,
                    "guidance_hard_collision_rate_mean": 0.0,
                    "guidance_offroad_rate_mean": 0.0,
                    "guidance_dynamics_energy_mean": 0.1,
                    "guidance_edit_distance_mean": 0.0,
                },
                "ego_stress": {
                    "ade_mean": 2.5,
                    "non_target_preservation_ADE_mean": 0.4,
                    "token_change_rate_vs_gt_mean": 0.4,
                    "guidance_ego_risk_success_rate_mean": 0.3,
                    "guidance_ego_near_miss_success_rate_mean": 0.0,
                    "guidance_target_event_success_rate_mean": 0.0,
                    "guidance_hard_collision_rate_mean": 0.0,
                    "guidance_offroad_rate_mean": 0.0,
                    "guidance_dynamics_energy_mean": 0.1,
                    "guidance_edit_distance_mean": 0.0,
                },
                "seed": {
                    "ade_mean": 0.0,
                    "non_target_preservation_ADE_mean": 0.0,
                    "token_change_rate_vs_gt_mean": 0.0,
                    "guidance_ego_risk_success_rate_mean": 0.0,
                    "guidance_ego_near_miss_success_rate_mean": 0.0,
                    "guidance_target_event_success_rate_mean": 0.0,
                    "guidance_hard_collision_rate_mean": 0.0,
                    "guidance_offroad_rate_mean": 0.0,
                    "guidance_dynamics_energy_mean": 0.0,
                    "guidance_edit_distance_mean": 0.0,
                },
                "bad": {
                    "ade_mean": 9.0,
                    "non_target_preservation_ADE_mean": 0.9,
                    "token_change_rate_vs_gt_mean": 0.9,
                    "guidance_ego_risk_success_rate_mean": 0.0,
                    "guidance_ego_near_miss_success_rate_mean": 0.0,
                    "guidance_target_event_success_rate_mean": 0.0,
                    "guidance_hard_collision_rate_mean": 0.5,
                    "guidance_offroad_rate_mean": 0.5,
                    "guidance_dynamics_energy_mean": 5.0,
                    "guidance_edit_distance_mean": 1.0,
                },
            }
        }

        pareto = _pareto_modes(summary)
        modes = {entry["mode"] for entry in pareto}

        self.assertNotIn("seed", modes)
        self.assertIn("safe", modes)
        self.assertIn("ego_stress", modes)
        self.assertNotIn("bad", modes)
        safe = next(entry for entry in pareto if entry["mode"] == "safe")
        self.assertAlmostEqual(safe["objectives"]["minimality"], 0.4)

        stress_objectives = _pareto_objectives(summary["by_mode"]["ego_stress"])
        self.assertAlmostEqual(stress_objectives["criticality"], 0.3)


if __name__ == "__main__":
    unittest.main()
