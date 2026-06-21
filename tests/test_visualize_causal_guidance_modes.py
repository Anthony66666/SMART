import json
import tempfile
import unittest
from pathlib import Path

import torch

from scripts.visualize_causal_guidance_modes import (
    _agent_future,
    _clean_scenario_id,
    _closest_time_aligned_pair,
    _cpu_tensor_tree,
    _ego_agent_index,
    _first_corridor_entry,
    _mode_list,
    _panel_title,
    _uses_corridor_entry_marker,
    _target_window,
    _trajectory_extent,
    _write_manifest,
)


class VisualizeCausalGuidanceModesTest(unittest.TestCase):
    def test_mode_list_deduplicates_and_preserves_order(self):
        self.assertEqual(
            _mode_list("seed,ego_edit,ego_stress,ego_edit, none "),
            ["seed", "ego_edit", "ego_stress", "none"],
        )

    def test_target_window_parses_start_end_pair(self):
        self.assertEqual(_target_window("2,4"), (2, 4))
        self.assertIsNone(_target_window(""))
        with self.assertRaises(ValueError):
            _target_window("1,2,3")

    def test_clean_scenario_id_handles_batched_list_strings(self):
        self.assertEqual(_clean_scenario_id("['abc/def']"), "abc_def")
        self.assertEqual(_clean_scenario_id("plain/id"), "plain_id")

    def test_cpu_tensor_tree_detaches_nested_tensors(self):
        value = {
            "output": {
                "pred": torch.tensor([1.0], requires_grad=True),
            },
            "items": [torch.tensor([2])],
        }

        cpu = _cpu_tensor_tree(value)

        self.assertFalse(cpu["output"]["pred"].requires_grad)
        self.assertEqual(cpu["output"]["pred"].device.type, "cpu")
        self.assertEqual(cpu["items"][0].tolist(), [2])

    def test_trajectory_extent_uses_history_gt_prediction_and_padding(self):
        history = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        gt = torch.tensor([[3.0, 0.0], [4.0, 0.0]])
        predictions = {
            "none": torch.tensor([[2.0, -2.0], [5.0, -3.0]]),
            "ego_edit": torch.tensor([[2.0, 3.0]]),
        }

        extent = _trajectory_extent(history, gt, predictions, min_window_m=1.0)

        self.assertLessEqual(extent[0], -0.5)
        self.assertGreaterEqual(extent[1], 5.5)
        self.assertLessEqual(extent[2], -3.5)
        self.assertGreaterEqual(extent[3], 3.5)

    def test_trajectory_extent_can_include_ego_context_paths(self):
        history = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
        gt = torch.tensor([[2.0, 0.0]])
        predictions = {"ego_edit": torch.tensor([[3.0, 0.0]])}
        ego_path = torch.tensor([[30.0, -12.0], [31.0, -13.0]])

        extent = _trajectory_extent(
            history,
            gt,
            predictions,
            min_window_m=1.0,
            extra_paths=[ego_path],
        )

        self.assertLessEqual(extent[2], -13.5)
        self.assertGreaterEqual(extent[1], 31.5)

    def test_ego_agent_index_reads_batched_av_index(self):
        data = {
            "agent": {
                "position": torch.zeros(4, 3, 2),
                "av_index": torch.tensor([2]),
            }
        }

        self.assertEqual(_ego_agent_index(data), 2)

    def test_closest_time_aligned_pair_returns_nearest_same_step_points(self):
        target_path = torch.tensor([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        ego_path = torch.tensor([[10.0, 0.0], [2.5, 0.0], [9.0, 0.0]])

        pair = _closest_time_aligned_pair(target_path, ego_path)

        self.assertIsNotNone(pair)
        target_xy, ego_xy, distance = pair
        self.assertEqual(target_xy.tolist(), [2.0, 0.0])
        self.assertEqual(ego_xy.tolist(), [2.5, 0.0])
        self.assertAlmostEqual(distance, 0.5)

    def test_first_corridor_entry_detects_target_entering_ego_path(self):
        target_path = torch.tensor([[4.0, 0.0], [2.5, 0.0], [0.8, 0.0]])
        ego_path = torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 4.0]])

        entry = _first_corridor_entry(target_path, ego_path, corridor_width=1.0)

        self.assertIsNotNone(entry)
        start_xy, entry_xy, entry_index, distance = entry
        self.assertEqual(start_xy.tolist(), [4.0, 0.0])
        self.assertAlmostEqual(float(entry_xy[0]), 0.8)
        self.assertAlmostEqual(float(entry_xy[1]), 0.0)
        self.assertEqual(entry_index, 2)
        self.assertAlmostEqual(distance, 0.8)

    def test_first_corridor_entry_ignores_paths_already_inside_corridor(self):
        target_path = torch.tensor([[0.5, 0.0], [0.8, 0.0]])
        ego_path = torch.tensor([[0.0, 0.0], [0.0, 2.0]])

        self.assertIsNone(_first_corridor_entry(target_path, ego_path, corridor_width=1.0))

    def test_corridor_entry_marker_is_only_for_legacy_cut_in_spec(self):
        self.assertTrue(_uses_corridor_entry_marker("cut_in"))
        self.assertTrue(_uses_corridor_entry_marker("lane-cut-in"))
        self.assertFalse(_uses_corridor_entry_marker("ego_risk"))
        self.assertFalse(_uses_corridor_entry_marker(""))

    def test_panel_title_includes_core_metrics_without_overflow_noise(self):
        title = _panel_title(
            "ego_stress",
            {
                "ade": 1.234,
                "fde": 5.678,
                "guidance_metrics": {
                    "ego_min_distance": 1.25,
                    "ego_min_ttc": 20.0,
                    "ego_risk_min_ttc": 2.5,
                    "hard_collision_rate": 0.0,
                    "ego_risk_success_rate": 0.1,
                    "edit_distance": 0.0,
                },
            },
        )

        self.assertIn("ego_stress", title)
        self.assertIn("ADE 1.23", title)
        self.assertIn("ego_d 1.25", title)
        self.assertIn("risk_ttc 2.50", title)
        self.assertIn("risk 0.10", title)

    def test_agent_future_masks_selected_agent_without_double_indexing(self):
        output = {
            "pred_traj": torch.tensor([
                [[0.0, 0.0], [1.0, 1.0]],
                [[2.0, 2.0], [3.0, 3.0]],
            ]),
            "pred_valid_mask": torch.tensor([
                [True, True],
                [False, True],
            ]),
        }

        future = _agent_future(output, 1, "pred_traj")

        self.assertEqual(future.tolist(), [[3.0, 3.0]])

    def test_write_manifest_serializes_paths_and_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            _write_manifest(
                path,
                config="config.yaml",
                ckpt="model.ckpt",
                raw_dir="data/valid_demo",
                indices=[2],
                seed=7,
                load_info={"missing": 0, "unexpected": 0},
                figures=[Path(tmpdir) / "scene.png"],
                records=[{"mode": "seed", "ade": 0.0}],
            )

            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["config"], "config.yaml")
        self.assertEqual(payload["indices"], [2])
        self.assertEqual(payload["figures"], [str(Path(tmpdir) / "scene.png")])
        self.assertEqual(payload["records"][0]["mode"], "seed")


if __name__ == "__main__":
    unittest.main()
