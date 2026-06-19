import subprocess
import sys
import unittest
from pathlib import Path

from smart.utils.config import load_config_act

from scripts.compare_motion_models import parse_model_spec


TRAIN_DIR = "/home/anthony/SimAgentJEPA/data/waymo/training_subset_10pct"
VAL_DIR = "/home/anthony/SimAgentJEPA/data/waymo/validation"


class CompareMotionModelsTest(unittest.TestCase):
    def test_parse_model_spec(self):
        spec = parse_model_spec("ar=configs/train/a.yaml=checkpoints/ar/epoch=00.ckpt")

        self.assertEqual(spec.name, "ar")
        self.assertEqual(spec.config_path, "configs/train/a.yaml")
        self.assertEqual(spec.ckpt_path, "checkpoints/ar/epoch=00.ckpt")

    def test_script_help_runs_from_file_path(self):
        result = subprocess.run(
            [sys.executable, "scripts/compare_motion_models.py", "--help"],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("--model", result.stdout)

    def test_1000_step_configs_use_identical_data_and_expected_predictors(self):
        expected = {
            "configs/train/train_scalable_ar_diffusion_baseline_1000.yaml": (
                "smart_ar_diffusion",
                "maskgit",
            ),
            "configs/train/train_scalable_ar_diffusion_frontier_local.yaml": (
                "smart_ar_diffusion",
                "causal_frontier_v1",
            ),
            "configs/train/train_scalable_causal_diffusion_1000.yaml": (
                "smart_causal_diffusion",
                "discrete_frontier_v2",
            ),
            "configs/train/train_scalable_causal_flow_matching_1000.yaml": (
                "smart_causal_flow_matching",
                "flow_matching_v1",
            ),
            "configs/train/train_scalable_elf_1000.yaml": (
                "smart_elf",
                "embedded_language_flow_v1",
            ),
            "configs/train/train_scalable_hybrid_diffusion_1000.yaml": (
                "smart_hybrid_diffusion",
                "closed_loop_frontier_v1",
            ),
            "configs/train/train_scalable_ar_action_chunk_1000.yaml": (
                "smart_action_chunk_diffusion",
                "maskgit",
            ),
            "configs/train/train_scalable_continuous_action_diffusion_1000.yaml": (
                "smart_continuous_action_diffusion",
                "diffusion_policy_v1",
            ),
            "configs/train/train_scalable_discrete_diffusion_policy_2000.yaml": (
                "smart_discrete_diffusion_policy",
                "chunk_rerank_v1",
            ),
        }
        for path, (predictor, objective) in expected.items():
            with self.subTest(path=path):
                self.assertTrue(Path(path).exists())
                cfg = load_config_act(path)
                self.assertEqual(cfg.Dataset.train_raw_dir, [TRAIN_DIR])
                self.assertEqual(cfg.Dataset.val_raw_dir, [VAL_DIR])
                expected_steps = 2000 if predictor == "smart_discrete_diffusion_policy" else 1000
                self.assertEqual(cfg.Trainer.max_steps, expected_steps)
                self.assertEqual(cfg.Trainer.val_check_interval, expected_steps)
                self.assertEqual(cfg.Trainer.checkpoint_every_n_train_steps, expected_steps)
                self.assertTrue(cfg.Trainer.save_last_checkpoint)
                self.assertEqual(cfg.Trainer.devices, 1)
                self.assertEqual(cfg.Trainer.strategy, "auto")
                self.assertEqual(cfg.Model.predictor, predictor)
                if predictor in ("smart_ar_diffusion", "smart_action_chunk_diffusion"):
                    self.assertEqual(cfg.Model.diffusion.ar_objective, objective)
                elif predictor == "smart_continuous_action_diffusion":
                    self.assertEqual(cfg.Model.diffusion.continuous_action_objective, objective)
                elif predictor == "smart_discrete_diffusion_policy":
                    self.assertEqual(cfg.Model.diffusion.discrete_policy_objective, objective)
                elif predictor == "smart_elf":
                    self.assertEqual(cfg.Model.diffusion.elf_objective, objective)
                elif predictor == "smart_hybrid_diffusion":
                    self.assertEqual(cfg.Model.diffusion.hybrid_objective, objective)
                else:
                    self.assertEqual(cfg.Model.diffusion.causal_objective, objective)


if __name__ == "__main__":
    unittest.main()
