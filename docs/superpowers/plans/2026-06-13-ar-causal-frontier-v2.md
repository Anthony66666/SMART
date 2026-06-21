# AR Causal Frontier v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a config-gated causal-frontier objective to the original AR diffusion path, then run matched short-budget training and validation comparisons.

**Architecture:** Preserve `SMARTAutoregressiveDiffusion` as the public predictor and keep the old MaskGIT-style objective as the default. Add a new `diffusion.ar_objective: causal_frontier_v1` mode that can use causal temporal edges, frontier-only supervision, closed-loop rollout training views, retokenized recovery targets, and current-motion context.

**Tech Stack:** PyTorch, PyTorch Lightning, Torch Geometric, existing SMART token vocabulary, existing validation visualization callback, unittest, repo-local YAML configs.

---

### Task 1: Training Step Budget

**Files:**
- Modify: `train.py`
- Test: `tests/test_train_entrypoint_config.py`

- [ ] **Step 1: Write the failing test**

Add a test that constructs a dummy trainer config with `max_steps = 1000` and verifies a helper returns `1000`.

```python
import unittest
from types import SimpleNamespace

import train


class TrainEntrypointConfigTest(unittest.TestCase):
    def test_resolves_configured_max_steps(self):
        cfg = SimpleNamespace(max_steps=1000)
        self.assertEqual(train.resolve_max_steps(cfg), 1000)

    def test_missing_max_steps_uses_lightning_default(self):
        cfg = SimpleNamespace()
        self.assertEqual(train.resolve_max_steps(cfg), -1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_train_entrypoint_config -v`

Expected: FAIL with `AttributeError: module 'train' has no attribute 'resolve_max_steps'`.

- [ ] **Step 3: Implement minimal code**

Add `resolve_max_steps(trainer_config)` in `train.py` and pass
`max_steps=resolve_max_steps(trainer_config)` to `pl.Trainer`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest tests.test_train_entrypoint_config -v`

Expected: PASS.

### Task 2: AR Causal Decoder Selection

**Files:**
- Modify: `smart/model/smart_ar_diffusion.py`
- Test: `tests/test_smart_ar_diffusion.py`

- [ ] **Step 1: Write the failing test**

Add a shell-style unit test that builds an AR model object, calls a new
`_use_causal_temporal_decoder()` helper, and verifies it returns true only when
the config flag is true.

```python
def test_causal_temporal_decoder_flag_is_config_driven(self):
    model = _ar_shell()
    model.ar_causal_temporal_edges = False
    self.assertFalse(model._use_causal_temporal_decoder())
    model.ar_causal_temporal_edges = True
    self.assertTrue(model._use_causal_temporal_decoder())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_smart_ar_diffusion.SMARTAutoregressiveDiffusionTest.test_causal_temporal_decoder_flag_is_config_driven -v`

Expected: FAIL because the helper does not exist.

- [ ] **Step 3: Implement minimal code**

Parse `diffusion.causal_temporal_edges`, add `_use_causal_temporal_decoder()`,
and when true replace the inherited `DiffusionDecoder` with
`CausalDiffusionDecoder` using the same constructor arguments.

- [ ] **Step 4: Run focused test**

Run the same unittest target.

Expected: PASS.

### Task 3: AR Frontier Loss

**Files:**
- Modify: `smart/model/smart_ar_diffusion.py`
- Test: `tests/test_smart_ar_diffusion.py`

- [ ] **Step 1: Write the failing test**

Add a unit test using a shell model with `ar_objective = "causal_frontier_v1"`.
Patch `_sample_frontier_ids()` to select chunk 1, feed packed chunk ids
`[0, 1, 2, 3]`, and assert `_compute_diffusion_loss()` masks chunks 1-3 while
supervising only chunk 1.

- [ ] **Step 2: Run test to verify it fails**

Run the focused unittest target.

Expected: FAIL because AR still delegates to the base MaskGIT loss.

- [ ] **Step 3: Implement minimal code**

Add `_sample_frontier_ids()` and `_compute_frontier_diffusion_loss()` to AR.
Have `_compute_diffusion_loss()` dispatch to the frontier objective only when
`self.ar_objective == "causal_frontier_v1"`.

- [ ] **Step 4: Run focused and AR tests**

Run: `python -m unittest tests.test_smart_ar_diffusion -v`

Expected: Existing unrelated config drift assertions may fail until Task 6
updates configs; the new frontier test must pass.

### Task 4: Closed-Loop AR Training Views

**Files:**
- Modify: `smart/model/smart_ar_diffusion.py`
- Test: `tests/test_smart_ar_diffusion.py`

- [ ] **Step 1: Write the failing test**

Add a test that sets `ar_closed_loop_batch_ratio_max = 0.5` and asserts
`_ar_closed_loop_curriculum(0)` returns `(0.0, 0.5)` and epoch 4 returns
`(0.25, 0.5)`.

- [ ] **Step 2: Run test to verify it fails**

Run the focused unittest target.

Expected: FAIL because the AR curriculum helper does not exist.

- [ ] **Step 3: Implement minimal code**

Move the causal closed-loop helper logic into AR-safe methods:
`_ar_closed_loop_curriculum()`, `_select_closed_loop_anchor()`,
`_build_model_rollout_training_view()`, `_retokenize_training_view()`, and
`_continuous_recovery_loss()`. Keep causal branch compatibility by letting
`SMARTCausalDiffusion` use or override those methods where needed.

- [ ] **Step 4: Run focused tests**

Run: `python -m unittest tests.test_smart_ar_diffusion tests.test_smart_causal_diffusion -v`

Expected: PASS except documented pre-existing config assertions until configs
are updated.

### Task 5: Matched Configs and Comparison Script

**Files:**
- Create: `configs/train/train_scalable_ar_diffusion_frontier_local.yaml`
- Create: `configs/train/train_scalable_ar_diffusion_baseline_1000.yaml`
- Create: `configs/train/train_scalable_causal_diffusion_1000.yaml`
- Create: `configs/train/train_scalable_causal_flow_matching_1000.yaml`
- Create: `scripts/compare_motion_models.py`
- Test: `tests/test_compare_motion_models.py`

- [ ] **Step 1: Write config tests**

Add tests that read each YAML and verify `max_steps: 1000`, the requested
training/validation paths, and the expected predictor/objective pair.

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_compare_motion_models -v`

Expected: FAIL because files do not exist.

- [ ] **Step 3: Add configs and script**

Add four local 1000-step configs. Add a script that accepts repeated
`--model name=config=checkpoint` specs, evaluates fixed validation indices,
writes `records.csv`, `summary.csv`, `manifest.json`, and saves per-model PNGs
through `save_validation_visualization()`.

- [ ] **Step 4: Run config/script tests**

Run: `python -m unittest tests.test_compare_motion_models -v`

Expected: PASS.

### Task 6: Verification and Experiment Launch

**Files:**
- Modify: `docs/progress.md`
- Modify: `docs/next.md`

- [ ] **Step 1: Static verification**

Run:

```bash
python -m py_compile train.py smart/model/smart_ar_diffusion.py scripts/compare_motion_models.py tests/test_train_entrypoint_config.py tests/test_smart_ar_diffusion.py tests/test_compare_motion_models.py
python -m unittest tests.test_train_entrypoint_config tests.test_smart_ar_diffusion tests.test_compare_motion_models -v
```

- [ ] **Step 2: Train with GPU access**

Run each config with `--save_ckpt_path checkpoints/<run_name>` in the `smart`
conda environment. Each run must use the same dataset paths and `max_steps:
1000`.

- [ ] **Step 3: Compare validation scenes**

Run `scripts/compare_motion_models.py` with the resulting checkpoints and fixed
validation indices `[0, 1, 2, 3, 4, 5, 6, 7]`.

- [ ] **Step 4: Record results**

Append the final summary table location and visualization directories to
`docs/progress.md`, refresh `docs/next.md`, and report the ADE/FDE table to the
user.
