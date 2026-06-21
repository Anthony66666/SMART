# Ego-Risk Guidance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace predefined cut-in / hard-brake event guidance defaults with a generic ego-risk objective that edits real scenes to reduce ego TTC while preserving realism and minimality.

**Architecture:** Keep the existing causal diffusion inference interface and ego/non-target seed locking. Add generic ego-risk reward/success metrics inside `TrajectoryEnergy.ego_interaction_metrics`, consume those metrics in `SMARTCausalDiffusion._guided_frontier_tokens`, and make smoke/visualization summaries report ego-risk rather than target-event success as the default criticality objective.

**Tech Stack:** Python, PyTorch tensors, unittest, existing SMART causal diffusion scripts.

---

### Task 1: Add Generic Ego-Risk Metrics

**Files:**
- Modify: `smart/modules/trajectory_energy.py`
- Test: `tests/test_trajectory_energy.py`

- [ ] **Step 1: Write failing tests**
  Add tests that call `TrajectoryEnergy.ego_interaction_metrics(..., target_spec="ego_risk")` and assert a closing low-TTC candidate has positive `ego_risk_reward` and `ego_risk_success=True`, while a distant non-closing candidate does not.

- [ ] **Step 2: Verify red**
  Run: `python -m unittest tests.test_trajectory_energy`
  Expected: import/key assertion failure because `ego_risk_reward` and `ego_risk_success` do not exist yet.

- [ ] **Step 3: Implement generic risk**
  Add `ego_risk_reward` and `ego_risk_success` to `ego_interaction_metrics`. The reward should combine low-TTC reward, close-distance reward, path-intrusion reward, conflict-time reward, and required-deceleration reward, and mask out hard collisions. For `target_spec in ("ego_risk", "risk", "low_ttc")`, set `target_event_reward = ego_risk_reward` and `target_event_success = ego_risk_success`.

- [ ] **Step 4: Verify green**
  Run: `python -m unittest tests.test_trajectory_energy`
  Expected: all tests pass.

### Task 2: Make Causal Guidance Use Ego-Risk Defaults

**Files:**
- Modify: `smart/model/smart_causal_diffusion.py`
- Modify: `configs/train/train_scalable_causal_diffusion.yaml`
- Modify: `configs/train/train_scalable_causal_diffusion_local.yaml`
- Modify: `configs/validation/validation_scalable_causal_diffusion.yaml`
- Test: `tests/test_smart_causal_diffusion.py`

- [ ] **Step 1: Write failing tests**
  Add tests that instantiate causal guidance config defaults and assert `guidance_target_spec == "ego_risk"`, and that diagnostics include `ego_risk_success_rate`.

- [ ] **Step 2: Verify red**
  Run: `python -m unittest tests.test_smart_causal_diffusion`
  Expected: failure because defaults still use `cut_in` and diagnostics lack ego-risk aliases.

- [ ] **Step 3: Implement default and diagnostics**
  Change the Python fallback/default from `cut_in` to `ego_risk`. Add diagnostics keys `ego_risk_reward` and `ego_risk_success_rate`, preserving old `target_event_*` aliases for compatibility.

- [ ] **Step 4: Verify green**
  Run: `python -m unittest tests.test_smart_causal_diffusion`
  Expected: all focused causal tests pass.

### Task 3: Update Smoke and Visualization Semantics

**Files:**
- Modify: `scripts/smoke_causal_guidance_modes.py`
- Modify: `scripts/visualize_causal_guidance_modes.py`
- Test: `tests/test_causal_guidance_smoke.py`
- Test: `tests/test_visualize_causal_guidance_modes.py`

- [ ] **Step 1: Write failing tests**
  Add assertions that default CLI/config target spec is `ego_risk`, summaries use `guidance_ego_risk_success_rate`, and Pareto criticality uses ego-risk success rather than predefined event success.

- [ ] **Step 2: Verify red**
  Run: `python -m unittest tests.test_causal_guidance_smoke tests.test_visualize_causal_guidance_modes`
  Expected: failure where defaults or summary fields still reference `cut_in`.

- [ ] **Step 3: Implement script changes**
  Change default `--target-spec` behavior to ego-risk, add zero/default metrics for `ego_risk_reward` and `ego_risk_success_rate`, and update Pareto criticality to prefer ego-risk success plus near-miss/TTC pressure.

- [ ] **Step 4: Verify green**
  Run: `python -m unittest tests.test_causal_guidance_smoke tests.test_visualize_causal_guidance_modes`
  Expected: all script tests pass.

### Task 4: Regenerate One Ego-Risk Controlled Scene

**Files:**
- Output: `outputs/causal_ego_risk_guidance/`
- Modify: `docs/progress.md`
- Modify: `docs/decisions.md`
- Modify: `docs/next.md`

- [ ] **Step 1: Run focused smoke/visualization**
  Run a one-scene checkpoint visualization with `--target-spec ego_risk`, `--ego-interaction-alpha`, `--edit-gamma`, and `--near-miss-distance` set explicitly.

- [ ] **Step 2: Verify generated artifacts**
  Use PIL to verify the PNG and inspect `summary.json` / `records.csv` for lower ego TTC and nonzero ego-risk success if the checkpoint produces one.

- [ ] **Step 3: Update docs**
  Record that the research direction is now generic ego-risk counterfactual editing, not predefined event generation.

- [ ] **Step 4: Final verification**
  Run: `python -m unittest tests.test_trajectory_energy tests.test_smart_causal_diffusion tests.test_causal_guidance_smoke tests.test_visualize_causal_guidance_modes`
  Run: `python -m py_compile smart/modules/trajectory_energy.py smart/model/smart_causal_diffusion.py scripts/smoke_causal_guidance_modes.py scripts/visualize_causal_guidance_modes.py`
  Run: `git diff --check`
