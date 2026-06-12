# Causal Diffusion Config Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a complete Chinese reference for every field in `configs/train/train_scalable_causal_diffusion.yaml`.

**Architecture:** Trace each YAML field through `train.py`, the data module, visualization callbacks, and the `SMARTCausalDiffusion` inheritance chain. Document the current value, runtime behavior, tuning effect, constraints, and whether the field is active, inherited, forced, or unused.

**Tech Stack:** YAML, PyTorch Lightning 2.0.3, PyTorch Geometric, SMART causal discrete diffusion.

---

### Task 1: Build the parameter inventory

**Files:**
- Read: `configs/train/train_scalable_causal_diffusion.yaml`
- Create: `docs/train_scalable_causal_diffusion_config.md`

- [x] Enumerate every expanded leaf field, including values introduced by the `time_info` YAML anchor.
- [x] Group fields into YAML anchors, Dataset, Trainer, Visualization, Model, decoder, and diffusion sections.

### Task 2: Trace runtime behavior

**Files:**
- Read: `train.py`
- Read: `smart/datamodules/scalable_datamodule.py`
- Read: `smart/model/smart.py`
- Read: `smart/model/smart_diffusion.py`
- Read: `smart/model/smart_ar_diffusion.py`
- Read: `smart/model/smart_causal_diffusion.py`
- Read: `smart/modules/causal_diffusion_decoder.py`
- Read: `smart/modules/trajectory_energy.py`
- Read: `smart/callbacks/validation_visualization.py`
- Read: `smart/callbacks/step_visualization.py`

- [x] Mark each field as directly active, inherited active, forced/overridden, compatibility-only, or currently unused.
- [x] Record units, valid ranges, interactions, and failure modes visible in the implementation.

### Task 3: Write and verify the reference

**Files:**
- Create: `docs/train_scalable_causal_diffusion_config.md`
- Modify: `docs/progress.md`
- Modify: `docs/next.md`

- [x] Explain every field with its current value, meaning, tuning effect, and recommendation.
- [x] Include formulas for rollout rounds, effective batch size, scheduler behavior, and safety reranking.
- [x] Programmatically compare YAML leaf keys against the Markdown reference.
- [x] Run `git diff --check` and review the final diff.
