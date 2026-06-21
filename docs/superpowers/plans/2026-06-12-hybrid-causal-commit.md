# Hybrid Causal Commit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stabilize causal diffusion rollout by making the executable commit token prefer SMART-style progress and speed consistency while preserving diffusion tail proposals for scene editing.

**Architecture:** Keep the existing receding-horizon loop: sample four future tokens, commit only the first token, and carry the tail as a revisable proposal. Add a commit-only progress/speed energy term to guidance reranking so safe/ego modes stop preferring near-static tokens when the observed vehicle is moving.

**Tech Stack:** PyTorch, existing SMART trajectory token codebook, `SMARTCausalDiffusion`, `SMARTAutoregressiveDiffusion`, existing unittest suite and `data/valid_demo` smoke scripts.

---

### Task 1: Commit Speed Energy

**Files:**
- Modify: `smart/model/smart_causal_diffusion.py`
- Test: `tests/test_smart_causal_diffusion.py`

- [ ] Add a focused unit test that constructs candidate 5-frame trajectories for a moving vehicle and verifies a commit-speed energy helper penalizes near-static candidates more than progress-matched candidates.
- [ ] Run the focused test and confirm it fails because the helper does not exist yet.
- [ ] Implement a small helper on `SMARTCausalDiffusion` that compares candidate displacement speed against current observed speed only for chunk-0 commit candidates.
- [ ] Include the helper in `_guided_frontier_tokens()` through `invalid_energy` / safe energy terms, guarded by config weights with conservative defaults.
- [ ] Run the focused test and causal unit tests.

### Task 2: Demo Validation

**Files:**
- Modify: `docs/progress.md`
- Modify: `docs/next.md`

- [ ] Compile changed Python files.
- [ ] Run causal guidance smoke over `/home/anthony/SimAgentJEPA/external/SMART/data/valid_demo` with checkpoint `/mnt/d/causal_v2_epoch=02.ckpt`.
- [ ] Compare `moving_speed_ratio_p50`, `pred_speed`, `offroad_rate`, and FDE for `none` and `safe`.
- [ ] Record concise evidence and next actions in repo docs.
