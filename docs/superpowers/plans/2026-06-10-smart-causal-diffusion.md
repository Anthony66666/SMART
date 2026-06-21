# SMART Causal Diffusion Implementation Plan

> Implement this plan in the existing working tree without altering the current
> SMART, full-horizon diffusion, or AR diffusion defaults.

**Goal:** Add a from-scratch causal absorbing discrete diffusion predictor with
closed-loop retokenization and top-k safety-energy reranking.

**Architecture:** Reuse `SMARTAutoregressiveDiffusion` for batch preparation,
encoding, token decoding, and one-token receding-horizon rollout. Override the
temporal graph, corruption/loss, sampling, and training-view construction in a
new predictor. Keep safety scoring in a standalone module with tensor-only
interfaces so it can be unit tested independently.

**Tech Stack:** PyTorch, PyTorch Geometric, PyTorch Lightning, OmegaConf, SMART
trajectory-token codebooks.

---

### Task 1: Strictly Causal Token Graph

**Files:**
- Create: `smart/modules/causal_diffusion_decoder.py`
- Create: `tests/test_smart_causal_diffusion.py`

1. Write a failing test that constructs three chunks for one agent and asserts
   every temporal edge has `source_chunk < target_chunk`.
2. Run the focused test and confirm it fails because the module is absent.
3. Subclass `DiffusionDecoder` and override only temporal edge construction with
   `causal=True`.
4. Run the focused test.

### Task 2: Absorbing Prefix Corruption and Frontier Loss

**Files:**
- Create: `smart/model/smart_causal_diffusion.py`
- Modify: `tests/test_smart_causal_diffusion.py`

1. Add failing tests for prefix-closed masks, first-masked frontier selection,
   and monotonic reveal schedules.
2. Implement model construction, four-token/one-token invariants, sequential
   prefix corruption, frontier masks, and monotonic reveal counts.
3. Override diffusion loss to supervise masked suffix tokens plus a weighted
   executable-frontier term.
4. Run focused helper and loss-shape tests.

### Task 3: Closed-Loop Curriculum and Retokenization

**Files:**
- Modify: `smart/model/smart_causal_diffusion.py`
- Modify: `tests/test_smart_causal_diffusion.py`

1. Add failing tests for epoch curriculum probabilities, rollout-depth bounds,
   local-frame retokenization, and per-type invalid thresholds.
2. Implement deterministic curriculum lookup and sampled training-state mode.
3. Implement codebook-center retokenization from an arbitrary predicted anchor,
   returning nearest ids, match errors, and threshold-valid masks.
4. Build perturbed and no-grad model-rollout training views; retokenize the
   continuation before computing diffusion loss.
5. Route invalid discrete targets to continuous endpoint/heading recovery loss.
6. Run focused tests.

### Task 4: Safety Energy and Top-k Reranking

**Files:**
- Create: `smart/modules/trajectory_energy.py`
- Create: `tests/test_trajectory_energy.py`
- Modify: `smart/model/smart_causal_diffusion.py`

1. Add failing tests showing lane-aligned candidates beat off-lane candidates,
   smooth candidates beat high-acceleration candidates, and separated agents
   beat colliding candidates.
2. Implement lane distance, lane heading, dynamics, and collision energies with
   explicit validity masks.
3. Add `(1-t)^2` energy scheduling and top-k log-probability reranking at each
   executable frontier.
4. Run energy and sampler tests.

### Task 5: Predictor Registration and Configs

**Files:**
- Modify: `smart/model/__init__.py`
- Modify: `train.py`
- Modify: `val.py`
- Modify: `eval_waymo_official.py`
- Create: `configs/train/train_scalable_causal_diffusion.yaml`
- Create: `configs/train/train_scalable_causal_diffusion_local.yaml`
- Create: `configs/validation/validation_scalable_causal_diffusion.yaml`
- Modify: `tests/test_smart_causal_diffusion.py`

1. Add a failing registry/config test.
2. Register `smart_causal_diffusion` without changing existing predictor names.
3. Add from-scratch configs with four-token windows, one-token commits, 32
   epochs, encoder LR scale 0.5, curriculum, retokenization, and safety energy.
4. Load all three configs and instantiate the predictor.

### Task 6: Rollout Metrics

**Files:**
- Modify: `smart/model/smart_causal_diffusion.py`
- Modify: `tests/test_smart_causal_diffusion.py`

1. Add failing tests for horizon slicing and weighted rollout score.
2. Log 2/4/6/8-second ADE/FDE, final-4-second ADE, energy terms, coverage, and
   invalid-retokenization rate while preserving official SMART metrics.
3. Run focused tests.

### Task 7: Verification and Project Memory

**Files:**
- Modify: `docs/spec.md`
- Modify: `docs/decisions.md`
- Modify: `docs/progress.md`
- Modify: `docs/next.md`

1. Run causal diffusion and trajectory-energy tests.
2. Run existing diffusion/AR/parity/history-context regression tests and report
   the known config-drift assertions separately.
3. Run `py_compile`, config load/instantiation smoke, and `git diff --check`.
4. Append concise implementation results and remaining server-only validation
   work to project docs.
