# SMART Causal Diffusion Design

## Objective

Add an independent SMART predictor that keeps the 2048-entry discrete trajectory
token space but trains and samples in the same causal, closed-loop state
distribution used at rollout time. The primary objective is road adherence and
late-horizon stability, followed by displacement quality.

## Non-Goals

- Do not change the existing `smart`, `smart_diffusion`, or
  `smart_ar_diffusion` behavior.
- Do not hard-project trajectories onto lanes.
- Do not encode traffic-light state as a hard rule in the first implementation.
- Do not load an old diffusion checkpoint into the new predictor.

## Model

The predictor name is `smart_causal_diffusion`. It inherits the current SMART
encoder and AR rollout plumbing, but replaces the denoising contract:

- Predict four future SMART tokens (2 seconds).
- Commit one token (0.5 seconds) per outer rollout.
- Use strictly causal temporal token edges: earlier chunks may condition later
  chunks, never the reverse.
- Sample a visible prefix followed by one absorbing masked suffix during
  training.
- Reveal that suffix monotonically during inference. Released tokens are never
  remasked, and stale uncommitted proposals are not used as executable state.
- Add an explicit execution-frontier objective for the first masked token.

The absorbing corruption distribution is parameterized by the existing
continuous diffusion timestep. For each agent, sequential Bernoulli survival
draws create a valid visible prefix. This preserves a single global noise level
without the previous per-chunk mask multipliers.

## Closed-Loop Training

Training uses a 32-epoch curriculum:

- Current implementation note, updated 2026-06-13: model-rollout state
  probability starts at `closed_loop_batch_ratio_max` from epoch 0.
- Epochs 0-3: model rollout plus clean rolling anchors.
- Epochs 4-31: model rollout plus 25% correlated pose perturbation; remaining
  views are clean rolling anchors.

Rollout depth is sampled uniformly from one to four committed tokens. After a
perturbed or predicted state is created, the remaining ground-truth trajectory
is transformed into that state frame and retokenized against the original SMART
codebook. Retokenization returns both the nearest token and its continuous
matching error. Targets above configurable per-agent-type thresholds are
excluded from token cross-entropy and supervised by continuous recovery losses
instead.

The thresholds are intended to be calibrated from the training set at the 99th
percentile and then frozen in config. The implementation exposes the errors and
thresholds so calibration does not require changing model code.

## Safety Energy

The sampler evaluates the top-k candidate SMART tokens at each executable
frontier with a differentiable-free scoring module:

- lane distance to nearby map points,
- lane heading mismatch,
- acceleration and yaw-rate feasibility,
- pairwise collision overlap.

The energy weight follows `(1 - t)^2`, so early denoising remains diverse and
late decisions become safety-focused. Energy only reranks top-k discrete
candidates; it does not project decoded trajectories or make invalid map states
impossible.

## Validation

In addition to existing SMART-compatible ADE/FDE, the predictor reports:

- ADE/FDE at 2, 4, 6, and 8 seconds,
- late-horizon ADE over the final 4 seconds,
- mean lane-distance and lane-heading energies,
- dynamics energy,
- collision energy,
- prediction coverage and retokenization-invalid rate.

Checkpoint selection should use a weighted rollout score led by map and
collision quality. Official SMART metrics remain available for baseline
comparison.

## Experiment Matrix

1. Clean causal absorbing diffusion.
2. Add correlated perturbation.
3. Add closed-loop rollout states and retokenization.
4. Add soft top-k safety energy.
5. Tune only energy weights and rollout-state probability.

All five runs start from scratch with encoder learning rate at half the decoder
learning rate.
