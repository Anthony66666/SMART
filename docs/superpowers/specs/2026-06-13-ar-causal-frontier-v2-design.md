# AR Causal Frontier v2 Design

## Objective

Improve the existing `smart_ar_diffusion` path without replacing its public
training, validation, inference, or visualization surface. The goal is to keep
AR diffusion as the comparable baseline branch, then add a configurable
frontier-aligned AR variant that trains on the state distribution used by its
closed-loop rollout.

## Motivation

The current AR diffusion path already uses receding-horizon execution:

- two SMART history tokens,
- a short future token window,
- one committed executable token by default,
- revisable tail proposal carry,
- SMART-style map context refresh.

Its main remaining mismatch is train/inference distribution. Training mostly
sees clean teacher-forced short windows with masked-token denoising, while
inference repeatedly commits model tokens, rolls history, rebuilds map and
agent context, and conditions the next window on model-induced states. This is
the same exposure-bias problem described by Scheduled Sampling, DAgger, and
Professor Forcing. It also matches the motivation behind receding-horizon
Diffusion Policy: a short plan is useful only if training covers the states
created by repeatedly executing the plan.

Diffusion Forcing suggests a second useful change: near executable tokens and
far proposal tokens should not be treated as identical. The first frontier token
should be the reliable action, while later chunks remain softer proposal
context. MaskGIT-style bidirectional remasking is still useful as a baseline,
but it is not the best execution contract for a closed-loop planner.

## Design

Add a config-gated `smart_ar_diffusion` mode named
`diffusion.ar_objective: causal_frontier_v1`.

The default objective remains `maskgit`, preserving old checkpoints and old
configs. The new objective changes only behavior selected by config:

- Use a causal temporal token decoder inside the AR window when
  `diffusion.causal_temporal_edges: true`.
- Train by sampling one frontier chunk per packed sequence, masking that chunk
  and all later chunks, and supervising only the selected frontier.
- Optionally build some training views by committing one to four model-sampled
  tokens, rolling history, rebuilding the AR window, and retokenizing the
  remaining ground-truth continuation in the rolled state frame.
- Reuse causal retokenization thresholds and endpoint recovery loss for cases
  where the rolled state makes the nearest token assignment unreliable.
- Add current-motion conditioning to AR packed inputs when
  `diffusion.current_state_enabled: true`.
- Keep AR inference as propose window, commit configured tokens, carry tail
  proposal, roll history, and rebuild context.

This migrates the causal branch's most portable advantages into the original AR
branch while keeping the original AR baseline selectable.

## Non-Goals

- Do not delete `smart_causal_diffusion` or `smart_causal_flow_matching`.
- Do not force all AR configs to the new objective.
- Do not project trajectories onto lanes.
- Do not make safety or ego-edit guidance part of the first AR frontier
  training objective.
- Do not claim trained quality until a 1000-step run and matched validation
  comparison have completed.

## Experiment Contract

All short-run comparisons use the same data and approximate optimization
budget:

- `train_raw_dir: ["/home/anthony/SimAgentJEPA/data/waymo/training_subset_10pct"]`
- `val_raw_dir: ["/home/anthony/SimAgentJEPA/data/waymo/validation"]`
- `Trainer.max_steps: 1000`
- same validation scene indices for summary metrics and PNG visualizations.

The primary comparison is:

- `ar_baseline_1000`: current AR mask-diffusion objective,
- `ar_frontier_v2_1000`: new AR causal frontier objective,
- `causal_diffusion_1000`: existing causal branch under the same short budget,
- `causal_flow_matching_1000`: optional additive flow branch under the same
  short budget.

The comparison report writes one table with ADE/FDE and rollout diagnostics,
plus one visualization directory per model.

## Acceptance Criteria

- Existing AR behavior remains available through `ar_objective: maskgit`.
- The new AR objective has unit coverage for frontier masking, closed-loop
  training view selection, causal decoder selection, and `Trainer.max_steps`.
- A unified local config set can train each candidate for about 1000 optimizer
  steps on the specified data.
- A comparison script can load model configs/checkpoints, evaluate the same
  validation indices, write `summary.csv` and `records.csv`, and save PNG
  visualizations per model.
- If the current shell cannot access GPU, the implementation and smoke tests
  still complete locally, and the exact training/evaluation commands are
  recorded for the GPU-enabled run.
