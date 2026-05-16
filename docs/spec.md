# SMART Self-Distillation Spec

## Goal
Add an ablation-friendly `smart_self_distill` training mode for quality-diverse closed-loop distillation on top of SMART.

## Scope
- Preserve the existing `smart` predictor and inference return formats.
- Add closed-loop rollout prefix scoring for training only.
- Add CAT-K-style recovery, EMA teacher KL, reward-weighted rollout NLL, and entropy regularization switches.
- Use tensor-only proxy rewards in v1: ADE, collision, kinematic smoothness, and rollout diversity.

## Constraints
- Do not implement privileged future teacher in v1.
- Do not change validation defaults.
- Keep generated artifacts, checkpoints, cache, and outputs out of source changes.

## Acceptance Criteria
- `Model.predictor: "smart"` keeps the baseline path.
- `Model.predictor: "smart_self_distill"` instantiates `SMARTSelfDistill`.
- `distill.enabled: false` returns baseline CE loss.
- `distill.enabled: true` logs CE, CAT-K, rollout KL, rollout NLL, entropy, reward, and diversity losses/metrics.
- Static compile and distillation unit tests pass.
