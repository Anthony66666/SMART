# Next

## In Flight

- Run AR validation inference/visualization after finetuning with receding-horizon `commit_tokens=1`, `carry_tail_proposal=true`, `causal_noise_schedule: false`, and `visible_token_corruption_prob: 0.15`; compare straight-vehicle speeds against the no-corruption and fixed causal schedule runs.
- Inspect `next_token_idx` predicted-token speed versus `next_token_idx_gt` for current-valid straight vehicles, split by `category == 3`, non-target generation agents, commit/proposal mode, visible-token corruption on/off, and causal-disabled/multiplier/fixed schedule variants.
- Continue comparing `smart_ar_diffusion` against full-horizon `smart_diffusion` on boundary exits, collisions, map violations, and official export zero-fallback checks.

## Blockers

- Waymo official evaluation dependencies are not installed in the current environment, so official export assertion tests still skip here.

## Next Actions

- For AR diffusion server debugging, inspect `[SMARTDiffusion]`, `[ValidationVisualization]`, and `[StepVisualization]` stdout logs to see whether slowdowns happen in window loss, 16-round receding-horizon rollout, per-round sampling, or rank0 visualization.
- For AR diffusion training, remember epoch-end validation is now bounded by `Trainer.limit_val_batches` in the AR train configs; remove or raise that field for full official validation.
- For AR diffusion, treat `val_ar_window_loss` as a short-window denoising diagnostic only; use rollout metrics such as `val_minADE`/`val_minFDE` for checkpoint selection.
- Keep `metric_mode: smart_val_compatible` for official SMART parity; use `metric_mode: smart_category3` only for target-only ablations.
- Use `source /home/anthony/anaconda3/etc/profile.d/conda.sh && conda activate smart` before running SMART tests in noninteractive shells.
- If straight-vehicle speed remains high after visible-token neighbor corruption, add a diagnostic that logs predicted and GT token speed percentiles by agent/category/current-speed bin.
