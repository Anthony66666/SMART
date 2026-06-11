# Next

## In Flight

- Launch causal v2 from scratch; do not resume `/mnt/d/causal_epoch=00.ckpt`, because the objective, state conditioning, proposal path, and decoder inputs changed.
- Use the frozen 10,000-scene perturbed P99 thresholds already written to all causal configs.
- Compare causal v2 against original SMART, the old causal checkpoint, and the current AR diffusion checkpoint using identical validation scenes.
- Compare `val_rollout_score`, 2/4/6/8-second ADE/FDE, late ADE, lane/collision energies, map violations, and official SMART metrics against original SMART and the current AR diffusion checkpoint.
- Inspect `outputs/ar_map_rollout_query_debug_6c1658d_epoch00/` first for `/mnt/d/6c1658d_epoch=00.ckpt`; verify raw q overlap, proposal-refreshed q expansion, sampled q placement, rolled agents/history, and raw/proposal/sampled map-edge coverage before tuning self-conditioning/corruption schedules.
- After the history-context map mask alignment, rerun AR diffusion validation/visualizations with the checkpoint-matched config and compare agent context/map-edge coverage for category-3 targets versus non-target generation agents.
- Inspect `next_token_idx` predicted-token speed versus `next_token_idx_gt` for current-valid straight vehicles, split by `category == 3`, non-target generation agents, commit/proposal mode, visible-token corruption on/off, and causal-disabled/multiplier/fixed schedule variants.
- Continue comparing `smart_ar_diffusion` against full-horizon `smart_diffusion` on boundary exits, collisions, map violations, and official export zero-fallback checks.

## Blockers

- Waymo official evaluation dependencies are not installed in the current environment, so official export assertion tests still skip here.
- `tests.test_smart_ar_diffusion` has two unrelated config-drift failures because the user-modified server AR train YAML enables causal noise and disables visible corruption while old tests still assert the prior defaults.

## Next Actions

- Run the five-epoch local demo command documented in `README.md` and confirm the model can overfit the 11 scenes before launching the full server run.
- Use `docs/train_scalable_causal_diffusion_config.md` as the field-by-field reference when editing the server causal config; verify all linked time/token fields remain consistent.
- Follow the `README.md` causal diffusion server checklist; the committed server config already uses epoch-based `warmup_steps: 2` and `total_steps: 32`.
- Archive `outputs/calibration/causal_retokenization_p99.json` with the run metadata; recalibrate only if the training dataset or perturbation policy changes.
- Start causal training with `configs/train/train_scalable_causal_diffusion.yaml`; checkpoint selection uses `val_rollout_score`, while `val_minADE`/`val_minFDE` remain baseline-comparison metrics.
- For AR diffusion finetunes initialized from older checkpoints, prefer `--pretrain_ckpt` over `--ckpt_path`; confirm LR is nonzero and GPU0 does not retain per-rank checkpoint-loading contexts after startup.
- For AR diffusion server debugging, inspect `[SMARTDiffusion]`, `[ValidationVisualization]`, and `[StepVisualization]` stdout logs to see whether slowdowns happen in window loss, 16-round receding-horizon rollout, per-round sampling, or rank0 visualization.
- For AR diffusion training, remember epoch-end validation is now bounded by `Trainer.limit_val_batches` in the AR train configs; remove or raise that field for full official validation.
- For AR diffusion, treat `val_ar_window_loss` as a short-window denoising diagnostic only; use rollout metrics such as `val_minADE`/`val_minFDE` for checkpoint selection.
- Keep `metric_mode: smart_val_compatible` for official SMART parity; use `metric_mode: smart_category3` only for target-only ablations.
- Use `source /home/anthony/anaconda3/etc/profile.d/conda.sh && conda activate smart` before running SMART tests in noninteractive shells.
- If straight-vehicle speed remains high after visible-token neighbor corruption, add a diagnostic that logs predicted and GT token speed percentiles by agent/category/current-speed bin.
