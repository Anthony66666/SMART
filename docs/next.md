# Next

## In Flight

- Local CUDA smoke on `data/valid_demo` passed three optimizer steps plus a full validation rollout; proceed to server setup after full-dataset retokenization calibration.
- Calibrate `retokenization_error_thresholds` on the full server training set with `scripts/calibrate_causal_retokenization.py --with_perturbation`, then update all causal train/validation configs with the frozen vehicle/pedestrian/cyclist P99 values.
- Launch the five from-scratch causal runs in order: clean causal, perturbation, closed-loop retokenization, safety energy, then weight/probability tuning. Keep `Model.total_steps` as an optimizer-step budget and do not initialize from the old AR checkpoint.
- Compare `val_rollout_score`, 2/4/6/8-second ADE/FDE, late ADE, lane/collision energies, map violations, and official SMART metrics against original SMART and the current AR diffusion checkpoint.
- Inspect `outputs/ar_map_rollout_query_debug_6c1658d_epoch00/` first for `/mnt/d/6c1658d_epoch=00.ckpt`; verify raw q overlap, proposal-refreshed q expansion, sampled q placement, rolled agents/history, and raw/proposal/sampled map-edge coverage before tuning self-conditioning/corruption schedules.
- After the history-context map mask alignment, rerun AR diffusion validation/visualizations with the checkpoint-matched config and compare agent context/map-edge coverage for category-3 targets versus non-target generation agents.
- Inspect `next_token_idx` predicted-token speed versus `next_token_idx_gt` for current-valid straight vehicles, split by `category == 3`, non-target generation agents, commit/proposal mode, visible-token corruption on/off, and causal-disabled/multiplier/fixed schedule variants.
- Continue comparing `smart_ar_diffusion` against full-horizon `smart_diffusion` on boundary exits, collisions, map violations, and official export zero-fallback checks.

## Blockers

- Waymo official evaluation dependencies are not installed in the current environment, so official export assertion tests still skip here.
- `tests.test_smart_ar_diffusion` has two unrelated config-drift failures because the user-modified server AR train YAML enables causal noise and disables visible corruption while old tests still assert the prior defaults.

## Next Actions

- Follow the `README.md` causal diffusion server checklist; because LR scheduling is intentionally epoch-based, set `Model.warmup_steps: 2` and `Model.total_steps: 32` for the 32-epoch run.
- Use `python scripts/calibrate_causal_retokenization.py --config configs/train/train_scalable_causal_diffusion.yaml --split train --max_samples <budget> --quantile 0.99 --with_perturbation --output_json <path>` before the first server run.
- Start causal training with `configs/train/train_scalable_causal_diffusion.yaml`; checkpoint selection uses `val_rollout_score`, while `val_minADE`/`val_minFDE` remain baseline-comparison metrics.
- For AR diffusion finetunes initialized from older checkpoints, prefer `--pretrain_ckpt` over `--ckpt_path`; confirm LR is nonzero and GPU0 does not retain per-rank checkpoint-loading contexts after startup.
- For AR diffusion server debugging, inspect `[SMARTDiffusion]`, `[ValidationVisualization]`, and `[StepVisualization]` stdout logs to see whether slowdowns happen in window loss, 16-round receding-horizon rollout, per-round sampling, or rank0 visualization.
- For AR diffusion training, remember epoch-end validation is now bounded by `Trainer.limit_val_batches` in the AR train configs; remove or raise that field for full official validation.
- For AR diffusion, treat `val_ar_window_loss` as a short-window denoising diagnostic only; use rollout metrics such as `val_minADE`/`val_minFDE` for checkpoint selection.
- Keep `metric_mode: smart_val_compatible` for official SMART parity; use `metric_mode: smart_category3` only for target-only ablations.
- Use `source /home/anthony/anaconda3/etc/profile.d/conda.sh && conda activate smart` before running SMART tests in noninteractive shells.
- If straight-vehicle speed remains high after visible-token neighbor corruption, add a diagnostic that logs predicted and GT token speed percentiles by agent/category/current-speed bin.
