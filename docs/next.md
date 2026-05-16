# Next

## In Flight

- Run a SMART-Diffusion GPU smoke with `configs/train/train_scalable_diffusion_local.yaml` after the BD3-LM-style vectorized block objective and schedule update.
- Compare diffusion validation rollout metrics and interaction diagnostics against the baseline SMART checkpoint.
- Inspect regenerated step/epoch visualizations for large jumps, static trajectories, collisions, map violations, skipped samples from empty diffusion batches, and any remaining red traces connected to the origin.
- Watch whether target-category-only training, 6-layer decoding, vectorized all-block training, data-driven clipping search, monotonic block sampling, and 4-chunk block rollout reduce out-of-map and collision failures.
- Watch whether `geometry_dropout_prob: 0.25` stabilizes rollout quality; lower it to `0.1` if loss becomes noisy or convergence slows.

## Blockers

- None.

## Next Actions

- Start local diffusion training from scratch with `train_scalable_diffusion_local.yaml`; do not continue old block-diffusion checkpoints because the block mask schedule and loss scaling changed again.
- Run one local Trainer train/validation pass to verify epoch-end `sampling_eps_min/max` updates and `valid_var_*` logging.
- Validate the selected diffusion checkpoint with `configs/validation/validation_scalable_diffusion.yaml`.
- Track `train_empty_diffusion_batch`, `val_empty_diffusion_batch`, `train_block_mask_ratio`, `val_block_mask_ratio`, `train_sampling_eps_min`, `train_sampling_eps_max`, `val_sampling_eps_min`, `val_sampling_eps_max`, `sampling_eps_min`, `sampling_eps_max`, `valid_var_*`, `val_loss`, `val_diffusion_loss`, `val_diffusion_loss_full`, `val_ntp_loss`, `val_mask_acc`, `val_minADE`, `val_minFDE`, `val_conflict_rate`, and `val_interaction_consistency` on target-category agents.
- Confirm future-token, temporal, and map-to-future edge counts are nonzero on typical batches, and flat `map_context.shape[0]` matches `map_valid_mask.sum()`.
- Watch GPU memory and step time after vectorizing all block views and increasing diffusion decoder depth to 6; use `block_vectorized_training: false` first if vectorized map duplication OOMs, then reduce only yaml `diffusion.num_layers` to 4 if needed.
- Compare one short run with `geometry_dropout_prob: 0.0` if the new run underperforms, to isolate the effect of geometry dropout.
- Compare ablations with `remask_sampling: true`, `block_size_chunks: 8`, `block_size_chunks: 16`, and `fix_clipping: true` if block diffusion underperforms, to isolate sampler, block-rollout, and schedule effects.
- Regenerate the step visualization at the next interval and confirm predicted trajectories use `pred_valid_mask` with no `(0,0)` artifacts.
- If local metrics/visualizations look sane, run the server config `configs/train/train_scalable_diffusion.yaml` with the full training and validation data paths.
