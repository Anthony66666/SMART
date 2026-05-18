# Next

## In Flight

- Run a longer SMART-Diffusion GPU smoke with `configs/train/train_scalable_diffusion_local.yaml` after the prefix-constrained sampling/training update.
- Compare diffusion validation rollout metrics and interaction diagnostics against the baseline SMART checkpoint.
- Inspect regenerated step/epoch visualizations for large jumps, static trajectories, collisions, map violations, skipped samples from empty diffusion batches, and any remaining red traces connected to the origin.
- Watch whether target-category-only training, 6-layer decoding, remask sampling, and prefix-constrained geometry reduce out-of-map and collision failures.
- Watch whether `geometry_dropout_prob: 0.25` stabilizes rollout quality; lower it to `0.1` if loss becomes noisy or convergence slows.

## Blockers

- None.

## Next Actions

- Start local diffusion training from scratch with `train_scalable_diffusion_local.yaml`; save checkpoints under a new diffusion-specific directory.
- Validate the selected diffusion checkpoint with `configs/validation/validation_scalable_diffusion.yaml`.
- Track `train_empty_diffusion_batch`, `val_empty_diffusion_batch`, `val_loss`, `val_diffusion_loss`, `val_ntp_loss`, `val_mask_acc`, `val_minADE`, `val_minFDE`, `val_conflict_rate`, and `val_interaction_consistency` on target-category agents.
- Compare one short ablation with `prefix_constrained_sampling: false` and `prefix_constrained_training: false` only if the new run underperforms, to isolate the prefix constraint effect.
- Confirm future-token, temporal, and map-to-future edge counts are nonzero on typical batches, and flat `map_context.shape[0]` matches `map_valid_mask.sum()`.
- Watch GPU memory and step time after increasing diffusion decoder depth to 6; reduce only yaml `diffusion.num_layers` to 4 if the server OOMs.
- Compare one short run with `geometry_dropout_prob: 0.0` if the new run underperforms, to isolate the effect of geometry dropout.
- Regenerate the step visualization at the next interval and confirm predicted trajectories use `pred_valid_mask` with no `(0,0)` artifacts.
- If local metrics/visualizations look sane, run the server config `configs/train/train_scalable_diffusion.yaml` with the full training and validation data paths.
