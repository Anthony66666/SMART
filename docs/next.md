# Next

## In Flight

- Run a longer SMART-Diffusion GPU smoke with `configs/train/train_scalable_diffusion_local.yaml` after the flat-map and dynamic-geometry updates.
- Compare diffusion validation rollout metrics and interaction diagnostics against the baseline SMART checkpoint.
- Inspect regenerated step/epoch visualizations for large jumps, static trajectories, collisions, map violations, skipped samples from empty diffusion batches, and any remaining red traces connected to the origin.
- Watch whether dynamic future-token geometry reduces the straight-line collapse seen in early diffusion visualizations.

## Blockers

- None.

## Next Actions

- Start local diffusion training from scratch with `train_scalable_diffusion_local.yaml`; save checkpoints under a new diffusion-specific directory.
- Validate the selected diffusion checkpoint with `configs/validation/validation_scalable_diffusion.yaml`.
- Track `train_empty_diffusion_batch`, `val_empty_diffusion_batch`, `val_loss`, `val_diffusion_loss`, `val_ntp_loss`, `val_mask_acc`, `val_minADE`, `val_minFDE`, `val_conflict_rate`, and `val_interaction_consistency`.
- Confirm future-token, temporal, and map-to-future edge counts are nonzero on typical batches, and flat `map_context.shape[0]` matches `map_valid_mask.sum()`.
- Watch GPU memory and step time after removing per-scene map padding; reintroduce `max_map_tokens` only if server memory requires it.
- Regenerate the step visualization at the next interval and confirm predicted trajectories use `pred_valid_mask` with no `(0,0)` artifacts.
- If local metrics/visualizations look sane, run the server config `configs/train/train_scalable_diffusion.yaml` with the full training and validation data paths.
