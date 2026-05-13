# Next

## In Flight

- Run a fresh SMART-Diffusion local training job with `configs/train/train_scalable_diffusion_local.yaml` after the graph-edge/map-context/loss updates.
- Compare diffusion validation rollout metrics and interaction diagnostics against the baseline SMART checkpoint.
- Inspect regenerated step/epoch visualizations for large jumps, static trajectories, collisions, map violations, skipped samples from empty diffusion batches, and any remaining red traces connected to the origin.
- Watch whether diffusion loss variance drops after valid-token normalization; batch size 1 still limits low-discrepancy timestep benefits.

## Blockers

- None.

## Next Actions

- Start local diffusion training from scratch with `train_scalable_diffusion_local.yaml`; save checkpoints under a new diffusion-specific directory.
- Validate the selected diffusion checkpoint with `configs/validation/validation_scalable_diffusion.yaml`.
- Track `train_empty_diffusion_batch`, `val_empty_diffusion_batch`, `val_loss`, `val_diffusion_loss`, `val_ntp_loss`, `val_mask_acc`, `val_minADE`, `val_minFDE`, `val_conflict_rate`, and `val_interaction_consistency`.
- Confirm future-token, temporal, and map-to-future edge counts are nonzero on typical batches, and `map_valid_mask.sum()` remains nonzero.
- Watch whether type-specific physical token embeddings reduce early straight-line collapse compared with the previous random token-id embedding run.
- Regenerate the step visualization at the next interval and confirm predicted trajectories use `pred_valid_mask` with no `(0,0)` artifacts.
- If local metrics/visualizations look sane, run the server config `configs/train/train_scalable_diffusion.yaml` with the full training and validation data paths.
- Keep using graph-first exploration before direct file reads for future substantial tasks.
