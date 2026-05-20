# Next

## In Flight

- Run a short SMART-Diffusion training smoke with `configs/train/train_scalable_diffusion_local.yaml` after enabling training-time self-conditioning.
- Compare diffusion validation rollout metrics and interaction diagnostics against the baseline SMART checkpoint.
- Watch `train_self_condition_frac`, `train_self_condition_input_acc`, and `train_self_condition_acc` for finite values and useful nonzero coverage.
- Inspect regenerated step/epoch visualizations for large jumps, static trajectories, collisions, map violations, skipped samples from empty diffusion batches, and any remaining red traces connected to the origin.

## Blockers

- None.

## Next Actions

- Run `train_scalable_diffusion_local.yaml` with self-conditioning enabled and save to a new diffusion-specific checkpoint directory.
- Run one short ablation with `self_condition_prob: 0.0` under the same setup to isolate the self-conditioning effect.
- Validate the selected diffusion checkpoints with `configs/validation/validation_scalable_diffusion.yaml`.
- Track `train_empty_diffusion_batch`, `train_self_condition_frac`, `train_self_condition_input_acc`, `train_self_condition_acc`, `val_empty_diffusion_batch`, `val_loss`, `val_diffusion_loss`, `val_mask_acc`, `val_minADE`, `val_minFDE`, `val_conflict_rate`, and `val_interaction_consistency` on target-category agents.
- If the self-conditioned run improves free-running metrics, carry the same config to the server training config `configs/train/train_scalable_diffusion.yaml` with full data paths.
- If it destabilizes loss or visuals, reduce only `self_condition_prob` to `0.10` before changing other diffusion settings.
