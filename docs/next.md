# Next

## In Flight

- Run a real single-batch train smoke with `configs/train/train_scalable_ar_diffusion_local.yaml`; confirm finite `diffusion_loss`, 4 short-horizon chunk coverage logs, and no empty diffusion batch.
- Run a validation inference smoke with `configs/validation/validation_scalable_ar_diffusion.yaml`; confirm 8 AR rounds produce `pred_traj=(A,80,2)` and full `pred_valid_mask` coverage for generated sim agents.
- Compare `smart_ar_diffusion` against full-horizon `smart_diffusion` on boundary exits, collisions, map violations, and official export zero-fallback checks.

## Blockers

- Waymo official evaluation dependencies are not installed in the current environment, so official export assertion tests still skip here.
- `pytest` is not installed in the `smart` conda environment; current targeted verification uses `python -m unittest`.

## Next Actions

- Start from `configs/train/train_scalable_ar_diffusion_local.yaml` for local smoke.
- Watch `train_empty_diffusion_batch`, `diffusion_loss`, `train_mask_acc`, `train_loss_chunk_00` through `train_loss_chunk_03`, and self-conditioning metrics.
- Inspect AR validation visualizations for whether per-round local map rescreening reduces off-road trajectories and collisions.
- If real smoke passes, launch the server-scale `configs/train/train_scalable_ar_diffusion.yaml` run and keep full-horizon `smart_diffusion` as the controlled baseline.
