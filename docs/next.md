# Next

## In Flight

- Run a validation inference smoke with `configs/validation/validation_scalable_ar_diffusion.yaml`; confirm current-valid non-background agents are visible in official validation visualizations even when `category != 3`.
- Compare official-view visualizations against optional supervision/target view to verify only the latter changes when `WaymoTargetBuilder` randomly selects category-3 targets.
- Continue comparing `smart_ar_diffusion` against full-horizon `smart_diffusion` on boundary exits, collisions, map violations, and official export zero-fallback checks.

## Blockers

- Waymo official evaluation dependencies are not installed in the current environment, so official export assertion tests still skip here.

## Next Actions

- Inspect `pred_valid_mask` and `official_valid_mask` together when diagnosing missing trajectories; do not infer missing generation from category-filtered `valid_mask` alone.
- Keep `metric_mode: smart_val_compatible` for official SMART parity; use `metric_mode: smart_category3` only for target-only ablations.
- Use `source /home/anthony/anaconda3/etc/profile.d/conda.sh && conda activate smart` before running SMART tests in noninteractive shells.
