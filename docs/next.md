# Next

## In Flight

- Run AR validation inference/visualization after the rollout history-token-validity fix and compare straight-vehicle speeds against the previous checkpoint/code path.
- Inspect `next_token_idx` predicted-token speed versus `next_token_idx_gt` for current-valid straight vehicles, split by `category == 3` and non-target generation agents.
- Continue comparing `smart_ar_diffusion` against full-horizon `smart_diffusion` on boundary exits, collisions, map violations, and official export zero-fallback checks.

## Blockers

- Waymo official evaluation dependencies are not installed in the current environment, so official export assertion tests still skip here.

## Next Actions

- Keep `metric_mode: smart_val_compatible` for official SMART parity; use `metric_mode: smart_category3` only for target-only ablations.
- Use `source /home/anthony/anaconda3/etc/profile.d/conda.sh && conda activate smart` before running SMART tests in noninteractive shells.
- If straight-vehicle speed remains high after the mask fix, add a diagnostic that logs predicted and GT token speed percentiles by agent/category/current-speed bin.
