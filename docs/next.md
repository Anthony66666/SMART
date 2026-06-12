# Next

## In Flight

- When diagnosing causal v2 checkpoints, use the updated `scripts/smoke_causal_guidance_modes.py` speed fields (`pred_speed`, `gt_speed`, `moving_speed_ratio`, `moving_speed_ratio_p10/p50/p90`) alongside ADE/FDE. A low moving-speed ratio is now the primary signal for the conservative/near-static collapse.
- The current hybrid commit path keeps causal diffusion proposals/editing but adds chunk-0 speed-reference reranking. On `data/valid_demo`, it improves `safe` speed ratios but does not fully remove low-speed bias; next tuning should compare `commit_speed_energy_weight`, `commit_min_speed_ratio`, and `commit_speed_reference_decay` on real validation scenes, not only demo scenes.
- Do not interpret old smoke FDE artifacts from non-contiguous valid masks as map exits; FDE now uses the last actual valid frame. Re-run old smoke/visual results before comparing FDE numbers.
- Re-run old ego-risk smoke/visual outputs before comparing `ego_risk_min_ttc` or `token_change_rate_vs_gt`; no-finite TTC now stays infinite and token-change rate now uses only valid token slots as the denominator.
- The local workspace does not currently have the causal v2 epoch-1 checkpoint referenced by `/mnt/d/causal_v2_epoch=02.ckpt`; `/mnt/d/90113f8_epoch=01.ckpt` and `/mnt/d/90113f8_v2_epoch=01.ckpt` are `smart_ar_diffusion`, not causal v2. Copy `/raid/haoq_lab/wangshijie/SMART/checkpoints/causal_v2/epoch=01.ckpt` locally before epoch-1/epoch-2 causal comparison.
- Validate generic ego-risk guidance on checkpoint scenes with `seed,none,safe,ego_stress,ego_edit`; focus on ego-only low-TTC, near-miss, hard-collision/offroad, edit distance, and non-target preservation. Do not optimize the main study around predefined target-event success.
- Use `outputs/causal_ego_risk_guidance/` as the first generic low-TTC smoke sweep over the 11 demo scenes. It is useful for debugging scoring and aggregation, not yet for paper claims.
- Use `outputs/causal_ego_risk_guidance_visual/idx_00010_1ce0b4bbd35a6ad1_guidance_modes.png` as the current generic ego-risk visual smoke. It shows the controlled target and ego/SDC without cut-in-specific markers, but the visual effect is still subtle.
- Use `docs/ego_risk_waymo_validation_candidates.md` as the current real-validation candidate list. The best visual examples from the first 100 validation scenes are indices 72, 87, 31, 17, 41, and 33.
- Treat `outputs/causal_ego_guidance_cut_in_success/` and `outputs/causal_ego_guidance_cut_in_scene9_agent4_marked/` as legacy cut-in ablation artifacts only. They are no longer the primary direction.
- Extend ego-risk sweeps beyond the first 100 validation scenes and improve target-agent/window selection around ego interactions because the current evidence is still a screen, not paper-scale evaluation.
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

- Use `docs/aaai_causal_diffusion_method_zh.md` as the paper-style Chinese Method draft before writing the final AAAI English version.
- Use `docs/smart_causal_diffusion_overview.md` with `docs/train_scalable_causal_diffusion_config.md` as the current causal v2 architecture/config reference before editing or launching causal runs.
- Use `scripts/visualize_causal_guidance_modes.py --modes seed,none,safe,ego_stress,ego_edit --target-spec ego_risk --target-event-eta 0.0 --output-dir outputs/causal_ego_risk_guidance_visual` as the quick visual comparison command for generic ego-risk guidance.
- For final figures, first scan candidate scenes/agents by low ego TTC, small ego-target distance, route proximity, zero hard collision/offroad, and small non-target preservation error; only then render selected scenes.
- Run cut-in or lead-hard-brake target specs only as legacy ablations. For those runs, pass explicit target agents/windows after geometry scanning instead of relying on the default target selector.
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
