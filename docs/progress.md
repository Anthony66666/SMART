# Progress

## 2026-06-20 CST
- Task: Fixed the SMART agent token embedding crash under `16-mixed` / AMP training.
- Result: `SMARTAgentDecoder.agent_token_embedding()` now allocates the packed token-embedding buffer from the token embedding output dtype instead of default FP32, so autocast FP16 token embeddings can be indexed back into the agent-token tensor without dtype mismatch. Inference trajectory-token buffers also allocate with their source trajectory-token dtype. The full server discrete diffusion-policy config now uses `Trainer.precision: "16-mixed"` while keeping `train_batch_size: 4`.
- Files: `smart/modules/agent_decoder.py`, `configs/train/train_scalable_discrete_diffusion_policy.yaml`, `tests/test_agent_decoder_history_context.py`, `tests/test_smart_discrete_diffusion_policy.py`, `docs/progress.md`, `docs/next.md`
- Validation: Added a regression that forces token embeddings to FP16 and calls the real `agent_token_embedding()` inference path; it failed on the previous FP32 buffer and now passes. Added a server-config regression that failed while precision was `32` and now requires `16-mixed`. Related agent-decoder, discrete-policy, compare-model tests and py_compile passed locally.
- Next: Retry the server `16-mixed` discrete diffusion-policy run. If another AMP error appears, inspect the next scratch tensor created with default `torch.zeros(...)` in that stack.

## 2026-06-20 CST
- Task: Added batched multi-anchor training for `smart_discrete_diffusion_policy`.
- Result: Replaced the default per-anchor training loop with a batched anchor-view path: selected anchors are converted into `(scene, anchor)` graph samples, packed into one PyG batch, and passed through one diffusion input/loss call. Each anchor keeps its own teacher-forced history context; the implementation does not reuse a stale single `h_t` across anchors.
- Config: Enabled `discrete_policy_batched_multi_anchor: true` and set `self_condition_prob: 0.0` in discrete-policy train/smoke/validation configs so span supervision does not trigger repeated no-grad denoiser self-conditioning.
- Validation: Added regressions that fail if the batched path calls per-anchor `_build_ar_training_view`, if anchor/source metadata is lost, or if config disables batched multi-anchor / re-enables self-conditioning.

## 2026-06-19 CST
- Task: Converted `smart_discrete_diffusion_policy` from SMART-prior-plus-rerank training to a pure diffusion-policy chunk objective.
- Result: Removed the dense SMART NTP CE second forward from `SMARTDiscreteDiffusionPolicy.training_step`, kept chunk-weighted forced full-window x0 CE and optional overlap KL, added per-horizon `loss_x0_chunk0..3`, `chunk0_acc..3`, `loss_overlap`, and supervision-coverage logs, and added shape/mask assertions so the denoiser supervises every valid target in each window.
- Config: Updated train/smoke/validation configs to `discrete_policy_objective: pure_chunk_v1`, `ntp_aux_loss_weight: 0.0`, `use_smart_ntp_head: false`, `use_smart_prior_fusion: false`, `proposal_memory.enabled: false`, `temporal_ensemble.enabled: false`, `sampling_guidance.enabled: false`, `prediction_horizon: 4`, `execution_horizon: 1`, `chunk_loss_weights: [1.0, 0.3, 0.15, 0.075]`, `overlap_loss_weight: 0.05`, and single-sample/no-energy inference.
- Validation: Added regressions that fail if training calls SMART NTP CE, if per-chunk metrics are missing, if terminal PAD windows count invalid targets, or if local discrete-policy config re-enables SMART prior/memory/guidance/rerank semantics.

## 2026-06-19 CST
- Task: Completed the local 2000-step smoke run for `smart_discrete_diffusion_policy`.
- Result: Updated `configs/train/train_scalable_discrete_diffusion_policy_2000.yaml` into a tiny local smoke config using `data/valid_demo`, reduced model layers/context, global step-based validation, and the same discrete-policy training terms: dense SMART NTP, chunk-weighted four-token diffusion CE, and overlap KL.
- Run: `python -u train.py --config configs/train/train_scalable_discrete_diffusion_policy_2000.yaml --save_ckpt_path checkpoints/discrete_diffusion_policy_2000` reached `max_steps=2000`, ran one validation batch, and wrote the current smoke checkpoint to `checkpoints/discrete_diffusion_policy_2000/last.ckpt` (`global_step=2000`, `epoch=181`, `hidden_dim=64`). A stale full-config checkpoint in the same directory was preserved as `last_full_config_2000.ckpt`.
- Metrics: TensorBoard `version_98` recorded final smoke metrics at step 1999: `val_minADE=4.3555`, `val_minFDE=10.0936`, `val_ar_window_loss=3.0050`, `val_ar_window_mask_acc=0.3556`, `train_loss_epoch=4.6166`, `smart_ntp_loss_epoch=2.2379`, `discrete_policy_chunk_loss_epoch=9.4120`, and `discrete_policy_overlap_loss_epoch=0.5158`.
- Outputs: Step visualizations are under `outputs/step_discrete_diffusion_policy_2000/smart_discrete_diffusion_policy/step_002000/`; validation visualizations are under `outputs/val_discrete_diffusion_policy_2000/smart_discrete_diffusion_policy/epoch_182/`. PNGs were verified as nonblank 1440x1440 RGBA images.
- Next: Use the smoke result only to verify the train/validation/visualization path. For quality claims, train the full server config `configs/train/train_scalable_discrete_diffusion_policy.yaml`.

## 2026-06-19 CST
- Task: Added a discrete diffusion-policy branch that commits one token from reranked four-token candidates.
- Result: Added `smart_discrete_diffusion_policy`, a SMART-token diffusion-policy ablation that keeps the AR receding-horizon shell, predicts four-token windows, commits only chunk 0, discards the tail chunks, and scores multiple sampled candidate windows with decayed chunk energy weights before commit.
- Training objective: Combines dense original SMART next-token CE (`ntp_aux_loss_weight: 1.0`), chunk-weighted full-window diffusion CE with weights `[1.0, 0.3, 0.15, 0.075]`, terminal-safe incomplete windows, deterministic contiguous anchor spans, and overlap KL from source tail logits to the corresponding future chunk-0 logits.
- Files: `smart/model/smart_discrete_diffusion_policy.py`, predictor registries, local/server/validation discrete-policy configs, `tests/test_smart_discrete_diffusion_policy.py`, `tests/test_compare_motion_models.py`, and durable docs.
- Validation: Focused unit tests cover one-token commit without temporal voting, decayed candidate-window scoring, contiguous anchor-window training, overlap KL wiring, total loss composition, and config invariants. The local 2000-step config is a tiny `data/valid_demo` functional smoke with reduced layers/context so it can finish locally; do not use it for quality comparisons.
- Next: Train `configs/train/train_scalable_discrete_diffusion_policy.yaml` on the server and compare against AR rerank, discrete action-chunk, continuous action diffusion, causal flow matching, and original SMART on a fixed validation slice.

## 2026-06-19 CST
- Task: Added a diffusion-policy continuous action-chunk branch.
- Result: Added `smart_continuous_action_diffusion`, which reuses SMART map/history context and AR receding-horizon rollout, but trains a continuous denoising head over four-token / twenty-frame local action chunks. Inference samples continuous action chunks, ensembles overlapping predictions in world trajectory space, commits chunk 0, and retokenizes the committed chunk only to keep SMART history/interface compatibility.
- Files: `smart/model/smart_continuous_action_diffusion.py`, predictor registries, continuous-action train/validation configs, `tests/test_smart_continuous_action_diffusion.py`, `tests/test_compare_motion_models.py`, and durable docs.
- Validation: Added regressions for local/world action transforms, continuous temporal ensembling, final-step clean denoising, local/server config invariants, and compare-script config coverage. `python -m unittest tests.test_smart_continuous_action_diffusion tests.test_compare_motion_models tests.test_smart_ar_diffusion -v` passed 56 tests; `py_compile`, model construction smoke, and `git diff --check` passed locally.
- Next: Train `configs/train/train_scalable_continuous_action_diffusion.yaml` on the server and compare against the discrete ACT-style branch, AR rerank, causal flow matching, and original SMART on the same fixed validation slice.

## 2026-06-19 CST
- Task: Stabilized action-chunk AR diffusion after weak server-training quality.
- Result: Restored interval dense SMART CE replay in action-chunk training, added an action-chunk-only adjacent-window shift-consistency auxiliary loss, aligned consistency comparisons by scene id / agent id / shifted chunk id, and added a full server training config at `configs/train/train_scalable_ar_action_chunk.yaml`.
- Files: `smart/model/smart_action_chunk_diffusion.py`, `smart/model/smart_ar_diffusion.py`, action-chunk train configs, `tests/test_smart_action_chunk_diffusion.py`, and durable docs.
- Validation: Added red/green regressions for overlap KL behavior, packed scene-row alignment, local/server config invariants, and action-chunk temporal voting. Focused action-chunk tests passed locally.
- Next: Retrain action-chunk from scratch before comparing quality; previous `checkpoints/ar_action_chunk_1000/last.ckpt` predates these training changes.

## 2026-06-19 CST
- Task: Added an ACT/ALOHA-style action-chunk AR diffusion ablation.
- Result: Added `smart_action_chunk_diffusion`, which reuses `SMARTAutoregressiveDiffusion` training and sampling but overrides commit selection with temporal ensembling across overlapping four-token windows. The initial 1000-step config disabled tail proposal carry/conditioning and dense full-sequence SMART CE replay, supervised all chunk slots equally, kept cadf_lite single-window training with local NTP, and wrote validation/step visualizations under `outputs/val_ar_action_chunk_1000` and `outputs/step_ar_action_chunk_1000`.
- Files: `smart/model/smart_action_chunk_diffusion.py`, `smart/model/smart_ar_diffusion.py`, predictor registries, action-chunk train/validation configs, `tests/test_smart_action_chunk_diffusion.py`, `tests/test_compare_motion_models.py`, and durable docs.
- Validation: Added regressions for overlapping tail-token temporal voting and config invariants. Focused action-chunk tests, 1000-step config tests, AR diffusion regressions, py_compile, and config construction smoke passed locally. The local 1000-step run completed under `checkpoints/ar_action_chunk_1000/last.ckpt`; TensorBoard `version_90` recorded `val_minADE=3.2032`, `val_minFDE=11.4179`, `val_ar_window_loss=1.3118`, and `val_ar_window_mask_acc=0.1866`.
- Outputs: Generated 4 validation PNGs under `outputs/val_ar_action_chunk_1000/smart_action_chunk_diffusion/epoch_001/` and 4 step PNGs under `outputs/step_ar_action_chunk_1000/smart_action_chunk_diffusion/step_001000/`; all opened as 1440x1440 RGBA images.
- Next: Inspect the eight PNGs and compare the checkpoint against AR baseline/rerank, causal diffusion, flow matching, ELF, and hybrid on a larger fixed validation slice.

## 2026-06-18 CST
- Task: Replaced active AR rerank all-anchor training with a single-forward cadf_lite path.
- Result: `SMARTAutoregressiveDiffusion` now supports `ar_training_mode: cadf_lite`: each non-replay step selects one deterministic future anchor, uses a terminal-safe PAD window, force-masks valid chunks with commit-primary weights, computes local chunk0 NTP CE from the same encoder context, and cycles all-mask/carry-over proposal initialization. Active rerank train configs disable explicit shift KL and replay full-sequence SMART CE every 8 steps.
- Files: `smart/model/smart_ar_diffusion.py`, `smart/model/smart_diffusion.py`, AR rerank train configs, `tests/test_smart_ar_diffusion.py`, and durable docs.
- Validation: Added cadf_lite regressions for deterministic anchor cycling, proposal init modes, and single-window training-step behavior. `tests.test_smart_ar_diffusion` and diffusion/causal/smart-parity regressions passed locally.
- Next: Retrain AR rerank from scratch with the cadf_lite configs before comparing speed, ADE/FDE, or late map adherence against all-anchor or older rerank checkpoints.

## 2026-06-18 CST
- Task: Converted AR rerank diffusion training to commitment-aware all-anchor windows.
- Result: `SMARTAutoregressiveDiffusion` now supports `commitment_aware_training` and `proposal_shift_consistency_loss_weight`. In this mode the MaskGIT diffusion loss enumerates every future anchor, permits terminal PAD windows, force-masks valid chunks with commit-primary loss weights, injects shifted proposal inputs for non-initial anchors, and adds adjacent-window proposal shift consistency. Rerank train configs now use `causal_loss_weights: [1.0, 0.3, 0.1, 0.05]`, `commitment_aware_training: true`, `proposal_shift_consistency_loss_weight: 0.1`, and disable the older standalone proposal-carry auxiliary loss.
- Files: `smart/model/smart_ar_diffusion.py`, `smart/model/smart_diffusion.py`, AR rerank train configs, `tests/test_smart_ar_diffusion.py`, and durable docs.
- Validation: Added regressions for terminal PAD windows, all-future chunk0 anchor coverage, forced valid-window masks, proposal shift consistency, and supervision-weight loss normalization. Focused AR rerank tests passed.
- Next: Retrain AR rerank from scratch and expect slower steps because one batch now runs all future-anchor diffusion windows.

## 2026-06-18 CST
- Task: Added a fast auxiliary-loss schedule for AR rerank training.
- Result: `SMARTAutoregressiveDiffusion` added `dense_smart_ce_interval`, `proposal_carry_interval`, and `proposal_carry_detach_encoder`. Dense original-SMART CE and proposal-carry diffusion can be skipped on non-interval steps, and proposal-carry input construction can run under `no_grad` while the diffusion decoder loss remains trainable. This was the fast auxiliary schedule before the later commitment-aware all-anchor rerank objective disabled the standalone proposal-carry auxiliary in active rerank configs.
- Files: `smart/model/smart_ar_diffusion.py`, AR rerank train configs, `tests/test_smart_ar_diffusion.py`, and durable docs.
- Validation: Added red/green regressions for dense CE interval skipping, proposal-carry interval skipping, proposal-carry encoder detach, and fast rerank config fields. Focused tests passed after implementation.
- Next: Run full AR diffusion regression tests and then retrain AR rerank with the fast schedule before judging speed/quality.

## 2026-06-17 CST
- Task: Implemented dense SMART CE and proposal-carry training for AR rerank.
- Result: `smart_ar_diffusion` MaskGIT training can now combine the sampled diffusion-window loss with dense original-SMART next-token CE over the prepared full scene and an auxiliary next-window proposal-carry diffusion loss. Proposal-carry training builds the post-commit window, corrupts the previous window's tail tokens as noisy proposals, injects them into proposal geometry/embedding conditioning, and supervises the whole next window with a forced mask. Causal loss weighting is now an explicit config gate so old AR baseline configs are not changed accidentally.
- Files: `smart/model/smart_ar_diffusion.py`, `smart/model/smart_diffusion.py`, AR rerank train/validation configs, `tests/test_smart_ar_diffusion.py`, and durable docs.
- Validation: Added regressions for explicit causal loss weighting, forced-mask proposal diffusion loss, proposal-carry view construction, terminal-window skipping, and total training-loss composition. `python -m unittest tests.test_smart_ar_diffusion -v` passed 34 tests.
- Next: Retrain AR rerank from scratch; older rerank checkpoints did not train dense full-sequence CE or proposal-carry conditioning and should be treated as stale for map/speed conclusions.

## 2026-06-17 CST
- Task: Fixed SMART ELF map/context drift in receding-horizon rollout.
- Result: `smart_elf` now aligns generation, supervision, metric, and map-attention masks for sim-agent rollout by training active ELF configs on all current-valid non-background agents and passing an explicit generation-agent mask into SMART history-context map attention. Receding ELF training views and inference commits now roll token history and frame history instead of only overwriting the final history anchor.
- Files: `smart/model/smart_elf.py`, `smart/modules/agent_decoder.py`, `smart/modules/smart_decoder.py`, ELF train/validation configs, and `tests/test_smart_elf.py`.
- Validation: Added regressions for all-generation-agent map masks, rolling training history tokens, rolling committed inference history tokens, and all-agent ELF config invariants. `python -m unittest tests.test_smart_elf tests.test_compare_motion_models -v` passed 20 tests; `python -m unittest tests.test_agent_decoder_history_context -v` passed; `py_compile` passed for touched Python files; `git diff --check` passed. A real `data/valid_demo` CPU inference smoke produced finite output with `pred_traj=(58, 80, 2)`, `next_token_idx=(58, 16)`, and `valid_frames=3777`.

## 2026-06-15 CST
- Task: Removed road centerline rendering from the multi-camera layout default.
- Result: `scripts/render_multicamera_layout.py` now keeps map polylines disabled by default so camera-layout conditioning images contain agent 3D boxes without road centerline overlays. A new `--draw-map-polylines` flag keeps the old map-line drawing path available only for debugging.
- Output: Re-rendered a real checkpoint smoke from `checkpoints/causal_diffusion_1000/last.ckpt` to `outputs/multicamera_layout_causal_diffusion_idx0_no_centerline/` with 4 frames, 6 cameras, `draw_map_polylines: false`, and a 6-view preview at `preview_frame_0000_grid.png`.
- Validation: Added regressions for default no-map-line rendering and opt-in debug map-line rendering. `python -m unittest tests.test_multicamera_layout -v` passed 5 tests; `python -m py_compile scripts/render_multicamera_layout.py tests/test_multicamera_layout.py` passed.

## 2026-06-15 CST
- Task: Added MVP-0 multi-camera layout export for SMART-generated scenarios.
- Result: Added `scripts/render_multicamera_layout.py`, which loads a normal SMART validation config/checkpoint, runs inference on the original SMART `HeteroData` / `Batch` input, converts predicted trajectories into a scene layout, and renders six synthetic camera-view PNG streams plus `manifest.json`. The renderer also exposes pure geometry helpers and a synthetic scene path for fast tests.
- Files: `scripts/render_multicamera_layout.py`, `tests/test_multicamera_layout.py`, and durable docs.
- Validation: `python -m unittest tests.test_multicamera_layout -v` passed 3 tests; `python -m py_compile scripts/render_multicamera_layout.py tests/test_multicamera_layout.py` passed; `python scripts/render_multicamera_layout.py --help` printed the CLI usage successfully.
- Next: Run the exporter on a trained checkpoint and inspect projected agent counts/images before attaching a downstream video generator.

## 2026-06-15 CST
- Task: Added a composition-based SMART hybrid diffusion predictor.
- Result: Added `smart_hybrid_diffusion` as a `pl.LightningModule` that does not inherit existing SMART predictor classes. It keeps original SMART batch inputs, composes the causal closed-loop SMART-token rollout core, registers train/validation/comparison entry points, and adds hybrid configs for local, matched 1000-step, and validation runs. The hybrid commit-speed energy now penalizes executable chunk-0 tokens that are too slow or too fast relative to the observed/reference speed band.
- Files: `smart/model/smart_hybrid_diffusion.py`, `smart/model/__init__.py`, `train.py`, `val.py`, `scripts/compare_motion_models.py`, hybrid train/validation configs, focused tests, and durable docs.
- Validation: `python -m unittest tests.test_smart_hybrid_diffusion tests.test_train_entrypoint_config tests.test_compare_motion_models -v` passed 14 tests; `python -m unittest tests.test_smart_causal_diffusion -v` passed 42 tests; `py_compile` passed for touched Python files; `git diff --check` passed. A config construction smoke printed `SMARTHybridDiffusion SMARTCausalDiffusion closed_loop_frontier_v1 True 1.25`.
- Next: Train `configs/train/train_scalable_hybrid_diffusion_1000.yaml`, then rerun `scripts/compare_motion_models.py` with the hybrid checkpoint included on the same validation slice.

## 2026-06-15 CST
- Task: Aligned SMART ELF training loss with one-token commit inference.
- Result: `SMARTEmbeddedLanguageFlow` now trains with commit-primary ELF loss: chunk-0 committed tokens use full weight and uncommitted tail proposal chunks use `elf_tail_loss_weight` as an auxiliary loss. The 1000-step and 3-epoch ELF configs keep `prediction_tokens: 4`, `commit_tokens: 1`, `carry_tail_proposal: true`, rolling-anchor training, and model-rollout state exposure, with `elf_tail_loss_weight: 0.25`.
- Validation: Added regressions for commit-primary ELF loss weighting and ELF config invariants. `python -m unittest tests.test_smart_elf -v` passed after the implementation change.

## 2026-06-15 CST
- Task: Fixed two SMART ELF review issues against the original ELF behavior.
- Result: `SMARTEmbeddedLanguageFlow` now computes flow MSE with an embedding-dimension mean before token averaging, matching original ELF loss scaling, and applies current-state context when `current_state_enabled` is true even though the ELF config keeps `ar_objective: maskgit`.
- Validation: Added regressions for ELF flow-loss scaling and current-state context injection. The two focused tests passed after the code change.

## 2026-06-15 CST
- Task: Added a local 3-epoch SMART ELF training config.
- Result: Added `configs/train/train_scalable_elf_3epoch_local.yaml`, which trains `smart_elf` for three full epochs on `/home/anthony/SimAgentJEPA/data/waymo/training_subset_10pct`, disables step-based truncation with `max_steps: -1`, validates once per epoch on eight validation batches, saves epoch checkpoints plus `last.ckpt`, and writes visualizations under `outputs/val_elf_3epoch/` and `outputs/step_elf_3epoch/`.
- Validation: `python -m unittest tests.test_smart_elf.EmbeddedLanguageFlowConfigTest -v` passed; `python -m py_compile tests/test_smart_elf.py` passed; `git diff --check` passed.

## 2026-06-15 CST
- Task: Diagnosed and fixed the SMART ELF visualization issue where visibly moving vehicles appeared to only turn right.
- Result: The issue was not a plotting bug. With the original sampler, the 1000-step ELF checkpoint's visibly moving agents on validation indices 0-3 had `dist>=10m` left/straight/right counts of `0/3/25`, and committed token ids collapsed to 40 unique ids. The root cause was final sampling from the weak auxiliary decoder logits rather than from the integrated ELF embedding state.
- Fix: `SMARTEmbeddedLanguageFlow._diffusion_sample()` now projects final integrated embeddings back to token ids with `_elf_proxy_token_ids()` and uses auxiliary decoder logits only for confidence. Added `EmbeddedLanguageFlowSamplerTest.test_sampling_projects_final_embeddings_instead_of_auxiliary_logits` to prevent regression.
- Evidence: Re-evaluating the same `checkpoints/elf_ar_1000/last.ckpt` with the fixed sampler on indices 0-3 wrote repaired PNGs under `outputs/elf_proxy_fix_diag/`; the diagnostic nearest-neighbor sampler produced `dist>=10m` left/straight/right counts of `32/15/20` and 525 unique committed token ids. The 4-scene repaired comparison summary was ADE/FDE `8.010/18.059` with coverage `1.0`.
- Validation: `python -m unittest tests.test_smart_elf tests.test_compare_motion_models tests.test_smart_causal_flow_matching -v` passed 17 tests; the touched files passed `py_compile`; `git diff --check` passed.

## 2026-06-15 CST
- Task: Added a SMART Embedded Language Flow predictor and ran the matched 1000-step local training job.
- Result: `smart_elf` now reuses the SMART autoregressive rollout shell for committed-state, map, and agent-context refresh, while its inner prediction window uses non-causal full-window embedding flow matching and a final token decoder. Registries and configs now support `Model.predictor: smart_elf`; `configs/train/train_scalable_elf_1000.yaml` completed `max_steps=1000` and wrote `checkpoints/elf_ar_1000/last.ckpt` with `global_step=1000`.
- Metrics: Final logged validation metrics were `val_minADE=3.638`, `val_minFDE=9.644`, `val_ar_window_loss=52.870`, `val_ar_window_mask_acc=0.232`, `val_conflict_rate=0.217`, and `val_interaction_consistency=0.985`. ELF does not log causal-specific `val_rollout_score`, so the ELF 1000-step config now monitors `val_minADE`.
- Outputs: Validation visualization wrote four 1440x1440 PNGs under `outputs/val_elf_1000/smart_elf/epoch_001/`; step visualization wrote four 1440x1440 PNGs under `outputs/step_elf_1000/smart_elf/step_001000/`.
- Validation: `python -m unittest tests.test_smart_elf tests.test_compare_motion_models tests.test_smart_causal_flow_matching -v` passed 16 tests; `python -m py_compile smart/model/smart_elf.py smart/modules/elf_decoder.py tests/test_smart_elf.py tests/test_compare_motion_models.py train.py val.py scripts/compare_motion_models.py` passed; `git diff --check` passed. A real CUDA batch smoke produced finite ELF training loss before the 1000-step run.
- Next: Add `smart_elf` to the matched comparison script invocation and compare it against `ar_baseline`, `ar_frontier`, `causal_diffusion`, and `causal_flow_matching` on the same validation slice before judging quality.

## 2026-06-14 CST
- Task: Fixed misleading causal flow-matching objective diagnostics.
- Result: `smart_causal_flow_matching` no longer averages flow MSE across the 2048 token vocabulary dimension before reducing over supervised tokens; loss now sums per-token velocity-vector error and then averages over frontier tokens. `val_ar_window_mask_acc` also no longer uses the teacher-forced interpolated `flow_state` to decide accuracy; it estimates the final token from `source + predicted_velocity` so a zero-velocity decoder cannot score as correct.
- Finding: The previous flow-matching `val_ar_window_loss`/`val_ar_window_mask_acc` could be misleading: a zero velocity field has MSE near `1 / token_size` under the old reduction, and the old accuracy can be `1.0` because `flow_state` already has the GT token as argmax for any `t > 0`. This explains why later checkpoints can show very low loss/high acc while visual rollout quality is poor.
- Validation: Added regressions in `tests/test_smart_causal_flow_matching.py`; the new tests fail on the old reduction/accuracy and pass after the fix. `python -m unittest tests.test_smart_causal_flow_matching tests.test_smart_causal_diffusion -v` passes.
- Next: Retrain causal flow matching before comparing quality again; old flow-matching window loss/acc values are not comparable with the fixed objective diagnostics.

## 2026-06-14 CST
- Task: Completed the matched 1000-step causal flow-matching local run and four-model comparison.
- Result: `configs/train/train_scalable_causal_flow_matching_1000.yaml` completed `max_steps=1000`, triggered step-1000 validation, and wrote `checkpoints/causal_flow_matching_1000/last.ckpt`. Validation/step visualization outputs were generated under `outputs/val_causal_flow_matching_1000/` and `outputs/step_causal_flow_matching_1000/`.
- Metrics: Final logged flow-matching validation metrics were `val_minADE=2.450`, `val_minFDE=8.410`, `val_ar_window_loss=0.000464`, `val_ar_window_mask_acc=1.000`, `val_rollout_score=19.40`, and `train_loss_epoch=0.158`.
- Comparison: `scripts/compare_motion_models.py` produced `outputs/model_comparison_1000/records.csv`, `summary.csv`, `manifest.json`, and 32 per-model/per-scene PNGs for indices 0-7. The 8-scene summary was: `causal_flow` ADE/FDE `5.4096/11.4135`, `ar_baseline` `5.8834/12.1168`, `causal_diffusion` `6.4491/13.6338`, and `ar_frontier` `7.9828/15.6503`, all with coverage `1.0`.
- Fix: The comparison script now inserts the repo root into `sys.path` when run as `python scripts/compare_motion_models.py`, and `tests/test_compare_motion_models.py` covers `--help` execution from the file path.
- Next: Commit/push the tracked code, config, and docs updates while keeping generated `checkpoints/`, `outputs/`, `cache/`, and Lightning artifacts out of Git.

## 2026-06-14 CST
- Task: Ran the matched 1000-step causal diffusion model locally.
- Result: `configs/train/train_scalable_causal_diffusion_1000.yaml` completed `max_steps=1000` on the same 10% Waymo training subset, triggered the step-1000 validation pass, and wrote `checkpoints/causal_diffusion_1000/last.ckpt`. Validation/step visualization outputs were generated under `outputs/val_causal_diffusion_1000/` and `outputs/step_causal_diffusion_1000/`.
- Metrics: Final logged validation metrics were `val_minADE=3.020`, `val_minFDE=9.990`, `val_ar_window_loss=4.040`, `val_ar_window_mask_acc=0.249`, and `val_rollout_score=24.10`.
- Next: Run the remaining matched 1000-step causal flow matching job, then compare all four checkpoints.

## 2026-06-14 CST
- Task: Ran the matched 1000-step AR causal-frontier diffusion variant locally.
- Result: `configs/train/train_scalable_ar_diffusion_frontier_local.yaml` completed `max_steps=1000` on the same 10% Waymo training subset, triggered the step-1000 validation pass, and wrote `checkpoints/ar_frontier_1000/last.ckpt`. Validation/step visualization outputs were generated under `outputs/val_ar_frontier_1000/` and `outputs/step_ar_frontier_1000/`.
- Metrics: Final logged validation metrics were `val_minADE=4.520`, `val_minFDE=12.70`, `val_ar_window_loss=4.020`, and `val_ar_window_mask_acc=0.187`.
- Next: Run the remaining matched 1000-step jobs for causal diffusion and causal flow matching, then compare all four checkpoints.

## 2026-06-14 CST
- Task: Ran the matched 1000-step AR diffusion baseline locally.
- Result: `configs/train/train_scalable_ar_diffusion_baseline_1000.yaml` completed `max_steps=1000` on the 10% Waymo training subset, triggered the step-1000 validation pass, and wrote `checkpoints/ar_baseline_1000/last.ckpt`. Validation/step visualization outputs were generated under `outputs/val_ar_baseline_1000/` and `outputs/step_ar_baseline_1000/`.
- Metrics: Final logged validation metrics were `val_minADE=2.920`, `val_minFDE=8.670`, `val_ar_window_loss=3.440`, and `val_ar_window_mask_acc=0.290`.
- Next: Run the remaining matched 1000-step jobs for AR frontier, causal diffusion, and causal flow matching, then compare all four checkpoints.

## 2026-06-14 CST
- Task: Fixed the matched 1000-step training configs so local single-GPU runs validate and save checkpoints at the intended step boundary.
- Result: `train.py` now supports config-driven `val_check_interval`, step-based checkpoint cadence, epoch-checkpoint fallback, and `save_last_checkpoint`. The four 1000-step comparison configs now use `strategy: auto`, validate at step 1000, save at step 1000, and write `last.ckpt`, including the causal diffusion and causal flow-matching configs.
- Validation: Added resolver/config regression coverage in `tests/test_train_entrypoint_config.py` and `tests/test_compare_motion_models.py`. A two-step real training smoke with validation and step checkpointing saved `last.ckpt`.
- Next: Let the four 1000-step jobs finish, then run `scripts/compare_motion_models.py` on the resulting `last.ckpt` files.

## 2026-06-14 CST
- Task: Added a SMART-AR causal-frontier v2 training path and matched 1000-step comparison configs.
- Result: `smart_ar_diffusion` now supports config-gated `diffusion.ar_objective: causal_frontier_v1` while keeping `maskgit` as the default-compatible baseline. The new path can swap in `CausalDiffusionDecoder`, train with frontier-only causal supervision, inject current-motion context, build clean/perturbed/model-rollout training views, retokenize targets from closed-loop states, and use differentiable endpoint recovery for invalid retokenized targets. `train.py` now honors `Trainer.max_steps`.
- Files: `smart/model/smart_ar_diffusion.py`, `train.py`, `configs/train/train_scalable_ar_diffusion_baseline_1000.yaml`, `configs/train/train_scalable_ar_diffusion_frontier_local.yaml`, `configs/train/train_scalable_causal_diffusion_1000.yaml`, `configs/train/train_scalable_causal_flow_matching_1000.yaml`, `scripts/compare_motion_models.py`, and focused tests.
- Validation: `python -m unittest tests.test_train_entrypoint_config tests.test_smart_ar_diffusion tests.test_compare_motion_models -v` passed; `python -m unittest tests.test_smart_causal_diffusion tests.test_smart_causal_flow_matching -v` passed; `py_compile` and `git diff --check` passed. A real `data/valid_demo` batch through the AR frontier `training_step` produced finite `loss=16.1374`.
- Blocker: The current local shell cannot run the requested 1000-step GPU training because `nvidia-smi` fails with `GPU access blocked by the operating system`.
- Next: Run the four matched 1000-step configs on a GPU-enabled machine, then compare checkpoints with `scripts/compare_motion_models.py`.

## 2026-06-13 CST
- Task: Added the server training config for causal SMART-token flow matching.
- Result: Added `configs/train/train_scalable_causal_flow_matching.yaml` for the 14-GPU server path with `/raid/haoq_lab/wangshijie/data/waymo/{training,validation}`, `smart_causal_flow_matching`, `flow_matching_v1`, and bounded first-run rollout validation via `flow_integration_steps: 1`.
- Validation: Config loading and model construction should be verified before launch; use `--save_ckpt_path checkpoints/causal_flow_matching` for server runs.
- Next: Submit the flow-matching code/config changes to GitHub, then start the server run from the new config.

## 2026-06-13 CST
- Task: Added a causal SMART-token flow-matching predictor as an additive replacement path for causal diffusion.
- Result: New `smart_causal_flow_matching` predictor reuses the causal receding-horizon sim-agent rollout shell, preserves 4-token proposal / 1-token commit semantics, and replaces the discrete frontier diffusion objective/sampler with token-simplex flow matching and Euler frontier integration. Train/validation entrypoints and the causal smoke/visualization scripts now support the new predictor without deleting `smart_causal_diffusion`.
- Files: `smart/model/smart_causal_flow_matching.py`, `smart/model/__init__.py`, `train.py`, `val.py`, `scripts/smoke_causal_guidance_modes.py`, `configs/train/train_scalable_causal_flow_matching_local.yaml`, `configs/validation/validation_scalable_causal_flow_matching.yaml`, `tests/test_smart_causal_flow_matching.py`
- Validation: Focused flow tests passed; the combined flow/causal smoke/visual test bundle passed 75 tests; `py_compile` passed for touched Python files. A CUDA real-batch loss smoke on `data/valid_demo` produced finite `flow_loss=17.5625` over 192 packed tokens. An untrained CUDA inference smoke wrote `outputs/causal_flow_matching_untrained_smoke/metrics.json`, and a visual smoke wrote `outputs/causal_flow_matching_untrained_visual_smoke/idx_00000_1c83f56236e33b4_guidance_modes.png` (`RGBA`, `1488x930`).
- Next: Train a real flow-matching checkpoint from the new local config before comparing ADE/FDE, speed ratios, or guidance quality against causal diffusion.

## 2026-06-13 CST
- Task: Changed causal diffusion training to use model-rollout states from epoch 0.
- Result: `_closed_loop_curriculum()` now returns `closed_loop_batch_ratio_max` rollout probability immediately, keeping perturb disabled for epochs 0-3 and enabling 25% perturb from epoch 4 onward. Updated schedule docs and the durable project plan/decision notes.
- Evidence: Added/updated `ClosedLoopCurriculumTest.test_curriculum_uses_model_rollout_from_epoch_zero`; the focused test passes.

## 2026-06-13 CST
- Task: Swept epoch-4 causal diffusion inference guidance settings to test whether poor results come from restrictive reranking.
- Result: Added `scripts/sweep_causal_guidance_settings.py` to reuse the existing causal guidance smoke path while changing only inference-time model attributes. Ran two CUDA sweeps on all 11 `data/valid_demo` scenes with `/mnt/d/casual_v2_epoch=04.ckpt`. The core sweep covered `none`, deterministic top-1/no-energy safe, default safe, weak energy, and speed4 variants; the speed-floor sweep covered higher `commit_min_speed_ratio` and no reference-speed decay.
- Finding: The main issue is not the default safety energy alone. `safe_top1_no_energy` already predicts slowly (`moving_speed_ratio=0.6199`) but with much better ADE/FDE than stochastic `none`, so the top-ranked token logits are biased toward conservative trajectories while multinomial `none` reaches higher speed by sampling worse tokens. The best tested setting is `commit_min_speed_ratio=0.75` and `commit_speed_reference_decay=1.0`, giving ADE/FDE `2.2930/5.2324`, pred speed `1.2919`, and moving-speed ratio `0.7485`, versus default safe `2.5538/5.7098`, `1.0400`, and `0.6118`.
- Outputs: `outputs/causal_guidance_epoch04_setting_sweep_core/` and `outputs/causal_guidance_epoch04_setting_sweep_speed_floor/` contain `metrics.json`, `records.csv`, `summary.csv`, `ranked_summary.csv`, and `settings.csv`.
- Validation: `python -m py_compile scripts/sweep_causal_guidance_settings.py` passed before both sweeps.

## 2026-06-13 CST
- Task: Generated whole-scene visual comparisons for `none` versus `safe` guidance on the current epoch-4 causal checkpoint.
- Result: Used `/mnt/d/casual_v2_epoch=04.ckpt` with `missing=0`, `unexpected=0` to render all 11 `data/valid_demo` scenes. Each scene has separate full-scene `none` and `safe` PNGs plus a side-by-side whole-scene comparison PNG. These plots use the official validation visualization path, so they show all current-valid scene agents rather than only the selected guidance target and ego agent.
- Outputs: `outputs/causal_guidance_epoch04_whole_scene_none_safe/per_mode/` contains 22 single-mode PNGs, `outputs/causal_guidance_epoch04_whole_scene_none_safe/compare/` contains 11 `none_vs_safe` PNGs, and the run wrote `manifest.json`, `records.csv`, `summary.csv`, and `summary.json`.
- Validation: Verified PNG counts and dimensions (`per_mode` images are `1440x1440`, compare images are `1890x945`), inspected `idx_00000_1c83f56236e33b4_none_vs_safe_whole_scene.png`, and confirmed `summary.csv` has the expected `none`/`safe` rows.

## 2026-06-13 CST
- Task: Re-ran matched causal speed diagnostics on the actual epoch-4 checkpoint path.
- Result: The exact user-provided `/mnt/d/causal_v2_epoch=04.ckpt` path is not visible in this workspace, but the actual file `/mnt/d/casual_v2_epoch=04.ckpt` exists and loads as `epoch=4`, `global_step=21745`, with `missing=0`, `unexpected=0`. On all 11 `data/valid_demo` scenes, `none` predicts mean speed `1.9128` vs GT `1.7695`, moving-speed ratio `0.7944`, p50 ratio `0.7185`, and pair-weighted ratio `0.8236`. `safe` predicts mean speed `1.0400`, moving-speed ratio `0.6118`, p50 ratio `0.5651`, and pair-weighted ratio `0.5982`.
- Finding: The epoch-4 checkpoint changes the speed diagnosis: `none` is no longer the severe low-speed collapse seen at epoch 2, while `safe` still imposes a clear slowdown. ADE/FDE are better under `safe` (`3.7364/8.9915 -> 2.5538/5.7098`) despite the speed penalty, so further debugging should separate raw token-logit quality in `none` from guidance/rerank conservatism in `safe`.
- Outputs: `outputs/causal_speed_diag_epoch04_none_safe/metrics.json`, `records.csv`, and `summary.csv`.

## 2026-06-13 CST
- Task: Tested whether 4-token causal/diffusion proposals mis-handle chunk 1-3 coordinate anchors or heading when using real SMART tokens.
- Result: Added a regression that loads `smart/tokens/cluster_frame_5_2048.pkl`, selects high-displacement and turning tokens for `veh`, `ped`, and `cyc`, and checks five start headings (`0`, `30`, `90`, `-90`, and `180` degrees). `_token_chunk_world()` matches a manual query-relative decode, and `_refresh_token_geometry()` matches all-known and proposal-carried query anchors/headings. With only chunk 0 known, masked tails correctly fall back to the latest known pose/heading.
- Finding: The coordinate transform itself is not keeping chunk 1-3 in the first token's frame, and heading rotation is consistent for real token corners. The remaining mismatch risk is architectural: early masked tail chunks only have fallback/proposal geometry until they are sampled/committed, whereas original SMART updates query geometry, token embeddings, and motion features in a strict one-token recurrent loop.
- Validation: `python -m unittest tests.test_smart_diffusion_smart_parity.SMARTDiffusionSMARTParityTest.test_real_token_four_chunk_geometry_and_heading_are_query_relative` passed. The related four-test bundle covering endpoint, proposal confidence, real-token heading, and rollout visualization query refresh also passed.

## 2026-06-13 CST
- Task: Diagnosed causal v2 epoch-2 speed collapse with matched `none` versus `safe` guidance on all 11 `data/valid_demo` scenes.
- Result: `/mnt/d/causal_v2_epoch=02.ckpt` loaded with `missing=0`, `unexpected=0`. `none` already predicts slow trajectories: mean pred speed `0.9240` vs GT `1.7695`, moving-speed ratio `0.5518` and pair-weighted ratio `0.5486`. `safe` is only slightly slower: mean pred speed `0.8665`, moving-speed ratio `0.5271` and pair-weighted ratio `0.5153`.
- Finding: The primary speed issue is present without safety reranking, so the root cause is likely training distribution/token logits rather than `safe` energy alone. `safe` improves ADE/FDE on this smoke (`3.1711/6.8125 -> 2.8586/6.1969`) while adding a small additional speed penalty.
- Outputs: `outputs/causal_speed_diag_epoch02_none_safe/metrics.json`, `records.csv`, and `summary.csv`.

## 2026-06-12 CST
- Task: Fixed three causal diffusion review issues in sampler and guidance metrics.
- Result: Seed-locked ego/edit sampling now preserves global chunk timestep semantics, so editable later chunks decode at their own causal frontier `t`; no-finite ego TTC diagnostics now remain infinite instead of becoming zero; smoke and visualization token-change rates now divide by valid token count only.
- Evidence: Added regression coverage for seed-locked chunk-2 timestep, direct and sampled infinite `ego_risk_min_ttc`, and valid-token edit-rate denominator. `tests.test_smart_causal_diffusion`, `tests.test_trajectory_energy`, `tests.test_causal_guidance_smoke`, and `tests.test_visualize_causal_guidance_modes` pass 77 tests; touched files pass `py_compile` and `git diff --check`.
- Note: Existing smoke/visual outputs that compare `ego_risk_min_ttc` or `token_change_rate_vs_gt` should be regenerated before drawing conclusions.

## 2026-06-12 CST
- Task: Implemented a hybrid SMART-style commit speed constraint for causal diffusion.
- Result: `smart_causal_diffusion` now keeps diffusion tail proposals for editing but reranks executable chunk-0 candidates with a speed-reference energy. The reference starts from observed history speed and decays through the AR loop, so a slowed generated history cannot immediately erase the speed floor.
- Evidence: Added unit coverage for static-token penalty, token-internal speed instead of endpoint-only speed, and reference-speed use after history slowdown. `tests.test_smart_causal_diffusion` passes 39 tests; `py_compile` and `git diff --check` pass. On all 11 `data/valid_demo` scenes with `/mnt/d/causal_v2_epoch=02.ckpt`, `safe` improved versus the endpoint-only attempt: ADE `2.6516 -> 2.5450`, FDE `5.8995 -> 5.5614`, pred speed `0.7893 -> 0.8537`, moving speed ratio p10 `0.0316 -> 0.0839`, and p50 `0.4498 -> 0.4921`.
- Note: The improvement is measurable but not a full fix; `safe` is still slower than `none` (`pred_speed 0.8537` vs `0.9505`) and offroad/hard-collision guidance metrics are effectively unchanged on this smoke.

## 2026-06-12 CST
- Task: Fixed causal guidance smoke FDE and added speed-collapse diagnostics.
- Result: `scripts/smoke_causal_guidance_modes.py` now computes FDE from the last actual valid frame instead of `valid.sum()-1`, preventing invalid gaps with zero GT coordinates from inflating FDE. The smoke records `pred_speed`, `gt_speed`, moving-frame speed ratios, and moving-pair counts for checkpoint comparisons.
- Evidence: Added regression tests for non-contiguous valid masks and moving speed ratios in `tests/test_causal_guidance_smoke.py`. On Waymo validation scene 72 with `/mnt/d/causal_v2_epoch=02.ckpt`, fixed `safe` FDE is `12.95m` instead of the earlier invalid-gap artifact near `684m`; `safe` moving speed ratio median is `0.0355`, confirming severe low-speed collapse.
- Note: The checkpoint metadata says epoch 1 was best (`val_rollout_score=22.7363`), but the referenced causal v2 epoch-1 path is server-local and not available in this workspace. Local `/mnt/d/90113f8*epoch=01.ckpt` files are AR diffusion, not causal v2.

## 2026-06-11 CST
- Task: Screened real Waymo validation scenes for visually clear generic ego-risk edits.
- Result: Ran ego-risk numeric sweeps over the first 100 scenes from `/home/anthony/SimAgentJEPA/data/waymo/validation` in two 50-scene chunks, merged the results, and ranked candidates by clean invalidity, ego-risk success, near-miss success, risk reward, distance, edit size, and dynamics energy.
- Outputs: Combined records are in `outputs/causal_ego_risk_waymo_val_sweep_chunks/records_000_099.csv`; ranked candidates are in `outputs/causal_ego_risk_waymo_val_sweep_chunks/ranked_candidates_000_099.csv`; visual comparisons are in `outputs/causal_ego_risk_waymo_val_visual_top6/` and `outputs/causal_ego_risk_waymo_val_visual_top4_extra/`.
- Finding: The best visual candidates so far are scene indices 72, 87, 31, 17, 41, and 33. Scene 20 was the strongest numeric candidate but not visually obvious enough for a main figure.
- Documentation: Added `docs/ego_risk_waymo_validation_candidates.md` with figure paths, metric snapshots, and rejected high-metric examples.

## 2026-06-11 CST
- Task: Replaced predefined cut-in guidance with generic ego-risk scene editing.
- Result: `smart_causal_diffusion` now defaults to `target_spec: ego_risk` for `ego_stress` and `ego_edit`. Guidance scores use `ego_risk_reward` instead of a predefined target-event score, and diagnostics/logging include `ego_risk_min_ttc`, `ego_risk_reward`, and `ego_risk_success_rate`. Cut-in and lead-hard-brake remain as legacy ablation target specs.
- Scripts: `scripts/smoke_causal_guidance_modes.py` and `scripts/visualize_causal_guidance_modes.py` default to `ego_risk`; Pareto criticality no longer depends on target-event success, and the visualization no longer draws a cut-in corridor-entry marker unless a legacy cut-in spec is explicitly requested.
- Evidence: A generic ego-risk smoke over all 11 `data/valid_demo` scenes wrote `outputs/causal_ego_risk_guidance/smoke_metrics.json`. A visual ego-risk comparison for scene index 10 wrote `outputs/causal_ego_risk_guidance_visual/idx_00010_1ce0b4bbd35a6ad1_guidance_modes.png` and records nonzero `guidance_ego_risk_success_rate` with zero hard-collision/offroad in the selected ego modes.
- Validation: 67 focused tests passed; `py_compile` passed for the touched Python files; `git diff --check` passed; the generated ego-risk comparison PNG verifies as RGBA `3720x930`.
- Next: Treat this as a smoke artifact, not paper-scale evidence. Sweep more scenes and improve target-agent/window selection around ego interactions before choosing final AAAI figures.

## 2026-06-11 CST
- Task: Improved ego-centric cut-in visualization for causal guidance.
- Result: `scripts/visualize_causal_guidance_modes.py` now explicitly draws the ego/SDC current marker, ego history/future, ego corridor band, controlled target marker, controlled target future markers, target-ego closest relation, and first ego-corridor entry marker. Plot extents now include both ego and controlled target paths.
- Evidence: Added focused visualization helper tests and regenerated marked cut-in comparisons. The clearer visual artifact is `outputs/causal_ego_guidance_cut_in_scene9_agent4_marked/idx_00009_1d3daf744e65dd7c_guidance_modes.png`; ego modes have nonzero target-event success (`0.0208`) with zero hard collision/offroad. The older scene-3 artifact remains stronger numerically (`0.1146`) but uses a visually static default target.
- Next: For paper figures, prefer geometry-visible target/window choices like scene 9 agent 4 over the default target selector, and continue scanning beyond the 11 demo scenes for stronger visible cut-in cases.

## 2026-06-11 CST
- Task: Replaced global stress/edit guidance with ego-centric interaction guidance.
- Result: `smart_causal_diffusion` now supports `guidance.mode = none | safe | ego_stress | ego_edit`, packs ego/SDC reference trajectory, heading, and optional route corridor into candidate-token reranking, scores ego modes with ego interaction reward plus target-event reward minus invalid/edit penalties, and logs ego-only metrics including ego min distance/TTC, required decel, path intrusion, conflict TTA error, target-event success, hard collision, offroad, dynamics, and edit distance.
- Controls: Ego and non-target agents are seed-locked by default for ego modes at both token and committed raw-trajectory levels; only target agents inside the configured target token window remain editable. The smoke target selector now excludes `av_index`/`ego_agent_id` by default.
- Scripts: `scripts/smoke_causal_guidance_modes.py` and `scripts/visualize_causal_guidance_modes.py` default to `seed,none,safe,ego_stress,ego_edit`, expose runtime overrides for target spec and guidance weights, and use ego near-miss plus target-event success as Pareto criticality.
- Evidence: 58 focused tests passed; `py_compile` and `git diff --check` passed. CUDA sweeps over all 11 `data/valid_demo` scenes loaded the checkpoint with `missing=0`, `unexpected=0`.
- Finding: A cut-in sweep found a successful ego-centric counterfactual on demo scene index 3 (`target_event_success_rate=0.1146`, `non_target_preservation_ADE=0`, `token_change_rate_vs_gt=0.0051`) and wrote a visual comparison under `outputs/causal_ego_guidance_cut_in_success/`. A lead-hard-brake sweep did not trigger target-event success on the 11 demo scenes.

## 2026-06-11 CST
- Task: Added causal guidance visual comparison examples.
- Result: Added `scripts/visualize_causal_guidance_modes.py`, which reruns selected scenes for `seed`, `none`, `safe`, `stress`, and `edit`, plots one multi-panel PNG per scene, and writes `manifest.json`, `records.csv`, `summary.csv`, and `summary.json`.
- Evidence: Generated three checkpoint examples for demo scene indices `[0, 1, 2]` under `outputs/causal_guidance_visual_examples/`. Each PNG is valid RGBA image data at `3720x930`; the manifest records checkpoint load with `missing=0` and `unexpected=0`.
- Validation: `tests.test_visualize_causal_guidance_modes` passes 8 tests, the combined script test set passes 15 tests, and `py_compile` passes for the new script and tests.
- Finding: The visual examples match the scalar sweep finding: `edit` stays close to the seed/GT with very small edit distance, while `stress`/`safe` change more trajectory tokens but still show zero near-miss success on these three demo scenes.
- Next: Inspect the three PNGs, then run broader scene/window/alpha sweeps and regenerate visuals for cases with nonzero criticality.

## 2026-06-11 CST
- Task: Extended the causal guidance smoke into a multi-scene Pareto sweep.
- Result: `scripts/smoke_causal_guidance_modes.py` now supports `--indices` and `--num-scenes`, writes per-record CSV, per-mode summary CSV, JSON summaries, and generation-mode Pareto candidates while keeping `seed` as a baseline rather than a Pareto candidate.
- Metrics: The sweep records ADE/FDE, token change rate, `min_ttc`, `near_miss_rate`, `hard_collision_rate`, `offroad_rate`, `dynamics_energy`, `edit_distance`, `non_target_preservation_ADE`, and `target_success_rate`.
- Evidence: A CUDA sweep over demo scene indices `[0, 1, 2]` with `checkpoints/causal_diffusion/epoch=00.ckpt` completed 15 runs (`seed`, `none`, `safe`, `stress`, `edit` per scene) and wrote `outputs/causal_guidance_sweep/metrics.json`, `records.csv`, and `summary.csv`.
- Finding: On these three demo scenes, the current checkpoint did not produce collision-free near-miss successes (`near_miss_rate=0` and `target_success_rate=0` for all modes); `edit` dominates the generation-mode Pareto set because it stays closest to the GT seed with zero invalidity under the tested target window.
- Next: Sweep more scenes and target windows, then visualize selected stress/edit cases to inspect whether stronger `stress_alpha` or broader edit windows can create valid critical interactions.

## 2026-06-11 CST
- Task: Added and ran a checkpoint smoke for causal guidance modes.
- Result: Added `scripts/smoke_causal_guidance_modes.py` to compare `seed`, `none`, `safe`, `stress`, and `edit` on the same demo scene/checkpoint. The script attaches GT seed tokens/trajectories for edit mode, selects a target agent, supports a target token window, and writes JSON metrics.
- Fixes: Real smoke exposed two guidance bugs: seed trajectory packing did not support extra `[token_steps, 2]` dimensions, and guidance diagnostics aliased the same zero tensor across metric keys. A third semantic issue made criticality compare candidates against the same agent's nominal trajectory; criticality now uses same-scene other-agent nominal trajectories and excludes self-agent references.
- Evidence: `python scripts/smoke_causal_guidance_modes.py --config configs/train/train_scalable_causal_diffusion_local.yaml --ckpt checkpoints/causal_diffusion/epoch=00.ckpt --raw-dir data/valid_demo --index 0 --output outputs/causal_guidance_smoke/metrics.json` completed on CUDA with zero missing/unexpected checkpoint keys. The final JSON includes all five modes for scenario `1c83f56236e33b4`.
- Validation: `tests.test_trajectory_energy` and `tests.test_smart_causal_diffusion` pass 34 tests; `py_compile` and `git diff --check` pass for touched code and smoke script.
- Next: Run the smoke over multiple scenes and target windows, then aggregate realism-criticality-minimality Pareto tables and visualizations.

## 2026-06-11 CST
- Task: Added first-pass inference-time guidance modes for `smart_causal_diffusion`.
- Result: Implemented `guidance.mode = none | safe | stress | edit`, mode-specific top-k config, stress/edit scoring with collision-free near-miss/low-TTC criticality reward, invalid trajectory penalties, edit-distance penalties, seed-token locking through `editable_mask`, target-agent/time-window edit controls, optional `seed_trajs` edit distance, and validation logging for guidance metrics.
- Files: `smart/model/smart_causal_diffusion.py`, `smart/model/smart_ar_diffusion.py`, `smart/modules/trajectory_energy.py`, causal configs, and focused causal/energy tests.
- Validation: `tests.test_trajectory_energy` and `tests.test_smart_causal_diffusion` pass 30 tests; `py_compile` passes for touched Python files.
- Next: Run a checkpoint inference smoke for `none`, `safe`, `stress`, and `edit` on identical scenes, then inspect the realism-criticality-minimality Pareto metrics and visualizations.

## 2026-06-11 CST
- Task: Saved an AAAI-style Chinese Method draft for causal diffusion.
- Result: Added `docs/aaai_causal_diffusion_method_zh.md` explaining Causal SMART-Token Diffusion in paper-style sections: overview, token representation, causal decoder, discrete frontier objective, closed-loop retokenization, receding-horizon sampling, safety reranking, and validation.
- Next: Use this as the Chinese architecture draft before converting the method section into final English AAAI prose.

## 2026-06-11 CST
- Task: Wrote a current Chinese architecture/function overview for `smart_causal_diffusion`.
- Result: Added `docs/smart_causal_diffusion_overview.md` covering the implementation structure, inheritance, input packing, causal decoder, discrete frontier v2 loss, closed-loop training curriculum, retokenization, sampling, proposal carry, safety-energy reranking, validation metrics, configs, and current limitations.
- Next: Use this overview with `docs/train_scalable_causal_diffusion_config.md` when launching or debugging causal v2 training.

## 2026-06-11 CST
- Task: Implemented causal diffusion v2 as a revisable four-token receding-horizon planner.
- Result: Replaced continuous-time weighted suffix loss with uniform discrete frontier CE; enabled tail proposal carry and confidence-weighted proposal embeddings; added recency/current-motion context and all-mask chunk-0 interaction sources; made commit safety active at the first sampling step; and extended dynamics energy across the observed-to-candidate transition.
- Config: Causal configs now use epoch LR schedules (`2/32` server, `1/5` local) and the calibrated P99 thresholds `[0.7379697561, 0.8562850952, 1.2705252171]`.
- Local workflow: `train_scalable_causal_diffusion_local.yaml` now trains directly on the 11 `data/valid_demo` scenes, limits rollout validation to one batch, and disables extra visualization by default.
- Validation: 21 causal tests pass. A three-step RTX 4090 smoke on all 11 `valid_demo` scenes produced finite losses `16.3 -> 15.1 -> 14.3`; a complete 16-round, 80-frame rollout finished in 29.1 seconds with finite outputs and wrote `outputs/causal_v2_smoke/idx_00000.png`.
- Note: The three-step visualization is an execution smoke, not a trajectory-quality result. Causal v2 must be trained from scratch.

## 2026-06-11 CST
- Task: Diagnosed and fixed epoch-0 causal diffusion token mode collapse.
- Result: The sampler now requires one decoding step per causal frontier. Four-token windows decode at `t=[1.0, 0.75, 0.5, 0.25]`; schedules such as 16 steps for 4 chunks fail fast instead of idling for 12 iterations and releasing only at low noise.
- Evidence: `/mnt/d/causal_epoch=00.ckpt` had completed 4349 optimizer steps, but three demo scenes used only 36/2048 predicted tokens, repeated adjacent tokens 71.8% of the time, and concentrated 95.5% of predictions in ten tokens. Changing the sampling schedule from 16 to 4 steps reduced mean ADE/FDE from 10.24/24.27 m to 6.89/16.08 m on those scenes.
- Files: `smart/model/smart_causal_diffusion.py`, causal train/local/validation configs, `tests/test_smart_causal_diffusion.py`.
- Validation: All 17 causal tests passed; `py_compile` and `git diff --check` passed. The corrected validation config loaded the epoch-0 checkpoint with zero missing/unexpected keys and produced ADE 2.045 m / FDE 4.826 m on demo scene 1.
- Next: Revalidate the existing epoch-0 checkpoint with the corrected config, then resume or restart training while comparing identical scenes and token-diversity diagnostics.

## 2026-06-10 CST
- Task: Documented every field in `configs/train/train_scalable_causal_diffusion.yaml`.
- Result: Added `docs/train_scalable_causal_diffusion_config.md` with code-traced runtime behavior, units, tuning effects, inactive/overridden fields, formulas, constraints, and server recommendations.
- Finding: Several compatibility fields are not active in the causal path, including YAML `Trainer.ckpt_path`, `eval_inference_batches`, and causal `geometry_dropout_prob`; visible corruption remains disabled, while proposal carry is enabled in v2.
- Next: Use the reference when freezing the server config and record the calibrated P99 retokenization thresholds.

## 2026-06-10 CST
- Task: Added a server-side `smart_causal_diffusion` runbook to `README.md`.
- Result: Documented branch/environment checks, epoch-based LR configuration, dataset validation, retokenization calibration, from-scratch DDP launch, monitoring, checkpoint resume, and standalone validation commands.
- Next: Apply server-specific data paths and GPU counts, calibrate full-dataset P99 thresholds, then launch the first clean causal run.

## 2026-06-10 CST
- Task: Ran a real local CUDA training smoke for `smart_causal_diffusion` on the 11 Waymo demo scenes under `data/valid_demo`.
- Result: PyTorch Lightning completed one epoch with three optimizer steps and one full 80-frame validation rollout on an RTX 4090. Decoder parameters changed (`max delta 3.0e-6`), train/validation losses were finite, prediction coverage was 1.0, and all causal rollout metrics were emitted.
- Validation: `global_step=3`, elapsed 49.36 seconds, `train_loss_epoch=20.0784`, `val_ar_window_loss=44.8941`, `val_rollout_score=60.6912`. The untrained model's trajectory metrics are intentionally poor and are not a quality estimate.
- Performance: Full 16-round inference took about 9.2 seconds. Existing `ConflictRate` and `InteractionConsistency` metrics added about 12.0 and 21.6 seconds respectively; the new causal rollout metrics added about 1.5 seconds.
- Next: The training path is operational for server upload. Recalibrate retokenization thresholds on the full training set, then start the from-scratch run.

## 2026-06-10 CST
- Task: Implemented the independent SMART causal absorbing diffusion redesign.
- Result: Added `smart_causal_diffusion` with strictly causal temporal token edges, geometric prefix corruption, chunk-correct absorbing MDLM weights, monotonic frontier reveal, four-token/one-token receding-horizon rollout, and separate train/validation/official-eval registration.
- Files: `smart/model/smart_causal_diffusion.py`, `smart/modules/causal_diffusion_decoder.py`, model and entrypoint registries, causal train/validation configs.
- Validation: New causal/config tests pass; all three configs instantiate with four chunks and encoder/decoder LR ratio 0.5; real Waymo window training and full 16-round inference smokes produced finite outputs with full prediction coverage.

- Task: Added closed-loop curriculum, retokenization recovery, and calibration.
- Result: Epoch curriculum now progresses from clean anchors to correlated perturbations and 1-4-token model-rollout states. GT continuations are retokenized from the resulting anchor; threshold-invalid targets use differentiable expected-endpoint recovery. Added a CLI for per-type empirical threshold calibration.
- Files: `smart/model/smart_causal_diffusion.py`, `scripts/calibrate_causal_retokenization.py`, `tests/test_smart_causal_diffusion.py`.
- Validation: A real three-token rollout-state training smoke built a `(73,31,2)` view and finite loss. A 50-scene perturbed calibration, filtered to current-valid category-3 targets, produced P99 seeds `[0.65, 0.78, 0.62]`.

- Task: Added soft safety-energy reranking and rollout metrics.
- Result: Executable frontier candidates are top-k reranked by lane distance, lane heading, dynamics, and collision energies with `(1-t)^2` guidance. Validation logs 2/4/6/8-second ADE/FDE, final-four-second ADE, energy terms, coverage, retokenization-invalid rate, and `val_rollout_score`.
- Files: `smart/modules/trajectory_energy.py`, `smart/model/smart_causal_diffusion.py`, `smart/model/smart_ar_diffusion.py`, `tests/test_trajectory_energy.py`.
- Validation: Energy ordering tests pass; a full CPU Waymo inference completed 16 rounds in about 28.5 seconds with `pred_traj=(73,80,2)`, finite outputs, coverage 1.0, and aggregated safety energies.

- Correction: Lightning steps the bare `LambdaLR` once per epoch in this training path. Causal configs now intentionally use epoch units; the separate user-modified AR config remains outside this task.
- Regression: 55 targeted/new/existing tests ran with 52 passing, 1 Waymo-dependency skip, and the same 2 pre-existing AR-config drift failures (`causal_noise_schedule` and visible corruption expectations). The 34-test regression subset excluding those stale config assertions passed.

## 2026-06-08 CST
- Task: Aligned diffusion history-context map-to-agent mask with official SMART forward semantics.
- Result: `SMARTAgentDecoder.encode_history_context()` now masks map-to-agent attention by `category == 3`, matching official `SMARTAgentDecoder.forward()` instead of using `type != 3`. Added a focused regression test that first failed under the old type-based mask and now passes.
- Files: `smart/modules/agent_decoder.py`, `tests/test_agent_decoder_history_context.py`
- Validation: `tests.test_agent_decoder_history_context` passed after failing on the old behavior; `tests.test_smart_diffusion_smart_parity` passed 12 tests with 1 Waymo-dependency skip; `py_compile` and `git diff --check` passed for touched files. `tests.test_smart_ar_diffusion` has one existing config-drift failure in `configs/validation/validation_scalable_ar_diffusion.yaml`, unrelated to this code change.

- Task: Added raw/proposal/sampled query visualization for AR rollout input debugging.
- Result: `scripts/visualize_ar_map_rollout.py` now draws raw packed query nodes, proposal-refreshed query nodes produced through `_refresh_token_geometry()` with carried tail proposals, and sampled query nodes after diffusion sampling. Metadata now records raw/proposal/sampled map-edge counts plus proposal and sampled geometry confidence per round. New query-debug PNGs were generated under `outputs/ar_map_rollout_query_debug_6c1658d_epoch00/`.
- Files: `scripts/visualize_ar_map_rollout.py`, `tests/test_visualize_ar_map_rollout.py`, `outputs/ar_map_rollout_query_debug_6c1658d_epoch00/*.png`, `outputs/ar_map_rollout_query_debug_6c1658d_epoch00/metadata.json`
- Validation: `tests.test_visualize_ar_map_rollout` first failed on missing `proposal_query_positions`, then passed 2 tests; `python3 -m py_compile scripts/visualize_ar_map_rollout.py tests/test_visualize_ar_map_rollout.py` passed; `git diff --check` passed for the script and test; the checkpoint visualization command wrote three valid 2880x2760 PNGs.
- Next: Use the query-debug images to verify whether raw q overlap is expected while proposal q expands from round 2 onward; run with a config matching checkpoint training settings when diagnosing `prediction_tokens: 6` checkpoints.

## 2026-06-08 CST
- Task: Upgraded AR rollout map-token visualization to inspect true recurrent inputs.
- Result: `scripts/visualize_ar_map_rollout.py` now records sampling-before input snapshots per round, including rolled selected history, rolled current generation-agent positions, packed full-scene map tokens, input query chunk positions, input map-to-token edges, sampled map-to-token edges, and carried proposal tokens. New input-debug PNGs were generated for three fixed validation scenes under `outputs/ar_map_rollout_input_debug_6c1658d_epoch00/`.
- Files: `scripts/visualize_ar_map_rollout.py`, `tests/test_visualize_ar_map_rollout.py`, `outputs/ar_map_rollout_input_debug_6c1658d_epoch00/*.png`, `outputs/ar_map_rollout_input_debug_6c1658d_epoch00/metadata.json`
- Validation: `tests.test_visualize_ar_map_rollout` passed after first failing on the missing snapshot helper; `python3 -m py_compile scripts/visualize_ar_map_rollout.py tests/test_visualize_ar_map_rollout.py` passed; the checkpoint visualization command wrote three valid 2880x2760 PNGs and metadata with input/sampled map-edge counts, input agent counts, and proposal tokens.
- Next: Inspect the input-debug panels first; if q0/rolled selected current/rolled agents are correct and map input edges are nonempty, focus the next investigation on low-diversity token sampling and early-training underfit rather than missing AR map or agent inputs.

## 2026-06-08 CST
- Task: Generated AR rollout map-token diagnostics for checkpoint `/mnt/d/6c1658d_epoch=00.ckpt`.
- Result: Three validation scenes were rolled out with `configs/validation/validation_scalable_ar_diffusion.yaml`; each output image shows one category-3 vehicle across 16 recurrent steps with current/nearby agents, packed full-scene map tokens, per-round connected map tokens, GT future, and committed prediction.
- Files: `scripts/visualize_ar_map_rollout.py`, `outputs/ar_map_rollout_debug_6c1658d_epoch00/*.png`, `outputs/ar_map_rollout_debug_6c1658d_epoch00/metadata.json`
- Validation: `python3 -m py_compile scripts/visualize_ar_map_rollout.py` passed; the script loaded the checkpoint on CUDA and wrote three valid 2880x2760 PNGs plus metadata with per-round token ids and map-edge counts.
- Next: Inspect the generated images and metadata to decide whether the epoch-0 checkpoint failure mode is map-context coverage, repeated low-diversity token choices, or normal early-training underfit.

## 2026-06-07 CST
- Task: Aligned AR diffusion map context rescreening with original SMART map-edge semantics.
- Result: smart_ar_diffusion now keeps all visible map tokens for each packed scene under local_map_refresh=rescreen, so DiffusionDecoder rebuilds map-to-token radius edges from the full scene map feature set instead of a current-pose local prefilter.
- Files: smart/model/smart_ar_diffusion.py, tests/test_smart_ar_diffusion.py
- Validation: python3 -m py_compile smart/model/smart_ar_diffusion.py tests/test_smart_ar_diffusion.py passed; git diff --check passed; in the smart conda environment, tests.test_smart_ar_diffusion passed 15 tests, tests.test_smart_diffusion_prefix passed 8 tests, and tests.test_smart_diffusion_smart_parity passed 12 tests with 1 Waymo-dependency skip.
- Next: Run AR validation in the smart conda environment and compare late-horizon ADE/FDE, map violations, throughput, and map-token memory against the previous local-prefilter run.

## 2026-06-07 CST
- Task: Generated visual proof artifacts for SMART-style AR diffusion map rescreening.
- Result: Four PNG diagnostics under outputs/ar_map_context_proof_20260607 compare old current-pose local prefiltering against the new full-scene map candidate behavior and show dynamic radius edges at current and future token poses.
- Files: outputs/ar_map_context_proof_20260607/*.png, outputs/ar_map_context_proof_20260607/proof_metadata.json
- Validation: PNG files were created successfully with valid image headers; proof_metadata.json records old_prefilter_candidates=[0], new_packed_candidates=[0,1,2,3], old_edges_future=[], and new_edges_future=[2].
- Next: Use these diagnostics alongside validation visualizations when comparing late-horizon AR diffusion behavior.

## 2026-05-28 CST
- Task: Added visible-token neighbor corruption for SMART AR diffusion training.
- Result: Diffusion training can now replace unmasked visible future tokens with same-type top-k nearest SMART trajectory-token neighbors while preserving GT labels for loss. AR train configs enable `visible_token_corruption_prob: 0.15` and `visible_token_corruption_topk: 5`; validation sets corruption probability to `0.0`.
- Files: `smart/model/smart_diffusion.py`, `tests/test_smart_diffusion_prefix.py`, `tests/test_smart_ar_diffusion.py`, `configs/train/train_scalable_ar_diffusion*.yaml`, `configs/validation/validation_scalable_ar_diffusion.yaml`
- Validation: Targeted visible-token corruption tests passed; `py_compile`, diffusion prefix tests, AR diffusion tests, SMART diffusion parity tests, and `git diff --check` passed with the existing Waymo official dependency skip.
- Next: Finetune from the previous AR diffusion checkpoint with receding-horizon proposal carry plus visible-token neighbor corruption, then compare straight-vehicle speed and map-violation visualizations.

## 2026-05-28 CST
- Task: Reverted AR diffusion default causal schedules to a disabled ablation and added multiplier-based causal mask support.
- Result: AR configs now keep `causal_noise_schedule: false` by default, retain receding-horizon `commit_tokens: 1` plus proposal carry, and expose `causal_chunk_mask_multipliers` for experiments that preserve `mask_prob(t)` semantics. Legacy fixed `causal_chunk_mask_probs` remains supported only as fallback behavior.
- Files: `smart/model/smart_diffusion.py`, `tests/test_smart_diffusion_prefix.py`, `tests/test_smart_ar_diffusion.py`, `configs/train/train_scalable_ar_diffusion*.yaml`, `configs/validation/validation_scalable_ar_diffusion.yaml`
- Validation: Targeted TDD tests passed; `py_compile`, diffusion prefix tests, AR diffusion tests, SMART diffusion parity tests, and `git diff --check` passed with the existing Waymo official dependency skip.
- Next: Re-run visualization with the old checkpoint under receding-horizon proposal carry only; compare against the fixed causal schedule run before trying visible-token neighbor corruption.

## 2026-05-28 CST
- Task: Added discrete Diffusion Forcing-style causal schedules for AR diffusion training.
- Result: SMART diffusion loss can now use per-chunk mask probabilities and loss weights; AR diffusion configs enable `[0.20, 0.45, 0.70, 0.90]` mask probabilities and `[1.0, 0.8, 0.4, 0.2]` loss weights so near tokens are trained as executable and far tokens as softer proposals.
- Files: `smart/model/smart_diffusion.py`, `tests/test_smart_diffusion_prefix.py`, `tests/test_smart_ar_diffusion.py`, `configs/train/train_scalable_ar_diffusion*.yaml`, `configs/validation/validation_scalable_ar_diffusion.yaml`
- Validation: Causal schedule tests passed; AR diffusion tests passed; diffusion prefix tests passed; SMART diffusion parity tests passed with the existing Waymo official dependency skip; `py_compile` passed.
- Next: Run AR validation from the old checkpoint with receding-horizon plus causal schedules, then compare straight-vehicle predicted-token speed percentiles against the previous schedule.

## 2026-05-28 CST
- Task: Added receding-horizon proposal carry for SMART AR diffusion.
- Result: AR rollout can now predict a 4-token window, commit 1 token per round, and carry the uncommitted tail as proposal geometry/confidence for the next diffusion window; history token pose/heading rolling now supports `commit_tokens < history_tokens`. AR train/validation configs enable `commit_tokens: 1` with `carry_tail_proposal: true`.
- Files: `smart/model/smart_ar_diffusion.py`, `smart/model/smart_diffusion.py`, `tests/test_smart_ar_diffusion.py`, `configs/train/train_scalable_ar_diffusion*.yaml`, `configs/validation/validation_scalable_ar_diffusion.yaml`
- Validation: AR diffusion unit tests passed; diffusion prefix tests passed; SMART diffusion parity tests passed with the existing Waymo official dependency skip; `py_compile` and `git diff --check` passed.
- Next: Run AR validation visualization and compare straight-vehicle token speed percentiles for `commit_tokens=1 + carry_tail_proposal` against the previous 2-token commit checkpoint/code path.

## 2026-05-22 CST
- Task: Added SMART autoregressive discrete diffusion.
- Result: New `smart_ar_diffusion` predictor reuses SMART discrete trajectory tokens and the existing diffusion decoder with 2-token history, 4-token prediction, 2-token commit, 8-round/80-step rollout, rolling-anchor short-window training, synthetic history-state perturbation, and per-round local map token rescreening. Dedicated train/validation configs and unit tests were added.
- Files: `smart/model/smart_ar_diffusion.py`, `smart/model/__init__.py`, `train.py`, `val.py`, `eval_waymo_official.py`, `tests/test_smart_ar_diffusion.py`, `configs/train/train_scalable_ar_diffusion*.yaml`, `configs/validation/validation_scalable_ar_diffusion.yaml`
- Validation: `py_compile` passed for AR diffusion and entrypoints; all AR configs instantiate; AR diffusion unit tests passed; SMART parity and prefix regression tests passed.
- Next: Run a real single-batch AR train smoke and validation inference smoke, then compare `smart_ar_diffusion` against full-horizon `smart_diffusion` on map violations, collisions, and sim-agent export coverage.

## 2026-05-22 CST
- Task: Implemented SMART-Diffusion parity and uncertainty-aware joint diffusion.
- Result: Diffusion rollout now packs all SMART history-valid generation agents, keeps `loss_mask_base` limited to category-3 supervision agents, returns SMART-compatible metric `valid_mask` while preserving `pred_valid_mask` generation coverage, decodes tokens with SMART endpoint-token composition, and uses proposal geometry/confidence for masked chunks without turning sampling into prefix AR. Official Waymo export now rejects missing, non-finite, invalid-mask, and zero-fallback sim-agent predictions.
- Files: `smart/model/smart_diffusion.py`, `smart/modules/diffusion_decoder.py`, `eval_waymo_official.py`, `tests/test_smart_diffusion_smart_parity.py`, diffusion train/validation configs, `scripts/visualize_diffusion_map_tokens.py`
- Validation: `py_compile` passed for edited Python files; SMART parity unit tests passed with Waymo dependency skip; prefix ablation tests passed; a tiny `DiffusionDecoder` forward smoke passed with geometry confidence/source masks.
- Next: Run real single-batch train and validation inference smoke on local Waymo data, then compare current checkpoint/current code, mask-parity only, and mask-parity plus uncertainty-aware geometry.

## 2026-05-20 CST
- Task: Added training-time self-conditioning to SMART-Diffusion.
- Result: Diffusion loss can now run a no-grad first pass on selected visible tokens, feed argmax predictions back into the final denoising pass, and supervise those self-conditioned positions against GT. Train diffusion configs enable `self_condition_prob: 0.25` and `self_condition_visible_prob: 0.10`.
- Files: `smart/model/smart_diffusion.py`, `configs/train/train_scalable_diffusion.yaml`, `configs/train/train_scalable_diffusion_local.yaml`
- Validation: `python3 -m py_compile smart/model/smart_diffusion.py` passed.
- Next: Run a short diffusion training smoke and compare free-running validation metrics with self-conditioning enabled vs disabled.

## 2026-05-18 CST
- Task: Added a SMART-Diffusion map-token diagnostic visualization script.
- Result: `scripts/visualize_diffusion_map_tokens.py` randomly samples validation agents and plots packed map tokens, all-mask map edges, GT-geometry map edges, agent history, and GT future by chunk. A one-sample smoke generated two PNGs under `outputs/diffusion_map_token_debug_smoke/`.
- Files: `scripts/visualize_diffusion_map_tokens.py`
- Validation: `python3 -m py_compile scripts/visualize_diffusion_map_tokens.py` passed; smoke run wrote two diagnostic images.
- Next: Inspect multiple random samples, especially left-turn/right-turn lane cases, to determine whether bad early trajectories correlate with wrong or overly broad map-token context.

## 2026-05-18 CST
- Task: Added prefix-constrained SMART-Diffusion training and sampling to reduce out-of-road rollouts from unreliable future geometry.
- Result: Diffusion training suffix-closes masked chunks per agent and supervises only the first masked frontier chunk; sampling now releases only prefix-ready frontier chunks, remasks suffixes when earlier chunks are remasked, keeps sample-time confidence instead of recomputing unmasked confidence, and falls back to prefix-ordered final fill. Diffusion configs now expose `prefix_constrained_sampling` and `prefix_constrained_training`.
- Files: `smart/model/smart_diffusion.py`, `configs/train/train_scalable_diffusion.yaml`, `configs/train/train_scalable_diffusion_local.yaml`, `configs/validation/validation_scalable_diffusion.yaml`, `tests/test_smart_diffusion_prefix.py`
- Validation: `py_compile` passed; prefix helper unit tests passed; real validation batch loss/sampling smoke passed with full map context and no remaining mask tokens under both 4-step smoke and default 32-step sampling; inference smoke produced finite `pred_traj=(73,80,2)` and valid target token ids.
- Next: Train from scratch with prefix constraints enabled and compare off-road visualizations/metrics against the previous diffusion run.

## 2026-05-15 CST
- Task: Implemented SMART-Diffusion stability and fair-comparison upgrades.
- Result: Diffusion now uses quasi-random timestep sampling that works at batch size 1, exact-count low-variance masks, alternating antithetic token ranking, target-category-only training/evaluation by default, 6-layer diffusion configs, 32-step sampling, and MaskGIT-style remask/resample inference. Validation visualization now shows prediction/GT futures only for target-category agents while keeping other agents as context.
- Files: `smart/model/smart_diffusion.py`, `smart/callbacks/validation_visualization.py`, `configs/train/train_scalable_diffusion.yaml`, `configs/train/train_scalable_diffusion_local.yaml`, `configs/validation/validation_scalable_diffusion.yaml`
- Validation: `py_compile` passed; targeted checks covered low-discrepancy timestep coverage, exact-count masks, antithetic ranking, target-category filtering, and remask sampling; real local train batch and validation inference smoke passed; target-category visualization rendered to `/tmp/smart_diffusion_target_category_viz.png`.
- Next: Restart server training from scratch with the updated diffusion config and compare target-category validation metrics/visualizations against SMART.

## 2026-05-13 CST
- Task: Added SMART-Diffusion training-time geometry dropout to reduce teacher-forced geometry mismatch.
- Result: `_refresh_token_geometry` now accepts a `geometry_known_mask`; diffusion loss randomly drops a configurable fraction of visible GT tokens from geometry-chain advancement during training while keeping token inputs unchanged. Diffusion train configs set `geometry_dropout_prob: 0.25`; validation sets `0.0`.
- Files: `smart/model/smart_diffusion.py`, `configs/train/train_scalable_diffusion.yaml`, `configs/train/train_scalable_diffusion_local.yaml`, `configs/validation/validation_scalable_diffusion.yaml`
- Validation: `py_compile` passed; real-batch geometry check confirmed dropout changes refreshed positions while keeping finite tensors.
- Next: Run a longer GPU training smoke and inspect whether early visualizations reduce straight-line collapse without destabilizing diffusion loss.

## 2026-05-13 CST
- Task: Implemented SMART-Diffusion audit fixes for training speed, map memory, dynamic geometry, trainer config handling, and diagnostics.
- Result: `ntp_aux_loss_weight <= 0` now skips the NTP forward pass; diffusion map context is flat/ragged with packed-scene batch ids; future-token positions/headings refresh from currently unmasked tokens during training and sampling; token conditioning includes SMART agent shape embeddings; `train.py`/`val.py` respect yaml strategy and precision; `ConflictRate` bbox rotation now matches SMART decode; `pred_prob` now carries token-level diffusion selection confidence.
- Files: `smart/model/smart_diffusion.py`, `smart/modules/diffusion_decoder.py`, `train.py`, `val.py`, `smart/metrics/joint_consistency.py`
- Validation: `py_compile` passed; decoder toy forward passed with flat map context and no-map paths; local real-batch smoke confirmed NTP forward is skipped at weight 0 with finite loss; real-batch map/geometry check confirmed flat `map_context` and nonzero dynamic geometry change.
- Next: Run a longer GPU training smoke and inspect step visualizations for reduced straight-line collapse.

## 2026-05-12 CST
- Task: Changed SMART-Diffusion map context to directly reuse SMART-style radius map selection.
- Result: `max_map_tokens <= 0` now means no scene-level map-token truncation; diffusion packs all visible map tokens for each scene and relies on the shared SMART radius map-agent edge builder to select per-token map neighbors. Diffusion train/validation configs now set `max_map_tokens: 0`.
- Files: `smart/model/smart_diffusion.py`, `configs/train/train_scalable_diffusion.yaml`, `configs/train/train_scalable_diffusion_local.yaml`, `configs/validation/validation_scalable_diffusion.yaml`
- Validation: `py_compile` passed; local validation sample packed `map_context=(1,2328,128)` with `map_valid_count=2328`, matching the visible map token count, and diffusion loss/inference remained finite with `pred_traj=(73,80,2)`, `next_token_idx=(73,16)`.
- Next: Monitor server memory/throughput with full visible map tokens; if needed, cap `max_map_tokens` only as a resource fallback.

- Task: Reused SMART edge builders and physical token embeddings in SMART-Diffusion.
- Result: Edge construction was moved into `smart/modules/smart_edge_builder.py`; original `SMARTAgentDecoder` methods now wrap the shared builders, and `DiffusionDecoder` uses the same temporal, agent-agent, and map-agent raw relation builders with its own relation embeddings. Diffusion token inputs now use SMART's type-specific physical trajectory token MLPs for visible token ids and the learned mask token only for masked ids.
- Files: `smart/modules/smart_edge_builder.py`, `smart/modules/agent_decoder.py`, `smart/modules/diffusion_decoder.py`, `smart/model/smart_diffusion.py`
- Validation: `py_compile` passed; decoder toy forward passed with/without map and with physical token embeddings; local single-batch smoke produced finite train/diffusion loss, confirmed gradient reaches `encoder.agent_encoder.token_emb_veh`, and inference returned `pred_traj=(73,80,2)`, `next_token_idx=(73,16)`.
- Next: Restart local diffusion training from scratch and compare early loss stability plus visual trajectory diversity against the previous random-id embedding run.

- Task: Disabled active JEPA entry points and strengthened SMART-Diffusion geometry, loss, and map conditioning.
- Result: `train.py`/`val.py` now register only `smart` and `smart_diffusion`; common validation/step visualization no longer builds JEPA overlays. Diffusion now uses SMART-style future-token and map-to-future graph edges with edge-relative Fourier embeddings, selected visible map token context, and valid-token-normalized diffusion NLL with a connected zero-loss no-mask path.
- Files: `train.py`, `val.py`, `smart/model/__init__.py`, `smart/model/smart_diffusion.py`, `smart/modules/diffusion_decoder.py`, `smart/callbacks/validation_visualization.py`, `smart/callbacks/step_visualization.py`, `scripts/export_val_quad_video.py`, `configs/train/train_scalable_diffusion*.yaml`, `configs/validation/validation_scalable_diffusion.yaml`
- Validation: `py_compile` passed; config load confirmed diffusion predictor and map fields; local single-batch smoke produced finite loss, `token_positions=(1,1152,2)`, `map_context=(1,128,128)`, and inference outputs `pred_traj=(73,80,2)`, `next_token_idx=(73,16)`. A graph-edge visualization smoke image was written to `/tmp/smart_diffusion_graph_edge_viz.png`.
- Next: Restart diffusion local training from scratch and compare loss stability/rollout quality against the previous diffusion run.

- Task: Fixed SMART-Diffusion visualizations connecting invalid predictions to `(0, 0)`.
- Result: Diffusion inference now returns `pred_valid_mask` for decoded prediction steps, and validation/step visualization uses it instead of GT future validity when drawing predicted trajectories. Invalid chunks can remain zero-filled internally without being rendered as red lines to the origin.
- Files: `smart/model/smart_diffusion.py`, `smart/callbacks/validation_visualization.py`
- Validation: `py_compile` passed; targeted decode check confirmed invalid chunks keep `pred_valid_mask=False`; a real validation sample from `checkpoints/diffusion_smoke/epoch=03.ckpt` had zero valid predicted points at `(0,0)` and no nonzero invalid predicted points, then rendered successfully to `/tmp/smart_diffusion_pred_valid_check.png`.
- Next: Regenerate step visualizations after the next training interval and verify the red prediction traces no longer connect to the origin.

- Task: Added low-discrepancy timestep sampling for SMART-Diffusion training.
- Result: Diffusion loss now samples stratified timesteps across each packed scene batch before masking, matching the minibatch low-discrepancy strategy used to reduce MDLM timestep variance; `batch_size=1` still degenerates to ordinary uniform sampling, so variance reduction becomes meaningful once multiple scenes are packed together.
- Files: `smart/model/smart_diffusion.py`
- Validation: `py_compile` passed; targeted sampler checks confirmed one timestep per stratum for `B=8` and `B=16`, and local diffusion single-batch train/validation smoke still ran through.
- Next: If the current local run remains batch size 1, assess whether memory allows a larger batch or whether a separate cross-step variance reduction scheme is needed.

- Task: Hardened SMART-Diffusion decode and validation edge cases after reviewing empty-agent and visualization failure modes.
- Result: Invalid or unavailable diffusion chunks now keep `next_token_idx=-1` and zero trajectories/headings instead of decoding as token `0`; empty diffusion training batches return a parameter-connected zero loss with explicit empty-batch metrics; validation visualization skips `None` predictions. While running the requested smoke check, a separate `ConflictRate` bbox matmul shape bug was fixed so validation inference can complete.
- Files: `smart/model/smart_diffusion.py`, `smart/callbacks/validation_visualization.py`, `smart/metrics/joint_consistency.py`
- Validation: `py_compile` passed; targeted Python checks covered invalid decode outputs, empty-batch zero loss, visualization `None` predictions, and `ConflictRate` bbox tensor shape; local diffusion config single-batch train/validation/inference smoke passed with `pred_traj=(73,80,2)` and `next_token_idx=(73,16)`.
- Next: Resume local diffusion training and watch the new empty-batch counters alongside rollout and interaction metrics.

- Task: Fixed SMART-Diffusion visualization-time inference crash under PyTorch 1.12 CUDA eval/no_grad.
- Result: `DiffusionDecoder` now disables the PyTorch 1.12 fused TransformerEncoderLayer fast path while still passing `src_key_padding_mask`, avoiding the CUDA mask-shape fallback error.
- Files: `smart/modules/diffusion_decoder.py`
- Validation: `py_compile` passed; eval/no_grad decoder forward with padding mask passed; toy diffusion sampling still leaves no mask tokens on valid positions and keeps padding zero.
- Next: Resume local diffusion training with step visualization enabled.

- Task: Implemented the SMART-Diffusion enhancement plan for from-scratch discrete mask diffusion with NTP auxiliary training.
- Result: Diffusion denoising now uses per-agent history context, agent type embeddings, Transformer padding masks, corrected iterative unmask sampling, idempotent batch preparation, NTP auxiliary loss, and interaction diagnostics.
- Files: `smart/model/smart_diffusion.py`, `smart/modules/diffusion_decoder.py`, `smart/metrics/joint_consistency.py`, `smart/callbacks/*visualization.py`, `configs/train/train_scalable_diffusion*.yaml`, `configs/validation/validation_scalable_diffusion.yaml`
- Validation: Static py_compile passed; diffusion configs load in the `smart` conda env; a CPU single-sample smoke confirmed finite training loss and inference output shapes `pred_traj=(A,80,2)`, `next_token_idx=(A,16)`.
- Next: Launch a short local diffusion training run and compare validation rollout/interaction metrics against baseline SMART.

## 2026-05-07 CST
- Task: Fixed standalone validation dataset initialization after `val.py` defaulted SMART tokenization to 512 tokens.
- Result: `val.py` now passes configured `token_size` into `MultiDataset` and accepts either `Dataset.batch_size` or `Dataset.val_batch_size` for validation loading.
- Validation: `python3 -m py_compile val.py` succeeded; `smart/tokens/cluster_frame_5_2048.pkl` and `smart/tokens/map_traj_token5.pkl` are present.
- Next: Rerun `python val.py --config configs/validation/validation_scalable.yaml --pretrain_ckpt /mnt/d/epoch\=20.ckpt` in the `smart` conda environment.

## 2026-04-29 CST
- Task: Added an explicit NTP second-stage training config for the JEPA ego30 hidden-history pretrain path.
- Result: `configs/train/train_scalable_ntp_from_jepa_ego30_hidden.yaml` was created from the baseline SMART config with `Model.predictor: smart`; it is intended to be launched with `--pretrain_ckpt <jepa_pretrain_ckpt>`.
- Validation: Config load check in the `smart` conda environment confirmed `predictor == smart`, no `Model.jepa`, and `monitor_metric == val_cls_acc`.
- Next: Commit and push only the new config file; keep existing unrelated worktree changes unstaged.

- Task: User reported completion of first-stage forecast-aligned JEPA pretraining with hidden masked-agent history.
- Result: Next workflow is to treat the JEPA checkpoint as an encoder initialization source, then start second-stage NTP training with the baseline SMART predictor rather than joint JEPA training.
- Files: `configs/train/train_scalable_jepa_pretrain_forecast_aligned_ego30_hidden_history.yaml`, `checkpoints/jepa_pretrain_ego30_hidden`
- Validation: Standalone `val.py` is not required for the JEPA-only pretrain stage unless checking held-out JEPA loss as a sanity check.
- Next: Use `configs/train/train_scalable.yaml` with `--pretrain_ckpt <jepa_pretrain_ckpt>` for the NTP stage after confirming full data paths and `total_steps`.

## 2026-04-21 15:26 CST
- Task: Added a forecast-aligned JEPA pretrain config variant that hides target-agent history as well as the ego-region map.
- Result: The forecast-aligned objective no longer hard-codes masked-agent history to `visible`; a new train/validation config pair was added for the `hidden` history ablation.
- Files: `smart/model/smart_jepa.py`, `configs/train/train_scalable_jepa_pretrain_forecast_aligned_ego30_hidden_history.yaml`, `configs/validation/validation_scalable_jepa_pretrain_forecast_aligned_ego30_hidden_history.yaml`
- Validation: `python3 -m py_compile smart/model/smart_jepa.py` succeeded, and both forecast-aligned configs were instantiated to confirm `visible` vs `hidden` history modes resolve correctly.
- Next: Run the hidden-history config smoke check, then compare visible-history vs hidden-history pretrain curves.

## 2026-04-21 14:55 CST
- Task: Installed a code graph workflow and durable project memory workflow for this repository.
- Result: `codebase-memory-mcp` was installed and the SMART repo was indexed; repo-local workflow instructions and memory files were added.
- Files: `AGENTS.md`, `docs/spec.md`, `docs/progress.md`, `docs/decisions.md`, `docs/next.md`
- Validation: Verified the code graph binary version, confirmed the SMART project was indexed, and ran a successful architecture query against the indexed project.
- Open: Codex still needs a restart before future sessions can use the new MCP server directly.
- Next: Restart Codex, then use graph-first exploration and keep `docs/*.md` updated after meaningful tasks.

## 2026-05-24 CST
- Task: Aligned SMART-Diffusion and AR-Diffusion validation/visualization masks with official SMART validation semantics.
- Result: `smart_val_compatible` now uses current-history-valid agents for metrics, inference rollout no longer requires GT future token validity, validation ADE/FDE runs on every validation batch, model outputs include `official_valid_mask`, and visualization/video rendering use `pred_valid_mask` only for display/coverage rather than official metric filtering.
- Files: `smart/model/smart_diffusion.py`, `smart/model/smart_ar_diffusion.py`, `smart/callbacks/validation_visualization.py`, `scripts/export_val_quad_video.py`, `tests/test_smart_diffusion_smart_parity.py`
- Validation: In the `smart` conda environment, `python -m py_compile smart/model/smart_diffusion.py smart/model/smart_ar_diffusion.py smart/callbacks/validation_visualization.py scripts/export_val_quad_video.py tests/test_smart_diffusion_smart_parity.py` passed; `python -m unittest tests.test_smart_diffusion_smart_parity -v` passed 12 tests with 1 Waymo-dependency skip; `tests.test_smart_ar_diffusion` passed 6 tests; `tests.test_smart_diffusion_prefix` passed 3 tests.
- Next: Inspect validation visualizations in official view for missing current-valid vehicles, then run a real validation smoke on AR diffusion configs.

## 2026-05-25 CST
- Task: Fixed AR discrete diffusion rollout history-token validity.
- Result: AR rollout views now preserve real/rolling history token validity and mask all token-history context for agents that are not current-valid generation agents, preventing invalid token-0 history from creating spurious high-speed motion features.
- Files: `smart/model/smart_ar_diffusion.py`, `tests/test_smart_ar_diffusion.py`
- Validation: `tests.test_smart_ar_diffusion` passed 8 tests; `py_compile` passed for AR files; diffusion prefix tests passed; SMART parity tests passed with the existing Waymo-dependency skip; `git diff --check` passed.
- Next: Re-run AR validation visualization and inspect straight-vehicle predicted token speeds versus GT token speeds after this mask fix.

## 2026-05-25 CST
- Task: Clarified AR diffusion validation loss and removed unused self-conditioning config fields.
- Result: `SMARTAutoregressiveDiffusion.validation_step()` now computes deterministic AR short-window validation loss under `val_ar_window_*` metrics while full rollout ADE/FDE remains under `val_minADE`/`val_minFDE`; AR training configs monitor `val_minADE` instead of misleading `val_loss`. Unused `self_condition_visible_prob` and `self_condition_loss_weight` fields were removed from code and diffusion train configs.
- Files: `smart/model/smart_ar_diffusion.py`, `smart/model/smart_diffusion.py`, `tests/test_smart_ar_diffusion.py`, diffusion/AR train configs
- Validation: `tests.test_smart_ar_diffusion` passed 11 tests; `py_compile` passed for AR/diffusion files; diffusion prefix tests passed; SMART parity tests passed with the existing Waymo-dependency skip; `git diff --check` passed.
- Next: Use `val_minADE`/`val_minFDE` for AR checkpoint selection and compare `val_ar_window_loss` only as a local short-window denoising diagnostic.

## 2026-05-25 CST
- Task: Limited epoch-end AR diffusion validation batches during training.
- Result: `train.py` now passes optional `Trainer.limit_val_batches` and `Trainer.check_val_every_n_epoch` through to PyTorch Lightning. AR diffusion train configs set `limit_val_batches: 50` and `check_val_every_n_epoch: 1` so epoch-end validation can run a bounded subset while preserving full validation when these fields are omitted.
- Files: `train.py`, `configs/train/train_scalable_ar_diffusion.yaml`, `configs/train/train_scalable_ar_diffusion_local.yaml`
- Validation: `python3 -m py_compile train.py` passed.
- Next: If validation remains too slow, profile AR inference to separate repeated encoder/map-context cost from diffusion denoising cost before changing model logic.

## 2026-05-26 CST
- Task: Added validation debug logging for AR diffusion server runs.
- Result: Added `diffusion.debug_validation_logging` and rank-aware stdout logs around AR validation window loss, full rollout inference, per-round diffusion sampling, validation visualization, and step visualization. AR train/validation configs enable the flag so server logs show where epoch-end validation is spending time.
- Files: `smart/model/smart_diffusion.py`, `smart/model/smart_ar_diffusion.py`, `smart/callbacks/validation_visualization.py`, `smart/callbacks/step_visualization.py`, AR diffusion configs
- Validation: `python3 -m py_compile smart/model/smart_diffusion.py smart/model/smart_ar_diffusion.py smart/callbacks/validation_visualization.py smart/callbacks/step_visualization.py` passed; `git diff --check` passed for edited files.
- Next: Use the new `[SMARTDiffusion]`, `[ValidationVisualization]`, and `[StepVisualization]` logs on the server to separate validation window loss, full AR rollout, and visualization bottlenecks.

## 2026-05-29 CST
- Task: Fixed `--pretrain_ckpt` checkpoint initialization to avoid GPU0 CUDA context pollution under DDP.
- Result: `train.py` and `val.py` now load pretrain checkpoints with `to_cpu=True`, so weights are mapped through CPU before Lightning/DDP places model replicas on their assigned devices. A regression test asserts both entry points keep CPU-mapped pretrain loading.
- Files: `train.py`, `val.py`, `tests/test_pretrain_checkpoint_loading.py`
- Validation: The new pretrain checkpoint loading test passed; `py_compile` passed for both entry points and the test.
- Next: Use `--pretrain_ckpt` rather than `--ckpt_path` for new AR diffusion finetunes, and confirm server LR is nonzero plus GPU0 no longer accumulates per-rank checkpoint-loading contexts.

## 2026-06-15 CST
- Task: Rewrote `smart_elf` as a standalone official-ELF-style predictor instead of the previous AR-inherited wrapper.
- Result: `SMARTEmbeddedLanguageFlow` now inherits only `pl.LightningModule`, composes the SMART map/history encoder, packs the full 16-token future sequence, and uses an independent ELF decoder with embedding-space flow plus factored token decoding. ELF configs no longer include AR/causal rollout, retokenization, or guidance fields.
- Files: `smart/model/smart_elf.py`, `smart/modules/elf_decoder.py`, `configs/train/train_scalable_elf_1000.yaml`, `configs/train/train_scalable_elf_3epoch_local.yaml`, `configs/validation/validation_scalable_elf.yaml`, `tests/test_smart_elf.py`, `docs/spec.md`, `docs/next.md`, `docs/decisions.md`
- Validation: `tests.test_smart_elf` and `tests.test_compare_motion_models` passed; `py_compile` passed for ELF code and entrypoint tests; a real validation-sample CPU smoke produced finite loss and `pred_traj=(73, 80, 2)`, `next_token_idx=(73, 16)`.
- Next: Retrain standalone ELF before comparing loss/accuracy or visual quality; old `elf_ar_*` checkpoints and visualizations are architecture-incompatible.

## 2026-06-16 CST
- Task: Converted standalone `smart_elf` from one-shot full-horizon inference to receding-horizon simulation-agent rollout.
- Result: ELF now samples four-token windows, commits one token, updates the rolled history anchor, and re-encodes map/history context before the next window. Training samples shifted GT windows and moves the previous GT anchor into the history state so later windows learn with local map context.
- Files: `smart/model/smart_elf.py`, ELF train/validation configs, `tests/test_smart_elf.py`, `docs/spec.md`, `docs/next.md`, `docs/decisions.md`
- Validation: `tests.test_smart_elf` and `tests.test_compare_motion_models` passed; `py_compile` passed; real `data/valid_demo` CPU smoke passed for both a two-token rolling check and full 16-token rollout with finite `pred_traj=(58, 80, 2)` and `next_token_idx=(58, 16)`.
- Next: Retrain with `checkpoints/elf_receding_3epoch`, then inspect map-constraint violations on trajectory tails before comparing against causal/hybrid runs.

## 2026-06-16 CST
- Task: Added an AR-first rerank diffusion variant without causal-frontier training.
- Result: `smart_ar_diffusion` can now keep the MaskGIT AR window objective while applying safe-speed top-k reranking only to committed tokens after sampling. The rerank variant also exposes map-token noise and history-context dropout as config-gated SMART-style conditioning perturbations.
- Files: `smart/model/smart_ar_diffusion.py`, `smart/model/smart_diffusion.py`, `configs/train/train_scalable_ar_diffusion_rerank_1000.yaml`, `configs/validation/validation_scalable_ar_diffusion_rerank.yaml`, `tests/test_smart_ar_diffusion.py`, `docs/spec.md`, `docs/next.md`, `docs/decisions.md`
- Validation: `tests.test_smart_ar_diffusion` plus `tests.test_train_entrypoint_config` passed 34 tests locally; `py_compile` passed for touched Python files; `git diff --check` passed.
- Next: Train `configs/train/train_scalable_ar_diffusion_rerank_1000.yaml`, then compare straight-vehicle moving-speed ratios against AR baseline, AR frontier, causal diffusion, and hybrid diffusion on the same validation scenes.

## 2026-06-16 CST
- Task: Added the server training config for the AR-first rerank diffusion run.
- Result: `configs/train/train_scalable_ar_diffusion_rerank.yaml` uses the full server Waymo paths, 14-GPU DDP settings, 32 epochs, 4-token prediction, 1-token commit, `ar_objective: maskgit`, safe-speed commit rerank, map-token noise, and history-context dropout.
- Files: `configs/train/train_scalable_ar_diffusion_rerank.yaml`, `tests/test_smart_ar_diffusion.py`, `docs/next.md`, `docs/progress.md`
- Validation: Focused AR rerank config tests passed; YAML syntax/field check passed for the server config; `git diff --check` passed.
- Next: Upload this config with the rerank code and launch `python -u train.py --config configs/train/train_scalable_ar_diffusion_rerank.yaml --save_ckpt_path checkpoints/ar_rerank`.

## 2026-06-16 CST
- Task: Fixed standalone ELF window attention to match receding-horizon causality.
- Result: `EmbeddedLanguageFlowDecoder` now builds a chunk-causal pairwise attention mask. Same-chunk agents can attend to each other, later chunks are hidden from earlier chunk queries, and prefix tokens cannot aggregate data-token content.
- Files: `smart/modules/elf_decoder.py`, `tests/test_smart_elf.py`, `docs/next.md`, `docs/decisions.md`, `docs/progress.md`
- Validation: `tests.test_smart_elf` and `tests.test_compare_motion_models` passed; `py_compile` and `git diff --check` passed; a real `data/valid_demo` CPU inference smoke produced finite `pred_traj=(58, 80, 2)` and `next_token_idx=(58, 16)`.
- Next: Retrain receding ELF after the attention-mask change before interpreting map-compliance metrics.

## 2026-06-16 CST
- Task: Fixed AR rerank diffusion token attention semantics before server training.
- Result: AR rerank train/validation configs now enable `causal_temporal_edges: true` while keeping `ar_objective: maskgit`. `DiffusionDecoder` now builds spatial token radius-graph groups in contiguous `(scene, chunk)` order and remaps edges back to the packed sequence, so same-window agents interact only within the same chunk.
- Files: `smart/modules/diffusion_decoder.py`, AR rerank train/validation configs, `tests/test_smart_causal_diffusion.py`, `tests/test_smart_ar_diffusion.py`, and durable docs.
- Validation: Focused decoder/config tests passed; `tests.test_smart_ar_diffusion` passed 28 tests; `tests.test_smart_causal_diffusion` passed 43 tests.
- Next: Retrain AR rerank checkpoints; older rerank runs should be treated as stale for attention-causality comparisons.

## 2026-06-17 CST
- Task: Added a map-conditioned commit scorer to standalone receding `smart_elf`.
- Result: ELF sampling can now combine final embedding-token similarity with a map-conditioned token score from the SMART history-context feature and optional top-k map-geometry energy. Training adds a commit-scorer CE loss controlled by `elf_map_commit_loss_weight`; ELF configs enable the scorer and geometry energy while preserving the standalone ELF path.
- Files: `smart/model/smart_elf.py`, ELF train/validation configs, `tests/test_smart_elf.py`, `docs/spec.md`, `docs/next.md`, `docs/decisions.md`
- Validation: `tests.test_smart_elf`, `tests.test_compare_motion_models`, and `tests.test_agent_decoder_history_context` passed together; `py_compile` and `git diff --check` passed; a real validation-sample CPU smoke with the old receding checkpoint produced finite `pred_traj=(73, 80, 2)` and `next_token_idx=(73, 16)`.
- Next: Retrain ELF with the map-conditioned scorer before judging map compliance; old receding ELF checkpoints only validate code loading, not quality.

## 2026-06-18 CST
- Task: Hardened diffusion proposal conditioning and future-token geometry refresh.
- Result: `DiffusionDecoder` now treats non-finite proposal or geometry confidence as zero/finite bounded confidence before embedding. `_refresh_token_geometry()` now advances each agent's future pose only through a contiguous chain of known or proposal-backed chunks, so a later visible token after a masked/proposal-missing gap no longer becomes a reliable geometry source from a stale pose.
- Files: `smart/modules/diffusion_decoder.py`, `smart/model/smart_diffusion.py`, `tests/test_smart_causal_diffusion.py`, `tests/test_smart_diffusion_smart_parity.py`, `docs/spec.md`, `docs/next.md`, `docs/decisions.md`
- Validation: `python -m unittest tests.test_smart_causal_diffusion tests.test_smart_ar_diffusion tests.test_smart_diffusion_smart_parity -v` passed 92 tests with 1 expected Waymo-dependency skip.
- Next: Retrain AR rerank or rerun checkpoint diagnostics before judging proposal-carry map adherence, because geometry-source confidence now differs after masked chunk gaps.

## 2026-06-18 CST
- Task: Removed two additional AR rerank state-consistency hazards from the code review follow-up.
- Result: AR inference no longer mutates the caller's `data['agent']` with `commit_speed_reference`; guidance speed context is kept in packed sampling state instead. Physical retokenization now keeps the previous heading for stationary decoded tokens, matching the non-physical path's norm guard and avoiding cumulative heading drift from zero-displacement tokens.
- Files: `smart/model/smart_ar_diffusion.py`, `tests/test_smart_ar_diffusion.py`, `docs/next.md`, `docs/decisions.md`, `docs/progress.md`
- Validation: `python -m unittest tests.test_smart_ar_diffusion tests.test_smart_causal_diffusion tests.test_smart_diffusion_smart_parity -v` passed 94 tests with 1 expected Waymo-dependency skip; `python -m py_compile smart/model/smart_ar_diffusion.py tests/test_smart_ar_diffusion.py` passed.
- Next: Treat older AR rerank diagnostics as stale for side-effect and stationary-heading checks; retrain before final map-adherence comparison.

## 2026-06-20 CST
- Task: Fixed the local training crash in discrete diffusion-policy batched multi-anchor mode.
- Result: Batched anchor construction now splits any PyG batch through `to_data_list()`, builds raw per-anchor views, merges them once, and runs map-token preparation on the merged anchor batch. This preserves one diffusion forward for multi-anchor training while avoiding stale/missing `pt_valid_mask` fields and malformed `pt_token -> map_polygon` dynamic edge batching.
- Files: `smart/model/smart_discrete_diffusion_policy.py`, `tests/test_smart_discrete_diffusion_policy.py`, `docs/progress.md`, `docs/next.md`
- Validation: Real `data/valid_demo` training smoke ran `training_step` plus `backward` with loss `4.662879943847656`; `tests.test_smart_discrete_diffusion_policy` and `tests.test_compare_motion_models` passed; `py_compile` and `git diff --check` passed.
- Next: Retry the local or server discrete diffusion-policy run after this anchor-view batching fix.

## 2026-06-20 CST
- Task: Reduced discrete diffusion-policy batched multi-anchor training overhead.
- Result: Batched `training_step()` no longer prepares the original batch before anchor batching; only the merged anchor batch is map-token prepared. Batched overlap KL now uses tensor key matching with `torch.searchsorted` instead of Python dict/list loops while preserving the shifted `p_t^k -> p_{t+k}^0` target semantics.
- Files: `smart/model/smart_discrete_diffusion_policy.py`, `tests/test_smart_discrete_diffusion_policy.py`, `docs/progress.md`, `docs/next.md`
- Validation: Real `data/valid_demo` training smoke ran `training_step` plus `backward` with loss `4.834183216094971`; `tests.test_smart_discrete_diffusion_policy` and `tests.test_compare_motion_models` passed; `py_compile` and `git diff --check` passed.
- Next: Retrain or rerun a short server throughput smoke with this optimized discrete-policy path before comparing wall-clock speed.
