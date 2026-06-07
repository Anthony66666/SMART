# Progress

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
