# Progress

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
