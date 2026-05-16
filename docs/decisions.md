# Decisions

## Decision: Use graph-first repository discovery
- Date: 2026-04-21
- Context: Broad grep-based exploration wastes context and scales poorly as the repository grows.
- Decision: Use `codebase-memory-mcp` as the default discovery layer before direct file scanning.
- Why: Symbol-aware graph queries reduce noisy search, constrain file reads to the relevant subset, and preserve more context for reasoning.
- Impact: Future work should start with architecture and symbol queries, then read only the specific files that matter.

## Decision: Store durable task memory in repo-local Markdown files
- Date: 2026-04-21
- Context: Long sessions and session restarts make it easy to lose project intent, progress, and next actions.
- Decision: Keep persistent project state in `docs/spec.md`, `docs/progress.md`, `docs/decisions.md`, and `docs/next.md`.
- Why: Simple files are easy to inspect, diff, edit, and recover in any environment without extra infrastructure.
- Impact: Meaningful tasks should update these files so future sessions can resume from workspace state instead of chat history alone.

## Decision: Forecast-aligned JEPA should keep history visibility configurable
- Date: 2026-04-21
- Context: The first forecast-aligned JEPA pretrain config forced target-agent history to remain visible, which prevented direct comparison against a stricter region-masked variant.
- Decision: Keep the forecast-aligned region and full-horizon supervision semantics fixed, but let `masked_agent_history_mode` stay config-driven for this objective.
- Why: This enables controlled ablations between visible-history and hidden-history student views without introducing a second objective implementation.
- Impact: Forecast-aligned configs can now differ only by `masked_agent_history_mode`, while code paths for region selection and future supervision remain shared.

## Decision: SMART-Diffusion uses agent-conditioned type-aware mask diffusion
- Date: 2026-05-12
- Context: The first SMART-Diffusion prototype conditioned denoising mostly on a scene-level summary and had weak per-agent identity/type information.
- Decision: Condition each future token on pooled per-agent history context, agent type, position, chunk id, and scene summary; keep NTP as a low-weight auxiliary objective.
- Why: Discrete diffusion needs enough agent-specific context to generate meaningful futures from an all-mask initial state, while NTP regularizes the shared SMART encoder during from-scratch training.
- Impact: Diffusion configs now include `num_layers`, `ntp_aux_loss_weight`, `use_agent_context`, and `use_type_embedding`; diffusion validation should track both rollout quality and interaction diagnostics.

## Decision: JEPA is no longer an active SMART pipeline
- Date: 2026-05-12
- Context: Current work has shifted from SMARTJEPA to discrete SMART-Diffusion.
- Decision: Keep historical JEPA source files and configs in the repository, but remove `smart_jepa` from active train/validation registries and common visualization/export paths.
- Why: This avoids accidental use of stale JEPA paths while preserving earlier experiment artifacts for reference.
- Impact: Current `train.py` and `val.py` support `smart` and `smart_diffusion`; JEPA-specific historical files should not be treated as supported entry points.

## Decision: SMART-Diffusion conditions on SMART-style edge-relative graph geometry
- Date: 2026-05-12
- Context: Raw world coordinates and indirect map conditioning made diffusion training noisy and weakly grounded.
- Decision: Use explicit future-token and map-to-future graph edges with edge-relative Fourier embeddings; raw coordinates are only used for `radius` / `radius_graph` neighbor construction. Include selected visible map token features in the diffusion graph and normalize diffusion NLL over valid tokens instead of masked-token mean.
- Why: This more directly reuses the original SMART/QCNet geometry style than scene-centering, avoids scene-global coordinate scale issues, and reduces small-t gradient variance.
- Impact: Diffusion configs include `use_map_context` and `max_map_tokens`; `max_map_tokens <= 0` disables scene-level map truncation so SMART's radius map-agent edge builder controls map access. Validation should inspect graph edge construction, `map_context`, and loss stability during local/server runs.

## Decision: Reuse SMART physical token embeddings for diffusion tokens
- Date: 2026-05-12
- Context: A plain token-id embedding forces diffusion to relearn the physical meaning of SMART trajectory token ids from scratch.
- Decision: Embed non-mask diffusion token ids with the original SMART type-specific trajectory token MLPs (`veh` / `ped` / `cyc`) and keep the learned mask token only for masked positions.
- Why: This gives diffusion direct access to the token codebook's physical trajectory shape semantics and keeps NTP and diffusion aligned around the same token representation.
- Impact: Diffusion gradients now update SMART's type-specific token embedding MLPs unless the encoder is frozen.

## Decision: SMART-Diffusion graph geometry is diffusion-state dependent
- Date: 2026-05-13
- Context: Repeating every future-token node at the historical pose made map and interaction edges blind to the already released token chain.
- Decision: Refresh future-token graph positions/headings from currently unmasked tokens during both training and sampling; masked chunks fall back to the most recent available pose. Represent map context as flat/ragged tensors with packed-scene batch ids, and skip NTP forward entirely when `ntp_aux_loss_weight <= 0`.
- Why: This keeps graph edges aligned with the denoising state, avoids padded map memory blowups, and removes unnecessary auxiliary compute when NTP is disabled.
- Impact: Diffusion graph construction now depends on the current noisy token state; `pred_prob` reports token-level diffusion selection confidence rather than a SMART autoregressive probability.

## Decision: Use geometry dropout for SMART-Diffusion training
- Date: 2026-05-13
- Context: Training used exact GT geometry for visible future tokens, while inference starts from all-mask geometry and only gradually improves as sampled tokens are released.
- Decision: Add `diffusion.geometry_dropout_prob` so training can randomly prevent visible GT tokens from advancing the temporary geometry chain. Validation and inference keep dropout disabled.
- Why: This exposes the decoder to incomplete geometry states similar to early denoising and reduces over-reliance on perfect teacher-forced future geometry.
- Impact: Training configs default to `geometry_dropout_prob: 0.25`; validation config sets it to `0.0`.

## Decision: Align SMART-Diffusion with SMART target-category evaluation
- Date: 2026-05-15
- Context: Diffusion was training and evaluating all non-background agents, while original SMART primarily supervises target-category agents selected by `WaymoTargetBuilder`.
- Decision: Default SMART-Diffusion training, validation metrics, and visualized prediction trajectories to agents with `category == 3` and `type != 3`.
- Why: This makes SMART vs diffusion comparisons fairer and avoids judging diffusion on agents that the baseline objective does not emphasize.
- Impact: Diffusion configs now include `target_category_only: true`; non-target agents remain available as context but are not default prediction targets.

## Decision: Use low-variance mask training and remask sampling for SMART-Diffusion
- Date: 2026-05-15
- Context: Batch size can be one, Bernoulli mask counts add avoidable variance, and one-way confidence release can lock in early wrong tokens.
- Decision: Use quasi-random timestep sampling across global steps, exact-count masked token selection with alternating antithetic ranking, 6-layer diffusion decoding, and MaskGIT-style remask/resample sampling by default.
- Why: This directly targets training variance, capacity mismatch with SMART, and early denoising error lock-in without changing SMART tokenization.
- Impact: Diffusion configs default to `num_layers: 6`, `num_steps: 32`, exact low-variance masking, and `remask_sampling: true`.

## Decision: Use temporal block diffusion for SMART-Diffusion
- Date: 2026-05-15
- Context: Full-horizon diffusion denoises all 16 future chunks at once, which makes early sampling weakly conditioned and leaves later chunks without generated trajectory context.
- Decision: Default SMART-Diffusion to 4-chunk temporal blocks with 8 denoising steps per block. Training masks and supervises only the current block while conditioning on previous GT blocks; inference samples blocks autoregressively and conditions later blocks on earlier sampled blocks.
- Why: This adapts Block Diffusion's block-autoregressive idea to SMART trajectory tokens without replacing SMART's graph decoder or token vocabulary.
- Impact: Diffusion configs include `block_training`, `block_size_chunks`, `block_denoise_steps`, and block mask-probability bounds. Full-horizon diffusion remains available by disabling block training or setting the block size to cover all future chunks.

## Decision: Align SMART-Diffusion block training with the Block Diffusion objective
- Date: 2026-05-16
- Context: The first block rollout trained only one temporal block per step and clamped mask probability separately from the loss weight, which increased loss variance and deviated from the paper's sum-over-blocks objective.
- Decision: Train all temporal blocks every step, use direct clipped effective mask-rate sampling with exact mask counts, scale masked NLL by the matching `1 / p_actual`, and default block sampling to monotonic unmasking.
- Why: This is closer to Block Diffusion's objective while keeping SMART's graph decoder and avoiding a text-model KV-cache dependency.
- Impact: Diffusion configs now use `block_train_all_blocks: true`, `block_loss_weight: clipped_consistent`, `block_denoise_steps: 16`, and `remask_sampling: false`. Old block-diffusion checkpoints should be restarted from scratch because the loss schedule and objective changed.

## Decision: Port BD3-LM vectorized training as graph-equivalent block views
- Date: 2026-05-16
- Context: Official Block Diffusion avoids per-block forward loops with clean/noisy token packing and specialized attention masks, but SMART-Diffusion uses PyTorch 1.12 graph attention instead of dense SDPA/FlexAttention.
- Decision: Batch each scene/block as a separate graph view in one `DiffusionDecoder` forward, duplicate ragged map context with new packed view ids, and use BD3-LM-style `t == move_chance` masking plus validation-time clipping variance search.
- Why: This preserves SMART's graph decoder and repo environment while adding the paper's two practical ingredients: vectorized all-block training and data-driven schedule selection.
- Impact: Configs now default to `block_vectorized_training: true`, `sampling_eps_min/max`, `var_min: true`, and `clip_search_grid`; `block_mask_prob_min/max` remain deprecated compatibility aliases. Old diffusion checkpoints should be restarted again because mask sampling and loss scaling changed.
