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
