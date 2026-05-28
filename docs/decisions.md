# Decisions

## Decision: Use visible-token neighbor corruption for AR diffusion training
- Date: 2026-05-28
- Context: SMART's official rolling tokenization noise trains the model to continue from slightly perturbed token states, while AR diffusion training still used clean visible future-token context after masking.
- Decision: During SMART-Diffusion training, keep GT labels unchanged but randomly replace some unmasked visible future tokens with same-type top-k nearest trajectory-token neighbors before decoder conditioning and geometry refresh.
- Why: This matches the robustness role of SMART's token noise without corrupting the supervised target, exposing the diffusion decoder to plausible sampled-token context drift.
- Impact: AR diffusion train configs enable `visible_token_corruption_prob: 0.15` with `visible_token_corruption_topk: 5`; validation keeps corruption disabled.

## Decision: Keep causal chunk schedules as disabled AR diffusion ablations
- Date: 2026-05-28
- Context: Fixed per-chunk mask probabilities made AR diffusion visualizations worse, likely because they overrode `mask_prob(t)` while sampling still uses the global diffusion noise schedule.
- Decision: Disable causal chunk schedules in the default AR diffusion configs. When enabled for ablation, use chunk multipliers on the global mask probability instead of fixed per-chunk probabilities.
- Why: This keeps the diffusion timestep embedding aligned with the actual mask rate and lets receding-horizon proposal carry be evaluated without the confound of a mismatched causal schedule.
- Impact: `commit_tokens: 1` and `carry_tail_proposal: true` remain the AR default; `causal_chunk_mask_multipliers` is available for controlled experiments, while `causal_chunk_mask_probs` is legacy fallback behavior.

## Decision: Use causal chunk noise and loss schedules for AR diffusion
- Date: 2026-05-28
- Context: Uniform mask/noise over a 4-token AR diffusion window treats near executable tokens and far planning tokens equally, which can let uncertain high-speed straight tokens affect closed-loop rollout.
- Decision: Enable an AR diffusion training schedule with lower mask probability and higher loss weight for near chunks, and higher mask probability with lower loss weight for far chunks.
- Why: This implements the discrete-token analogue of Diffusion Forcing without changing SMART tokenization: near tokens are trained as reliable actions while far tokens remain soft proposal context.
- Impact: The first AR chunk receives the strongest supervision, later chunks are still learned but downweighted; full-horizon `smart_diffusion` remains unchanged unless causal schedule config is explicitly enabled.

## Decision: Use receding-horizon proposal carry for SMART AR diffusion
- Date: 2026-05-28
- Context: The previous 4-token prediction / 2-token commit rollout could commit two jointly sampled straight-vehicle tokens before refreshing map context and motion features, amplifying high-speed straight-token errors.
- Decision: Keep 4-token diffusion windows for short-horizon planning, but commit only the first token by default and carry the uncommitted tail as proposal geometry/confidence into the next window.
- Why: This restores 0.5s physical closed-loop correction while preserving diffusion's multi-token planning signal as a soft, reversible proposal rather than hard state.
- Impact: AR diffusion inference uses 16 rollout rounds for 80 future steps when `commit_tokens: 1`; validation is slower than 2-token commit but should be more stable for straight-vehicle speed.

## Decision: Add SMART autoregressive discrete diffusion as a separate predictor
- Date: 2026-05-22
- Context: Full-horizon joint diffusion can refresh proposal geometry internally, but it cannot re-query local map context after committed agent motion the way an autoregressive rollout can.
- Decision: Implement `smart_ar_diffusion` as a new predictor that reuses SMART trajectory tokens and the existing discrete diffusion decoder. Each outer step uses 2 history tokens, predicts 4 future tokens jointly, commits the first 2 tokens, updates agent state/history, and rescreens local map tokens before the next step.
- Why: This preserves discrete diffusion inside each short window while restoring the dynamic map-query behavior that made SMART autoregressive rollout robust.
- Impact: Existing `smart_diffusion` remains available as a full-horizon comparison path; AR diffusion configs are separate and should be used for closed-loop experiments.

## Decision: SMART-Diffusion uses SMART generation parity with uncertainty-aware joint diffusion
- Date: 2026-05-22
- Context: Target-category-only generation and prefix-frontier denoising diverged from upstream SMART inference and degraded sim-agent rollouts with zero/fallback trajectories, boundary exits, and collisions.
- Decision: Generate all SMART history-valid agents, supervise diffusion loss only on SMART category-3 targets, keep SMART-compatible category filtering in returned metric masks, and use proposal token geometry plus explicit confidence for masked future chunks during joint denoising.
- Why: This preserves original SMART rollout semantics while retaining discrete diffusion's simultaneous prediction over all masked valid tokens instead of reverting to autoregressive frontier-only sampling.
- Impact: `prefix_constrained_sampling` and `prefix_constrained_training` remain available only as ablations; default configs use `use_proposal_geometry: true`, `geometry_confidence_source_threshold: 0.35`, and SMART parity mask modes.

## Decision: Use prefix-constrained geometry for SMART-Diffusion denoising
- Date: 2026-05-18
- Context: Out-of-road rollouts persisted even though map tokens were retained and map-to-token edges were rebuilt each denoising step; masked future chunks still lacked reliable poses and fell back to weak historical geometry.
- Decision: During training, suffix-close masked chunks per agent and supervise only the first masked frontier chunk. During sampling, release only prefix-ready frontier chunks and remask suffixes whenever an earlier chunk is remasked.
- Why: Map conditioning should be queried from poses supported by an already sampled trajectory prefix, not from an early all-mask clean-trajectory guess that can amplify off-road errors.
- Impact: SMART-Diffusion remains joint across agents at each frontier chunk, while long-horizon generation becomes chunk-prefix ordered per agent. Configs expose `prefix_constrained_sampling` and `prefix_constrained_training` for ablation.

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

## Decision: Use training-time self-conditioning for SMART-Diffusion
- Date: 2026-05-20
- Context: SMART-Diffusion sampling keeps model-generated future tokens as visible denoising context, while training previously used only GT visible future tokens.
- Decision: Add optional self-conditioning that masks a subset of visible positions in a no-grad first pass, feeds argmax predictions back as visible tokens in the final pass, and supervises those positions against GT.
- Why: This exposes training to sampled-token context without changing the inference sampler or adding random visible-token corruption.
- Impact: Diffusion train configs now expose `self_condition_prob`, `self_condition_visible_prob`, `self_condition_mode`, and `self_condition_loss_weight`; validation loss remains teacher-forced.

## Decision: SMART-Diffusion validation parity follows official current-valid SMART metrics
- Date: 2026-05-24
- Context: Official SMART trains token supervision on category-3 targets, but validation ADE/FDE uses current-history-valid agents and raw future validity rather than the category-filtered inference return mask.
- Decision: Treat `smart_val_compatible` as official current-valid validation semantics. Keep `category == 3` for diffusion/token supervision and explicit `smart_category3` ablations only. Validation ADE/FDE must use raw official future validity directly and run on every validation batch, matching upstream SMART; `pred_valid_mask` is not part of the official metric mask.
- Why: This matches upstream SMART validation behavior and prevents validation visualizations from hiding generated non-target vehicles because the randomized target category changed between steps.
- Impact: Diffusion and AR diffusion outputs now expose `official_valid_mask`; visualization defaults to official view while target/supervision view remains optional for debugging training targets. Missing prediction coverage should be diagnosed separately from ADE/FDE.
