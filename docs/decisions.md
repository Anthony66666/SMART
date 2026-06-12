# Decisions

## Decision: Use SMART-style speed-referenced commit reranking for causal diffusion
- Date: 2026-06-12
- Context: Causal v2 safe sampling remained conservative because commit selection could prefer low-speed tokens and then use the slowed predicted history as the next speed reference.
- Decision: Keep four-token causal diffusion proposals, but add a commit-only speed energy that compares token-internal median frame speed against a decaying initial observed speed reference. Only the first executable chunk is hard-constrained; uncommitted tail tokens remain revisable proposals for guidance/editing.
- Why: This preserves diffusion-based scene editing while making the executed token closer to official SMART's stable closed-loop state update.
- Impact: Causal configs expose `commit_speed_energy_weight`, `commit_min_speed_ratio`, `commit_speed_threshold`, and `commit_speed_reference_decay`; validation logs `commit_speed_energy`.

## Decision: Use generic ego-risk guidance instead of predefined event targets
- Date: 2026-06-11
- Context: The paper direction should edit a real scene into a safety-critical counterfactual for the ego/SDC, not require the user to predeclare a cut-in, lead-hard-brake, or other scenario template. The previous cut-in target could produce nonzero scalar event scores while the visual result did not clearly read as a cut-in.
- Decision: Make `target_spec: ego_risk` the default for causal guidance. Ego stress/edit modes now score candidate tokens with a generic ego-risk reward that combines low TTC, close ego-target distance, ego path intrusion, route proximity, conflict timing, and required deceleration while masking hard collisions. `target_event_eta` can be set to `0.0` for generic runs, and legacy event specs remain available only as ablations.
- Why: This better matches safety-critical scene editing: choose an editable real target agent and optimize for making the ego's future interaction riskier while preserving the ego and non-target agents, instead of forcing a named maneuver class.
- Impact: Causal configs and smoke/visualization scripts default to `ego_risk`; validation logs `ego_risk_min_ttc`, `ego_risk_reward`, and `ego_risk_success_rate`; Pareto criticality uses ego-risk plus near-miss success instead of target-event success; visualization only draws the old corridor-entry marker for legacy cut-in specs.

## Decision: Replace global stress/edit guidance with ego-centric interaction guidance
- Date: 2026-06-11
- Context: The first stress/edit guidance used global arbitrary agent-pair min distance / TTC as the main criticality reward. That can generate or score risks unrelated to the ego/SDC and is a weak fit for safety-critical counterfactual scenario editing.
- Decision: Use `guidance.mode = none | safe | ego_stress | ego_edit`. Ego modes score top-k candidate tokens with ego/SDC-only interaction reward, target-event reward, invalid-energy penalty, and edit-distance penalty. The first implemented event targets are `cut_in` and `lead_hard_brake`; `crossing_conflict` and `yield_failure` have basic event hooks for follow-up.
- Why: The paper direction should demonstrate minimal, interpretable target-agent edits that create risk for ego while preserving ego and non-target agents, instead of trying to beat official SMART on displacement metrics or rewarding unrelated scene-wide danger.
- Impact: Causal configs expose `ego_stress_topk`, `ego_edit_topk`, `ego_interaction_alpha`, `target_event_eta`, `path_corridor_width`, `conflict_tta_threshold`, and `target_spec`. Smoke and visualization scripts default to `seed,none,safe,ego_stress,ego_edit`.

## Decision: Add inference-time guidance modes for causal diffusion editing and stress testing
- Date: 2026-06-11
- Context: `smart_causal_diffusion` may not beat official SMART on displacement metrics, but its discrete diffusion sampler is useful for controllable trajectory editing and safety-critical counterfactual generation with existing checkpoints.
- Decision: Keep the causal training objective unchanged and add inference-time `guidance.mode = none | safe | stress | edit`. `none` uses unguided token sampling, `safe` preserves the existing safety-energy rerank, `stress` rewards collision-free near misses/low TTC while penalizing invalid trajectories, and `edit` locks seed tokens outside the target agent/time window while minimizing edit distance.
- Why: This reuses the four-token causal frontier sampler, tail proposal carry, current-state conditioning, and top-k reranking path without retraining.
- Impact: Causal configs now expose nested `diffusion.guidance` fields. Existing safe behavior remains the default; stress/edit are inference-time modes for scenario generation and pressure testing.

## Decision: Rebuild causal diffusion as a revisable receding-horizon planner
- Date: 2026-06-11
- Context: The v1 model discarded three of four sampled tokens, trained epoch 0 only on clean histories, had no all-mask chunk-0 interaction sources, and applied zero safety guidance to the executed token. Static vehicles became moving after model histories replaced GT histories and then never recovered.
- Decision: Use uniform discrete frontier CE, carry uncommitted tail tokens as revisable geometry/embedding proposals, add recency-weighted history and explicit current motion, expose chunk-0 current anchors to causal/spatial attention, and apply fixed commit safety with observation-to-first-frame dynamics.
- Why: These changes align the supervised decision with the executed action and preserve short-horizon planning without committing uncertain future tokens.
- Impact: Causal v2 must train from scratch. SMART and `smart_ar_diffusion` interfaces remain unchanged.

## Decision: Keep causal LR scheduling epoch-based
- Date: 2026-06-11
- Context: Lightning steps the scheduler returned by `configure_optimizers()` once per epoch. Values `2000/120000` therefore keep a 32-epoch run inside an extremely low warmup regime.
- Decision: Server and validation configs use `warmup_steps: 2`, `total_steps: 32`; the five-epoch local config uses `1/5`.
- Why: This matches the user's requested epoch-level LR schedule and the actual scheduler interval.
- Impact: Do not interpret these two fields as optimizer-step counts unless the scheduler return metadata is explicitly changed to `interval: step`.

## Decision: Release one causal frontier per sampling step
- Date: 2026-06-11
- Context: With four prediction chunks and `num_steps: 16`, the sampler skipped decoding for 12 iterations and released all chunks only at `t=0.25, 0.1875, 0.125, 0.0625`. This activated strong `(1-t)^2` safety guidance before the undertrained decoder had produced a reliable executable-token distribution and contributed to severe token mode collapse.
- Decision: Require `diffusion.num_steps == diffusion.prediction_tokens`. Four-token causal windows therefore decode once per frontier at `t=1.0, 0.75, 0.5, 0.25`, with no idle sampling iterations.
- Why: The all-mask state starts at high noise, each irreversible frontier release receives its own decoder call, and safety guidance ramps from weak to strong instead of dominating the first commit.
- Impact: All causal train/validation configs use `num_steps: 4`; invalid schedules fail during model construction. Existing checkpoints remain weight-compatible when loaded into a model created from the corrected config.

## Decision: Add the initial causal closed-loop diffusion predictor
- Date: 2026-06-10
- Context: Existing AR diffusion trains mostly on clean short windows but performs 16 recurrent commits at inference. Its non-causal future-token attention, proposal carry, remasking, and lack of explicit road/dynamics scoring allow early token errors to compound into late-horizon map exits.
- Decision: Add `smart_causal_diffusion` as an independent predictor with four-token windows, one-token commits, strictly causal temporal edges, and monotonic frontier release. Its original continuous-time loss was subsequently replaced by the v2 discrete frontier objective above.
- Why: The executable token must not depend on uncertain future chunks or be locked at high noise. The training corruption marginal and loss weighting must describe the same stochastic process used by the model.
- Impact: Existing SMART and diffusion predictors remain unchanged. New train/validation configs and registry entries select the causal path explicitly.

## Decision: Train causal diffusion on predicted states with retokenized recovery targets
- Date: 2026-06-10
- Context: Clean teacher-forced token labels become geometrically inconsistent after a predicted or perturbed state drifts from the ground-truth anchor.
- Decision: Use a 32-epoch clean/perturb/model-rollout curriculum. Commit one to four model tokens to form rollout states, transform the GT continuation into that state frame, and rematch the SMART codebook. Targets above per-type P99 error thresholds leave discrete CE and use differentiable expected-endpoint recovery instead.
- Why: This aligns labels with the actual closed-loop state distribution without assigning impossible discrete targets.
- Impact: `scripts/calibrate_causal_retokenization.py` computes per-type thresholds. The 10,000-scene perturbed P99 calibration is frozen as `[0.7379697561, 0.8562850952, 1.2705252171]`.

## Decision: Use commit-aware top-k safety-energy reranking without hard projection
- Date: 2026-06-10
- Context: Map attention alone does not guarantee that a high-probability token stays lane-aligned, dynamically feasible, or collision-free.
- Decision: Rerank top-k candidates with lane-distance, lane-heading, acceleration/yaw-rate, and collision energies. The executable chunk uses fixed `commit_safety_weight`; later chunks retain `(1-t)^2` scaling. Dynamics includes the observed-state-to-first-frame transition.
- Why: Soft reranking preserves the SMART token manifold and model diversity while making the irreversible low-noise commit explicitly safety-aware.
- Impact: Validation logs horizon metrics, late ADE, energy terms, coverage, retokenization-invalid rate, and a safety-led `val_rollout_score`.

## Decision: AR diffusion rescreening should keep full scene map candidates like SMART
- Date: 2026-06-07
- Context: Original SMART computes map features once, then rebuilds map-to-agent radius edges from the full scene map token set at each recurrent step. AR diffusion previously prefiltered map context by current agent pose before the decoder rebuilt map-to-token edges, so future/proposal positions could not connect to map tokens outside that local subset.
- Decision: For smart_ar_diffusion with local_map_refresh=rescreen, pack all visible map tokens for each packed scene and rely on DiffusionDecoder._build_map2token_edges() to perform the current-geometry radius search.
- Why: This matches original SMART's map access semantics while preserving diffusion's existing map-to-token graph and static one-time x_pt map feature encoding.
- Impact: AR diffusion may use more map memory/compute per packed scene, but late-horizon rollout should no longer be limited by an early current-pose map prefilter. If resource use is too high, add an explicit resource fallback rather than changing the default SMART-parity behavior.

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
