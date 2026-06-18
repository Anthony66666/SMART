# Decisions

## Decision: Throttle AR rerank auxiliary losses for faster training
- Date: 2026-06-18
- Context: AR rerank training step was doing three encoder passes by default: the main AR diffusion window, dense original-SMART CE on the full scene, and proposal-carry diffusion on the next window. The latter two are useful auxiliary signals, but running both every step made training much slower than the original SMART-style baseline.
- Decision: Keep the auxiliary objectives, but make them explicitly scheduled. `dense_smart_ce_interval` and `proposal_carry_interval` gate their execution by global step, and `proposal_carry_detach_encoder` can build the proposal-carry encoder context under `no_grad` while still training the diffusion decoder on that loss. The active rerank train configs use dense CE every 4 steps, proposal carry every 2 steps, and detach the proposal-carry encoder path.
- Why: This reduces repeated encoder forward/backward work without silently changing the default model semantics or pretending that different original/window/proposal views can share one encoder output.
- Impact: New AR rerank speed/quality comparisons should report the auxiliary intervals. Default configs that omit these fields keep interval `1` and proposal detach disabled.

## Decision: Train AR rerank with dense SMART CE plus proposal-carry supervision
- Date: 2026-06-17
- Context: The AR rerank variant predicted four tokens but committed one; the three tail tokens only acted as inference-time proposals and were not represented as proposal inputs during training. Random single-window MaskGIT training also meant each scene contributed far less dense next-token supervision than original SMART.
- Decision: Keep `ar_objective: maskgit` and avoid frontier loss, but train AR rerank with three terms: a downweighted diffusion-window loss, dense original-SMART next-token CE over the prepared full scene, and an auxiliary next-window diffusion loss conditioned on corrupted carried tail proposals. Enable causal loss weights through an explicit `causal_loss_weighting_enabled` gate so old AR baseline configs remain unchanged unless they opt in.
- Why: This preserves the intended AR-first design while making the four-token proposal mechanism visible to the objective and restoring full-sequence SMART CE signal.
- Impact: Retrain AR rerank checkpoints. Existing rerank checkpoints trained before this change are stale for judging tail proposal usefulness, late map adherence, or straight-vehicle speed.

## Decision: Align SMART ELF with all-agent sim-agent rollout state
- Date: 2026-06-17
- Context: Receding ELF was generating and validating all current-valid non-background agents, but the active configs supervised only `category == 3` and SMART history-context map attention still connected map tokens only to category-3 agents. Its receding commit path also overwrote only the last history anchor, leaving older history tokens/frames stale after each generated step.
- Decision: Treat active `smart_elf` as an all-agent sim-agent rollout path. ELF configs use `supervision_mode: all_agents` and `target_category_only: false`; `SMARTDecoder.encode_history_context()` accepts an optional `map_agent_mask` so ELF can give every generated agent direct map2agent attention while other paths keep category-3 default behavior. ELF training views and inference commits roll the token and frame history windows before re-encoding.
- Why: The agent set used for loss, map conditioning, validation, and visualization must match the generated rollout set. Receding-history context must represent the latest generated state sequence, not a stale original-history token plus a new final anchor.
- Impact: Existing ELF checkpoints trained with category-only supervision or overwrite-only history should be treated as stale for map-compliance evaluation. Retrain ELF before judging late-horizon lane adherence.

## Decision: Use chunk-causal token attention for AR rerank without frontier loss
- Date: 2026-06-16
- Context: The AR rerank variant should keep the MaskGIT AR objective and SMART-style perturbations, but the executable earlier chunks must not read later future-token states inside the four-token window.
- Decision: Set the AR rerank train/validation configs to use `causal_temporal_edges: true`, and make diffusion spatial token edges robustly same-scene/same-chunk by building radius-graph batches in contiguous `(scene, chunk)` order before remapping to the packed sequence.
- Why: This preserves the user's requested AR-first rerank path without frontier supervision, while matching the intended attention semantics: same-timestep agents can interact, but earlier chunks cannot depend on later chunks.
- Impact: Retrain AR rerank checkpoints after this change. Existing rerank checkpoints trained with bidirectional temporal attention or non-contiguous spatial batching should be treated as stale for attention-causality comparisons.

## Decision: Add a synthetic multi-camera layout exporter before video generation
- Date: 2026-06-15
- Context: The user wanted a VectorWorld-style bridge from generated trajectory scenarios to multi-camera layout conditioning for downstream driving video generation, but the first step should be minimal and not commit to a specific video generator.
- Decision: Add `scripts/render_multicamera_layout.py` as an additive MVP-0 exporter. It loads normal SMART configs/checkpoints, runs inference on the original SMART `HeteroData` / `Batch` validation input, and renders six synthetic nuScenes-like pinhole camera layout streams plus `manifest.json`.
- Why: This isolates the trajectory-to-layout adapter from the much heavier video synthesis stage and keeps the SMART model interface unchanged. The exporter consumes generated trajectories after inference; it does not predict tokens and then convert tokens back into model input.
- Impact: The first layouts are camera-conditioning artifacts, not calibrated sensor renderings. They show 3D agent boxes with an approximate camera rig and do not draw road/map centerlines by default; `--draw-map-polylines` is only for geometry debugging. The next quality gate is visual inspection on real checkpoints before choosing a downstream video model.

## Decision: Add composition-based SMART hybrid diffusion
- Date: 2026-06-15
- Context: AR diffusion rollouts can over-speed straight vehicles and drift off map late, while causal diffusion safe guidance can become too conservative. The user wanted a new model path that does not inherit existing SMART predictor classes, while keeping original SMART inputs and reusing official SMART encoder/token structures.
- Decision: Add `smart_hybrid_diffusion` as an additive `pl.LightningModule` predictor that composes the causal SMART-token closed-loop rollout core instead of inheriting from existing predictor classes. The public model keeps the original SMART `HeteroData` / `Batch` input and uses `diffusion.hybrid_objective: closed_loop_frontier_v1`.
- Why: This preserves train/validation/visualization compatibility and avoids another deep inheritance layer, while letting the first hybrid experiment focus on closed-loop quality and commit-token speed calibration.
- Impact: Hybrid configs set `commit_min_speed_ratio: 0.75`, `commit_max_speed_ratio: 1.25`, and `commit_speed_reference_decay: 1.0`. The speed energy now penalizes both too-slow and too-fast executable chunk-0 tokens; tail proposals remain revisable.

## Decision: Train SMART ELF with commit-primary proposal loss
- Date: 2026-06-15
- Context: `smart_elf` inference predicts a four-token window but only commits the first token into history before rebuilding the next AR view. Equal full-window ELF supervision over-optimized uncommitted tail proposals that are revisable at inference.
- Decision: Keep the four-token prediction window, one-token commit, tail proposal carry, and rolling-anchor/model-rollout training view, but weight the ELF objective as committed chunk loss plus `elf_tail_loss_weight` times tail proposal loss. The active ELF configs set `elf_tail_loss_weight: 0.25`.
- Why: This keeps lookahead proposals available for geometry and warm-starting while making the dominant training signal match the executed receding-horizon action.
- Impact: Old `elf_ar_1000` metrics are not comparable to new ELF training losses. New ELF runs should report the tail weight and use rollout ADE/FDE, not full-window loss alone, for quality comparison.

## Decision: Discretize SMART ELF sampling from final flow embeddings
- Date: 2026-06-15
- Context: The first `smart_elf` 1000-step visualization showed that visibly moving vehicles almost all turned to one side. A diagnostic on validation indices 0-3 showed the original decoder-logit sampler produced `dist>=10m` left/straight/right counts of `0/3/25` and only 40 unique committed token ids, while the same checkpoint with final-embedding nearest-neighbor projection produced `32/15/20` and 525 unique ids.
- Decision: Use the integrated ELF embedding state as the source of truth for final token ids via `_elf_proxy_token_ids()`. Keep the auxiliary decoder logits for confidence scoring and training CE, but do not let the weak auxiliary head choose sampled token ids.
- Why: ELF's core prediction is the embedding flow. Early 1000-step auxiliary logits can collapse to a few frequent token ids even when the final embedding state carries more diverse trajectory information.
- Impact: Existing `checkpoints/elf_ar_1000/last.ckpt` can be re-evaluated with the fixed sampler without retraining. Use `outputs/elf_proxy_fix_diag/` as the first repaired visualization check.

## Decision: Build SMART ELF as AR outer loop plus non-causal embedded language flow
- Date: 2026-06-15
- Context: The user wanted an ELF-style model adapted to the sim-agent SMART task and clarified that the original ELF objective is not a causal frontier decoder. The task still needs dynamic map refresh and surrounding-agent context refresh during rollout.
- Decision: Add `smart_elf` as an independent predictor that inherits the SMART autoregressive rollout shell, but replaces the inner short-window objective with full-window embedding-space flow matching and a final token decoder. The ELF inner window uses `diffusion.elf_objective: embedded_language_flow_v1`; it does not use `causal_objective` or causal frontier sampling.
- Why: The AR outer loop preserves the receding-horizon sim-agent behavior needed for updated map/agent context, while the inner ELF objective stays faithful to ELF's non-causal full-window embedding flow idea.
- Impact: Use `configs/train/train_scalable_elf_1000.yaml` for the matched 1000-step local ELF run and `configs/validation/validation_scalable_elf.yaml` for checkpoint validation. Select ELF checkpoints with `val_minADE`, because this path logs standard AR validation metrics but not causal-specific `val_rollout_score`.

## Decision: Add a config-gated AR causal-frontier objective
- Date: 2026-06-14
- Context: The user wanted a major improvement to original `smart_ar_diffusion` by transplanting the most useful causal diffusion ideas while preserving the original AR rollout surface for comparison.
- Decision: Keep the existing `maskgit` AR objective as the baseline path and add `diffusion.ar_objective: causal_frontier_v1` as an opt-in objective. The new AR path uses causal temporal decoder edges, frontier-only supervision, current-motion state context, closed-loop model-rollout training views, and retokenized recovery targets.
- Why: This tests whether the causal diffusion advantages come from causal frontier training and closed-loop state exposure rather than from replacing the AR interface itself.
- Impact: Use `configs/train/train_scalable_ar_diffusion_baseline_1000.yaml` for the old AR baseline and `configs/train/train_scalable_ar_diffusion_frontier_local.yaml` for the new AR frontier model. Compare both against the matched causal diffusion and causal flow matching 1000-step configs before drawing quality conclusions.

## Decision: Add causal SMART-token flow matching as a separate predictor
- Date: 2026-06-13
- Context: The user wanted a flow-matching model to replace the current causal diffusion model for the same sim-agent rollout task, without deleting existing code.
- Decision: Add `smart_causal_flow_matching` as an independent predictor that inherits the causal receding-horizon rollout shell, keeps the 4-token proposal / 1-token commit interface, and replaces the discrete frontier CE/sampler with token-simplex flow matching and Euler frontier integration.
- Why: This preserves the stable training, validation, guidance, and visualization surfaces while letting flow matching be compared directly against `smart_causal_diffusion`.
- Impact: New flow runs should use the dedicated flow configs and checkpoints; existing causal diffusion configs and checkpoints remain valid.

## Decision: Start causal model-rollout training at epoch 0
- Date: 2026-06-13
- Context: Later causal checkpoints showed conservative token logits and the user wanted to remove the clean-only warmup so training sees its own closed-loop states immediately.
- Decision: `_closed_loop_curriculum()` now returns nonzero model-rollout probability from epoch 0, using `closed_loop_batch_ratio_max` as the rollout rate. Perturb state training remains disabled for epochs 0-3 and starts at 25% from epoch 4 onward.
- Why: This directly trains the causal decoder on the closed-loop state distribution used at inference instead of spending the first epochs on clean teacher-forced anchors only.
- Impact: New causal runs are not comparable to older three-phase curriculum runs without noting the curriculum change. Monitor `train_state_mode`, `train_rollout_depth`, `train_retokenization_invalid_frac`, and speed-ratio diagnostics early in training.

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

## Decision: Rebuild SMART ELF as a standalone official-ELF-style predictor
- Date: 2026-06-15
- Context: The first `smart_elf` implementation inherited `SMARTAutoregressiveDiffusion` and reused `DiffusionDecoder`, so low loss / acceptable token accuracy could still come from AR-shell or decoder-logit behavior rather than a true ELF implementation.
- Decision: Make `SMARTEmbeddedLanguageFlow` inherit only `pl.LightningModule`, compose the SMART map/history encoder, pack the full 16-token future sequence, and use an independent ELF decoder adapted from the official ELF architecture: RMSNorm/SwiGLU blocks, time prefix tokens, embedding-space flow output, and factored token decoder head.
- Why: This satisfies the requested from-scratch ELF path while keeping SMART data/token interfaces and train/validation entry points stable.
- Impact: Old `elf_ar_*` checkpoints and visualizations are architecture-incompatible with the standalone ELF model. Retrain before interpreting ELF loss, token accuracy, ADE/FDE, or visual quality.

## Decision: Make standalone SMART ELF receding-horizon for map-grounded rollout
- Date: 2026-06-16
- Context: Full-16-token standalone ELF encoded map/history once, so late generated tokens had no refreshed map query after the trajectory moved away from the original anchor.
- Decision: Keep `smart_elf` standalone, but make inference sample short ELF windows (`elf_window_tokens: 4`), commit one token (`elf_commit_tokens: 1`), write the committed anchor into the history state, and re-encode before the next window. Train the same window objective from randomly shifted GT anchors so later windows see map context around future positions.
- Why: This preserves the official-ELF-style embedded flow decoder while restoring the sim-agent invariant that each executed step gets map/history context at the current rolled state.
- Impact: Previous standalone full-window ELF checkpoints are stale. New ELF runs should use the receding configs and inspect map compliance separately from token loss.

## Decision: Use chunk-causal attention inside SMART ELF windows
- Date: 2026-06-16
- Context: The first standalone ELF decoder used only a padding key mask inside the four-token window, so chunk-0 queries could attend to chunks 1-3 and leak future-token information.
- Decision: Build a pairwise attention mask from `chunk_ids`: data queries can attend to prefix tokens and valid data keys with `key_chunk <= query_chunk`; prefix queries attend only to prefix tokens. Agents at the same chunk remain mutually visible.
- Why: This keeps same-time interaction modeling while preserving the sim-agent causality requirement that earlier committed-token decisions cannot see later token content.
- Impact: ELF loss still supervises tail chunks, but chunk-0 predictions can no longer use future chunk embeddings through attention. Retrain receding ELF checkpoints after this mask change.

## Decision: Ground SMART ELF commit selection with map-conditioned scoring
- Date: 2026-06-17
- Context: Receding ELF re-encoded map/history every committed token, but tail rollouts still drifted off map because final token ids were chosen from ELF embedding projection without an explicit map-conditioned commit score.
- Decision: Keep ELF as the primary latent proposal, but score commit candidates with a product-style combination of final ELF embedding-token similarity, a map-conditioned token scorer from the SMART history-context feature, and optional top-k map-geometry energy. Do not replace ELF with the original SMART `token_predict_head`.
- Why: This injects map attention directly into token selection while preserving a clean difference from original SMART for paper ablations.
- Impact: Existing receding ELF checkpoints are stale for quality comparisons because the new `elf_map_score_proj` and commit-scorer loss need training.

## Decision: Add AR MaskGIT rerank without frontier training
- Date: 2026-06-16
- Context: Hybrid diffusion kept causal-frontier semantics, so it did not isolate the original AR diffusion advantage for straight-vehicle speed.
- Decision: Keep `smart_ar_diffusion` on `ar_objective: maskgit` for the rerank variant, with 4-token prediction, 1-token commit, and carried tail proposals. Add sampling-time top-k reranking only for committed slots using lane, dynamics, collision, and bidirectional commit-speed energy. Add SMART-style map-token noise and history-context dropout as training perturbations, while validation keeps these perturbations disabled.
- Why: This tests the user's requested AR-first hypothesis directly: use AR rollout semantics, avoid frontier loss, and borrow only the sampling rerank plus original SMART conditioning perturbations.
- Impact: Use `configs/train/train_scalable_ar_diffusion_rerank_1000.yaml` for the matched short run and `configs/validation/validation_scalable_ar_diffusion_rerank.yaml` for deterministic evaluation.

## Decision: Trust only contiguous future-token geometry chains
- Date: 2026-06-18
- Context: Diffusion geometry refresh could skip an unknown chunk and still decode a later known/proposal token from the last stale pose, marking that later token as a confident geometry source for graph attention.
- Decision: Keep fallback query poses available for target nodes, but only assign geometry confidence and advance an agent's future pose through a contiguous chain of known or proposal-backed chunks. Sanitize proposal and geometry confidence before embedding so NaN/inf confidence cannot poison decoder activations.
- Why: A masked chunk with no proposal means the next chunk's absolute pose is not recoverable from token-relative geometry. Treating later tokens as reliable sources creates misleading spatial/map edges, especially under high mask rates or carried proposals.
- Impact: Older AR rerank/diffusion checkpoints remain loadable, but graph geometry confidence during training and sampling is now stricter after chunk gaps. Retrain or rerun diagnostics before comparing late-horizon map adherence against older outputs.

## Decision: Keep AR rollout inputs pure and stationary headings stable
- Date: 2026-06-18
- Context: AR inference wrote `commit_speed_reference` into the caller's input batch, and physical retokenization updated heading directly from token box orientation even when the decoded token had near-zero final movement.
- Decision: Keep speed-reference state local to the sampling/packed guidance path rather than mutating `data`. During physical retokenization, advance heading only when the decoded token has a non-zero final displacement, mirroring the non-physical path's norm guard.
- Why: Validation/inference callers should be able to reuse input batches without hidden new keys. Retokenization should not accumulate arbitrary heading changes from stationary token boxes.
- Impact: AR rerank checkpoints remain loadable, but future retokenized training views can differ for zero-displacement tokens. Retrain before judging late-horizon map adherence against older runs.
