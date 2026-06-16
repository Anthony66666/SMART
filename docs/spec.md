# Spec

## Goal

Maintain this repository as the primary implementation repo for SMART baseline reproduction and SMART-Diffusion experiments, while keeping SMART data/token interfaces stable. Historical JEPA files and configs may remain in the tree, but JEPA is no longer an active train/validation pipeline.

## Scope

- Baseline SMART training and validation flows
- SMART-Diffusion discrete mask diffusion training and inference experiments
- SMART-AR causal-frontier experiments that preserve the AR rollout interface while testing causal diffusion training ideas
- Inference-time causal diffusion guidance for safe rollout and ego-centric safety-critical counterfactual editing without retraining
- SMART causal flow-matching experiments that reuse the same causal rollout and visualization interfaces as causal diffusion
- SMART embedded-language-flow experiments implemented as a standalone predictor: compose the SMART map/history encoder, then run official-ELF-style short-window embedding flow in a receding-horizon loop without inheriting AR, causal, or diffusion predictors
- SMART hybrid diffusion experiments that keep the original SMART input batch, avoid existing predictor inheritance, and combine closed-loop causal frontier rollout with bidirectional commit-speed calibration
- Multi-camera layout export for SMART-generated scenarios, producing camera-view conditioning frames for downstream driving video generation without changing SMART model inputs
- Config, evaluation, visualization, and workflow support needed to compare baseline SMART and SMART-Diffusion
- Historical JEPA code/config retention without active `train.py` / `val.py` predictor support
- Repo-local operating workflow for graph-first code exploration and durable task memory

## Non-goals

- Replacing the external `SimAgentJEPA` scaffold repo
- Redesigning the SMART inference interface
- Deleting historical JEPA source files or configs
- Building a full external vector database or hosted memory service for this repo

## Constraints

- Preserve existing training and validation entry points (`train.py`, `val.py`)
- Preserve data preprocessing and tokenization interfaces unless a task explicitly changes them
- Prefer graph-based repository discovery before broad file scanning
- Keep persistent task memory in workspace Markdown files under `docs/`

## Acceptance Criteria

- The repo can be explored with code graph tools before broad grep-based search
- Durable task state survives long sessions through `docs/spec.md`, `docs/progress.md`, `docs/decisions.md`, and `docs/next.md`
- New tasks can be resumed from repo-local files without reconstructing context from chat history alone

## Current Plan

1. Use the code graph for architecture and symbol discovery.
2. Read only the files needed to execute the current task.
3. Keep SMART-Diffusion aligned with SMART/QCNet-style graph-relative geometry: raw coordinates may be used for neighbor search, but embeddings should consume edge-relative features.
4. Represent diffusion map context as flat/ragged tokens with packed-scene batch ids, not per-scene padding.
5. Refresh future-token graph geometry from currently unmasked diffusion tokens during training and sampling, with training-time geometry dropout to reduce train/inference mismatch.
6. Compare SMART-Diffusion variants with SMART parity: full-horizon joint MaskGIT denoising remains available, while `smart_ar_diffusion` now uses a receding-horizon rollout by default: 2-token history, 4-token prediction, 1-token commit, uncommitted tail-token proposal carry, and training-time visible-token neighbor corruption. Causal chunk schedules remain available as disabled ablations and must preserve the global diffusion timestep semantics when enabled. The opt-in `smart_ar_diffusion` causal-frontier objective preserves the AR interface while testing causal temporal edges, frontier-only loss, current-state context, closed-loop training views, and retokenized recovery targets. The opt-in AR rerank variant keeps the MaskGIT AR objective, avoids frontier loss, uses causal temporal token attention, and adds final commit-token safe-speed reranking plus SMART-style map/history conditioning perturbations.
7. Train `smart_causal_diffusion` v2 from scratch as the road-stability redesign: discrete frontier supervision, four-token revisable plans with one-token commits, proposal carry, recency/current-motion context, current-anchor interaction edges, epoch-0 model-rollout curriculum, retokenization recovery, and commit-aware safety reranking.
8. Use `smart_causal_diffusion` checkpoints for inference-time ego-centric scene editing through `guidance.mode = none | safe | ego_stress | ego_edit`; keep the training objective unchanged while steering editable target agents with a generic ego-risk / low-TTC objective. Predefined event targets such as cut-in or lead-hard-brake are legacy ablations, not the default research direction.
9. Keep `smart_causal_flow_matching` additive: it should not delete or rename the causal diffusion path, and should preserve the same receding-horizon sim-agent inference output surface for validation and visualization.
10. Keep `smart_elf` additive but standalone: do not inherit `SMARTAutoregressiveDiffusion`, `SMARTCausalDiffusion`, or `SMARTDiffusion`. Compose the SMART map/history encoder, sample four-token ELF windows, commit one token per round, roll the committed anchor into the history state, and re-encode before the next window so map queries follow the generated trajectory. Train window views from randomly shifted GT anchors; keep chunk 0 full weight and downweight later window chunks with `elf_tail_loss_weight`.
11. Keep `smart_hybrid_diffusion` additive: expose a new Lightning predictor that does not inherit existing SMART predictor classes, keeps original SMART `HeteroData` / `Batch` inputs, reuses the causal closed-loop SMART-token rollout surface by composition, and replaces one-sided slow-token speed reranking with a bidirectional commit-speed band.
12. Keep the first multi-camera video-conditioning bridge as an additive exporter: consume normal SMART validation batches/checkpoints, render synthetic camera-view layout PNGs plus a manifest, and defer calibrated real-camera projection or video-model integration until the layout artifact is inspected.
13. Keep causal LR scheduling epoch-based: 32-epoch server runs use `warmup_steps: 2`, `total_steps: 32`, and encoder LR scale 0.5.
14. After meaningful work, update `docs/progress.md` and refresh `docs/next.md`.
15. Update `docs/spec.md` and `docs/decisions.md` only when project direction or durable decisions change.

## Risks

- Graph indexes can become stale if the local MCP service is unavailable
- Memory files become noisy if every trivial action is logged
- Repo state can drift if workflow instructions are ignored in future sessions
