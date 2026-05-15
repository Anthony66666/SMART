# Spec

## Goal

Maintain this repository as the primary implementation repo for SMART baseline reproduction and SMART-Diffusion experiments, while keeping SMART data/token interfaces stable. Historical JEPA files and configs may remain in the tree, but JEPA is no longer an active train/validation pipeline.

## Scope

- Baseline SMART training and validation flows
- SMART-Diffusion discrete mask diffusion training and inference experiments
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
6. Train SMART-Diffusion with low-variance timestep/mask selection, SMART target-category alignment, and MaskGIT-style remask/resample sampling.
7. Use temporal block diffusion by default: train on one future chunk block at a time, condition on previous blocks, exclude future blocks from decoder attention, and sample blocks autoregressively.
8. After meaningful work, update `docs/progress.md` and refresh `docs/next.md`.
9. Update `docs/spec.md` and `docs/decisions.md` only when project direction or durable decisions change.

## Risks

- Graph indexes can become stale if the local MCP service is unavailable
- Memory files become noisy if every trivial action is logged
- Repo state can drift if workflow instructions are ignored in future sessions
