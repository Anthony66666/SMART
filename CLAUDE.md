# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and Commands

```bash
# Activate environment
conda activate smart

# Verify imports
python -c "from smart.model import SMART, SMARTJEPA; print('ok')"

# Preprocess Waymo data
python data_preprocess.py --input_dir <raw_dir> --output_dir <processed_dir>

# Syntax check
python -m compileall train.py val.py smart/model smart/modules
```

**Training:**
```bash
# From scratch
python train.py --config configs/train/train_scalable.yaml --save_ckpt_path checkpoints/baseline
# From pretrained weights (initialize model params, not Lightning resume)
python train.py --config configs/train/train_scalable_jepa.yaml --pretrain_ckpt <ckpt> --save_ckpt_path checkpoints/jepa
# Resume training (Lightning resume with optimizer state)
python train.py --config configs/train/train_scalable_jepa.yaml --save_ckpt_path checkpoints/jepa --ckpt_path checkpoints/jepa/last.ckpt
```

**Validation:**
```bash
python val.py --config configs/validation/validation_scalable.yaml --pretrain_ckpt <ckpt>
```

**Offline evaluation (Waymo official metrics):**
```bash
python eval_waymo_official.py --config configs/validation/validation_scalable.yaml --pretrain_ckpt <ckpt>
```

**JEPA mask debugging:**
```bash
python scripts/visualize_jepa_masks.py --config <config> --split train --indices 0 1 2 3 --output-dir outputs/jepa_mask_debug
```

## Config System

Configs are YAML loaded as `EasyDict` via `smart.utils.config.load_config_act()`. The model predictor class is selected by `Model.predictor` in the config: `"smart"` → `SMART`, `"smart_jepa"` → `SMARTJEPA`.

Always pair training and validation configs:
- Baseline: `train_scalable.yaml` ↔ `validation_scalable.yaml`
- JEPA: `train_scalable_jepa*.yaml` ↔ `validation_scalable_jepa*.yaml`

Config fields that must be edited per-server: `Dataset.train_raw_dir`, `Dataset.val_raw_dir`, `Dataset.train_processed_dir`, `Dataset.val_processed_dir`, `Trainer.accelerator`, `Trainer.devices`, `Trainer.num_nodes`.

## Architecture

### Model Hierarchy

```
pl.LightningModule
├── SMART                    # Baseline: next-token prediction
│   └── SMARTDecoder         # Combines map + agent encoders
│       ├── SMARTMapDecoder  # Encode map polylines/polygons via GNN
│       └── SMARTAgentDecoder # Encode agents via GNN + token prediction head
│           - shift = 5      # History token granularity (every 0.5s)
│           - Tokenized trajectory prediction (2048 clusters from K-means)
└── SMARTJEPA(SMART)         # Inherits all SMART structure, adds JEPA module
    └── JointEmbeddingPredictiveModule  # Self-supervised future prediction
        ├── FutureBlockEncoder (online + EMA target)
        ├── MapBlockEncoder (online + EMA target)
        └── predictor (2-layer TransformerEncoder)
    └── target_map_backbone  # Separate EMA copy of map_encoder for stability
```

### JEPA Training Flow

1. **Context encoding**: `encoder.encode_history_context(data, map_visible_mask, agent_history_mask)` extracts history-only features from map tokens and agent history — this is a separate path from the main SMART forward.

2. **Future target building**: `_build_future_targets()` extracts future positions/headings/speeds relative to the last history step, grouped into chunks of `future_chunk_steps` (default 5).

3. **Mask construction**: `_build_jepa_masks()` decides which agent chunks and map polygons to mask. Two mask strategies:
   - `random_chunk`: per-agent random chunk masking
   - `interaction_multiblock`: spatially-aware joint agent+map masking with anchor-based region selection
   - `forecast_aligned_ego30_v1` (pretrain_objective): ego-centered 30m radius regions with full-future supervision

4. **JEPA module forward**: Online encoder produces latents for visible tokens + mask tokens for hidden ones → 2-layer Transformer predictor → cosine similarity loss against EMA target encoder output.

5. **Loss**: Joint mode = `cls_loss + aux_loss_weight * jepa_loss`; Pretrain mode = `jepa_loss` only.

### Key Architecture Invariants

- `num_future_steps % future_chunk_steps == 0` (e.g., 80 // 5 = 16 chunks)
- `num_historical_steps % agent_encoder.shift == 0` (history tokens every 5 steps)
- `SMARTJEPA` preserves the SMART inference/rollout interface — only the training objective changes
- `--pretrain_ckpt` initializes model weights (shape-compatible partial load); `--ckpt_path` resumes a Lightning run with full training state
- Map encoder weights are EMA-tracked separately (`target_map_backbone`) to prevent target drift from the online encoder
- `find_unused_parameters=True` in DDPStrategy — needed because JEPA target encoders skip gradients

### Data Flow

```
Waymo scenarios → data_preprocess.py → .pkl files
    → MultiDataset (torch_geometric Dataset) → HeteroData graphs
        → MultiDataModule (Lightning DataModule) → DataLoader
            → train.py / val.py → SMART / SMARTJEPA
```

Each `HeteroData` graph has node types: `agent`, `map_polygon`, `pt_token`. The graph is batched with `torch_geometric.data.Batch`.

## Docs Convention

This repo uses `docs/*.md` for durable task memory:
- `docs/spec.md` — current goal, scope, constraints
- `docs/progress.md` — one entry per meaningful completed task
- `docs/decisions.md` — technical decisions and rationale
- `docs/next.md` — live next actions and blockers

Read these before substantial work; update them after meaningful changes.
