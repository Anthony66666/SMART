# Training Guide

This document is the practical server-side training guide for this repo. Use it when you want to start baseline SMART training, run SMARTJEPA, or launch the two-stage JEPA pretrain plus scenario-generation finetune workflow.

## 1. Environment

If the environment already exists on the server:

```bash
cd /home/anthony/SimAgentJEPA/external/SMART
source ~/anaconda3/etc/profile.d/conda.sh
conda activate smart
```

If you still need to create it:

```bash
conda env create -f environment.yml
source ~/anaconda3/etc/profile.d/conda.sh
conda activate smart
pip install -r requirements.txt
```

Quick import check:

```bash
python -c "from smart.model import SMART, SMARTJEPA; print('ok')"
```

## 2. What To Edit Before Training

Before launching real runs, edit the config files to match the server:

- `Dataset.train_raw_dir`
- `Dataset.val_raw_dir`
- `Dataset.train_processed_dir`
- `Dataset.val_processed_dir`
- `Trainer.accelerator`
- `Trainer.devices`
- `Trainer.num_nodes`

The default JEPA configs still point to the demo dataset, so do not start training before fixing these paths.

## 3. Config Files

Baseline:

- `configs/train/train_scalable.yaml`
- `configs/validation/validation_scalable.yaml`

Default JEPA:

- `configs/train/train_scalable_jepa.yaml`
- `configs/validation/validation_scalable_jepa.yaml`

Joint JEPA ablations:

| Ablation | Meaning | Train config | Validation config |
| --- | --- | --- | --- |
| `A0` | masked-agent history fully visible | `configs/train/train_scalable_jepa_a0_visible.yaml` | `configs/validation/validation_scalable_jepa_a0_visible.yaml` |
| `A1` | masked-agent history partial dropout | `configs/train/train_scalable_jepa_a1_partial_dropout.yaml` | `configs/validation/validation_scalable_jepa_a1_partial_dropout.yaml` |
| `A2` | masked-agent history hidden | `configs/train/train_scalable_jepa_a2_hidden.yaml` | `configs/validation/validation_scalable_jepa_a2_hidden.yaml` |

JEPA pretrain ablations:

| Ablation | Meaning | Train config | Validation config |
| --- | --- | --- | --- |
| `A0-pretrain` | masked-agent history fully visible | `configs/train/train_scalable_jepa_pretrain_a0_visible.yaml` | `configs/validation/validation_scalable_jepa_pretrain_a0_visible.yaml` |
| `A1-pretrain` | masked-agent history partial dropout | `configs/train/train_scalable_jepa_pretrain_a1_partial_dropout.yaml` | `configs/validation/validation_scalable_jepa_pretrain_a1_partial_dropout.yaml` |
| `A2-pretrain` | masked-agent history hidden | `configs/train/train_scalable_jepa_pretrain_a2_hidden.yaml` | `configs/validation/validation_scalable_jepa_pretrain_a2_hidden.yaml` |

The only intended difference across `A0/A1/A2` is `Model.jepa.masked_agent_history_mode`. The pretrain configs additionally set:

- `Model.jepa.training_stage: pretrain`
- `Model.inference_token: false`
- `Visualization.enabled: false`
- `Trainer.monitor_metric: val_jepa_loss`
- `Trainer.monitor_mode: min`

## 4. Recommended Experiment Order

1. `B0`: train a baseline SMART model from scratch.
2. `B1`: train the joint SMARTJEPA baseline from scratch.
3. `P0-pretrain`: run pure JEPA pretraining, defaulting to `A1 partial_dropout`.
4. `P0-finetune`: initialize baseline SMART from the JEPA checkpoint and finetune for scenario generation.
5. Compare `B0`, `B1`, and `P0` using the same offline evaluation pipeline.

Default fair-budget comparison:

- `B1 joint`: `32` epochs
- `P0`: `16` epochs pretrain + `16` epochs finetune

If the two-stage pipeline is already better, extend it to `32 + 32`.

## 5. Baseline Commands

Train baseline:

```bash
python train.py \
  --config configs/train/train_scalable.yaml \
  --save_ckpt_path checkpoints/baseline
```

Validate baseline:

```bash
python val.py \
  --config configs/validation/validation_scalable.yaml \
  --pretrain_ckpt checkpoints/baseline/last.ckpt
```

## 6. JEPA Commands

Train default JEPA:

```bash
python train.py \
  --config configs/train/train_scalable_jepa.yaml \
  --save_ckpt_path checkpoints/jepa_default
```

Validate default JEPA:

```bash
python val.py \
  --config configs/validation/validation_scalable_jepa.yaml \
  --pretrain_ckpt checkpoints/jepa_default/last.ckpt
```

Initialize JEPA from a baseline checkpoint:

```bash
python train.py \
  --config configs/train/train_scalable_jepa.yaml \
  --pretrain_ckpt checkpoints/baseline/last.ckpt \
  --save_ckpt_path checkpoints/jepa_from_baseline
```

Resume JEPA training:

```bash
python train.py \
  --config configs/train/train_scalable_jepa.yaml \
  --save_ckpt_path checkpoints/jepa_default \
  --ckpt_path checkpoints/jepa_default/last.ckpt
```

## 7. Two-Stage JEPA Pretrain Then Finetune

Recommended main path:

- pretrain with `A1 partial_dropout`
- finetune `SMART` with the JEPA checkpoint using baseline configs

### P0 Step 1: JEPA pretrain

```bash
python train.py \
  --config configs/train/train_scalable_jepa_pretrain_a1_partial_dropout.yaml \
  --save_ckpt_path checkpoints/jepa_pretrain_a1
```

```bash
python val.py \
  --config configs/validation/validation_scalable_jepa_pretrain_a1_partial_dropout.yaml \
  --pretrain_ckpt checkpoints/jepa_pretrain_a1/last.ckpt
```

### P0 Step 2: Scenario-generation finetune with SMART

```bash
python train.py \
  --config configs/train/train_scalable.yaml \
  --pretrain_ckpt checkpoints/jepa_pretrain_a1/last.ckpt \
  --save_ckpt_path checkpoints/smart_from_jepa_a1
```

```bash
python val.py \
  --config configs/validation/validation_scalable.yaml \
  --pretrain_ckpt checkpoints/smart_from_jepa_a1/last.ckpt
```

### Offline evaluation

```bash
python eval_waymo_official.py \
  --config configs/validation/validation_scalable.yaml \
  --pretrain_ckpt checkpoints/smart_from_jepa_a1/last.ckpt
```

## 8. Joint Ablation Commands

### A0: history visible

```bash
python train.py \
  --config configs/train/train_scalable_jepa_a0_visible.yaml \
  --save_ckpt_path checkpoints/jepa_a0_visible
```

```bash
python val.py \
  --config configs/validation/validation_scalable_jepa_a0_visible.yaml \
  --pretrain_ckpt checkpoints/jepa_a0_visible/last.ckpt
```

### A1: partial history dropout

```bash
python train.py \
  --config configs/train/train_scalable_jepa_a1_partial_dropout.yaml \
  --save_ckpt_path checkpoints/jepa_a1_partial_dropout
```

```bash
python val.py \
  --config configs/validation/validation_scalable_jepa_a1_partial_dropout.yaml \
  --pretrain_ckpt checkpoints/jepa_a1_partial_dropout/last.ckpt
```

### A2: history hidden

```bash
python train.py \
  --config configs/train/train_scalable_jepa_a2_hidden.yaml \
  --save_ckpt_path checkpoints/jepa_a2_hidden
```

```bash
python val.py \
  --config configs/validation/validation_scalable_jepa_a2_hidden.yaml \
  --pretrain_ckpt checkpoints/jepa_a2_hidden/last.ckpt
```

## 9. JEPA Pretrain Ablation Commands

### A0-pretrain: history visible

```bash
python train.py \
  --config configs/train/train_scalable_jepa_pretrain_a0_visible.yaml \
  --save_ckpt_path checkpoints/jepa_pretrain_a0
```

### A1-pretrain: partial history dropout

```bash
python train.py \
  --config configs/train/train_scalable_jepa_pretrain_a1_partial_dropout.yaml \
  --save_ckpt_path checkpoints/jepa_pretrain_a1
```

### A2-pretrain: history hidden

```bash
python train.py \
  --config configs/train/train_scalable_jepa_pretrain_a2_hidden.yaml \
  --save_ckpt_path checkpoints/jepa_pretrain_a2
```

## 10. Running On A Server

Single GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/train/train_scalable_jepa_pretrain_a1_partial_dropout.yaml \
  --save_ckpt_path checkpoints/jepa_pretrain_a1
```

Background run example:

```bash
nohup python train.py \
  --config configs/train/train_scalable_jepa_pretrain_a1_partial_dropout.yaml \
  --save_ckpt_path checkpoints/jepa_pretrain_a1 \
  > outputs/train_pretrain_a1.log 2>&1 &
```

Multi-GPU runs are controlled through the config:

- `Trainer.accelerator`
- `Trainer.devices`
- `Trainer.num_nodes`

`train.py` already uses Lightning `DDPStrategy`, so you normally do not need to modify the launcher script.

## 11. What To Monitor

Baseline:

- `train_loss`
- `cls_loss`
- `val_cls_acc`
- `val_loss`
- `val_minADE`
- `val_minFDE`

JEPA:

- `jepa_loss`
- `jepa_cosine`
- `agent_jepa_loss`
- `map_jepa_loss`
- `masked_agent_count`
- `masked_map_count`
- `masked_agent_history_visible_fraction`
- `val_jepa_loss`

Joint JEPA only:

- `val_total_loss`
- `val_minADE`
- `val_minFDE`

Two-stage finetune only:

- `val_cls_acc`
- `val_minADE`
- `val_minFDE`

Interpretation of the new history metric:

- `A0` should stay near `1.0`
- `A1` should stay between `0` and `1`
- `A2` should stay near `0.0`

If those values do not match the intended ablation, the wrong config is being used.

## 12. Debugging The JEPA Mask

To visualize the current joint agent-map mask and confirm the history mode:

```bash
python scripts/visualize_jepa_masks.py \
  --config configs/train/train_scalable_jepa_pretrain_a1_partial_dropout.yaml \
  --split train \
  --indices 0 1 2 3 \
  --output-dir outputs/jepa_mask_debug_a1
```

The figure title box will show:

- masked agent count
- masked polygon count
- `history=<mode>`
- `hist vis=<fraction>`

## 13. Common Failure Modes

### Wrong dataset paths

The configs still use demo paths by default. Fix the dataset paths before running.

### Wrong validation config

Always validate with the matching validation config:

- `A0` train pairs with `A0` validation
- `A1` train pairs with `A1` validation
- `A2` train pairs with `A2` validation
- `A0-pretrain` train pairs with `A0-pretrain` validation
- `A1-pretrain` train pairs with `A1-pretrain` validation
- `A2-pretrain` train pairs with `A2-pretrain` validation

### Invalid future chunk setup

`num_future_steps` must be divisible by `Model.jepa.future_chunk_steps`.

### Checkpoint load confusion

- `--pretrain_ckpt` means initialize model weights from a checkpoint
- `--ckpt_path` means resume a Lightning run from an existing training checkpoint

### Watching only token accuracy

For JEPA pretraining, do not judge runs using only `val_cls_acc`. The primary comparison should be:

- `val_minADE`
- `val_minFDE`
- `val_jepa_loss`
- `val_agent_jepa_loss`
- `val_map_jepa_loss`

## 14. Minimal Start-From-Zero Workflow

```bash
cd /home/anthony/SimAgentJEPA/external/SMART
source ~/anaconda3/etc/profile.d/conda.sh
conda activate smart

# edit the config paths first

# B0 baseline
python train.py --config configs/train/train_scalable.yaml --save_ckpt_path checkpoints/baseline
python val.py --config configs/validation/validation_scalable.yaml --pretrain_ckpt checkpoints/baseline/last.ckpt

# B1 joint JEPA
python train.py --config configs/train/train_scalable_jepa_a1_partial_dropout.yaml --save_ckpt_path checkpoints/jepa_joint_a1

# P0 JEPA pretrain -> SMART finetune
python train.py --config configs/train/train_scalable_jepa_pretrain_a1_partial_dropout.yaml --save_ckpt_path checkpoints/jepa_pretrain_a1
python train.py --config configs/train/train_scalable.yaml --pretrain_ckpt checkpoints/jepa_pretrain_a1/last.ckpt --save_ckpt_path checkpoints/smart_from_jepa_a1
```
