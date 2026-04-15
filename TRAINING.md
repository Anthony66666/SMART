# Training Guide

This document is the practical server-side training guide for this repo. Use it when you want to start baseline SMART training, run SMARTJEPA, or launch the three masked-agent-history ablations.

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

JEPA ablations:

| Ablation | Meaning | Train config | Validation config |
| --- | --- | --- | --- |
| `A0` | masked-agent history fully visible | `configs/train/train_scalable_jepa_a0_visible.yaml` | `configs/validation/validation_scalable_jepa_a0_visible.yaml` |
| `A1` | masked-agent history partial dropout | `configs/train/train_scalable_jepa_a1_partial_dropout.yaml` | `configs/validation/validation_scalable_jepa_a1_partial_dropout.yaml` |
| `A2` | masked-agent history hidden | `configs/train/train_scalable_jepa_a2_hidden.yaml` | `configs/validation/validation_scalable_jepa_a2_hidden.yaml` |

The only intended difference across `A0/A1/A2` is `Model.jepa.masked_agent_history_mode`.

## 4. Recommended Experiment Order

1. Train a baseline SMART model.
2. Validate the baseline checkpoint.
3. Train `A0`.
4. Train `A1`.
5. Train `A2`.
6. Compare `val_minADE`, `val_minFDE`, `val_jepa_loss`, `val_agent_jepa_loss`, `val_map_jepa_loss`.

If you want JEPA to start from a trained baseline checkpoint, use `--pretrain_ckpt`.

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

## 7. Ablation Commands

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

## 8. Running On A Server

Single GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --config configs/train/train_scalable_jepa_a0_visible.yaml \
  --save_ckpt_path checkpoints/jepa_a0_visible
```

Background run example:

```bash
nohup python train.py \
  --config configs/train/train_scalable_jepa_a0_visible.yaml \
  --save_ckpt_path checkpoints/jepa_a0_visible \
  > outputs/train_a0.log 2>&1 &
```

Multi-GPU runs are controlled through the config:

- `Trainer.accelerator`
- `Trainer.devices`
- `Trainer.num_nodes`

`train.py` already uses Lightning `DDPStrategy`, so you normally do not need to modify the launcher script.

## 9. What To Monitor

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
- `val_total_loss`
- `val_minADE`
- `val_minFDE`

Interpretation of the new history metric:

- `A0` should stay near `1.0`
- `A1` should stay between `0` and `1`
- `A2` should stay near `0.0`

If those values do not match the intended ablation, the wrong config is being used.

## 10. Debugging The JEPA Mask

To visualize the current joint agent-map mask and confirm the history mode:

```bash
python scripts/visualize_jepa_masks.py \
  --config configs/train/train_scalable_jepa_a1_partial_dropout.yaml \
  --split train \
  --indices 0 1 2 3 \
  --output-dir outputs/jepa_mask_debug_a1
```

The figure title box will show:

- masked agent count
- masked polygon count
- `history=<mode>`
- `hist vis=<fraction>`

## 11. Common Failure Modes

### Wrong dataset paths

The configs still use demo paths by default. Fix the dataset paths before running.

### Wrong validation config

Always validate with the matching validation config:

- `A0` train pairs with `A0` validation
- `A1` train pairs with `A1` validation
- `A2` train pairs with `A2` validation

### Invalid future chunk setup

`num_future_steps` must be divisible by `Model.jepa.future_chunk_steps`.

### Checkpoint load confusion

- `--pretrain_ckpt` means initialize model weights from a checkpoint
- `--ckpt_path` means resume a Lightning run from an existing training checkpoint

### Watching only token accuracy

For JEPA, do not judge runs using only `val_cls_acc`. The primary comparison should be:

- `val_minADE`
- `val_minFDE`
- `val_jepa_loss`
- `val_agent_jepa_loss`
- `val_map_jepa_loss`

## 12. Minimal Start-From-Zero Workflow

```bash
cd /home/anthony/SimAgentJEPA/external/SMART
source ~/anaconda3/etc/profile.d/conda.sh
conda activate smart

# edit the config paths first

# baseline
python train.py --config configs/train/train_scalable.yaml --save_ckpt_path checkpoints/baseline
python val.py --config configs/validation/validation_scalable.yaml --pretrain_ckpt checkpoints/baseline/last.ckpt

# JEPA ablations
python train.py --config configs/train/train_scalable_jepa_a0_visible.yaml --save_ckpt_path checkpoints/jepa_a0_visible
python train.py --config configs/train/train_scalable_jepa_a1_partial_dropout.yaml --save_ckpt_path checkpoints/jepa_a1_partial_dropout
python train.py --config configs/train/train_scalable_jepa_a2_hidden.yaml --save_ckpt_path checkpoints/jepa_a2_hidden
```
