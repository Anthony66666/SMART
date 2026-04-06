# SMART Fork for JEPA Experiments

这个仓库是基于原始 SMART 改出的研究分支，用来做：

- 原始 SMART baseline 复现
- `SMARTJEPA` 实验
- 在不改变 SMART rollout / inference 接口的前提下，把 JEPA 作为训练期辅助目标接入

当前建议把这个仓库当成你的正式算法实现仓库；外围的 `SimAgentJEPA` 仓库继续只做脚手架、可视化和补充评估。

## Current Status

当前分支支持两条模型线：

- `predictor: smart`
  原始 SMART baseline
- `predictor: smart_jepa`
  新增的 `SMARTJEPA`

`SMARTJEPA` 的设计原则是：

- 不改数据预处理格式
- 不改 tokenization 流程
- 不改 inference / rollout 接口
- 只在训练期增加 JEPA auxiliary loss

JEPA 版本当前已经包含：

- `FutureBlockEncoder`
- `JointEmbeddingPredictiveModule`
- history-only scene context 编码
- EMA target future encoder
- 独立训练/验证配置

## Repository Layout

你最应该关心的几个文件：

- `train.py`
  训练入口，支持 `smart` 和 `smart_jepa`
- `val.py`
  验证入口，支持 `smart` 和 `smart_jepa`
- `smart/model/smart.py`
  原始 SMART LightningModule
- `smart/model/smart_jepa.py`
  新增的 JEPA 版本 LightningModule
- `smart/model/jepa.py`
  JEPA 模块本体
- `smart/modules/smart_decoder.py`
  decoder 总入口，新增了 history-only context 编码接口
- `smart/modules/agent_decoder.py`
  agent/map/history 交互主逻辑
- `configs/train/train_scalable.yaml`
  baseline 训练配置
- `configs/train/train_scalable_jepa.yaml`
  JEPA 训练配置
- `configs/validation/validation_scalable.yaml`
  baseline 验证配置
- `configs/validation/validation_scalable_jepa.yaml`
  JEPA 验证配置

## Recommended Git Workflow

建议一直在你的 fork 分支上工作，不要直接在 `main` 上改：

```bash
git checkout codex/jepa-smart-v1
git pull --ff-only origin codex/jepa-smart-v1
```

同步上游 SMART：

```bash
git fetch upstream
git checkout main
git pull --ff-only upstream main
git checkout codex/jepa-smart-v1
git rebase main
git push --force-with-lease
```

日常提交：

```bash
git add <changed-files>
git commit -m "your message"
git push
```

## Environment Setup

### 1. Create the environment

```bash
conda env create -f environment.yml
conda activate SMART
pip install -r requirements.txt
```

如果 PyG 相关依赖安装失败，先运行：

```bash
bash install_pyg.sh
```

### 2. Verify the environment

建议先跑一个最简单的导入检查：

```bash
python -c "from smart.model import SMART, SMARTJEPA; print('ok')"
```

如果你的服务器上出现 OpenMP / shared memory 问题，优先检查：

- `torch`
- `torch_geometric`
- `torch_cluster`
- `pytorch_lightning`

以及当前服务器的共享内存和 OpenMP 运行环境。

## Data Preparation

### 1. Download WOMD scenario data

建议组织成：

```text
SMART
├── data
│   ├── waymo
│   │   ├── scenario
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── testing
```

### 2. Install Waymo Open Dataset API

按照官方仓库安装 Waymo Open Dataset API。

### 3. Preprocess the data

训练集：

```bash
python data_preprocess.py \
  --input_dir ./data/waymo/scenario/training \
  --output_dir ./data/waymo_processed/training
```

验证集：

```bash
python data_preprocess.py \
  --input_dir ./data/waymo/scenario/validation \
  --output_dir ./data/waymo_processed/validation
```

处理后的目录建议组织成：

```text
SMART
├── data
│   ├── waymo_processed
│   │   ├── training
│   │   ├── validation
│   │   ├── testing
```

## Configuration Guide

### Baseline config

baseline 默认配置：

- `configs/train/train_scalable.yaml`
- `configs/validation/validation_scalable.yaml`

需要至少改这些字段：

- `Dataset.train_raw_dir`
- `Dataset.val_raw_dir`
- `Dataset.train_processed_dir`
- `Dataset.val_processed_dir`
- `Trainer.accelerator`
- `Trainer.devices`

### JEPA config

JEPA 默认配置：

- `configs/train/train_scalable_jepa.yaml`
- `configs/validation/validation_scalable_jepa.yaml`

除了和 baseline 一样要改数据路径、设备参数，还要注意：

- `Model.predictor: smart_jepa`
- `Trainer.monitor_metric: val_minADE`
- `Trainer.monitor_mode: min`
- `Model.inference_token: True`

JEPA 子配置在：

```yaml
Model:
  jepa:
    enabled: true
    future_chunk_steps: 5
    mask_ratio: 0.5
    ema_decay: 0.99
    aux_loss_weight: 0.25
```

默认约束：

- `num_future_steps` 必须能被 `future_chunk_steps` 整除
- 当前默认 `80 / 5 = 16` 个 future blocks

## Training

### 1. Train baseline SMART

```bash
python train.py \
  --config configs/train/train_scalable.yaml \
  --save_ckpt_path /path/to/checkpoints/smart_baseline
```

### 2. Train SMARTJEPA

```bash
python train.py \
  --config configs/train/train_scalable_jepa.yaml \
  --save_ckpt_path /path/to/checkpoints/smart_jepa
```

### 3. Resume training

Lightning resume：

```bash
python train.py \
  --config configs/train/train_scalable_jepa.yaml \
  --save_ckpt_path /path/to/checkpoints/smart_jepa \
  --ckpt_path /path/to/checkpoints/smart_jepa/last.ckpt
```

### 4. Initialize SMARTJEPA from baseline SMART checkpoint

`SMARTJEPA` 保留了 shape-compatible checkpoint 加载逻辑，所以可以先训练 baseline，再拿 baseline ckpt 初始化 JEPA：

```bash
python train.py \
  --config configs/train/train_scalable_jepa.yaml \
  --pretrain_ckpt /path/to/baseline.ckpt \
  --save_ckpt_path /path/to/checkpoints/smart_jepa
```

## Validation

### Validate baseline

```bash
python val.py \
  --config configs/validation/validation_scalable.yaml \
  --pretrain_ckpt /path/to/baseline.ckpt
```

### Validate SMARTJEPA

```bash
python val.py \
  --config configs/validation/validation_scalable_jepa.yaml \
  --pretrain_ckpt /path/to/smart_jepa.ckpt
```

## Recommended Experiment Order

建议按这个顺序做：

1. 跑 baseline 的最小 smoke test
2. 跑 JEPA 的最小 smoke test
3. 训练 baseline 正式模型
4. 训练 SMARTJEPA 正式模型
5. 在同一 validation protocol 下比较：
   - `val_cls_acc`
   - `val_loss`
   - `val_minADE`
   - `val_minFDE`
   - `val_jepa_loss`
   - `val_total_loss`

## What SMARTJEPA Actually Changes

`SMARTJEPA` 相比原始 SMART，核心变化只有训练目标：

- baseline SMART：
  只优化 token classification loss
- SMARTJEPA：
  优化 `cls_loss + aux_loss_weight * jepa_loss`

JEPA 部分做的是：

- 从历史可见 token 编码 history-only context
- 从真实未来轨迹在线构造 future blocks
- 对 mask 掉的 future latent blocks 做预测
- 使用 EMA target encoder 提供目标表示

重要的是：

- inference 路径保持和原始 SMART 一样
- rollout 机制保持和原始 SMART 一样
- challenge submission 接口不因为 JEPA 改变

## Logging And Monitoring

baseline 训练默认主要看：

- `train_loss`
- `cls_loss`
- `val_cls_acc`
- `val_loss`

SMARTJEPA 会额外记录：

- `jepa_loss`
- `jepa_cosine`
- `jepa_masked_fraction`
- `val_jepa_loss`
- `val_total_loss`
- `val_jepa_cosine`

checkpoint monitor 现在由 config 控制：

- baseline 默认建议：
  - `monitor_metric: val_cls_acc`
  - `monitor_mode: max`
- JEPA 默认建议：
  - `monitor_metric: val_minADE`
  - `monitor_mode: min`

## Smoke Tests

开始正式训练前，建议至少做下面 3 件事：

### 1. Syntax check

```bash
PYTHONPYCACHEPREFIX=/tmp/pycache \
python3 -m compileall train.py val.py smart/model smart/modules
```

### 2. One-batch / one-epoch debug run

把 config 里的数据路径改成小样本，先确认：

- baseline 能正常 forward / backward
- JEPA 能正常 forward / backward
- validation 能正常跑

### 3. Check checkpoint loading

至少验证两种情况：

- baseline ckpt -> baseline
- baseline ckpt -> SMARTJEPA

## Common Pitfalls

### 1. `main` 和实验分支混着改

不要在 `main` 上直接做实验。始终在：

```bash
git checkout codex/jepa-smart-v1
```

### 2. 训练和验证用错配置

baseline 和 JEPA 配置是分开的，注意不要混用：

- baseline 用 `train_scalable.yaml`
- JEPA 用 `train_scalable_jepa.yaml`

### 3. 数据路径没改

这套配置文件默认还是 demo 路径，必须改成你的服务器路径。

### 4. `future_chunk_steps` 不整除

如果你把 `num_future_steps` 改了，一定同步检查：

```text
num_future_steps % future_chunk_steps == 0
```

### 5. 只看 token acc 不看 rollout

JEPA 可能不会提升 token classification，但会改善 rollout 几何误差，所以要同时看：

- `val_cls_acc`
- `val_minADE`
- `val_minFDE`

## Suggested Server Workflow

如果你在服务器上从头开始，建议按这个顺序：

```bash
cd ~/SimAgentJEPA/external/SMART
git checkout codex/jepa-smart-v1
git pull

conda env create -f environment.yml
conda activate SMART
pip install -r requirements.txt

python data_preprocess.py --input_dir <training_raw> --output_dir <training_processed>
python data_preprocess.py --input_dir <validation_raw> --output_dir <validation_processed>

# 先跑 baseline
python train.py --config configs/train/train_scalable.yaml --save_ckpt_path <baseline_ckpt_dir>
python val.py --config configs/validation/validation_scalable.yaml --pretrain_ckpt <baseline_ckpt>

# 再跑 JEPA
python train.py --config configs/train/train_scalable_jepa.yaml --save_ckpt_path <jepa_ckpt_dir>
python val.py --config configs/validation/validation_scalable_jepa.yaml --pretrain_ckpt <jepa_ckpt>
```

## Citation

如果你使用原始 SMART，请引用原论文：

```bibtex
@article{wu2024smart,
  title={SMART: Scalable Multi-agent Real-time Simulation via Next-token Prediction},
  author={Wu, Wei and Feng, Xiaoxin and Gao, Ziyan and Kan, Yuheng},
  journal={arXiv preprint arXiv:2405.15677},
  year={2024}
}
```

如果你后续写 `SMARTJEPA` 论文，建议把本 fork 的改动、commit SHA 和实验配置一起记录清楚。

## Acknowledgements

原始 SMART 工作受 [QCNet](https://github.com/ZikangZhou/QCNet) 启发。这个 fork 在研究流程上也借助了外层 `SimAgentJEPA` 仓库做快速原型、可视化和补充评估。
