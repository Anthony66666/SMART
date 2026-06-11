# SMART Fork for JEPA Experiments

这个仓库是基于原始 SMART 改出的研究分支，用来做：

- 原始 SMART baseline 复现
- `SMARTJEPA` 实验
- 在不改变 SMART rollout / inference 接口的前提下，把 JEPA 作为训练期辅助目标接入

当前建议把这个仓库当成你的正式算法实现仓库；外围的 `SimAgentJEPA` 仓库继续只做脚手架、可视化和补充评估。

服务器训练和 ablation 启动说明见：

- `TRAINING.md`

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

## Validation Visualization During Training

训练现在支持自动从固定的 `val` 样本保存可视化图，用来直接比较 baseline 和 `SMARTJEPA`。

当前实现方式是：

- 在每个 validation epoch 结束后触发
- 从 `Visualization.sample_indices` 指定的固定索引读取样本
- 对每个样本运行 `model.inference(...)`
- 保存地图、历史、GT 和预测轨迹图

这样只要 baseline 和 JEPA 使用同一组 `sample_indices`，两边的图就是一一对应的。

默认训练配置里已经把两条线都设成了：

```yaml
Visualization:
  enabled: true
  interval_epochs: 1
  sample_indices: [0, 1, 2, 3]
  output_dir: "./outputs/val_visualizations"
  max_agents: 0
```

输出目录结构是：

```text
outputs/val_visualizations/
├── smart/
│   ├── epoch_001/
│   ├── epoch_002/
├── smart_jepa/
│   ├── epoch_001/
│   ├── epoch_002/
```

所以你后面比较时，直接对齐看：

- `smart/epoch_005/idx_00000_*.png`
- `smart_jepa/epoch_005/idx_00000_*.png`

如果你想改固定样本，只需要同时修改两份训练配置里的：

- `configs/train/train_scalable.yaml`
- `configs/train/train_scalable_jepa.yaml`

并保证 `sample_indices` 完全一致。

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

## Causal Diffusion Server Training

下面是 `smart_causal_diffusion` 的服务器训练流程。该模型应当从头训练，不要加载旧的
`smart_ar_diffusion` checkpoint。

### Local small-data training

本地配置默认使用仓库中的 11 个 Waymo demo：

```bash
cd /home/anthony/SimAgentJEPA/external/SMART
source /home/anthony/anaconda3/etc/profile.d/conda.sh
conda activate smart

mkdir -p checkpoints/causal_v2_local

CUDA_VISIBLE_DEVICES=0 python -u train.py \
  --config configs/train/train_scalable_causal_diffusion_local.yaml \
  --save_ckpt_path checkpoints/causal_v2_local
```

该配置训练 5 个 epoch、每个 epoch 只验证 1 个 batch，并默认关闭额外
visualization rollout。训练日志可用下面的命令查看：

```bash
tensorboard --logdir lightning_logs --port 6006
```

这是训练链路和过拟合能力检查，不用于判断完整 Waymo 数据集上的最终效果。

### 1. Update the repository

```bash
cd /path/to/SMART
git checkout codex/smart-discrete-diffusion-noblock
git pull --ff-only origin codex/smart-discrete-diffusion-noblock
git log --oneline -5
```

确认提交历史中包含：

```text
64aa4ed Add causal closed-loop SMART diffusion
```

### 2. Activate and verify the environment

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate smart

python -c "import torch; print(torch.__version__); print(torch.cuda.device_count())"
python -c "from smart.model import SMARTCausalDiffusion; print('import ok')"
nvidia-smi
```

检查 SMART trajectory token 文件：

```bash
ls -lh \
  smart/tokens/cluster_frame_5_2048.pkl \
  smart/tokens/map_traj_token5.pkl
```

### 3. Configure the training run

编辑：

```bash
vim configs/train/train_scalable_causal_diffusion.yaml
```

至少确认以下字段：

```yaml
Dataset:
  train_raw_dir: ["/path/to/waymo/training"]
  val_raw_dir: ["/path/to/waymo/validation"]
  train_batch_size: 4
  val_batch_size: 1
  num_workers: 4

Trainer:
  devices: 4
  max_epochs: 32

Model:
  warmup_steps: 2
  total_steps: 32
  diffusion:
    causal_objective: discrete_frontier_v2
    prediction_tokens: 4
    commit_tokens: 1
    carry_tail_proposal: true
    proposal_conditioning_enabled: true
    current_state_enabled: true
    current_state_edges: true
    closed_loop_batch_ratio_max: 0.5
    retokenization_error_thresholds:
      [0.7379697561264038, 0.8562850952148438, 1.2705252170562744]
```

当前 `LambdaLR` 由 PyTorch Lightning 按 epoch 更新，因此这里的
`warmup_steps` 和 `total_steps` 都表示 epoch。不要使用旧的
`warmup_steps: 2000` 和 `total_steps: 120000`，否则学习率会长期过低。

GPU 数量可以调整，但 `Trainer.devices` 必须与 `CUDA_VISIBLE_DEVICES`
暴露的 GPU 数量一致。

### 4. Configure standalone validation

编辑：

```bash
vim configs/validation/validation_scalable_causal_diffusion.yaml
```

至少修改：

```yaml
Dataset:
  val_raw_dir: ["/path/to/waymo/validation"]

Trainer:
  devices: 1

Model:
  warmup_steps: 2
  total_steps: 32
```

训练和验证配置中的 `retokenization_error_thresholds` 必须保持一致。

### 5. Check the dataset

```bash
find /path/to/waymo/training -maxdepth 1 -type f | head
find /path/to/waymo/validation -maxdepth 1 -type f | head

find /path/to/waymo/training -maxdepth 1 -type f | wc -l
find /path/to/waymo/validation -maxdepth 1 -type f | wc -l
```

### 6. Run focused tests

```bash
python -m unittest \
  tests.test_smart_causal_diffusion \
  tests.test_trajectory_energy -v
```

### 7. Calibrate retokenization thresholds

当前配置已经写入这次 10,000 scene 标定的 vehicle、pedestrian 和 cyclist
P99。数据集或 perturbation 策略变化时，使用下面的命令重新估计：

```bash
mkdir -p outputs/calibration

python scripts/calibrate_causal_retokenization.py \
  --config configs/train/train_scalable_causal_diffusion.yaml \
  --split train \
  --max_samples 10000 \
  --quantile 0.99 \
  --with_perturbation \
  --output_json outputs/calibration/causal_retokenization_p99.json
```

检查结果：

```bash
cat outputs/calibration/causal_retokenization_p99.json
```

将 JSON 中的 `config_order` 按原顺序填写到训练和验证配置。当前结果为：

```yaml
retokenization_error_thresholds:
  [0.7379697561264038, 0.8562850952148438, 1.2705252170562744]
```

如果某一类的 `counts` 为 `0` 或阈值为 `null`，不要启动正式训练。应先扩大
`--max_samples` 或检查数据中的 agent type/category 分布。

### 8. Start training from scratch

以下示例使用 4 张 GPU：

```bash
mkdir -p checkpoints/causal_diffusion logs

CUDA_VISIBLE_DEVICES=0,1,2,3 \
nohup python -u train.py \
  --config configs/train/train_scalable_causal_diffusion.yaml \
  --save_ckpt_path checkpoints/causal_diffusion \
  > logs/causal_diffusion.log 2>&1 &

echo $!
```

不要给该命令添加旧 AR checkpoint 的 `--pretrain_ckpt`。第一次正式实验应当
从头训练，以免旧模型的非因果状态和错误训练预算影响结果。

### 9. Monitor training

```bash
tail -f logs/causal_diffusion.log
```

```bash
watch -n 2 nvidia-smi
```

```bash
tensorboard --logdir lightning_logs --port 6006
```

重点检查：

- 学习率是否按 epoch warmup 和衰减
- `train_loss` / diffusion loss 是否为有限值
- `train_frontier_chunk_0_frac` 到 `train_frontier_chunk_3_frac` 是否都被采样
- `train_state_mode` 和 `train_rollout_depth` 是否按 curriculum 变化
- `train_retokenization_invalid_rate`
- `val_rollout_score`
- `val_minADE` 和 `val_minFDE`
- 2/4/6/8-second ADE/FDE 和 late ADE
- lane、dynamics 和 collision energy
- prediction coverage
- validation 和 step visualization 中是否仍有后段出地图现象

checkpoint 选择优先使用最低的 `val_rollout_score`，不要只根据训练 loss
或短窗口 diffusion loss 判断模型质量。

### 10. Resume an interrupted run

先找到需要恢复的 checkpoint：

```bash
find checkpoints/causal_diffusion -maxdepth 1 -name '*.ckpt' -print
```

然后使用 `--ckpt_path` 恢复 optimizer、scheduler、epoch 和 global step：

```bash
python -u train.py \
  --config configs/train/train_scalable_causal_diffusion.yaml \
  --save_ckpt_path checkpoints/causal_diffusion \
  --ckpt_path checkpoints/causal_diffusion/epoch=XX.ckpt
```

恢复训练不要使用 `--pretrain_ckpt`，因为它只加载模型参数，不恢复 optimizer
和训练进度。

### 11. Validate a checkpoint

```bash
python val.py \
  --config configs/validation/validation_scalable_causal_diffusion.yaml \
  --pretrain_ckpt checkpoints/causal_diffusion/epoch=XX.ckpt
```

训练配置默认只验证有限批次以控制开销。最终比较 checkpoint 时，应使用独立
validation 配置运行完整验证，并比较 late-horizon、地图约束和碰撞相关指标。

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
