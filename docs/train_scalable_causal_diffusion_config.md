# `train_scalable_causal_diffusion.yaml` 参数详解

本文只解释配置文件：

```text
configs/train/train_scalable_causal_diffusion.yaml
```

对应模型为 `smart_causal_diffusion`，继承关系如下：

```text
SMARTCausalDiffusion
  -> SMARTAutoregressiveDiffusion
    -> SMARTDiffusion
      -> SMART
```

因此一个参数可能在 causal 模型、AR 父类、diffusion 父类或原始 SMART
encoder 中生效。

## 1. 参数状态标记

本文使用以下标记：

| 标记 | 含义 |
| --- | --- |
| **生效** | 当前 causal 训练或推理直接使用 |
| **继承生效** | 由父类或原始 SMART encoder 使用 |
| **固定约束** | causal 模型要求固定值，修改后会报错或被覆盖 |
| **条件生效** | 只有特定开关、模式或入口下才生效 |
| **当前无效** | YAML 中存在，但当前训练路径没有读取或实际使用 |

## 2. 当前配置的关键结论

当前配置表示：

- 使用 11 帧历史和 80 帧未来，Waymo 时间间隔为 `0.1 s`。
- 一个 SMART trajectory token 表示 5 帧，即 `0.5 s`。
- 每次 causal diffusion 预测 4 个 token，即规划未来 `2.0 s`。
- 每轮只提交 1 个 token，即每 `0.5 s` 重新规划一次。
- 完成 80 帧 rollout 需要：

```text
80 / (1 commit token * 5 frames) = 16 rounds
```

- 每轮 diffusion 有 4 个循环 step，每个 step 恰好释放一个 causal frontier
  token。
- DDP 使用 14 张 GPU，每张 GPU batch size 为 4，梯度累积为 1，因此每个
  optimizer update 的名义全局 batch size 为：

```text
4 * 14 * 1 node * 1 accumulation = 56 scenes
```

- 当前 `LambdaLR` 以裸 scheduler 返回给 PyTorch Lightning 2.0.3，默认
  `interval="epoch"`。因此 `warmup_steps` 和 `total_steps` 实际按 epoch
  解释，而不是按 optimizer step。

> **当前配置：** `warmup_steps: 2`、`total_steps: 32`，与 32 epoch
> 训练和 epoch 级 scheduler 对齐。

## 3. YAML anchor: `time_info`

```yaml
time_info: &time_info
  num_historical_steps: 11
  num_future_steps: 80
  use_intention: True
  token_size: 2048
```

`time_info` 是 YAML anchor，只用于减少重复。顶层 `time_info` 本身不会传给
训练器；`<<: *time_info` 会把这 4 个字段复制到 `Dataset`、`Model` 和
`Model.decoder`。

### `time_info.num_historical_steps: 11`

- **状态：生效，但不同副本的使用情况不同。**
- 表示历史序列共有 11 帧。
- Waymo 为 `10 Hz` 时，对应从 `t=-1.0 s` 到 `t=0 s`。
- `Model.num_historical_steps` 是模型实际读取的历史长度。
- `Dataset.num_historical_steps` 用于构造 `WaymoTargetBuilder`。
- `Model.decoder.num_historical_steps` 当前不会被模型读取。
- 修改时必须保证历史长度与 SMART token shift 匹配：

```text
history_tokens = (num_historical_steps - 1) / token_steps
```

当前为 `(11 - 1) / 5 = 2`。

### `time_info.num_future_steps: 80`

- **状态：生效，但模型实际读取 decoder 副本。**
- 表示未来 80 帧，即 `8.0 s`。
- `Dataset.num_future_steps` 控制数据 transform 截取长度。
- `Model.decoder.num_future_steps` 决定模型的 `self.num_future_steps`。
- `Model.num_future_steps` 当前不直接参与模型初始化。
- 必须满足：

```text
num_future_steps % future_chunk_steps == 0
```

### `time_info.use_intention: True`

- **状态：当前无效。**
- `Dataset.use_intention` 只被保存，没有传给 transform 或 dataset。
- `Model.use_intention` 没有被 `SMART` 读取。
- `Model.decoder.use_intention` 没有从 `SMART` 传入 `SMARTDecoder`，且
  `SMARTDecoder` 内部也未使用该参数。
- 修改该值不会改变当前 causal diffusion 行为。

### `time_info.token_size: 2048`

- **状态：部分生效。**
- `Dataset.token_size` 决定 `TokenProcessor` 加载
  `cluster_frame_5_2048.pkl`。
- `Model.decoder.token_size` 决定 agent token vocabulary、diffusion
  decoder 输出类别数以及 mask token id。
- `Model.token_size` 当前不直接使用。
- Dataset 与 decoder 的 token size 必须一致，否则 token id 和输出维度不匹配。

## 4. `Dataset`

### `Dataset.root: null`

- **状态：生效。**
- `null` 表示直接从 `train_raw_dir` / `val_raw_dir` 中枚举 pickle 文件。
- 如果设置为目录，`MultiDataset` 会尝试读取：

```text
<root>/split_datainfo.pkl
```

- 当前 causal 训练推荐保持 `null`，使用 raw pickle 目录模式。

### `Dataset.train_batch_size: 4`

- **状态：生效。**
- 每个 DDP rank、每张 GPU 的训练 batch size。
- 不是全局 batch size。
- 增大可提高吞吐和 timestep 覆盖，但显著增加 agent/map token 显存。
- OOM 时优先降到 `2` 或 `1`，再通过
  `Trainer.accumulate_grad_batches` 补偿全局 batch。

### `Dataset.val_batch_size: 1`

- **状态：生效。**
- 每个 rank 的 validation batch size。
- 完整 80 帧、16 轮 causal rollout 开销较大，建议保持 `1`。

### `Dataset.test_batch_size: 1`

- **状态：条件生效。**
- 只用于 `test_dataloader()`。
- `train.py` 的 `trainer.fit()` 不运行测试，因此正式训练过程中通常不使用。

### `Dataset.shuffle: true`

- **状态：生效。**
- 控制训练 DataLoader 是否打乱 scene。
- validation/test DataLoader 始终不 shuffle。
- 正式训练建议保持 `true`。

### `Dataset.num_workers: 4`

- **状态：生效。**
- 每个 DDP rank 的 DataLoader worker 数。
- 14 张 GPU 时总 worker 数可达到约 `14 * 4 = 56`。
- 如果共享文件系统压力大、打开文件过多或 CPU 抢占严重，应降低。

### `Dataset.pin_memory: true`

- **状态：生效。**
- DataLoader 使用 page-locked host memory，通常能加快 CPU 到 GPU 传输。
- 会增加主机内存占用。

### `Dataset.persistent_workers: true`

- **状态：生效。**
- worker 在 epoch 之间保持存活，减少重复启动开销。
- 只有 `num_workers > 0` 时才真正启用。

### `Dataset.train_raw_dir`

当前值：

```yaml
["/raid/haoq_lab/wangshijie/data/waymo/training"]
```

- **状态：生效。**
- 可以填写多个目录，dataset 会合并目录中的所有文件。
- 目录内容必须是当前 SMART 预处理后可直接 `pickle.load()` 的 scene 文件，
  不是原始 WOMD TFRecord。
- 服务器迁移时最需要确认的字段之一。

### `Dataset.val_raw_dir`

当前值：

```yaml
["/raid/haoq_lab/wangshijie/data/waymo/validation"]
```

- **状态：生效。**
- validation scene pickle 目录列表。
- validation visualization 的 `sample_indices` 也基于这里枚举后的索引。

### `Dataset.test_raw_dir: null`

- **状态：条件生效。**
- 只供 test dataset 使用。
- 当前训练不执行 test，因此可以为空。

### `Dataset.transform: WaymoTargetBuilder`

- **状态：生效。**
- 在 `MultiDataModule.transforms` 中查找对应 transform 类。
- 负责 category 标注、历史/未来截取、坐标处理等。
- 名称拼错会在 DataModule 初始化时触发 `KeyError`。

### `Dataset.train_processed_dir: null`

### `Dataset.val_processed_dir: null`

### `Dataset.test_processed_dir: null`

- **状态：当前 raw 目录模式下不使用。**
- 这些字段来自旧的 processed dataset 接口。
- 当前 `MultiDataset.get()` 实际从 `raw_paths` 读取 pickle。
- 不建议仅设置 `processed_dir`；若启用 `root/split_datainfo.pkl` 模式，需要
  确保 metadata 数量与 raw 文件顺序完全一致。

### `Dataset.dataset: scalable`

- **状态：生效。**
- 在 `MultiDataModule.dataset` 注册表中选择 `MultiDataset`。
- 当前注册表只支持 `"scalable"`。

### `Dataset.num_historical_steps: 11`

- **状态：生效。**
- 传给 train/val/test 的 `WaymoTargetBuilder`。
- 应与 `Model.num_historical_steps` 相同。

### `Dataset.num_future_steps: 80`

- **状态：生效。**
- 传给 `WaymoTargetBuilder`。
- 应与 `Model.decoder.num_future_steps` 和
  `Model.diffusion.total_rollout_steps` 一致。

### `Dataset.use_intention: true`

- **状态：当前无效。**
- DataModule 只保存该值，没有继续传递。

### `Dataset.token_size: 2048`

- **状态：生效。**
- 决定 `TokenProcessor` 使用哪一份 trajectory token 文件。
- 当前对应 `smart/tokens/cluster_frame_5_2048.pkl`。

## 5. `Trainer`

### `Trainer.strategy: ddp_find_unused_parameters_true`

- **状态：生效。**
- `train.py` 将它转换为：

```python
DDPStrategy(find_unused_parameters=True, gradient_as_bucket_view=True)
```

- causal 模型存在条件分支和可选 loss，部分参数在某些 batch 中可能没有梯度，
  因而当前使用 `true` 更稳妥。
- `find_unused_parameters=True` 会增加 DDP graph traversal 开销。

### `Trainer.accelerator: gpu`

- **状态：生效。**
- 传给 `pl.Trainer(accelerator=...)`。
- 无 CUDA 环境可设为 `cpu`，但完整训练和 rollout 会非常慢。

### `Trainer.devices: 14`

- **状态：生效。**
- 表示使用 14 个 GPU process。
- 若通过 `CUDA_VISIBLE_DEVICES` 限制 GPU，暴露数量必须与这里一致。
- 单机 4 卡示例应改为 `4`。

### `Trainer.max_epochs: 32`

- **状态：生效。**
- 训练最多运行 32 个 epoch。
- causal closed-loop curriculum 也是按 `current_epoch` 设计的：

| epoch，代码从 0 开始 | clean/rollout 分布 |
| --- | --- |
| 0-3 | 50% rollout，50% clean |
| 4+ | 50% rollout，50% clean |

- model-rollout 从第 0 个 epoch 开始参与训练；连续高斯 state perturb 已从代码和配置中移除。

### `Trainer.save_ckpt_path: null`

- **状态：当前无效。**
- `train.py` 不读取该 YAML 字段。
- checkpoint 目录由命令行决定：

```bash
--save_ckpt_path checkpoints/causal_diffusion
```

### `Trainer.num_nodes: 1`

- **状态：生效。**
- DDP 节点数。
- 多机训练需要同时配置集群启动方式、通信环境和每节点设备数。

### `Trainer.mode: null`

- **状态：当前无效。**
- `train.py` 和模型没有读取该字段。

### `Trainer.ckpt_path: null`

- **状态：当前无效。**
- 恢复训练由命令行 `--ckpt_path` 控制。
- YAML 中填写路径不会自动恢复 optimizer/scheduler。

### `Trainer.precision: 32`

- **状态：生效。**
- 使用 FP32 训练。
- `16-mixed` 或 `bf16-mixed` 可节省显存和提高速度，但需要重新做数值稳定性
  smoke test，尤其是 diffusion 权重和 energy 计算。

### `Trainer.accumulate_grad_batches: 1`

- **状态：生效。**
- 每多少个 batch 做一次 optimizer update。
- 有效全局 batch size：

```text
train_batch_size * devices * num_nodes * accumulate_grad_batches
```

- 显存不足时可减小 batch size 并增大该值。

### `Trainer.limit_val_batches: 50`

- **状态：生效。**
- 每个 validation epoch 只运行有限数量的 validation batch。
- 当前每个 validation batch 都会进行完整 causal rollout，因此该值直接影响
  epoch-end validation 时间。
- 这是训练期间的快速评估，不等价于完整验证集官方结果。

### `Trainer.check_val_every_n_epoch: 1`

- **状态：生效。**
- 每个 epoch 结束后运行一次 validation。
- 增大可减少训练耗时，但 checkpoint 指标和可视化更新会变稀疏。

### `Trainer.monitor_metric: val_rollout_score`

- **状态：生效。**
- `ModelCheckpoint` 根据该 metric 保存最佳 5 个 checkpoint。
- `val_rollout_score` 综合 8 秒 ADE、后 4 秒 ADE 和安全 energy。
- 如果 validation 没有执行 rollout 或 metric 没有被记录，checkpoint callback
  会无法正常监控。

### `Trainer.monitor_mode: min`

- **状态：生效。**
- `val_rollout_score` 越低越好，因此使用 `"min"`。
- 改成 `"max"` 会保存最差模型。

## 6. `Visualization`

### `Visualization.enabled: true`

- **状态：生效。**
- 启用 epoch-end validation visualization。
- 同时也是创建 `step_viz` callback 的总开关；父级为 `false` 时，
  `step_viz.enabled: true` 也不会生效。

### `Visualization.interval_epochs: 1`

- **状态：生效。**
- 每 1 个 epoch 生成一次固定 validation scene 的图。
- callback 只在 global rank 0 执行。

### `Visualization.sample_indices: [0, 1, 2, 3]`

- **状态：生效。**
- `val_dataset` 中用于 epoch visualization 的固定样本索引。
- 越界索引会被静默跳过。

### `Visualization.output_dir: ./outputs/val_causal_diffusion`

- **状态：生效。**
- 实际输出结构为：

```text
outputs/val_causal_diffusion/
  smart_causal_diffusion/
    epoch_001/
```

### `Visualization.max_agents: 0`

- **状态：生效。**
- `0` 表示不限制绘图 agent 数。
- 正整数表示只画距离 ego 最近的一部分 agent，并保留 ego。
- 只影响绘图，不影响模型输入和指标。

### `Visualization.step_viz.enabled: true`

- **状态：生效。**
- 启用训练过程中按 global optimizer step 生成可视化。
- 需要父级 `Visualization.enabled: true`。

### `Visualization.step_viz.interval_steps: 2000`

- **状态：生效。**
- 当 `trainer.global_step` 是 2000 的倍数时生成图。
- 这里的 step 是 optimizer step，不是 epoch。
- 完整 causal inference 较慢，设置过小会明显降低训练吞吐。

### `Visualization.step_viz.sample_indices: [110, 111, 112, 113]`

- **状态：生效。**
- step visualization 使用的 validation dataset 索引。
- 必须确认 validation 集至少有 114 个样本。

### `Visualization.step_viz.output_dir: ./outputs/step_causal_diffusion`

- **状态：生效。**
- 实际输出按 predictor 和 global step 分目录。

## 7. `Model` 通用参数

### `Model.mode: train`

- **状态：当前无效。**
- `train.py` 已经决定执行训练，模型代码没有读取 `model_config.mode`。

### `Model.predictor: smart_causal_diffusion`

- **状态：生效。**
- 在 `train.py` 的 predictor 注册表中选择 `SMARTCausalDiffusion`。
- 改成其他名称会训练不同模型或触发 `KeyError`。

### `Model.dataset: waymo`

- **状态：继承生效。**
- 传给原始 SMART map/agent encoder。
- 当前数据特征、类别和 metric 均按 Waymo 约定实现，不应随意修改。

### `Model.input_dim: 2`

- **状态：继承生效。**
- 原始 SMART 图构建和 Fourier edge embedding 使用平面 `x/y`。
- 当前 encoder 支持 `2` 或 `3`；改成 `3` 需要确保数据有一致的 z 维语义。

### `Model.hidden_dim: 128`

- **状态：生效。**
- 原始 SMART encoder 和 causal diffusion decoder 的主特征维度。
- 增大通常提高容量，但 map/agent/token attention 的显存和计算都会增加。
- 必须与 attention 投影等模块保持一致。

### `Model.output_dim: 2`

- **状态：当前 causal 路径无效。**
- 仅在 `SMART.__init__` 中保存，没有用于 causal decoder 输出。
- causal decoder 输出是 `token_size=2048` 类 logits。

### `Model.output_head: false`

- **状态：当前 causal 路径无效。**
- 仅保存为成员变量，当前 SMART/causal 模块没有根据它切换 head。

### `Model.num_heads: 8`

- **状态：生效。**
- 原始 SMART encoder 和 causal diffusion decoder 的 attention head 数。
- 与 `head_dim: 16` 配合，attention 内部总 head 宽度为 `8 * 16 = 128`，
  正好等于 `hidden_dim`。

### `Model.head_dim: 16`

- **状态：生效。**
- 每个 attention head 的维度。
- 修改时应同时评估 `num_heads * head_dim` 与 `hidden_dim` 的关系。

### `Model.dropout: 0.1`

- **状态：生效。**
- 用于原始 SMART 和 diffusion attention layer。
- 增大可增强正则，但可能降低短期 token 精度。

### `Model.num_freq_bands: 64`

- **状态：生效。**
- FourierEmbedding 的频带数，用于相对距离、方向、时间等连续特征。
- 增大可表示更丰富的尺度，但增加 embedding 参数和计算。

### `Model.lr: 0.0005`

- **状态：生效。**
- causal diffusion decoder 和其他非 encoder 参数的基础 AdamW LR。
- SMART encoder 使用：

```text
lr * encoder_lr_scale
= 0.0005 * 0.5
= 0.00025
```

- 两个 param group 共用同一 LambdaLR 倍率。

### `Model.warmup_steps: 2`

- **状态：生效，但当前实际单位是 epoch。**
- scheduler 的变量名叫 `current_step`，但 Lightning 对裸 scheduler 默认
  按 epoch 调用。
- 当前第一阶段倍率约为：

```text
lambda(epoch) = (epoch + 1) / warmup_steps
```

- 前两个 epoch 完成 warmup；decoder 和 encoder param group 使用相同倍率。

### `Model.total_steps: 32`

- **状态：生效，但当前实际单位是 epoch。**
- warmup 后使用 cosine decay，达到 `total_steps` 后 LR 变为 0：

```text
0.5 * (1 + cos(pi * (epoch - warmup) / (total - warmup)))
```

- 当前与 `Trainer.max_epochs: 32` 对齐。
- 必须满足 `total_steps > warmup_steps`，否则 cosine 区间退化。

### `Model.inference_token: true`

- **状态：生效。**
- validation 时执行完整 80 帧 rollout，并记录 ADE/FDE、energy 和
  `val_rollout_score`。
- 设为 `false` 会跳过 rollout；此时当前 checkpoint monitor
  `val_rollout_score` 也不会产生。

### `Model.rollout_num: 1`

- **状态：当前 causal 路径无效。**
- 原始 `SMART` 只保存该字段，当前 causal inference 不读取它。
- 不会生成多条 stochastic rollout。

### `Model.num_historical_steps: 11`

- **状态：生效。**
- 原始 SMART encoder 和 causal rollout 使用的历史帧数。
- 必须与 Dataset transform 一致。

### `Model.num_future_steps: 80`

- **状态：当前模型不直接读取。**
- 模型实际从 `Model.decoder.num_future_steps` 获取未来长度。
- 为避免配置语义混乱，应保持两个值相同。

### `Model.use_intention: true`

- **状态：当前无效。**

### `Model.token_size: 2048`

- **状态：当前模型不直接读取。**
- 实际 vocabulary 大小来自 `Model.decoder.token_size`。

## 8. `Model.decoder`

这部分同时配置原始 SMART history/map encoder。causal diffusion 不会绕过
SMART encoder，而是先调用 `encode_history_context()` 获取历史和地图特征。

### `Model.decoder.num_historical_steps: 11`

- **状态：当前无效。**
- `SMART` 构造 encoder 时使用的是 `Model.num_historical_steps`。
- 建议保持一致，仅作为配置可读性副本。

### `Model.decoder.num_future_steps: 80`

- **状态：生效。**
- `SMART.__init__` 用它设置 `self.num_future_steps`。
- 决定 future chunk 总数和 validation GT 截取范围。

### `Model.decoder.use_intention: true`

- **状态：当前无效。**
- 没有被传入 encoder。

### `Model.decoder.token_size: 2048`

- **状态：生效。**
- 原始 agent token classification head 和 diffusion output projection 的类别数。
- diffusion 的 mask token id 为 `2048`，正常 token id 为 `0-2047`。

### `Model.decoder.num_map_layers: 3`

- **状态：继承生效。**
- 原始 SMART map encoder 的 map-token self-attention 层数。
- 增大提高地图表示容量，也增加 map encoding 成本。
- 不等于 diffusion decoder 的 `num_layers`。

### `Model.decoder.num_agent_layers: 6`

- **状态：继承生效。**
- 原始 SMART history encoder 的 temporal、map-to-agent 和
  agent-to-agent attention 层数。
- 主要影响 history context 编码成本。

### `Model.decoder.a2a_radius: 60`

- **状态：生效，单位为米。**
- 原始 SMART history agent-to-agent 图和 diffusion future-token
  同时刻空间交互图都使用该半径。
- 增大可覆盖更多交互 agent，但边数量和显存可能快速增加。

### `Model.decoder.pl2pl_radius: 10`

- **状态：继承生效，单位为米。**
- 原始 SMART map-token 到 map-token 图的半径。
- 只影响 map encoder，不直接控制 future token 查地图的范围。

### `Model.decoder.pl2a_radius: 30`

- **状态：生效，单位为米。**
- 原始 map-to-agent context 和 diffusion map-to-future-token 动态边半径。
- 对后段出地图问题非常关键：过小可能让偏移后的 token 查不到道路上下文；
  过大则引入过多无关道路和计算。

### `Model.decoder.time_span: 30`

- **状态：生效，单位为 frame。**
- 原始 SMART history temporal edge 使用该时间范围。
- causal decoder 中转换为最大 token chunk 差：

```text
max_chunk_delta = floor(time_span / future_chunk_steps)
= floor(30 / 5)
= 6
```

- 当前预测窗口只有 4 个 chunk，因此所有更早 chunk 都可作为严格因果 source。

## 9. `Model.diffusion`: 时间窗口与闭环 rollout

### `Model.diffusion.future_chunk_steps: 5`

- **状态：固定约束。**
- 每个 trajectory token 代表 5 个 future frame，即 `0.5 s`。
- 必须同时满足：

```text
num_future_steps % future_chunk_steps == 0
future_chunk_steps == SMARTAgentDecoder.shift
```

- 当前 SMART codebook 本身就是 frame-5 token，不能只改配置来改变它。

### `Model.diffusion.history_tokens: 2`

- **状态：固定约束。**
- 每个短窗口保留 2 个历史 token，即 1 秒 token history。
- 必须等于：

```text
(num_historical_steps - 1) / token_steps = 2
```

### `Model.diffusion.prediction_tokens: 4`

- **状态：causal 模型硬性要求为 4。**
- 每轮预测 4 个 token，即 20 帧、2 秒 planning window。
- 修改成其他值会在 `SMARTCausalDiffusion.__init__()` 报错。

### `Model.diffusion.commit_tokens: 1`

- **状态：causal 模型硬性要求为 1。**
- 每轮只执行第一个 0.5 秒 token，再根据新状态重新编码地图和 agent context。
- 修改成其他值会报错。

### `Model.diffusion.carry_tail_proposal: true`

- **状态：生效。**
- 每轮只提交 chunk 0，未提交的 3 个 token 左移到下一轮，作为可修订 proposal：

```text
previous chunks [1, 2, 3] -> next proposal chunks [0, 1, 2]
```

- proposal 不会直接写入执行历史；下一轮仍重新采样所有四个 token。

### `Model.diffusion.proposal_conditioning_enabled: true`

- **状态：生效。**
- carried proposal 同时提供置信度加权的物理 token embedding 和临时几何。
- 一旦该位置被本轮正式采样，proposal 条件自动失效，因此 tail 始终可修订。

### `Model.diffusion.causal_objective: discrete_frontier_v2`

- **状态：固定约束。**
- 训练均匀选择一个有效 frontier，mask 该 frontier 及其后缀，但离散 CE
  只监督被选择的 frontier。
- 该目标与推理时“一次释放一个 chunk”的决策严格对应，并移除了旧连续时间
  `1/t` 权重和无 mask 时强制插入 token 的逻辑。

### `Model.diffusion.token_steps: 5`

- **状态：固定约束。**
- AR rollout 解码一个 token 时产生的 frame 数。
- 必须等于 `future_chunk_steps`，否则模型初始化报错。

### `Model.diffusion.total_rollout_steps: 80`

- **状态：生效。**
- 完整闭环 rollout 的 frame 数。
- 必须满足：

```text
total_rollout_steps % (commit_tokens * token_steps) == 0
```

- 当前得到 16 个 rollout round。
- 通常应与 `num_future_steps` 保持一致。

### `Model.diffusion.local_map_refresh: rescreen`

- **状态：生效。**
- `rescreen` 模式每轮保留该 scene 的全部可见 map token feature，然后按照
  当前 future token 位置，用 `pl2a_radius` 重建 map-to-token 边。
- 这是闭环后段仍能查询新道路区域的关键机制。
- 非 `rescreen` 值会回退到父类的 scene-level map candidate 选择逻辑。

### `Model.diffusion.rolling_anchor_training: true`

- **状态：生效。**
- 训练不总是固定在当前帧，而是构造短 AR window，并启用 clean / model-rollout
  state curriculum。
- 设为 `false` 会跳过 `_build_causal_training_view()`，闭环分布训练失效。

### Gaussian state perturbation

- **状态：已移除。**
- 旧的 `state_perturb_pos_sigma_m` / `state_perturb_heading_sigma_rad` 配置不再存在，
  代码也不再对历史 position 或 heading 加连续高斯噪声。
- 当前鲁棒性主要来自 model-rollout state curriculum 和 SMART-style top-k
  history-token noise；future target 会从 noised history anchor 做 deterministic
  retokenization。

### `Model.diffusion.closed_loop_max_depth: 4`

- **状态：生效。**
- model-rollout curriculum 最多先提交多少个模型 token，再从漂移状态训练。
- 代码强制裁剪到 `[1, 4]`。
- depth 4 对应先 rollout 2 秒，再 retokenize 后续 GT continuation。

### `Model.diffusion.closed_loop_batch_ratio_max: 0.5`

- **状态：生效。**
- 从 epoch 0 开始最多 50% batch 使用模型 rollout state；其余 batch 保持 clean。
- 在当前 1-4 token rollout depth 下，目标是把总体训练开销增量控制在约 30%。

### `Model.diffusion.current_state_enabled: true`

- **状态：生效。**
- 从最近两帧构造纵向/横向速度、速度大小、yaw rate 和 current-valid 特征，
  经 MLP 加入每个 future token 的 agent context。

### `Model.diffusion.current_state_edges: true`

- **状态：生效。**
- 即使四个 future token 全部为 mask，chunk 0 的当前锚点仍可作为 causal
  temporal 和 same-chunk agent-agent source。
- 这样第一个可执行 token 能在采样前感知当前多车交互，而不是只有 scene mean。

### `Model.diffusion.history_recency_decay: 0.5`

- **状态：生效，范围 `(0, 1]`。**
- 两个历史 token 按 `[0.5, 1.0]` 加权，最近 token 权重更高。
- 设为 `1.0` 等价于旧的 masked mean pooling。

## 10. `Model.diffusion`: retokenization 与训练 loss

### `Model.diffusion.retokenization_error_thresholds`

- **状态：生效，顺序固定。**
- 顺序为：

```text
[vehicle, pedestrian, cyclist]
```

- 单位为米。
- 每个 future chunk 在预测/扰动后的局部坐标系中重新匹配 SMART codebook，
  error 是有效 frame 上轨迹点欧氏距离的平均值。
- 低于对应阈值：使用离散 token CE。
- 高于阈值：认为没有合理离散 token，改用 continuous recovery loss。
- 当前使用 10,000 scene calibration 得到的 P99：

```text
[0.7379697561264038, 0.8562850952148438, 1.2705252170562744]
```

### `Model.diffusion.continuous_recovery_loss_weight: 1.0`

- **状态：生效，非负。**
- 对 retokenization-invalid target，计算 token softmax 下的期望 endpoint，
  与真实局部 endpoint 做 Smooth L1 loss。
- 总 loss：

```text
causal diffusion loss
+ continuous_recovery_loss_weight * recovery_loss
```

- 设为 `0` 会让 invalid target 不产生有效恢复监督。

### `Model.diffusion.frontier_loss_weight: 1.0`

- **状态：生效，非负。**
- 直接缩放被选中 frontier 的平均 CE；后缀 token 不计算离散 CE。
- 当前 `1.0` 不改变 CE 尺度。

### `Model.diffusion.encoder_lr_scale: 0.5`

- **状态：生效，非负。**
- 原始 SMART encoder 使用基础 LR 的 50%。
- causal decoder、energy 相关可训练参数等使用完整 `Model.lr`。
- 从头训练时过小可能导致 history/map encoder 学习慢；过大可能破坏已有 SMART
  token representation。

### `Model.diffusion.ntp_aux_loss_weight: 0.0`

- **状态：生效。**
- 原始 SMART next-token-prediction auxiliary loss 权重。
- `<= 0` 时不仅权重为 0，而且会直接跳过 NTP forward，节省计算。
- 当前模型完全由 causal diffusion/recovery loss 训练。

## 11. `Model.diffusion`: diffusion decoder 与噪声过程

### `Model.diffusion.num_steps: 4`

- **状态：生效。**
- causal sampling 外层循环次数。
- 必须满足 `num_steps == prediction_tokens`。当前四个预测 chunk 对应四个
  sampling step。
- 每个 step 依次释放一个 chunk：

```text
step 0, t=1.00: release chunk 0
step 1, t=0.75: release chunk 1
step 2, t=0.50: release chunk 2
step 3, t=0.25: release chunk 3
```

- 不允许添加不执行 decoder 的空转 step。模型初始化时会拒绝
  `num_steps != prediction_tokens` 的配置。
- 这样全 mask 初态从高噪声 `t=1` 开始，后续随着已知前缀增加逐步降低
  `t`，同时让 `(1-t)^2` safety guidance 从弱到强平滑增加。

### `Model.diffusion.num_layers: 6`

- **状态：生效。**
- causal diffusion decoder 的层数。
- 每层依次执行：

```text
strictly causal temporal attention
map-to-future attention
same-chunk agent-to-agent attention
```

- 与 `decoder.num_agent_layers` 是两套不同网络。

### `Model.diffusion.eps: 1.0e-3`

- **状态：v2 frontier loss 中不使用。**
- 仍用于构造兼容的 `LogLinearNoise` 模块，但离散 frontier v2 不再从连续噪声
  分布采样训练 mask。

### `Model.diffusion.min_t: 1.0e-3`

- **状态：兼容性下限。**
- v2 训练和采样固定使用 `t=[1.0, 0.75, 0.5, 0.25]`，当前值不会改变它们。

### `Model.diffusion.remask_confidence_temperature: 1.0`

- **状态：生效，但名称来自旧 remask 实现。**
- causal 模型不会 remask 已释放 token，但仍使用：

```text
softmax(logits / temperature)
```

- `< 1` 使分布更尖锐，`> 1` 增加随机性。
- 代码最小裁剪为 `1e-6`。

## 12. `Model.diffusion`: 条件信息与图构建

### `Model.diffusion.use_agent_context: true`

- **状态：生效。**
- 将每个 agent 的历史 token feature 做 recency-weighted pooling，并叠加
  显式 current-motion embedding，再投影到每个 future token。
- 关闭后 causal decoder 仍有 scene summary、type、shape、map 和图边，但失去
  agent-specific history summary。

### `Model.diffusion.use_type_embedding: true`

- **状态：生效。**
- 加入 vehicle/pedestrian/cyclist/background type embedding。
- 同时决定 causal decoder 是否创建 4 类 type embedding table。

### `Model.diffusion.use_map_context: true`

- **状态：生效。**
- 将 SMART map encoder 的 `x_pt` 输入 diffusion decoder，并建立动态
  map-to-future-token radius edges。
- 关闭后 lane energy 如果仍能拿到 map positions 也会受影响，因为当前
  `_pack_map_context()` 会整体返回空 map context/geometry。

### `Model.diffusion.max_map_tokens: 0`

- **状态：当前 `local_map_refresh: rescreen` 下基本无效。**
- 父类路径中，正数表示每个 scene 最多保留多少个最近 map token，`<=0`
  表示不截断。
- 但 AR/causal 的 `rescreen` 分支明确保留 scene 中全部可见 map token，
  不读取 `max_map_tokens`。
- 当前真正控制连接数量的是 `pl2a_radius`。

### `Model.diffusion.geometry_dropout_prob: 0.0`

- **状态：当前 causal loss 中无效。**
- 父类 `SMARTDiffusion._compute_diffusion_loss()` 会调用 geometry dropout，
  但 causal 模型覆盖了该 loss，没有调用 `_maybe_apply_geometry_dropout()`。
- 当前设为 0 与实际行为一致；改成非 0 也不会影响 causal loss。

### `Model.diffusion.visible_token_corruption_prob: 0.0`

- **状态：固定关闭。**
- 父类可用近邻 token 污染 visible context。
- causal 初始化会无条件把该概率改为 `0.0`，以保持单一 absorbing state。
- YAML 改成其他值也不会生效。

### `Model.diffusion.geometry_confidence_source_threshold: 0.2`

- **状态：继承生效。**
- geometry confidence 高于阈值的 future token 才能作为 temporal/spatial
  graph source。
- proposal/已释放 token 按 confidence 成为 source；此外 v2 的 chunk-0 当前锚点
  在 `current_state_edges: true` 时始终可作为 source。
- 阈值过高会削弱已预测 token 的上下文传播；过低会允许低置信 geometry
  影响后续 token。

## 13. `Model.diffusion`: agent 选择、监督与指标

### `Model.diffusion.target_category_only: true`

- **状态：被显式 `supervision_mode` 部分遮蔽。**
- 它主要用于决定未指定 `supervision_mode` 时的默认值，以及旧的
  `legacy_target_category_only` 模式。
- 当前已经显式设置 `supervision_mode: smart_category3`，因此修改该布尔值
  不会改变监督 agent。

### `Model.diffusion.agent_selection_mode: smart_inference`

- **状态：生效。**
- 决定哪些 agent 参与生成。
- `smart_inference` 表示当前历史帧有效的所有 agent。
- 支持的其他模式包括 `non_background` 和 `smart_category3`。
- 生成所有 history-valid agent 有助于交互一致性；loss 仍可只监督 category 3。

### `Model.diffusion.supervision_mode: smart_category3`

- **状态：生效。**
- 只有 `category == 3` 且属于 generation set 的 agent 接受 diffusion loss。
- 其他 agent 仍可生成并作为交互上下文。
- 可选 `smart_generation` 会监督全部 generation agent，改变实验定义。

### `Model.diffusion.metric_mode: smart_val_compatible`

- **状态：生效。**
- validation metrics 使用当前历史帧有效的 agent，与 SMART validation
  兼容。
- `smart_category3` 只评 target agents，会得到不同口径，不能直接与官方
  SMART 全量口径混用。

## 14. `Model.diffusion`: safety energy reranking

候选选择公式：

```text
total_energy =
    lane_distance_weight * lane_distance
  + lane_heading_weight  * lane_heading
  + dynamics_weight      * dynamics
  + collision_weight     * collision

guidance_scale =
  commit_safety_weight,                 chunk == 0
  safety_energy_weight * (1 - t)^2,     chunk > 0

score = log_probability - guidance_scale * total_energy
```

因此执行 token 在 `t=1` 也受 safety 影响，后续 proposal token 仍随噪声降低
逐步增强约束。

### `Model.diffusion.safety_energy_enabled: true`

- **状态：生效。**
- 对每个 frontier token 的 top-k 候选执行 energy reranking。
- 关闭后从 frontier softmax 分布直接 multinomial sample。

### `Model.diffusion.safety_topk: 16`

- **状态：生效，最小为 1。**
- 只对概率最高的 16 个 token 计算 trajectory energy。
- 增大可让安全项纠正更多低概率候选，但 `torch.cdist`、dynamics 和 collision
  计算成本上升。

### `Model.diffusion.safety_energy_weight: 1.0`

- **状态：生效，非负。**
- 控制整个 total energy 相对于 log probability 的全局强度。
- `0` 相当于在 top-k 内纯 argmax log probability。
- 过大可能使模型偏向保守 token并降低多样性。

### `Model.diffusion.commit_safety_weight: 1.0`

- **状态：生效，非负。**
- 只用于每轮真正执行的 chunk 0，不乘 `(1-t)^2`。
- 解决旧实现中首轮 `t=1` 导致执行 token safety 权重恰好为 0 的问题。

### `Model.diffusion.lane_distance_energy_weight: 1.0`

- **状态：生效。**
- 惩罚候选轨迹点到同 scene 最近 map token 的平均距离，单位基础为米。
- 对抑制驶离道路最直接。

### `Model.diffusion.lane_heading_energy_weight: 0.5`

- **状态：生效。**
- 惩罚候选 heading 与最近 map token orientation 的平均绝对角差，单位为弧度。
- 在路口或多方向重叠车道附近可能存在 nearest-map ambiguity。

### `Model.diffusion.dynamics_energy_weight: 0.25`

- **状态：生效。**
- 惩罚超过阈值的加速度和 yaw rate，采用平方 excess。
- chunk 0 额外包含观测当前位置/速度/heading 到候选首帧的跃迁，静止车辆突然
  选择高速直行 token 不再具有接近零的 dynamics energy。
- 当前权重低于 lane/collision，作为软约束而不是硬过滤。

### `Model.diffusion.collision_energy_weight: 2.0`

- **状态：生效。**
- 惩罚候选轨迹与同 scene 其他 frontier agent nominal trajectory 的近距离重叠。
- 当前是候选 energy 中最大的显式权重。

### `Model.diffusion.trajectory_dt: 0.1`

- **状态：生效，单位为秒。**
- 用于从位置差计算速度/加速度，从 heading 差计算 yaw rate。
- 必须与 Waymo 数据频率一致。

### `Model.diffusion.max_acceleration: 6.0`

- **状态：生效，单位为 `m/s^2`。**
- 只惩罚超过 6 的部分：

```text
relu(|acceleration| - 6)^2
```

- 不是硬上限。

### `Model.diffusion.max_yaw_rate: 1.2`

- **状态：生效，单位为 `rad/s`。**
- 只惩罚超过该阈值的 yaw rate。
- `1.2 rad/s` 约为 `68.8 deg/s`。

### `Model.diffusion.collision_distance: 2.0`

- **状态：生效，单位为米。**
- 两条轨迹距离小于 2 米时产生：

```text
relu(2.0 - distance)^2
```

- 当前是基于中心点距离，不是精确旋转 bounding box 碰撞。

## 15. `Model.diffusion`: validation 与训练控制

### `Model.diffusion.eval_inference_batches: 2`

- **状态：当前无效。**
- 值会被保存为 `self.diffusion_eval_batches`。
- 但当前 `_should_run_validation_inference()` 只检查
  `Model.inference_token`，没有使用 batch index 或该上限。
- 所以只要 `inference_token: true`，`limit_val_batches` 范围内的每个
  validation batch 都会运行完整 rollout。

### `Model.diffusion.debug_validation_logging: true`

- **状态：生效。**
- 输出 `[SMARTDiffusion]`、`[ValidationVisualization]` 和
  `[StepVisualization]` 的耗时/阶段日志。
- 有助于定位 16-round rollout、sampling 或 visualization 卡顿。
- 会增加 stdout 内容，但一般不影响数值结果。

### `Model.diffusion.freeze_encoder: false`

- **状态：生效。**
- `false` 表示原始 SMART map/history encoder 从头共同训练。
- `true` 会把 encoder 参数设为 `requires_grad=False`，只训练 causal decoder
  等其余参数。
- 从头训练不建议冻结；从可靠 SMART checkpoint 初始化时可作为 ablation。

## 16. 推荐的服务器修改项

正式训练前至少修改：

```yaml
Dataset:
  train_raw_dir: ["/actual/path/to/training"]
  val_raw_dir: ["/actual/path/to/validation"]

Trainer:
  devices: <实际GPU数>
  max_epochs: 32

Model:
  warmup_steps: 2
  total_steps: 32
  diffusion:
    retokenization_error_thresholds:
      [0.7379697561264038, 0.8562850952148438, 1.2705252170562744]
```

显存不足时建议按以下顺序调整：

1. 降低 `Dataset.train_batch_size`。
2. 增大 `Trainer.accumulate_grad_batches` 保持有效全局 batch。
3. 降低 `Dataset.num_workers` 只会缓解 CPU/内存压力，不缓解 GPU OOM。
4. 不建议首先减小 map 半径或关闭 map context，因为这会直接影响道路约束。

训练速度过慢时重点检查：

1. `Trainer.limit_val_batches` 是否过大。
2. step visualization 是否过频繁。
3. `safety_topk` 是否过大。
4. `debug_validation_logging` 日志是否显示耗时集中在 rollout、energy metric
   或 visualization。

## 17. 参数联动约束速查

```text
Dataset.num_historical_steps
  == Model.num_historical_steps

Dataset.num_future_steps
  == Model.decoder.num_future_steps
  == Model.diffusion.total_rollout_steps

Dataset.token_size
  == Model.decoder.token_size

Model.diffusion.future_chunk_steps
  == Model.diffusion.token_steps
  == SMARTAgentDecoder.shift
  == 5

Model.diffusion.history_tokens
  == (Model.num_historical_steps - 1) / token_steps
  == 2

Model.diffusion.prediction_tokens
  == 4

Model.diffusion.commit_tokens
  == 1

total_rollout_steps
  % (commit_tokens * token_steps)
  == 0

Model.diffusion.num_steps
  == prediction_tokens

Model.total_steps
  > Model.warmup_steps
```

## 18. 主要代码位置

- 训练入口：[`train.py`](../train.py)
- DataModule：[`smart/datamodules/scalable_datamodule.py`](../smart/datamodules/scalable_datamodule.py)
- Dataset：[`smart/datasets/scalable_dataset.py`](../smart/datasets/scalable_dataset.py)
- 原始 SMART：[`smart/model/smart.py`](../smart/model/smart.py)
- diffusion 父类：[`smart/model/smart_diffusion.py`](../smart/model/smart_diffusion.py)
- AR 父类：[`smart/model/smart_ar_diffusion.py`](../smart/model/smart_ar_diffusion.py)
- causal 模型：[`smart/model/smart_causal_diffusion.py`](../smart/model/smart_causal_diffusion.py)
- causal decoder：[`smart/modules/causal_diffusion_decoder.py`](../smart/modules/causal_diffusion_decoder.py)
- safety energy：[`smart/modules/trajectory_energy.py`](../smart/modules/trajectory_energy.py)
