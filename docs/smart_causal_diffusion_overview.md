# SMART Causal Diffusion 当前实现说明

本文说明当前代码里的 `smart_causal_diffusion`。用户口头提到的
`casual diffusion`，在代码和配置里对应的是 `causal diffusion`。

本文以当前实现为准，主要覆盖：

- 代码结构和入口
- 模型继承关系
- 训练数据流和 loss
- 采样/rollout 过程
- safety energy rerank
- validation 指标
- 当前配置和限制

## 1. 一句话概括

`smart_causal_diffusion` 是一个基于 SMART 离散轨迹 token 的闭环、因果、
receding-horizon diffusion planner。

它不是一次性预测完整 8 秒，也不是把 4 个 future token 平等地当成普通
MaskGIT token。当前实现每次只规划一个短窗口：

```text
历史: 2 个 SMART token = 10 帧 = 1.0 s
规划: 4 个 SMART token = 20 帧 = 2.0 s
提交: 1 个 SMART token = 5 帧 = 0.5 s
完整 rollout: 80 帧 = 16 轮
```

每轮内部用 4 次 diffusion decoder 调用，依次释放 chunk 0/1/2/3。第一个
chunk 是真正要执行的 action，后面 3 个 chunk 是可修改的短期 plan proposal。

## 2. 主要文件

核心实现：

```text
smart/model/smart_causal_diffusion.py
smart/modules/causal_diffusion_decoder.py
smart/modules/trajectory_energy.py
```

复用的父类和模块：

```text
smart/model/smart_ar_diffusion.py
smart/model/smart_diffusion.py
smart/modules/diffusion_decoder.py
smart/modules/smart_edge_builder.py
```

入口注册：

```text
train.py
val.py
eval_waymo_official.py
smart/model/__init__.py
```

配置：

```text
configs/train/train_scalable_causal_diffusion.yaml
configs/train/train_scalable_causal_diffusion_local.yaml
configs/validation/validation_scalable_causal_diffusion.yaml
```

测试和工具：

```text
tests/test_smart_causal_diffusion.py
tests/test_trajectory_energy.py
scripts/calibrate_causal_retokenization.py
docs/train_scalable_causal_diffusion_config.md
```

## 3. 继承结构

当前继承链是：

```text
SMARTCausalDiffusion
  -> SMARTAutoregressiveDiffusion
    -> SMARTDiffusion
      -> SMART
```

各层职责大致是：

| 层 | 主要职责 |
| --- | --- |
| `SMART` | 原始 SMART encoder、历史 agent/map 编码、token codebook 相关能力 |
| `SMARTDiffusion` | diffusion token 打包、物理 token embedding、地图上下文、动态几何刷新、基础 validation |
| `SMARTAutoregressiveDiffusion` | 外层 receding-horizon rollout、rolling history、短窗口训练 view、tail proposal carry |
| `SMARTCausalDiffusion` | 严格因果 temporal graph、离散 frontier objective、闭环 retokenization、current-state conditioning、safety rerank、causal rollout metrics |

所以 causal diffusion 不是完全独立重写，而是在 AR diffusion 的外层闭环框架上，
替换了窗口内部的 denoising 规则、训练目标和候选 token 选择策略。

## 4. 构造时的硬约束

`SMARTCausalDiffusion.__init__()` 会强制检查这些条件：

```text
diffusion.prediction_tokens == 4
diffusion.commit_tokens == 1
diffusion.num_steps == num_future_chunks == 4
diffusion.causal_objective == discrete_frontier_v2
```

原因是当前 v2 设计要求“一次采样 step 释放一个 causal frontier”。如果
`num_steps` 不是 4，就会出现 idle step 或多个 frontier 混在同一步释放，训练和
推理语义会重新错位。

构造时还会固定或关闭一些父类能力：

```text
remask_sampling = False
prefix_constrained_sampling = True
prefix_constrained_training = True
causal_noise_schedule = False
visible_token_corruption_prob = 0.0
self_condition_prob = 0.0
use_proposal_geometry = proposal_conditioning_enabled
ar_carry_tail_proposal = carry_tail_proposal
```

这意味着当前 causal diffusion 不是 MaskGIT 式反复 remask，也不再使用旧
AR diffusion 的 visible-token corruption/self-conditioning 默认路径。它走的是
单调 frontier release。

## 5. Decoder 结构

### 5.1 `CausalDiffusionDecoder`

`smart/modules/causal_diffusion_decoder.py` 很薄，只继承 `DiffusionDecoder` 并覆盖
temporal token edge 构造。

核心变化是调用：

```text
build_temporal_edges_from_flat(..., causal=True)
```

也就是同一 agent 的 future-token temporal attention 只允许从更早 chunk 指向
更晚 chunk。后面的 chunk 不能反向影响前面的 executable chunk。

### 5.2 仍然复用 `DiffusionDecoder`

`DiffusionDecoder.forward()` 仍负责：

- token/mask embedding
- diffusion timestep embedding
- chunk embedding
- scene summary projection
- agent context projection
- agent shape embedding
- agent type embedding
- proposal token embedding
- geometry confidence embedding
- temporal attention
- map-to-token attention
- spatial token-token attention
- token-id logits 输出

因此 causal decoder 的“因果性”主要体现在 temporal edge 方向；空间边、地图边、
feature composition 和输出头仍沿用原 diffusion decoder。

## 6. 输入打包和 conditioning

### 6.1 基础打包

`SMARTDiffusion._build_diffusion_inputs()` 做基础工作：

1. 调用 `encoder.encode_history_context(data)` 得到历史 agent/map 表征。
2. 构造 future token target、valid mask、generation agent mask、supervision agent mask。
3. 按 scene 把每个 agent 的 4 个 future token 展平成 `[B, L]` padded 序列。
4. 保存 `token_ids`、`chunk_ids`、`valid_mask`、`loss_mask_base`、`agent_context`、
   `agent_type_ids`、`agent_shape_embeddings`、`agent_maps` 等。
5. 打包 map context。AR/causal 路径的 `local_map_refresh: rescreen` 会保留 scene
   内可见 map token，再由 decoder 根据当前 token geometry 做 radius edge。

一个 packed sequence 的顺序可以理解为：

```text
agent0_chunk0, agent0_chunk1, agent0_chunk2, agent0_chunk3,
agent1_chunk0, agent1_chunk1, agent1_chunk2, agent1_chunk3,
...
```

### 6.2 causal 额外 conditioning

`SMARTCausalDiffusion._build_diffusion_inputs()` 在父类打包后追加 current-state 信息。

它从当前帧和上一帧计算：

```text
velocity
current_heading
longitudinal velocity
lateral velocity
speed
yaw_rate
current_valid
```

然后把 5 维 state feature 过 `current_state_projection` 投到 hidden dim，并加到该
agent 的 4 个 chunk 的 `agent_context` 上。

此外，它还把当前速度和 heading pack 到：

```text
packed["current_velocities"]
packed["current_headings"]
```

这些主要给 commit token 的 dynamics energy 使用，用来约束“观测状态到第一个预测
token”的速度/转角跳变。

### 6.3 history context 的 recency pooling

causal path 覆盖了 `_pool_agent_context()`。它不是简单平均历史 token，而是按
`history_recency_decay` 对越新的历史 token 给越大权重。

当前配置：

```text
history_recency_decay: 0.5
```

这让最新的历史 motion state 在 agent context 中更突出。

## 7. 训练视图

训练入口是 `SMARTCausalDiffusion.training_step()`：

```text
prepare batch
-> build causal training view
-> build diffusion inputs
-> pack retokenization metadata
-> compute causal diffusion loss
-> optional NTP loss
-> log train metrics
```

### 7.1 rolling anchor

由于 causal diffusion 每次只训练一个 4-token window，父类
`SMARTAutoregressiveDiffusion._build_ar_training_view()` 会从完整序列里选择一个
anchor：

```text
历史 token: anchor 前 2 个
目标 token: anchor 后 4 个
历史帧: 当前 token anchor 对应的最近 11 帧
未来帧: 4 token * 5 帧
```

训练时如果 `rolling_anchor_training: true`，anchor 可以在可用时间范围内滚动采样。

### 7.2 闭环 curriculum

当前 `_closed_loop_curriculum(epoch)` 为了兼容旧调用仍返回两个概率槽位：

```text
unused_state_noise_prob, rollout_prob
```

schedule 是：

| epoch | Gaussian state perturb | model rollout |
| --- | --- | --- |
| 0-3 | 0.00 | `closed_loop_batch_ratio_max`，当前 0.5 |
| 4+ | 0.00 | `closed_loop_batch_ratio_max`，当前 0.5 |

model-rollout view 会先用当前模型 no-grad 采样 1 到 4 个 committed token，把 agent
状态滚动到预测状态，再从这个新状态继续构造训练目标。

### 7.3 retokenization

一旦历史状态被模型 rollout 改写，原始 GT future token 就可能不再是这个新 anchor
下的合理 token。causal path 会重新 retokenize：

1. 取新 anchor 后的 4 个 token 对应的 GT world-frame future positions。
2. 对每个 chunk，把 world future 转到当前 token anchor 的 local frame。
3. 在 veh/ped/cyc 各自 SMART codebook 中找最近 token。
4. 得到 nearest token id、匹配误差、local endpoint。
5. 匹配误差超过阈值的目标不参与离散 CE，改走连续 recovery loss。

当前 frozen P99 阈值是：

```text
veh: 0.7379697561264038
ped: 0.8562850952148438
cyc: 1.2705252170562744
```

`scripts/calibrate_causal_retokenization.py` 用来从数据集中统计这些阈值。

## 8. 训练目标：`discrete_frontier_v2`

当前 causal loss 在 `_compute_diffusion_loss()` 中实现。

它的关键点是：每个 packed sequence 随机选择一个可监督 frontier chunk，而不是对所有
masked suffix token 平均 CE。

流程：

1. 从 `raw_loss_mask_base` 中找可监督 chunk。
2. 对每个 sequence 随机选一个 `frontier_id`。
3. mask 掉 `chunk_id >= frontier_id` 的 suffix。
4. decoder 看到可见 prefix 和 masked suffix。
5. 离散 CE 只监督 `chunk_id == frontier_id` 的 token。
6. 如果该 frontier 由于 retokenization error 超阈值无效，不做 CE。
7. 对这些 invalid frontier 位置计算连续 recovery loss。

也就是说：

```text
可见: chunks < frontier
masked: chunks >= frontier
离散分类监督: chunk == frontier
连续 recovery: frontier 但离散 retokenization 无效的位置
```

训练中的 diffusion timestep 由 frontier 决定：

```text
t = 1.0 - frontier_id / num_future_chunks
```

当前 4 个 chunk 对应：

```text
frontier 0 -> t = 1.00
frontier 1 -> t = 0.75
frontier 2 -> t = 0.50
frontier 3 -> t = 0.25
```

这和采样时的 4 次释放对齐。

## 9. 采样流程

窗口内采样由 `SMARTCausalDiffusion._diffusion_sample()` 实现。

初始状态：

```text
sampled = 全 mask token
confidence = 0
previous_reveal_count = 0
```

每一步：

1. `_causal_reveal_count(step, num_steps, num_chunks)` 返回 `step + 1`。
2. 通过 `_prefix_frontier_mask()` 找到每个 agent 当前最早的 masked valid chunk。
3. decoder 只对当前 frontier 采样。
4. 采到的 token 写回 `sampled`，不再 remask。
5. 进入下一步，frontier 自动推进。

当前配置：

```text
num_steps: 4
prediction_tokens: 4
```

所以每轮采样释放顺序固定为：

```text
step 0: chunk 0
step 1: chunk 1
step 2: chunk 2
step 3: chunk 3
```

如果最终还有 valid token 保持 mask，会直接抛错：

```text
RuntimeError("Causal diffusion sampling ended with masked valid tokens.")
```

### 9.1 geometry refresh

采样和训练 decode 前都会调用 `_refresh_token_geometry()`：

- 已知 token 用自身 token trajectory 推进 geometry。
- masked token fallback 到最近已知 pose。
- 如果 proposal geometry 开启，上一轮未提交的 proposal token 可以给 masked chunk
  提供几何和 embedding conditioning。
- `geometry_confidence_source_threshold` 控制哪些 token geometry 可以作为 temporal/spatial
  source。

causal path 还允许 chunk 0 的 current-state anchor 在 all-mask 初始状态下作为 source，
避免第一个 executable token 完全没有 interaction source。

### 9.2 proposal carry

外层 AR rollout 中，每次 4-token window 只提交第 1 个 token。剩余 3 个 token 通过
`_next_tail_proposal()` 作为下一轮的 proposal：

```text
上一轮 sampled chunks: [c0, c1, c2, c3]
提交: c0
下一轮 proposal: [c1, c2, c3, empty]
```

proposal 不是 hard state，也不会被直接执行。它只是：

- 提供 masked token 的 proposal geometry。
- 提供 confidence-weighted physical token embedding。
- 在下一轮重新采样时可以被修改。

这就是“four-token revisable plan with one-token commit”的实现方式。

## 10. Safety energy rerank

当前采样不是直接按 logits multinomial 抽最终 token。如果
`safety_energy_enabled: true`，会对 frontier 的 top-k token 做 soft rerank。

入口是：

```text
SMARTCausalDiffusion._guided_frontier_tokens()
TrajectoryEnergy
```

当前配置：

```text
safety_topk: 16
safety_energy_weight: 1.0
commit_safety_weight: 1.0
lane_distance_energy_weight: 1.0
lane_heading_energy_weight: 0.5
dynamics_energy_weight: 0.25
collision_energy_weight: 2.0
```

### 10.1 energy 项

`TrajectoryEnergy` 提供三类能量：

| energy | 含义 |
| --- | --- |
| lane distance | candidate trajectory 到最近 map point 的平均距离 |
| lane heading | candidate heading 与最近 map orientation 的角度差 |
| dynamics | 加速度和 yaw rate 超出阈值的平方惩罚 |
| collision | 与其他 agent nominal candidate 的近距离 overlap 惩罚 |

其中 collision 是在先用 lane/dynamics 初筛出 nominal candidate 后，再计算相互碰撞能量。

### 10.2 rerank 公式

候选 token 的调整分数是：

```text
adjusted_score = log_prob - scale * energy
```

一般 chunk 的 scale 是：

```text
safety_energy_weight * (1 - t)^2
```

但 chunk 0 是即将执行的 commit token，会使用固定的：

```text
commit_safety_weight
```

这保证第一步执行 token 从一开始就有 safety 约束，而后续可修改 plan 仍然随噪声水平逐步增强 safety。

注意：这是 top-k soft rerank，不是把轨迹硬投影到车道线上。模型仍然只能选择 SMART
离散 token codebook 中的候选。

## 11. 外层 16 轮 rollout

`SMARTAutoregressiveDiffusion.inference()` 负责完整 80 帧预测。

当前 causal config 下：

```text
total_rollout_steps = 80
commit_tokens = 1
token_steps = 5
rounds = 80 / (1 * 5) = 16
```

每一轮：

1. 用当前 rolling history 构造 `rollout_view`。
2. 重新 encode history context。
3. 重新打包 agent token 和 map context。
4. 调用 causal `_diffusion_sample()` 得到 4 个 token。
5. 只提交第 1 个 token。
6. 解码这个 token 的 5 帧轨迹和 heading。
7. 写入 `pred_traj` / `pred_head` / `pred_valid_mask`。
8. 把 committed token 滚进 history。
9. 把未提交 tail token 作为下一轮 proposal。
10. 进入下一轮。

这和 full-horizon diffusion 最大区别是：地图、agent history、当前运动状态都会每 0.5 秒
闭环刷新一次。

## 12. Validation 和日志

causal validation 复用 AR validation 框架：

- 先计算一个短窗口 `val_ar_window_loss`。
- 再按配置决定是否跑完整 inference。
- 完整 inference 后更新 SMART-compatible metrics。

常规指标：

```text
val_minADE
val_minFDE
val_conflict_rate
val_interaction_consistency
```

causal 额外指标由 `_log_additional_rollout_metrics()` 记录：

```text
val_ADE_2s / val_FDE_2s
val_ADE_4s / val_FDE_4s
val_ADE_6s / val_FDE_6s
val_ADE_8s / val_FDE_8s
val_late_ADE_4s
val_energy_lane_distance
val_energy_lane_heading
val_energy_dynamics
val_energy_collision
val_prediction_coverage
val_retokenization_invalid_rate
val_rollout_score
```

checkpoint 默认监控：

```text
monitor_metric: val_rollout_score
monitor_mode: min
```

`val_rollout_score` 是安全和 late-horizon 质量导向的加权分数，不是官方唯一指标。
官方 SMART 风格的 ADE/FDE 仍用于 baseline 对比。

## 13. 当前 server/local 配置含义

server config：

```text
configs/train/train_scalable_causal_diffusion.yaml
```

关键值：

```text
devices: 14
max_epochs: 32
train_batch_size: 4
limit_val_batches: 50
lr: 0.0005
warmup_steps: 2
total_steps: 32
encoder_lr_scale: 0.5
```

local smoke config：

```text
configs/train/train_scalable_causal_diffusion_local.yaml
```

关键值：

```text
devices: 1
max_epochs: 5
train_raw_dir: data/valid_demo
val_raw_dir: data/valid_demo
limit_val_batches: 1
warmup_steps: 1
total_steps: 5
```

当前 Lightning 返回的是 bare `LambdaLR`，实际按 epoch step。因此 server 的
`warmup_steps: 2`、`total_steps: 32` 是 epoch 级调度。

## 14. 和其他 predictor 的区别

### 14.1 和 `smart`

`smart` 是原始 SMART autoregressive token predictor。它按 SMART 原始接口做预测，不使用
diffusion mask denoising，也没有 causal frontier release。

### 14.2 和 `smart_diffusion`

`smart_diffusion` 是 full-horizon/joint diffusion，对完整未来 token 序列做 denoising。
它更像一次性预测 8 秒，而 causal diffusion 每 0.5 秒闭环刷新一次。

### 14.3 和 `smart_ar_diffusion`

`smart_ar_diffusion` 已经有短窗口 AR rollout、proposal carry 和 map rescreening。
但 causal diffusion 在此基础上新增/替换了：

- 严格 causal temporal token graph。
- `discrete_frontier_v2` 训练目标。
- 单调 frontier release，不 remask。
- 训练时 clean/model-rollout state curriculum。
- predicted/noised-history anchor 下的 deterministic retokenization，以及 SMART-style top-k history-token noise。
- retokenization invalid 的 continuous recovery。
- current motion conditioning。
- all-mask chunk-0 current-state source。
- commit-aware safety energy rerank。
- causal rollout score 和 horizon/safety metrics。

## 15. 当前状态和限制

当前 causal diffusion 是 v2 redesign，应当从头训练。

不要把旧 checkpoint 当成当前 v2 的质量判断依据，尤其不要继续沿用旧
`/mnt/d/causal_epoch=00.ckpt` 作为可恢复训练点。v2 的 objective、state conditioning、
proposal path 和 decoder input 语义都已经改变。

当前显式非目标：

- 不改变原始 `smart`、`smart_diffusion`、`smart_ar_diffusion` 默认行为。
- 不把轨迹硬投影到 lane 上。
- 不在第一版把 traffic light 作为硬规则。
- 不把 safety energy 当成可微训练损失；它目前是 sampling-time top-k rerank。

当前需要继续验证的重点：

- local 5-epoch demo 是否能 overfit 11 个 demo scenes。
- 从零 server run 的 `val_rollout_score`、ADE/FDE、late ADE、map/safety energy。
- 和原始 SMART、旧 causal checkpoint、当前 AR diffusion checkpoint 在同一 validation
  scenes 上的对比。

## 16. 推荐阅读顺序

如果要继续改 causal diffusion，建议按这个顺序读：

1. `docs/smart_causal_diffusion_overview.md`
2. `docs/train_scalable_causal_diffusion_config.md`
3. `smart/model/smart_causal_diffusion.py`
4. `smart/model/smart_ar_diffusion.py`
5. `smart/model/smart_diffusion.py`
6. `smart/modules/causal_diffusion_decoder.py`
7. `smart/modules/trajectory_energy.py`
8. `tests/test_smart_causal_diffusion.py`
9. `tests/test_trajectory_energy.py`
