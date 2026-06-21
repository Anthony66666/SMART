# AAAI Method 草稿：Causal SMART-Token Diffusion

本文是一版偏 AAAI 论文风格的中文 Method 草稿，用来解释当前
`smart_causal_diffusion` 的模型架构和方法逻辑。它不是最终英文投稿稿，而是作为
写作和理解模型结构的中文基础稿。

## 3. Method

### 3.1 Overview

我们提出一种面向多智能体轨迹预测的因果离散扩散框架，称为
**Causal SMART-Token Diffusion**。该方法继承 SMART 的离散轨迹 token 表示，将
连续未来轨迹预测转化为在 SMART trajectory codebook 上的离散 token 生成问题。
与一次性预测完整未来轨迹的 full-horizon diffusion 不同，我们采用闭环
receding-horizon 规划：每次预测 4 个未来 token，但只提交第 1 个 token，随后根据
已执行的预测结果刷新历史状态、地图上下文和交互关系，再进入下一轮规划。

具体而言，历史输入包含 11 帧，即 1.0 秒历史；每个 SMART token 表示 5 帧，即
0.5 秒。模型每轮规划 4 个 token，对应 2.0 秒未来，但仅提交 1 个 token，对应
0.5 秒。完整 8 秒预测因此由 16 轮闭环 rollout 组成。这样的设计使模型既能保留
短期多 token planning 的能力，又能避免过早锁定远期不确定 token。

### 3.2 SMART Token Representation

给定场景中智能体的历史状态和地图元素，SMART encoder 首先编码 agent 历史、
agent-agent 关系和 map-agent 关系。未来轨迹被划分为固定长度片段，每个片段通过
SMART trajectory codebook 离散化为一个 token。对于 agent \(i\)，未来轨迹可表示为
token 序列：

```text
z_i = (z_i^1, z_i^2, ..., z_i^K)
```

其中每个 \(z_i^k\) 对应 0.5 秒轨迹片段。在 causal diffusion 中，每一轮只处理一个
短窗口：

```text
(z_i^1, z_i^2, z_i^3, z_i^4)
```

模型的目标是在给定历史、地图和其他智能体上下文的条件下，生成这一窗口内的未来
token。

### 3.3 Causal Diffusion Decoder

我们使用一个离散 mask diffusion decoder 对未来 token 进行去噪。输入 token 可以是
真实可见 token，也可以是 absorbing mask token。decoder 的输入特征由以下部分组成：

- token embedding 或 mask embedding；
- diffusion timestep embedding；
- future chunk embedding；
- scene-level context；
- agent history context；
- agent type embedding；
- agent shape embedding；
- geometry confidence embedding；
- 可选的 proposal token embedding；
- map context features。

与普通 diffusion decoder 的关键区别在于 temporal token graph 是严格因果的。对于同一
agent，chunk \(p\) 只能向更晚的 chunk \(q\) 传递信息：

```text
p < q
```

因此，后续尚未确定的未来 token 不会反向影响当前即将执行的 token。这一点对于闭环
控制尤其重要，因为第一个 token 是马上要提交的 action，不能依赖不可执行的远期猜测。

### 3.4 Discrete Frontier Objective

当前模型采用 `discrete_frontier_v2` 目标。训练时，模型不再对所有 masked suffix
token 平均计算分类损失，而是随机选择一个 **frontier chunk** 作为当前训练目标。

设选择的 frontier 为 \(f\)。则：

- chunks `< f` 作为可见 prefix；
- chunks `>= f` 被 mask；
- 离散交叉熵只监督 chunk `f`。

也就是说，训练样本对应的去噪任务是：

```text
p_theta(z^f | z^{<f}, history, map, agents)
```

该设计直接对齐推理过程：推理时模型也是从 chunk 0 到 chunk 3 单调释放，每一步只生成
当前 frontier。当前 4 个 chunk 对应的 timestep 为：

```text
t in {1.0, 0.75, 0.5, 0.25}
```

这避免了旧版本中 sampling step 与实际释放 frontier 不一致的问题。

### 3.5 Closed-Loop Training with Retokenization

为了减少训练和推理之间的状态分布偏移，我们引入闭环训练 curriculum。训练早期使用
干净历史状态；随后加入扰动状态；再进一步加入模型自身 rollout 后的预测状态。

当历史状态被扰动或由模型预测滚动得到时，原始 ground-truth token 可能不再是当前
anchor 下的合理 token。因此我们对未来轨迹重新进行 retokenization：将 ground-truth
future 从 world frame 转换到当前 predicted anchor 的 local frame，并在 SMART codebook
中重新寻找最近 token。

若最近 token 的匹配误差低于类别阈值，则该 token 用于离散交叉熵监督；若误差过大，
则认为离散 token supervision 不可靠，转而使用连续 endpoint recovery loss。这样可以
避免模型在几何上不一致的状态下学习错误 token 标签。

### 3.6 Receding-Horizon Sampling

推理时，模型执行 16 轮闭环 rollout。每一轮中，decoder 从全 mask 状态开始，按 causal
frontier 顺序生成 4 个 token：

```text
z^1 -> z^2 -> z^3 -> z^4
```

但是只有第一个 token \(z^1\) 被提交并解码为未来 0.5 秒轨迹。随后模型将该 token
对应的轨迹滚入历史状态，刷新 agent 位置、heading、历史 token、地图边和交互边，再
进入下一轮预测。

未提交的 \(z^2, z^3, z^4\) 不会成为硬状态，而是作为下一轮的 revisable proposal。
它们以 confidence-weighted proposal embedding 和 proposal geometry 的形式参与下一轮
条件化，但仍然可以被重新采样和修改。

### 3.7 Safety-Aware Token Reranking

为了提高已提交 token 的道路一致性和动态可行性，我们在采样阶段引入 top-k safety
energy reranking。对于当前 frontier，模型先根据 logits 选出 top-k token candidates，
然后计算每个 candidate 的安全能量：

```text
E = lambda_d E_lane-dist
  + lambda_h E_lane-heading
  + lambda_m E_dynamics
  + lambda_c E_collision
```

其中 lane distance 衡量轨迹到最近地图点的距离，lane heading 衡量轨迹朝向与地图方向
的一致性，dynamics 衡量加速度和 yaw rate 是否过大，collision 衡量与其他 agent
candidate 的潜在重叠。

最终 token 不是简单选择最大概率，而是最大化：

```text
log p_theta(z) - alpha(t) E(z)
```

对于普通未来 chunk，`alpha(t) = (1 - t)^2`，使得低噪声阶段安全约束更强；对于第一个
即将提交的 chunk，我们使用固定的 commit safety weight，使执行 token 从第一步就受到
安全约束。

### 3.8 Training and Validation

模型优化时使用两组学习率：SMART encoder 使用较小学习率，decoder 和新增模块使用基础
学习率。当前实现中 encoder learning rate scale 为 0.5。validation 阶段除常规
ADE/FDE 外，还记录 2/4/6/8 秒 ADE/FDE、late-horizon ADE、安全能量、预测覆盖率和
retokenization invalid rate。checkpoint selection 使用以安全性和后半程质量为主导的
`val_rollout_score`。

整体上，该方法的核心思想是：**用 SMART token 保持轨迹 manifold，用 causal frontier
diffusion 对齐训练和执行，用闭环 retokenization 缓解状态分布偏移，用 proposal carry
保留短期计划，用 safety reranking 改善提交动作的道路和交互质量。**
