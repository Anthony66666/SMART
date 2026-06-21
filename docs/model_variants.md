# SMART 模型变体总览

这份文档总结当前 checkout 中存在的主要模型变体，并重点说明它们在继承关系、预测对象、训练目标、推理 commit 方式、proposal / memory / rerank / guidance 机制上的差异。这里把当前 `train.py` / `val.py` 可以直接路由的 predictor 和仍保留在代码树中的历史 JEPA 分支分开写，避免把旧实验配置误认为当前主线模型。

## 当前入口支持的模型

当前 `train.py` 和 `val.py` 注册了以下 predictor：

| Predictor | 类 | 主文件 |
| --- | --- | --- |
| `smart` | `SMART` | `smart/model/smart.py` |
| `smart_diffusion` | `SMARTDiffusion` | `smart/model/smart_diffusion.py` |
| `smart_ar_diffusion` | `SMARTAutoregressiveDiffusion` | `smart/model/smart_ar_diffusion.py` |
| `smart_causal_diffusion` | `SMARTCausalDiffusion` | `smart/model/smart_causal_diffusion.py` |
| `smart_causal_flow_matching` | `SMARTCausalFlowMatching` | `smart/model/smart_causal_flow_matching.py` |
| `smart_elf` | `SMARTEmbeddedLanguageFlow` | `smart/model/smart_elf.py` |
| `smart_hybrid_diffusion` | `SMARTHybridDiffusion` | `smart/model/smart_hybrid_diffusion.py` |
| `smart_action_chunk_diffusion` | `SMARTActionChunkDiffusion` | `smart/model/smart_action_chunk_diffusion.py` |
| `smart_continuous_action_diffusion` | `SMARTContinuousActionDiffusion` | `smart/model/smart_continuous_action_diffusion.py` |
| `smart_discrete_diffusion_policy` | `SMARTDiscreteDiffusionPolicy` | `smart/model/smart_discrete_diffusion_policy.py` |

## 1. 原始 SMART

配置示例：

- `configs/train/train_scalable.yaml`
- `configs/validation/validation_scalable.yaml`
- `configs/train/train_scalable_ntp_from_jepa_ego30_hidden.yaml`

`SMART` 是原始 SMART token predictor。它直接使用 SMART 的 map decoder 和 agent decoder，训练时预测下一个 motion token 的 logits，并用 cross entropy 监督；推理时使用原始 SMART 的 autoregressive token rollout。

关键点：

- Predictor：`smart`
- 类：`SMART`
- 训练目标：原始 next-token classification CE。
- 推理方式：原始 SMART AR token rollout。
- 地图使用：原始 SMART map-token / map-agent attention。
- Diffusion：没有。
- Proposal / rerank / guidance：没有。
- 主要用途：作为 baseline 和 checkpoint 兼容参考。

如果某个 diffusion 变体后段不贴地图，原始 SMART 是最重要的参照：它可以帮助判断问题来自数据/tokenization，还是来自新 rollout / diffusion objective。

## 2. Full-Horizon SMART Diffusion

配置示例：

- `configs/train/train_scalable_diffusion.yaml`
- `configs/train/train_scalable_diffusion_local.yaml`
- `configs/validation/validation_scalable_diffusion.yaml`

`SMARTDiffusion` 把原始 SMART 的 token-by-token 生成改成了整段 future token 的 discrete mask diffusion。它仍然复用 SMART encoder/context，但后面接 diffusion decoder，对未来 token 序列进行迭代 denoise。

关键点：

- Predictor：`smart_diffusion`
- 类：`SMARTDiffusion`，继承 `SMART`。
- 训练目标：future SMART tokens 上的 mask diffusion x0 CE。
- 预测范围：完整 future horizon，通常是 80 帧 / 5 帧一个 token，即 16 个 future tokens。
- 推理方式：一开始把 future token 全部 mask，然后 joint denoise 整个未来序列。
- Commit 语义：没有 receding-horizon commit loop；一次采完整 future token sequence，再 chain decode 成轨迹。
- Proposal carry：没有 AR rolling proposal 机制。
- NTP 辅助：常用配置里 `ntp_aux_loss_weight: 0`。
- 优点：同一场景内多 agent / 多 future chunk 可以联合 denoise，交互一致性更强。
- 弱点：没有每 commit 一个 token 后重新刷新历史和地图上下文，长时段贴图错误可能在 decode 后累积。

这个模型适合作为非 AR 的 diffusion baseline，用来回答“全未来联合 masked denoising 是否比 recurrent token rollout 更好”。

## 3. AR Diffusion Baseline

配置示例：

- `configs/train/train_scalable_ar_diffusion.yaml`
- `configs/train/train_scalable_ar_diffusion_local.yaml`
- `configs/train/train_scalable_ar_diffusion_baseline_1000.yaml`
- `configs/validation/validation_scalable_ar_diffusion.yaml`

`SMARTAutoregressiveDiffusion` 把 discrete diffusion 放进一个 receding-horizon AR 控制器里。每一轮根据当前 committed history 构造短窗口，预测 4 个未来 token，只 commit 第 1 个 token，然后滚动历史、刷新地图，再进入下一轮。

关键点：

- Predictor：`smart_ar_diffusion`
- 类：`SMARTAutoregressiveDiffusion`，继承 `SMARTDiffusion`。
- 基础目标：`ar_objective: maskgit`。
- 每轮预测：`prediction_tokens: 4`。
- 每轮执行：`commit_tokens: 1`。
- Proposal carry：AR baseline 配置中通常 `carry_tail_proposal: true`。
- 推理方式：80 帧 future 下，如果一个 token 是 5 帧、每轮 commit 1 token，则一共 16 轮 rollout。
- NTP 辅助：常用配置里 `ntp_aux_loss_weight: 0`。
- 地图刷新：每次 committed history 滚动后重新筛选/刷新 local map context。
- 重要行为：如果 carry 开启，预测出的 chunk1-3 虽然不 commit，但会作为下一轮 provisional proposal 影响几何、map edge 和 decoder 条件。

相比 full-horizon diffusion，它更贴近 SMART 的闭环 rollout 接口；相比原始 SMART，它不是单 token CE 预测，而是用 diffusion 一次预测短窗口。

## 4. AR Causal-Frontier 变体

配置示例：

- `configs/train/train_scalable_ar_diffusion_frontier_local.yaml`

这是 `SMARTAutoregressiveDiffusion` 内部的一个 opt-in 训练模式，不是单独的类。它保留 AR rollout 入口，但把训练目标向 causal-frontier 思路靠近。

关键点：

- Predictor：`smart_ar_diffusion`
- 类：仍然是 `SMARTAutoregressiveDiffusion`。
- 目标：配置启用时为 `ar_objective: causal_frontier_v1`。
- 目的：在不改变 AR 外壳的情况下，测试 causal temporal edges、frontier-only supervision、current-state context 等想法。
- 和 `smart_causal_diffusion` 的区别：它只是 AR diffusion 的实验模式，不是严格的 causal v2 predictor，也不强制 `num_steps == prediction_tokens`。

这个分支主要是 AR diffusion 和 causal diffusion 之间的过渡 ablation。

## 5. AR Diffusion Rerank

配置示例：

- `configs/train/train_scalable_ar_diffusion_rerank.yaml`
- `configs/train/train_scalable_ar_diffusion_rerank_1000.yaml`
- `configs/train/train_scalable_ar_diffusion_rerank_local.yaml`
- `configs/validation/validation_scalable_ar_diffusion_rerank.yaml`

AR rerank 也不是一个单独的类，而是 `smart_ar_diffusion` 的一套特定配置。它保留 AR diffusion 的 4-token prediction / 1-token commit / tail proposal carry，但加入 commit-aware 训练、causal temporal attention、proposal 初始化增强，以及可选的采样时 rerank。

关键点：

- Predictor：`smart_ar_diffusion`
- 类：`SMARTAutoregressiveDiffusion`。
- 基础目标：`ar_objective: maskgit`。
- 训练模式：`ar_training_mode: cadf_lite`。
- 预测 / 执行：预测 4 个 token，只 commit 1 个。
- Proposal carry：`carry_tail_proposal: true`。
- Proposal conditioning：当前 rerank 配置中启用。
- Causal temporal edges：启用。
- Chunk loss 权重：`[1.0, 0.3, 0.1, 0.05]`，chunk0 权重最高。
- Terminal window：最后不足 4 个 token 的窗口使用 valid mask 保留，不直接丢弃。
- Anchor coverage：deterministic chunk0 cycling，保证每个 future timestep 周期性作为 chunk0 被监督。
- Local NTP CE：`cadf_lite_local_ntp_loss_weight: 1.0`，来自同一次 window forward，不是额外完整 SMART forward。
- Dense SMART CE replay：按 interval 启用，常见为 `dense_smart_ce_interval: 8`。
- Sampling guidance：训练配置里关闭；validation 配置可以把 safe-speed rerank 当作推理 ablation 打开。

这个分支的目标是保留 AR baseline 中比较有用的 rolling proposal 连续性，同时让监督目标更关注真正被执行的 chunk0。

## 6. Causal Diffusion

配置示例：

- `configs/train/train_scalable_causal_diffusion.yaml`
- `configs/train/train_scalable_causal_diffusion_local.yaml`
- `configs/train/train_scalable_causal_diffusion_1000.yaml`
- `configs/validation/validation_scalable_causal_diffusion.yaml`

`SMARTCausalDiffusion` 是更严格的 causal discrete-frontier 模型。它继承 AR diffusion，但强制四个 future token、一次 commit 一个 token，并让 sampling step 和 token frontier 一一对应。

关键点：

- Predictor：`smart_causal_diffusion`
- 类：`SMARTCausalDiffusion`，继承 `SMARTAutoregressiveDiffusion`。
- 训练目标：`causal_objective: discrete_frontier_v2`。
- 预测 / 执行：`prediction_tokens: 4`，`commit_tokens: 1`。
- 采样步数：`num_steps: 4`，每一步释放一个 causal frontier。
- Proposal carry：默认开启。
- Proposal conditioning：默认开启。
- Current state：启用当前运动特征和 current-state edges。
- 训练策略：clean / rollout / perturb 风格的 closed-loop curriculum，并有 retokenization recovery。
- Safety guidance：支持 `guidance.mode = none | safe | ego_stress | ego_edit`。
- 采样选择：safe guidance 会用 lane / dynamics / collision / ego-risk energy 对 top-k frontier token 做 rerank。

这个模型不只是为了最小 ADE/FDE，更偏 controllable closed-loop rollout，尤其适合 safety guidance、ego-risk counterfactual 和可控场景编辑。

## 7. Causal Flow Matching

配置示例：

- `configs/train/train_scalable_causal_flow_matching.yaml`
- `configs/train/train_scalable_causal_flow_matching_local.yaml`
- `configs/train/train_scalable_causal_flow_matching_1000.yaml`
- `configs/validation/validation_scalable_causal_flow_matching.yaml`

`SMARTCausalFlowMatching` 保留 causal diffusion 的 rollout 和 guidance 接口，但把 discrete frontier CE 换成 probability simplex 上的 flow matching。

关键点：

- Predictor：`smart_causal_flow_matching`
- 类：`SMARTCausalFlowMatching`，继承 `SMARTCausalDiffusion`。
- 目标：`causal_objective: flow_matching_v1`。
- 状态空间：token probability distribution，而不是只处理 token id。
- Loss：从 source distribution 到目标 one-hot token distribution 的 velocity regression。
- 采样方式：对当前 frontier 积分 flow，再用 argmax 或 multinomial 选 token。
- Rollout / guidance：继承 causal diffusion 的 receding-horizon rollout 和 guidance 接口。
- Proposal carry：causal flow 配置中开启。
- 当前作用：比较 flow matching 是否比 discrete frontier CE 更容易学出平滑、稳定的 token 分布。

它是 causal diffusion 的替代训练目标，不是完全不同的 rollout 架构。

## 8. Hybrid Diffusion

配置示例：

- `configs/train/train_scalable_hybrid_diffusion_local.yaml`
- `configs/train/train_scalable_hybrid_diffusion_1000.yaml`
- `configs/validation/validation_scalable_hybrid_diffusion.yaml`

`SMARTHybridDiffusion` 是 composition-based 模型。它的公开类不继承其他 predictor，而是在内部持有一个 `SMARTCausalDiffusion` core，并替换 commit-speed scoring 逻辑。

关键点：

- Predictor：`smart_hybrid_diffusion`
- 类：`SMARTHybridDiffusion`，独立 `LightningModule`。
- 内部 core：`SMARTCausalDiffusion`。
- 目标：`hybrid_objective: closed_loop_frontier_v1`。
- 输入：原始 SMART `HeteroData` / `Batch`。
- 预测 / 执行：沿用 causal 的四 token proposal / 一 token commit。
- Proposal carry：开启。
- 主要修改：把原先偏一侧的 slow-token speed penalty 换成双向速度带约束，使用 `commit_min_speed_ratio` 和 `commit_max_speed_ratio`。
- 目的：缓解 commit token 过慢，而不是完全替换 causal diffusion 架构。

这个分支更像 causal diffusion 的 commit-selection / speed calibration 版本。

## 9. SMART ELF

配置示例：

- `configs/train/train_scalable_elf_3epoch_local.yaml`
- `configs/train/train_scalable_elf_1000.yaml`
- `configs/validation/validation_scalable_elf.yaml`

`SMARTEmbeddedLanguageFlow` 是当前最独立的 active branch。它不继承 `SMART`、`SMARTDiffusion`、`SMARTAutoregressiveDiffusion` 或 `SMARTCausalDiffusion`，而是组合 SMART map/history encoder 和 official-ELF-style embedding flow decoder。

关键点：

- Predictor：`smart_elf`
- 类：`SMARTEmbeddedLanguageFlow`，独立 `LightningModule`。
- 目标：`elf_objective: embedded_language_flow_v1`。
- 状态空间：continuous token embedding，而不是主路径上的 token-id logits。
- Window：`elf_window_tokens: 4`。
- Commit：`elf_commit_tokens: 1`。
- 推理：采样 4-token ELF window，只 commit 第 1 个 token，然后滚动 token/frame history，重新 encode，再采下一轮。
- Loss：embedding-flow loss，加可选 decoder CE branch；tail loss 由 `elf_tail_loss_weight` 下调。
- Commit selection：以 ELF embedding similarity 为主，用 map-conditioned commit score 和可选 geometry energy 进行地图 grounding。
- Attention：为了适配 rollout，采用 chunk-causal compromise；同一 chunk 的 agent 可以互相看，前序 chunk query 不看后序 chunk target。
- Supervision：当前 ELF 配置使用 all-agent supervision，而不是只监督 category-3 target。

这是论文创新性最强的分支之一，但由于架构变化较多，旧 checkpoint 很容易 stale，质量判断必须基于当前 standalone 架构重新训练后的结果。

## 10. Discrete Action-Chunk Diffusion

配置示例：

- `configs/train/train_scalable_ar_action_chunk.yaml`
- `configs/train/train_scalable_ar_action_chunk_1000.yaml`
- `configs/validation/validation_scalable_ar_action_chunk.yaml`

`SMARTActionChunkDiffusion` 是 ACT / ALOHA 风格的 discrete action-chunk ablation，建立在 AR diffusion 上。它预测重叠的 4-token chunks，并对同一个将要 commit 的 timestep 聚合多个 overlapping prediction。

关键点：

- Predictor：`smart_action_chunk_diffusion`
- 类：`SMARTActionChunkDiffusion`，继承 `SMARTAutoregressiveDiffusion`。
- 基础目标：AR diffusion，`ar_training_mode: cadf_lite`。
- 预测 / 执行：预测 4 个 token，commit 1 个。
- Proposal carry：当前 action-chunk 配置中关闭。
- Temporal ensemble：默认开启。
- Ensemble 对象：来自当前和之前 overlapping chunks 的 discrete token predictions。
- Shift consistency：可选相邻窗口 overlap logits 的一致性 loss。
- Dense SMART CE replay：当前 server config 中开启。
- Sampling guidance：训练配置里关闭。

它和 AR rerank 的问题意识不同：AR rerank 是 carry tail 作为下一轮条件；action-chunk 是把 tail 当作多个 overlapping action proposal，并在 commit 前做 temporal voting / ensemble。

## 11. Continuous Action Diffusion

配置示例：

- `configs/train/train_scalable_continuous_action_diffusion.yaml`
- `configs/train/train_scalable_continuous_action_diffusion_1000.yaml`
- `configs/validation/validation_scalable_continuous_action_diffusion.yaml`

`SMARTContinuousActionDiffusion` 把 diffusion-policy 思路从 discrete token id 转到 continuous local trajectory chunk。它仍然使用 SMART context，并在 commit 后把连续动作 retokenize 回 SMART token，以兼容历史和接口。

关键点：

- Predictor：`smart_continuous_action_diffusion`
- 类：`SMARTContinuousActionDiffusion`，继承 `SMARTAutoregressiveDiffusion`。
- 目标：`continuous_action_objective: diffusion_policy_v1`。
- 预测对象：每个 future token slot 对应的连续 local 5-frame x/y action chunk。
- 预测 / 执行：预测四 token / 二十帧 action window，只 commit 一个 5-frame chunk。
- Proposal carry：关闭。
- Temporal ensemble：在连续世界轨迹空间里开启。
- Loss：continuous action chunks 上的 MSE denoising loss。
- Tokenization：commit 后用 nearest SMART token 恢复 token id，只是为了维持 SMART history/interface。
- Dense SMART CE replay：当前配置中开启。

这是最接近机器人 diffusion policy 的分支：动作本身连续且更平滑，SMART token 主要用于保持原有管线兼容。

## 12. Pure Discrete Diffusion Policy

配置示例：

- `configs/train/train_scalable_discrete_diffusion_policy.yaml`
- `configs/train/train_scalable_discrete_diffusion_policy_2000.yaml`
- `configs/validation/validation_scalable_discrete_diffusion_policy.yaml`

`SMARTDiscreteDiffusionPolicy` 是最新的 pure discrete diffusion-policy ablation。它继承 AR 外壳只是为了复用 context / rollout plumbing，但刻意关闭大部分 AR / SMART 辅助机制。

关键点：

- Predictor：`smart_discrete_diffusion_policy`
- 类：`SMARTDiscreteDiffusionPolicy`，继承 `SMARTAutoregressiveDiffusion`。
- 目标：`discrete_policy_objective: pure_chunk_v1`。
- 预测对象：四个未来 SMART tokens，即 `[z_{t+1}, z_{t+2}, z_{t+3}, z_{t+4}]`。
- 执行：`execution_horizon: 1`，只 commit chunk0。
- Tail 行为：chunk1-3 在推理时直接丢弃。
- Proposal carry：禁止；构造函数会拒绝 `carry_tail_proposal: true`。
- SMART NTP / prior fusion：禁止。
- Proposal memory：关闭且禁止。
- Temporal ensemble：关闭且禁止。
- Candidate rerank：当前配置关闭。
- Sampling guidance：当前配置关闭。
- 训练：batched multi-anchor windows，forced full-window supervision，chunk-weighted x0 CE，加可选 overlap KL。
- Chunk 权重：`[1.0, 0.3, 0.15, 0.075]`。
- 日志：`loss_x0_chunk0..3`、`chunk0_acc..3`、`loss_overlap`、supervision coverage。

这个分支最适合测试“纯 discrete diffusion-policy 目标本身是否有用”。它也最容易丢失 AR baseline 的 rolling proposal map-continuity，因为后 3 个 lookahead tokens 不 carry、不 ensemble、不 commit。

## 13. SMART JEPA 历史分支

仍存在的配置示例：

- `configs/train/train_scalable_jepa.yaml`
- `configs/train/train_scalable_jepa_a0_visible.yaml`
- `configs/train/train_scalable_jepa_a1_partial_dropout.yaml`
- `configs/train/train_scalable_jepa_a2_hidden.yaml`
- 对应的 `configs/validation/validation_scalable_jepa*.yaml`

`SMARTJEPA` 和 JEPA modules 仍保留在 `smart/model/smart_jepa.py` 和 `smart/model/jepa.py` 中。但当前 `train.py` 和 `val.py` 的 predictor map 没有注册 `smart_jepa`，所以它不是当前可直接运行的主线 predictor。

关键点：

- 旧配置中的 predictor：`smart_jepa`。
- 类：`SMARTJEPA`，继承 `SMART`。
- 当前入口状态：当前 `train.py` / `val.py` 未注册。
- 目标族：根据配置做 future / map latent block 的 JEPA-style prediction。
- 当前角色：历史实验代码保留，不是当前 active train/validation pipeline。

如果之后要恢复 JEPA，第一步应该是重新注册 predictor，然后重新验证配置、loss 和 checkpoint 兼容性。

## 核心差异表

| 变体 | 预测对象 | 训练目标 | 推理 commit | Tail token 处理 | Rerank / guidance | 最适合验证的问题 |
| --- | --- | --- | --- | --- | --- | --- |
| `smart` | 下一个 SMART token | 原始 CE | 原始 SMART rollout | 无 | 无 | 原始 baseline |
| `smart_diffusion` | 完整 future token sequence | full-window mask diffusion CE | 采完整序列后 decode | 非 AR tail | 无 | 联合 denoising baseline |
| AR diffusion baseline | 4 个离散 token | short-window mask diffusion CE | commit 1 | carry 成 proposal | 通常无 | AR diffusion baseline |
| AR frontier | 4 个离散 token | causal-frontier ablation | commit 1 | 依配置 | 通常无 | AR/causal 过渡 ablation |
| AR rerank | 4 个离散 token | cadf_lite + local NTP + CE replay | commit 1 | carry 成 proposal | validation 可开 rerank | 当前最 AR-compatible 的增强离散分支 |
| Causal diffusion | 4 个离散 token | discrete frontier v2 + recovery | 按 frontier commit 1 | carry proposal | safe / ego guidance | 可控 safety rollout |
| Causal flow matching | token distributions | flow velocity loss | 按 frontier commit 1 | carry proposal | 继承 causal guidance | flow 替代 causal CE |
| Hybrid diffusion | 4 个离散 token | causal frontier core | commit 1 | carry proposal | speed-band commit scoring | causal speed calibration |
| SMART ELF | token embeddings | embedding flow + decoder aux | commit 1 | receding re-encode，不是 AR carry | map score / geometry energy | ELF 论文分支 |
| Action chunk | 4 个离散 token | cadf_lite + shift consistency | ensemble 后 commit 1 | ensemble candidates | 训练 guidance 关闭 | ACT-style discrete overlap |
| Continuous action | 连续 5-frame action chunks | continuous denoising MSE | commit 1 continuous chunk | continuous ensemble | 训练 guidance 关闭 | 平滑 diffusion-policy action |
| Discrete diffusion policy | 4 个离散 token | batched chunk-weighted x0 CE + overlap KL | 只 commit chunk0 | 直接丢弃 | 关闭 | pure policy ablation |
| SMART JEPA | future / map latent blocks | JEPA-style latent prediction | 当前入口未启用 | 当前入口未启用 | 当前入口未启用 | 历史分支 |

## 实际选择建议

如果目标是和原始 SMART 最接近地比较，优先看 `smart` 和 AR diffusion baseline。

如果目标是在现有 discrete AR 路线上做增强，同时保留 rollout 连续性，优先看 AR rerank。

如果目标是 safety editing、ego-risk counterfactual 或 guided scene generation，优先看 causal diffusion 或 causal flow matching。

如果目标是测试 diffusion-policy 思路，重点比较 continuous action diffusion 和 pure discrete diffusion policy。Continuous action diffusion 更强调连续几何平滑和 temporal ensembling；pure discrete diffusion policy 更干净，但会丢弃 lookahead tail。

如果目标是论文创新和 ELF 路线，使用 `smart_elf`，但必须基于当前 standalone 架构重新训练后再判断质量。
