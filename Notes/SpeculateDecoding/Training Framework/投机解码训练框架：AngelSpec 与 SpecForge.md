# 投机解码训练框架：AngelSpec 与 SpecForge

基于以下两个仓库的实现与文档整理：

- `AngelSpec`：多架构投机解码 draft 模型训练平台。
- `SpecForge`：面向 SGLang 服务生态的投机解码训练运行时。

## 1. 这类框架训练什么

投机解码训练框架训练的是小型草稿模型 `q_phi`，让它低成本地一次提出多个 token；大模型 target `p` 负责并行验证。正确执行拒绝采样时，服务最终输出的分布仍等于 target 单独解码的分布。

因此，它的目标不是提高 target 的知识、推理能力或对齐程度，而是在不改变 target 输出语义的前提下，提高其解码吞吐、降低延迟。

对于给定前缀的单个位置，draft 分布与 target 分布的理论接受率为：

```text
alpha = sum_v min(p(v), q_phi(v)) = 1 - TV(p, q_phi)
```

连续草稿的价值取决于连续接受的前缀。理想化地，长度为 `K` 的草稿的期望接受长度近似为：

```text
E[accepted length] = sum_{i=1..K} product_{j=1..i} alpha_j
```

这解释了为什么普通 next-token CE 不是充分目标：后面位置会受到 draft 自己先前预测的影响，且真正需要优化的是接受率和接受前缀长度。

## 2. 投机解码训练必须解决的问题

### 2.1 目标函数要和线上接受率一致

仅对真实文本 token 做 teacher forcing，容易出现离线 accuracy 或 CE 改善、线上接受长度却不改善的现象。原因包括：target 的偏好不一定等于数据 token、后续草稿位置会发生 exposure bias，以及 block-parallel 预测的不同位置难度不同。

所以框架需要支持：

- target 分布蒸馏：CE、KL、top-k KL、L1/TV 等；
- 直接或间接对齐接受率的目标，如 LK、D-PACE、端到端多步 TV；
- TTT 或 on-policy 多步展开，让后续位置以 draft 自己的预测为条件；
- 按位置统计接受率、模拟接受长度，以及接入真实 serving engine 的线上 acceptance evaluation。

这里的 "on-policy" 不等于 RL：它只是让 draft 使用自己的历史 token 来生成训练样本或后续输入；监督仍来自 target 分布，而不是奖励和 advantage。

### 2.2 教师数据不是普通 SFT 数据

Feature-space drafter 不只需要文本 token，还依赖 target 在 prefill 时的中间 hidden states、最后层状态或 logits。这带来一组严格的兼容性约束：

- 从 vLLM、SGLang 或 HF target 中捕获正确层号、norm 阶段和张量布局；
- 保证 tokenizer、词表映射、target/draft `lm_head`、mask token 与训练、导出、服务三端一致；
- 将 embedding/lm_head 等从 target 复制并冻结，减少参数量且保持投影语义一致；
- 为 EAGLE/MTP 的自回归 TTT 与 DFlash 一类 block-parallel 模型提供不同的特征和 attention mask。

### 2.3 推理和训练的资源形态不对称

target 是大规模、偏 inference 的 workload；draft 较小、偏 FSDP 训练的 workload；而多层 hidden states 又是高带宽、大体积数据。把二者塞进同一进程或同一组 GPU，会使两侧被迫同规模扩缩容。

典型数据流是：

```text
prompt
  -> frozen target prefill / feature capture
  -> tensor store (hidden states or logits)
  -> draft trainer forward + backward
  -> serving-compatible draft checkpoint
```

控制面只传样本 id、形状、版本、URI 等元数据；大 tensor 通过 Mooncake 或本地 feature store 直接在 producer 与 trainer 间流动。系统还要有 backpressure、预取、资源回收和 checkpoint 对齐，避免 target 快于训练端时耗尽显存或存储。

### 2.4 分布式正确性与长上下文

该类训练同时遇到 target TP/EP、draft DP/FSDP、以及 USP sequence parallel 等拓扑。训练框架需要明确：

- optimizer accumulation 的全局边界；
- 每个 rank 完整且一致的 batch/window，避免 collective 死锁；
- online 流的消费确认、重试、去重与故障恢复；
- 长上下文下的 hidden-state 内存、全词表损失和跨文档 sequence packing；
- block anchor 不能跨越 document 边界，position id 和 RoPE 也需按文档重置。

### 2.5 训练产物必须可直接服务

一个 loss 收敛的 draft checkpoint 并不一定能被服务引擎加载。框架还需要负责 architecture/config 导出、权重 key 转换、词表裁剪映射、target layer layout 与 serving runtime 的版本兼容，并以真实 spec-decode 的 accepted length 和吞吐验证成效。

## 3. AngelSpec 与 SpecForge 的分工

| 维度 | AngelSpec | SpecForge |
| --- | --- | --- |
| 核心定位 | 多架构、目标函数和长上下文训练能力 | SGLang 生态下可部署、可恢复的统一训练运行时 |
| draft 方法 | DFly、DFlash、DFlare、Eagle3、DSpark、MTP | EAGLE3、P-EAGLE、EAGLE3.1、DFlash、Domino、DSpark |
| 训练优化 | TTT、部分 on-policy、CE/top-k KL/LK/D-PACE/TV、packing、USP | typed config、算法 registry、统一 Trainer/Strategy、offline/online-disaggregated 路径 |
| 推理后端 | vLLM、SGLang、HuggingFace | 以 SGLang server capture 与 serving 兼容为中心 |
| 数据/控制面 | Ray controller + Mooncake + FSDP2，异步生产消费 | inference/data/control/training 四平面；FeatureStore、SampleRef、SQLite ledger、durable ack |
| 重点问题 | 如何把 draft 本身训练得更接近 target 并覆盖更多架构 | 如何让在线流、恢复、资源回收、并行拓扑和服务交付可靠可复现 |

### AngelSpec

AngelSpec 更接近投机解码的算法训练平台。它把不同预测范式放进同一 pipeline：Eagle3/MTP 是带 TTT 的自回归草稿；DFlash/DFlare/DFly/DSpark 以 block-parallel 方法用较少的 draft forward 提出整个 token block。其重点在接受率对齐目标、长上下文 USP、文档感知 packing，以及训练中接入真实 serving engine 的 acceptance evaluation。

其基础架构是 Ray actor 管理的推理与训练 worker group，二者通过 Mooncake 传递 target feature tensor，因此可以独立扩展 target feature 生成吞吐和 draft 优化吞吐。

### SpecForge

SpecForge 更接近生产化 runtime。它以 `specforge train --config ...` 作为统一入口，算法差异被收敛到 strategy/provider，训练 loop 本身不因 EAGLE3、DFlash 或 Domino 分支。

它明确划分了：

- inference plane：外部 patched SGLang 捕获 target feature；
- data plane：FeatureStore 读写大 tensor；
- control plane：PromptTask/SampleRef 等无 tensor 元数据、排队、去重和确认；
- training plane：同一 Trainer -> DataLoader -> Controller -> Core -> Strategy 路径。

在线模式是 consume-once stream，因此其重要价值不只是“把 tensor 送到 GPU”，还包括 SQLite durable ledger、按 optimizer window 的 ack、partial tail 清理和 consumer-only recovery。这些机制避免在线流在进程失败、重复投递或不完整 batch 时出现静默的数据丢失、重复训练或分布式不同步。

## 4. 与 RL 训练框架的区别

| 维度 | 投机解码训练框架 | RL 训练框架 |
| --- | --- | --- |
| 被训练对象 | 小 draft `q_phi`；target 通常冻结 | actor policy；常配 critic、reference policy、reward model/verifier |
| 学习信号 | target hidden states、logits、token 分布、验证结果 | environment / rule / reward model 输出的 reward，以及 advantage |
| 优化目标 | 拟合 target、提高接受率和解码吞吐 | 改变 policy 行为、最大化期望回报 |
| 常用损失 | CE、KL、TV、LK、D-PACE、multi-step distillation | PPO/GRPO policy loss、value loss、entropy/KL regularization |
| rollout 含义 | target feature capture 或 draft 多步展开 | 当前 policy 生成轨迹，计算 reward / old logprob / advantage 后更新 |
| 正确性约束 | 拒绝采样下不能改变 target 的输出分布 | 更新本来就要有意改变模型的输出分布 |
| 系统中心 | serving engine 捕获、feature transport、服务 checkpoint 兼容、acceptance | actor/critic/reference/reward workers、trajectory buffer、权重同步、policy lag |
| 主要风险 | feature contract 不一致、接受率低、曝光偏差、tensor 流与并行死锁 | reward hacking、策略崩塌、价值估计误差、奖励延迟和 off-policy 偏差 |

两类框架都可能有 rollout、Ray、FSDP、队列、checkpoint 和在线数据流，但这些相似的基础设施服务于不同的闭环：

```text
投机训练：target feature -> draft distillation -> 更快地服务同一个 target
RL 训练：policy rollout -> reward/advantage -> 改变 policy 行为
```

## 5. 可以组合，而不是互斥

RL 训练本身依赖大量 rollout；将投机 draft 接入 rollout engine 可以降低 RL 的生成成本。更进一步，可以从 actor rollout 或 old-logprob 计算中捕获 feature，周期性训练 draft 并热更新到 rollout engine。

这时职责仍然清晰：RL 框架决定 actor 应当学会什么行为；投机解码训练框架负责让该 actor 更快地产生 rollout。二者共享部分基础设施，但 draft 的蒸馏训练不应被误认为 reward-driven RL。

## 6. 代码阅读依据

- `AngelSpec/README.md`：六类 draft、TTT、接受率目标、packing、online evaluation 和 Mooncake 分离式架构。
- `AngelSpec/docs/concepts/disaggregated_architecture.md`：target feature capture 到 trainer 的异步数据流与 backpressure。
- `AngelSpec/docs/concepts/draft_model_family.md`：自回归 TTT 与 block-parallel draft 的结构取舍。
- `SpecForge/README.md`：SGLang 兼容与统一 typed training entry。
- `SpecForge/specforge/runtime/ARCHITECTURE.md`：offline、disaggregated offline、online 三种运行契约。
- `SpecForge/specforge/runtime/contracts.py` 与 `control_plane/controller.py`：metadata-only control plane、FeatureStore、durable ack/recovery。
