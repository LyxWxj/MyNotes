# RL 后训练框架设计：核心问题与项目对比

> 来源：分析 `/media/lyxwxj/Data/common/Workspace/Omni-infra/reinforcement-learning` 目录下的 5 个项目（Relax / slime / verl / verl-SpeCo / verl-omni），2026-08-04。

## 一、项目总览

| 项目 | 来源 | 一句话定位 |
|---|---|---|
| verl | verl-project（HybridFlow, EuroSys'25） | 通用 LLM RL 后训练框架，两级数据流抽象，生态最大 |
| Relax | redai-infra（小红书） | 全异步 omni-modal RL 引擎，六层服务化架构，Megatron + SGLang |
| slime | THUDM（智谱） | GLM 系列背后的轻量、生产验证框架，只走 Megatron + SGLang 单一路径 |
| verl-omni | verl-project | verl 的多模态/扩散模型扩展，独立仓库但 pin 住 verl 版本作为依赖 |
| verl-SpeCo | verl-project | verl 的轻量 overlay：投机解码 drafter 共训练（SPECO），不是完整框架 |

**关系**：`verl` 是生态中心，`verl-omni` 与 `verl-SpeCo` 构建在它之上；`Relax`、`slime` 与 `verl` 是"同代人"的独立实现，解决同一组问题。

## 二、设计 RL 框架要解决的核心问题

### 1. 控制流与计算流的分离（HybridFlow 核心洞察）
RL 算法是高层"控制流"（rollout → 算 advantage → 更新模型），神经网络计算是"计算流"。框架要让 PPO/GRPO/GSPO 的差异只体现在控制流上，同时复用同一套训练/推理计算流。
- verl：`SingleController + WorkerGroup` 实现分离（单进程控制流 + 多进程计算流）。
- Relax：把每个角色拆成独立 Ray Serve 服务。

### 2. 训练与推理的解耦
训练（Megatron/FSDP，矩阵反传）与推理（SGLang/vLLM，高吞吐生成）计算特征、并行策略（TP/PP/CP/EP/DP）完全不同，需要决定 GPU 如何划分：
- colocate（同卡共享）：Relax 的 hybrid 模式、slime 默认、verl 默认。
- disaggregated（独立集群）：Relax 全异步模式、verl v1、slime external rollout。

### 3. on-policy 约束 vs 吞吐的权衡
RL 要求用最新策略采样的数据训练，同步模式让训练/推理互相干等 GPU。解法谱系：
- slime `train_async.py`：简单一步预取（double buffering）。
- verl v1：TransferQueue 做轨迹 KV 存储 + 异步采样。
- Relax：彻底全异步，`--max-staleness` 显式控制数据新旧，接受一定 off-policy 换吞吐。

### 4. 权重同步（weight sync）
训练端每步更新权重后必须同步到推理引擎，否则策略漂移：
- colocate：进程内内存拷贝。
- 分离：跨节点 NCCL 广播（Relax DCS）、增量 delta 同步（slime delta-weight-sync）、文件传输（slime external rollout）。
- verl-SpeCo：还要把 drafter 权重热更新回推理引擎。

### 5. 批数据抽象与轨迹管理
rollout 产生的 tokens/logprobs/rewards/masks 需在多阶段高效流转，且变长、多模态（图像/音频）。解法：
- verl：`DataProto`（tensordict 统一数据格式，自动 padding/unpadding）。
- slime：Data Buffer（还支持 partial rollout 回收被 abort 的半成品样本）。
- Relax：TransferQueue partition + StreamingDataLoader。

### 6. 奖励计算与验证
规则奖励（verifiable）、模型奖励（GenRM/LLM-as-judge）、多奖励服务、异步奖励重叠（verl-omni reward_loop）、agentic 环境交互（tool use、sandbox）。slime 的哲学："agentic workflow 就是数据生成"——全部插到同一条数据路径，不 fork 训练内核。

### 7. 算法可插拔
PPO、GRPO、GSPO、DAPO、REINFORCE++、ReMax、DPO……本质是控制流 + loss 的差异，框架必须让算法、奖励、模型后端可组合。

### 8. 工程可靠性
长任务 RL 失败模式与 SFT 不同（rollout 挂掉、长尾样本拖住整轮）：
- checkpoint：Relax 异步保存 + 轮转。
- 容错：Relax HealthManager 自动恢复、slime fault-tolerance。
- 可观测/复现：metrics、tracing、CI 都是一等公民。

## 三、异同点

### 相同点
- 全部基于 Ray 做分布式编排（placement group、actor group、future）。
- 全部采用"训练后端 + 推理引擎"分离架构，共用 Megatron/SGLang/vLLM 生态。
- 解决同一组问题：数据流、权重同步、rollout/train 调度、reward 接入。
- 都以 PPO/GRPO 系算法为默认主线。

### 关键差异

| 维度 | verl | Relax | slime | verl-omni | verl-SpeCo |
|---|---|---|---|---|---|
| 架构哲学 | 两级数据流抽象（控制流/计算流解耦），编程模型驱动 | 六层服务化：每个角色是独立 Ray Serve deployment | 刻意轻量：一条 train/rollout/Data Buffer 主路径，不造多余抽象层 | verl 内核 + 按模型 pipeline 化（pipelines/） | verl 内核 + overlay（verl_speco），import-only 不 patch verl |
| 训练后端 | FSDP/FSDP2/Megatron 多选 | Megatron（TP/PP/CP/EP） | 只 Megatron（原生参数透传） | FSDP2/VeOmni | verl 的 FSDP |
| 推理后端 | vLLM/SGLang/HF 多选 | 只 SGLang | 只 SGLang（--sglang- 参数透传） | vLLM-Omni（多模态加速） | vLLM/SGLang（投机解码集成） |
| 异步程度 | 同步为主，v1 引入 TransferQueue 异步 | 全异步 + 可配置 staleness，另有 hybrid 模式 | 同步 round + Ray future 预取 | 同步为主 + 异步奖励重叠 | 同步 loop + 周期性 drafter 热更新 |
| 模态范围 | LLM/VLM | 全模态（文/图/视频/音频，Qwen3-Omni） | 以 LLM 为主（GLM/Qwen/DeepSeek） | 扩散模型（Qwen-Image/SD3.5/Wan2.2）+ omni | LLM |
| 独有特性 | 算法最全、HybridFlow 编程范式、生态最大 | 服务级弹性扩缩容、DCS 权重广播、GenRM、弹性 rollout | 数据生成自由（agent/sandbox/verifier 即数据源）、partial rollout、动态采样过滤 | 扩散模型 RL 全家桶（FlowGRPO/NFT/DPO）、多奖励服务、rollout correction | drafter 共训（EAGLE1/2/3、DFlash、DSpark、Domino）、NPU 支持 |

### 本质差异一句话概括
- **verl**：追求"抽象正确"——统一编程模型覆盖所有算法和模型。
- **Relax**：追求"解耦极致"——所有角色服务化 + 流式数据，榨干 GPU 利用率，代价是系统复杂度最高。
- **slime**：追求"小而战斗过"——只保留一条被 GLM 系列验证过的路径，反对多引擎最低公分母抽象，上游优化零成本回传。
- **verl-omni**：追求"域扩展"——保留 verl 内核，聚焦扩散/多模态特有难题（latent 打分、VAE、图像/视频 I/O、异步奖励）。
- **verl-SpeCo**：追求"单点增强"——只解决"推理太慢"一个问题，让 drafter 在 RL loop 里跟着变强。

## 四、结论与选型参考
- 要覆盖全模态、追求极致吞吐：看 Relax。
- 要稳定复现 SOTA LLM 后训练、快速接上游优化：看 slime。
- 要做算法研究、多后端灵活切换：看 verl。
- 要做扩散/视频/图像生成 RL：看 verl-omni。
- 要加速现有 verl 训练的推理：看 verl-SpeCo。
