---
type: Note
status: Active
related_to:
  - "[[moe-from-basics-to-stable-latentmoe]]"
  - "[[vllm-moe-inference]]"
  - "[[expert-parallelism-and-moe]]"
---

# MoE 大模型推理调研：LLM、固定长度视频与因果流式视频

> 调研时间：2026-08-08。本文只把论文、作者项目、官方代码或主流框架文档当作事实依据；带有“工程推导”标记的内容是从公开执行模型得到的设计建议，并非现有系统已经完整实现。

## 1. 一页结论

MoE 并不是一种统一的运行时形态。视频生成中至少要区分三类结构：

| 类型 | 路由粒度 | 每次前向激活什么 | 代表模型 | 推理核心矛盾 |
| --- | --- | --- | --- | --- |
| Token-choice sparse MoE | 每个视觉/文本/音频 token | Top-k 小专家 + shared expert | LingBot-Video、Mamoda2.5 | 巨量 token 的路由、All-to-All、负载倾斜 |
| Timestep/stage MoE | 整个扩散时间步 | 一整套高噪或低噪 DiT | Wan2.2 A14B | 两套大权重的驻留/换入，以及阶段批处理 |
| Modality-aware MoE | 模态或模态内 token | 视频、音频或共享专家 | MOVA、JavisDiT++、StreamChar | 模态 token 数和算力不对称、同步截止时间 |

三者都叫 MoE，但不能共享同一套性能结论：

1. **LLM decode 是小 token 批、逐 token、偏权重带宽受限；视频 DiT 是大 token 批、重复去噪、Attention 与专家 GEMM 都很重。** 因此 LLM 低延迟下“把 MoE 当 dense GEMM 算”的做法，在视频数万 token 上通常会产生不可接受的冗余计算。
2. **Wan2.2 的 27B total / 14B active 只保证每个去噪步约算一套 14B DiT，不保证两套权重同时常驻时仍只占 14B 的显存。** 官方代码会按噪声边界选择模型，并在启用 `offload_model` 时把非活跃模型移到 CPU。
3. **固定长度视频没有 LLM 式逐 token KV cache，但会在相同 latent 网格上反复执行几十个时间步。** 这使 step distillation、TeaCache/DBCache、阶段级预取成为视频独有的第一优先级。
4. **因果视频同时承担两类状态：历史帧 KV cache 和 MoE 权重。** 它还必须满足首帧/首块延迟与持续 FPS，而不只是完成整段视频；离线吞吐最优策略可能直接破坏播放截止时间。
5. **降低延迟、节省显存、提高吞吐不是同一个配置。** 低延迟倾向少跨节点通信和权重常驻；显存受限倾向量化、分片和 offload；吞吐倾向大 batch、High-Throughput All-to-All、请求/块流水和热点专家复制。

## 2. 阅读路径

```text
基础数学与 EP
  -> [[moe-from-basics-to-stable-latentmoe]]
  -> [[expert-parallelism-and-moe]]

现有框架实现
  -> [[vllm-moe-inference]]

本次推理调研
  -> [[01-llm-moe-inference-systems|LLM MoE 推理系统]]
  -> [[02-fixed-length-video-moe-inference|固定长度视频 MoE 推理]]
  -> [[03-causal-streaming-video-moe-inference|因果与流式视频 MoE 推理]]
  -> [[04-video-moe-vs-llm-challenges-and-playbook|视频与 LLM 的差异及三场景选型]]
```

## 3. 统一性能模型

对 token-choice MoE，一层的前向可以拆成：

$$
T_{\mathrm{MoE}} = T_{\mathrm{router}} + T_{\mathrm{pack}}
+ T_{\mathrm{dispatch}} + T_{\mathrm{experts}}
+ T_{\mathrm{combine}} + T_{\mathrm{unpack}}.
$$

若本 rank 有 $T$ 个 token、Top-k 为 $k$、hidden size 为 $d$、每元素 $b$ bytes，忽略元数据和本地路由比例，两次 EP 数据移动的逻辑量级约为：

$$
B_{\mathrm{EP}} \approx 2T k d b.
$$

这个式子解释了两种相反现象：

- LLM decode 的 $T$ 小，通信消息很小，固定启动/同步延迟显著，专家 GEMM 也容易退化成 GEMV 或小 GEMM。
- 视频的 $T$ 可达数万到数十万，消息量很大，但专家 GEMM 更容易达到高利用率；瓶颈转为带宽、拓扑和热点 rank 的尾部时间。

对 timestep MoE，路由本身几乎没有 token dispatch：

$$
E(t)=
\begin{cases}
E_{\mathrm{high}}, & t \ge t_{\mathrm{boundary}}\\
E_{\mathrm{low}}, & t < t_{\mathrm{boundary}}.
\end{cases}
$$

其问题不是 token All-to-All，而是：

$$
T_{\mathrm{stage\ switch}}
= T_{\mathrm{weight\ transfer}} + T_{\mathrm{sync}} + T_{\mathrm{warmup}},
$$

以及两个专家是同时常驻、跨设备常驻，还是在阶段边界换入换出。

## 4. 代表模型和证据边界

### 4.1 固定长度/双向视频

| 模型 | MoE 形式 | 已公开信息 | 备注 |
| --- | --- | --- | --- |
| Wan2.2 A14B | 两个 timestep experts | 约 27B total、每步约 14B active；按 SNR/时间步切换高噪与低噪专家 | 不是 token-choice MoE；[官方仓库](https://github.com/Wan-Video/Wan2.2) |
| LingBot-Video | token-choice sparse MoE | 128 routed experts、Top-8；报告 13B-A1.4B 到 120B-A11B | 单流 DiT；[论文](https://arxiv.org/abs/2607.07675)、[官方仓库](https://github.com/robbyant/lingbot-video) |
| Mamoda2.5 | token-choice DiT-MoE | 128 experts、Top-8，25B total / 3B active | 同时使用 few-step distillation；[论文](https://arxiv.org/abs/2605.02641) |
| MOVA | 音视频 MoE | 32B total / 18B active | 联合音视频生成；[论文](https://arxiv.org/abs/2602.08794) |
| JavisDiT++ | modality-specific MoE | 视频/音频模态专属专家与共享建模 | [论文](https://arxiv.org/abs/2602.19163) |

### 4.2 因果/流式视频

| 模型/系统 | 是否 MoE | 因果执行特点 | 在本调研中的角色 |
| --- | --- | --- | --- |
| CausalWan2.2 I2V A14B Preview | 是，继承 Wan2.2 双 Transformer | 8-step causal pipeline；模型卡明确要求 MoE DiT offload | 当前直接公开的因果 MoE 案例；[模型卡](https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers) |
| StreamChar | 是，modality-aware FFN MoE | 长时音视频流、两阶段蒸馏、sink-chunk memory | 多模态流式 MoE 案例；[论文](https://arxiv.org/abs/2605.25659) |
| MAGI-1 | 否，公开描述为 dense DiT | 24 帧一块、最多四块并发、恒定峰值成本 | 块自回归和流水基线；[论文](https://arxiv.org/abs/2505.13211)、[官方仓库](https://github.com/SandAI-org/MAGI-1) |
| StreamDiffusionV2 | 基础模型不以 MoE 为前提 | rolling KV、SLO batching、step/layer pipeline | 流式 serving 基线；[论文](https://arxiv.org/abs/2511.07399) |
| MiniWorld | 否，公开描述为 block-causal DiT | rolling KV + 异步流水去噪 | 有界状态基线；[论文](https://arxiv.org/abs/2608.01127) |

> [!IMPORTANT]
> MAGI-1、StreamDiffusionV2 和 MiniWorld 在这里用于分析因果执行约束，不能据此称为 MoE 模型。

## 5. 三个部署目标的最短答案

| 目标 | LLM MoE | 固定长度视频 MoE | 因果/流式视频 MoE |
| --- | --- | --- | --- |
| 降低延迟 | 少跨节点 EP；LL All-to-All；融合 router/permute/GEMM；量化；小批专用 kernel | 先降采样步/CFG，再 cache；SP/CFG parallel；阶段专家常驻或一次性预取 | 1--4 step causal distillation；rolling KV；首块优先；按 step/layer 做流水 |
| 显存受限 | expert quant + EP；热点缓存；CPU/NVMe offload；避免冗余专家挤占 KV | timestep expert 做 phase offload；token experts 用 EP/HSDP；量化；VAE tile/parallel | 除 expert quant/offload 外必须限制 KV window；避免每块重复换权重 |
| 提高吞吐 | continuous batching；HT All-to-All；DP+EP；EPLB/热点复制；DBO | 同 shape/timestep/expert 分桶；请求级 batch/DP；长序列 SP；双专家流水 | deadline/slack-aware chunk batching；多块异步流水；expert affinity；背压控制 |

完整的优先级、反例和推荐部署拓扑见 [[04-video-moe-vs-llm-challenges-and-playbook]]。

## 6. 使用数字时的四条规则

1. **总参数、激活参数、每 rank 常驻参数分开报。** `A14B` 不等于权重文件只有 14B 参数。
2. **延迟数字必须带步数、分辨率、帧数、CFG、精度、GPU 和并行度。** 缺任一项都很难横比。
3. **videos/s 与 FPS 分开。** 离线一次生成 81 帧的平均 FPS，不等于流式系统每帧都能按播放 deadline 产出。
4. **论文峰值 speedup 不直接相乘。** distillation、cache、量化、稀疏注意力和多卡并行常优化同一段执行时间，组合收益受 Amdahl 定律限制。

## 7. 主要来源

### 模型与视频系统

- [Wan2.2 官方仓库](https://github.com/Wan-Video/Wan2.2)
- [LingBot-Video 论文](https://arxiv.org/abs/2607.07675)
- [CausalWan2.2 模型卡](https://huggingface.co/FastVideo/CausalWan2.2-I2V-A14B-Preview-Diffusers)
- [StreamDiffusionV2](https://arxiv.org/abs/2511.07399)
- [Causal Forcing](https://arxiv.org/abs/2602.02214) 与 [Causal Forcing++](https://arxiv.org/abs/2605.15141)

### MoE 推理系统

- [DeepEP](https://github.com/deepseek-ai/DeepEP)
- [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/)
- [MegaScale-Infer](https://arxiv.org/abs/2504.02263)
- [MoE-Infinity](https://arxiv.org/abs/2401.14361)
- [Fiddler](https://arxiv.org/abs/2402.07033)
- [Klotski](https://arxiv.org/abs/2502.06888)

### 视频推理框架

- [vLLM-Omni diffusion parallelism](https://docs.vllm.ai/projects/vllm-omni/en/latest/user_guide/diffusion/parallelism/overview/)
- [xDiT](https://github.com/xdit-project/xdit)
- [LightX2V](https://github.com/ModelTC/LightX2V)
- [TeaCache](https://arxiv.org/abs/2411.19108)
