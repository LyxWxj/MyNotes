---
type: Note
status: Active
related_to:
  - "[[00-moe-inference-research-index]]"
  - "[[vllm-moe-inference]]"
  - "[[expert-parallelism-and-moe]]"
---

# LLM MoE 推理系统：从低延迟 decode 到高吞吐 Expert Parallel

## 1. 问题边界

本文关注 decoder-only LLM 中以 FFN 为专家的 token-choice sparse MoE。模型结构和 vLLM 调用链已分别整理在 [[moe-from-basics-to-stable-latentmoe]] 与 [[vllm-moe-inference]]；这里回答部署问题：不同负载下瓶颈在哪里，应怎样组合 kernel、并行、负载均衡和 offload。

在线 LLM serving 的三个常用指标是：

- **TTFT**：prefill 完成并产生首 token 的时间；
- **TPOT**：decode 每个输出 token 的间隔；
- **goodput**：满足 TTFT/TPOT SLO 的请求或 token 吞吐，而非无约束峰值吞吐。

## 2. Prefill 和 decode 是两种 MoE 工作负载

| 属性 | Prefill | Decode |
| --- | --- | --- |
| 每次进入 MoE 的 token 数 | prompt tokens 的大批量 | 活跃序列各 1 token |
| 专家 GEMM | 较大的 grouped GEMM，较易 compute-bound | 每专家只有少量 token，常 memory-bound |
| EP 通信 | 大消息，带宽重要 | 小消息，启动/同步延迟重要 |
| 主要 SLO | TTFT | TPOT、抖动 |
| 推荐通信模式 | High Throughput、层次化聚合 | Low Latency、直接 GPU-initiated 路径 |

[NCCL EP](https://arxiv.org/abs/2603.13606) 对这一区分给出清楚的工程边界：LL 模式面向约 1--128 token 的小批，HT 模式面向约 4096+ token 的大批；[DeepEP](https://github.com/deepseek-ai/DeepEP) 同样提供面向低延迟和高吞吐的 dispatch/combine 实现。

### 2.1 为什么 decode 可能“算得少却不快”

假设一层有 $E$ 个 routed experts，每 token 选 $k$ 个。平均每 expert 的 assignment 数近似为：

$$
m_e \approx \frac{T k}{E}.
$$

当 $T$ 很小时，$m_e$ 可能只有 0、1 或几项。此时：

- 读取专家权重的代价远大于对少量 token 的乘加；
- pack/sort、TopK、kernel launch 和同步成为固定成本；
- grouped GEMM 被切成许多窄小问题，Tensor Core 利用率低；
- 某个 rank 多收到几个热点 token 就可能决定整个 collective 的尾延迟。

TensorRT-LLM 的 [MoE as Dense GEMM](https://nvidia.github.io/TensorRT-LLM/1.3.0rc15/blogs/tech_blog/blog24_MoE_as_Dense_GEMM.html) 展示了一个反直觉结果：在 Blackwell、NVFP4、TP8 的特定 64--208 token 区间，把 routed experts 表示为一次 dense GEMM 再用 alpha mask 保留 TopK，可能比许多小 grouped GEMM 更快；但在更小或更大批次都不占优。这说明 kernel 必须按 token 数和路由分布动态选，而不是“稀疏一定更快”。

## 3. 参数放置：TP、EP 与 DP+EP

### 3.1 Tensor Parallel over experts

每张卡持有所有 experts 的一部分权重；token 不需要按 expert owner 远程迁移，但每层需要 TP collective。

适合：

- 低并发、延迟优先；
- expert 数少或每 expert 很大；
- 单个 NVLink domain 内，TP collective 很快；
- 希望避免极端路由倾斜导致的 EP straggler。

代价是所有 rank 都涉及所有 expert，权重局部性和规模扩展较差。

### 3.2 Expert Parallel

每张卡持有部分完整 experts，token 通过 dispatch All-to-All 到 owner，计算后 combine 回源 rank。其显存优势是每 rank routed expert 参数约降至 $1/EP$，但 dense attention、shared experts 和其他层通常仍复制或按其他轴分片。

适合：

- routed experts 是主要权重，单 rank 放不下；
- 有 NVLink/NVSwitch 或高质量 RDMA；
- token batch 足以摊薄通信；
- 可以用 EPLB 或冗余 expert 控制热点。

### 3.3 DP attention + EP experts

现代宽 EP 常让 attention 按 DP 复制、MoE expert 跨 DP ranks 组成 EP group。这降低 attention 的 TP 通信并扩大 expert 池，但每个 MoE layer 都出现 DP token 到 EP owner 的全局 dispatch/combine。

vLLM 的 [Expert Parallel Deployment](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) 和 [TensorRT-LLM EP 文档](https://nvidia.github.io/TensorRT-LLM/latest/legacy/advanced/expert-parallelism.html) 都支持 EP 或 TP×EP 混合。实际最优点取决于模型 expert 粒度、batch 和拓扑，不应只按 GPU 数等分。

## 4. 五层优化栈

### 4.1 Router 与数据布局

标准数据流是：

```text
router logits -> TopK -> histogram/prefix-sum -> permute/pack
 -> dispatch -> expert grouped GEMM -> combine -> weighted unpermute
```

优化目标包括：

- TopK、归一化和 expert-id remap 融合；
- 避免 CPU 读取动态 token count；
- contiguous layout 用于 prefill/大批，batched/masked layout 用于 CUDA Graph decode；
- 将 TopK weight 融入 activation 或 down projection；
- 预分配最大容量 buffer，减少动态分配和 host sync。

vLLM 的 [Fused MoE Modular Kernel](https://docs.vllm.ai/en/stable/design/fused_moe_modular_kernel.html) 把 prepare/finalize 与 expert kernel 解耦；[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) 分别提供 contiguous grouped GEMM 和适合 decode/CUDA Graph 的 masked grouped GEMM。

### 4.2 Expert kernel

按工作区间选择：

| 工作区间 | 常见实现 | 重点 |
| --- | --- | --- |
| 极小 token | fused GEMV/小 GEMM、TP、特化 kernel | launch 与权重读取 |
| 低延迟小批 | masked grouped GEMM，部分平台可 dense GEMM | 固定 shape、CUDA Graph |
| 中大批 | contiguous grouped GEMM | 减 padding、提高 occupancy |
| 量化专家 | FP8/FP4/W4A8 fused MoE | scale layout、激活量化、通信 dtype |

不要只用总 token 数选 kernel。2026 年的 [DA-MoE](https://arxiv.org/abs/2607.23099) 表明，同样的 token 数下，路由倾斜会改变 tile padding、权重复用和最优 kernel；它用 GPU resident histogram 做运行时 dispatch，在论文工作负载上降低 fused-MoE latency。

### 4.3 EP 通信

低延迟与高吞吐的优化方向不同：

- **LL decode**：直接 NVLink/RDMA mesh、减少 RTT、GPU 发起、双缓冲；
- **HT prefill**：节点内先聚合，再跨节点传输，追求有效带宽；
- **低精度 dispatch**：hidden activation 用 FP8 等格式发送，combine 保留 BF16/适当精度；
- **通信计算重叠**：限制通信 kernel 使用的 SM，把 dispatch/combine 和其他 microbatch 的 attention/GEMM 重叠；
- **拓扑感知**：group-limited routing 或 owner placement 尽量先利用 NVLink domain，再使用 RDMA。

[DeepEP V1 文档](https://github.com/deepseek-ai/DeepEP/blob/main/docs/legacy.md) 给出了 prefill/训练型 normal kernel 和 decode low-latency kernel 的独立测量；DeepEP V2 将两者统一到 `ElasticBuffer` 接口，并减少通信占用的 SM。

### 4.4 负载均衡和 expert placement

训练时的 aux loss 只改善逻辑 expert 的平均使用率，不能保证生产请求、每层、每个短窗口都均衡。运行时仍需：

1. 记录每层 expert token histogram；
2. 将多个热 experts 分散到不同 ranks；
3. 必要时复制热点 expert，router 的 logical id 映射到多个 physical copies；
4. 周期性重排，但避免迁移权重本身造成停顿；
5. 以 `avg_load / max_load`、P95/P99 rank time 而非均值判断效果。

vLLM 的 [EPLB](https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/) 支持滑动窗口统计、异步重排和 redundant experts。冗余并非免费：文档给出的 DeepSeek-V3 示例中，每 rank 多一个冗余 expert 约增加 2.4 GB，因此显存受限时不应照搬吞吐配置。

### 4.5 Serving 调度

- continuous batching 增大每层 token 数，使 expert GEMM 从 memory-bound 向 compute-bound 移动；
- chunked prefill 避免长 prompt 阻塞 decode，但会改变 MoE 每次 forward 的 token shape；
- prefill/decode 分离后，两边应使用不同 EP kernel 和 batch 上限；
- attention 与 MoE 通信可用双 microbatch 重叠。

vLLM 的 [Dual Batch Overlap](https://docs.vllm.ai/en/latest/design/dbo/) 将 batch 分为两个 microbatch，通过两个执行线程把一个 microbatch 的 MoE All-to-All 与另一个的计算重叠。它主要改善 DP+EP 吞吐，不等价于单请求 TPOT 必然下降。

## 5. 场景一：降低延迟

### 5.1 推荐优先级

1. **先分开测 prefill 与 decode。** 否则大 prompt 的 TTFT 会掩盖 decode 优化，或反之。
2. **能在一个高速互联域内放下就避免跨节点。** 跨节点 EP 的固定延迟常直接进入每层 TPOT。
3. **decode 选择 LL dispatch/combine 和小批 MoE kernel。** 不要用 prefill 吞吐最优配置。
4. **减少 Python/host sync，使用 CUDA Graph 和静态 bucket。** bucket 应包含 token 数、TopK、expert layout 和量化格式。
5. **量化 expert 权重和 activation。** 但 router、归一化和少量敏感层可保留更高精度。
6. **在 shared expert 与 routed expert 之间做 stream overlap。** 前提是不会因争抢 SM/HBM 反而延长关键路径。
7. **对极小 batch 比较 TP、EP 和 dense-GEMM-style backend。** 稀疏 FLOPs 最少不代表延迟最低。

### 5.2 不适合低延迟的做法

- 为省显存逐层从 NVMe/CPU 同步加载 expert；
- 为追求均衡把 EP 扩到大量跨节点 ranks；
- 等待大 batch 才执行；
- 高频 EPLB 权重迁移；
- 无条件复制大量 experts，挤压 KV cache 后导致调度并发下降。

## 6. 场景二：显存受限

### 6.1 量化优先

MoE 的绝大多数总参数在 experts 中，expert-only quantization 往往有最高收益：

- BF16 -> FP8 近似将 expert weight footprint 减半；
- W4A16/NVFP4 可继续降低权重占用，但必须有匹配的 fused kernel，否则只省显存不提速；
- dispatch activation 量化还能减少网络字节，但 combine 精度需验证；
- KV cache 与 expert weights 竞争 HBM，二者应联合做容量规划。

### 6.2 EP 分片

先估算每 rank：

$$
M_{\mathrm{rank}} \approx M_{\mathrm{dense/shared}}
+ \frac{M_{\mathrm{routed}}}{EP}
+ M_{\mathrm{KV}} + M_{\mathrm{workspace}} + M_{\mathrm{redundant}}.
$$

只用 `total_params / EP` 会漏掉 replicated attention/shared expert、KV cache、MoE pack buffer 和 EPLB copies。

### 6.3 Expert offload

当集群总 HBM 仍放不下权重时有三条路线：

| 系统 | 核心思想 | 更适合 |
| --- | --- | --- |
| [MoE-Infinity](https://arxiv.org/abs/2401.14361) | 请求级 activation trace、expert cache 与 prefetch | expert 路径有重复和可预测性 |
| [Fiddler](https://arxiv.org/abs/2402.07033) | 冷 expert 在 CPU 直接计算，避免频繁搬权重 | PCIe 搬运比 CPU expert 计算更差时 |
| [Klotski](https://arxiv.org/abs/2502.06888) | expert-aware multi-batch pipeline，联合规划 I/O 与计算 | 吞吐优先的资源受限离线/在线批处理 |

offload 的基本下界是：若必须为每层传一个此前不在 GPU 的 expert 权重，PCIe/NVMe 时间无法靠 sparse activation 自动消失。有效系统都在利用至少一种结构：热点、跨层/跨 token 相关性、多 batch 重叠，或让 CPU 直接完成冷 expert 计算。

### 6.4 显存配置的取舍

- 关闭或减少 redundant experts；
- 缩小 CUDA Graph capture bucket 和预分配 buffer 上界；
- KV cache 使用 FP8/分页管理，并控制最大并发与上下文；
- 热 expert 常驻 GPU、冷 expert 在 CPU/NVMe；
- 若只有少量 GPU，比较“量化后单域 TP”与“跨节点 EP”，前者可能延迟更好。

## 7. 场景三：提高吞吐

### 7.1 让专家 GEMM 变大

- continuous batching 合并更多活跃序列；
- prefill 使用较大 token budget，decode 使用 SLO 允许的最大 batch；
- 将同 expert 的 token 完整 coalesce 后再做 grouped GEMM；
- 避免过度 microbatch，太碎会再次回到 memory-bound。

### 7.2 扩大 DP+EP

- attention 用 DP 提高请求并行；
- experts 用 EP 汇聚整个 DP group 的 token，提高每 expert batch；
- 使用 HT All-to-All 和层次化拓扑；
- 用 EPLB + 少量热点复制减少最慢 rank；
- 用 DBO/多 batch 把 All-to-All 隐藏在计算后面。

### 7.3 Attention-FFN disaggregation

[MegaScale-Infer](https://arxiv.org/abs/2504.02263) 将 attention 与 FFN 放到可独立扩缩的设备池，并用 ping-pong microbatch pipeline 往返两类资源，论文报告最高 1.90x per-GPU throughput。它解决的是 attention 与 sparse FFN 的资源属性不同，但不是普适答案：额外的 token 传输会受 scale-out 带宽限制。

2026 年的 [AFD 边界分析](https://arxiv.org/abs/2602.09721) 指出，标准集群可能存在增加 FFN instances 也无法提高利用率的 dead zone；只有互联足够强、expert 较粗或稀疏度合适时，算子级解耦才更可能获益。因此部署前必须把每层 hidden-state 流量计入网络 roofline。

## 8. 推荐的基准矩阵

至少测试下列维度，而不是只跑一个离线吞吐点：

| 维度 | 建议取值 |
| --- | --- |
| 阶段 | prefill、decode 分开，再测混合流量 |
| token/batch | 1、8、32、128、512、4K、16K 或模型实际区间 |
| 路由 | 均匀、真实 trace、热点/长尾 synthetic |
| 并行 | TP、EP、TP×EP、DP×EP |
| 拓扑 | 单卡、单 NVLink domain、跨节点 RDMA |
| 精度 | BF16、FP8、FP4/INT4（若有原生 kernel） |
| 指标 | TTFT、TPOT P50/P99、tokens/s、goodput、HBM、网络带宽、最慢 rank |

## 9. 结论

LLM MoE 的最优策略由 `每次进入一层的 token 数` 主导：

- 小批 decode：减少固定开销、减少跨节点、使用 LL 通信和小批特化 kernel；
- 大批 prefill：提高 grouped GEMM 强度、使用 HT 通信和计算重叠；
- 显存不够：先量化和 EP，再考虑 activation-aware offload；
- 吞吐服务：DP+EP、EPLB、continuous batching 和双 batch overlap；
- attention/FFN 解耦：只在网络 roofline 和实际负载证明有收益时采用。
