---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Dual Batch Overlap (DBO)

## 核心动机

DBO 是 vLLM 中的一个性能优化系统，目标是**将 MoE 层中的稀疏 all-to-all 通信与周围的计算重叠执行**，从而隐藏通信延迟。目前仅针对 **DP+EP**（数据并行 + 专家并行）部署场景。

## 工作原理

DBO 的核心思想是**微批次（microbatch/ubatch）+ 线程乒乓（ping-pong）**：

1. **拆分批次**：`GPUModelRunner` 将一个大 batch 拆成两个微批次（microbatch 0 和 microbatch 1）
2. **创建两个线程**：为每个微批次创建一个 CPU worker 线程（UBatch thread）
3. **乒乓调度**：两个线程交替执行，当一个线程在做**计算**时，另一个线程在等待**通信**（all-to-all），反之亦然

这种设计使得通信延迟被计算完全"隐藏"掉。

## 调度时序

```text
Comp: |-A0₀-A1₀-||-MLP₁-||-S₁-MLP₀-||-S₀-A0₁-A1₁-|
Comm: |----D₁---||--D₀--||----C₁---||-----C₀-----|
```

其中：
- **A0** = MLA 的 qkv 投影
- **A1** = Core attention + output projection + MoE gate
- **D** = Dispatch（专家分发，all-to-all 通信）
- **C** = Combine（专家结果合并，all-to-all 通信）
- **S** = Shared expert（共享专家计算）
- **MLP** = 专家 MLP 计算
- 下标 ₀/₁ 表示属于哪个微批次

执行顺序：

```text
D₁ send → A0₀ → A1₀ → D₁ recv → D₀ send → MLP₁ → D₀ recv →
C₁ send → S₁ → MLP₀ → C₁ recv → C₀ send → S₀ → A0₁ → A1₁ → C₀ recv
```

关键观察：**通信和计算是交错的**。例如 D₁ 的 send/recv 包裹了 A0₀ 和 A1₀ 的计算；D₀ 的 send/recv 包裹了 MLP₁ 的计算。

## 三大组件

| 组件 | 职责 |
|------|------|
| **GPUModelRunner** | 负责批次拆分、跨 DP rank 协调、决定是否启用微批次 |
| **UBatchWrapper** | 管理线程生命周期、CUDA graph 捕获与回放，对 Model Runner 透明 |
| **UBatchContext** | 包装 `ForwardContext`，通过 `dbo_yield` 实现两个线程的同步 |

## 关键同步机制

`UBatchContext` 实现了乒乓调度的核心：

- **`dbo_yield`**：当前线程休眠，唤醒另一个线程
- **`dbo_register_recv_hook`**：注册一个回调，用于等待 all-to-all 通信完成
- **`dbo_maybe_run_recv_hook`**：执行对方线程注册的回调

所有这些 yield 点都位于 `FusedMoEModularKernel.forward` 方法中。

## 批次拆分的协调逻辑

微批次的启用需要**所有 DP rank 统一**：

1. 跨所有 rank 协调，确定是否可以微批次化
2. 如果任何 rank 不可行，全部 rank 都不微批次
3. 如果可以，将所有 rank 的 token 数 pad 到最大值
4. 如果 pad 后某个 rank 的第二个微批次为空，则中止微批次化

## 启用方式

```bash
vllm serve deepseek-ai/DeepSeek-V2-Lite \
  --trust-remote-code \
  --data-parallel-size 2 \
  --enable-expert-parallel \
  --enable-dbo \
  --all2all-backend deepep_low_latency
```

前提条件：
- `--data-parallel-size N`（N >= 2）
- `--enable-expert-parallel`
- 安装 DeepEP
- 至少 2 块 GPU

可调参数：
- `--dbo-decode-token-threshold`：纯 decode batch 启用 DBO 的最小 token 数
- `--dbo-prefill-token-threshold`：含 prefill 的 batch 启用 DBO 的最小 token 数

## 一句话总结

DBO 通过将 batch 拆成两个微批次、用两个线程交替执行，使得 MoE 层的 all-to-all 通信被另一个微批次的计算所掩盖，从而提升 DP+EP 场景下的吞吐量。
