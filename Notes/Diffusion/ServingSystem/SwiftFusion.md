# SwiftFusion: Scalable Sequence Parallelism for Distributed Inference of Diffusion Transformers

**arXiv:** 2601.20273
**Date:** 2026-05-22
**Authors:** Yang, Jiacheng; Wu, Jun; Ding, Yaoyao; et al.
**Affiliation:** University of Toronto, Microsoft

---

## 1. 论文概述

SwiftFusion（论文中称 StreamFusion）是一个拓扑感知的高效 DiT 推理引擎，通过优化序列并行（Sequence Parallelism）来解决高分辨率/长视频生成的扩展性问题。

**核心贡献**：
- 提出拓扑感知序列并行，考虑机内/机间带宽差异
- 设计 **Torus Attention**，实现机间 All-to-All 与计算重叠
- 采用单边通信（one-sided communication）减少 GPU 同步开销
- 平均 **1.35× 加速**（最高 1.77×）

---

## 2. 背景与问题

### 2.1 序列并行的必要性

随着分辨率/视频长度增加：
- **激活值过大**：单 GPU 显存无法容纳
- **延迟过高**：串行推理无法满足实时需求

### 2.2 现有 SP 方法的局限

| 方法 | 问题 |
|------|------|
| **Ulysses Attention** | All-to-All 通信成为瓶颈 |
| **Ring Attention** | 同步开销大，无法重叠计算 |

具体问题：

1. **通信模式不匹配**：
   - 现代 GPU 集群：机内 NVLink（600GB/s） vs 机间 InfiniBand（100Gb/s）
   - 现有方法未考虑这种异构性

2. **All-to-All 瓶颈**：
   - 机间 All-to-All 通信延迟高
   - 无法与计算重叠

3. **双边通信开销**：
   - 使用 MPI 等双边通信库
   - 需要 GPU 间同步，增加延迟

---

## 3. 核心技术

### 3.1 拓扑感知序列并行

**核心思想**：将序列分片优先分配到机内 GPU，减少跨机通信

```
集群拓扑:
Machine 1: [GPU0, GPU1, GPU2, GPU3] ← NVLink
Machine 2: [GPU4, GPU5, GPU6, GPU7] ← NVLink
           ↑_________________________↑
                 InfiniBand

序列分片策略:
- 优先在 Machine 1 内分片: [seq0, seq1, seq2, seq3]
- 需要扩展时再跨机: [seq0-3 in M1] + [seq4-7 in M2]
```

### 3.2 Torus Attention

**问题**：机间 All-to-All 通信延迟高

**解决方案**：将 All-to-All 与计算重叠

```
传统 Ring Attention:
GPU0: [All-to-All] → [Compute] → [All-to-All] → [Compute]
GPU1: [All-to-All] → [Compute] → [All-to-All] → [Compute]
       ↑ 跨机通信阻塞计算

Torus Attention:
GPU0: [Compute chunk0] → [All-to-All chunk1] → [Compute chunk1] → ...
GPU1: [All-to-All chunk0] → [Compute chunk0] → [All-to-All chunk1] → ...
       ↑ 通信与计算流水线化
```

**关键**：
- 将序列分成多个 chunk
- 通信 chunk $i$ 与计算 chunk $i-1$ 重叠
- 使用环形拓扑，每个 GPU 同时发送和接收

### 3.3 单边通信（One-sided Communication）

**问题**：双边通信（如 MPI_Send/Recv）需要发送方和接收方同步

**解决方案**：使用 RDMA 单边通信

```python
# 双边通信（需要同步）
GPU0: MPI_Send(data, dest=GPU1)  # 等待 GPU1 调用 MPI_Recv
GPU1: MPI_Recv(data, src=GPU0)   # 阻塞等待

# 单边通信（无需同步）
GPU0: RDMA_Write(remote_addr, data)  # 直接写入 GPU1 显存
GPU1: 继续计算，无需等待              # 无需显式接收
```

**优势**：
- 消除 GPU 间同步开销
- 允许更细粒度的重叠
- 减少通信延迟

---

## 4. 系统实现

### 4.1 架构

```
┌─────────────────────────────────────────┐
│           SwiftFusion Runtime           │
│  ┌───────────────────────────────────┐  │
│  │      Topology-aware Partitioner   │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │      Torus Attention Engine       │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │      RDMA Communication Layer     │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### 4.2 关键优化

1. **Chunk 流水线**：
   ```python
   for chunk_idx in range(num_chunks):
       # 通信 chunk_idx（异步）
       comm_handle = start_communication(chunk_idx)
       # 计算 chunk_idx - 1（与通信重叠）
       if chunk_idx > 0:
           compute_attention(chunk_idx - 1)
       # 等待通信完成
       wait(comm_handle)
   ```

2. **显存管理**：
   - 预分配通信缓冲区
   - 使用 CUDA stream 管理并发

---

## 5. 实验结果

### 5.1 实验设置

- **模型**：Stable Video Diffusion
- **硬件**：8× A100（2 机，机间 InfiniBand）
- **序列长度**：4K - 32K tokens

### 5.2 性能对比

| 方法 | 4K tokens | 16K tokens | 32K tokens |
|------|-----------|------------|------------|
| Ulysses | 1.0× | 1.0× | 1.0× |
| Ring Attention | 0.95× | 1.1× | 1.2× |
| **SwiftFusion** | **1.2×** | **1.4×** | **1.77×** |

### 5.3 通信分析

| 方法 | 通信时间占比 | 同步开销 |
|------|-------------|----------|
| Ulysses | 45% | 高 |
| Ring Attention | 35% | 中 |
| **SwiftFusion** | **20%** | **低** |

---

## 6. 优势与局限

### 优势

1. **拓扑感知**：充分利用机内高带宽
2. **重叠计算**：通信不阻塞计算
3. **低同步开销**：单边通信减少等待

### 局限

1. **硬件依赖**：需要 RDMA 支持
2. **实现复杂**：Torus Attention 调度复杂
3. **适用场景**：主要针对跨机场景

---

## 7. 与其他 SP 方法的对比

| 方法 | 通信模式 | 同步需求 | 适用场景 |
|------|----------|----------|----------|
| Ulysses | All-to-All | 高 | 机内 |
| Ring Attention | P2P Ring | 中 | 跨机 |
| **SwiftFusion** | **Torus + RDMA** | **低** | **跨机** |

---

## 8. 关键术语

| 术语 | 解释 |
|------|------|
| Sequence Parallelism (SP) | 序列维度并行 |
| Torus Attention | 环形拓扑 attention，通信与计算重叠 |
| One-sided Communication | RDMA 单边通信，无需接收方同步 |
| Topology-aware | 拓扑感知，考虑机内/机间带宽差异 |
| All-to-All | 全对全通信模式 |
