---
type: Note
related_to: "[[Diffusion-Serving]]"
status: Active
url: https://arxiv.org/abs/2309.14509
---

# DeepSpeed-Ulysses: Sequence Parallelism via All-to-All

> Microsoft Research（Sam Ade Jacobs, Samyam Rajbhandari, Yuxiong He 等）
> arXiv:2309.14509
> 集成于 DeepSpeed v0.13+

---

## 一、背景：为什么需要序列并行？

Transformer 的计算可以沿四个维度并行化：

| 维度 | 并行方式 | 解决什么问题 |
|------|---------|-------------|
| Batch size | 数据并行（DP） | 加速训练 |
| Hidden dimension | 张量并行（TP） | 模型太大放不下 |
| Layer depth | 流水线并行（PP） | 模型太深 |
| **Sequence length** | **序列并行（SP）** | **序列太长** |

前三者已被广泛研究，但**序列维度的并行**在长序列场景下变得越来越重要：

- 长文档理解（数万~数十万 token）
- 高分辨率图像生成（DiT 的 patch 序列）
- 视频生成（时空 patch 序列可达百万级 token）
- 基因组序列（人类基因组有 64 亿个碱基）

---

## 二、Ulysses 的核心设计

### 2.1 整体流程

```
输入序列 N（长度为 N 的 token 序列）
        ↓
沿序列维度切分为 P 份，每份 N/P
        ↓
每个 GPU 持有 N/P 的 Q, K, V 投影
        ↓
┌───────────────────────────────────┐
│     All-to-All 通信（第一次）       │
│                                   │
│  从：每 GPU 持有 N/P 的全部 head    │
│  到：每 GPU 持有 N 的部分 head      │
│                                   │
│  即：序列维度 → head 维度的转置      │
└───────────────────────────────────┘
        ↓
每个 GPU 独立计算自己负责的 head 的注意力
  Output = softmax(QKᵀ/√d) · V
        ↓
┌───────────────────────────────────┐
│     All-to-All 通信（第二次）       │
│                                   │
│  从：每 GPU 持有 N 的部分 head      │
│  到：每 GPU 持有 N/P 的全部 head    │
│                                   │
│  即：head 维度 → 序列维度的转置      │
└───────────────────────────────────┘
        ↓
每个 GPU 持有 N/P 的输出，继续后续 MLP 等操作
```

### 2.2 图示

```
GPU 0: [Q₀,K₀,V₀]  ← 序列 [0, N/P)
GPU 1: [Q₁,K₁,V₁]  ← 序列 [N/P, 2N/P)
GPU 2: [Q₂,K₂,V₂]  ← 序列 [2N/P, 3N/P)
GPU 3: [Q₃,K₃,V₃]  ← 序列 [3N/P, N)

        ↓ All-to-All ↓

GPU 0: [Q_all, K_all, V_all] 但只有 head [0, H/P)
GPU 1: [Q_all, K_all, V_all] 但只有 head [H/P, 2H/P)
GPU 2: [Q_all, K_all, V_all] 但只有 head [2H/P, 3H/P)
GPU 3: [Q_all, K_all, V_all] 但只有 head [3H/P, H)

每个 GPU 计算自己负责的 head 的完整注意力（完整序列）

        ↓ All-to-All ↓

GPU 0: 输出 [0, N/P)，全部 head
GPU 1: 输出 [N/P, 2N/P)，全部 head
GPU 2: 输出 [2N/P, 3N/P)，全部 head
GPU 3: 输出 [3N/P, N)，全部 head
```

### 2.3 关键洞察

**Ulysses 的本质是把"序列并行"转化为"head 并行"：**

- All-to-All 之前：每个 GPU 持有**部分序列、全部 head**
- All-to-All 之后：每个 GPU 持有**全部序列、部分 head**
- 注意力计算：每个 GPU 独立计算自己负责的 head（完整序列上的注意力）
- All-to-All 转回来：恢复为部分序列、全部 head

这样做的好处是：**注意力计算本身不需要通信**，通信只发生在 All-to-All 的两次转置中。

---

## 三、通信复杂度分析

### 3.1 Ulysses 的通信量

对于隐藏维度 h、序列长度 N、并行度 P 的模型：

```
第一次 All-to-All：QKV 投影，聚合消息大小 = 3Nh
第二次 All-to-All：输出投影，聚合消息大小 = Nh

每次 All-to-All 每条链路的通信量 = 聚合大小 / P
总通信量每条链路 = 4Nh/P = O(N/P)
```

**关键：当 N 和 P 成比例增加时，通信量保持不变！**

### 3.2 与其他方法的对比

| 方法 | 每条链路通信量 | 复杂度 | 特点 |
|------|-------------|--------|------|
| **Ulysses** | 4Nh/P | **O(N/P)** | N 和 P 成比例时通信量恒定 |
| Megatron-SP | 4Nh | O(N) | 通信量随 N 线性增长，与 P 无关 |
| Ring Attention | O(N) | O(N) | 通信可重叠但总量大 |

**Ulysses 比 Megatron-SP 小 P 倍的通信量。**

### 3.3 为什么 All-to-All 更高效？

```
All-to-All：每个 GPU 发送 M/P 的数据给每个其他 GPU
            总发送量 = M/P × P = M
            但每条链路只传输 M/P（并行传输）

All-Gather：每个 GPU 发送 M/P 的数据，最终每个 GPU 收到完整的 M
            每条链路传输量 = M（所有数据都要到达每个 GPU）
```

在 NVSwitch/InfiniBand 等高带宽互联下，All-to-All 可以充分利用网络并行度。

---

## 四、与 Ring Attention 的对比

### 4.1 Ring Attention 回顾

```
GPU 0 → GPU 1 → GPU 2 → GPU 3 → GPU 0（环形传递 KV）

每个 GPU 持有部分 Q，逐步接收所有 GPU 的 KV
D 轮后每个 GPU 都计算了完整注意力
```

### 4.2 核心区别

| 维度 | Ulysses | Ring Attention |
|------|---------|---------------|
| **通信模式** | 两次 All-to-All | D 轮点对点环形传递 |
| **并行轴** | Head 维度 | Sequence 维度 |
| **通信次数** | 固定 2 次 | D 轮（D = GPU 数） |
| **通信-计算重叠** | 不重叠（通信在计算前后） | 重叠（每轮边传边算） |
| **每链路通信量** | O(N/P) | O(N) |
| **适用场景** | 中等规模（NVLink 高带宽） | 大规模（网络带宽有限） |
| **实现复杂度** | 简单（标准集合通信） | 复杂（需要在线 softmax） |

### 4.3 何时选择哪个？

```
Ulysses 适合：
  ✓ 节点内或高带宽互联（NVLink/NVSwitch）
  ✓ 中等规模 GPU 数（2-64）
  ✓ 实现简单，兼容 FlashAttention
  ✓ DiT 推理（通常在单节点 8 GPU 内）

Ring Attention 适合：
  ✓ 跨节点、低带宽互联
  ✓ 极大规模（数百 GPU）
  ✓ 通信可以被计算掩盖
  ✓ 训练超长序列（百万级 token）
```

**在 DiT 服务场景中，Ulysses 更常用**——因为 DiT 推理通常在单节点 8 GPU 内完成，NVLink 带宽足够高，两次 All-to-All 的延迟远低于 D 轮环形传递。

---

## 五、内存效率

### 5.1 激活内存

序列并行将激活内存从 O(N) 降低到 O(N/P)：
- 每个 GPU 只持有 N/P 的激活值
- 注意力计算时临时需要完整序列，但计算完即释放

### 5.2 与 ZeRO-3 集成

Ulysses 可以与 ZeRO-3 结合：
- ZeRO-3 在**数据并行 + 序列并行**的组合组内分区模型状态
- 梯度在数据和序列并行 rank 上共同归约
- 实现模型大小和序列长度的双重扩展

```
传统 DP：每个 GPU 持有完整模型，数据不同
DP + SP：每个 GPU 持有部分模型（ZeRO-3），部分序列（SP）
         → 模型和序列都可以扩展
```

---

## 六、通用性：Attention-Agnostic

Ulysses 的设计是**注意力无关的**——它支持任何注意力机制：

| 注意力类型 | 是否支持 |
|-----------|---------|
| Self-attention | ✓ |
| Cross-attention | ✓ |
| Causal attention | ✓ |
| Sparse attention | ✓ |
| FlashAttention v1/v2/v3 | ✓ |
| 任何自定义注意力 | ✓ |

**原因：** All-to-All 只负责数据重分布，注意力计算本身在每个 GPU 上独立完成。只要本地注意力实现正确，整体就是正确的。

---

## 七、实验结果

### 7.1 序列长度扩展性

在 1.2B 参数 GPT 上，序列长度从 8K 扩展到 **1M tokens**：
- 序列长度与 GPU 数成比例增加
- 计算吞吐量保持稳定

### 7.2 与 Megatron-SP 的对比

| 模型 | GPU 数 | Ulysses 吞吐量 | Megatron-SP 吞吐量 | 加速比 |
|------|--------|---------------|-------------------|--------|
| 7B (dense) | 32 | 更高 | 基准 | ~2× |
| 30B (dense) | 64 | 更高 | 基准 | ~2× |
| 7B (sparse) | 32 | 更高 | 基准 | ~2× |
| 30B (sparse) | 64 | 更高 | 基准 | ~2× |

**Ulysses 支持比 Megatron-SP 长 4× 的序列长度。**

### 7.3 并行扩展性

| 序列长度 | GPU 数 | 迭代时间 (ms) | TFLOPs/GPU |
|---------|--------|-------------|-----------|
| 131072 | 64 | 32,432 | 165.5 |
| 131072 | 128 | 17,053 | 157.4 |
| 131072 | 256 | 9,887 | 136.1 |

| 序列长度 | GPU 数 | 迭代时间 (ms) | TFLOPs/GPU |
|---------|--------|-------------|-----------|
| 65536 | 64 | 9,677 | 161.4 |
| 131072 | 128 | 17,053 | 157.4 |
| 262144 | 256 | 33,487 | 147.4 |

达到硬件峰值的 **54%** 以上。

### 7.4 收敛性

Ulysses 是纯系统优化，**不影响模型收敛质量**：
- 与 Megatron-SP 在相同配置下收敛曲线一致
- 不同 ZeRO 阶段（0/1/2/3）均不影响收敛

---

## 八、在 DiT 服务中的应用

### 8.1 为什么 DiT 服务需要序列并行？

DiT 的输入是图像/视频的 patch 序列：

| 图像大小 | Token 数 | 单 GPU 能否处理 |
|---------|---------|---------------|
| 256×256 | 256 | ✓ |
| 512×512 | 1,024 | ✓ |
| 1024×1024 | 4,096 | ✓（勉强） |
| 2048×2048 | 16,384 | ✗（需要 SP） |
| 4096×4096 | 65,536 | ✗（需要大 SP） |
| 视频（长） | 100K+ | ✗（需要大 SP） |

### 8.2 DiT 服务中 Ulysses vs Ring Attention

在 DiT 服务系统中：

| 系统 | 使用的 SP 方式 |
|------|--------------|
| **xDiT** | 支持 Ulysses 和 Ring Attention |
| **DiT-Serve** | 基于 Ring Attention，提出 Brick Attention |
| **TetriServe** | 使用 Ulysses（Ulysses attention） |
| **TridentServe** | 使用 Ring Attention |

**实践中的选择：**
- 单节点 8 GPU（NVLink）→ Ulysses 通常更快
- 跨节点 → Ring Attention 可能更合适
- TetriServe 论文明确指出："Ulysses attention is often preferred on systems with high-bandwidth interconnects like NVLink, as its use of collective primitives can be more efficient"

---

## 九、实现要点

### 9.1 通信原语

```python
# All-to-All 通信（PyTorch 分布式）
# 前：每个 GPU 持有 [N/P, H, d]
# 后：每个 GPU 持有 [N, H/P, d]

# 第一次 All-to-All：序列维度 → head 维度
dist.all_to_all_single(output, input)

# 本地注意力计算
attn_output = flash_attention(q, k, v)

# 第二次 All-to-All：head 维度 → 序列维度
dist.all_to_all_single(final_output, attn_output)
```

### 9.2 与 FlashAttention 的兼容性

Ulysses 天然兼容 FlashAttention：
- All-to-All 后，每个 GPU 持有完整序列但部分 head
- 本地注意力可以用 FlashAttention 高效计算
- 不需要修改 FlashAttention 的实现

### 9.3 进程组管理

```
需要创建两类进程组：
1. 序列并行组（SP group）：负责 All-to-All 通信
2. 数据并行组（DP group）：负责梯度同步

SP group 和 DP group 正交：
  4 GPU，SP=2，DP=2：
  SP group: {GPU 0, GPU 1}, {GPU 2, GPU 3}
  DP group: {GPU 0, GPU 2}, {GPU 1, GPU 3}
```

---

## 十、总结

### Ulysses 的三个关键

| 关键点 | 解释 |
|--------|------|
| **All-to-All 转置** | 将"部分序列、全部 head"转置为"全部序列、部分 head"，使注意力计算无需通信 |
| **通信量 O(N/P)** | 当 N 和 P 成比例增加时通信量恒定，比 Megatron-SP 小 P 倍 |
| **Attention-Agnostic** | 支持任何注意力机制，兼容 FlashAttention |

### Ulysses vs Ring Attention 一句话总结

> **Ulysses 用两次 All-to-All 转置把序列并行转化为 head 并行，通信量小但不重叠；Ring Attention 用 D 轮环形传递逐步累积 KV，通信量大但与计算重叠。在高带宽互联下 Ulysses 更快，在低带宽或大规模下 Ring Attention 更可扩展。**
