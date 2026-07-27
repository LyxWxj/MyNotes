# DIT Offloading 技术方案

> 场景：PCIe 卡型 + 长序列 DIT 推理（Cosmos3, LingBot-Video）
> 目标：消费卡/华为卡上支持长序列 DIT 推理，做好计算-通信 overlap

---

## 一、问题分析

### 1.1 显存瓶颈

DIT 推理的显存占用：

$$\text{Mem}_{\text{total}} = \text{Mem}_{\text{weights}} + \text{Mem}_{\text{activation}} + \text{Mem}_{\text{kv\_cache}} + \text{Mem}_{\text{buffer}}$$

| 组件 | Cosmos3 (16B) | LingBot-Video (13B MoE) |
|------|--------------|------------------------|
| 权重 (BF16) | ~32 GB | ~26 GB（活跃 2.8GB） |
| 激活 (1M tokens) | ~8-16 GB | ~8-16 GB |
| KV Cache | ~2-4 GB | ~2-4 GB |
| SP/TP Buffer | ~4-8 GB | ~4-8 GB |
| **总计** | **~46-60 GB** | **~40-54 GB** |

单卡 24-32 GB 显存无法容纳。

### 1.2 通信瓶颈

PCIe 架构：

```
CPU
 │
 │ PCIe 5.0 x16 (单向 ~57 GB/s)
 ▼
┌───────────────────────┐
│      PCIe Switch       │
└─┬───────┬───────┬─────┘
  │       │       │
GPU0   GPU1   GPU2   GPU3
(H2D)  (P2P)  (P2P)  (P2P)
```

关键约束：
- PCIe 全双工：可同时 h2d + d2h
- P2P 带宽：~57 GB/s（空闲）
- NUMA 感知：跨 NUMA 节点带宽下降

### 1.3 DIT 推理的计算特征

```
去噪循环：
for t in timesteps:           # 外循环：35-50 步
    for chunk in chunks:      # 内循环：长序列分块
        for layer in layers:  # 层循环：28-64 层
            x = layer(x, t, context)
```

**关键观察**：
1. **层间独立性**：每层计算完即可卸载权重
2. **步骤间独立性**：去噪步骤间无状态依赖（除 KV Cache）
3. **chunk 间依赖**：GEN 路径双向注意力需要全序列 K/V
4. **UND 路径**（Cosmos3）：只运行一次，K/V 缓存后复用

---

## 二、方案设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                        GPU                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 当前层权重     │  │ 下一层权重     │  │ 激活缓存      │       │
│  │ (Layer_i)     │  │ (Layer_i+1)  │  │ (固定大小)    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ UND KV Cache  │  │ 路由预测器     │                         │
│  │ (跨步骤复用)   │  │ (MoE 专用)    │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
        ↕ h2d (预取)              ↕ d2h (卸载)
┌─────────────────────────────────────────────────────────────┐
│                        CPU                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ 其他层权重     │  │ 卸载的激活     │  │ 非活跃专家     │       │
│  │ (Pinned Mem)  │  │ (Pinned Mem)  │  │ (Pinned Mem)  │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└─────────────────────────────────────────────────────────────┘
        ↕ NVMe (可选)
┌─────────────────────────────────────────────────────────────┐
│                        NVMe                                  │
│  ┌──────────────┐                                           │
│  │ 溢出的权重/激活 │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 权重 Offloading 方案

#### 2.2.1 层级滑动窗口（借鉴 SlideFormer）

**核心思想**：GPU 只保留当前计算层和下一层的权重。

```python
class LayerSlidingOffloader:
    """层级滑动权重 offloader"""
    
    def __init__(self, layers, gpu_cache_slots=2):
        self.layers = layers
        self.gpu_cache_slots = gpu_cache_slots
        self.gpu_cache = {}  # layer_id -> weight_tensor
        
        # 预分配 Pinned Memory
        self.cpu_buffer = allocate_pinned_memory(total_weight_size)
    
    def prefetch_next(self, current_layer_id):
        """异步预取下一层权重"""
        next_id = current_layer_id + 1
        if next_id < len(self.layers):
            with torch.cuda.stream(self.h2d_stream):
                self.gpu_cache[next_id] = self.layers[next_id].to(
                    device='cuda', non_blocking=True
                )
    
    def offload_current(self, current_layer_id):
        """异步卸载当前层权重"""
        if current_layer_id in self.gpu_cache:
            with torch.cuda.stream(self.d2h_stream):
                self.gpu_cache[current_layer_id].to(
                    device='cpu', non_blocking=True
                )
            del self.gpu_cache[current_layer_id]
```

**Overlap 条件**（借鉴 SlideFormer）：

$$\eta = \frac{T_{\text{compute}}}{T_{\text{h2d\_prefetch}}} \geq 1$$

当 $\eta \geq 1$ 时，权重预取完全被计算掩盖。

**实测数据参考**（SlideFormer Table 1）：

| 模型 | GPU | Batch | T_compute (ms) | T_h2d (ms) | η |
|------|-----|-------|---------------|------------|---|
| Qwen2.5-14B | RTX 4090 | 32 | 340 | 25 | 13.6 |
| Qwen2.5-14B | RTX 4090 | 64 | 660 | 25 | 26.4 |

DIT 推理中，每层计算时间通常远大于权重传输时间，overlap 条件容易满足。

#### 2.2.2 MoE 专家 Offloading（LingBot-Video 专用）

LingBot-Video 有 128 个专家，总权重巨大，但每 token 只激活 8 个。

```python
class MoEExpertOffloader:
    """MoE 专家权重 offloader"""
    
    def __init__(self, num_experts=128, top_k=8):
        self.num_experts = num_experts
        self.top_k = top_k
        self.gpu_experts = set()  # 当前驻留 GPU 的专家
        self.cpu_experts = {}     # 卸载到 CPU 的专家
    
    def predict_active_experts(self, router_logits, chunk_id):
        """预测下一 chunk 的活跃专家"""
        # 使用当前 chunk 的路由结果预测下一 chunk
        top_indices = torch.topk(router_logits, self.top_k, dim=-1)[1]
        return top_indices.unique()
    
    def prefetch_experts(self, expert_ids):
        """异步预取专家权重"""
        for eid in expert_ids:
            if eid not in self.gpu_experts:
                with torch.cuda.stream(self.h2d_stream):
                    self.cpu_experts[eid].to('cuda', non_blocking=True)
                self.gpu_experts.add(eid)
    
    def offload_inactive(self, active_experts):
        """卸载非活跃专家"""
        inactive = self.gpu_experts - set(active_experts)
        for eid in inactive:
            with torch.cuda.stream(self.d2h_stream):
                self.gpu_experts[eid].to('cpu', non_blocking=True)
            self.gpu_experts.remove(eid)
            self.cpu_experts[eid] = self.gpu_experts.pop(eid)
```

### 2.3 激活 Offloading 方案

#### 2.3.1 Chunk 分块策略（借鉴 ChunkFlow）

**核心思想**：按 Chunk 组织计算，峰值内存由 ChunkSize 决定。

$$\text{Peak Mem} \approx \text{Mem}_{\text{weights}} + \text{ChunkSize} \times \text{hidden\_dim} \times \text{num\_layers}$$

**DIT 推理的 Chunk 策略**：

| 策略 | ChunkSize | 适用场景 | 优劣 |
|------|-----------|---------|------|
| 按时间帧 | K 帧 | 视频生成 | 帧间独立，易分块 |
| 按空间块 | H'×W' | 高分辨率 | 需处理边界依赖 |
| 混合分块 | K 帧 × H'×W' | 超长序列 | 最灵活，管理复杂 |

```python
class ActivationChunkManager:
    """激活分块管理器"""
    
    def __init__(self, chunk_size, hidden_dim, num_layers):
        self.chunk_size = chunk_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 预分配激活缓存
        self.activation_cache = torch.zeros(
            chunk_size, hidden_dim, num_layers,
            device='cuda', dtype=torch.bfloat16
        )
    
    def forward_with_chunking(self, x, layers):
        """分块前向传播"""
        chunks = x.split(self.chunk_size, dim=1)
        outputs = []
        
        for chunk_id, chunk in enumerate(chunks):
            # 卸载上一个 chunk 的激活
            if chunk_id > 0:
                self.offload_activation(chunk_id - 1)
            
            # 预取下一个 chunk 的输入
            if chunk_id < len(chunks) - 1:
                self.prefetch_input(chunk_id + 1)
            
            # 计算当前 chunk
            for layer_id, layer in enumerate(layers):
                chunk = layer(chunk)
                self.activation_cache[:, :, layer_id] = chunk
            
            outputs.append(chunk)
        
        return torch.cat(outputs, dim=1)
```

#### 2.3.2 UND vs GEN 路径差异

**Cosmos3 特殊考虑**：

```
UND 路径（文本处理）：
  - 只运行一次
  - K/V 缓存保留在 GPU
  - 激活很小（文本序列长度有限）
  - 策略：权重在首次运行后完全卸载

GEN 路径（视觉生成）：
  - 每个去噪步骤都运行
  - 激活巨大（1M+ tokens）
  - 策略：分 chunk 处理，激活按需卸载
```

### 2.4 PCIe 全双工利用

#### 2.4.1 三流并行

```python
# 三个独立 CUDA Stream
compute_stream = torch.cuda.Stream()   # 计算
h2d_stream = torch.cuda.Stream()       # CPU → GPU（预取）
d2h_stream = torch.cuda.Stream()       # GPU → CPU（卸载）

def denoise_step_with_overlap(input, timesteps):
    """带 overlap 的去噪步骤"""
    for step_id, t in enumerate(timesteps):
        # 异步卸载上一步的激活
        if step_id > 0:
            with torch.cuda.stream(d2h_stream):
                offload_activations(prev_activations)
        
        # 异步预取下一步的权重
        if step_id < len(timesteps) - 1:
            with torch.cuda.stream(h2d_stream):
                prefetch_weights(next_layer_weights)
        
        # 计算当前步
        with torch.cuda.stream(compute_stream):
            output = dit_forward(input, t, current_weights)
        
        # 同步
        torch.cuda.current_stream().wait_stream(h2d_stream)
        torch.cuda.current_stream().wait_stream(d2h_stream)
        
        input = output
        prev_activations = get_current_activations()
    
    return output
```

#### 2.4.2 PCIe 带宽打满策略

**观察**：PCIe 5.0 x16 理论带宽 64 GB/s，实测 ~57 GB/s。

**打满策略**：

1. **Pinned Memory**：所有 CPU-GPU 传输使用 Pinned Memory
2. **大块传输**：避免小块传输，合并为大块
3. **全双工利用**：同时 h2d + d2h
4. **NUMA 感知**：权重放在对应 GPU 的 NUMA 节点

```python
class PCIeOptimizedTransfer:
    """PCIe 优化传输器"""
    
    def __init__(self, gpu_id):
        self.gpu_id = gpu_id
        self.pinned_buffer = torch.zeros(
            max_weight_size, dtype=torch.bfloat16
        ).pin_memory()
    
    def async_transfer(self, src, dst, stream):
        """异步传输"""
        # 确保源是 Pinned Memory
        if src.device.type == 'cpu' and not src.is_pinned():
            src = src.pin_memory()
        
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
    
    def batch_transfer(self, tensors, stream):
        """批量传输，打满带宽"""
        # 合并小块为大块
        merged = torch.cat([t.flatten() for t in tensors])
        self.async_transfer(merged, self.gpu_buffer, stream)
```

#### 2.4.3 多卡权重分发

**方案对比**：

| 方案 | 通信量 | 同步点 | 复杂度 |
|------|--------|--------|--------|
| Rank0 H2D + P2P | N × W | N-1 次 P2P | 低 |
| FSDP Scatter | N × W | N-1 次 Scatter | 中 |
| 同时 P2P | N × W | 1 次同步 | 高（DMA 竞争） |

**推荐方案**：Rank0 H2D + P2P

```python
def distribute_weights_to_gpus(weights, num_gpus):
    """Rank0 统一 H2D，然后 P2P 分发"""
    if rank == 0:
        # Rank0 从 CPU 加载到 GPU
        gpu_weights = weights.to('cuda:0', non_blocking=True)
    
    # 同步
    dist.barrier()
    
    # P2P 分发
    if rank == 0:
        for gpu_id in range(1, num_gpus):
            with torch.cuda.stream(p2p_stream):
                gpu_weights.to(f'cuda:{gpu_id}', non_blocking=True)
```

---

## 三、实现计划

### 3.1 Phase 1：基础 Offloading 框架

**目标**：单卡权重 offloading，支持 Cosmos3 基本推理。

**任务**：
1. 实现 `LayerSlidingOffloader`，支持层级权重滑动
2. 实现 Pinned Memory 预分配
3. 实现三流并行（compute/h2d/d2h）
4. 集成到 Cosmos3 推理管道

**验证指标**：
- 单卡能否运行 16B Cosmos3
- overlap 效率（η ≥ 1）
- PCIe 带宽利用率

### 3.2 Phase 2：激活 Offloading + Chunk 分块

**目标**：支持长序列（1M+ tokens）推理。

**任务**：
1. 实现 `ActivationChunkManager`，支持按帧/空间分块
2. 实现激活异步卸载/预取
3. 集成 ChunkFlow 的状态感知调度
4. 优化 ChunkSize 选择策略

**验证指标**：
- 最大支持序列长度
- 峰值显存占用
- 推理延迟

### 3.3 Phase 3：MoE 专家 Offloading

**目标**：支持 LingBot-Video 128 专家 MoE 推理。

**任务**：
1. 实现 `MoEExpertOffloader`，支持专家级卸载
2. 实现路由预测，提前预取活跃专家
3. 实现 grouped_mm 与 offloading 的集成
4. 优化专家缓存策略

**验证指标**：
- 专家命中率（预测准确率）
- 专家预取延迟
- 整体推理吞吐

### 3.4 Phase 4：多卡优化

**目标**：多卡 PCIe 卡型的高效推理。

**任务**：
1. 实现 Rank0 H2D + P2P 权重分发
2. 实现 SP + Offloading 的集成
3. NUMA 感知的内存分配
4. 全双工 PCIe 带宽优化

**验证指标**：
- 多卡扩展效率
- P2P 带宽利用率
- 整体推理延迟

---

## 四、风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| PCIe 带宽不足 | 预取延迟大，overlap 失败 | 减小 ChunkSize，增加计算密度 |
| 激活内存溢出 | OOM | 更激进的 chunk 分块，或 NVMe 溢出 |
| MoE 预测不准 | 专家预取失败，回退到 CPU 计算 | 增大预取窗口，或保留更多专家在 GPU |
| NUMA 跨节点带宽下降 | P2P 效率降低 | 每个 NUMA 节点独立加载权重 |
| CUDA Stream 同步开销 | 隐藏的延迟 | 减少同步点，使用 event-based 同步 |

---

## 五、参考论文

1. **SlideFormer** (arxiv 2503.02356v3)：Layer-Sliding Architecture，异步引擎，预分配 GPU 缓存
2. **ChunkFlow** (arxiv 2603.16428v2)：Chunk 构建，状态感知调度，Pipeline 优化
3. **ZeRO-Offload/Infinity**：CPU/NVMe offloading 基础框架
4. **Cosmos3**：双流 MoT 架构，UND/GEN 路径分离
5. **LingBot-Video**：单流 MoE 架构，128 专家 top-8 路由
