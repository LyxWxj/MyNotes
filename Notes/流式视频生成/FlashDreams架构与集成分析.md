---
type: Note
related_to: "[[FlashDreams]]"
status: Active
---

# FlashDreams 架构与集成分析

> FlashDreams 是 NVIDIA 开发的高性能流式视频生成推理框架。本文档整理了项目架构、核心组件和各模型集成的实现细节。

---

## 🏗️ 项目架构概览

> [!info] 核心设计
> FlashDreams 采用分层架构：**infra（基础设施层）** → **recipes（模型配方层）** → **integrations（模型集成层）**

### 目录结构

```
flashdreams/
├── flashdreams/          # 核心包
│   ├── configs/          # 配置系统
│   ├── core/             # 核心功能（attention, checkpoint, distributed）
│   ├── infra/            # 基础设施层（encoder, diffusion, decoder, pipeline, runner）
│   ├── plugins/          # 插件系统
│   ├── recipes/          # 模型配方（template, cosmos, wan, taehv）
│   ├── scripts/          # CLI 工具
│   └── serving/          # WebRTC 服务
├── integrations/         # 模型集成
│   ├── causal_forcing/   # 因果强迫
│   ├── cosmos_predict2/  # Cosmos 世界模型
│   ├── flashvsr/         # 视频超分辨率
│   ├── hy_worldplay/     # 交互式世界模型
│   ├── lingbot/          # 相机可控 I2V
│   ├── self_forcing/     # 自强迫
│   └── wan21/            # Wan 2.1
└── tests/                # 测试
```

### 核心抽象

> [!important] 三大核心组件
> 1. **StreamInferencePipeline** - 流式推理管道
> 2. **Runner** - CLI 驱动的推理执行器
> 3. **Encoder/Decoder** - 流式编解码器

```python
# 流式推理管道
StreamInferencePipeline
  ├── StreamingEncoder    # 输入编码
  ├── DiffusionModel      # 扩散模型（Transformer + Scheduler）
  └── StreamingDecoder    # 输出解码

# Runner 驱动
Runner
  ├── RunnerConfig        # 配置
  ├── Pipeline            # 管道实例
  └── run()               # 生成并持久化输出
```

---

## 🔧 核心组件详解

### SDPA（Scaled Dot-Product Attention）

> [!note] SDPA 是什么？
> **S**caled **D**ot-**P**roduct **A**ttention（缩放点积注意力），是 Transformer 的核心操作。

**公式**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**FlashDreams 中的两种后端**：

```python
# cp.py 中的别名定义
_sdpa_cudnn = torch.ops.aten._scaled_dot_product_cudnn_attention  # CuDNN 后端
_sdpa_flash = torch.ops.aten._scaled_dot_product_flash_attention   # Flash Attention 后端
```

| 后端 | 实现 | 优势 | 适用场景 |
|------|------|------|----------|
| **CuDNN** | NVIDIA CuDNN 库 | 针对 NVIDIA GPU 优化 | 标准 GPU 推理 |
| **Flash Attention** | Tri Dao | 内存高效 O(N) | 超长序列 |

### KV Cache 管理

> [!important] 流式推理的关键
> KV Cache 用于自回归生成的上下文复用，避免重复计算。

**滑动窗口机制**：

```
窗口大小 = sink_size + window_size

Token 序列: [S1, S2, ..., S_sink, W1, W2, ..., W_window, NEW]
             ↑_______________↑   ↑______________________↑
                   sink                window

当新 token 进来时，最旧的 window token 被移除
```

**代码实现**（`core/attention/kvcache.py`）：

```python
class BlockKVCache:
    """滑动窗口 KV Cache"""
    
    def __init__(self, k_shape, v_shape, seq_dim, chunk_size, window_size, sink_size):
        self.chunk_size = chunk_size      # 每次追加的 token 数
        self.window_size = window_size    # 窗口大小
        self.sink_size = sink_size        # sink token 数
    
    def update(self, k, v):
        """追加新的 K/V"""
        # 移除最旧的 token，追加新的
    
    def cached_k(self):
        """获取缓存的 K"""
    
    def cached_v(self):
        """获取缓存的 V"
```

### Context Parallel

> [!tip] 多 GPU 并行
> Context Parallel 用于多 GPU 间的 token 级分片，实现分布式注意力计算。

**两种方法**：

| 方法 | 说明 | 适用场景 |
|------|------|----------|
| **Ring** | 环形通信 | 长序列 |
| **Ulysses** | 全局通信 | 短序列 |

---

## 📊 集成对比分析

> [!summary] 四种集成的架构对比
> 不同的集成针对不同的应用场景，采用不同的架构设计。

### 生成模式对比

| 集成 | 生成模式 | 循环次数 | 每次帧数 | 流式输出 |
|------|----------|----------|----------|----------|
| **Cosmos** | 单次生成 | 1 | 全部 | ❌ |
| **Causal-Forcing** | 流式架构 | 60 | 2 | ❌ |
| **FlashVSR** | 流式处理 | 可变 | 8/16 | ✅ gRPC |
| **HY-WorldPlay** | 流式架构 | 4 | 4 | ❌ |

### Cosmos vs Causal-Forcing

> [!question] 为什么 Cosmos 只生成一次，而 Causal-Forcing 循环 60 次？

**Cosmos：双向注意力（Bidirectional）**

```python
# 一次性生成完整视频
generated = self.pipeline.generate(autoregressive_index=0, cache=cache)
stats = self.pipeline.finalize(autoregressive_index=0, cache=cache)
```

- 每个 token 可以 attend 到所有其他 token
- 需要完整的时间窗口
- 适合固定长度的视频生成

**Causal-Forcing：因果注意力（Causal）**

```python
# 循环生成 60 个 chunk
for i in range(config.total_blocks):
    video_chunk = self.pipeline.generate(autoregressive_index=i, cache=cache)
    stats = self.pipeline.finalize(autoregressive_index=i, cache=cache)
    chunks.append(video_chunk.cpu())
```

- 每个 chunk 只能 attend 到之前的 chunks
- 使用滑动窗口 KV Cache
- 适合无限长度的流式生成

> [!warning] 注意
> Causal-Forcing 虽然有流式架构，但 Runner 没有实现真正的流式输出（最后才一次性保存）。

### FlashVSR：真正的流式交互

> [!success] 真正的流式实现
> FlashVSR 通过 gRPC 服务 + 浏览器查看器实现了真正的流式交互。

**架构**：

```
┌──────────────┐      gRPC       ┌──────────────┐
│   Client     │ ◄──────────────► │   Server     │
│ (uplift_     │   streaming     │ (uplift_     │
│  client.py)  │                 │  server.py)  │
└──────────────┘                 └──────────────┘
       │                                │
       │ 8-frame chunks                 │
       ▼                                ▼
┌──────────────┐                 ┌──────────────┐
│ Input Video  │                 │ FlashVSR     │
│ (30 fps)     │                 │ Pipeline     │
└──────────────┘                 └──────────────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │ Browser      │
                                │ Viewer       │
                                │ (MJPEG)      │
                                └──────────────┘
```

**关键组件**：

| 组件 | 作用 |
|------|------|
| **gRPC Server** | 保持 Pipeline 预热，接受流式请求 |
| **双向流** | `upscale_video` 支持实时输入输出 |
| **多线程** | 接收、处理、发送并行 |
| **帧合并** | 8 帧 → 13/16 帧，匹配模型需求 |
| **浏览器查看器** | MJPEG 实时显示 |

**启动方式**：

```bash
# 启动服务端
uv run python -m flashvsr.grpc.uplift_server \
    --port 50051 --viewer_port 8080

# 启动客户端（流式）
uv run python -m flashvsr.grpc.uplift_client \
    --continuous --server localhost:50051 --input clip.mp4

# 浏览器查看
# 打开 http://localhost:8080/
```

---

## 🎮 HY-WorldPlay 深度分析

> [!info] 交互式世界模型
> HY-WorldPlay 是腾讯混元的实时交互式世界模型，支持动作 + 相机轨迹控制。

### 核心创新

> [!important] 三大核心创新
> 1. **PRoPE** - 相机感知的位置编码
> 2. **双分支注意力** - RoPE + PRoPE
> 3. **记忆机制** - FOV 重叠选择

### PRoPE（Projective Positional Encoding）

> [!note] 什么是 PRoPE？
> **P**rojective **R**otary **P**osition **E**ncoding，将相机参数编码到位置编码中。

**数学原理**：

```python
# 1. 构建投影矩阵
P = lift(K) @ viewmats           # image ← world
P_inv = inv(viewmats) @ lift(inv(K))  # world ← image
P_T = P.transpose(-1, -2)        # 转置

# 2. 应用到 Q/K/V
q_prope = q @ P_T        # 查询在图像空间
k_prope = k @ P_inv      # 键在世界空间
v_prope = v @ P_inv      # 值在世界空间

# 3. 注意力计算
attn = q_prope @ k_prope.T  # 跨视角的注意力

# 4. 输出投影
out = attn @ v_prope
out = out @ P              # 变换回原始空间
```

**代码实现**（`_prope.py`）：

```python
def prope_qkv(q, k, v, *, viewmats, Ks):
    """应用 PRoPE 投影位置编码"""
    
    # 构建投影矩阵
    if Ks is not None:
        P = torch.einsum("...ij,...jk->...ik", _lift_K(Ks_norm), viewmats)
        P_T = P.transpose(-1, -2)
        P_inv = torch.einsum("...ij,...jk->...ik", _invert_SE3(viewmats), ...)
    else:
        P = viewmats
        P_T = P.transpose(-1, -2)
        P_inv = _invert_SE3(viewmats)
    
    # 构建应用函数
    apply_fn_q = partial(_apply_tiled_projmat, matrix=P_T)
    apply_fn_kv = partial(_apply_tiled_projmat, matrix=P_inv)
    apply_fn_o = partial(_apply_tiled_projmat, matrix=P)
    
    # 应用到 Q/K/V
    return apply_fn_q(q), apply_fn_kv(k), apply_fn_kv(v), apply_fn_o
```

**为什么需要 PRoPE？**

```
场景：多相机视角的视频生成

相机 1 (正面)          相机 2 (侧面)
    ┌─────┐              ┌─────┐
    │  A  │              │  B  │
    │     │              │     │
    └─────┘              └─────┘

问题：同一个 3D 点在不同相机中的 2D 坐标不同
解决：PRoPE 编码相机参数，让模型理解视角关系
```

### 双分支自注意力

> [!important] 双分支设计
> 同时使用 RoPE（编码时间）和 PRoPE（编码空间），实现时空联合建模。

**架构**：

```
输入 tokens
    │
    ├──► Q, K, V
    │
    ├─────────────────────┬─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌───────────┐       ┌───────────┐       ┌───────────┐
│ RoPE      │       │ PRoPE     │       │ Memory    │
│ Branch    │       │ Branch    │       │ Cache     │
│ (时间)    │       │ (空间)    │       │ (历史帧)  │
└───────────┘       └───────────┘       └───────────┘
    │                     │                     │
    ▼                     ▼                     ▼
┌───────────┐       ┌───────────┐       ┌───────────┐
│ Standard  │       │ Camera    │       │ Memory    │
│ Attention │       │ Attention │       │ Prepend   │
└───────────┘       └───────────┘       └───────────┘
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          │
                          ▼
                 out_rope + out_prope
                          │
                          ▼
                 o(x) + o_prope(x)
```

**代码实现**（`_camera.py`）：

```python
class HyWorldPlayPRoPESelfAttention(SelfAttention):
    """双分支自注意力：RoPE + PRoPE"""
    
    def __init__(self, ...):
        super().__init__(...)
        
        # PRoPE 分支的输出投影（零初始化）
        self.o_prope = nn.Linear(self.inner_dim, self.query_dim)
        nn.init.zeros_(self.o_prope.weight)
    
    def forward_dual_branch(self, x, kv_cache, prope_kv_cache, 
                           rope_freqs, viewmats, Ks, memory_kv_cache=None):
        """双分支前向传播"""
        
        # 计算 Q, K, V
        q_raw = self.norm_q(self.q(x))
        k_raw = self.norm_k(self.k(x))
        v_raw = self.v(x)
        
        # RoPE 分支
        k_for_rope = apply_rope_freqs(k_raw, rope_freqs_k)
        kv_cache.update(k_for_rope, v_raw)
        cached_k = kv_cache.cached_k()
        if memory_kv_cache.has_rope_kv:
            cached_k = torch.cat([memory_kv_cache.k_rope, cached_k], dim=-3)
        out_rope = self.attn_op(q_rope, cached_k, cached_v)
        out_rope = self.o(out_rope)
        
        # PRoPE 分支
        q_prope, k_prope, v_prope, apply_fn_o = prope_qkv(...)
        prope_kv_cache.update(k_prope, v_prope)
        prope_cached_k = prope_kv_cache.cached_k()
        if memory_kv_cache.has_prope_kv:
            prope_cached_k = torch.cat([memory_kv_cache.k_prope, prope_cached_k], dim=-3)
        out_prope = self.attn_op_prope(q_prope, prope_cached_k, prope_cached_v)
        out_prope = self.o_prope(out_prope)
        
        # 合并两个分支
        return out_rope + out_prope
```

### 记忆机制

> [!tip] FOV 重叠选择
> 基于相机 FOV 重叠度选择历史帧，保留视觉相关的历史信息。

**记忆帧选择算法**：

```python
def select_memory_frame_indices(
    rollout_viewmats,      # [F, 4, 4] 完整轨迹
    rollout_Ks,            # [F, 3, 3] 内参
    current_chunk_idx,     # 当前 chunk 索引
    temporal_context_size, # 时间上下文大小
    memory_frames,         # 记忆帧预算
    fov_h_deg,             # 水平 FOV
    fov_v_deg,             # 垂直 FOV
    points_local,          # 蒙特卡洛点云
) -> list[int]:
    """选择记忆帧索引"""
    
    # 1. 时间上下文：最近的 N 帧（无条件保留）
    temporal_frames = list(range(
        max(0, current_start - temporal_context_size),
        current_start
    ))
    
    # 2. FOV 重叠选择：从更早的帧中选择
    if len(temporal_frames) < memory_frames:
        # 计算当前相机的 FOV 锥体
        current_fov = compute_fov_cone(viewmats[current_chunk_idx], Ks[current_chunk_idx])
        
        # 对每个历史帧，计算 FOV 重叠度
        scores = []
        for i in range(0, current_start - temporal_context_size):
            overlap = calculate_fov_overlap_similarity(
                current_fov, rollout_viewmats[i], rollout_Ks[i], points_local
            )
            scores.append((i, overlap))
        
        # 贪心选择重叠度最高的帧
        scores.sort(key=lambda x: x[1], reverse=True)
        fov_frames = [idx for idx, _ in scores[:memory_frames - len(temporal_frames)]]
    
    return sorted(temporal_frames + fov_frames)
```

**缓存架构**：

```
HyWorldPlayPRoPEBlockCache
├── self_attn: BlockKVCache          # RoPE 分支缓存
├── prope_self_attn: BlockKVCache    # PRoPE 分支缓存
├── cross_attn: BlockKVCache         # 交叉注意力缓存
└── memory: HyWorldPlayMemoryKVCache # 记忆缓存
    ├── k_rope: [batch, mem_len, heads, dim]
    ├── v_rope: [batch, mem_len, heads, dim]
    ├── k_prope: [batch, mem_len, heads, dim]
    └── v_prope: [batch, mem_len, heads, dim]
```

### 完整流程

```
Chunk 0（冷启动）：
┌─────────────────────────────────────────────────────────┐
│ 1. 初始化缓存                                            │
│ 2. 生成第一块（无记忆）                                   │
│ 3. finalize → 更新 KV Cache                              │
└─────────────────────────────────────────────────────────┘

Chunk 1+（稳态）：
┌─────────────────────────────────────────────────────────┐
│ 1. 选择记忆帧（时间上下文 + FOV 重叠）                    │
│ 2. 预填充记忆缓存（prefill_memory_kv）                    │
│    - 计算记忆帧的 Q, K, V                                │
│    - 写入 memory.k_rope, memory.v_rope                   │
│    - 写入 memory.k_prope, memory.v_prope                 │
│ 3. 生成当前块                                            │
│    - 双分支注意力：RoPE + PRoPE                          │
│    - 拼接记忆缓存：[memory K/V, current K/V]             │
│ 4. finalize → 更新 KV Cache                              │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 设计模式总结

### 生成模式分类

> [!abstract] 三种生成模式
> 1. **单次生成** - 一次性生成完整视频（Cosmos）
> 2. **流式架构** - 逐块生成，但最后拼接（Causal-Forcing, HY-WorldPlay）
> 3. **流式交互** - 逐块生成，实时输出（FlashVSR, LingBot）

| 模式 | 代表 | 特点 | 适用场景 |
|------|------|------|----------|
| 单次生成 | Cosmos | 双向注意力，质量高 | 离线视频生成 |
| 流式架构 | Causal-Forcing, HY-WorldPlay | 因果注意力，可无限生成 | 长视频生成 |
| 流式交互 | FlashVSR, LingBot | 实时输出，可交互 | 实时应用 |

### 流式程度对比

> [!warning] 架构流式 vs 应用流式
> 有流式架构不等于有流式交互！

| 集成 | 架构流式 | 实时输出 | 交互控制 | 服务层 |
|------|----------|----------|----------|--------|
| **Cosmos** | ❌ | ❌ | ❌ | ❌ |
| **Causal-Forcing** | ✅ | ❌ | ❌ | ❌ |
| **HY-WorldPlay** | ✅ | ❌ | ✅ (相机) | ❌ |
| **FlashVSR** | ✅ | ✅ | ✅ | ✅ gRPC |
| **LingBot** | ✅ | ✅ | ✅ | ✅ WebRTC |

### 优化建议

> [!tip] 代码优化点
> 1. **预分配 tensor** - 避免最后的 cat 操作
> 2. **异步拷贝** - 使用 CUDA stream
> 3. **双缓冲** - 一个在处理，一个在写入

**当前实现**（FlashVSR Runner）：

```python
chunks_out: list[torch.Tensor] = []
for chunk_idx, (start, size) in enumerate(chunks):
    video_chunk = self.pipeline.generate(...)
    chunks_out.append(video_chunk.cpu())

# 最后拼接
generated = torch.cat(chunks_out, dim=2)
```

**优化方案**：

```python
# 1. 预分配
generated = torch.zeros(
    (1, 3, total_output_frames, output_H, output_W),
    dtype=dtype, device='cpu'
)

# 2. 逐块填入
current_frame = 0
for chunk_idx, (start, size) in enumerate(chunks):
    video_chunk = self.pipeline.generate(...)
    chunk_frames = video_chunk.shape[2]
    generated[:, :, current_frame:current_frame + chunk_frames] = video_chunk.cpu()
    current_frame += chunk_frames
```

---

## 🔗 相关文档

- [[论文阅读清单]] - 流式视频生成论文列表
- [[todolist]] - 学习规划和任务跟踪
- [[Audio Interaction Model]] - 流式音频交互模型

---

*最后更新: 2026-06-04*
*标签: #FlashDreams #流式视频生成 #PRoPE #双分支注意力 #记忆机制*
