# LingBot-World 2.0 (Infinity): 无限交互世界模型

> **论文**: Infinite Worlds with Versatile Interactions
> **作者**: Robbyant Team (21 人)
> **来源**: arXiv 2607.07534 | 14B 参数 + 1.3B 轻量版
> **代码**: `/media/lyxwxj/Data/common/Workspace/Omni-infra/world/lingbot-world-v2`

---

## 一、核心升级

| 特性 | LingBot-World v1 | LingBot-World 2.0 |
|------|-----------------|-------------------|
| 交互时长 | 分钟级 | **小时级（无限）** |
| 实时性 | ✓ | **720p @ 60fps** |
| 交互元素 | 有限 | **攻击、射箭、施法、射击 + 文本事件** |
| 智能体 | 无 | **Pilot + Director 双智能体** |

---

## 二、架构总览

![Overview](../../流式视频生成/assets/lingbot-world/overview_page-0001.jpg)

### 基础骨干

- **Diffusion Transformer (DiT)**，基于 Wan2.2
- **因果视频生成**（自回归逐块生成）
- **条件流匹配** (Conditional Flow Matching) 训练目标

### 两种推理模式

| 模式 | 模型 | 采样步数 | CFG |
|------|------|---------|-----|
| `causal_fast` | 蒸馏少步模型 | **4 步/块** | 无 |
| `causal_pretrain` | 预训练因果模型 | 40 步/块 | 有 (scale=5.0) |

---

## 二（续）、模型架构详解

### 单流 vs 双流架构

LingBot-World 使用**单流架构**：视觉和条件 token 拼接成一条序列，共享同一套 Transformer 参数。

```
单流 (LingBot-World):                    双流 (Cosmos3):
[视觉 tokens, 条件 tokens]                [文本 tokens]    [视觉 tokens]
        ↓                                      ↓                ↓
  同一套 QKV + FFN                         UND QKV + FFN    GEN QKV + FFN
  (Self-Attention 自然融合)                 (因果自注意力)      (交叉注意力 → [K_und; K_gen])
```

| | 单流 | 双流 |
|---|---|---|
| Token 序列 | 一条 `[视觉, 条件]` | 两条 AR `[文本]` + DM `[视觉]` |
| 参数共享 | 全部共享 | 各自独立 |
| GEMM 效率 | 统一大 GEMM → MFU 高 | 分离小 GEMM → MFU 低 |
| 文本注入 | Cross-Attention（K/V 缓存） | UND 塔 → 交叉注意力 |

### 整体数据流

> 📎 可视化流程图：[lingbot-world-dataflow.drawio](lingbot-world-dataflow.drawio)

```
输入图像 ──→ VAE.encode() ──→ y [B, 20, F, H, W] ──┐
                                                     ├─→ cat → Conv3d patch_emb
噪声 latent ──────────────────────────→ x [B, 16, F, H, W] ──┘

文本 prompt ──→ T5 Encoder ──→ context [L, 4096] ──→ text_embedding (Linear-GELU-Linear)

摄像机轨迹 ──→ Plücker 编码 ──→ c2ws [B, 384, F, hp, wp] ──→ cam_embedding (Linear)

时间步 t ──→ sinusoidal ──→ time_embedding (MLP) ──→ time_projection → 6D 调制信号
```

### 组件规格

| 组件 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `patch_embedding` | `x concat y` [B, 36, F, H, W] | [B, F·hp·wp, dim] | Conv3d, patch_size=(1,2,2) |
| `text_embedding` | T5 输出 [B, L, 4096] | [B, L, dim] | Linear-GELU-Linear |
| `time_embedding` | sinusoidal(t) [B, F, 256] | [B, F, dim] | Linear-SiLU-Linear |
| `time_projection` | time_embedding | [B, F, 6, dim] | SiLU-Linear，6D 调制 |
| `patch_embedding_wancamctrl` | Plücker [B, 384, F, hp, wp] | [B, F·hp·wp, dim] | 摄像机条件注入 |

### Block 结构（CausalWanAttentionBlock）

每个 Block 内部的数据流：

```
输入 x [B, S, dim]
    │
    ├──→ norm1(x) → AdaLN 调制 (scale_msa, shift_msa)
    │         │
    │         ▼
    │    Self-Attention (因果 RoPE + KV Cache)
    │         │
    │         ▼
    │    x = x + gate_msa * attn_out
    │
    ├──→ 摄像机注入（如果存在 c2ws_plucker_emb）
    │    cam_scale, cam_shift = MLP(c2ws)
    │    x = (1 + cam_scale) * x + cam_shift
    │
    ├──→ norm3(x) → Cross-Attention (text context, 缓存)
    │    x = x + cross_attn_out
    │
    ├──→ norm2(x) → AdaLN 调制 (scale_mlp, shift_mlp)
    │         │
    │         ▼
    │    FFN (Linear-GELU-Linear)
    │         │
    │         ▼
    │    x = x + gate_mlp * ffn_out
    │
    └──→ 输出 x [B, S, dim]
```

### 三种条件的注入方式

| 条件类型 | 注入位置 | 注入方式 |
|---------|---------|---------|
| **文本** | Cross-Attention | K/V 来自 T5 编码，缓存后复用 |
| **摄像机** | Self-Attention 后 | `x = (1+scale) * x + shift`，scale/shift 从 Plücker MLP 得到 |
| **时间步** | 每个 Block | AdaLN：`norm(x) * (1+scale) + shift`，gate 控制残差 |

### Self-Attention 的 KV Cache 管理

```python
# 三种情况
if local_attn_size == -1:
    # 全局缓存：直接写入 [current_start:current_end]
    kv_cache["k"][:, start:end] = roped_key
elif cache_full:
    # 滚动窗口：丢弃最旧的 token，保留 sink tokens
    kv_cache["k"][:, sink:sink+roll] = kv_cache["k"][:, sink+evict:sink+evict+roll].clone()
    kv_cache["k"][:, new_start:new_end] = roped_key
else:
    # 直接追加
    kv_cache["k"][:, local_start:local_end] = roped_key
```

**参数**：
- `local_attn_size=18`：只看最近 18 个 chunk 的 KV
- `sink_size=6`：保留前 6 个 chunk 不被驱逐（注意力锚点）

### Cross-Attention 的缓存

```python
# 首次调用：计算并缓存 text 的 K/V
if is_first:
    crossattn_cache["k"] = self.norm_k(self.k(context))
    crossattn_cache["v"] = self.v(context)
    crossattn_cache["is_init"] = 1
# 后续调用：直接复用缓存
else:
    k = crossattn_cache["k"]
    v = crossattn_cache["v"]
```

**关键**：文本 K/V 只算一次，所有 chunk 所有去噪步骤复用。

---

## 三、因果生成范式

### 形式化

$$p_\theta(x_{1:T} | a_{1:T}) = \prod_t p_\theta(x_t | x_{<t}, a_{\leq t})$$

### 逐块生成

```
Chunk 1: [x₁, x₂, x₃, x₄] ← 从噪声生成，KV Cache 更新
Chunk 2: [x₅, x₆, x₇, x₈] ← 条件化 Chunk 1 的 KV Cache
Chunk 3: ...
```

**代码实现** (`image2video.py` L649-710)：

```python
for chunk_id in range(num_inference_chunk):
    current_latent = latents_chunk[chunk_id]
    current_condition = condition_chunk[chunk_id]

    # 每个 chunk 内部迭代 4 步
    for timestep_idx in range(len(timesteps)):
        noise_pred = self.model(
            x=[current_latent], t=timestep,
            kv_cache=self_kv_cache,           # 跨 chunk 复用
            crossattn_cache=cross_kv_cache,    # 跨 chunk 复用
            current_start=chunk_id * chunk_size * frame_seqlen,
            ...
        )
        x0 = self._convert_flow_pred_to_x0(noise_pred, ...)

    # 用干净 latent 更新 KV Cache
    self.model(x=[x0], t=timestep0, ...)
```

---

![General Pipeline](../../流式视频生成/assets/lingbot-world/general_pipeline_page-0001.jpg)

---

## 四、MoBA 注意力掩码（核心创新）

**MoBA = Mixture of Bidirectional and Autoregressive**

### 问题

纯 teacher forcing 会导致模型过度依赖上下文，不预测未来帧。

### 解决方案

在 teacher forcing 掩码中集成**双向注意力块**：

```
自注意力掩码:
┌─────────────────────────────┐
│ x₁  x₂  x₃  x₄  [b₁ b₂] │  ← x₄ 可以看到 b₁, b₂
│ ✓   ✓   ✓   ✓   [✓  ✓ ] │
│ ─────────────────────────── │
│ Teacher Forcing (下三角)    │  ← 标准因果
│ + 双向块 (右下角)           │  ← 帮助适应变长生成
└─────────────────────────────┘
```

### 交叉注意力

- **Teacher Forcing 部分**：每帧 attend to 背景提示 $a_B$ + 块级提示 $a_{\leq i}$（下三角）
- **双向部分**：全局提示 $a_G$ 描述整个视频的所有事件

---

## 五、动作表示：Plücker 嵌入

### 摄像机控制

将观看射线编码为 **6D Plücker 坐标**（每像素）：

```python
# image2video.py L578-586
c2ws_plucker_emb = get_plucker_embeddings(c2ws_infer, Ks, h, w)
c2ws_plucker_emb = rearrange(
    c2ws_plucker_emb,
    'f (h c1) (w c2) c -> (f h w) (c c1 c2)',
    c1=int(h // lat_h), c2=int(w // lat_w),
)
```

- 输入：摄像机外参 $c2ws$（4×4 矩阵）+ 内参 $K$
- 输出：每像素 6D 坐标，通过 AdaLN 注入扩散过程

### 文本控制

- **块级提示**：每个视频块有独立字幕
- **背景提示**：描述整体场景
- **全局提示**：描述所有事件（双向部分用）

---

## 六、训练流程

### 6.1 因果预训练

**训练目标**：条件流匹配

$$\mathcal{L}_{fm} = \mathbb{E}_{x,i,t,\epsilon} \| v_\theta(x_i^t, t | x_{<i}, p_{\leq i}, a_{\leq i}) - (\epsilon - x_i) \|^2$$

其中 $x_i^t = (1-t)x_i + t\epsilon$，$t \sim \mathcal{U}(0,1)$，$\epsilon \sim \mathcal{N}(0, I)$

### 6.2 蒸馏

**两阶段蒸馏**：

#### 阶段 1：一致性蒸馏 (Consistency Distillation)

$$\mathcal{L}_{CD} = \mathbb{E}\left[d\left(G_\theta(x_i^t, t|c), G_{\theta^-}(\tilde{x}_i^{t-\Delta t}, t-\Delta t|c)\right)\right]$$

- 同一 PF-ODE 轨迹上的状态映射到相同预测
- $\theta^-$ 是 $\theta$ 的 EMA

#### 阶段 2：DMD (Distribution Matching Distillation)

$$\nabla_\theta \mathbb{E}\left[D_{KL}(p_{\theta,t} \| p_{data,t})\right] = -\mathbb{E}\left[(s_{real} - s_{fake}) \frac{\partial \hat{x}_i}{\partial \theta}\right]$$

**关键**：DMD 在**长自展开轨迹**上应用，优化学生自己的预测分布，减少累积漂移。

---

## 七、推理优化

### 7.1 KV Cache 管理

```python
# 局部注意力窗口
self.local_attn_size = 18  # 只看最近 18 个 chunk
self.sink_size = 6         # 保留前 6 个 chunk（sink tokens）

# KV Cache 大小计算
if self.local_attn_size > -1:
    kv_size = frame_seqlen * self.local_attn_size  # 有界
else:
    kv_size = frame_seqlen * lat_f                 # 全量
```

**动态 KV Cache 调度**：
- 保留最有信息量的历史
- 丢弃低价值条目
- 减少有效上下文 → 更快推理
- 抑制陈旧内容干扰 → 更好质量

### 7.2 跨 Chunk KV Cache 复用

```python
# 自注意力 KV Cache
self_kv_cache = [{
    'k': torch.zeros(shape),           # [B, kv_size, H_kv, D]
    'v': torch.zeros(shape),
    'global_end_index': torch.tensor([0]),
    'local_end_index': torch.tensor([0])
} for _ in range(num_layers)]

# 交叉注意力 KV Cache（T5 编码结果，只算一次）
crossattn_cache = [{
    'k': torch.zeros(shape),           # [B, text_len, H, D]
    'v': torch.zeros(shape),
    'is_init': torch.tensor(0)         # 首次计算标志
} for _ in range(num_layers)]
```

### 7.3 T5 缓存

```python
# 相同 prompt 不重复编码
cache_key = hashlib.sha256(input_prompt.encode('utf-8')).hexdigest()
if cache_key in self._t5_cache:
    context = self._t5_cache[cache_key]
else:
    context = self.text_encoder([input_prompt], self.device)
    self._t5_cache[cache_key] = context
```

### 7.4 Prewarm（预热）

```python
def prewarm(self, img, max_area, frame_num, chunk_size, text_seq_len):
    """运行一次 dummy forward，预热 CUDA kernel、FSDP all-gather、Ulysses all-to-all"""
    # 避免首次 generate() 的 ~7s warmup 开销
```

### 7.5 分布式策略

| 策略 | 说明 |
|------|------|
| FSDP | DiT 和 T5 分片 |
| Ulysses SP | 序列并行，注意力头分区 |
| CPU Offload | 单 GPU 时模型在 CPU/GPU 间交换 |

---

## 八、数据管道

![Data Pipeline](../../流式视频生成/assets/lingbot-world/data_pipeline_page-0001.jpg)

### 三阶段

1. **数据获取**：自采第一人称视频 + 合成数据（游戏/UE）+ 大规模网络视频
2. **数据画像**：技术过滤（解码、镜头边界、质量评分）+ VLM 画像（Qwen 模型）
3. **多维标注**：
   - 视频级全局字幕
   - 块级时间局部描述
   - 多轨道事件级标注（主体可见性、运动状态、交互状态、环境动态）

---

## 九、智能体系统

### Director-Pilot 协同模拟

```
┌─────────────────┐     事件提案      ┌─────────────────┐
│   VLM Director   │ ──────────────→ │  DiT Pilot       │
│  (因果推理)       │                  │  (物理渲染)       │
│  分析当前画面      │ ←────────────── │  生成下一帧        │
│  预测后果          │     视觉反馈      │                  │
└─────────────────┘                  └─────────────────┘
```

### 两种交互模式

**模式 A：直接语义交互**
- VLM 分析当前帧 → 生成动态事件卡 → 用户选择

**模式 B：跟踪辅助对象交互**
- SAM 跟踪对象 → 用户选择 → VLM 推断状态变化 → 视频模型渲染

### 用户干预

- **全局状态转移**：时间、天气、全局事件
- **局部实体注入**：生成新实体，VLM 确定时空切入点

---

## 十、关键代码结构

```
lingbot-world-v2/
├── generate.py              # 入口，CLI 参数解析
├── wan/
│   ├── __init__.py          # 导出 WanI2VCausal
│   ├── image2video.py       # 核心推理管道
│   │   ├── WanI2VCausal     # 主类
│   │   ├── _generate_causal_fast    # 蒸馏少步推理
│   │   ├── _generate_causal_pretrain # 预训练 40 步推理
│   │   └── prewarm          # 预热
│   ├── modules/
│   │   ├── model_fast.py    # WanModelFast（蒸馏模型）
│   │   ├── model_causal.py  # WanModelCausal（预训练模型）
│   │   ├── t5.py            # T5 编码器
│   │   └── vae2_1.py        # VAE
│   ├── distributed/         # FSDP、SP
│   └── utils/               # 调度器、相机工具
└── run_fast.sh              # 快速启动脚本
```

---

## 十一、与你研究方向的关联

1. **因果逐块生成**：KV Cache 管理是核心瓶颈
2. **MoBA 掩码**：混合双向+因果的注意力设计
3. **DMD 蒸馏**：4 步推理 vs 40 步，关键加速手段
4. **动态 KV Cache**：有界窗口（local_attn_size + sink_size）
5. **预热机制**：避免首次推理的 warmup 开销
