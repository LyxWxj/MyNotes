# Cosmos 3 模型架构深度解析（面向推理优化）

> 基于论文 arXiv 2606.02800 + vllm-omni 代码实现
> 阅读目标：理解架构细节以指导推理优化

---

## 一、整体架构：双塔 Mixture-of-Transformers (MoT)

Cosmos 3 的核心是**双塔 MoT 架构**——每个 Transformer 解码器层包含**两套独立参数**：

![MoT Architecture](../../流式视频生成/assets/tikz_mot_architecture_page-0001.jpg)

```
┌───────────────────────────────────────────────────────────────┐
│                    Transformer Layer                          │
│                                                               │
│  ┌────────────────────────┐  ┌──────────────────────────┐     │
│  │   UND Tower (Reasoner) │  │   GEN Tower (Generator)  │     │
│  │                        │  │                          │     │
│  │  Causal Self-Attn      │  │  Cross-Attention         │     │
│  │  (text tokens only)    │  │  (Q_gen → [K_und;K_gen]) │     │
│  │                        │  │                          │     │
│  │  GatedMLP              │  │  GatedMLP                │     │
│  └────────────────────────┘  └──────────────────────────┘     │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 代码对应

```python
# transformer_cosmos3.py

class Cosmos3VFMTransformer(nn.Module):
    self.language_model = Cosmos3LanguageModel(...)  # UND tower
    self.gen_layers = nn.ModuleList([               # GEN tower
        Cosmos3GenDecoderLayer(...) for _ in range(num_hidden_layers)
    ])
```

**关键点**：两套参数**独立**，不共享权重。

---

## 二、两条路径的注意力机制

### 2.1 UND 路径：因果自注意力

```python
class Cosmos3CausalAttention(nn.Module):
    # causal=True, 只看 text tokens
    self.attn = FrameworkAttention(
        causal=True,
        softmax_scale=1.0 / (self.head_dim ** 0.5),
        skip_sequence_parallel=True,  # UND 不做 SP！
    )
```

**输入**：text_ids → embed_tokens → 层层因果自注意力
**输出**：每层的 (K, V) 缓存，供 GEN 交叉注意力使用

### 2.2 GEN 路径：交叉注意力

```python
class Cosmos3CrossAttention(nn.Module):
    # causal=False, 全连接
    self.attn = FrameworkAttention(
        causal=False,
        softmax_scale=1.0 / (self.head_dim ** 0.5),
        # 不 skip SP，GEN 路径做序列并行
    )
```

**核心公式**（论文 Eq.7-8）：

```
O_AR = Attn_causal(Q_AR, K_AR, V_AR)           # UND: 只看自己
O_DM = Attn_full(Q_DM, [K_AR; K_DM], [V_AR; V_DM])  # GEN: 看 AR + DM
```

**关键约束**：AR tokens **永远不会被 DM tokens 更新**——因果完整性保持。

### 代码实现（非 SP 路径）

```python
def _forward_local(self, q, k, v, k_und, v_und):
    # 拼接 UND 和 GEN 的 K/V
    k_all = torch.cat([k_und, k], dim=1)  # [B, S_und + S_gen, H_kv, D]
    v_all = torch.cat([v_und, v], dim=1)
    out = self.attn(q, k_all, v_all)
    return out.reshape(B, S_gen, -1)
```

### 代码实现（SP 路径）

```python
def _forward_sp(self, q, k, v, k_und, v_und):
    # Ulysses: k_und/v_und 作为 joint_key/joint_value
    attn_metadata = AttentionMetadata(
        joint_query=q.new_empty(B, 0, ...),  # 空的 joint_query
        joint_key=k_und,
        joint_value=v_und,
        joint_strategy="front",
    )
    out = self.attn(q, k, v, attn_metadata)
```

---

## 三、Token 排列与生成模式（Sequence Packing）

![Multiresolution Sequence Packing](../../流式视频生成/assets/tikz_multiresolution_sequence_packing_page-0001.jpg)

### 什么是 Sequence Packing？

不同模态（文本、视频、音频、动作）的**分辨率完全不同**，Packing 就是把它们**打包成一条统一的一维序列**，通过 mRoPE 位置编码保留语义关系。

```
|←──── AR 子序列 ────→|←──────────── DM 子序列 ────────────→|
[语言 tokens, ViT tokens | 控制tokens, 视频tokens, 音频tokens, 动作tokens]
```

### 3.1 两个子序列

```
[S_AR, S_DM]
  │      │
  │      └── VAE 视频/图像 tokens + 音频 tokens + 动作 tokens（噪声/干净）
  └── 语言 tokens + ViT 视觉 tokens（理解用）
```

### 3.2 统一格式规则

1. AR tokens 在 DM tokens 之前
2. DM 内部：干净条件 tokens → 噪声目标 tokens
3. 条件和扩散内部：视觉 → 音频 → 动作

### 3.3 各模式的 token 排列

| 模式 | 排列 |
|------|------|
| T2I | [S_AR, ṽ₁] |
| T2V | [S_AR, ṽ₁:ₙ] |
| I2V | [S_AR, v₁, ṽ₂:ₙ] |
| V2V | [S_AR, v₁:ₚ, ṽₚ₊₁:ₙ] |
| Transfer | [S_AR, v^ctrl₁:ₙ, ṽ₁:ₙ] |
| Policy | [S_AR, ṽ₁:ₙ, ã₁:ₘ] |

![Action Modes](../../流式视频生成/assets/tikz_action_modes_page-0001.jpg)

![Action Representation](../../流式视频生成/assets/tikz_action_representation_page-0001.jpg)

### 3.4 各模式排列详解

**三条核心规则**：
1. AR 在前，DM 在后
2. DM 内部：干净条件 → 噪声目标
3. 同类型内部：视觉 → 音频 → 动作

**T2I**：`[语言 | ṽ₁]` — 最简单，1 个噪声帧

**T2V**：`[语言 | ṽ₁:ₙ]` — 全部噪声帧，从零生成

**I2V**：`[语言 | v₁, ṽ₂:ₙ]` — 第一帧干净（不加时间步嵌入），其余噪声

**V2V**：`[语言 | v₁:ₚ, ṽₚ₊₁:ₙ]` — 前 P 帧干净输入，后面噪声

**Transfer**：`[语言 | v^ctrl₁:ₙ, ṽ₁:ₙ]` — 控制信号（深度/边缘）在前，噪声 RGB 在后

**Policy**：`[语言 | ṽ₁:ₙ, ã₁:ₘ]` — 视频和动作同时去噪

### 3.5 Packing 的代码实现

```python
# 1. 各模态独立投影到 hidden_size
hidden_video = self.proj_in(self.patchify(hidden_states, t, h, w))
hidden_action = self.action_proj_in(action_tokens, domain_ids)
hidden_sound = self.audio_proj_in(sound_tokens)

# 2. 添加模态嵌入（区分模态）
hidden_action = hidden_action + self.action_modality_embed

# 3. 只对噪声帧添加时间步嵌入
hidden_video = hidden_video + time_embed * token_noisy_mask

# 4. 拼接成一条序列
hidden_gen = torch.cat([*hidden_controls, hidden_video, hidden_action, hidden_sound], dim=1)
```

### 3.6 mRoPE 如何区分模态

拼接后所有 token 位置连续 `[0, 1, 2, ..., N]`，但 mRoPE 分配**3D 物理坐标**：

| 模态 | t | h | w |
|------|---|---|---|
| 视频 | 帧索引 | 空间网格 | 空间网格 |
| 音频 | hop 索引 | 0 | 0 |
| 动作 | 步索引 | 0 | 0 |

AR-DM 之间插入 **15,000** 时间间隔，防止初始帧伪影。

---

## 四、3D mRoPE 位置编码

![mRoPE Coordinate Assignment](../../流式视频生成/assets/tikz_mrope_coordinate_assignment_page-0001.jpg)

### 4.1 核心设计

Cosmos 3 使用 **3D mRoPE + 绝对时间索引** 对齐不同模态的 token：

```
时间轴 t: 视频帧索引、音频 hop 索引、动作步索引
空间轴 h, w: 视频/图像的空间位置
```

### 4.2 各模态的位置 ID

| 模态 | t | h | w |
|------|---|---|---|
| 语言 token | 单调递增 | = t | = t |
| ViT 视觉 token | 每帧共享 | 空间网格 | 空间网格 |
| VAE 视频 token | 潜在帧索引 | 空间网格 | 空间网格 |
| 音频 token | 每 hop 递增 | 0 | 0 |
| 动作 token | 每步递增 | 0 | 0 |

### 4.3 AR-DM 时间边距

**关键**：AR 和 DM 子序列之间插入 **15,000** 的固定时间间隔。

```python
# transformer_cosmos3.py
self.temporal_modality_margin = 15000
media_temporal_offset = t_offset + self.temporal_modality_margin
```

**目的**：防止初始视频帧的"过饱和和棋盘伪影"。

### 4.4 FPS 调制

不同帧率的视频通过 TPS（Temporal Steps Per Second）归一化到同一时间轴：

```python
tps = fps / temporal_compression_factor
base_tps = base_fps / effective_base_tcf  # base_tps = 24/4 = 6
t_index = (frame_indices / tps * base_tps + temporal_offset)
```

---

## 五、模型变体

| 变体 | 总参数 | 稠密参数 | 层数 | 隐藏维度 | 注意力头 | KV 头 | FFN 维度 |
|------|--------|---------|------|---------|---------|-------|---------|
| **Edge** | ~4B | ~2B | 28 | 2,048 | 16 | 8 | 9,216 |
| **Nano** | ~16B | ~8B | 36 | 4,096 | 32 | 8 | 12,288 |
| **Super** | ~64B | ~32B | 64 | 5,120 | 64 | 8 | 25,600 |

---

## 六、推理优化关键点

### 6.1 UND 缓存（最重要）

```python
# Cosmos3VFMTransformer.forward()
if self.cached_kv is None:
    # UND 只运行一次！
    with self._offload_context("reasoner"):
        cached_kv_full = self.language_model(text_ids, freqs_und)
    self.cached_kv = [(k[:, :max_real_len], v[:, :max_real_len]) ...]

# GEN 每个去噪步骤都运行
with self._offload_context("generator"):
    for layer, (k_und, v_und) in zip(self.gen_layers, self.cached_kv):
        hidden_gen = layer(hidden_gen, k_und=k_und, v_und=v_und, ...)
```

**优化机会**：
- UND K/V 缓存：只计算一次，所有去噪步骤复用
- UND 不做序列并行（skip_sequence_parallel=True）
- UND K/V 是 replicated（所有 rank 持有相同副本）

### 6.2 序列并行 (Ulysses)

```python
_sp_plan = {
    "gen_sp_prepare": {
        0: SequenceParallelInput(split_dim=1, expected_dims=3, split_output=True),
        1: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
        2: SequenceParallelInput(split_dim=1, expected_dims=4, split_output=True),
    },
    "gen_sp_gather": SequenceParallelOutput(gather_dim=1, expected_dims=3),
}
```

**关键**：只有 GEN 路径做 SP，UND 路径不做。

### 6.3 CPU Offload（组件级）

```python
def _model_cpu_offload_components(self):
    return {
        "reasoner": [self.language_model.layers],  # UND 层
        "generator": [self.gen_layers],             # GEN 层
    }
```

**策略**：reasoner 和 generator **互斥**交换——运行 UND 时卸载 GEN，运行 GEN 时卸载 UND。

### 6.4 Patchify / Unpatchify

```python
def patchify(self, latents, t, h, w):
    # [B, C, t, h, w] → [B, t*hp*wp, p*p*C]
    # p = latent_patch_size = 2
    # C = latent_channel_size = 48
    # patch_latent_dim = 2*2*48 = 192
```

**优化点**：投影层很小（192 → hidden_size），不值得量化。

### 6.5 时间步嵌入

```python
# fp32 精度
with torch.autocast(..., enabled=False):
    time_embed = self.time_embedder((timestep * self.timestep_scale).float())
```

**注意**：时间步嵌入强制 fp32，post_load_weights 时转为 fp32。

### 6.6 噪声帧掩码（I2V/V2V）

```python
if noisy_frame_mask is not None:
    # 只对噪声帧添加时间步嵌入
    token_noisy_mask = noisy_frame_mask[:, 0, :, 0, 0].unsqueeze(-1).expand(...)
    hidden_video = hidden_video + time_embed.unsqueeze(1) * token_noisy_mask
else:
    hidden_video = hidden_video + time_embed.unsqueeze(1)  # 全部加
```

**I2V 优化**：条件帧（frame 0）不加时间步嵌入，每步重新注入干净 latent。

### 6.7 CFG（Classifier-Free Guidance）

标准模式：cond + uncond 两路

Transfer 模式：最多 3 路
- `control_and_text`: cond_full + cond_no_control + uncond_full
- `control_only`: cond_full + cond_no_control
- `text_only`: cond_full + uncond_full

### 6.8 Cache-DiT 集成

```python
_cache_dit_adapter_config = CacheDiTAdapterConfig(
    block_forward_patterns={"gen_layers": ForwardPattern.Pattern_3},
    has_separate_cfg=True,
    check_forward_pattern=False,
)
```

**注意**：cache-dit 的 `has_separate_cfg=True` 要求顺序 CFG 循环保持 cond/uncond 配对。

---

## 七、数据流总结

```
输入文本 → Tokenizer → text_ids [B, S_text]
                              │
                              ▼
                    ┌─── UND (Reasoner) ───┐
                    │  embed_tokens         │
                    │  N 层 CausalAttention  │  ← 只运行一次
                    │  输出: cached_kv       │
                    └───────────────────────┘
                              │
输入视频/图像 → VAE → latents [B, C, t, h, w]
                              │
                              ▼
                    ┌─── patchify + proj_in ──┐
                    │  [B, t*hp*wp, 192]      │
                    │  → [B, t*hp*wp, hidden] │
                    └─────────────────────────┘
                              │
                    + time_embed (只对噪声帧)
                    + control/action/sound tokens (可选)
                              │
                              ▼
                    ┌─── GEN (Generator) ───┐
                    │  N 层 CrossAttention    │  ← 每个去噪步骤
                    │  Q_gen → [K_und;K_gen] │
                    │  + GatedMLP            │
                    └────────────────────────┘
                              │
                              ▼
                    ┌─── norm + proj_out ───┐
                    │  unpatchify           │
                    │  → [B, C, t, h, w]   │
                    └───────────────────────┘
                              │
                              ▼
                         Scheduler step
                              │
                              ▼
                         重复 N 步
```

---
