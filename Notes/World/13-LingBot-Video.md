# LingBot-Video: MoE 视频预训练用于具身智能

> **论文**: Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence
> **作者**: Robbyant Team (27 人)
> **来源**: arXiv 2607.07675
> **代码**: `/media/lyxwxj/Data/common/Workspace/Omni-infra/world/lingbot-video`

---

## 一、核心定位

LingBot-Video 是一个**任务统一的单流扩散 Transformer**，使用 **Sparse MoE** 扩展参数容量，支持 T2I、T2V、TI2V 任务。

**关键特性**：
- 单流架构（非双流）→ 更高 MFU
- Sparse MoE → 容量-计算解耦
- 级联 Refiner → 480p → 1080p

---

## 二、整体架构

![LingBot-Video Architecture](../../流式视频生成/assets/lingbot-video/architecture_page-lingbot-video.jpg)

### 级联设计

```
Base Generator (480p) → Refiner (1080p)
     ↓                      ↓
  低分辨率生成            超分辨率细化
```

### Base Generator 组件

| 组件 | 说明 |
|------|------|
| **文本编码器** | Qwen3-VL-4B 提取条件 |
| **视觉编码器** | Wan2.1-VAE 压缩视觉潜在 |
| **扩散 Transformer** | 单流 DiT + Sparse MoE |

---

## 三、任务统一的单流 DiT

### 3.1 统一输入公式

所有任务（T2I/T2V/TI2V）统一为一条 token 序列：

```
[视觉 latent patches, 条件 tokens]
```

- 图像 = 单帧视频（T=1）
- 条件和视觉 token 投影到同一隐藏维度
- 通过 **3D MM-RoPE** 分离

### 3.2 Unified Input 组成

统一输入由**条件 tokens + 视觉 latent patches** 拼接成一条序列：

| 组件 | 来源 | Token 数 | 3D MM-RoPE 坐标 |
|------|------|---------|----------------|
| 条件 tokens | Qwen3-VL-4B 编码 | L 个 | `(i, 0, 0)` |
| 视觉 patches | Wan2.1-VAE + 噪声 | F×H×W 个 | `(L+1+f, h, w)` |

```python
# 序列维度拼接
unified_tokens = torch.cat([condition_tokens, visual_patches], dim=1)
```

注：LingBot-World 的 Unified Input 是噪声 latent + 条件 latent **通道维度拼接**后 patchify，文本通过 Cross-Attention 注入。

### 3.3 为什么选单流（非双流）？

| | 单流 (LingBot-Video) | 双流 (Cosmos3) |
|---|---|---|
| 参数复用 | 最大化 | 分离 |
| 跨模态交互 | 每层密集 | 交叉注意力 |
| GEMM 效率 | 统一大 GEMM | 分离小 GEMM |
| 内存带宽 | 连续处理 | 频繁拼接/分割 |
| 通信模式 | 简化 | 复杂 |

**论文原话**："groups multi-modal features into larger, unified GEMM computations, aiming to improve MFU"

### 3.3 3D MM-RoPE

条件 token 和视觉 token 在**不重叠的时间坐标范围**内：

```
条件 tokens:  (i, 0, 0)  for i = 1, ..., L
视觉 tokens:  (L+1+f, h, w)  for f=0..F-1, h=0..H-1, w=0..W-1
```

- 保持空间局部性和时间顺序
- 注意力完全单流
- 无需任务特定架构

### 3.4 QK-Norm

```python
q = self.norm_q(q)  # Per-head RMSNorm
k = self.norm_k(k)
q = apply_rotary_emb(q, rotary_emb)
k = apply_rotary_emb(k, rotary_emb)
```

稳定深层 Transformer 的注意力，控制注意力 logit 增长。

### 3.5 AdaLN-Single 调制

```
时间步 → time_embedder → time_modulation → 共享 6D 调制信号
                                              ↓
每个 Block: scale_shift_table + 共享信号 → (shift, scale, gate)
```

**关键**：时间步 MLP 只算一次，每层加一个可训练调制表。

---

## 四、Sparse MoE 架构

### 4.1 为什么用 MoE？

视频生成需要建模：
- 复杂物理过程（流体运动、3D 空间一致性）
- 多样运动轨迹
- 丰富材质纹理

**问题**：密集 FFN 强制所有 token 共享同一参数路径 → 子任务干扰

**MoE 解决**：容量-计算解耦
- 总参数容量 ∝ 专家池大小
- 每 token 活跃 FLOPs 恒定

### 4.2 MoE 公式

$$m(\mathbf{u}_t) = \sum_{i=1}^{N_s} E_i^{(s)}(\mathbf{u}_t) + \sum_{j \in \mathcal{R}_b(\mathbf{u}_t)} g_{t,j} E_j^{(r)}(\mathbf{u}_t)$$

- $N_s$：共享专家数
- $\mathcal{R}_b$：选中的路由专家集合
- $g_{t,j}$：路由权重

### 4.3 路由机制

**Sigmoid 路由器**（非 softmax）：

$$\alpha_{t,j} = \text{Sigmoid}(\mathbf{u}_t^\top \mathbf{r}_j)$$

**分组限制路由**（DeepSeek 风格）：
1. 将 $N_r$ 专家分为 $N_g$ 组
2. 每组取 top-2 偏差校正亲和度之和作为组分
3. 选 top $K_g$ 组
4. 在选中组内选 top $K_r$ 专家

### 4.4 在线偏差校正（无辅助损失）

$$b_j \leftarrow b_j - \eta \cdot \text{sign}(n_j - \bar{n})$$

- $n_j$：专家 $j$ 的有效 token 分配数
- $\bar{n}$：每专家平均负载
- 选择时用 $\tilde{\alpha}_{t,j} = \alpha_{t,j} + b_j$
- 门控值用原始 $\alpha_{t,j}$（无偏差）

### 4.5 序列级辅助损失

$$\mathcal{L}_{\text{seq}} = \frac{1}{S}\sum_{s=1}^{S}\sum_{j=1}^{N_r} f_j^{(s)} P_j^{(s)}$$

- 鼓励每个打包视频序列内的平衡使用
- 而非仅全局 batch 级别

---

## 五、Block 结构

```python
class LingBotVideoBlock(nn.Module):
    def forward(self, x, temb6, rotary_emb, ...):
        # AdaLN 调制
        mod = temb6 + self.scale_shift_table
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6)

        # 注意力分支
        attn_in = (norm1(x) * scale_msa + shift_msa)
        attn_out = self.attn(attn_in, rotary_emb, ...)
        x = x + gate_msa * norm_post_attn(attn_out)

        # FFN 分支（MoE 或 Dense）
        ffn_in = (norm2(x) * scale_mlp + shift_mlp)
        if isinstance(self.ffn, SparseMoeBlock):
            ffn_out = self.ffn(ffn_in, ...)
        else:
            ffn_out = self.ffn(ffn_in)
        x = x + gate_mlp * norm_post_ffn(ffn_out)
```

**MoE 层选择**：
```python
if layer_idx not in mlp_only_layers and (num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0):
    self.ffn = SparseMoeBlock(...)  # MoE 层
else:
    self.ffn = MLP(...)             # 密集层
```

---

![ACWM Architecture](../../流式视频生成/assets/lingbot-video/acwm_page-linbot-video.jpg)

---

## 六、MoE 后端实现

代码支持多种 MoE 执行后端：

| 后端 | 说明 |
|------|------|
| `grouped_mm` | PyTorch `_grouped_mm`（默认） |
| `sglang_triton` | SGLang Triton 融合专家 |
| `sglang_triton_fp8` | SGLang FP8 量化专家 |

### Token 重排 → 专家计算 → Token 恢复

```python
# 1. 重排 token 按专家分组
permuted_tokens, counts, sorted_positions, sorted_scores = self._reorder_tokens(...)

# 2. 分组专家计算（grouped_mm 或 for-loop）
expert_output = self._run_grouped_experts(permuted_tokens, counts)

# 3. 恢复 token 顺序
return self._restore_tokens(expert_output, sorted_positions, sorted_scores, ...)
```

---

## 七、级联 Refiner

### 7.1 设计

```
Base (480p) → 像素空间上采样 → VAE 编码 → Refiner → 1080p
```

### 7.2 条件 Rectified Flow

不是从纯噪声去噪，而是从**退化条件** $\mathbf{x}_{\text{lr}}$ 开始：

$$\mathbf{x}_t = \left(1-\frac{t}{\tau}\right)\mathbf{x}_0 + \frac{t}{\tau}\mathbf{x}_\tau$$

$$v^*_{\text{ref}} = \frac{\mathbf{x}_\tau - \mathbf{x}_0}{\tau}$$

- $\tau \sim \text{Uniform}(0.85, 0.95)$：最大噪声水平
- Refiner 专注于恢复高频细节、锐化纹理、纠正局部伪影

---

## 八、模型变体

| 变体 | 总参数 | 活跃参数 | 专家数 | 每 token 选中 |
|------|--------|---------|--------|-------------|
| Dense 1.3B | 1.3B | 1.3B | - | - |
| MoE 13B-A1.4B | 13B | 1.4B | 128 | 8 |
| MoE 30B-A3B | 30B | 3B | - | - |
| MoE 60B-A6B | 60B | 6B | - | - |
| MoE 120B-A11B | 120B | 11B | - | - |

**实验结论**：
- MoE 13B-A1.4B ≈ Dense 3B 的活跃参数，但性能远超
- MoE 30B-A3B ≈ Dense 14B 的性能
- 1M tokens 时，MoE 30B 比 Dense 30B 快 3.18×

---

## 九、推理优化关键点

### 9.1 FP32 敏感模块

```python
LINGBOT_VIDEO_FP32_MODULES = (
    "time_embedder", "time_modulation", "scale_shift_table",
    "norm", "norm1", "norm2", "norm_q", "norm_k",
    "norm_post_attn", "norm_post_ffn", "norm_out", "norm_out_modulation",
    "router",
)
```

这些模块在混合精度下保持 FP32。

### 9.2 融合 QKV

```python
if os.environ.get("LINGBOT_FUSED_QKV_LINEAR") == "1":
    weight = torch.cat((self.to_q.weight, self.to_k.weight, self.to_v.weight), dim=0)
    qkv = F.linear(x, weight, bias)
    q, k, v = qkv.view(B, S, 3, H, D).unbind(2)
```

### 9.3 MoE 后端选择（环境变量）

| 变量 | 选项 | 说明 |
|------|------|------|
| `LINGBOT_MOE_EXPERT_BACKEND` | `grouped_mm`, `sglang_triton`, `sglang_triton_fp8` | 专家计算后端 |
| `LINGBOT_MOE_PAD_BACKEND` | `loop`, `vectorized` | Token 填充后端 |
| `LINGBOT_MOE_REORDER_BACKEND` | `sort`, `triton_pack` | Token 重排后端 |
| `LINGBOT_MOE_RESTORE_BACKEND` | `scatter`, `chunked_scatter`, `triton` | Token 恢复后端 |

### 9.4 Ulysses 序列并行

```python
# All-to-All 分发
q_global = _all_to_all_split_cat(q, scatter_dim=2, gather_dim=1, group=group)
# 本地注意力
out = flash_attn_varlen_func_v3(q_flat, k_flat, v_flat, ...)
# All-to-All 收集
out = _all_to_all_split_cat(out_global, scatter_dim=1, gather_dim=2, group=group)
```

---

## 十、代码结构

```
lingbot-video/
├── lingbot_video/
│   ├── transformer_lingbot_video.py  # 核心 Transformer + MoE
│   ├── pipeline_lingbot_video.py     # T2V 推理管道
│   ├── pipeline_lingbot_video_i2v.py # I2V 推理管道
│   ├── inference_backend.py          # 推理后端抽象
│   ├── native_backend.py             # 原生 PyTorch 后端
│   ├── fsdp_inference.py             # FSDP 推理
│   ├── scheduling_flow_unipc.py      # FlowUniPC 调度器
│   ├── moe_pack_kernels.py           # Triton MoE 打包核
│   ├── moe_restore_kernels.py        # Triton MoE 恢复核
│   └── sglang_moe_shim.py            # SGLang MoE 适配
└── rewriter/                         # 提示重写
```

---

## 十一、与 Cosmos3 的对比

| | LingBot-Video | Cosmos3 |
|---|---|---|
| 架构 | **单流** DiT | **双流** MoT |
| FFN | **Sparse MoE** (128 专家) | 密集 GatedMLP |
| 参数扩展 | 容量-计算解耦 | 固定活跃参数 |
| 位置编码 | 3D MM-RoPE | 3D mRoPE + 15000 边距 |
| 条件注入 | 统一序列拼接 | AR/DM 分离 |
| 调制 | AdaLN-Single | AdaLN |
| 超分 | 级联 Refiner | 无 |

---

## 十二、与你研究方向的关联

1. **单流 vs 双流**：MFU 优化的架构选择
2. **Sparse MoE**：容量-计算解耦，128 专家 top-8
3. **grouped_mm**：PyTorch 原生 MoE 计算
4. **SGLang MoE 后端**：Triton 融合专家核
5. **FP8 量化**：sglang_triton_fp8 后端
6. **Ulysses SP**：All-to-All 序列并行
