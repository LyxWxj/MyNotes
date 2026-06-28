# Bagel 模型架构分析

## 一、概述

Bagel 是一个**统一多模态模型**，将文本/图像理解和图像生成融合在同一个架构中。核心创新是 **MoT (Mixture-of-Tokens)** 设计——在同一个 LLM 骨干中支持理解模式和生成模式。

代码位置：`vllm-omni/vllm_omni/diffusion/models/bagel/`

## 二、目录结构

| 文件 | 用途 |
|------|------|
| `autoencoder.py` | VAE 变分自编码器，负责图像的编码和解码 |
| `bagel_transformer.py` | 核心 MoT Transformer 架构和去噪逻辑（约 2300 行） |
| `pipeline_bagel.py` | 顶层推理 Pipeline，编排完整推理流程 |

配套集成文件：
- `model_executor/models/bagel/bagel.py` — vLLM 在线服务的 AR 阶段模型
- `model_executor/stage_input_processors/bagel.py` — CFG prompt 展开和 KV cache 收集
- `vllm/transformers_utils/configs/bagel.py` — `BagelConfig` 配置类

## 三、核心架构：Mixture-of-Tokens (MoT)

Bagel 在同一个 Qwen2 LLM 骨干中支持两种运行模式：

- **理解模式 (und)**：使用基础 Q/K/V/MLP 投影，处理文本和 ViT 视觉特征
- **生成模式 (gen)**：使用独立的 `qkv_proj_moe_gen` / `o_proj_moe_gen` / `mlp_moe_gen` 权重矩阵处理 VAE latent tokens

### 整体流程

```
用户 Prompt (+ 可选图像)
    │
    ├─→ Qwen2 Tokenizer + Embedding (文本编码)
    ├─→ SigLIP ViT (视觉理解)
    ├─→ Conv VAE Encoder (图像潜空间编码)
    │
    ▼
┌─────────────────────────────────────────┐
│         Qwen2MoT Language Model         │
│  每层有双权重矩阵: und-mode + gen-mode  │
└─────────────────────────────────────────┘
    │
    ▼
  Flow Matching 去噪循环 (LLM 作为去噪骨干)
    │
    ▼
  VAE Decoder → 生成的图像
```

## 四、LLM 作为去噪网络

### 为什么可以用 LLM 去噪？

传统去噪模型（DiT）的架构本质是 **Transformer + timestep embedding + patch embedding**。Bagel 的 LLM（Qwen2）本身就是一个超大的 Transformer，因此可以直接替换 DiT 中的 Transformer 骨干。

```
DiT:   噪声 latent → patch embed → Transformer → 预测速度
Bagel: 噪声 latent → vae2llm    → Qwen2 LLM  → llm2vae → 预测速度
```

核心思想：**既然已经有了一个巨大的 Transformer（LLM），何必再单独训练一个 DiT？**

### 单个去噪步的计算流程

```
1. 噪声 latent (N 个 patch) → vae2llm → N 个 embedding
2. 文本 token → embed_tokens → M 个 embedding
3. 拼成一个序列: [text_emb × M, noise_emb × N]
4. 送入 Qwen2 LLM (mode="gen", is_causal=False)
   - 文本 token 用 und-mode 权重
   - 噪声 latent 用 gen-mode 权重
5. 取出噪声 latent 对应位置的 hidden states
6. llm2vae 投影回 latent 空间 → 得到速度 v_t
7. x_t = x_t - v_t × dt  (欧拉法更新)
```

每一步都是完整的 LLM 前向传播，LLM 的每一层都在参与去噪方向的预测。

### 去噪循环与 LLM 层数的关系

**LLM 层数 ≠ 去噪步数**，两者完全独立：

- **LLM 层数**（如 28 层）：LLM 做一次前向传播，每层产出 K/V，存入 KV cache。只做一次。
- **去噪步数**（如 50 步）：从纯噪声迭代去噪到干净图像。每步都跑一遍完整的 LLM。

```
LLM (做一次 prefill)              DiT (做 N 次去噪)
┌──────────────────┐             ┌──────────────────┐
│ Layer 1  → KV₁   │             │ Step 1: 噪声→略去噪 │
│ Layer 2  → KV₂   │ 一次性产出  │ Step 2: 再去噪      │
│ ...              │ ─────────→  │ ...               │
│ Layer 28 → KV₂₈  │ KV cache    │ Step 50: 基本干净   │
└──────────────────┘             └──────────────────┘
```

## 五、两种推理模式

### 1. 离线推理 (pipeline_bagel.py)

自包含的端到端 Pipeline，在 vLLM 框架外独立运行。自己加载所有组件，自己跑完整去噪循环。

```python
# pipeline_bagel.py 中自己做 prefill
gen_context["past_key_values"] = NaiveCache(...)
# forward_cache_update_text / vae / vit 逐步构建 KV cache
# 然后调用 generate_image()
```

### 2. 在线服务 (两阶段架构)

拆分为 AR 阶段和 DiT 阶段，利用 vLLM 的调度能力提升吞吐。

```
┌─────────────────────────────────────────┐
│  Stage 0: AR 阶段                       │
│  model_executor/models/bagel/bagel.py   │
│                                         │
│  输入: 文本 prompt (+ 可选图像)          │
│  输出: KV cache (多层 K/V 表示)          │
│  在 vLLM 调度框架中运行                  │
└──────────────────┬──────────────────────┘
                   │ KV Transfer (SharedMemory)
                   ▼
┌─────────────────────────────────────────┐
│  Stage 1: DiT 阶段                      │
│  diffusion/models/bagel/pipeline_bagel.py│
│                                         │
│  输入: KV cache + 随机噪声               │
│  输出: 生成的图像                        │
│  跑 flow matching 去噪循环               │
└─────────────────────────────────────────┘
```

### KV Cache 传递链路

1. AR 阶段在 vLLM 引擎中运行，引擎自动管理 PagedAttention KV cache
2. AR scheduler 检测到触发条件，通过 `kv_transfer_manager` 将 KV block 序列化到共享内存
3. `diffusion_model_runner.execute_model()` 在调用 `pipeline.forward(req)` **之前**，先调用 `kv_transfer_manager.receive_multi_kv_cache_distributed()` 从共享内存读出 KV 数据
4. KV 数据挂载到 `req.sampling_params.past_key_values` 和 `req.sampling_params.kv_metadata`
5. `pipeline_bagel.forward()` 从 `req.sampling_params` 读取注入的 KV cache

关键代码：`pipeline_bagel.py:344-378`

```python
injected_kv = req.sampling_params.past_key_values
if injected_kv is not None:
    gen_context["past_key_values"] = injected_kv
    gen_context["ropes"] = req.sampling_params.kv_metadata["ropes"]
    image_shape = tuple(req.sampling_params.kv_metadata["image_shape"])
    cfg_text_kv = req.sampling_params.cfg_text_past_key_values
    cfg_img_kv  = req.sampling_params.cfg_img_past_key_values
else:
    # 离线模式：自己 prefill
```

## 六、KV Cache 的内容

KV cache 不仅仅是文本编码，而是**经过 LLM 全部层处理后的多模态上下文**。

### text2img（纯文生图）

KV cache 只包含文本编码结果（经 LLM 处理后的 K/V）。

### img2img（图生图）

AR 阶段处理的是拼接的多模态序列（`bagel.py:639`）：

```python
combined = torch.cat([se, vae_embeds, ee, se, vit_emb, ee], dim=0)
```

KV cache 包含：

| 区段 | 内容 | 模式 |
|------|------|------|
| VAE latent patches | 图像的潜空间表示（生成用） | gen-mode |
| ViT features | 图像的语义理解特征 | und-mode |
| Text tokens | 文本 prompt 编码 | und-mode |
| Markers + separator | 结构标记 | und-mode |

传给 DiT 的不是原始 VAE latent 或 ViT feature，而是经过 LLM 全部 transformer 层深度交互后的 K/V 表示。

## 七、与传统模型的区别

| | 传统模型 (如 SD/Qwen-Image) | Bagel |
|---|---|---|
| **文本编码器** | 独立模型（CLIP/T5），通常冻结 | LLM 自己就是编码器 |
| **去噪骨干** | 独立的 DiT / U-Net | LLM 本身 |
| **编码结果** | 固定的 embedding 向量 | LLM 每一层的 K/V 表示 |
| **模态融合** | TextEncoder 只看文本 | LLM 内部文本、ViT、VAE 做 cross-attention |
| **是否端到端** | TextEncoder 冻结，DiT 单独训练 | LLM + DiT 可联合训练 |

传统模型有三个独立组件（TextEncoder + VAE + DiT），Bagel 用一个统一的 LLM 替代了 TextEncoder + DiT 两个组件。

## 八、CFG (Classifier-Free Guidance)

Bagel 的 CFG 使用三个分支：

- **gen**：条件分支（用户 prompt + 图像）
- **cfg_text**：文本无条件分支（空/negative prompt）
- **cfg_img**：图像无条件分支（用户 prompt，无图像）

三个分支各有独立的 KV cache，在去噪循环中分别计算速度，然后通过 `_combine_cfg()` 融合。支持三种并行策略：
- **Batched CFG**：所有分支在单次 LLM forward 中处理
- **CFG Parallel**：每个分支在不同 GPU 上计算
- **SP + CFG**：序列并行 + CFG 分支

## 九、关键类速查

| 类 | 文件 | 职责 |
|---|---|---|
| `AutoEncoder` | autoencoder.py | VAE 编解码 |
| `PackedAttentionMoT` | bagel_transformer.py | 双模式注意力层 |
| `Qwen2MoTDecoderLayer` | bagel_transformer.py | 双 layernorm + 双 MLP 解码层 |
| `Qwen2MoTForCausalLM` | bagel_transformer.py | LLM 因果语言模型包装 |
| `Bagel` | bagel_transformer.py | 中央编排模块，含去噪循环 |
| `BagelPipeline` | pipeline_bagel.py | 顶层推理 Pipeline |
| `OmniBagelForConditionalGeneration` | model_executor/bagel.py | vLLM 在线服务 AR 阶段 |
| `OmniKVTransferManager` | distributed/omni_connectors/kv_transfer_manager.py | KV cache 跨阶段传输 |
