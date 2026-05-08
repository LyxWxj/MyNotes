# 娄雨轩 Daily Report — 2026-05-08

## VS Code Server glibc 兼容性问题解决

### 背景

延续昨日 VS Code Remote SSH 连接华为云 ModelArts 服务器的问题。核心矛盾在于 VS Code 1.86 版本的 Remote 模块要求 glibc 2.28 及以上，而旧版本服务器（如 Ubuntu 18.04）的系统 glibc 版本低于此要求。直接升级系统 glibc 是极其危险的操作，可能导致整个系统崩溃。

### 解决方案：patchelf 手动指定动态库

利用 `patchelf` 工具手动修改 vscode-server 的 `node` 二进制文件的动态链接器和运行时库搜索路径，避免重新编译系统 glibc。

#### Step 1：下载 glibc 动态链接库

使用 GitHub 上的 `glibc-all-in-one` 仓库下载所需版本的 glibc：

```bash
# 查看支持的版本
cat list

# 下载指定版本（以 2.31 为例）
./download 2.31-0ubuntu9.14_amd64
```

下载完成后，当前文件夹下会生成 `libs` 文件夹，包含所下载版本的动态库。

如需编译生成完整 glibc 目录：

```bash
./build 2.31 arm64
```

该命令会在根目录下生成 `/glibc` 文件夹，可移动至 `glibc-all-in-one` 文件夹中。

#### Step 2：使用 patchelf 修改 vscode-server 依赖

1. 先删除 `~/.vscode-server` 文件夹，用 VS Code 重新连接服务器，让它自动下载 vscode-server。此时 `~/.vscode-server/bin` 下应只有一个由数字和字母组成的随机字符串文件夹（如 `863d2581ecda6849923a2118d93a088b0745d9d6`）
2. 进入该文件夹，找到 `node` 二进制文件
3. 执行 patchelf 命令：

```bash
patchelf \
  --set-interpreter ~/pack/glibc-all-in-one/libs/2.31-0ubuntu9.14_amd64/ld-linux-x86-64.so.2 \
  --set-rpath ~/pack/glibc-all-in-one/libs/2.31-0ubuntu9.14_amd64/:~/pack/glibc-all-in-one/glibc/2.31/amd64/lib \
  --force-rpath \
  ~/.vscode-server/bin/<your-hash>/node
```

**参数说明**：

| 参数 | 作用 | 路径 |
|------|------|------|
| `--set-interpreter` | 指定动态链接器路径 | `glibc-all-in-one/libs/[version]/ld-linux-x86-64.so.2` |
| `--set-rpath` | 设置运行时库搜索路径 | `libs/[version]` + `glibc/[version]/[arch]/lib` |
| `--force-rpath` | 强制覆盖已有 rpath | 目标可执行文件路径 |

**常见问题**：若报错 `patchelf: open: Text file busy`，需先关闭本机 VS Code 的远程连接，再执行命令。

## QwenImage 多模态生成模型架构分析

### 模型 Pipeline 架构

QwenImage 采用经典的三阶段多模态生成架构：

1. **Text Encoder**（Qwen2.5-7B）：将文本 prompt 编码为 hidden states 作为 diffusion 的条件输入
2. **Diffusion Transformer**（QwenImageTransformer2DModel）：60 层 dual-stream DiT，执行 50 步去噪
3. **VAE Decoder**（AutoencoderKLQwenImage）：将 latent 空间解码为像素空间图像

### 源码分析

#### Text Encoder：为何使用 HuggingFace 实现而非 vllm

`pipeline_qwen_image.py` 中 Text Encoder 的加载方式：

```python
from transformers import Qwen2_5_VLForConditionalGeneration
self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model, subfolder="text_encoder", ...
)
```

实际推理时仅传入 `input_ids` 和 `attention_mask`，只用到 language model 部分的 hidden states，视觉编码器完全未参与计算。

**为何不用 vllm 的 `Qwen2_5VLForConditionalGeneration`**：

- vllm 的模型类为自回归 serving 设计，依赖 `VllmConfig`、分布式组（`get_pp_group()`）、PagedAttention 等基础设施，无法直接作为 `nn.Module` 嵌入 diffusion pipeline
- `init_vllm_registered_model`（`model_executor/models/utils.py`）的工作流程：根据 `architectures` 参数从 vllm 注册表解析模型类 → `initialize_model` 实例化 → `DefaultModelLoader.load_weights` 加载权重，整个链路都需要完整的 vllm engine 上下文
- Text Encoder 仅做单次前向提取特征，不需要 vllm 的 KV-cache、continuous batching 等优化，HuggingFace transformers + Flash Attention 2 已足够高效

**可优化点**：当前加载完整的 `Qwen2_5_VLForConditionalGeneration`（含视觉编码器），可改为仅加载 `Qwen2ForCausalLM`（`subfolder="text_encoder"` 的 config 中 architectures 即为 `Qwen2ForCausalLM`），避免加载无用的视觉编码器权重。

#### Qwen2Model 结构（vllm 实现）

`qwen2.py` 中的 `Qwen2Model` 是标准 Transformer decoder，包含三个核心部分：

- **`embed_tokens`**（`VocabParallelEmbedding`）：词嵌入层，支持 pipeline parallel，仅第一 pp rank 加载
- **`layers`**（N 层 `Qwen2DecoderLayer`）：每层含 QKV 投影（`qkv_proj`）、O 投影（`o_proj`）、SwiGLU MLP（`gate_up_proj` + `down_proj`）。支持 tensor parallel（QKV/O 为 ColumnParallel/RowParallelLinear）。每层 forward 为 `hidden_states, residual = layer(positions, hidden_states, residual)`
- **`norm`**（`RMSNorm`）：最终归一化，仅最后一个 pp rank

forward 流程：`embed_tokens → N 层 decoder layers → RMSNorm → hidden_states`。不包含 lm_head，这正是 diffusion pipeline 可将其用作 text encoder 的原因——只需中间层特征，无需 token 预测。

#### Diffusion Transformer 源码结构

`qwen_image_transformer.py` 中的核心类：

**`QwenImageTransformer2DModel`**：继承 `CachedTransformer`，包含：
- `img_in` / `txt_in`：输入投影（ReplicatedLinear）
- `time_text_embed`：时间步 + 文本条件嵌入
- `transformer_blocks`：60 层 `QwenImageTransformerBlock`
- `norm_out` + `proj_out`：输出归一化和投影
- 支持 Sequence Parallelism（`_sp_plan`）和 CFG Parallel（`QwenImageCFGParallelMixin`）

**`QwenImageTransformerBlock`**（dual-stream 设计）：
- **Image stream**：`img_mod`（SiLU + Linear dim→6dim，生成调制参数）→ `img_norm1`（AdaLayerNorm）→ attention → `img_norm2` → `img_mlp`（FeedForward，dim→4dim→dim，GELU-approximate）
- **Text stream**：`txt_mod` → `txt_norm1` → 共享 attention → `txt_norm2` → `txt_mlp`
- **Joint attention**（`QwenImageCrossAttention`）：image 和 text 各自计算 QKV，concat 后做联合 attention，再 split 回各自 stream。使用 QK Norm（RMSNorm）+ RoPE

**`FeedForward`**：`ColumnParallelApproxGELU`（dim→4dim）→ `RowParallelLinear`（4dim→dim），mult=4，inner_dim=12288

#### VAE Decoder 源码结构

`autoencoder_kl_qwenimage.py` 中的 `QwenImageDecoder3d`：

- **conv_in**：`QwenImageCausalConv3d`（z_dim=16 → 384），3D 因果卷积，支持时间维度缓存
- **mid_block**：`QwenImageMidBlock`（384ch），含 1 个 ResBlock + 1 个 AttentionBlock + 1 个 ResBlock
- **up_blocks**（3 个）：逐步上采样并降通道（384→192→96），每个 block 含 `num_res_blocks+1` 个 `QwenImageResidualBlock`（两个 3×3 CausalConv3d + RMSNorm + SiLU + 残差连接）
- **conv_out**：96 → 3（RGB 输出）

**`QwenImageAttentionBlock`**：单头自注意力，`to_qkv`（Conv2d 1×1）→ `scaled_dot_product_attention` → `proj`（Conv2d 1×1），在空间维度上做 attention

**显存优化特性**：支持 `feat_cache` 机制的因果 3D 卷积，用于视频生成时的帧间缓存；支持 spatial tiling 降低大图解码的显存峰值

### 各组件计算量与参数量分析

#### Text Encoder（Qwen2.5-7B）

- 参数量：~7.0B（embed_tokens 545M + 28 层 Transformer 6.5B）
- 架构：标准 Transformer decoder（hidden_size=3584, intermediate_size=18944, 28 heads, 4 KV heads）
- 单次前向 FLOPs：~6.3 PFLOPs
- 关键设计：仅使用 hidden_states[-1] 提取文本特征，不需要 lm_head，不涉及自回归生成

#### Diffusion Transformer（QwenImageTransformer2DModel）

- 参数量：~38.8B（60 层 dual-stream block，每层 ~641M）
- 架构：dual-stream Transformer（image stream + text stream 独立 MLP/Modulation，共享 joint attention）
  - dim = 3072, num_heads = 24, head_dim = 128, FeedForward inner_dim = 12288
  - 每层包含：img_mod, txt_mod（条件调制）, joint cross-attention, img_mlp, txt_mlp
- 每步 FLOPs：~14.1 PFLOPs（1024×1024 图片，S_img=4096, S_txt=512）
- 50 步总 FLOPs：~703 PFLOPs
- 显存占用：~44GB（含权重、激活值、KV cache）

#### VAE Decoder（AutoencoderKLQwenImage）

- 参数量：~0.16B（base_dim=96, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2）
- 架构：纯卷积 3D 解码器（Conv3d + ResBlock + AttentionBlock）
- 单次前向 FLOPs：~0.7 PFLOPs（512×512 输出）

#### 计算量占比

| 组件 | 参数量 | FLOPs | 占比 |
|------|--------|-------|------|
| Text Encoder | 7.0B | 6.3 PFLOPs | ~0.9% |
| VAE Decoder | 0.16B | 0.7 PFLOPs | ~0.1% |
| Diffusion Transformer | 38.8B | 703 PFLOPs (50步) | **~99%** |

Diffusion Transformer 的去噪循环是绝对瓶颈，Text Encoder 和 VAE Decoder 的开销可忽略。

### 华为昇腾芯片算力对比

| 芯片 | BF16 算力 (TFLOPS) | 显存 | 定位 |
|------|-------------------|------|------|
| 昇腾 310P | ~128 | 24GB | 推理 |
| 昇腾 910B | ~320 | 64GB HBM2e | 训练/推理 |
| 昇腾 910C | ~320 | 64/96GB HBM2e | 训练/推理 |
| NVIDIA A100 | 312 | 80GB HBM2e | 训练/推理 |
| NVIDIA H100 | 989 | 80GB HBM3 | 训练/推理 |

### 硬件流水线编排方案

基于三级筛选机制进行硬件匹配：

**第一级：显存约束筛选**

| 组件 | 峰值显存 | 可用硬件 |
|------|---------|---------|
| Text Encoder | ~18GB | 310P(24G), 910B(64G), 910C(64/96G) |
| Diffusion | ~44GB | 910B(64G), 910C(64/96G) |
| VAE Decoder | ~1GB | 全部 |

**第二级：算力匹配筛选**

| 组件 | FLOPs | 310P (128T) | 910B (320T) |
|------|-------|------------|------------|
| Text Encoder | 6.3P | 49ms (匹配) | 20ms (过剩) |
| Diffusion | 703P | 5.5s (不足) | 2.2s (匹配) |
| VAE Decoder | 0.7P | 5.5ms (匹配) | 2.2ms (过剩) |

**第三级：最优编排**

| 组件 | 分配硬件 | 理由 |
|------|---------|------|
| Text Encoder | 310P | 显存满足，算力匹配，避免浪费高端卡 |
| Diffusion | 910B | 显存满足（44G < 64G），算力最佳匹配 |
| VAE Decoder | 310P（与 Text Encoder 分时复用） | 显存满足，计算量极小 |

总硬件：1×910B + 1×310P，流水线调度下 batch=15 时吞吐约 6 req/s。

### 专利方案撰写

多模态推理流水线硬件匹配专利，三级硬件匹配方法论：

1. **显存约束筛选**：$M_i^{peak} = M_i^{weight} + M_i^{act}(B) + M_i^{cache}$，筛选满足显存阈值的候选硬件
2. **算力匹配筛选**：$T_i = FLOPs_i / (P_j^{peak} \times \eta_j)$，选取算力冗余比 $R \approx 1$ 的硬件
3. **流水线全局优化**：$\Theta = \min_i (B_i / T_i)$，在显存约束下求解最优 stage-to-hardware 分配
