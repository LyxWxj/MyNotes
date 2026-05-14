---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Fusion torch.compile Passes

## 核心思想

vLLM 通过 `torch.compile` 的自定义 Inductor 编译 pass，在**编译时**将多个相邻的算子/内核融合为一个，减少中间结果的显存读写开销。这些融合与模型代码解耦，由 `PassConfig` 统一控制，按优化级别自动启用。

## 所有融合一览

| 融合 | 控制标志 | 融合内容 | 默认级别 | 加速 | 适用场景 |
|------|----------|----------|----------|------|----------|
| **AllReduce + RMSNorm** | `fuse_allreduce_rms` | AllReduce → RMSNorm (+残差加) → 可选量化 | O2 (Hopper/Blackwell, TP>1) | 5-20% | 小 token 数 |
| **Attention + Quant** | `fuse_attn_quant` | Attention 输出 → FP8/NVFP4 量化 | 默认关闭 | 3-7% | 始终 |
| **MLA Attention + Quant** | `fuse_attn_quant` | MLA Attention 输出 → FP8/NVFP4 量化 | 默认关闭 | TBD | 始终 |
| **RoPE + KV-Cache** | `fuse_rope_kvcache` | 旋转位置编码 → KV cache 写入 | O2 (ROCm/AITER) | 2-4% | 小 token 数 |
| **QK Norm + RoPE** | `enable_qk_norm_rope_fusion` | Q/K RMSNorm → RoPE | 默认关闭 | 2-3% | 小 token 数 |
| **Sequence Parallelism** | `enable_sp` | AllReduce → ReduceScatter + AllGather | 默认关闭 | AsyncTP 前置条件 | 大 token 数 |
| **AsyncTP GEMM + Collective** | `fuse_gemm_comms` | GEMM → reduce-scatter/all-gather → GEMM | 默认关闭 | 7-10% | 大 token 数 |
| **RMSNorm + Quant** | `fuse_norm_quant` | RMSNorm (+残差加) → FP8/FP4 量化 | O1 (条件性) | 1-4% | 始终 |
| **SiLU+Mul + Quant** | `fuse_act_quant` | SiLU+Mul 激活 → FP8/FP4 量化 | O1 (条件性) | 1-4% | 始终 |
| **RMSNorm + Padding** | `fuse_act_padding` | 残差加 + RMSNorm → Padding | O1 (ROCm/AITER) | TBD | 始终 |

## 逐个详解

### AllReduce + RMSNorm (`fuse_allreduce_rms`)

将张量并行的 AllReduce 集合通信与后续的残差加、RMSNorm、可选量化步骤融合为一个 FlashInfer/TRT-LLM 通信内核。小 token 数场景下有效。

支持模式：
- `AllReduce → RMSNorm(+残差加)` — Hopper/Blackwell + FlashInfer
- `AllReduce → RMSNorm(+残差加) → FP8 static` — Hopper+
- `AllReduce → RMSNorm(+残差加) → NVFP4 dynamic` — Blackwell+

### Attention + Quant (`fuse_attn_quant`)

将 Attention 输出的量化直接融合到 Attention 计算中，**消除一次全精度的显存读写**。支持标准 Attention 和 MLA Attention（DeepSeek-V2/V3/R1）。需要全图可见。MLA 当前尚无加速。

### RoPE + KV-Cache Update (`fuse_rope_kvcache`)

将旋转位置编码计算与 KV cache 的 scatter/write 融合，避免对 key/value 张量的分别读写。仅 ROCm/AITER，默认 `num_tokens <= 256`。

### QK Norm + RoPE (`enable_qk_norm_rope_fusion`)

将 `split QKV → reshape → Q/K RMSNorm → reshape → RoPE` 整个链路融合为单个 CUDA 内核。适用于 Qwen 等模型。

### Sequence Parallelism (`enable_sp`)

将 AllReduce 替换为 `ReduceScatter → local RMSNorm → AllGather`。不是直接性能优化，而是 **AsyncTP 的前置条件**。

### AsyncTP GEMM + Collective (`fuse_gemm_comms`)

在 SP 变换之后，利用对称内存原语将 GEMM 与 reduce-scatter/all-gather **重叠执行**。大 token 数场景才有效。

### RMSNorm + Quant (`fuse_norm_quant`)

将 `rms_norm` / `fused_add_rms_norm` 与后续量化融合。NVIDIA 上 Inductor 自己生成的融合内核更快，只在使用自定义内核时启用。

### SiLU+Mul + Quant (`fuse_act_quant`)

将 gate-up 投影的 `silu_and_mul` 激活函数与后续量化融合，避免全精度 post-activation 张量的物化。

### RMSNorm + Padding (`fuse_act_padding`)

将残差加 + RMSNorm 与 padding 操作融合。仅 ROCm/AITER，针对 GPT-OSS 模型。

## 硬件支持矩阵

- **Blackwell (SM100)**：支持最多，包括 NVFP4
- **Hopper (SM90)**：大部分融合支持
- **Ada (SM89)**：部分支持
- **Ampere (SM80)**：仅 QK Norm+RoPE
- **ROCm**：RoPE+KVCache、RMSNorm+Padding、RMSNorm+Quant、SiLU+Quant、Attention+Quant

## 启用方式

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

llm = LLM(
    model="...",
    optimization_level=2,
    compilation_config=CompilationConfig(
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_allreduce_rms=False,
        )
    ),
)
```

命令行：`vllm serve model -O2 -cc.pass_config.fuse_allreduce_rms=False`

## 一句话总结

Fusion passes 是 vLLM 在 `torch.compile` 编译管线中注入的图优化 pass，通过模式匹配将相邻的计算和通信算子融合为单个内核，减少显存读写和内核启动开销，从而在不修改模型代码的情况下获得 1-20% 的端到端加速。
