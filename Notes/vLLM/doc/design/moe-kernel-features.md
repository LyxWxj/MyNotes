---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# MoE Kernel Features

## 概述

vLLM 中有多种 MoE 内核（模块化和非模块化），本文档帮助根据场景选择合适的内核组合。

## All2All 通信后端

All2All 后端实现专家并行（EP）的 `FusedMoE` 层通信。各后端通过 `FusedMoEPrepareAndFinalizeModular` 子类提供接口。

| 后端 | 输出格式 | 量化类型 | 量化格式 | 异步 | Weight On Input | 子类 |
|------|----------|----------|----------|------|-----------------|------|
| **naive** | standard | all | G,A,T | N | 依实现 | `FusedMoE` |
| **deepep_high_throughput** | standard | fp8 | G(128),A,T | Y | Y | `DeepEPHTPrepareAndFinalize` |
| **deepep_low_latency** | batched | fp8 | G(128),A,T | Y | Y | `DeepEPLLPrepareAndFinalize` |
| **flashinfer_nvlink_two_sided** | standard | nvfp4,fp8 | G,A,T | N | N | `FlashInferNVLinkTwoSidedPrepareAndFinalize` |
| **flashinfer_nvlink_one_sided** | standard | nvfp4 | G,A,T | N | N | `FlashInferNVLinkOneSidedPrepareAndFinalize` |

格式说明：G=Grouped, G(N)=Grouped block size N, A=Per activation token, T=Per tensor

异步后端支持 DBO 和 shared expert overlap。通过 `--all2all-backend` 控制。

## 专家内核

| 内核 | 输入格式 | 量化类型 | 量化格式 | 激活函数 | Weight On Input | 模块化 |
|------|----------|----------|----------|----------|-----------------|--------|
| **triton** | standard | all | G,A,T | silu,gelu,swigluoai 等 | Y | Y |
| **triton (batched)** | batched | all | G,A,T | silu,gelu | - | Y |
| **deep gemm** | standard/batched | fp8 | G(128),A,T | silu,gelu | - | Y |
| **cutlass_fp4** | standard/batched | nvfp4 | A,T | silu | Y | Y |
| **cutlass_fp8** | standard/batched | fp8 | A,T | silu,gelu | Y | Y |
| **flashinfer** | standard | nvfp4,fp8 | T | SwiGlu | N | Y |
| **gpt oss triton** | standard | N/A | N/A | SwiGlu | Y | Y |
| **marlin** | standard/batched | uint4/uint8/fp8/fp4 | - | silu,swigluoai | Y | Y |
| **trtllm** | standard | mxfp4,nvfp4 | G(16),G(32) | SwiGlu | N | Y |
| **rocm aiter moe** | standard | mxfp4,fp8 | G(32),G(128),A,T | silu,gelu,swigluoai | Y | N |
| **cpu_fused_moe** | standard | N/A | N/A | silu | N | N |
| **naive batched** | batched | int8,fp8 | G,A,T | silu,gelu | - | Y |

## 模块化内核家族

推荐的后端 + 专家组合：

| 后端 | Prepare/Finalize 子类 | 专家子类 |
|------|----------------------|----------|
| deepep_high_throughput | `DeepEPHTPrepareAndFinalize` | `DeepGemmExperts`, `TritonExperts`, `TritonOrDeepGemmExperts`, `CutlassExpertsFp8`, `MarlinExperts` |
| deepep_low_latency | `DeepEPLLPrepareAndFinalize` | `BatchedDeepGemmExperts`, `BatchedTritonExperts`, `CutlassBatchedExpertsFp8`, `BatchedMarlinExperts` |
| flashinfer | `FlashInferCutlassMoEPrepareAndFinalize` | `FlashInferExperts` |

## 兼容性要求

MoE 内核与 Prepare/Finalize 子类配对时需满足：激活格式兼容、量化类型兼容、量化格式兼容。

## 一句话总结

MoE kernel features 文档是 vLLM 中 All2All 后端和专家内核的选型参考，列出各后端/内核的格式、量化、异步支持等属性，以及推荐的模块化内核组合。
