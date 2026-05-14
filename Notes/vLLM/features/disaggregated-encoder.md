---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Disaggregated Encoder

## 概述

Disaggregated Encoder（分离式编码器）将多模态 LLM 的视觉编码器阶段与预填充/解码阶段分离到不同进程中运行，带来三大优势：

1. **独立细粒度扩展**：编码器轻量，语言模型庞大，可独立扩缩容
2. **降低 TTFT**：纯语言请求可绕过编码器，缩短预填充关键路径
3. **跨进程复用与缓存**：远程共享缓存允许任何 worker 复用已有 embedding

## 架构

### 两阶段分离

- **Encoder Instance**：独立的 vLLM 实例，专门执行视觉编码
- **PD Instance**：运行语言预填充和解码
  - 可以是单个实例（E→PD 模式）
  - 也可以分离的 Prefill + Decode 实例（E→P→D 模式）

### 关键抽象

- **ECConnector**：负责从编码器检索 EC (Encoder-Cache) 缓存
  - *Scheduler 角色*：检查缓存存在性并调度加载
  - *Worker 角色*：将 embedding 加载到内存

### 数据流

1. Encoder 实例处理视觉输入，生成 embedding 缓存
2. ECConnector 将 embedding 传输到 PD 实例
3. PD 实例在注意力层注入 encoder 输出，完成语言生成

## 使用示例

```bash
# 1 Encoder + 1 PD 实例
examples/online_serving/disaggregated_encoder/disagg_1e1pd_example.sh

# 1 Encoder + 1 Prefill + 1 Decode 实例
examples/online_serving/disaggregated_encoder/disagg_1e1p1d_example.sh
```

## 实现细节

- 所有代码位于 `vllm/distributed/ec_transfer`
- 与 NixlConnector 配合实现 P→D 的 KV Cache 传输
