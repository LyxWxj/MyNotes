---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/architecture_overview.md
---

# Architecture Overview

vLLM-Omni是vLLM的扩展版本，专为全模态（Omni-Modality）模型推理和服务引擎设计。

## 设计目标

vLLM-Omni的核心目标是构建最快、最易用的开源全模态模型推理与服务引擎：

- **非文本输出**：支持图像、音频、视频等多种数据类型的集成、高效处理和输出
- **非自回归结构**：支持超越自回归的模型结构，特别是Diffusion Transformer（DiT）
- **与vLLM核心集成**：保持兼容性，利用现有vLLM关键模块和优化
- **可扩展性**：设计模块化、灵活的架构，便于容纳新模态、模型架构和输出格式

## 代表性全模态模型

大多数流行开源模型采用AR+DiT组合，可分为三类：

1. **DiT作为主结构，AR作为文本编码器**（如Qwen-Image）
   - 强大的图像生成基础模型，支持复杂文本渲染和精确图像编辑

2. **AR作为主结构，DiT作为多模态生成器**（如BAGEL）
   - 统一的多模态理解和生成模型，支持思维链文本输出和视觉生成

3. **AR+DiT并行**（如Qwen-Omni）
   - 原生端到端全模态LLM，支持多模态输入（文本/图像/音频/视频）和输出（文本/音频）

## 主要架构组件

| 组件 | 描述 |
|------|------|
| **OmniRouter** | 为全模态请求提供智能路由分发 |
| **EntryPoints** | 定义离线/在线服务的API（APIServer、Omni/AsyncOmni），AsyncOmniEngine和Orchestrator协调多阶段AR/DiT执行 |
| **AR** | 适配全模态模型，继承vLLM的高效特性（如缓存管理） |
| **Diffusion** | 原生实现并使用加速组件进行优化 |
| **OmniConnector** | 基于E/P/D/G（编码/处理/解码/生成）阶段分离支持完全解聚 |

## 主要特性

### 性能和加速

- **高效AR支持**：继承vLLM的高效KV缓存管理
- **流水线执行**：使用流水线阶段执行重叠确保高吞吐量
- **完全解聚**：依赖OmniConnector和跨阶段的动态资源分配
- **Diffusion加速**：
  - 缓存：DBCache、TeaCache、第三方集成（如cache-dit）
  - 并行：TP、CP、USP、CFG
  - 注意力：第三方集成接口（如FA3、SAGE、MindIE-SD）
  - 量化：FP8、AWQ
  - 融合操作：自定义和第三方集成

### 无分类器引导（CFG）伴随流

vLLM-Omni原生建模CFG，通过"伴随请求"范式消除冗余的文本/多模态上下文计算：

1. **提示扩展**：在初始AR阶段，自定义的`prompt_expand_func`钩子拦截传入的生成提示，动态配对负伴随提示
2. **同步KV缓存传输**：AR阶段同时评估主序列和伴随序列批次，OmniConnector通过共享内存或网络协议跨阶段边界传递正负结果KV缓存
3. **KV缓存收集与注入**：在下游Diffusion引擎中，`cfg_kv_collect_func`自动拦截映射的伴随缓存

### 灵活性和易用性

- 异构管道抽象
- Hugging Face集成
- 分布式推理（张量、管道、数据、专家并行）
- 流式输出
- 统一API接口（兼容vLLM）
- OpenAI兼容API服务器

## 接口设计

### 离线推理

```python
from vllm_omni.entrypoints.omni import Omni

omni = Omni(model="Qwen/Qwen3-Omni-30B-A3B-Instruct")

om_inputs = {"prompt": prompt,
             "multi_modal_data": {
                 "video": video_frames,
                 "audio": audio_signal,
             }}

outputs = omni.generate(om_inputs, sampling_params_list)
```

### 在线服务

启动服务器：
```bash
vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni --port 8091
```

发送请求示例：
```bash
curl -sS -X POST http://localhost:8091/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d @- <<EOF
{
  "model": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
  "sampling_params_list": $sampling_params_list,
  "messages": [
    {
      "role": "user",
      "content": $user_content
    }
  ]
}
EOF
```

## 参考链接

- [vLLM-Omni GitHub](https://github.com/vllm-project/vllm-omni)
- [示例代码](https://github.com/vllm-project/vllm-omni/tree/main/examples)
