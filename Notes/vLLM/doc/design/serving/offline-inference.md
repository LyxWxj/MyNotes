---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/offline_inference.md
---

# Offline Inference（离线推理）

离线推理可以在自己的代码中使用vLLM的`LLM`类进行。

## 基本用法

```python
from vllm import LLM

# 初始化vLLM引擎
llm = LLM(model="facebook/opt-125m")
```

初始化`LLM`实例后，使用可用API执行模型推理。可用API取决于模型类型：

- [生成模型](../models/generative_models.md)：输出logprobs，从中采样得到最终输出文本
- [池化模型](../models/pooling_models/README.md)：直接输出隐藏状态

## Ray Data LLM API

Ray Data LLM是一个替代的离线推理API，使用vLLM作为底层引擎。此API添加了多项内置功能，简化大规模、GPU高效推理：

| 功能 | 描述 |
|------|------|
| **流式执行** | 处理超过集群聚合内存的数据集 |
| **自动分片** | 自动分片、负载均衡和自动扩展 |
| **连续批处理** | 保持vLLM副本饱和，最大化GPU利用率 |
| **透明并行** | 支持张量和管道并行 |
| **文件格式支持** | 读写大多数流行文件格式和云对象存储 |
| **无代码扩展** | 无需代码更改即可扩展工作负载 |

### 示例

```python
import ray  # 需要ray>=2.44.1
from ray.data.llm import vLLMEngineProcessorConfig, build_llm_processor

config = vLLMEngineProcessorConfig(model_source="unsloth/Llama-3.2-1B-Instruct")
processor = build_llm_processor(
    config,
    preprocess=lambda row: {
        "messages": [
            {"role": "system", "content": "You are a bot that completes unfinished haikus."},
            {"role": "user", "content": row["item"]},
        ],
        "sampling_params": {"temperature": 0.3, "max_tokens": 250},
    },
    postprocess=lambda row: {"answer": row["generated_text"]},
)

ds = ray.data.from_items(["An old silent pond..."])
ds = processor(ds)
ds.write_parquet("local:///tmp/data/")
```

## 相关链接

- [Ray Data LLM文档](https://docs.ray.io/en/latest/data/working-with-llms.html)
- [API参考](../api/README.md#offline-inference)
