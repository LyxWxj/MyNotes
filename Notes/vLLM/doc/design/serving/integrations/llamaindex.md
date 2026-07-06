---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/integrations/llamaindex.md
---

# LlamaIndex Integration

vLLM也可通过[LlamaIndex](https://github.com/run-llama/llama_index)使用。

## 安装

```bash
pip install llama-index-llms-vllm -q
```

## 使用

在单个或多个GPU上运行推理，使用`llamaindex`中的`Vllm`类：

```python
from llama_index.llms.vllm import Vllm

llm = Vllm(
    model="microsoft/Orca-2-7b",
    tensor_parallel_size=4,
    max_new_tokens=100,
    vllm_kwargs={"gpu_memory_utilization": 0.5},
)
```

## 相关链接

- [LlamaIndex vLLM教程](https://docs.llamaindex.ai/en/latest/examples/llm/vllm/)
