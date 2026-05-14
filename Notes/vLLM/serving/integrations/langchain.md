---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm/blob/main/docs/serving/integrations/langchain.md
---

# LangChain Integration

vLLM也可通过[LangChain](https://github.com/langchain-ai/langchain)使用。

## 安装

```bash
pip install langchain langchain_community -q
```

## 使用

在单个或多个GPU上运行推理，使用`langchain`中的`VLLM`类：

```python
from langchain_community.llms import VLLM

llm = VLLM(
    model="Qwen/Qwen3-4B",
    trust_remote_code=True,  # HuggingFace模型必需
    max_new_tokens=128,
    top_k=10,
    top_p=0.95,
    temperature=0.8,
    # 对于分布式推理
    # tensor_parallel_size=...,
)

print(llm("What is the capital of France ?"))
```

## 相关链接

- [LangChain vLLM教程](https://python.langchain.com/docs/integrations/llms/vllm)
