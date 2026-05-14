---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Custom Logits Processors

## 概述

Custom Logits Processors 允许用户定义自定义的 logits 变换逻辑，在采样前对模型输出的 logits 进行修改，实现高级控制（如强制格式、禁止特定 token 等）。

## 核心概念

### LogitsProcessor 类

用户需要继承 `LogitsProcessor` 基类并实现 `__call__` 方法：

```python
from vllm.logits_processor import LogitsProcessor

class MyProcessor(LogitsProcessor):
    def __call__(self, logits, tokens):
        # 自定义 logits 变换逻辑
        return modified_logits
```

### BatchUpdate 机制

- 支持批量处理多个请求的 logits
- 通过 `BatchUpdate` 同步批次状态
- 确保批处理效率

## 加载方式

### 1. 完全限定类名 (FQCN)

```bash
vllm serve <model> --logits-processors my_module.MyProcessor
```

### 2. Entry Points

通过 Python package 的 entry_points 注册：

```toml
[project.entry-points."vllm.logits_processors"]
my_processor = "my_module:MyProcessor"
```

### 3. 类对象直接传递

```python
llm = LLM(model="...", logits_processors=[MyProcessor()])
```

## 典型应用场景

- **格式约束**：强制 JSON、XML 等输出格式
- **内容过滤**：禁止生成特定词汇
- **语言控制**：限制输出语言
- **置信度校准**：调整输出概率分布

## 注意事项

- 处理器应尽量高效，避免成为瓶颈
- 确保处理器的数值稳定性
- 在流式输出中，处理器需正确处理增量 token
