---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# LoRA Adapters

## 概述

LoRA (Low-Rank Adaptation) 允许在基础模型上高效加载和切换微调适配器，支持按请求级别的 LoRA 切换，开销极小。

## 基本使用

### 离线推理

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
)
```

### 在线服务

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

请求中通过 `model` 参数指定 LoRA 适配器名称。

## 动态 LoRA 加载

需要设置环境变量：`VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`

### API 端点

- `POST /v1/load_lora_adapter`：动态加载 LoRA
- `POST /v1/unload_lora_adapter`：卸载 LoRA

### Plugin 方式

- **lora_filesystem_resolver**：从本地目录加载
- **lora_hf_hub_resolver**：从 Hugging Face Hub 加载
- 支持自定义 LoRAResolver 插件（如 S3）

### In-Place 重载

`load_inplace=True` 可替换同名适配器，适用于异步 RL 场景中持续更新适配器。

## 高级配置

### max_lora_rank

设置为所有适配器中的最大 rank，避免设置过高浪费内存。

```bash
vllm serve model --enable-lora --max-lora-rank 64
```

### lora-target-modules

限制 LoRA 应用的模块：

```bash
vllm serve model --enable-lora --lora-target-modules o_proj qkv_proj
```

## 多模态 LoRA

### Tower 和 Connector 支持

实验性支持多模态模型的 Tower 和 Connector 组件的 LoRA。

### Default Multimodal LoRAs

为特定模态注册默认 LoRA，当输入包含该模态时自动应用：

```python
llm = LLM(
    model=model_id,
    enable_lora=True,
    default_mm_loras={"audio": model_id},
)
```

## 模型卡片展示

LoRA 模型在 `/models` 端点中显示 `parent`（基础模型）和 `root`（适配器路径）字段。
