---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/features/custom_pipeline.md
---

# Custom Pipeline Extension Guide

本指南演示如何使用新添加的功能扩展vLLM-Omni的扩散管道。

## 概述

三个主要功能支持自定义管道扩展：

| 功能 | 描述 |
|------|------|
| **`WorkerWrapperBase`** | 包装类，支持使用自定义功能进行动态工作进程扩展 |
| **`load_format`** | 控制扩散模型加载方式的参数，包括对自定义管道的支持 |
| **`CustomPipelineWorkerExtension`** | 扩展类，支持使用自定义实现重新初始化管道 |

## 功能详解

### WorkerWrapperBase

`WorkerWrapperBase`是包装类，创建带有可选扩展支持的`DiffusionWorker`实例。它支持动态继承，允许向工作进程添加自定义方法和功能，而无需修改基础工作进程类。

**关键能力**：
- 通过`worker_extension_cls`动态扩展工作进程类
- 通过`custom_pipeline_args`支持自定义管道初始化
- 方法委托到底层工作进程
- 属性访问转发

**位置**：`vllm_omni/diffusion/worker/diffusion_worker.py`

### load_format参数

`load_format`参数控制扩散模型的加载方式，支持以下值：

| 值 | 描述 |
|----|------|
| `"default"` | 使用模型注册表的标准模型加载（默认行为） |
| `"custom_pipeline"` | 加载由`custom_pipeline_name`指定的自定义管道类 |
| `"dummy"` | 跳过模型加载（用于测试或管道将单独初始化时） |

**位置**：`vllm_omni/diffusion/model_loader/diffusers_loader.py`

### CustomPipelineWorkerExtension

`CustomPipelineWorkerExtension`是mixin类，扩展`DiffusionWorker`以支持使用自定义实现重新初始化管道。

**关键方法**：
- `re_init_pipeline(custom_pipeline_args)`：使用自定义参数重新初始化管道，正确清理旧管道

**位置**：`vllm_omni/diffusion/worker/diffusion_worker.py`

## 使用示例

### 步骤1：创建自定义管道

创建扩展现有管道的自定义管道类：

```python
# custom_pipeline.py
from vllm_omni.diffusion.data import DiffusionOutput, OmniDiffusionConfig
from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import QwenImageEditPipeline
import torch

class CustomPipeline(QwenImageEditPipeline):
    def __init__(self, *, od_config: OmniDiffusionConfig, prefix: str = ""):
        super().__init__(od_config=od_config, prefix=prefix)

    def forward(self, req, prompt=None, negative_prompt=None, **kwargs):
        # 调用父类的forward获取正常输出
        output = super().forward(req=req, prompt=prompt, negative_prompt=negative_prompt, **kwargs)

        # 添加自定义轨迹数据
        actual_num_steps = req.sampling_params.num_inference_steps or kwargs.get('num_inference_steps', 50)
        output.trajectory_timesteps = torch.linspace(1000, 0, actual_num_steps, dtype=torch.float32)
        output.trajectory_latents = torch.randn(actual_num_steps, 1, 16, 64, 64, dtype=torch.float32)

        return output
```

### 步骤2：使用自定义管道与Omni

使用自定义管道配置初始化`Omni`引擎：

```python
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# 使用自定义管道初始化
omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    diffusion_load_format="dummy",  # 跳过初始加载
    custom_pipeline_args={
        "pipeline_class": "custom_pipeline.CustomPipeline"
    },
)

# 使用自定义管道生成
outputs = omni.generate(...)

# 访问自定义轨迹数据
output = outputs[0].request_output
print(f"Trajectory timesteps shape: {output.metrics['trajectory_timesteps'].shape}")
print(f"Trajectory latents shape: {output.latents.shape}")
```

### 步骤3：运行示例

```bash
cd examples/offline_inference/custom_pipeline/image_to_image

# 使用自定义管道运行
python image_edit.py \
    --model Qwen/Qwen-Image-Edit-2511 \
    --image cherry_blossom.jpg \
    --prompt "Let this mascot dance under the moon, surrounded by floating stars" \
    --output output_image_edit.png \
    --num-inference-steps 10
```

## 高级用法

### 自定义工作进程扩展

可以创建自定义工作进程扩展以添加超出管道重新初始化的新方法：

```python
from typing import Any
from vllm_omni.diffusion.worker.diffusion_worker import CustomPipelineWorkerExtension

class MyCustomExtension(CustomPipelineWorkerExtension):
    def custom_method(self):
        """自定义工作进程方法。"""
        return "custom_result"

    def another_method(self, data: Any):
        """另一个自定义方法。"""
        # 通过self访问工作进程内部
        return self.model_runner.some_operation(data)

omni = Omni(
    model="Qwen/Qwen-Image-Edit",
    diffusion_load_format="dummy",
    custom_pipeline_args={
        "pipeline_class": "custom_pipeline.CustomPipeline"
    },
    worker_extension_cls=MyCustomExtension,
    # 注意：worker_extension_cls是内部参数
    # 当提供custom_pipeline_args时，CustomPipelineWorkerExtension将自动初始化管道
)
```

## 相关链接

- [Hugging Face自定义管道文档](https://github.com/huggingface/diffusers/blob/main/docs/source/en/using-diffusers/custom_pipeline_overview.md)
