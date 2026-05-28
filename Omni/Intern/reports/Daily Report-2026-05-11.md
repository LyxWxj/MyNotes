# 娄雨轩 Daily Report — 2026-05-11

## 1. 环境搭建

### 1.1 镜像选择

gcc 版本小于 8.5 无法编译 vllm。经查找，镜像 `verl-0.8.0-pytorch_2.9.0-cann_8.5.1-py_3.11-hce_2.0.2512-aarch64-snt9b` 符合要求，预安装了 vllm-0.16.0、torch 2.9.0、torch-npu 2.9.0 等库。

### 1.2 虚拟环境重建

分支使用 vllm-0.14.0，与镜像预装的 vllm-0.16.0 不兼容，需重建虚拟环境：

```bash
cd vllm-omni
uv sync
uv pip install -e .
```
设置uv镜像源（仅在当前上下文环境）
```bash
export UV_INDEX_URL=https://mirrors.huaweicloud.com/repository/pypi/simple
```
### 1.3 依赖问题及解决

**问题 1：fa3-fwd 版本不存在**

```
ERROR: Could not find a version that satisfies the requirement fa3-fwd==0.0.1
```

原因：镜像源中仅有 0.0.2、0.0.3 版本。修改 `pyproject.toml` 中 `fa3-fwd==0.0.1` 为 `fa3-fwd>=0.0.1`。

**问题 2：vllm 与 vllm-ascend 的 torch 版本冲突**

```
vllm==0.14.0 depends on torch==2.9.1
vllm-ascend==0.14.0rc1 depends on torch==2.9.0
```

解决方案：分开安装，绕过依赖解析冲突：

```bash
uv pip install vllm==0.14.0
uv pip install vllm-ascend==0.14.0rc1
```

---

## 2. 启动服务

### 2.1 Dataclass 装饰器错误

启动脚本：

```bash
bash vllm-omni/scripts/start_multi_instance.sh start -m /home/ma-user/work/model -p 9000 -n 4
```

报错：

```
TypeError: non-default argument 'stage_connector_config' follows default argument
```

**根因分析**：`vllm_omni/config/model.py` 中 `OmniModelConfig` 使用了双层装饰器：

```python
@config
@dataclass(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
```

而 vllm-omni 0.20 版本中仅使用单层：

```python
@config(config=ConfigDict(arbitrary_types_allowed=True))
class OmniModelConfig(ModelConfig):
```

双层装饰器导致 pydantic 在处理继承自 `ModelConfig`（本身也是 pydantic dataclass）的子类时，字段排序出错。

**解决方案**（任选其一）：
- 去掉 `@config`，保留 `@dataclass(config=ConfigDict(...))`
- 去掉 `@dataclass(config=ConfigDict(...))`，将 `@config` 改为 `@config(config=ConfigDict(arbitrary_types_allowed=True))`

### 2.2 NPU 显存不足 (OOM)

修复装饰器问题后再次启动，报 OOM：

```
torch.OutOfMemoryError: NPU out of memory. Tried to allocate 74.00 MiB
(NPU 2; 29.50 GiB total capacity; 28.21 GiB already allocated)
```

**根因**：当前租赁的 NPU 型号为 910B4，单卡 HBM 仅 32 GB（可用 ~29.5 GB），而 Diffusion Transformer 模型约 39 GB（fp16），单卡放不下。

---

## 3. 硬件规格

```
NPU 型号: 910B4
单卡 HBM: 32768 MB (32 GB)
可用 HBM: ~29.5 GB
NPU 数量: 4
```

---

## 4. 下一步计划

- 等待 910B1 规格显卡（更大显存）再次测试多实例服务
- 当前先验证包版本依赖之间有无其他问题

## 5. 最小化测试脚本

编写了基于 `accelerate.dispatch_model` 的最小推理脚本，验证依赖和模型加载流程。Transformer 通过 `infer_auto_device_map` 自动分片到多张 NPU，Text Encoder 和 VAE 放置在 NPU 3：

```python
import torch
import torch_npu
from accelerate import dispatch_model, infer_auto_device_map

MODEL = "/home/ma-user/work/model"

def main():
    from diffusers import FlowMatchEulerDiscreteScheduler, DiffusionPipeline
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer
    from diffusers.models import QwenImageTransformer2DModel, AutoencoderKLQwenImage

    n_gpus = torch.npu.device_count()
    print(f"检测到 {n_gpus} 张 NPU")

    # 1. Transformer: CPU 加载 -> 4 张 NPU 均匀分片
    print("加载 Transformer 到 CPU...")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        MODEL, subfolder="transformer", torch_dtype=torch.float16,
    )
    print(f"  参数量: {sum(p.numel() for p in transformer.parameters())/1e9:.1f}B")

    max_mem = {i: "14GiB" for i in range(n_gpus)}
    device_map = infer_auto_device_map(
        transformer, max_memory=max_mem,
        no_split_module_classes=["QwenImageTransformerBlock"],
        dtype=torch.float16,
    )
    counts = {}
    for dev in device_map.values():
        counts[dev] = counts.get(dev, 0) + 1
    print(f"  分配: {dict(sorted(counts.items()))}")

    transformer = dispatch_model(transformer, device_map, main_device="cpu")
    print("  Transformer 分片完成!")

    # 2. Text Encoder + VAE -> NPU 3
    print("加载 Text Encoder 到 NPU 3...")
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL, subfolder="text_encoder", torch_dtype=torch.float16,
    ).to("npu:3")
    print("加载 VAE 到 NPU 3...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        MODEL, subfolder="vae", torch_dtype=torch.float16,
    ).to("npu:3")

    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL, subfolder="tokenizer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL, subfolder="scheduler"
    )

    # 3. 组装 Pipeline
    print("组装 Pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        MODEL,
        transformer=transformer, text_encoder=text_encoder, vae=vae,
        tokenizer=tokenizer, scheduler=scheduler, torch_dtype=torch.float16,
    )

    # 4. 生成
    prompt = "a cute cat sitting on a table"
    print(f"生成中: {prompt}")
    generator = torch.Generator(device="cpu").manual_seed(42)
    image = pipe(
        prompt=prompt, num_inference_steps=30, generator=generator
    ).images[0]
    image.save("output.png")
    print("图片已保存到: output.png")

if __name__ == "__main__":
    main()
```

**NPU 分配方案**：
| NPU | 组件 | 说明 |
|-----|------|------|
| 0, 1, 2 | Transformer blocks | 分片，每张 ~17 GB |
| 3 | Text Encoder + VAE | 共享，约 10 GB |
