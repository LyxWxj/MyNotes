---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# Quantization

## 概述

量化通过降低模型精度换取更小的内存占用，使大模型能在更广泛的设备上运行。

## 支持的量化格式

| 格式 | 说明 |
| --- | --- |
| AutoAWQ | 权重量化，支持 INT4 |
| BitsAndBytes | 4-bit/8-bit 量化 |
| GGUF | 通用格式，支持多种精度 |
| GPTQModel | GPTQ 量化 |
| Intel Neural Compressor | Intel 平台优化 |
| INT4 W4A16 | 权重 INT4，激活 FP16 |
| INT8 W8A8 | 权重和激活均为 INT8 |
| FP8 W8A8 | 权重和激活均为 FP8 |
| NVIDIA ModelOpt | NVIDIA 量化工具 |
| AMD Quark | AMD 平台量化 |
| Quantized KV Cache | KV Cache 量化 |
| TorchAO | PyTorch 原生量化 |

## 硬件兼容性

| 实现 | Volta | Turing | Ampere | Ada | Hopper | AMD | Intel | x86 CPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AWQ | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| GPTQ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ |
| Marlin | ❌ | ✅* | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| INT8 | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| FP8 | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |
| bitsandbytes | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| GGUF | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |

*Turing 不支持 Marlin MXFP4。

## 自定义量化插件

通过 `@register_quantization_config` 装饰器注册自定义量化方法：

```python
@register_quantization_config("my_quant")
class MyQuantConfig(QuantizationConfig):
    def get_name(self): return "my_quant"
    def get_supported_act_dtypes(self): return [torch.float16]
    def get_min_capability(self): return -1
    def get_config_filenames(self): return []
    def from_config(cls, config): return cls()
    def get_quant_method(self, layer, prefix): ...
```

### 必须实现的方法

- `get_name()`：量化方法名称
- `get_supported_act_dtypes()`：支持的激活数据类型
- `get_min_capability()`：最低 GPU 计算能力
- `get_config_filenames()`：配置文件名列表
- `from_config()`：从配置字典创建实例
- `get_quant_method()`：根据层类型返回量化方法

### 量化层实现

- Linear 层：继承 `UnquantizedLinearMethod`，实现 `create_weights` 和 `apply`
- MoE 层：继承 `FusedMoEMethodBase`，实现 `create_weights`、`apply`、`get_fused_moe_quant_config`

## 推荐入门

使用 [LLM Compressor](https://github.com/vllm-project/llm-compressor) 进行 FP8、INT8、INT4 等格式的量化。
