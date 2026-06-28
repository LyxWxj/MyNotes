# vLLM 自定义算子集成：从 csrc 到 model_executor

> [!abstract] 核心主题
> vLLM 如何将 `csrc/` 下的 CUDA/C++ 自定义算子集成到 `model_executor/` 下的模型中。
> 以 `qwen3_omni_moe_thinker.py` 为例，追踪完整调用链。

---

## 一、整体架构

```
csrc/*.cu, *.cpp              ← CUDA/C++ kernel 实现
       ↓
csrc/torch_bindings.cpp       ← TORCH_LIBRARY 注册（schema + impl）
       ↓
CMakeLists.txt                ← 编译为 Python 扩展模块 (.abi3.so)
       ↓
vllm/platforms/interface.py   ← import_kernels() 加载 .so
       ↓
vllm/_custom_ops.py           ← Python 包装层
       ↓
vllm/model_executor/          ← 模型层调用
```

> [!important] 关键点：不使用 pybind11
> vLLM 使用 **PyTorch 原生的 `TORCH_LIBRARY` 注册机制**，而非 pybind11。
> 算子通过 `ops.def()` 定义 schema，`ops.impl()` 绑定实现，注册到 `torch.ops._C` 命名空间。

---

## 二、csrc 目录结构

| 子目录 | 功能 |
|--------|------|
| `attention/` | dtype 特定的 attention 辅助函数 |
| `core/` | `registration.h` — 注册宏定义 |
| `libtorch_stable/` | 稳定 ABI 算子（paged attention, layernorm, rotary embedding, 量化） |
| `moe/` | MoE 算子（topk_softmax, moe_align_block_size, permute/unpermute） |
| `quantization/` | Marlin, Machete, GGUF, FP8/INT8 量化 |
| `cutlass_extensions/` | CUTLASS 扩展类型和工具 |

---

## 三、注册机制

### 3.1 注册宏

> [!note] `TORCH_LIBRARY_EXPAND` vs `STABLE_TORCH_LIBRARY`
> vLLM 有两种注册风格：
> - **`TORCH_LIBRARY_EXPAND`**：用于 `_C` 和 `_moe_C`，与 PyTorch 版本强耦合
> - **`STABLE_TORCH_LIBRARY_FRAGMENT`**：用于 `_C_stable_libtorch`，向前兼容

`csrc/core/registration.h` 定义了关键宏：

```cpp
#define REGISTER_EXTENSION(NAME) \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() { \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module); \
  }
```

### 3.2 注册示例

```cpp
// csrc/torch_bindings.cpp
TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  // 1. 定义 schema
  ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
  // 2. 绑定实现
  ops.impl("silu_and_mul", torch::kCUDA, &silu_and_mul);
}
REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```

> [!tip] `TORCH_EXTENSION_NAME` 由 CMake 设置
> CMake 通过 `-DTORCH_EXTENSION_NAME=_C` 传入，所以 `TORCH_EXTENSION_NAME` 展开为 `_C`，
> 算子可通过 `torch.ops._C.<op_name>` 访问。

### 3.3 稳定 ABI 注册

```cpp
// csrc/libtorch_stable/torch_bindings.cpp
STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) {
  ops.def("rms_norm(Tensor! result, Tensor input, Tensor weight, float epsilon) -> ()");
}
STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, ops) {
  ops.impl("rms_norm", TORCH_BOX(&rms_norm));
}
```

> [!note] `TORCH_BOX` 的作用
> `TORCH_BOX` 确保与未来 PyTorch 版本的前向兼容性。
> `_C` 和 `_C_stable_libtorch` 共存，注册到同一个 `_C` 命名空间。

---

## 四、编译系统

### 4.1 setup.py

```python
# 声明 CMake 扩展模块
ext_modules = [
    CMakeExtension(name="vllm._C"),                  # 核心算子
    CMakeExtension(name="vllm._C_stable_libtorch"),  # 稳定 ABI 算子
    CMakeExtension(name="vllm._moe_C"),              # MoE 算子
]
```

### 4.2 CMakeLists.txt

> [!info] `define_extension_target` 函数
> 定义在 `cmake/utils.cmake:559`，调用 `Python_add_library(MODULE ...)` 创建可被 Python 加载的共享库。

| 目标 | 命名空间 | 用途 |
|------|---------|------|
| `_C` | `_C` | 核心算子：custom all-reduce, 激活函数, Marlin/Machete 量化 |
| `_C_stable_libtorch` | `_C` | 稳定 ABI：paged attention, layernorm, CUTLASS w8a8 |
| `_moe_C` | `_moe_C` | MoE：topk_softmax, moe_align_block_size |
| `_rocm_C` | `_rocm_C` | ROCm 专用 |

---

## 五、Python 加载与包装

### 5.1 平台加载

> [!example] `import_kernels()` 机制
> `vllm/platforms/interface.py:242` 定义了 `import_kernels()`，
> 在模块加载时触发 `import vllm._C`，将算子注册到 `torch.ops._C` 命名空间。

```python
# vllm/platforms/interface.py
@classmethod
def import_kernels(cls) -> None:
    try:
        import vllm._C
    except ImportError as e:
        logger.warning("Failed to import from vllm._C: %r", e)
    with contextlib.suppress(ImportError):
        import vllm._moe_C
```

### 5.2 _custom_ops 包装层

`vllm/_custom_ops.py` 提供类型化 Python 函数：

```python
# vllm/_custom_ops.py
def rms_norm(out, input, weight, epsilon):
    torch.ops._C.rms_norm(out, input, weight, epsilon)

def fused_add_rms_norm(input, residual, weight, epsilon):
    torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)
```

### 5.3 两种调用模式

> [!tip] 模式 A：通过 `_custom_ops` 包装层
> ```python
> from vllm import _custom_ops as ops
> ops.cutlass_scaled_mm(...)
> ```
>
> 模式 B：直接调用
> ```python
> torch.ops._C.silu_and_mul(out, x)
> torch.ops._moe_C.topk_softmax(...)
> ```

---

## 六、以 Qwen3 Omni MoE Thinker 为例

### 6.1 模型结构

```
Qwen3OmniMoeThinkerForCausalLM
  └─ Qwen3OmniMoeThinkerModel
       ├─ Qwen3OmniMoeAudioEncoder        (音频编码器)
       │    └─ Qwen3OmniMoeAudioEncoderLayer
       │         ├─ QKVParallelLinear      ← ops.cutlass_scaled_mm (量化)
       │         ├─ MMEncoderAttention      ← attention backend
       │         └─ ColumnParallelLinear / RowParallelLinear
       ├─ Qwen3Omni_VisionTransformer      (视觉编码器)
       │    └─ Qwen3_VisionBlock
       │         ├─ Qwen3_VisionAttention
       │         └─ Qwen3_VisionMLP
       └─ Qwen3MoeLLMForCausalLM           (语言模型)
            └─ Qwen3MoeModel
                 └─ Qwen3MoeDecoderLayer
                      ├─ RMSNorm           ← torch.ops._C.rms_norm
                      ├─ QKVParallelLinear ← ops.cutlass_scaled_mm
                      ├─ SiluAndMul        ← torch.ops._C.silu_and_mul
                      └─ FusedMoE          ← torch.ops._moe_C.*
```

### 6.2 SiluAndMul 完整链路

> [!example] 链路追踪：SiluAndMul
>
> **模型层** `qwen3_moe.py:123`：
> ```python
> self.act_fn = SiluAndMul()
> ```
>
> **Layer 定义** `activation.py:117-148`：
> ```python
> @CustomOp.register("silu_and_mul")
> class SiluAndMul(CustomOp):
>     def __init__(self):
>         if current_platform.is_cuda_alike():
>             self.op = torch.ops._C.silu_and_mul
>
>     def forward_cuda(self, x):
>         out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
>         self.op(out, x)
>         return out
> ```
>
> **C++ 注册** `libtorch_stable/torch_bindings.cpp:290`：
> ```cpp
> ops.def("silu_and_mul(Tensor! result, Tensor input) -> ()");
> ops.impl("silu_and_mul", TORCH_BOX(&silu_and_mul));
> ```
>
> **CUDA Kernel** `libtorch_stable/activation_kernels.cu`

### 6.3 RMSNorm 完整链路

> [!example] 链路追踪：RMSNorm
>
> **模型层** `qwen3_moe.py:411`：
> ```python
> self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
> ```
>
> **Layer 定义** `layernorm.py:37-116`：
> ```python
> @CustomOp.register("rms_norm")
> class RMSNorm(CustomOp):
>     def forward_cuda(self, x, residual=None):
>         return self.forward_native(x, residual)  # 走 IR 调度
>
>     def forward_native(self, x, residual=None):
>         return ir.ops.rms_norm(x, self.weight.data, self.variance_epsilon, ...)
> ```
>
> **IR 调度层** — 自动选择最优实现：
> - `torch.ops._C.rms_norm` → CUDA kernel
> - `torch.ops.oink.rms_norm` → Oink kernel
> - `torch.ops.vllm_aiter.rms_norm` → AITER kernel (ROCm)
>
> **CUDA Kernel** `libtorch_stable/layernorm_kernels.cu`

### 6.4 FusedMoE 完整链路

> [!example] 链路追踪：FusedMoE
>
> **模型层** `qwen3_moe.py:211`：
> ```python
> self.experts = FusedMoE(
>     num_experts=config.num_experts,
>     top_k=config.num_experts_per_tok,
>     intermediate_size=config.moe_intermediate_size,
> )
> ```
>
> **Layer 定义** `fused_moe/layer.py:68`，内部调用多个 C++ 算子：

| 功能 | Python 调用 | C++ 实现 |
|------|------------|---------|
| topk_softmax | `torch.ops._moe_C.topk_softmax(...)` | `csrc/moe/topk_softmax_kernels.cu` |
| moe_align_block_size | `ops.moe_align_block_size(...)` | `csrc/moe/moe_align_block_size.cu` |
| moe_permute | `torch.ops._moe_C.moe_permute(...)` | `csrc/moe/moe_permute_unpermute_kernels.cu` |
| fused_experts GEMM | `ops.cutlass_moe_mm(...)` | `csrc/quantization/` 或 Triton |
| 激活函数 | `torch.ops._C.silu_and_mul(...)` | `csrc/libtorch_stable/activation_kernels.cu` |

---

## 七、关键文件索引

> [!tip] 快速导航
>
> **C++ 注册层**：
> - `csrc/core/registration.h` — 注册宏
> - `csrc/torch_bindings.cpp` — 主 `_C` 绑定
> - `csrc/libtorch_stable/torch_bindings.cpp` — 稳定 ABI 绑定
> - `csrc/moe/torch_bindings.cpp` — MoE 绑定
>
> **编译系统**：
> - `CMakeLists.txt` — 构建定义
> - `cmake/utils.cmake:559` — `define_extension_target` 函数
> - `setup.py` — 扩展注册
>
> **Python 层**：
> - `vllm/_custom_ops.py` — Python 包装层
> - `vllm/platforms/interface.py:242` — kernel 加载机制
> - `vllm/model_executor/layers/activation.py` — 激活函数层
> - `vllm/model_executor/layers/layernorm.py` — 归一化层
> - `vllm/model_executor/layers/fused_moe/layer.py` — MoE 层
>
> **模型定义**：
> - `vllm/model_executor/models/qwen3_omni_moe_thinker.py` — Qwen3 Omni MoE
> - `vllm/model_executor/models/qwen3_moe.py` — Qwen3 MoE 基础模型

---

## 八、添加自定义算子的步骤

> [!warning] 新增算子的完整流程
>
> 1. **实现 CUDA Kernel**：在 `csrc/` 下编写 `.cu` 文件
> 2. **注册算子**：在对应的 `torch_bindings.cpp` 中添加 `ops.def()` 和 `ops.impl()`
> 3. **更新 CMakeLists.txt**：确保新文件被编译
> 4. **添加 Python 包装**（可选）：在 `_custom_ops.py` 中添加包装函数
> 5. **在 model_executor 中使用**：通过 `torch.ops._C.<op_name>` 或 `ops.<op_name>` 调用

---

## 九、自动融合问题：标准 torch ops 会自动调用 csrc kernel 吗？

> [!danger] 答案：不会
> 如果你用标准 PyTorch 函数（`torch.silu`、`F.rms_norm`、`torch.mm`）写代码，
> **不会**自动调用 `csrc/` 下的自定义高性能算子。

### 9.1 为什么不会？

> [!important] `torch.compile` 不会自动发现 csrc 算子
> `csrc/` 下的 CUDA kernel 对 `torch.compile` 来说是**不透明的自定义算子**（opaque custom ops）。
> Inductor 不会扫描 `csrc/` 目录，也不会把标准 PyTorch op 替换为自定义 kernel。

```python
# ❌ 这样写 — 走 PyTorch 原生实现，不会调用 csrc kernel
def my_rms_norm(x, weight, eps):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight

# ✅ 这样写 — 走 csrc kernel
def my_rms_norm(x, weight, eps):
    return ir.ops.rms_norm(x, weight, eps)  # 调度到 torch.ops._C.rms_norm
```

### 9.2 vLLM 的三层调度机制

vLLM 用一套**显式的三层系统**来调用自定义算子，而非自动发现：

```
┌─────────────────────────────────────────────────────────┐
│  第1层：模型代码显式使用 IR ops 或 torch.ops._C          │
│    ir.ops.rms_norm()  或  torch.ops._C.silu_and_mul()   │
├─────────────────────────────────────────────────────────┤
│  第2层：IR 优先级调度                                     │
│    按优先级选择最优实现：vllm_c > oink > aiter > native   │
├─────────────────────────────────────────────────────────┤
│  第3层：Fusion Pattern Matcher                           │
│    将多个 IR op 组合（如 RMSNorm+Quant）融合为单个 kernel  │
└─────────────────────────────────────────────────────────┘
```

### 9.3 IR 优先级调度详解

> [!info] IR ops 的工作原理
> `ir.ops.rms_norm(x, w, eps)` 被调用时，`IrOp.__call__` 通过 `torch.ops.vllm_ir.rms_norm.default` 路由。
> 运行时 `dispatch()` 方法按优先级列表选择第一个 `supports_args` 检查通过的实现。

优先级配置（`vllm/config/kernel.py`）：

| 平台 | 优先级顺序 | 说明 |
|------|-----------|------|
| CUDA | `["vllm_c", "native"]` | 优先用 vLLM C kernel，回退到纯 PyTorch |
| ROCm | `["aiter", "vllm_c", "native"]` | 优先用 AITER，再 vLLM C，最后纯 PyTorch |
| CPU | `["native"]` | 只用纯 PyTorch |

`vllm_c` provider 在 `vllm/kernels/vllm_c.py` 中注册：

```python
@ir.ops.rms_norm.register_impl("vllm_c", supports_args=rms_no_var_size, supported=CUDA_ALIKE)
def rms_norm(x, weight, epsilon, variance_size=None):
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    torch.ops._C.rms_norm(output, x, weight, epsilon)  # ← 调用 csrc kernel
    return output
```

### 9.4 Fusion Pattern Matcher

> [!warning] 融合只匹配 IR ops，不匹配标准 torch ops
> Pattern matcher 只识别 `vllm.ir.ops.*` 和 `torch.ops._C.*`，
> **不认识** `F.rms_norm`、`torch.silu`、`torch.mm` 等标准算子。

vLLM 的融合 pass（`vllm/compilation/passes/fusion/`）：

| 融合 Pass | 输入模式 | 输出 kernel |
|-----------|---------|------------|
| `rms_quant_fusion.py` | `ir.ops.rms_norm` + `quant` | `torch.ops._C.rms_norm_static_fp8_quant` |
| `act_quant_fusion.py` | `silu_and_mul` + `quant` | `torch.ops._C.silu_and_mul_quant` |
| `allreduce_rms_fusion.py` | `AllReduce` + `ir.ops.rms_norm` | 融合 allreduce+layernorm |
| `rope_kvcache_fusion.py` | `RoPE` + `KV cache` | 融合 rope+cache 写入 |
| `qk_norm_rope_fusion.py` | `QK norm` + `RoPE` | 融合 qk_norm+rope |

### 9.5 IR Lowering Pass

> [!example] 编译时的 IR 降低流程
> `VllmIRLoweringPass`（`compilation/passes/ir/lowering_pass.py`）在 Inductor 生成代码之前运行：
> 1. 在 FX 图中找到所有 `vllm_ir` 命名空间的 op
> 2. 用 `dispatch()` 选择最优实现
> 3. 用 `match.replace_by_example()` 将选中实现内联到图中
> 4. Inductor 随后对内联的代码做进一步优化

### 9.6 对比表

| 写法 | 是否调用 csrc kernel | 说明 |
|------|:-------------------:|------|
| `F.rms_norm(x, w, eps)` | ❌ | 走 PyTorch 原生实现 |
| `torch.silu(x) * y` | ❌ | 走 PyTorch 原生实现 |
| `torch.mm(a, b)` | ❌ | 走 cuBLAS |
| `ir.ops.rms_norm(...)` | ✅ | 调度到最优 csrc kernel |
| `torch.ops._C.silu_and_mul(...)` | ✅ | 直接调用 csrc kernel |
| `torch.ops._moe_C.topk_softmax(...)` | ✅ | 直接调用 MoE csrc kernel |

> [!tip] 结论
> vLLM 的高性能算子是**显式调用**的，不是自动发现的。
> 模型代码必须使用 `vllm.ir.ops.*` 或 `torch.ops._C.*` 才能触达自定义 kernel。
> 这是设计选择——确保行为可预测，而非依赖编译器的隐式优化。

### 9.7 关键文件索引

> [!tip] 调度与融合相关文件
>
> **IR ops 系统**：
> - `vllm/ir/op.py` — IR op 注册与调度
> - `vllm/ir/ops/layernorm.py` — IR op 定义（native 实现）
> - `vllm/kernels/vllm_c.py` — vLLM C kernel 作为 IR op provider
>
> **编译与融合**：
> - `vllm/compilation/backends.py` — VllmBackend
> - `vllm/compilation/passes/pass_manager.py` — Pass 编排
> - `vllm/compilation/passes/ir/lowering_pass.py` — IR lowering pass
> - `vllm/compilation/passes/fusion/rms_quant_fusion.py` — RMSNorm+Quant 融合
> - `vllm/compilation/passes/fusion/act_quant_fusion.py` — Activation+Quant 融合
> - `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` — AllReduce+RMSNorm 融合
>
> **配置**：
> - `vllm/config/kernel.py` — IR op 优先级配置

---

#vLLM #CUDA #自定义算子 #MoE #模型推理 #torch_compile #IR调度 #Kernel融合
