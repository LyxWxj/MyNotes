---
type: Note
related_to: "[[vllm]]"
status: Active
---

# vLLM 设备感知与权重分配机制

vLLM 的设备检测和权重分配完全自动，用户只需通过 `--tensor-parallel-size N` 指定 TP 大小。

## 一、设备检测：全自动

vLLM 通过插件式架构自动检测硬件平台，入口在 [vllm/platforms/\_\_init\_\_.py](Notes/Omni/vllm/vllm/platforms/__init__.py)，核心函数 `resolve_current_platform_cls_qualname()`（约第 213 行）依次尝试每种平台插件。

| 平台 | 检测方式 | 源码 |
|------|---------|------|
| NVIDIA GPU | `pynvml` 调用 `nvmlDeviceGetCount()`，无 NVML 时回退到 `torch.cuda.get_device_properties()` | [vllm/platforms/cuda.py](Notes/Omni/vllm/vllm/platforms/cuda.py) — `NvmlCudaPlatform`（第 584 行）、`NonNvmlCudaPlatform`（第 805 行） |
| AMD GPU | `amdsmi` 库调用 `amdsmi_get_processor_handles()` | [vllm/platforms/rocm.py](Notes/Omni/vllm/vllm/platforms/rocm.py) |
| Intel GPU | `torch.xpu.is_available()` | [vllm/platforms/xpu.py](Notes/Omni/vllm/vllm/platforms/xpu.py) |
| TPU | 检测 `libtpu` 是否存在 | [vllm/platforms/tpu.py](Notes/Omni/vllm/vllm/platforms/tpu.py) |
| CPU | 检查是否 CPU 构建或 macOS | [vllm/platforms/cpu.py](Notes/Omni/vllm/vllm/platforms/cpu.py) |

平台基类 [vllm/platforms/interface.py](Notes/Omni/vllm/vllm/platforms/interface.py) 统一暴露 `device_count()`、`get_device_capability()`、`get_device_total_memory()` 等接口。CUDA 平台还通过 `device_id_to_physical_device_id()` 解析 `CUDA_VISIBLE_DEVICES` 将逻辑设备映射到物理设备。

DeviceConfig（[vllm/config/device.py](Notes/Omni/vllm/vllm/config/device.py)，第 49-73 行的 `__post_init__`）在 `device="auto"` 时自动选择设备类型。

## 二、Tensor Parallel 大小：用户指定

TP 大小由用户通过 `--tensor-parallel-size` 指定，默认 1。vLLM 不会自动选择，但会校验。

- **配置定义**：[vllm/config/parallel.py](Notes/Omni/vllm/vllm/config/parallel.py)，第 112 行 `tensor_parallel_size: int = 1`
- **执行器后端选择**：同文件第 781-826 行，`world_size` 小且适合单机时默认使用 `"mp"`（多进程），跨节点时使用 Ray
- **GPU 数量校验**：第 796-803 行，如果 `device_count() < world_size` 会直接报错

## 三、权重分配：全自动

权重分配到不同 GPU 完全由 vLLM 自动完成，用户无需任何手动操作。分以下几个阶段：

### 3.1 Worker 创建与 GPU 绑定

多进程执行器为每个 GPU 创建一个 worker 进程：

- **Worker 创建**：[vllm/v1/executor/multiproc_executor.py](Notes/Omni/vllm/vllm/v1/executor/multiproc_executor.py)，第 172-199 行，为 `local_rank = 0 .. local_world_size-1` 各 fork 一个 worker 进程
- **设备绑定**：[vllm/v1/worker/gpu_worker.py](Notes/Omni/vllm/vllm/v1/worker/gpu_worker.py)，`init_device()` 方法（第 218-289 行）调用 `torch.device(f"cuda:{self.local_rank}")` 和 `torch.accelerator.set_device_index()` 将每个 worker 固定到唯一 GPU

### 3.2 分布式进程组（TP Group）初始化

- **入口**：[vllm/v1/worker/gpu_worker.py](Notes/Omni/vllm/vllm/v1/worker/gpu_worker.py)，第 1048-1053 行调用 `ensure_model_parallel_initialized()`
- **组创建**：[vllm/distributed/parallel_state.py](Notes/Omni/vllm/vllm/distributed/parallel_state.py)，`initialize_model_parallel()`（第 1484-1582 行），将 `all_ranks` reshape 后沿最后一维解绑形成 TP 组

每个 worker 可以通过 `get_tensor_model_parallel_rank()` 获取自己的 tp_rank。

### 3.3 TP 感知的权重分片（核心）

每个 worker 实例化**完整的模型图**（参数形状已按 TP 大小缩小），权重加载器根据 `tp_rank` 从完整 checkpoint 权重中选取对应分片。所有逻辑集中在 [vllm/model_executor/layers/linear.py](Notes/Omni/vllm/vllm/model_executor/layers/linear.py)：

| 层类型 | 分片维度 | 前向通信 | 源码行号 |
|--------|---------|---------|---------|
| `ColumnParallelLinear` | output_dim（列） | `all-gather` | `weight_loader` 第 528-563 行，`forward` 第 573-587 行 |
| `RowParallelLinear` | input_dim（行） | `all-reduce` | `weight_loader` 第 1475-1508 行，`forward` 第 1519-1541 行 |
| `QKVParallelLinear` | 按 head 分片 | `all-gather` + `all-reduce` | `weight_loader` 第 1164-1365 行 |
| `MergedColumnParallelLinear` | output_dim | `all-gather` | `weight_loader` 第 603 行附近 |
| `ReplicatedLinear` | 不分片 | 无需通信 | `weight_loader` 第 360-381 行 |

**分片公式示例**（列并行）：
```python
shard_size = param_data.shape[output_dim]  # 已经是 1/TP 的大小
start_idx = self.tp_rank * shard_size
loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
```

### 3.4 模型加载流程

1. **ModelRunner.load_model()**：[vllm/v1/worker/gpu/model_runner.py](Notes/Omni/vllm/vllm/v1/worker/gpu/model_runner.py)，第 257-319 行
2. **BaseModelLoader.load_model()**：[vllm/model_executor/model_loader/base_loader.py](Notes/Omni/vllm/vllm/model_executor/model_loader/base_loader.py)，第 43-83 行 — 初始化模型 → 调用 `load_weights()`
3. **DefaultModelLoader.load_weights()**：[vllm/model_executor/model_loader/default_loader.py](Notes/Omni/vllm/vllm/model_executor/model_loader/default_loader.py)，第 368-397 行 — 遍历 checkpoint 权重，调用每个参数的 `weight_loader()`

## 总结

```
用户输入: --tensor-parallel-size 4
           │
           ▼
┌─────────────────────────────────────────────────────┐
│  vllm/platforms/         自动检测 GPU 数量 = 8      │
│  vllm/config/parallel.py 校验 4 ≤ 8，通过           │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│  vllm/v1/executor/multiproc_executor.py              │
│  创建 4 个 worker 进程，每个绑定一个 GPU             │
│  vllm/v1/worker/gpu_worker.py  init_device()        │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│  vllm/distributed/parallel_state.py                  │
│  initialize_model_parallel() 构建 TP 进程组          │
│  每个 worker 获得 tp_rank ∈ {0,1,2,3}               │
└─────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────┐
│  vllm/model_executor/layers/linear.py                │
│  每个参数的 weight_loader() 根据 tp_rank 自动切分   │
│  完整 checkpoint 权重，只加载属于自己那 1/4          │
└─────────────────────────────────────────────────────┘
```