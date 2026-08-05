---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/async_diffusion_output.md
---

# 异步扩散输出（Async Diffusion Output）

## 概述

该特性将 D2H（设备→主机）拷贝与 SHM 打包从 Worker 主线程移到后台守护线程，使 GPU 默认流能立即开始下一个请求的 forward，消除同步 D2H/打包造成的 GPU 气泡。

**核心收益**：上一个请求的输出 D2H/打包与下一个请求的 forward 重叠。

**自动启用**：当 `step_execution=False`（默认，请求模式）时自动生效，无需额外配置。`step_execution=True`（逐步模式）时走原有同步路径——不启动 pump 线程与 Worker 后台线程。

## 性能

### HunyuanImage-3.0（TP4）

| 分辨率 | 异步关闭 (QPS) | 异步开启 (QPS) | 变化 |
|--------|---------------|---------------|------|
| 1024×1024 | 0.4773 | 0.4802 | +0.60% |
| 768×768 | 0.8370 | 0.8533 | +1.95% |

**分析**：分辨率越小收益越大。D2H/打包是相对固定开销，而 forward 随分辨率变化；小图 forward 更短，D2H/打包占比更高，重叠收益更明显。

## 架构

### 执行时间线

```
之前（同步 D2H）：
[forward req1] [D2H+SHM pack req1] [forward req2] [D2H+SHM pack req2]
                                 ^^^^^^^^^^^^^^^^ GPU 气泡

之后（异步 D2H）：
[forward req1] [forward req2]
[D2H+SHM pack req1 (side stream)] [D2H+SHM pack req2 (side stream)]
                                 ^^^^^^^^^^^^^^^^ 气泡消除
```

### 数据流

```
                    ┌─────────────────────────────────────────────────────┐
                    │                 Worker 进程                          │
                    │                                                    │
  execute_model ──> │ WorkerBusyLoop ──> forward() ──> DiffusionOutput   │
                    │       │                              │ (GPU tensor)│
                    │       │                    ┌─────────┴──────────┐  │
                    │       │                    │  async 输出路径     │  │
                    │       │                    │                    │  │
                    │       │              compute_done ──> result_mq │  │
                    │       │                    │                    │  │
                    │       │                    ▼                    │  │
                    │       │            AsyncOutputThread            │  │
                    │       │         (side CUDA stream)             │  │
                    │       │         wait_event(gpu_event)          │  │
                    │       │         pack_diffusion_output_shm()    │  │
                    │       │         D2H + SHM 写入                  │  │
                    │       │                    │                    │  │
                    │       │              output_ready ──> result_mq │  │
                    │       │                    │                    │  │
                    │       ▼                    ▼                    │  │
                    │  (dequeue 下一个请求)                            │  │
                    └─────────────────────────────────────────────────────┘
                                         │
                                         ▼
                    ┌─────────────────────────────────────────────────────┐
                    │             Executor（主进程）                      │
                    │                                                    │
                    │  ResultPumpThread（result_mq 唯一读取者）           │
                    │       │                                            │
                    │       ├── AsyncDiffusionOutput(COMPUTE_DONE)       │
                    │       │   → resolve _rpc_futures[rpc_id]           │
                    │       │   → RunnerOutput(async_output_id=...)      │
                    │       │                                            │
                    │       ├── AsyncDiffusionOutput(OUTPUT_READY)       │
                    │       │   → resolve _output_futures[async_output_id]│
                    │       │   → 或经 _batch_split_map 做批量拆分       │
                    │       │                                            │
                    │       └── 非异步消息                               │
                    │           → _sync_result_buffer（其它 RPC）        │
                    │                                                    │
                    │  wait_output_ready(async_output_id) → Future       │
                    └─────────────────────────────────────────────────────┘
```

### 关键组件

1. **AsyncDiffusionOutput**（`data.py`）：`result_mq` 上的协议信封，`kind` 字段路由消息：
   - `COMPUTE_DONE` — forward 完成，GPU 可开始下一个请求
   - `OUTPUT_READY` — D2H/SHM 打包完成，最终输出可用
   - `RPC_RESULT` — 普通 RPC 返回（含错误传播）

2. **Worker AsyncOutputThread**（`diffusion_worker.py`）：后台守护线程，等待 `gpu_event`（跨流同步，确保 forward 已写完张量）→ 在 side CUDA stream 上执行 `pack_diffusion_output_shm()` → 完成后向 `result_mq` 投递 `OUTPUT_READY`。

3. **ResultPumpThread**（`multiproc_executor.py`）：请求模式下 `result_mq` 的唯一读取者，分发消息：
   - `COMPUTE_DONE` / `RPC_RESULT` → 解析 `_rpc_futures[rpc_id]`
   - `OUTPUT_READY` → 解析 `_output_futures[async_output_id]`（批量时经 `_batch_split_map` 拆分）
   - 非异步消息 → `_sync_result_buffer` 供其它 RPC 使用

4. **collective_rpc() 双路径分发**（`multiproc_executor.py`）：
   - 路径 1：`execute_model` / `execute_model_batch` → 生成 `rpc_id`、注册 Future、等待 pump 投递 `compute_done`
   - 路径 2：其它所有 RPC → `_sync_result_buffer`（pump 供数）或 `_result_mq`（step 模式无 pump）

5. **record_device_event**（`platforms/`）：跨平台 GPU 事件录制，用于 side-stream 同步。已实现 CUDA、NPU、ROCm、XPU、MUSA；基类返回 `None`（安全 no-op）。

6. **IPC side-stream D2H**（`ipc.py`）：`pack_diffusion_output_shm()` 及辅助函数接受 `d2h_stream` 参数，使用 `pin_memory` + `copy_(non_blocking=True)` 在 side stream 上执行，替代同步 `.cpu()`。

### 批量拆分

当 `execute_model_batch` 以单个 `async_output_id` 返回整批的 `COMPUTE_DONE` 时，executor 通过 `_batch_split_map` 拆分为每请求一个的 `async_output_id`（格式 `{batch_id}/{request_id}`）。`OUTPUT_READY` 到达后，pump 从批量输出中提取每个请求的结果并独立解析各请求的输出 Future。

## 模型覆盖

### 当前支持（请求模式，`step_execution=False`）

请求模式下所有模型自动启用异步输出。已验证模型：

| 模型 | 类型 | `supports_request_batch` |
|------|------|--------------------------|
| **HunyuanImage-3.0** | 图像 | `False` |
| **Qwen-Image** | 图像 | `True` |

其它 `step_execution=False` 模型同样支持，但尚未验证。

### 暂不支持（逐步模式，`step_execution=True`）

`step_execution=True`（或 `streaming_output=True`，其会自动启用 step 模式）时，模型走 `execute_stepwise` 而非 `execute_model`，不在异步路径 1 的白名单内，异步输出不适用。已验证 Helios（`step_execution=True`，不受益于该特性）。

逐步模式适配需要额外设计，原因：
- 每一步的 `RunnerOutput` 必须同步可用，调度器才能推进
- 中间 latent 张量可能无需 D2H（只有最后一步需要 D2H + 后处理）
- step 之间串行（第 N+1 步依赖第 N 步输出），重叠收益仅限"最后一步的 D2H 与下一个请求的 forward 重叠"

**计划方案**：先只覆盖最后一步的异步 D2H，中间步骤保持同步，与请求模式的收益模式一致。

## 配置

无需配置。`step_execution=False`（默认）时自动启用。

| `step_execution` | 模式 | 异步输出 | Pump 线程 | Worker 后台线程 |
|------------------|------|---------|-----------|----------------|
| `False`（默认） | 请求模式 | ✅ 启用 | ✅ 启动 | ✅ 启动 |
| `True` | 逐步模式 | ❌ 禁用 | ❌ 不启动 | ❌ 不启动 |

## 相关文件

- `vllm_omni/diffusion/data.py`：`AsyncDiffusionOutput`、`AsyncOutputKind`、`DiffusionOutput.async_output_id`
- `vllm_omni/diffusion/worker/diffusion_worker.py`：`WorkerProc._return_result()`、`_async_output_loop()`、`_generate_async_output_id()`
- `vllm_omni/diffusion/executor/multiproc_executor.py`：`ResultPumpThread`、`collective_rpc()` 双路径分发、`wait_output_ready()`、`_batch_split_map`
- `vllm_omni/diffusion/diffusion_engine.py`：`step()` / `step_streaming()` / `add_req_and_wait_for_response()` 的异步输出等待
- `vllm_omni/diffusion/sched/request_scheduler.py`：`update_from_output()` — `async_output_id` → `FINISHED_COMPLETED`
- `vllm_omni/diffusion/ipc.py`：`pack_diffusion_output_shm()` side-stream D2H 路径
- `vllm_omni/platforms/interface.py`：`OmniPlatform.record_device_event()` 基类
- `vllm_omni/platforms/cuda/platform.py`、`vllm_omni/platforms/npu/platform.py`：`record_device_event()` 实现
- `vllm_omni/diffusion/models/hunyuan_image3/pipeline_hunyuan_image3.py`：`get_hunyuan_image3_post_process_func()`、后处理移除
- `vllm_omni/diffusion/registry.py`：`_DIFFUSION_POST_PROCESS_FUNCS` 后处理注册表
