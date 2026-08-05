---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/diffusion_continuous_batching.md
---

# 逐步扩散的连续批处理（Continuous Batching for Step-Wise Diffusion）

> ⚠️ **实验特性**：目前仅适用于以 `step_execution=True` 运行的原生扩散管道。

本文描述构建于 [[diffusion-step-execution|Diffusion Step Execution]] 之上的批处理扩展。基础 step 执行契约不变，批处理工作主要在调度器与 runner 层。

## 为什么有用

逐步执行把长的去噪循环拆成调度器可见的单元，运行时可以在 step 之间接入其它兼容请求，而不必等整个请求完成。

这主要对**低 MFU 或突发流量**场景有意义：
- 单个请求的去噪 step 可能无法打满 GPU
- 多个兼容请求可共享同一次 denoise forward
- 不改变请求本地调度状态即可提升吞吐与设备利用率

这不是单请求延迟的保证收益。主要收益通常是在存在多个在飞兼容请求时提高利用率与吞吐。

## 概述

启用连续批处理后：
- 调度器可同时保持多个兼容请求处于运行态
- runner 把请求本地的 step 状态打包进一个 `InputBatch`
- `denoise_step()` 在该 batch 上执行
- `step_scheduler()` 与 `post_decode()` 仍按请求独立执行

当前实现较保守：
- 只有兼容请求才会被合批
- 各请求的进度与完成保持独立

注意：这里的"连续批处理"指 `step_execution=True` 的逐步路径。请求模式的 `DiffusionRequestBatch` 是一次完整 `forward()` 调用的静态请求级批处理，不会在去噪 step 之间加入或移除请求。

## 启用方式

以 `--step-execution` 作为特性开关，再把 `--max-num-seqs` 调大到 1 以上即启用批处理：

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --step-execution \
  --max-num-seqs 8
```

`--max-num-seqs 1` 保持逐步路径但不启用批处理。

可复现的基准回放见 `benchmarks/diffusion/README.md` 与 `benchmarks/diffusion/performance_dashboard/qwen_image_serving_performance.md` 中的 Qwen-Image replay 命令。

## 调度器

调度器通过 `max_num_running_reqs` 从 `max_num_seqs` 推导批容量。

批次准入由 `StepBatchSamplingParamsKey`（`vllm_omni/diffusion/sched/interface.py`）把关，它由形状敏感与 CFG 敏感的采样字段构建。这是合批正确性的核心规则：**只有共享相同去噪张量契约的请求才能合批**。

三个重要细节：
- `num_inference_steps` 不在 key 中，因此总步数不同的请求仍可共享一个 batch
- 请求不需要处于相同去噪进度；活跃请求即使当前 step 索引不同也可继续合批
- 准入仍是 FIFO，等待队列头部的非兼容请求会阻塞后续兼容请求

当前兼容规则仍对形状敏感。`height`、`width`、`num_frames` 与 CFG 相关字段仍在 key 中，不同分辨率或不兼容 guidance 设置暂不合批。key 还覆盖 LoRA 身份（`lora_int_id`、`lora_scale`），不同适配器或尺度的请求分到不同 batch，worker 每步只激活一个适配器。

调度器批处理单元是一个逻辑 `OmniDiffusionRequest`。逐步路径中运行时张量批表示为 `StepInputBatch`。请求模式的 prompt 语义见 Request-Level Batching。

## Runner

runner 在 `StepRequestState`（`vllm_omni/diffusion/worker/utils.py`）中保存持久化的每请求执行状态；调度器持有独立的轻量 `SchedulerRequestState`（`vllm_omni/diffusion/sched/interface.py`）用于排队与生命周期跟踪。

每一步，runner 从活跃请求状态构建 `InputBatch`（`vllm_omni/diffusion/worker/input_batch.py`）：
- prompt embeddings 与 mask 归一化、padding
- `latents`、`timesteps` 等动态张量每步收集
- batch 组成不变时复用缓冲区

逐步批处理路径：
1. 对新准入请求执行 `prepare_encode()`
2. 构建或刷新 `InputBatch`
3. 执行一次批量 `denoise_step(input_batch)`
4. 将批量 `noise_pred` 切回每个请求
5. 按请求执行 `step_scheduler()`
6. 仅对完成去噪的请求执行 `post_decode()`
7. 通过 `scatter_latents()`（`vllm_omni/diffusion/worker/input_batch.py`）把更新后的 latents 写回持久请求状态

这样共享工作仅限于 denoise forward，请求本地调度器状态与输出保持不变。

## 引擎

`DiffusionEngine` 提供后台循环与异步 add-request 路径，使多个请求能在调度器中累积。

`step_execution=True` 时，引擎把工作路由到逐步执行路径。连续批处理行为由调度器侧兼容性把关与 runner 侧 `StepInputBatch` 打包共同定义。

## 当前限制

- 实验特性；保守的单请求逐步路径请用 `max_num_seqs=1`
- 仅支持已实现 `step_execution=True` 的原生管道
- 仅支持以 `StepBatchSamplingParamsKey` 为准的同构批次
- `cache_backend`、KV 传输及其它请求模式扩展尚未接入批量逐步路径
- 未来可通过更丰富的异构批处理策略（如按分辨率 bucketing 或 padding 执行）放宽当前同形状限制

## 相关文件

- 调度器基类：`vllm_omni/diffusion/sched/base_scheduler.py`
- 调度器接口：`vllm_omni/diffusion/sched/interface.py`
- Step 调度器：`vllm_omni/diffusion/sched/step_scheduler.py`
- Runner：`vllm_omni/diffusion/worker/diffusion_model_runner.py`
- Input batch：`vllm_omni/diffusion/worker/input_batch.py`
- 测试：`tests/diffusion/test_diffusion_scheduler.py`
