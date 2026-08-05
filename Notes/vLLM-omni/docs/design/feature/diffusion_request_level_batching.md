---
type: Note
related_to: "[[vllm-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/diffusion_request_level_batching.md
---

# 扩散请求级批处理（Request-Level Batching for Diffusion）

本文描述扩散管道的请求模式批处理路径。与 [[diffusion_continuous_batching|逐步扩散的连续批处理]] 不同：请求级批处理对**静态**的兼容请求批次执行一次完整管道 `forward()`；逐步连续批处理则在 `step_execution=True` 时于去噪 step 之间准入工作。

## 为什么有用

请求级设计避免把多个逻辑请求耦合到同一个请求对象上。这样请求身份、中止/错误处理与每请求元数据保持无歧义，同时仍能为突发或并发流量提供一次融合的管道 forward。

## 概述

启用请求级批处理后：
- 每个 `OmniDiffusionRequest` 包含一个 `prompt` 与一个 `request_id`
- 调度器把兼容的等待请求分成一个调度波（scheduler wave）
- `DiffusionRequestBatch` 包装被调度请求供管道 `forward()` 使用
- 支持批处理的管道返回 `list[DiffusionOutput]`，每个请求一个输出
- `BatchRunnerOutput` 把每个结果映射回其原始 `request_id`

管道通过 `supports_request_batch = True` 与接受 `DiffusionRequestBatch`、返回 `list[DiffusionOutput]` 的 `forward()` 方法选择加入。未选择加入的管道保持原有逐请求执行路径。

## 启用方式

请求级批处理是请求模式路径，因此必须保持 `step_execution` 关闭。把 `max_num_seqs` 调大到 1 以上，让调度器保持多个兼容请求活跃：

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --max-num-seqs 4
```

对突发在线流量，`request_batch_max_wait_ms` 可在调度波首次 `schedule()` 前增加一个有界的准入等待：

```bash
vllm serve Qwen/Qwen-Image --omni \
  --port 8091 \
  --max-num-seqs 4 \
  --request-batch-max-wait-ms 20
```

`request_batch_max_wait_ms=0` 禁用该等待，也是默认值。

## 请求契约

`OmniDiffusionRequest` 表示一个逻辑请求，拥有一个 prompt、采样参数、request id 与请求本地元数据。运行时批次由调度器形成，与请求载荷分开表示。

运行时批处理由以下对象表示：
- `DiffusionSchedulerOutput`（`vllm_omni/diffusion/sched/interface.py`）：被调度请求 id 与请求载荷
- `DiffusionRequestBatch`（`vllm_omni/diffusion/worker/request_batch.py`）：面向管道的请求批次
- `BatchRunnerOutput`（`vllm_omni/diffusion/worker/utils.py`）：每请求结果

`DiffusionRequestBatch` 有意暴露 `prompts`、`sampling_params`、`request_id`、`kv_sender_info` 等兼容属性，使迁移后的管道在贴近上游代码的同时使用 batch 感知契约。

## 调度器

调度器通过 `max_num_running_reqs` 从 `max_num_seqs` 推导容量，并暴露 waiting/running 队列计数，供引擎在调度新波前判断准入等待是否有用。

批次兼容性由 `RequestBatchSamplingParamsKey`（`vllm_omni/diffusion/sched/interface.py`）控制。key 包含形状敏感与 guidance 敏感字段，包括输出数量与 LoRA 身份。形状、CFG 设置、输出数量、LoRA 适配器或 LoRA 尺度不兼容的请求会被分到不同批次。

准入保守：
- 调度器只合批兼容请求
- 保持 FIFO 顺序
- 等待队列头部的非兼容请求会阻塞后续兼容请求

## 引擎

`DiffusionEngine` 在初始化时根据配置的管道类（含自定义管道类）解析请求批处理能力。

能力检查使用管道类属性 `supports_request_batch = True`。设置该属性的管道必须实现请求批兼容的 `forward()` 契约并每个请求返回一个 `DiffusionOutput`；runner 在运行时校验返回形状。

当选中的管道支持批处理且 `step_execution=False` 时，请求模式通过批处理执行器路径路由调度波；否则保持逐请求执行器路径。

可选准入等待仅在以下条件同时满足时运行：
- 支持请求批处理
- `step_execution=False`
- `request_batch_max_wait_ms > 0`
- 当前无请求在运行

等待在以下情况提前退出：等待队列达到容量、队列在短窗口内保持稳定、截止时间到达、或引擎停止。

## 执行器与 Runner

执行器暴露两个请求模式入口：
- `execute_request`：每个被调度请求一次 worker 调用
- `execute_batch`：整个 `DiffusionSchedulerOutput` 一次 worker 调用

在批处理路径上，worker 构建 `DiffusionRequestBatch` 并运行一次管道。请求本地初始化仍按请求进行：
- KV 传输元数据
- 随机生成器与 seed 处理
- 请求输出/错误/中止映射

共享的批设置尽可能每批一次：
- cache refresh
- 同构适配器 key 的 LoRA 激活
- 管道 `forward(req_batch)`

大张量 IPC 仍走共享内存打包路径。打包器同时遍历普通 `RunnerOutput.result` 包装与嵌套批结果，保证批输出的大张量不走 pickle IPC。

## 当前限制

- 只有声明请求批契约的管道才使用融合批执行
- 批次在 `RequestBatchSamplingParamsKey` 下同构；异构分辨率或不兼容 guidance 暂不合批
- FIFO 调度在队列前部存在不兼容请求时会减少合批机会
- `request_batch_max_wait_ms` 改善突发合并但会给调度波首个请求增加延迟；延迟敏感场景请保持较小值
- 逐步连续批处理单独成文，仅在 `step_execution=True` 时适用

## 相关文件

- 请求对象与请求批次：`vllm_omni/diffusion/request.py`
- 调度器接口：`vllm_omni/diffusion/sched/interface.py`
- 调度器基类：`vllm_omni/diffusion/sched/base_scheduler.py`
- 引擎：`vllm_omni/diffusion/diffusion_engine.py`
- Worker runner：`vllm_omni/diffusion/worker/diffusion_model_runner.py`
- 执行器接口：`vllm_omni/diffusion/executor/abstract.py`
- 测试：`tests/diffusion/test_diffusion_engine.py`
