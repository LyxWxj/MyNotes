#### DiffusionEngine 与 continuous batching

DiffusionEngine 是 stage 内的请求入口和执行循环 owner。它建立 scheduler、executor、后台 busy loop、输出流和 worker 控制面；scheduler 决定容量、可兼容请求、进度和终态；executor 把批次提交到 worker；worker/model runner 持有设备、pipeline 和模型状态。

执行 backend 可为 **uni**（engine 进程内 worker）、**mp**（每个设备一个 worker process）或 **ray**（跨节点 GPU actor）。这些选择改变 IPC、资源边界和故障表现，但上层 request lifecycle 由同一类 engine/scheduler 契约驱动。

- `vllm_omni/diffusion/diffusion_engine.py#L275`、`vllm_omni/diffusion/diffusion_engine.py#L431`
- `vllm_omni/diffusion/diffusion_engine.py#L537`、`vllm_omni/diffusion/diffusion_engine.py#L709`
- `docs/design/module/diffusion/diffusion_runtime.md`

### 两种 scheduler

- **RequestScheduler** 更偏向整次 request / request wave；等待队列按配置容量和兼容 key admission，只有兼容的请求可共享执行 wave。
- **StepScheduler** 维护每个请求的 current step、total steps、交互代数与等待期限。每个 tick 由 engine 将当前 batch 发往 executor，随后 scheduler 根据结果推进、完成、报错，或进入 wait-for-interaction 状态。
- 稳定 request id 是输出归属依据，不能用某一 tick 的 batch 下标作为身份。兼容性显式比较；这有意限制了能够一起运行的 request 组合。

执行路径可读为：add request → scheduler schedule → executor execute → worker/model runner → scheduler update → 输出 / 下一 tick。Request 级失败应尽量局限于单 request；worker group 死亡则会让 executor/stage 进入故障处理路径。

- `vllm_omni/diffusion/sched/request_scheduler.py`、`vllm_omni/diffusion/sched/step_scheduler.py`
- `docs/design/module/diffusion/continuous_batching.md`
- `tests/diffusion/test_step_request_lifecycle.py`、`tests/diffusion/test_step_interaction_wait.py`、`tests/diffusion/batching/test_diffusion_batching.py`

### 设备输出与同步

输出 tensor 交给异步消费者前，producer stream 上的写必须对 consumer 可见。Diffusion uni executor 的输出流对已 materialize 的 device output 记录/等待 CUDA event，并在 consumer stream 上登记 tensor storage；没有 device event 的 CPU 输出无需该等待。没有 event 能力的平台在返回前完成 producer。没有输出的 denoise step 不应被误当成每步都要全局同步。

这说明**提交 execute 调用**只表示工作已排入执行路径，不表示输出 tensor 已可安全读取。实际数据可见性由 event/stream 或平台同步保证。

- `docs/design/module/diffusion/diffusion_runtime.md#uniproc-backend`
- `tests/diffusion/test_result_pump.py`
- `tests/diffusion/test_uniproc_executor.py`
