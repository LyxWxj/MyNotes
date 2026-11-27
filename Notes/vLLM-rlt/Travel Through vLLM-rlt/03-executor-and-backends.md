#### Executor (`vllm_rlt/worker/executor.py`)

|成员|关系|
|---|---|
|`runner`|引用普通 `ModelRunner`|
|`speculation`|引用可选的 speculative runner|
|`submit(SchedulerOutput)`|speculative batch 交给 round controller；普通 batch 先绑定 `ExecutionBatch`|

```text
SchedulerOutput
  ├─ phase == SPECULATIVE
  │    -> SpeculativeRunner.submit(SchedulerOutput)
  └─ ordinary / mixed
       -> build_execution_batch(...)
       -> ModelRunner.submit(SchedulerOutput, batch=ExecutionBatch)
```

`build_execution_batch()` 的绑定：

|内容|来源|
|---|---|
|phase -> operation|`PREFILL/PREFILLING_PRELUDE/PRELUDE` -> PRELUDE；`PREFILLING_RECURRENT/RECURRENT` -> RECURRENT；`CODA` -> CODA|
|位置和深度|`token_start`、`token_count`、`depth`、`chunk_start`、`prefill_hidden_row_offset`|
|loop boundary|优先使用 `SchedulerOutput.loop_count`；直接调用时 PREFILL 为完整 depth、RECURRENT 为一轮|
|hidden 保留|`retain_chunk`、`retain_last` 描述 staged prefill 的 gather/scatter 生命周期|
|校验|同一 batch 必须共享 operation 和 execution boundary|

`Operation`、`StateSpan`、`ExecutionBatch` 当前统一定义在

`vllm_rlt/request.py`。Executor 只绑定 immutable execution descriptor，不更新

Request 进度、不创建 KV metadata。

#### Profiler

In `vllm_rlt/profiling.py`

```text
Profiler(device)
  -> 保存 device
  -> current = None
  -> 第一次 start() 时创建 session
```

`StageClient`/`LLM` 通过 `start_profile()`、`stop_profile()`、`profile_status()`

和 `wait_for_profile_artifacts()` 调用 worker 内的 profiler。

#### Attention Backend

In `vllm_rlt/attention/backend.py`：

```text
create_backend(backend, device, dtype, head_dim, block_size,
               required_capabilities, prefer_packed_prefill, ...)
  -> validate_backend_name()
  -> 创建候选 AttentionBackend
  -> 检查 packed、graph、async、device context、FP8、sliding window 等能力
  -> 返回 kernel + capabilities + selection info
```

|选择方式|行为|
|---|---|
|显式 backend|初始化或能力检查失败直接报错，不 fallback。|
|`auto` + CUDA|FA4 -> FA3 -> FA2 -> Triton -> Torch，选择第一个满足设备、dtype、形状与所需能力的候选。|
|`auto` + CPU|选择 Torch reference attention。|
|Ascend NPU|必须显式选择 `ascend_eager` 或 `torch`；`ascend_eager` 要求 NPU + BF16，使用 Torch SDPA。|
|运行期 kernel 错误|不会触发 backend fallback。|

|能力|当前实现|
|---|---|
|`packed_prefill`|FA4 原生支持；普通 Triton 可按 eligibility 自动启用，滑窗 packed 还需要 tile probe；Ascend eager 可使用共享 packed metadata。`use_uva` 只选择输入传输方式，并非 packed 的开关。|
|`packed_matches_decode`|FA4 的普通路径声明支持；静态 scale FP8 路径不再声明逐位一致。同步 speculative 只有同时满足该能力和模型约束时才使用 packed verification。|
|`async_scheduling` / `device_context_lengths`|Triton、FA2、FA3、FA4 支持两者；`ascend_eager` 声明 async，不声明 device-owned context lengths。|
|`sliding_window`|Torch、Triton 和满足版本/shape 条件的 FA4 支持 masking；masking 不等于回收窗口外的 KV pages。|
|`fp8_kv`|Torch/Triton 支持动态 per-vector scale；兼容的 FA4 支持启动时校准的固定 scale，属于实验路径。|

`AttentionBackend.__call__()` 服务普通 paged attention，`prefill()` 服务 packed prefill。异步 speculative 使用 device-owned single-query metadata，不依赖 packed FlashAttention verification。

| |Packed prefill|普通 paged 路径|
|---|---|---|
|Query|连续 token rows 按 sequence 拼成 ragged batch|每个 query row 使用自己的 context length/page table|
|Metadata|`cu_seqlens_q`、sequence IDs、packed table|逐 row block table/context length|
|KV|仍由 KVCacheManager 提供 request/depth page|仍由 KVCacheManager 提供 request/depth page|
|适用|需要 `packed_prefill`；若要与 decode 保持逐位一致，还需要 `packed_matches_decode`|通用的逐 query 路径|

滑窗 packed prefill 还要求受支持的 packed-window kernel 和普通、同步、eager、单 stage 执行，不能同时启用 speculation/preemption。对这条路径，batch 中每条 prompt 都长于 64 tokens 时才采用 packed，否则使用 per-query；不要把它推广为所有普通 packed prefill 的阈值。

#### Ascend NPU 路径

`vllm_rlt/device.py` 在显式请求 NPU 时导入 `torch_npu`；`StageEngine` 在 NPU 上选择继承自 `ModelRunner` 的 `AscendModelRunner`。`ascend_eager` 使用 Torch SDPA，按 page table 收集可见 KV；共享 paged KV writer 和 packed prefill metadata，固定深度 async 下可使用设备侧状态、流和 event。

当前准入范围是单 stage Ouro 四循环、BF16 权重/KV、greedy、`exit_threshold=1`，必须显式设置 `num_blocks` 或 `kv_cache_memory_bytes`。参考 `docs/recipes/ouro_ascend.md` 的固定深度、完整 prompt prefill、单流配置；该方案不使用 graph、speculation、PD、preemption、prefix caching 或 incremental KV。任意 chunked prefill、混合长度及 adaptive exit 的精度仍未完成全面验证；不能由单次 throughput 结果推断等质量。

