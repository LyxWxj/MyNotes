---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Model Runner V2 (MRV2)

## 动机

V1 的 model runner 存在设计缺陷和技术债：persistent batch 设计耦合过紧、异步调度靠 hack 实现、`dummy_run` 职责过多。MRV2 从第一性原理重新设计，更干净、高效、模块化。

## 核心设计

### 1. Persistent Batch 解耦

V1 的 persistent state tensor 直接作为模型输入，导致请求加入/退出时需要复杂的全张量重排。

MRV2 的方案：
- 预分配 `max_num_reqs` 行的固定大小 tensor（默认 1024）
- 每个请求分配永久行，直到完成或抢占
- 抢占视为完成，恢复时重新添加为新状态
- **解耦持久状态和每步输入**：根据 attention backend 确定的请求顺序，从持久状态 gather 出输入 tensor
- 消除了 `CachedRequestState` 冗余备份

### 2. Async-First

MRV2 假设核心模型执行循环是**无 CPU 同步点的 CUDA stream**，CPU 入口只负责排队工作到 stream 上。

### 3. 消除 Async Barrier

V1 用 async barrier 保护临界区，但容易遗漏、不灵活。

MRV2 的方案：**分离持久 CPU 状态和拷贝 tensor**。
- `self.states` 不使用 pin_memory
- 拷贝时先 `pin_memory()` 生成临时副本 `tmp_states`，再异步拷贝到 GPU
- CPU 写 `self.states`，GPU 读 `tmp_states`，消除竞争无需显式同步

### 4. StagedWriteTensor

对大 tensor（如 block table），避免每步全量 CPU→GPU 拷贝：
1. 基础 tensor 在 GPU 上
2. CPU 上暂存 diff
3. 打包 diff 到连续 buffer
4. 拷贝到 GPU
5. 启动一个 kernel 应用 diff

支持 ragged 更新，无 CPU-GPU 同步，最少 kernel 启动。

### 5. GPU-Native 输入元数据准备

用 Triton kernel 准备 `input_ids`、`positions`、`query_start_loc`、`seq_lens` 等。利用 UVA 让 GPU 直接访问 CPU 上的大 tensor（如 `prefill_token_ids`）。

### 6. Triton-Native Sampler

- **Gumbel Sampling**：避免显式 softmax 物化，kernel 内无状态 RNG
- **Efficient Top-K Logprobs**：先从 logits 识别 top-k token，再只计算这些 token 的 logprobs，降低峰值显存
- **Memory-Efficient Prompt Logprobs**：支持更细粒度 chunking，包括单个 prompt 内部 chunking
- **Speculative Decoding 兼容**：用 `idx_mapping` 间接映射，而非扩展 per-request sampling state

### 7. 模块化

V1 的 `gpu_model_runner.py` 庞大且耦合。MRV2 将功能逻辑拆分到独立文件（`mrope_utils.py`、`penalties.py` 等），合并模型输入到 `InputBatch` 类。

### 8. 不滥用 dummy_run

V1 的 `dummy_run` 承担过多职责（内存 profiling、CUDA graph capture、warmup、空 DP forward）。MRV2 简化为：
- `execute_model` 支持 dummy run 不影响状态
- `dummy_run` 委托给 `execute_model`
- CUDA graph capture 使用独立路径

### 9. 显式 CUDA Graph 管理

V1 的 CUDA graph 隐式且难理解。MRV2 用 `CUDAGraphManager` 通过标准 PyTorch API 显式捕获和启动完整 CUDA graph。可将多个 draft model forward pass 捕获为一个 CUDA graph。

## 开发哲学

MRV2 的变更应满足更高的代码质量标准。从第一性原理重新考虑功能，而非快速移植 V1 行为。保持模块化和清晰的抽象边界。

## 一句话总结

MRV2 通过解耦 persistent state、消除 async barrier、GPU-native 输入准备和 Triton sampler，从根本上简化了 V1 的复杂性，同时为异步调度和 CUDA graph 提供了更清晰的设计。
