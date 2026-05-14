---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
---

# Logits Processors

## 概述

Logits processor 调整 next-token 概率分布，引导模型生成期望的行为。在 vLLM 中，logits processor 在**批次粒度**上操作，消费 `(num_requests) x (vocab_size)` 的 logits 张量，对启用该 processor 的请求对应的行进行变换。

## 在引擎中的工作流

每个 engine step 有两个阶段：

### 1. 更新内部状态 (`update_state`)

每步开始时，persistent batch 可能添加、丢弃或重排请求。引擎调用每个 logits processor 的 `update_state()` 方法，传入 `BatchUpdate` 数据结构。

### 2. 应用变换 (`apply`)

模型推理得到 logits 后，sampler 调用 logits processor 的 `apply()` 方法变换 logits，然后传给 softmax。

## argmax-invariant 优化

- **argmax-invariant**：不改变最高 logit 值的 token ID（如 Min-P）。当所有请求都用 greedy sampling 时可跳过。
- **non-argmax-invariant**：可能改变最高 logit（如强制 EOS 的 mask）。greedy sampling 时不能跳过。

## BatchUpdate 数据结构

`BatchUpdate` 模型化 persistent batch 的状态变化，包含三种操作：

| 操作 | 说明 |
|------|------|
| **Remove** | 移除 index `i` 的请求，留空槽 |
| **Add** | 在 index `i` 添加/替换请求，包含 SamplingParams、prompt/output token ids 的引用 |
| **Move** | 单向移动（UNIDIRECTIONAL）或交换（SWAP）两个位置的请求 |

处理顺序：removes → adds → moves。当新请求少于完成请求时，先 Add 替换已完成的，再 Remove 多余的，最后压缩 batch（Unidirectional Move）使其连续。

## 编程模型

继承 `LogitsProcessor` 基类，实现以下方法：

| 方法 | 说明 |
|------|------|
| `__init__(vllm_config, device, is_pin_memory)` | 初始化 |
| `apply(logits) -> logits` | 在 batch 粒度上变换 logits，可原地或返回新张量 |
| `is_argmax_invariant() -> bool` | 是否不影响 greedy sampling 的 argmax |
| `update_state(batch_update)` | 根据 batch 状态变化更新内部状态 |
| `validate_params(sampling_params)` | 验证请求参数有效性 |

## 内置 Logits Processors

- Min-P
- Logit bias
- Min-tokens

以下功能目前硬编码在 sampler 中，未来将重构为 logits processor：
- Allowed token IDs, Bad words, Repetition/Frequency/Presence penalty, Temperature, Top-K, Top-P

## 最佳实践

- `apply()` 和 `update_state()` 尽量用向量化操作
- 稀疏场景可用字典只存启用该 processor 的请求
- 当无请求启用时，`apply()` 应直接返回未修改的输入张量
- `update_state()` 在 `batch_update is None` 时可提前退出

## 一句话总结

Logits processor 是 vLLM 中以 batch 粒度变换 logits 的有状态插件，通过 `BatchUpdate` 增量同步 batch 状态，并区分 argmax-invariant 和 non-argmax-invariant 以在 greedy sampling 时跳过不必要的计算。
