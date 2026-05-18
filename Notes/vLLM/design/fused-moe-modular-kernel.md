---
type: Note
related_to: "[[vLLM]]"
status: Active
url: https://github.com/vllm-project/vllm
_organized: true
---

# FusedMoEModularKernel

## 整体目标

FusedMoEModularKernel 是 vLLM 中 **MoE（Mixture of Experts）前向计算**的模块化框架。它将复杂的 FusedMoE 操作拆分为可组合的组件，解决以下问题：

- MoE 操作步骤多、实现变体多，组合方式爆炸式增长
- All2All 通信与专家计算的实现应该**解耦**，独立开发和测试
- 需要一个清晰的骨架（抽象类）供未来实现扩展

## 两种激活格式

根据 All2All Dispatch 返回的数据格式，分为两类：

| 格式 | 形状 | 说明 |
|------|------|------|
| **Contiguous / Standard** | `(M, K)` | 激活是连续张量，配合 TopK Ids 和 weights，形状 `(M, num_topk)`。典型实现：`DeepEPHTPrepareAndFinalize`（高吞吐） |
| **Batched** | `(num_experts, max_tokens, K)` | 同一专家的 token 被 batch 在一起，配合 `expert_num_tokens` 标记有效 token 数。典型实现：`DeepEPLLPrepareAndFinalize`（低延迟） |

两者的**唯一操作差异**在于是否有 Permute / Unpermute 步骤。

## 三大组件

### 1. FusedMoEPrepareAndFinalizeModular — 通信层

负责 All2All 通信的两端：

- **`prepare()`**：输入量化 + All2All Dispatch（将 token 分发到各专家所在的 GPU）
- **`prepare_no_receive()`**：同 prepare，但不等待通信完成，返回一个回调 thunk，用于**通信与计算的重叠**（如 DBO 中与共享专家重叠）
- **`finalize()`**：All2All Combine（收集专家计算结果）+ 可能的 TopK 权重应用与归约

关键接口：
- `activation_format()`：返回 Batched 或 Standard
- `topk_indices_dtype()`：TopK ids 的数据类型要求
- `max_num_tokens_per_rank()`：单次 Dispatch 的最大 token 数
- `num_dispatchers()`：分发单元总数，决定 Dispatch 输出大小

### 2. FusedMoEExpertsModular — 计算层

MoE 的核心计算逻辑，`apply()` 方法执行：

```text
Permute → Matmul(W1) → Act+Mul → Quantization → Matmul(W2) → Unpermute → [可选] TopK权重+归约
```

关键接口：
- **`workspace_shapes()`**：声明两个 workspace 的形状和输出形状，避免反复分配临时内存
- **`finalize_weight_and_reduce_impl()`**：返回 `TopKWeightAndReduce` 对象，告诉 `finalize()` 是否需要额外做权重应用和归约

### 3. TopKWeightAndReduce — 灵活的归约策略

TopK 权重应用和归约可以发生在两个地方：
- 在 `FusedMoEExpertsModular.apply()` 内部做（返回 `TopKWeightAndReduceNoOp`，finalize 不做额外操作）
- 在 `FusedMoEPrepareAndFinalizeModular.finalize()` 中做（返回具体实现类）

这种灵活性通过抽象类 `TopKWeightAndReduce` 实现。

## 整体流程伪代码

```python
class FusedMoEModularKernel:
    def forward(self, DP_A):
        # 1. 量化 + All2All Dispatch
        Aq, A_scale, _, _, _ = self.prepare_finalize.prepare(DP_A, ...)

        # 2. 分配 workspace
        ws13_shape, ws2_shape, _, _ = self.fused_experts.workspace_shapes(...)
        workspace_13 = torch.empty(ws13_shape, ...)
        workspace_2 = torch.empty(ws2_shape, ...)

        # 3. 执行专家计算
        fe_out = self.fused_experts.apply(Aq, A_scale, workspace_13, workspace_2, ...)

        # 4. 获取 TopK 归约策略
        war_impl = self.fused_experts.finalize_weight_and_reduce_impl()

        # 5. All2All Combine + 可选的 TopK 归约
        output = self.prepare_finalize.finalize(fe_out, war_impl, ...)
        return output
```

## 初始化流程

`FusedMoEMethodBase` 的三个方法协作构建 `FusedMoEModularKernel`：

1. **`maybe_make_prepare_finalize()`**：根据 all2all 后端和并行策略（EP+DP 等），构造 `FusedMoEPrepareAndFinalizeModular`
2. **`select_gemm_impl()`**：由子类实现，构造合适的 `FusedMoEExpertsModular`
3. **`init_prepare_finalize()`**：协调上述两者，组装最终的 `FusedMoEModularKernel`，并将其赋值给 `fused_experts`，使子类无需关心具体实现

## 如何扩展

- **添加新的 Prepare/Finalize**：先实现 All2All Manager，再继承 `FusedMoEPrepareAndFinalizeModular`
- **添加新的 Experts 实现**：继承 `FusedMoEExpertsModular`，实现 `apply()`、`workspace_shapes()` 等
- **兼容性测试**：`test_modular_kernel_combinations.py` 遍历所有组合进行正确性验证
- **性能分析**：`profile_modular_kernel.py` 可生成单次 `forward()` 的 Torch trace

## 一句话总结

FusedMoEModularKernel 是一个**插件式框架**，将 MoE 的通信（Dispatch/Combine）、计算（专家 Matmul）、归约（TopK 权重）三个阶段解耦为可独立替换的模块，通过抽象类定义契约，使得不同的 All2All 后端和专家内核实现可以自由组合，同时保持代码清晰和可测试性。
