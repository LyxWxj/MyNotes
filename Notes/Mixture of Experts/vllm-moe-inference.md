---
type: Note
status: Active
related_to:
  - "[[moe-from-basics-to-stable-latentmoe]]"
  - "[[expert-parallelism-and-moe]]"
---

# vLLM / vLLM-omni 的 MoE 推理实现

## 调研范围

本文以 2026-08-07 工作区中的代码为准：

- vLLM: `43d691ec6b1d26d3ef3d8725a7c7e4d8556eb984`
- vLLM-omni: `24ede9e226a62910455d5d8da4c71d5c5177c292`

重点追踪两条链路：

1. vLLM 语言模型的 token-choice sparse MoE，从 `LLM.generate()` 到专家 GEMM。
2. vLLM-omni 对上游 MoE 的复用、扩散模型 adapter，以及仍然保留原生 Python/Grouped-MM 的多模态模型。

同一个 DeepSeek 风格 MoE 同时具有“sigmoid + bias + grouped top-k”的路由特征、“shared expert”的结构特征、“EP + all-to-all”的通信特征和“FP8/AITER/Triton”的 kernel 特征。因此下面按代码层级分别计数。

## 结论与数量

### 1. 路由公式：13 个代码标签，10 个具体公式

`vllm/model_executor/layers/fused_moe/config.py:100-129` 的 `RoutingMethodType` 有 13 个标签：

| 标签 | 数学/流程 | 典型用途 |
| --- | --- | --- |
| `Default` | softmax -> top-k | Mixtral 等普通 MoE |
| `Renormalize` | top-k -> softmax | FlashInfer 兼容标签 |
| `DeepSeekV3` | sigmoid -> bias -> 选 group -> 选 expert -> 归一化 | DeepSeek V2/V3、部分衍生模型 |
| `Llama4` | top-1 -> sigmoid | Llama 4 风格 |
| `RenormalizeNaive` | softmax -> top-k -> 对 top-k 重新归一化 | 普通 softmax 路由的归一化版本 |
| `TopK` | 直接 top-k，不做概率变换 | 特殊模型/兼容路径 |
| `SigmoidRenorm` | sigmoid -> top-k -> 按 top-k 和归一化 | sigmoid 路由 |
| `MiniMax2` | sigmoid + bias -> top-k -> scaled sum normalize | MiniMax 风格 |
| `Sigmoid` | sigmoid -> top-k，不重新归一化 | 部分 sigmoid 路由 |
| `DeepseekV4` | sqrtsoftplus + bias + normalize | DeepSeek V4/Kimi 相关路径 |
| `Unspecified` | 不能映射到已知 FlashInfer 方法 | 回退/新组合 |
| `Custom` | 由模型提供 routing function | Hunyuan、Bailing 等 |
| `Simulated` | 仿真路由，仅用于测试/性能实验 | `VLLM_MOE_ROUTING_SIMULATION_STRATEGY` |

所以：如果只数有明确公式名称的内置方法，是 10 个；如果数枚举项，是 13 个。`Unspecified`、`Custom`、`Simulated` 是执行策略标签，不应当当作三种新的数学路由算法。

### 2. Router 工厂：7 类运行时实现

`router/router_factory.py:114-236` 根据配置选择下面 7 类 router：

1. `RoutingSimulatorRouter`
2. `ZeroExpertRouter`
3. `GroupedTopKRouter`
4. `CustomRoutingRouter`
5. `FusedTopKBiasRouter`，也处理 DeepSeek V4 的 hash table/输入 ID 相关映射
6. `AiterSharedRoutedFusedMoERouter`，ROCm shared-expert fusion 专用
7. `FusedTopKRouter`，普通 softmax/sigmoid top-k 回退

这是“router 类”的数量，不是“路由公式”的数量。比如 `GroupedTopKRouter` 内部还可以走 Python/`torch.compile` 路径、CUDA `grouped_topk` 路径或 ROCm AITER 路径。

### 3. vLLM 模型接入：48 个模型适配模块 + 1 个通用 Transformers adapter

对 vLLM 当前源码用 `FusedMoEFactory`、`SharedFusedMoE`、`FusedMoE(` 做静态盘点，可以看到约 48 个模型实现模块直接接入统一 MoE 工厂，另有 `vllm/model_executor/models/transformers/moe.py` 这个通用 Hugging Face Transformers adapter。

它们主要复用同一条 `FusedMoEFactory -> MoERunner -> RoutedExperts` 管线，只在 gate、top-k、shared expert、权重布局、激活函数和输入输出变换上做配置。模型层面可以归纳为：

- 普通 routed-only MoE：Mixtral、Qwen2/3、OLMoE、PhiMoE、DBRX、Cohere、Granite 等。
- shared-expert MoE：Qwen3、DeepSeek、Bailing、Nemotron、Laguna 等，routed 输出和 shared MLP 相加或重叠执行。
- grouped/bias MoE：DeepSeek V2/V3、Bailing、LongCat、MiniMax 等。
- latent MoE：`nemotron_h.py` 通过 `routed_input_transform` 把 routed experts 放在压缩维度，结束后用 `routed_output_transform` 投影回完整 hidden size；shared expert 仍处理原始 hidden size。
- 非标准激活/带 bias 的 MoE：GPT-OSS 的 `swigluoai`、带专家 bias 的 Bailing/GPT-OSS 等。
- zero/redundant/EPLB MoE：zero expert、冗余 physical experts 和 logical-to-physical load balancing 是运行时扩展，不是新的模型数学结构。
- 外部路由 MoE：模型先在自己的 Python 模块中计算 top-k，再通过 `custom_routing_function` 把结果交给统一专家 kernel。

### 4. 执行后端：10 个 prepare/finalize 模块，9 个未量化后端标签

`fused_moe/prepare_finalize/` 当前有 10 个实现模块（不计 `__init__.py`）：

- `no_dp_ep.py`：单卡/普通 TP，无 DP/EP dispatch
- `naive_dp_ep.py`：AllGather + ReduceScatter 的朴素 DP/EP
- `batched.py`：本地 batched expert layout，当前主要给 XPU batched path 使用
- `deepep_ht.py`：DeepEP high-throughput
- `deepep_ll.py`：DeepEP low-latency
- `deepep_v2.py`：DeepEP v2
- `mori.py`：MoRI high-throughput/low-latency
- `flashinfer_nvlink_two_sided.py`
- `flashinfer_nvlink_one_sided.py`
- `nixl_ep.py`

`oracle/unquantized.py:36-45` 的 `UnquantizedMoeBackend` 有 9 个标签：FlashInfer TRTLLM、FlashInfer CUTLASS、AITER、Triton、Batched Triton、CPU、XPU、TPU、OOT。CUDA 默认候选通常是前四/五个，ROCm 默认候选是 AITER、Triton、Batched Triton；9 个标签不能理解为 9 个都会在一张 CUDA 卡上同时出现。

此外，kernel oracle 还有 9 个具体 selector family：unquantized、FP8、INT8、INT-WNA16、NVFP4、MXFP4、MXFP8、W4A8、W4A8-INT8。它们和路由变体正交。

## 自顶向下的调用链

下面先描述标准 vLLM LLM 请求，再补充 vLLM-omni 的 stage/diffusion 入口。

```text
LLM.generate() / vllm serve
  -> LLMEngine
  -> EngineCore.step()
  -> Scheduler.schedule()
  -> GPUWorker.execute_model()
  -> GPUModelRunner.execute_model()
  -> set_forward_context(...)
  -> GPUModelRunner._model_forward()
  -> model(...)
  -> model-specific SparseMoeBlock
  -> MoERunner.forward()
  -> router + dispatch/prepare
  -> expert kernel: permute/sort -> GEMM1 -> activation -> GEMM2
  -> finalize: top-k weighting -> unpermute/sum -> EP combine
  -> shared expert/output transform/all-reduce
```

### A. 顶层 API 与调度

1. 离线 API 的 `LLM.__init__` 在 `vllm/entrypoints/llm.py:339-341` 通过 `LLMEngine.from_engine_args` 创建 engine；`LLM.generate` 在 `:414-477` 把 prompt 和 sampling 参数交给 completion 路径。
2. `LLMEngine` 在 `vllm/v1/engine/llm_engine.py:104-111` 创建 `EngineCoreClient`。请求由 `add_request` 送入 EngineCore；MoE 还没有参与，此处只管理请求、KV cache 和调度。
3. `vllm/v1/engine/core.py:580-610` 的 `EngineCore.step()` 先调用 scheduler，再调用 `model_executor.execute_model`。
4. `vllm/v1/worker/gpu_worker.py:1022-1086` 把 scheduler 输出交给 `GPUModelRunner`。
5. `vllm/v1/worker/gpu_model_runner.py:4463-4493` 设置 forward context、EPLB 统计元数据和 token 数，然后在 `:3927-3957` 的 `_model_forward` 中调用已经加载好的模型。

因此，MoE 并不是由 `LLM.generate()` 特判出来的。它是模型构造时注册的普通 `nn.Module`，在模型 forward 走到某个 sparse MLP block 时才出现。

### B. 模型 block 如何进入统一 MoE

以 Mixtral 为例，`vllm/model_executor/models/mixtral.py:69-147` 做三件事：

1. 用 `ReplicatedLinear` 计算 router logits。
2. 用 `FusedMoEFactory` 创建 experts。
3. 将输入 flatten 成 `[T, H]`，调用 `self.experts(hidden_states, router_logits)`。

Qwen3 的 block（`qwen3_moe.py:172-245`）额外传入 shared expert；DeepSeek V2（`deepseek_v2.py:300-428`）增加 grouped top-k、correction bias、scaling 和 shared expert；Nemotron-H（`nemotron_h.py:191-236`）增加 latent input/output transform。

`FusedMoEFactory` 的核心参数在 `vllm/model_executor/layers/fused_moe/layer.py:99-147`，包含：

- `num_experts`、`top_k`、hidden/intermediate size
- `scoring_func`、`renormalize`、grouped top-k、expert score bias
- shared experts、shared expert gate、zero expert
- `routed_input_transform` / `routed_output_transform`
- TP/DP/PCP、EPLB、冗余 experts、sequence parallel
- quantization config、activation、权重 checkpoint 名称和 custom router

工厂在 `layer.py:216-425` 依次构造：

```text
FusedMoEParallelConfig
  -> ExpertMapManager
  -> FusedMoERouter
  -> FusedMoEConfig
  -> RoutedExperts
  -> MoERunner
```

权重通常按如下布局保存：

- `w13_weight`: `[E_local, 2 * I, H]`，包含 gate/up projection
- `w2_weight`: `[E_local, H, I]`，包含 down projection

`RoutedExperts` 同时负责参数创建、checkpoint expert ID 到 local expert ID 的映射和 quant method；它的 `forward_modular` / `forward_monolithic` 在 `routed_experts.py:1199-1266` 分别把已选的 top-k 或原始 router logits 交给量化方法。

### C. Router：从 logits 到 `(topk_weights, topk_ids)`

`router_factory.py` 的选择优先级大致是：仿真 -> zero expert -> grouped top-k -> custom -> bias/hash -> AITER shared fusion -> 普通 fused top-k。

统一的 `BaseRouter._select_experts`（`router/base_router.py:260-305`）做四步：

1. 校验 EPLB 状态。
2. 调用具体 router 的 `_compute_routing`。
3. 保存 logical expert IDs；如果启用 EPLB，通过 Triton `eplb_map_to_physical_and_record` 把 logical ID 映射到 physical replica，并记录负载。
4. 把 `topk_ids` 转成当前 kernel 要求的索引类型。

普通 softmax/sigmoid 路由最后会进入 `topk_softmax` / `topk_sigmoid`。Grouped top-k 在 `router/grouped_topk_router.py:28-161` 中有两条路径：

- sigmoid：CUDA `ops.grouped_topk` 可以把 scoring、bias、分组筛选、top-k、归一化放在一个 fused kernel 中。
- softmax：先在 Python 中 softmax，再调用 grouped CUDA kernel；Python fallback 的算法是先在 group 内算分数，再选 `topk_group` 个 group，mask 其他专家，最后从保留 group 中选 `top_k` 个 expert。

如果有 correction bias，选择分数和最终路由权重是两套值：bias 参与选 expert，原始概率参与加权。这一点是 DeepSeek/Bailing 类模型复现精度时的关键。

### D. MoERunner：统一 forward 编排

`vllm/model_executor/layers/fused_moe/runner/moe_runner.py:218-308` 的 `MoERunner` 是当前主执行器。它持有 router、`RoutedExperts`、shared experts、gate 和输入输出 transform，并注册为 `torch.ops.vllm.moe_forward` / `moe_forward_shared` 的 layer lookup。

在 CUDA 上，`MoERunner._select_forward` 选择 custom op；CPU/TPU 走 Python `_forward_impl`。一次 forward 的关键顺序是：

1. `apply_routed_input_transform`：普通 MoE 原样传递；latent MoE 把 routed 分支投影到 latent dim，同时保留原始 hidden 给 shared expert。
2. 对齐/补齐 hidden dim，适配某些 FP4/FlashInfer kernel 的布局要求。
3. 如果配置有 internal gate，计算 router logits；如果模型自己计算 gate，则直接使用传入的 logits。
4. 运行 shared expert，或者在支持 async 的 prepare/finalize 中与 routed dispatch 重叠。
5. routed experts 选择 modular 或 monolithic 执行。
6. combine 后做 output transform、routed scaling、shared expert 相加、最终 all-reduce 和 zero-expert 输出。

`_apply_quant_method`（`moe_runner.py:574-623`）清晰地体现了两种执行模型：

```text
modular:
  router.select_experts() -> topk_weights/topk_ids
  -> routed_experts.forward_modular()

monolithic:
  router_logits
  -> routed_experts.forward_monolithic()
  -> kernel 内部完成 routing + expert GEMM
```

`MoERunner.forward`（`moe_runner.py:668-777`）再把结果接回模型维度。shared expert 可以使用独立 CUDA stream；latent MoE 则在 routed 输出回到 full hidden 后才与 shared 输出相加。

### E. TP、EP、DP、PCP 与 token dispatch

`FusedMoEParallelConfig.make`（`fused_moe/config.py:1208-1255`）的核心语义是：

- 不启用 EP：专家权重沿 TP 切分，所有 rank 都有每个逻辑 expert 的一个 TP shard。
- 启用 EP：flatten 后的设备数成为 `ep_size`，每个 rank 拥有一组完整 expert，MoE 内部 `tp_size=1`。
- DP/PCP/SP 与 EP 同时存在时，token 必须先发到拥有目标 expert 的 rank，再在 combine 阶段返回原 token 所在 rank。

`use_all2all_kernels` 在 `config.py:1055-1059` 定义为 `EP && (DP > 1 || PCP > 1 || SP)`。因此“启用 EP”不必然意味着每个配置都做 all-to-all；单一 EP/TP 形态可以走本地路径。

`all2all_utils.py:118-356` 选择 prepare/finalize：

1. 没有 all-to-all 时，单卡走 `no_dp_ep`；普通 DP 回退到 `naive_dp_ep` 的 AllGather + ReduceScatter。
2. DeepEP HT/LL/v2、MoRI、FlashInfer NVLink、NIXL EP 负责异构的 dispatch/combine。
3. dispatch 的输出通常包含 quantized activation、scale、expert token metadata，以及可能从其他 EP rank 收集来的 top-k IDs/weights。
4. expert kernel 只计算本 rank 负责的 physical experts；finalize 再按原 token 顺序合并。

EPLB 还会把一个 logical expert 复制为多个 physical experts，根据实时 load 选择副本。它改变的是“expert 放置/路由后的物理 ID”，不改变模型 checkpoint 的 logical expert 语义。

## 从 Python 到 CUDA/Triton 算子

### 1. Modular kernel 的抽象层

`vllm/model_executor/layers/fused_moe/modular_kernel.py` 将 MoE 拆成三层：

```text
FusedMoEPrepareAndFinalize
  -> FusedMoEExperts
  -> FusedMoEKernel
```

- `PrepareAndFinalize` 处理输入量化、EP dispatch、token layout 和结果 combine。
- `FusedMoEExpertsModular` 只接收 `topk_ids/topk_weights`，执行专家计算。
- `FusedMoEExpertsMonolithic` 接收 router logits，让外部 fused kernel 自己做 routing。

`FusedMoEKernelModularImpl.apply`（`modular_kernel.py:1417-1510`）的顺序是：

```text
prepare
  -> quantize / dispatch / permute
fused_experts.apply
  -> GEMM1
  -> activation
  -> GEMM2
finalize
  -> top-k weight / unpermute / reduce / EP combine
```

### 2. Token layout 与数学计算

设输入为 `X[T, H]`，每个 token 选 `K` 个 expert：

1. router 产生 `topk_ids[T, K]` 和 `topk_weights[T, K]`。
2. 按 expert 对 token assignment 排序，得到每个 expert 的连续 token 段。
3. 把 token 复制/permute 成 `X_perm[M, H]`，其中 `M` 约为 `T * K`，并为 GEMM block size 补 padding。
4. 对每个 expert 做：

   ```text
   U = X_perm @ W13[e]^T
   V = activation(U)       # 常见为 silu(gate) * up
   Y = V @ W2[e]^T
   ```

5. 每个 assignment 乘自己的 `topk_weight`，按 `(token, top-k slot)` 求和，恢复成 `Y[T, H]`。

当 `topk=1` 时，某些 backend 可以把 weight 乘到输入侧；当 `topk>1` 时，通常在第二个 GEMM 输出侧做加权。

### 3. vLLM 自带 `_moe_C` C++/CUDA 算子

`csrc/libtorch_stable/moe/torch_bindings.cpp:6-150` 注册了 `_moe_C`：

| Python-visible op | 实现作用 | 源码 |
| --- | --- | --- |
| `topk_softmax` | softmax + top-k + optional bias/renormalize | `topk_softmax_kernels.cu:822-860` |
| `topk_sigmoid` | sigmoid + top-k + scaling/renormalize | `topk_softmax_kernels.cu:862-900` |
| `topk_softplus_sqrt` | DeepSeek V4 风格 sqrtsoftplus + bias + top-k | `topk_softplus_sqrt_kernels.cu:837-878` |
| `grouped_topk` | group 选择 + expert 选择，支持 sigmoid/no-activation 两种 scoring | `grouped_topk_kernels.cu:1447-1525` |
| `moe_align_block_size` | 统计每个 expert token 数，按 block size padding，生成 `sorted_token_ids/expert_ids` | `moe_align_sum_kernels.cu:625-723` |
| `batched_moe_align_block_size` | batched expert layout 的 alignment | `moe_align_sum_kernels.cu:725-755` |
| `moe_permute` / `moe_unpermute` | CUDA 12+ 的高效 token 排序、复制、恢复 | `moe_permute_unpermute_op.cu:53-174`、`:275-324` |
| `moe_sum` | 把 `[T, K, H]` partial output 求和为 `[T, H]`，支持 EP pad-aware skip | `moe_align_sum_kernels.cu:757-825` |
| `moe_wna16_gemm` | W4A16/W8A16 weight-only quantized expert GEMM | `moe_wna16.cu:16-30`、`:277-320` |
| `dsv3_router_gemm` | DeepSeek V3 专用小 batch router GEMM；限制 token 数 <= 16 | `dsv3_router_gemm_entry.cu:112-145` |

这里的 CUDA 算子承担的是 routing、layout、量化 GEMM 和 reduce 的基础积木；FlashInfer、DeepGEMM、AITER、CUTLASS、TensorRT-LLM 等专家实现则可能来自外部库，并不都位于 vLLM 的 `csrc/` 目录。

`moe_align_block_size` 的核心原因是：每个 expert 收到的 token 数不同，而 tiled GEMM 要求 M 维按 `BLOCK_SIZE_M` 对齐。它还可以接受 `expert_map`，把不属于当前 EP rank 的 expert 标记为无效，避免本 rank 计算远端 expert。

### 4. 旧/回退的 Triton `fused_experts` 路径

除了新的 modular kernel，`vllm/model_executor/layers/fused_moe/fused_moe.py:1587-1853` 仍保留功能式 `fused_experts`：

1. `_prepare_expert_assignment` 选择 naive assignment，或调用 `moe_align_block_size` 生成排序和 padding。
2. `dispatch_fused_moe_kernel` 在普通场景使用 Triton `fused_moe_kernel`；W4A16/W8A16 会在条件满足时切换到 C++/CUDA `moe_wna16_gemm`，否则使用 Triton WNA16 kernel。
3. 第一个 grouped GEMM 写入 `intermediate_cache1`。
4. `apply_moe_activation` 做 SiLU/GELU/SwiGLU/SituGLU 等激活。
5. 第二个 grouped GEMM 写入 `intermediate_cache3`。
6. `ops.moe_sum` 合并 top-k partial output。

这条路径的语义和新 modular 路径相同，但调度层次更旧、量化和 EP overlap 能力较少。读性能问题时不能只搜索 `MoERunner`，还需要检查 `fused_experts` 是否被具体 quant method 选中。

### 5. 专家后端选择

未量化 MoE 的选择入口是 `oracle/unquantized.py:206-324`：

- CUDA：候选通常为 FlashInfer TRTLLM、FlashInfer CUTLASS、Triton、Batched Triton；设备架构、DP、激活类型、LoRA 和 shape 会动态调低某些候选的优先级。
- ROCm：AITER、Triton、Batched Triton。
- XPU/CPU/TPU：对应平台实现。
- 如果启用 `moe_backend`，用户配置可以覆盖自动 oracle，但 backend 仍会通过 `is_supported_config` 检查 shape/量化/并行条件。

量化时，`quant_config.get_quant_method` 在 `RoutedExperts._get_quant_method` 中优先返回具体量化 MoE method；没有量化配置才使用 `UnquantizedFusedMoEMethod`。因此“FP8 MoE”“Triton MoE”“DeepEP MoE”分别属于不同轴：前者偏权重/激活表示，第二个偏专家计算 kernel，第三个偏跨 rank token 通信。

## vLLM-omni 的 MoE 形态

### A. 语言模型：大多复用上游 vLLM

Qwen3-Omni 的 Thinker 从 `vllm.model_executor.models.qwen3_moe.Qwen3MoeForCausalLM` 继承；Talker 的 `qwen3_omni_moe_talker.py:348-368` 也明确说明其语言模型复用 Qwen3 MoE 和 shared-expert 支持，只把 embedding/LM head 改成 codec 任务。因此它们不是 vllm-omni 自己的第三套专家 kernel，而是上游 `FusedMoEFactory/MoERunner` 的多阶段包装。

Ming/Bailing 是更特殊的语言 MoE：

- `modeling_bailing_moe_v2.py:214-282` 使用 sigmoid + expert bias + grouped top-k，并把原始 logits 转成 fp32 做选择。
- `:307-363` 支持 `topN` 和 `MultiRouter`。
- `:395-424` 根据 text/image/audio mask 在三个 router 的结果之间逐 token 选择，再把 `(topk_weight, topk_idx)` pack 到 `gating_output`，通过 `custom_routing_function` 交给上游 FusedMoE。

这体现了 vLLM-omni 的常见扩展方式：模型可以保留自己的路由精度和模态规则，但专家计算、EP、量化和权重加载仍复用上游 runner。

### B. 扩散模型：FusedMoE adapter

`vllm_omni/diffusion/layers/fused_moe.py:52-127` 的 `FusedMoE` 不是另写一套专家计算。它在 `__new__` 中调用上游 `vllm.model_executor.layers.fused_moe.FusedMoE`，拿到真正的 `MoERunner`，然后：

- 在每次 forward 前设置 `ForwardContext.num_tokens`，避免路由 token 数缺失造成静默错误。
- 在 diffusion-only DP 下收集各 rank token 数，填充 vLLM 的 DP metadata。
- 调用 platform hook，为 CUDA/ROCm/NPU/XPU 加入额外 fused-MoE runtime 准备。

扩散 worker 在 `vllm_omni/diffusion/worker/diffusion_worker.py:276-329` 把 diffusion parallel config 翻译成 vLLM config：tensor/data/expert parallel、quantization 和 `moe_backend` 都进入统一的 `FusedMoE` 选择。HunyuanImage-3 的 block 在 `diffusion/models/hunyuan_image3/hunyuan_image3_transformer.py:1556-1648` 使用这个 adapter。

### C. HunyuanImage-3：FP32 custom routing

`vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py:1190-1354` 是 vllm-omni 中最典型的“自定义路由 + 统一专家 kernel”实现：

1. gate 权重使用 fp32。
2. hidden cast 到 fp32 后做 softmax、top-k 和 `clamp(min=1e-8)` 归一化。
3. 把 `topk_weights` 与 `topk_indices` pack 到一个 tensor。
4. `custom_routing_function` 只负责解包，绕过上游 bf16 `topk_softmax` CUDA op。
5. 解包后的 top-k 结果交给 `SharedFusedMoE`，shared MLP、EP、quant kernel 仍由统一 runner 处理。

这不是新的专家 GEMM 变体，而是为了逐 bit 对齐 Hugging Face reference 的路由变体。

### D. vLLM-omni 中没有走统一 FusedMoE 的原生实现

这些实现也属于 MoE 或 MoE-like，但不能假设它们会经过 `MoERunner`：

1. **HiDream：原生 ModuleList + 顺序 expert loop**

   `diffusion/models/hidream_image/hidream_image_transformer.py:347-480` 使用 DeepSeek 风格的 softmax top-k、shared expert 和 token 重排，但 `moe_infer` 逐 expert 顺序调用 ModuleList。源码明确标注 sequential routing 可能成为瓶颈，没有使用上游 grouped GEMM。

2. **LingBot Video：grouped top-k + `torch._grouped_mm`**

   `diffusion/models/lingbot_video/lingbot_video_transformer.py:299-365` 先以 softmax/sigmoid、bias 和 group-limited top-k 得到 token assignment；`:495-532` 使用 `torch._grouped_mm` 做 w1/w3/w2 三次 grouped GEMM，并在不支持 CUDA grouped MM 时回退到 expert loop；`:583-601` 再合并 shared expert。

3. **MagiHuman：按 modality dispatch 的 per-expert TP linear**

   `diffusion/models/magi_human/magi_human_dit.py:337-366` 的 `NativeMoELinear` 根据 `ModalityDispatcher` 把 token 分给不同 expert，然后逐 expert 做 BF16 linear。`:372-503` 的 `MoEQKVParallelLinear`、`MoEColumnParallelLinear`、`MoERowParallelLinear` 将每个 modality expert 包成 vLLM TP linear。它是硬 modality dispatch，不是 learned top-k router；甚至 attention QKV/O projection 也按 modality 分 expert。

4. **MammothModa2：understanding/generation 二路 hard-mask MoE**

   `model_executor/models/mammoth_moda2/mammoth_moda2.py:74-92` 按层类型和层号决定是否开启 MoE；`:95-176` 的 `moe_forward` 用 `gen_token_mask` 将 token 分给 `und_expert` 或 `gen_expert`，对混合 batch 做 permute、两路计算和 inverse reorder。这里没有 top-k、router logits 或 expert all-to-all。

5. **Bagel/Lance 的 `moe_gen` / MoT 分支**

   Bagel 在 `model_executor/models/bagel/bagel.py:487-533` 为生成模式安装独立的 `qkv_proj_moe_gen`、`o_proj_moe_gen`、`mlp_moe_gen`；`:898-984` 只把 VAE latent tokens 送进 generation-mode 参数，其他 token 仍走 understanding-mode 参数。这是 mode-of-thought/模态分支，不应与 sparse token-choice MoE 的 top-k 统计混为一谈。

6. **Wan2.2 的两级 denoiser cascade**

   `diffusion/models/wan2_2/pipeline_wan2_2.py:317-345` 根据是否存在 `transformer_2` 和 `boundary_ratio` 加载一个或两个 transformer；`:665-672` 按 denoising timestep 的 boundary 选择 high-noise/low-noise 模型。这是“模型级 MoE/cascade”，不是一个 transformer block 内把 token 路由给多个 experts。

因此，vLLM-omni 至少同时包含三种不同语义：

- learned token-choice sparse MoE；
- hard mask/modality/mode expert dispatch；
- denoising-stage model cascade。

如果只研究 CUDA fused sparse MoE，应以 Qwen3/Bailing/Hunyuan adapter 和上游 `FusedMoE` 为主；HiDream、MagiHuman、Mammoth、Bagel、Wan2.2 需要作为旁支单独阅读。

## 一次 CUDA MoE forward 的具体数据流

下面用标准 modular、EP 开启、SwiGLU、`top_k=2` 的情形说明：

```text
hidden_states                 [T, H]
router_logits                 [T, E]
       |
       | topk_softmax/sigmoid/grouped_topk
       v
topk_ids                      [T, 2]
topk_weights                  [T, 2]
       |
       | EPLB logical -> physical
       | EP all-to-all dispatch (optional)
       v
local expert token buffer    [M_local, H]
       |
       | permute/sort + block padding
       v
W13[e] GEMM                  [M_local, 2I]
       |
       | silu(gate) * up / other MoEActivation
       v
intermediate                 [M_local, I]
       |
       | W2[e] GEMM
       v
partial expert output        [M_local, H]
       |
       | top-k weight + unpermute + sum
       | EP all-to-all combine / reduce
       v
routed_output                [T, H]
       |
       | routed output transform / scaling
       | shared expert output add
       | final TP/EP all-reduce if needed
       v
next transformer block       [T, H]
```

几个容易出错的细节：

- `topk_ids` 在 router 阶段是 global logical ID；进入 EP kernel 前可能已经被 `expert_map` 或 EPLB 变成 physical/local ID。
- `sorted_token_ids` 中的 padding row 不是有效 token；`expert_ids=-1` 表示该 block 不应计算。
- `topk_weights` 可能是 fp32，即使 hidden/weights 是 bf16；这是为了稳定 combine 和保持 reference 语义。
- shared expert 不一定被 append 为一个 routed slot。ROCm AITER fusion 可以把 shared expert append 到 expert table 并用一个 grouped GEMM 合并；普通路径则独立执行后相加。
- `reduce_results=False` 不是“永远不 reduce”，工厂会结合 EP/sequence parallel 和 kernel 是否已经 reduce 来决定延迟或拆分 all-reduce。

## 如何定位性能或正确性问题

建议按以下顺序定位，而不是只看最终 GEMM：

1. **模型参数/路由语义**：检查模型 block 是否传入了正确的 `scoring_func`、`renormalize`、group/bias、shared expert 和 activation。
2. **router 输出**：检查 `topk_ids/topk_weights` 的 dtype、是否使用 bias 前/后的分数、是否在 fp32 中做 top-k。
3. **专家映射**：检查 `ExpertMapManager`、EPLB logical-to-physical 表、`expert_map` 是否与 checkpoint expert ID 对齐。
4. **并行配置**：确认 TP 和 EP 的语义；EP 下每个 rank 应拥有完整的 local experts，而不是 TP shard。
5. **prepare/finalize**：检查 all-to-all backend、DP token padding、batched activation format 和 async overlap。
6. **kernel oracle**：记录实际选中的 quant method、expert backend 和 activation format；不要根据命令行模型名推断 kernel。
7. **低层 op**：最后检查 `moe_align_block_size`、permute/unpermute、WNA16 或 Triton grouped GEMM 的 shape/stride/block size。

对 vLLM-omni 还要先确认当前模型是否真的进入上游 `MoERunner`：

- 能在 `diffusion/layers/fused_moe.py` 或模型代码中看到上游 `FusedMoE`，才沿 `MoERunner` 追踪。
- 看到 `ModuleList`、`torch._grouped_mm`、`ModalityDispatcher` 或 `gen_token_mask` 时，应改查模型自己的 dispatch/restore 逻辑。
- 看到 Wan2.2 的 `transformer_2` 和 `boundary_ratio` 时，应查 pipeline 的 timestep stage switch，而不是寻找 token-level top-k。

## 关键源码索引

- 顶层：`vllm/entrypoints/llm.py`、`vllm/v1/engine/core.py`、`vllm/v1/worker/gpu_model_runner.py`
- 统一工厂：`vllm/model_executor/layers/fused_moe/layer.py`
- 路由：`vllm/model_executor/layers/fused_moe/router/`
- runner：`vllm/model_executor/layers/fused_moe/runner/moe_runner.py`
- 专家权重/quant method：`vllm/model_executor/layers/fused_moe/routed_experts.py`
- modular kernel：`vllm/model_executor/layers/fused_moe/modular_kernel.py`
- dispatch/combine：`vllm/model_executor/layers/fused_moe/prepare_finalize/`、`all2all_utils.py`
- backend oracle：`vllm/model_executor/layers/fused_moe/oracle/`
- legacy Triton：`vllm/model_executor/layers/fused_moe/fused_moe.py`
- vLLM CUDA/C++：`csrc/libtorch_stable/moe/`
- vLLM-omni adapter：`vllm_omni/diffusion/layers/fused_moe.py`
- vLLM-omni native variants：`hidream_image_transformer.py`、`lingbot_video_transformer.py`、`magi_human_dit.py`、`modeling_bailing_moe_v2.py`、`hunyuan_image3.py`
