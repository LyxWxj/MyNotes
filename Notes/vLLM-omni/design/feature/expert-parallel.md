---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
url: https://github.com/vllm-project/vllm-omni/blob/main/docs/design/feature/expert_parallel.md
---

# Expert Parallel（专家并行）

## 概述

### 什么是Expert Parallel？

Expert Parallel是混合专家（MoE）模型中的一种并行策略，将不同的专家网络分布在不同的计算设备上。每个设备只持有并计算一部分专家（本地专家），通过集体通信操作（如All-to-All、All-Gather）在远程设备之间分发和收集token。

| 后端 | 描述 |
|------|------|
| `allgather_reducescatter` | 基于allgather/reducescatter原语的默认后端 |

## 配置

通过`--enable-expert-parallel`标志启用EP，EP大小自动计算：

```
EP_SIZE = TP_SIZE × SP_SIZE × CFG_SIZE × DP_SIZE
```

**注意**：
- 专家并行仅适用于MoE模型
- EP组在每个管道阶段创建
- 底层通信模式为EP组内的All-to-All

## 实现步骤

### 步骤1：配置专家并行设置

```python
ep_size = 8
num_experts = 64
num_local_experts = num_experts // ep_size  # 每卡8个专家

assert num_experts % ep_size == 0, "Experts must be divisible by EP size"
```

### 步骤2：使用Sparse MoE Block启用EP路由

```python
class HunYuanSparseMoeBlock(nn.Module):
    def __init__(self, config, layer_id=-1, prefix=""):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.n_routed_experts = config.num_experts

        # 路由门控（在所有rank上复制）
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
        )

        # EP专家层（工厂加载平台特定实现）
        self.experts = HunyuanFusedMoE(...)
```

**关键点**：
- gate是`ReplicatedLinear`（在所有rank上复制）
- experts通过`HunyuanFusedMoE`工厂创建，自动处理EP分发

### 步骤3：初始化EP运行时

```python
op_name = "hunyuan_fused_moe"
current_omni_platform.prepare_diffusion_op_runtime(op_name)

impl = resolve_obj_by_qualname(
    current_omni_platform.get_diffusion_model_impl_qualname(op_name)
)
```

### 步骤4：专家权重映射与加载

每个rank只加载分配给其本地的专家权重：

```python
expert_mapping = HunyuanFusedMoE.make_expert_params_mapping(
    model=self,
    ckpt_gate_proj_name="gate_proj",
    ckpt_down_proj_name="down_proj",
    ckpt_up_proj_name="up_proj",
    num_experts=64,
    num_redundant_experts=0,
)

for name, loaded_weight in weights:
    if "mlp.experts" in name:
        expert_id = parse_expert_id_from_name(name)
        local_expert_start = (ep_rank) * num_local_experts
        local_expert_end = (ep_rank + 1) * num_local_experts

        if not (local_expert_start <= expert_id < local_expert_end):
            continue  # 跳过非本地专家权重
```

### 步骤5：EP前向传播

```python
def forward(self, hidden_states):
    # 1. 全局路由计算（所有token，所有专家分数）
    router_logits, _ = self.gate(hidden_states)

    # 2. EP分发和计算（HunyuanFusedMoE内部处理all_to_all）
    final_hidden_states = self.experts(
        hidden_states=hidden_states,
        router_logits=router_logits,
    )

    # 3. 添加共享专家输出（非EP，所有rank计算）
    if self.shared_mlp is not None:
        shared_out = self.shared_mlp(hidden_states)
        final_hidden_states = final_hidden_states + shared_out

    # 4. 张量并行All-Reduce
    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )

    return final_hidden_states.view(orig_shape)
```

## 测试

```bash
python text_to_image.py \
    --model Your-org/your-model \
    --prompt "a cup of coffee on the table" \
    --enable-expert-parallel \
    --tensor-parallel-size 8
```

## 参考实现

| 模型 | 路径 | 备注 |
|------|------|------|
| HunyuanImage3.0 | `vllm_omni/diffusion/models/hunyuan_image_3/` | 完整实现 |
| EP Tests | `tests/e2e/offline_inference/test_expert_parallel.py` | E2E测试 |

## 总结

1. 识别MoE层 - 定位每个transformer块中的路由器和专家网络
2. 验证EP约束 - 确保num_experts可被expert_parallel_size整除
3. 测试 - 使用enable-expert-parallel运行，检查内存减少、加速和输出质量
