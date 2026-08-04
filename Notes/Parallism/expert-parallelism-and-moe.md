---
type: Note
related_to: "[[pipeline-parallelism]]"
status: Active
---

# 专家并行与 MoE（EP / DeepEP / Hybrid-EP）

> MoE（Mixture of Experts）把 FFN 换成多个**专家**，每个 token 只激活 top-k 个。稀疏激活让"总参数大、实际计算小"，但专家数量多到单卡放不下时，需要把专家**分布到多卡**——这就是**专家并行（Expert Parallelism, EP）**。

## 1. MoE 基础

```
        x
        │
   ┌────┴────┐
   │ router  │── top-k 选择 → 每个 token 选 k 个专家
   └────┬────┘
  ┌─────┼─────┐
  E0    E1 ... EN-1   （N 个专家，可能分布在不同卡上）
  └─────┼─────┘
        y（加权合并）
```

- 典型配置：DeepSeek-V3 共 **256 个专家**，每 token 激活 8 个 + 1 个共享专家；总参 671B、激活参 ~37B。
- 关键技术点：**负载均衡**（capacity factor、auxiliary loss）、**路由**（top-1/top-2/Expert Choice）、**共享专家**。

## 2. 为什么需要 EP：从 DP/TP 到 EP

| 并行方式 | 专家怎么放 | 问题 |
|---|---|---|
| DP | 每卡完整副本 | 显存冗余：671B 模型不可能每卡一份 |
| TP | 单个专家跨卡切分 | 专家数不变，all-reduce 通信随 TP 增大 |
| **EP** | 不同专家放不同卡 | 需要 **all-to-all** 把 token 送到对应专家 |

**EP 本质**：token 是"稀疏"的，每个 token 只需要少数专家 → 用 all-to-all 动态路由，而不是像 TP 那样所有卡算同一个算子。

## 3. EP 的通信结构：dispatch / combine

每个 MoE 层两段 all-to-all：

1. **Dispatch**：`[token → 专家]`——各 rank 把自己 token 的隐藏状态发给目标专家所在的 rank；
2. **Combine**：`[专家 → token]`——把专家输出按权重加权后送回原 rank。

```
rank0: tokens t1 t2 ──dispatch──▶ rank0 的专家 E0,E1
rank1: tokens t3 t4 ──all-to-all▶ rank1 的专家 E2,E3   （E 的权重常驻对应卡）
         ◀────── combine ──────┘
```

**通信量**：`O(EP × batch × hidden)`，与序列长度无关；在 130 TB/s 级 NVLink 上可控，跨节点则高度依赖 IB 带宽。

## 4. 工业实现

### DeepSpeed-MoE

- 支持大规模 MoE 训练；提供 **PR-MoE**（小专家置换合并，减 12.5% 参数量）、**MoS**（混合序列/并行策略）等压缩方案；
- 层级式 all-to-all 优化跨节点通信。

### Megatron-MoE（Megatron-Core）

- `--expert-model-parallel-size` 设置 EP 度；`--moe-grouped-gemm` 用分组 GEMM 加速专家计算；
- **TP + EP 组合必须开序列并行**（`--sequence-parallel`），否则激活重排与通信组不一致；
- 官方推荐配置示例（Megatron 指南）：
  - Mixtral 8x7B（64 卡）：TP=1, PP=4, EP=8；
  - Mixtral 8x22B（256 卡）：TP=4, PP=4, EP=8；
  - DeepSeek-V3（1024 卡）：TP=2, PP=16, EP=64。

### DeepEP（DeepSeek 开源，2025）

专门为 MoE 训练+推理设计的 **EP 通信库**：

- **普通 kernel**：用于训练/prefill 阶段，吞吐优先；
- **低延迟 kernel**：用于 decode 阶段，延迟优先（MoE 推理逐 token 路由抖动大）；
- **SM 资源分区**：调度部分 SM 专职做 dispatch/combine，与专家计算**真正重叠**，把通信隐藏进计算；
- 同时支持 NVLink（节点内）与 InfiniBand（跨节点），是 DeepSeek-V3/R1 训练推理的通信底座。

### Hybrid-EP（NVIDIA，2026）

- 把 EP 与 FSDP 式分片结合：**热专家 EP 复制、冷专家分片**（或按内存/带宽混合放置），解决 EP 下专家权重复制导致的显存浪费；
- 报告比纯 DeepEP 再提升 ~8% 训练性能；也支持与 MTP（多 token 预测）等新特性组合。

## 5. 负载均衡与路由技术（EP 的隐形瓶颈）

| 技术 | 思路 | 出处 |
|---|---|---|
| Capacity Factor | 每专家固定容量，超载 token 丢弃（需重算） | Switch Transformer |
| Auxiliary Loss | 辅助损失惩罚不均匀路由 | GShard / Switch |
| Expert Choice | 专家"选" token（top-k 反过来） | ECR（2022） |
| 偏置采样 + 低 aux loss | DeepSeek-V3：路由偏置动态调整，几乎去掉 aux loss | DeepSeek-V3 |

> [!IMPORTANT]
> EP 的吞吐上限往往不是算力，而是 **all-to-all 带宽 + 路由失衡**。负载均衡不是一次性超参，需要按训练过程动态监控（router 熵、专家利用率）。

## 6. 选型建议

- 专家总参 放不下单卡 → **EP**（配合 DeepEP 之类内核）；
- EP + TP 同时用 → 记得开 **序列并行**；
- 显存仍紧张 → 尝试 **Hybrid-EP**（FSDP 混合分片）；
- 跨节点训练 → 用 **DualPipe/V-Shape 流水**把 EP 的 all-to-all 通信与计算重叠（DeepSeek-V3 路线）。

## 参考

- Switch Transformer：https://arxiv.org/abs/2101.03961 ；GShard：https://arxiv.org/abs/2006.16668
- Megatron-MoE：https://arxiv.org/abs/2202.08906
- DeepEP：https://arxiv.org/abs/2501.05805
- DeepSeek-V3 技术报告：https://arxiv.org/abs/2412.19437
- NVIDIA Hybrid-EP 博客：https://developer.nvidia.com/blog/?p=112038/
