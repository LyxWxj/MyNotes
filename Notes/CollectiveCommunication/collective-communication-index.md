---
type: Note
related_to: "[[collective-communication-basics]]"
status: Active
---

# 集合通信与 NCCL 调优（索引）

> 集合通信（Collective Communication）是**所有并行技术的地基**：DP 的梯度同步、TP 的 all-reduce、SP 的 all-to-all、PP 的 p2p，底层全是集合通信原语。本目录单独成册，讲清楚**原语语义、算法与拓扑、NCCL 调优和问题排查**。

## 笔记导航

- [[collective-communication-basics|集合通信基础]] — 原语语义与通信量 / Ring / Tree / CollNet / NVLS / PAT / 带宽延迟模型 / 拓扑感知
- [[nccl-tuning-and-debugging|NCCL 调优与排查]] — 环境变量详解 / 调优方法论 / nccl-tests / profiling / 常见问题

## 与并行笔记的关系

并行笔记见 [[Parallel tech|并行训练技术全景]]（`Notes/Parallism/`）：

| 并行维度 | 依赖的集合通信 |
|---|---|
| DDP | all-reduce（= reduce-scatter + all-gather） |
| ZeRO-2/3、FSDP | reduce-scatter + all-gather |
| 张量并行 TP | 层内 all-reduce |
| 序列并行 Ulysses | all-to-all |
| Ring-Attention / CP | p2p 环传 |
| 流水线 PP | p2p send/recv |
| 专家并行 EP | all-to-all（dispatch/combine） |

## 快速上手

```bash
# 环境自检：打印 NCCL 版本与拓扑检测信息
NCCL_DEBUG=INFO python -c "import torch; torch.distributed.init_process_group('nccl'); print('ok')"

# 压测基准（NCCL 官方工具）
mpirun -np 8 ./build/all_reduce_perf -b 8M -e 8G -f 2 -g 8
# 关注输出中的 busbw（总线带宽）是否接近硬件理论值
```

## 参考

- NCCL 官方文档：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html
- nccl-tests：https://github.com/NVIDIA/nccl-tests
