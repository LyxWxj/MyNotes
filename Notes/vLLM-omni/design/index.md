---
type: Note
related_to: "[[vLLM-omni]]"
status: Active
---

# vLLM-Omni Design Documents

本目录包含vLLM-Omni的设计文档和架构规范笔记。

## 架构文档

- [Architecture Overview](architecture-overview.md) - vLLM-Omni整体架构设计

## 功能设计文档

### 加速与缓存
- [Async Chunk Design](async-chunk-design.md) - 异步分块处理设计
- [Cache-DiT](cache-dit.md) - Diffusion Transformer缓存加速
- [TeaCache](teacache.md) - 基于时间步相似性的缓存加速

### 并行策略
- [CFG Parallel](cfg-parallel.md) - 无分类器引导并行
- [Expert Parallel](expert-parallel.md) - 专家并行（MoE模型）
- [HSDP](hsdp.md) - 混合分片数据并行
- [Sequence Parallel](sequence-parallel.md) - 序列并行
- [Tensor Parallel](tensor-parallel.md) - 张量并行
- [VAE Parallel](vae-parallel.md) - VAE补丁并行

### 分布式与执行
- [Disaggregated Inference](disaggregated-inference.md) - 解聚推理
- [Ray-based Execution](ray-based-execution.md) - 基于Ray的分布式执行
- [Diffusion Step Execution](diffusion-step-execution.md) - 逐步扩散执行契约

## 模块设计文档

- [AR Module](ar-module.md) - 自回归模块设计
- [AsyncOmni Architecture](async-omni-architecture.md) - 异步Omni架构
- [DIT Module](dit-module.md) - 扩散模块架构设计
- [Entrypoint Module](entrypoint-module.md) - 入口点模块设计

## 功能特性

- [ComfyUI Integration](comfyui-integration.md) - ComfyUI集成
- [Custom Pipeline](custom-pipeline.md) - 自定义管道扩展指南
- [Sleep Mode](sleep-mode.md) - 睡眠模式

## 快速参考

### 并行策略对比

| 策略 | 用途 | 适用模型 | 关键参数 |
|------|------|----------|----------|
| TP | 分割模型权重 | 所有模型 | `tensor_parallel_size` |
| SP | 分割序列维度 | 长序列模型 | `ulysses_degree`、`ring_degree` |
| DP | 复制模型，分割批次 | 所有模型 | `data_parallel_size` |
| EP | 分割专家网络 | MoE模型 | `enable_expert-parallel` |
| CFG | 并行化正/负提示 | 所有扩散模型 | `cfg_parallel_size` |
| HSDP | 分片模型权重 | 大型模型 | `use_hsdp`、`hsdp_shard_size` |
| VAE | 并行化VAE编码/解码 | 扩散模型 | `vae_patch_parallel_size` |

### 缓存后端对比

| 后端 | 适用模型 | 加速倍数 | 描述 |
|------|----------|----------|------|
| TeaCache | 所有DiT模型 | 1.5x-2.0x | 基于时间步相似性的缓存 |
| Cache-DiT | 所有DiT模型 | 1.5x-2.0x | 动态块级缓存 |

### 连接器选择

| 场景 | 推荐连接器 | 备注 |
|------|-----------|------|
| 单节点 | SharedMemoryConnector | 自动配置 |
| 多节点（TCP） | MooncakeStoreConnector | 需要Mooncake Master |
| 多节点（RDMA） | MooncakeTransferEngineConnector | 最快 |
| 多节点（Yuanrong） | YuanrongConnector | 需要etcd |
