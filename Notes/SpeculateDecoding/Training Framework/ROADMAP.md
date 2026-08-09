# 投机解码训练框架导读路线

本文按“先建立一轮训练的心智模型，再理解为什么需要复杂 runtime”的顺序阅读 `AngelSpec` 与 `SpecForge`。

## 0. 先建立一条主线

先记住下面这条链路；后续模块都是在解决其中某个环节的算法或系统问题。

```text
YAML 配置
  -> target prefill / feature capture
  -> feature tensor store
  -> draft forward + distillation loss
  -> FSDP optimizer step / checkpoint
  -> serving engine 的真实 acceptance evaluation
```

这里的 target 通常冻结，draft 才是被优化的模型。投机训练不是为了改变 target 的输出质量，而是让 draft 更准确地预提议 token，从而提高 target 验证后的连续接受长度和解码吞吐。

## 1. 第一遍：用 AngelSpec 跑通端到端直觉

### 1.1 读 README，先认识产物和角色

阅读：

- `AngelSpec/README.md`
- `AngelSpec/docs/concepts/disaggregated_architecture.md`
- `AngelSpec/docs/concepts/draft_model_family.md`

需要回答：

- 谁是 target，谁是 draft？
- 为什么 hidden states 是训练数据的一部分？
- 为什么 target inference 和 draft training 要拆成不同 GPU worker group？
- Eagle3/MTP 的自回归 TTT 与 DFlash/DFly 的 block-parallel 有何取舍？

读完应形成的图：

```text
dataset prompt
  -> target prefill
  -> selected-layer hidden states
  -> Mooncake
  -> draft trainer
  -> draft checkpoint
  -> spec-decode evaluation
```

注意：prefill/capture 不等同于 RL rollout。它主要是在冻结 target 上提取教师特征，而不是生成轨迹、计算 reward 并更新 policy。

### 1.2 从入口看角色如何被组装

阅读：

- `AngelSpec/angelspec/train_entry.py`
- 重点：`train_async_no_generation`

关注它创建的三类角色：

1. `AsyncTrainingController`：管理 dataset、样本调度、背压、保存和评估。
2. `AsyncInferenceManager`：持有 target inference engines，提交 prompt，获得 feature 的引用。
3. `TrainerActor` / `TrainGroup`：每个 actor 持有一个 draft training rank。

阅读问题：为什么 controller 不直接携带 hidden-state tensor？

答案：控制面只负责调度；大 tensor 必须绕过控制器，经 Mooncake 在 producer/consumer 之间流动，否则 Python/Ray object transfer 会成为瓶颈并增加显存复制。

### 1.3 看一轮训练的调度

阅读：

- `AngelSpec/angelspec/controller/loop.py`
- `AngelSpec/angelspec/controller/inference_manager.py`
- `AngelSpec/angelspec/controller/training_controller.py`

一轮的逻辑是：

```text
controller dispatches prompts
  -> inference manager asks target engines to prefill and capture
  -> engines write feature tensors to Mooncake and return lightweight keys
  -> controller assigns keys to each draft DP rank
  -> every trainer rank consumes its assigned micro-batches
  -> all ranks complete one optimizer step
```

重点观察：in-flight pool/backpressure、每 rank 的 batch 对齐、gradient accumulation 和 checkpoint/eval 是以 optimizer step 而不是任意 micro-batch 为边界的。

### 1.4 看 feature 如何变成训练 batch

阅读：

- `AngelSpec/angelspec/training/trainer.py`
- `AngelSpec/angelspec/training/data_fetcher.py`

关键链路：

```text
Mooncake key
  -> MooncakeDataFetcher
  -> collator / loss mask / optional packing
  -> tensor batch
  -> trainer._train_step(...)
```

要理解的点：

- 训练样本除了 `input_ids` 还有哪些 feature；
- 为什么要预取，以及为什么预取时可以先放 CPU；
- 为什么 packing 必须避免跨文档 attention、anchor 和 position-id 污染；
- 为什么长上下文需要 USP，而不是单纯增大 batch。

### 1.5 最后才深入某个算法与 loss

建议先选一种方法读透：

- 自回归路径：`angelspec/training/eagle3_trainer.py` 与 `angelspec/models/eagle3.py`；
- block-parallel 路径：`angelspec/training/dflash_trainer.py` 与 `angelspec/models/dflash.py`；
- MTP 路径：`angelspec/training/mtp_trainer.py` 与 `angelspec/models/mtp.py`。

建议顺序：Eagle3 -> DFlash -> MTP/DFly/DSpark。

阅读时一直追问：

```text
target feature 是什么？
draft 如何用它提出多个 token？
teacher target 是硬 token、软分布还是 target hidden state？
多步位置如何处理 draft 自己造成的上下文偏移？
最终 metric 如何关联真实 acceptance length？
```

## 2. 第二遍：用 SpecForge 理解生产化边界

AngelSpec 先给出“训练如何发生”的直觉；SpecForge 的价值在于把在线/离线 feature 流、恢复语义和 serving 兼容性拆成清晰的平面。

### 2.1 从 Runtime Architecture 开始

阅读：

- `SpecForge/README.md`
- `SpecForge/specforge/runtime/ARCHITECTURE.md`
- `SpecForge/specforge/runtime/CONTRACTS.md`

记住 SpecForge 的标准链路：

```text
PromptTask
  -> RolloutWorker
  -> SampleRef
  -> FeatureDataLoader
  -> TrainBatch
  -> TrainerCore + DraftTrainStrategy
  -> durable acknowledgement
```

它将运行方式分成三种：

- colocated offline：读取已落盘的 feature；
- disaggregated offline：producer 发布固定 manifest，consumer 训练；
- online：patched SGLang 写 Mooncake，producer/consumer 使用一次性 feature stream。

### 2.2 先读 contracts：什么可以跨平面传递

阅读：

- `SpecForge/specforge/runtime/contracts.py`
- `SpecForge/specforge/runtime/control_plane/controller.py`

最重要的约束：`PromptTask` 和 `SampleRef` 只能包含 metadata，不能包含 tensor；`TrainBatch` 才是第一份携带 tensor 的跨模块数据结构。

这不是形式主义。它使得控制面可以做调度、去重、重试和 SQLite 持久化，而不会把超大 hidden states 放进队列、日志或数据库。

### 2.3 再读 producer：特征是怎样生成和发布的

阅读：

- `SpecForge/specforge/inference/rollout_worker.py`
- `SpecForge/specforge/inference/capture.py`
- `SpecForge/specforge/inference/adapters/server_capture.py`

关键过程：

```text
lease PromptTask
  -> request SGLang capture
  -> verify feature name / layer id / shape / target representation
  -> FeatureStore.put or adopt server-produced ref
  -> commit SampleRef to metadata ledger
```

重点：capture contract 的价值是尽早发现 layer-id、width、vocab map 或 hidden-state representation 不匹配，而不是让错误在 trainer 中以难以解释的 loss 崩溃形式出现。

### 2.4 再读 consumer：引用如何安全地变成一步优化

阅读：

- `SpecForge/specforge/runtime/data_plane/feature_dataloader.py`
- `SpecForge/specforge/training/trainer.py`
- `SpecForge/specforge/training/controller.py`

关键过程：

```text
SampleRef + FeatureStore
  -> get tensors / collate
  -> TrainBatch
  -> Strategy.forward_loss
  -> backward
  -> optimizer boundary
  -> durable ack and feature cleanup
```

必须理解为什么 ack 要在 optimizer step 后发生：如果在读取 feature 后立即确认，训练进程在反向或优化前失败会丢数据；如果永不确认，特征会泄漏且 producer 的 in-flight 流量无法下降。

### 2.5 最后看算法如何插入统一训练循环

阅读：

- `SpecForge/specforge/training/strategies/base.py`
- `SpecForge/specforge/algorithms/eagle3/providers.py`
- `SpecForge/specforge/algorithms/dflash/providers.py`

这里的核心设计是：

```text
TrainerCore 不知道算法、online/offline 或部署拓扑
DraftTrainStrategy 知道需要哪些 feature、如何 forward、如何计算 loss
AlgorithmProvider 知道该算法支持哪些 capture contract、模型和配置
```

因此新增一个 draft 算法，不应复制整套 trainer；应定义模型、feature contract、collator/capture、strategy forward-loss 和 checkpoint/export 规则。

## 3. 建议的实际阅读节奏

### 第一次：只要跑通概念，不钻实现

1. AngelSpec README 与 `disaggregated_architecture.md`。
2. AngelSpec `draft_model_family.md`。
3. SpecForge `runtime/ARCHITECTURE.md`。

目标：能徒手画出 target feature capture 到 draft optimizer step 的图。

### 第二次：沿一条具体实现调用链追踪

1. AngelSpec `train_entry.py`。
2. `controller/loop.py`。
3. `inference_manager.py`。
4. `trainer.py` / `data_fetcher.py`。
5. 选择 Eagle3 或 DFlash 的 trainer 和 model。

目标：知道每个 tensor 在何处产生、通过何种存储移动、在哪个函数第一次参与 loss。

### 第三次：理解为什么 online runtime 复杂

1. SpecForge `contracts.py`。
2. `rollout_worker.py`。
3. `feature_dataloader.py`。
4. `trainer.py` / `training/controller.py`。
5. `dp_ack.py`、`ref_distributor.py`、`metadata_store.py`。

目标：能说明 partial batch、进程重启、重复投递和 feature cleanup 各由什么机制处理。

### 第四次：扩展与优化

关注：

- 如何为新 target 选择 capture layers、norm、vocab map；
- 如何为新 draft 定义 feature contract 和 service checkpoint；
- TTT、block-parallel、packing、USP 的性能与准确率权衡；
- 为什么 real acceptance evaluation 比离线 CE 更值得信任；
- 如何把 drafter 训练嵌入 RL rollout，加速训练但不混淆 draft distillation 与 reward-driven RL。

## 4. 一句话检查理解

能回答下面四个问题，说明已经掌握基本流程：

1. target 的哪些 feature 进入 draft 训练，为什么不能只给 token？
2. 为什么 target inference、tensor store、draft trainer 要解耦？
3. 为什么 loss/accuracy 下降不一定代表线上投机加速有效？
4. 为什么 online feature 的确认必须与 optimizer checkpoint 边界关联？
