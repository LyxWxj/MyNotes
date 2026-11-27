#### PreemptionManager

In `vllm_rlt/engine/preemption.py`

|成员|作用|
|---|---|
|`engine`|访问 StageEngine、Scheduler、ModelRunner、KVCacheManager 和 speculative runner|
|`snapshots`|按 request ID 保存 phase、hidden/token、可选 KVSnapshot、可选 PipelineState|
|`suspend_callback`|Scheduler 选 victim 后调用；队列迁移仍由 Scheduler 完成|
|`resume_callback`|Scheduler admission 时尝试恢复 snapshot|

**suspend：可运行 phase -> WAITING**

```text
Scheduler.preempt(requesting_request)
  -> 选择安全 victim
  -> PreemptionManager.suspend(victim, offload_kv=not priority_only)
       -> 首次 suspend 时 synchronize ModelRunner
       -> rewind staged prefill phase
       -> 保存 phase + hidden/token + 可选 pipeline state
       -> runner.release() / ModelRunner.suspend()  # 保留 RNG
       -> capacity pressure 时将 KVSnapshot 放到 CPU 并 free KV allocation；priority-only 时保留 resident KV
  -> Scheduler.enqueue(victim, WAITING)
```

|模式|snapshot 中的 `kv`|后续行为|
|---|---|---|
|`offload_kv=True`|CPU `KVSnapshot`|释放 device pages；resume 时重新 allocate + restore|
|`offload_kv=False`|`None`，KV allocation 保留|只恢复 hidden/token/pipeline state，不重复分配 KV|
|已有 resident snapshot 再遇压力|先前状态仍在 snapshots|可再次 suspend 转成 CPU KV snapshot，不重复逻辑挂起；admission/恢复容量不足时可以依次回收此类 WAITING resident allocation|

|victim 检查|条件|
|---|---|
|phase|`RECEIVING` 不可抢占；`WAITING` 只有在仍持有 resident KV 且发生 capacity pressure 时才可被 offload|
|输出|`num_output_placeholders == 0`|
|speculation|`num_speculative_rounds == 0`|
|保护集合|不能是当前 requesting request 或本批 `excluded` request|
|transfer|不能持有 transfer lease|
|priority-only|只允许抢占 priority 数值更大的 request|
|排序|priority 优先，再按 `required_blocks(position + 1)` 选择 footprint 更大的 request|

**resume：WAITING -> snapshot 中的原 phase**

|返回值|含义|
|---|---|
|`None`|没有 snapshot，继续普通 admission|
|`False`|CPU KV snapshot 存在但 KV 暂不可分配，保留 WAITING|
|`RequestPhase`|KV/state/pipeline 恢复成功；Scheduler 重新 enqueue 到该 phase|

```text
resume(request)
  -> 若 kv != None：allocate frontier + restore KVSnapshot
  -> restore hidden/input token 到 cache device
  -> speculative runner.restore_pipeline(...)
  -> 如果曾 offload KV 且设备为 CUDA：current stream synchronize，保证恢复数据可见
  -> 删除 snapshot，返回 saved RequestPhase
```

挂起释放 runner slot，但保留 request RNG；终止路径才调用 `release()` 并删除

RNG generator。`discard_snapshot()` 只丢弃保存记录；正常 finish/abort 路径仍由

Scheduler/Engine 释放 request 的剩余 KV 和生命周期资源。

### PDEngine/PDStageConnector

In `vllm_rlt/pd/engine.py`

`PDEngine` 是 PD 部署的 CPU coordinator，不执行 model forward；它拥有多个

`StageClient`、P/D peer、跨 stage request 状态和 transfer 生命周期。

```text
PDEngine(model, ...)
  -> prepare_model() -> read-only ModelConfig
  -> PDConfig + P/D CacheConfig + P/D SchedulerConfig
  -> 校验 model family、KV layout/block size、attention backend
  -> peers / requests / transfers / metrics
  -> 对每个 (role, device)：
       StageClient(..., StageConfig(role), connector_factory=PDStageConnector.create)
         -> StageWorker -> StageEngine
         -> PDStageConnector.create()
  -> 等待所有 StageClient.ready
  -> 比较各 worker fingerprint
  -> 向异角色 peer 发送 connect
  -> 等待所有 StageClient.connected
```

|对象|初始化 owner|持有内容|
|---|---|---|
|`PDEngine`|主进程/HTTP worker|`Peer`、`Pending`、request 映射、transfer 状态、计时和 profile 状态|
|`StageClient`|PDEngine 创建|单个 stage 的进程、Pipe、generation 映射和 control 生命周期|
|`StageEngine`|对应 StageWorker|该 GPU 的 model、KV cache、Scheduler、Runner 和本地 request|
|`PDStageConnector`|对应 StageWorker|hidden staging、`RequestTransfer`、slot、KV import/export 协议|
|`NixlConnector`|PDStageConnector 创建|NIXL agent、registered KV/hidden memory、peer、in-flight transfer handle|

**`PDStageConnector.create()`**

```text
engine.scheduler.config.max_num_seqs
  + max_receiving_requests（仅 produces_tokens 的 stage）
  + max_draining_requests（仅 accepts_prompt 的 stage）
  -> 分配 device hidden staging
  -> NixlConnector(stage_id, cache, hidden, PDConfig)
  -> 注册 key_cache / value_cache / hidden memory
  -> 返回 PDStageConnector(engine, hidden, transport, config)
```

**启动约束**

|检查|作用|
|---|---|
|P/D model fingerprint|比较模型权重、dtype、KV layout、depth、attention 信息|
|KV geometry|P/D 必须使用相同 layout、block size、KV dtype、depth 和 block geometry；固定 scale FP8 还要校验校准 scale|
|角色配置|prefill worker 不启用 speculative；decode worker 才接收 speculative config|
|transport backend|NIXL 注册两侧 KV/hidden memory，建立 remote agent|

**传输所有权**

```text
PDEngine
  -> control channel: reserve / prefill / commit / ack / cancel 元数据
PDStageConnector
  -> reserve 本地 slot、维护 RequestTransfer、确认 compute/transfer 边界
NixlConnector
  -> 用 registered VRAM 地址执行 KV/hidden WRITE
  -> poll 完成状态并释放 transfer handle
StageEngine / KVCacheManager
  -> 保持 request phase、KV allocation、lease 和 compute 生命周期
```

控制通道不承载 KV tensor。目标侧先 reserve 本地 Request/KV，此时处于 `RECEIVING`；收到 commit **且**所有预期 chunk 到达后才能 `accept_import()`，再经过共享 admission 进入 `CODA` 生成首 token。源侧的 compute 完成、目标确认和远端访问结束共同决定何时安全释放，commit 单独不足以释放源 KV。PDEngine 负责跨 stage 顺序，单个 StageEngine 仍负责本地调度和执行。


#### PDEngine 的准入与消息推进

```text
PDEngine.step():
	-> _poll() 遍历各 peer 的 StageClient，_message() 按 transfer ID 更新状态
	-> 检查 request/transfer deadline，随后 _admit() 尝试为等待请求配对
		-> P 预留 prompt blocks，D 预留 prompt + 输出上限 - 1 对应的 blocks
		-> 检查各 peer 的 transfer slots、reserved blocks 和 watermark
		-> 优先选 block 占比最低的 peer，同占比时看 slot 占比
		-> 短时受阻的请求可被后续请求绕过，等待超过一半 request_timeout 的受阻请求会挡住后续准入
	-> 返回本轮由 D 返回的 outputs
```

```text
waiting -> reserving:
	-> coordinator 记账 P/D 双方额度，先向 D 发送 reserve
	-> D 分配 prompt KV 与 hidden slot，返回 reserved(tables, slot, cached_tokens)
	-> coordinator 用这些目标地址向 P 发 prefill，P 分配本地 prompt KV 并开始执行
prefill -> received -> decode:
	-> P 按完成的 prompt 范围排入传输，最后一段带最终 hidden
	-> P 的 chunk 全部提交后发 commit(count)，由 coordinator 转给 D
	-> D 拿齐 commit 和所有 sequence 后调用 accept_import()，再通知 received
	-> coordinator 向 P 发 ack
	-> D 经本地 admission 进入 CODA，connector 通知 activated
finished -> released:
	-> D 的最后一个 output 结束外部请求
	-> P/D 各自上报 released 后，coordinator 分别归还其额度
	-> 两端都释放后，记录 timings 并删除 transfer
```

Coordinator 的额度用于配对和预留，实际 KV 页由各 stage 的 Scheduler/KVCacheManager 管。D 先返回命中的 prefix 长度，P 跳过 D 已有的 KV。若 P 本地命中的范围更长，`reserve()` 会把这段差额先排进传输队列。`accept_import()` 将传入范围标成有效 KV、绑定最终 hidden，再把仍处于 `RECEIVING` 的请求加入本地 WAITING 队列。

取消时，尚未分配 peer 的请求可直接移除。已有传输时向 D 发 `cancel`，再通知 P 停止后续提交。P 清空未发队列，并等待本地 staging event 和已提交的远端写入结束。D 若尚未确认完整接收，需等 coordinator 在 P 释放后发 `cancel_safe` 才能回收。deadline 或 transport 错误进入失败处理并终止相关 worker，避免继续复用状态不明的注册内存。

代码：`vllm_rlt/pd/{engine,connector,transport}.py` 与 `StageEngine.accept_import()`。

