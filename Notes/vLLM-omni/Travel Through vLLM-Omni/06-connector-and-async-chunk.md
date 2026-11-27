#### OmniConnector、async_chunk 与 full payload

### OmniConnector 只管传输

Connector 抽象跨 stage/process/device/node 的 payload transport 与同步，不拥有 next-stage 路由，也不应承载某个模型的执行策略。producer 和 consumer 必须约定 payload identity、shape、dtype、placement、ownership 和完成语义；正常完成、取消、错误和 shutdown 都要有确定的资源清理路径。

### Async chunk

async_chunk 允许上游产生一个可消费 chunk 后就交给下游，不必等待整个 stage 完成。Orchestrator 可以先向 downstream chunk stage 提交 placeholder/prewarm；随后 scheduler/adapter 管理 chunk readiness，真正数据通过 connector 到达。Generation scheduler 必须以完整 ready chunk 为原子单位 admission，不能把一个 chunk 切成任意子片段；超大 chunk 应形成单一终态错误，避免部分提交后状态失配。

有些 native MRv2 接收路径由 **OmniSchedulingCoordinator** 追踪 runner connector 事件；legacy adapter 路径由 **OmniChunkTransferAdapter** 管理加载和保存循环。二者暴露给调度器的 readiness 语义应一致，但 transport owner 不同。

- `docs/design/feature/async_chunk.md`
- `vllm_omni/distributed/omni_connectors/transfer_adapter/chunk_transfer_adapter.py`
- `vllm_omni/core/sched/omni_ar_scheduler.py`
- `tests/core/sched/test_generation_scheduler_atomic_chunk.py`、`tests/engine/test_orchestrator_async_chunk_prewarm.py`

### Full payload 累积与丢弃

MRv2 full-payload 路径中，worker data plane 按内部 request id 持有尚未发布的 payload；scheduler 给出 terminal status 后再 flush 或丢弃。正常 stop/length-limit 完成应只 flush 一次；abort/error/ignored 通过 discarded request id 清掉待发 payload。已经提交给 connector 的写不再由 pending accumulator 控制。

如果 AR 请求被 preempt 后继续执行，累计输出和 token boundary 属于同一个逻辑 request：重算的 prefix 不能重复追加；对齐 token 边界的 tensor 可只追加新 suffix；无法按 token 切分的部分重叠应显式失败。retry 则必须建立新的 payload/position state。最近的 full-payload 传输失败修复也提醒：stage transfer 失败不能让请求留在等待态或静默成功。

- `vllm_omni/worker_v2/omni_data_plane.py`、`vllm_omni/core/sched/omni_scheduler_mixin.py`
- `docs/design/module/omni_connector.md`、`tests/worker/test_full_payload_lifecycle.py`
- 当前基线提交主题：full-payload stage transfer 不能成功时令请求失败（commit e5576969）。

### 排队发送不代表下游已消费

Output materialization / native output worker 将 payload 引用入队，enqueue 成功不等于 downstream 已收到或处理完成。source 的 model output 字段之后被清空，也不意味着队列中的对象或底层 storage 被释放。要定位 tensor 被过早复用，分别追 producer snapshot、队列持有时间、consumer event/wait、request terminal cleanup，不能只看 enqueue 返回。

- `docs/design/feature/omni_async_output_materialization.md`
- `vllm_omni/worker_v2/omni_data_plane.py`
- `tests/worker_v2/test_omni_ar_model_runner.py`
