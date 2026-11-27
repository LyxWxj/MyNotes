#### 同步、异步和退出

同步 `_step()` 对选中的 batch submit；只有 ticket 需要 host 结果时才 collect，否则可在 event 约束下推进逻辑状态。普通异步 `_step_async()` 可以先提交下一段工作，再回收前一段必要的 exit signal / CODA 结果；`num_output_placeholders` 防止未确认 EOS 前重复采样。自推测异步路径最多保留两个未退休 round，并对过期 request/round 结果进行检查。

普通 Ouro `exit_mode="ouro"` 用 gate hazard 更新累计退出概率；`ouro_delayed` 将这个决定推迟一轮，`random_lookahead` 使用独立的随机 lookahead head，`trace` 使用预先给定的 depth。`exit_threshold=1` 禁用自适应退出，`max_loops` 是上限。完整 prompt 始终跑模型完整深度；上述退出选择主要影响后续 decode。Dory 的固定 target 规则见 Model 小节。

同步更新由 `Scheduler.update_from_output()` 处理；异步提交时 `update_after_submission()` 可先更新 host 侧逻辑进度，event 仍负责设备依赖，后续 collect 再交付需要 host 确认的 token/score。请求队列中的 phase 因此不是 GPU 已完成的证明。

#### 输出、取消和资源回收

`ModelRunnerOutput` 的 token IDs、depth 和结束信息经 Scheduler 转成 `RequestOutput`。worker 输出携带内部 generation，`StageClient` 映射回外部 ID；HTTP worker 只增量解码新 token，再通过 `RequestChannel` 交付 text、usage 和 finish reason。一次 speculative round 可以产生多个已提交 token，不能假设一次事件恰好一个 token。

结束、取消或失败会停止后续调度并清理 runner/RNG 和 request/KV。逻辑结束不等于物理内存立即可复用：异步 round、readback、GPU reader 和 PD transfer lease 仍可能持有资源，必须等相关 event / transport 安全后才能退休。客户端过慢导致 channel 缓冲满时，HTTP 层会报错并安排取消。


### 精度回归与性能验证

```text
benchmarks.gsm8k prepare:
	-> 固定 checkpoint/revision、题目、3-shot CoT prompt、输入 token IDs 和生成控制
	-> 写出 protocol，并校验默认 fixture 的基准指纹
benchmarks.gsm8k run:
	-> 以同一 protocol 跑 native 或 transformers，保存逐题输出、token IDs、答案和停止原因
	-> 默认 Ouro-1.4B BF16、四轮固定深度、greedy、单请求，最多 1024 个新 token / 2048 总上下文
	-> 默认用 59/87 作为回归门槛，先用两道 train 题单独跑通流程
benchmarks.gsm8k compare:
	-> 用同一 protocol 的完整逐题记录做配对比较，报告准确率差、paired loss/gain 和答案差异
	-> 改 checkpoint、生成 recipe 或题集时，重新跑 HF 和 native 两端
```

遇到异常或非有限 logits 时保留已有记录并让运行失败，固定题集的分母不变。基准分数用于这组固定题的回归判断。运行入口和参数见 `docs/accuracy.md`。

|验证入口|看什么|
|---|---|
|`tests/test_scheduler_selection.py`、`tests/test_prefix_growth.py`、`tests/test_kv_cache.py`|组批选择、prefix 命中复用、KV 容量和页生命周期|
|`tests/test_sampler.py`、`tests/test_speculative_sampling.py`、`tests/test_async_speculative_lifecycle.py`|采样分布、推测验收、异步取消和资源回收|
|`tests/test_tensor_transfer.py`、`tests/test_submission_queue.py`、`tests/test_pd.py`|搬运与缓冲复用、PD 握手和传输生命周期|
|`benchmarks.cdb_runtime`|固定 toy workload 和 exit-depth trace，对照不同调度与设备提交配置，检查 token/depth 一致性|
|`benchmarks.preemption`|冻结到达序列和 KV 预算，配对比较抢占开关，检查 token/depth 一致性，并按优先级与长度看 TTFT/E2E|

吞吐验证沿用仓库固定版本的 benchmark 客户端，warm-up 后分别记录串行、并发和有限到达率的结果。报告失败数、request/token throughput 和 TTFT、TPOT、ITL、E2E 分位数。端到端计时与离线 engine step 分开记录。入口见 `docs/serving.md` 的 benchmark 小节。

为每次运行留存命令、commit、模型 revision、设备、亲和性、配置和原始结果。

### 阅读与验证入口

|问题|对应文件|
|---|---|
|架构、stage owner 与执行边界|`docs/unified_execution.md`、`vllm_rlt/engine/stage_engine.py`|
|admission、phase 和 mixed/wavefront 选择|`docs/scheduler.md`、`docs/split-prefill-wavefront.md`、`vllm_rlt/core/{scheduler,scheduling_policy}.py`|
|KV 布局、prefix、transfer pin、FP8|`docs/kv_cache_walkthrough.md`、`docs/kv_layout_computation.md`、`docs/runtime.md`、`vllm_rlt/core/kv_cache_manager.py`|
|自推测与异步生命周期|`docs/async-speculative.md`、`vllm_rlt/worker/{speculative,async_speculative}.py`|
|HTTP/PD、设备限制|`docs/serving.md`、`docs/recipes/ouro_ascend.md`、`vllm_rlt/pd/`|
|回归测试|`tests/test_scheduler_selection.py`、`tests/test_stage_engine.py`、`tests/test_serving.py`、`tests/test_async_speculative_random.py`、`tests/test_fp8_kv_cache.py`、`tests/test_dory*.py`、`tests/test_ascend*.py`|

上述测试文件是阅读入口；本笔记的静态代码核对不代表已在 GPU/NPU 上重新运行这些测试，也不构成精度或性能结论。
