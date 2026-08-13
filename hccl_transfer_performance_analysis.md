# Qwen-Image 服务通信后端性能

## 1. 结论摘要

1. HCCL 当前实现是“ZMQ 控制面握手 + HCCL P2P 数据面”的同步请求/响应协议。每个 tensor 独立 `send/recv`，发送端和接收端都逐个 `wait()`；没有 tensor 合并，也没有通信与计算重叠。它会增加单请求 handoff 延迟和尾延迟。
2. codec 对每次角色交接都抽取并迁移整个 `DiffusionRequestState` 中的张量集合。Denoiser 到 Decoder 的交接仍会传输 prompt embedding、mask 和 scheduler 等 Decoder 通常不需要的数据，造成不必要的通信、分配和同步。

## 2. 性能测试数据

### 2.1 汇总

| 指标 | HCCL | SHM | ZMQ |
| --- | ---: | ---: | ---: |
| Benchmark duration (s) | 330.29 | 324.06 | 319.43 |
| Request throughput (req/s) | 0.39 | 0.39 | 0.40 |
| Successful requests | 128/128 | 128/128 | 128/128 |
| Latency Mean (s) | 67.5625 | 67.4846 | 67.8068 |
| Latency Median (s) | 67.0377 | 69.0848 | 70.1709 |
| Latency P95 (s) | 105.8398 | 105.5538 | 105.6300 |
| Latency P99 (s) | 117.1062 | 112.3045 | 116.7687 |
| Encoder mean (s) | 0.1457 | 0.1273 | 0.1326 |
| Denoiser mean (s) | 16.4319 | 16.3959 | 16.4648 |
| Decoder mean (s) | 0.3244 | 0.1656 | 0.1693 |
| E2E mean (s) | 66.6763 | 66.6231 | 66.8901 |

### 2.2 数据解读

`e2e` 平均约 66--67 秒，而三个已计阶段的时间之和仅约 16.7--16.9 秒。原因是阶段时间在获得角色 worker 锁后才开始计时，等待 Denoiser replica 的排队时间没有记入阶段时间，但会记入 e2e。

当前 7 个 Denoiser replica 是全系统的主瓶颈：

```text
单 replica 服务能力  ~= 1 / 16.4 s = 0.061 req/s
7 replica 总能力    ~= 7 / 16.4 s = 0.427 req/s
实际吞吐            = 0.39--0.40 req/s
```

## 3. 当前 HCCL 问题一：逐 tensor send/recv、同步等待，未做合并或重叠

### 3.1 当前代码

发送端对 tensor 列表逐一提交 `send`，然后逐一等待：

```python
# vllm_diffusion/disaggregation/transfer_group.py
works = [
    self.pg.send([t], dst_rank, tag_base + i)
    for i, t in enumerate(tensors_on_device)
]
for w in works:
    w.wait()
```

接收端先通过 ZMQ 请求发送端开始发送，再为每个 tensor 分配 buffer、逐一 `recv` 并等待：

```python
# vllm_diffusion/disaggregation/transfer_group.py
sock.send(pickle.dumps(req, protocol=pickle.HIGHEST_PROTOCOL))

metas = refs["metas"]
bufs = [
    torch.empty(tuple(m["shape"]), dtype=_name_dtype(m["dtype"]), device=self.device)
    for m in metas
]
works = [
    self.pg.recv([b], int(refs["src_rank"]), tag_base + i)
    for i, b in enumerate(bufs)
]
for w in works:
    w.wait()
reply = pickle.loads(sock.recv())
```

对应位置：`vllm_diffusion/disaggregation/transfer_group.py:200-202`、`:223-236`。

### 3.2 影响

- 每次 handoff 都必须经过一次 ZMQ fetch 握手，之后才能发起配对 HCCL 操作。
- 张量数量越多，HCCL 调用、tag 管理和 work 对象越多；小 tensor 的启动开销无法摊薄。
- `wait()` 使发送端 fetch 线程和接收端 role worker 均在传输完成前阻塞。它不是全局 barrier，但属于 encoder->denoiser 和 denoiser->decoder 的关键路径同步点。
- 当前没有把 HCCL 通信放到独立 stream 并用 event 在真正消费前同步，也没有在传输期间预处理后续计算。因此通信无法与可并行的 CPU/NPU 工作重叠。
- Decoder 的阶段时间包含 denoiser->decoder 的 fetch、HCCL recv/wait、状态重建和 VAE decode；HCCL 的 decoder 均值比 shm/zmq 高约 0.16 秒，需通过传输分段计时确认其中有多少来自该同步路径。

### 3.3 建议

1. 先增加 handoff 指标：实际 `store`、tensor 数、总字节数、ZMQ manifest 时间、buffer 分配时间、HCCL post 时间和 wait 时间。
2. 按 dtype/device 合并小 tensor，或使用支持成批 P2P 提交的接口，降低逐 tensor 的 Python 与通信启动开销。
3. 将异步通信提交和完成等待分离：提前 post recv/send，在下游真正读取张量前以 event/Work 等待；仅在满足数据依赖的位置同步。
4. 压测需要分为低并发（测单请求 handoff 延迟）与饱和并发（测总吞吐），每种后端至少 warm-up 后重复多轮。

## 4. 当前 HCCL 问题二：每次交接迁移完整 `DiffusionRequestState`

### 4.1 当前代码

codec 定义了每次交接都要处理的状态字段：

```python
# vllm_diffusion/disaggregation/codec.py
_TENSOR_FIELDS = (
    "prompt_embeds",
    "prompt_embeds_mask",
    "negative_prompt_embeds",
    "negative_prompt_embeds_mask",
    "latents",
    "timesteps",
    "guidance",
)

_SCHEDULER_TENSOR_ATTRS = (
    "timesteps",
    "timesteps_ori",
    "model_outputs",
    "timestep_list",
)
```

HCCL/shm/rdma 的编码路径没有根据“下一跳角色”裁剪字段，而是构建完整 payload、递归抽取所有 tensor，再交给数据面：

```python
# vllm_diffusion/disaggregation/codec.py
payload = _build_payload(state, keep_device=device_tensors_preferred(backend))
bag: dict[str, torch.Tensor] = {}
meta = _extract_tensors(payload, bag, [0])
refs = put_tensors(bag, backend=backend)
envelope = {
    "v": _ENVELOPE_VERSION,
    "backend": backend,
    "meta": meta,
    "refs": refs,
}
return pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
```

对应位置：`vllm_diffusion/disaggregation/codec.py:29-46`、`:208-218`。

### 4.2 影响

- Denoiser->Decoder handoff 继续搬运 `prompt_embeds`、negative prompt embedding、mask、guidance 及无关 scheduler 状态；Decoder 一般只需要最终 latent 和少量输出/尺寸元数据。
- 对 HCCL 而言，这会直接增加 P2P 字节数、NPU buffer 分配数量及逐 tensor 同步次数。
- 对 shm 而言，会增加 D2H、`/dev/shm` 写入、Python bytes copy、H2D 和 segment 回收开销；因此 shm 并非真正零拷贝。
- 对 zmq 而言，会增加 CPU 张量化、pickle/unpickle 和 TCP payload。
- 该问题与后端无关，但 HCCL 当前逐 tensor 实现会将其放大。

### 4.3 建议

1. 将 codec API 扩展为带 destination role 的编码，例如 `encode_state(state, destination=Role.DECODER)`。
2. 定义显式 handoff schema：
   - Encoder->Denoiser：prompt embedding、mask、初始 latent、scheduler 所需状态；
   - Denoiser->Decoder：最终 latent、图像尺寸/输出元数据；
   - 双 DiT 分段时仅补充后续 Denoiser 必须的 scheduler state。
3. 每个 schema 记录 tensor 名、shape、dtype、字节数，并在接收端进行版本校验；不要依赖对任意 Python object 的递归抽取。
4. 先测量各字段字节数和 Decoder 实际读取字段，再裁剪。避免错误删除 scheduler 或图生图路径仍需要的状态。
