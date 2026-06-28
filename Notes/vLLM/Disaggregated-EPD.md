# vLLM EPD: Disaggregated Encoder / Prefill / Decode

> 参考文档：[Inside vLLM EPD](https://yuanlehome.github.io/blog/inside-vllm-epd-disaggregated-encoder-prefill-decode/)

## 1. 核心概念

EPD 是将多模态 LLM 推理的三个阶段分离到不同 GPU 实例的架构：

| 阶段 | 英文 | 职责 | 计算特征 |
|------|------|------|----------|
| **E** | Encoder | 处理多模态输入（图像/音频） | 轻量级，GPU 利用率低 |
| **P** | Prefill | 处理输入 prompt，生成 KV Cache | 计算密集型 |
| **D** | Decode | 逐 token 生成输出 | 内存带宽型 |

### 为什么分离 Encoder？

1. **资源浪费**：视觉编码器远比语言模型轻量，共享 GPU 导致无法独立调配和扩缩容
2. **TTFT 被拉高**：纯文本请求也必须经过含 Encoder 的完整 pipeline，增加不必要的首 token 延迟
3. **缓存无法跨进程复用**：同一张图片在不同请求中被重复编码，进程内 Encoder Cache 仅限单 Worker 复用

**核心目标**：把短时、重计算、强波动的视觉 encoder 从文本 generation 实例里拆出来，减少互相拖累。

---

## 2. 部署模式

### 2.1 E+PD 模式（1 Encoder + 1 PD）

最基础的 Encoder 分离模式，Encoder 独立部署，PD 合并：

```
┌─────────────┐
│   Encoder   │ ──save_caches──► 共享存储
│   (GPU 0)   │
└─────────────┘
                      ┌─────────────┐
                      │     PD      │ ◄──load_caches── 共享存储
                      │   (GPU 1)   │
                      └─────────────┘
```

**数据流**：
1. Proxy 从请求中提取多模态项（图片/音频）
2. 每个多模态项并发发送到 Encoder 实例
3. Encoder 执行视觉编码，将结果按 `mm_hash` 索引保存到共享存储
4. Proxy 将原始请求转发到 PD 实例
5. PD 实例通过 EC Connector 从共享存储加载 Encoder Cache
6. PD 实例执行语言模型推理并返回结果

### 2.2 E+P+D 模式（三级分离）

在 E+PD 基础上进一步将 Prefill 和 Decode 分离：

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Encoder   │ ──► │   Prefill   │ ──► │    Decode   │
│   (GPU 0)   │ EC  │   (GPU 1)   │ KV  │   (GPU 2)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

Prefill 实例同时是 EC 消费者和 KV 生产者；Decode 实例只是 KV 消费者，不需要 EC 配置。

| | Encoder | Prefill | Decode |
|---|---------|---------|--------|
| EC 角色 | ec_producer | ec_consumer | 无 |
| KV 角色 | 无 | kv_producer | kv_consumer |
| 核心操作 | 视觉编码→保存 EC | 加载 EC→Prefill→传 KV | 加载 KV→Decode |
| 显存占用 | 极低 | 中 | 中 |

### 2.3 ec_both（单实例自缓存）

同一实例既是 Producer 又是 Consumer，用于单机多请求下相同图片只编码一次，后续请求直接复用缓存。

---

## 3. 配置体系

### 3.1 ECTransferConfig

```python
@config
class ECTransferConfig:
    ec_connector: str | None = None       # EC 连接器实现名称
    engine_id: str | None = None          # 引擎 ID，自动生成 UUID
    ec_buffer_device: str | None = "cuda"  # 缓冲区设备
    ec_buffer_size: float = 1e9           # 缓冲区大小约 1GB
    ec_role: ECRole | None = None         # 当前实例角色
    ec_rank: int | None = None            # 传输集群中的 rank
    ec_parallel_size: int = 1             # 并行实例数
    ec_ip: str = "127.0.0.1"             # 连接器 IP
    ec_port: int = 14579                  # 连接器端口
    ec_connector_extra_config: dict = field(default_factory=dict)
    ec_connector_module_path: str | None = None  # 动态加载自定义连接器
```

### 3.2 角色与运行时行为矩阵

| ec_role | scheduler 侧行为 | worker 侧行为 | 是否走 encoder-only 快路径 |
|---------|-----------------|---------------|--------------------------|
| ec_producer | 不参与远端 EC 载入决策 | 只 save_caches() | 是，execute_model 提前返回空输出 |
| ec_consumer | 参与远端命中/加载决策 | 先 start_load_caches() 再正常 prefill/decode | 否 |
| ec_both | 具备 consumer 的远端命中语义 | 同时具备 load 与 save 能力 | 否 |

### 3.3 Encoder 实例专用配置

```bash
vllm serve $MODEL \
    --enforce-eager \              # 必须：禁用 CUDA Graph
    --no-enable-prefix-caching \   # 必须：禁用前缀缓存
    --max-num-batched-tokens 114688 \
    --mm-encoder-only \            # 跳过语言模型，节省显存
    --gpu-memory-utilization 0.01  # Encoder 很轻量
```

`mm_encoder_only` 启用后只加载视觉编码器权重，不加载语言模型权重，大幅减少显存占用。

---

## 4. ECConnector 核心抽象

### 4.1 双角色设计

ECConnector 采用双角色设计——SCHEDULER 角色在调度器中实例化，负责决策；WORKER 角色在 Worker 进程中实例化，负责执行。

#### Worker 侧方法

```python
def bind_connector_metadata(self, connector_metadata):
    """每次模型执行前绑定调度器传来的元数据"""

def start_load_caches(self, encoder_cache, **kwargs):
    """从连接器加载 Encoder Cache 到 encoder_cache 字典"""

def save_caches(self, encoder_cache, mm_hash, **kwargs):
    """将 Encoder Cache 保存到连接器"""

def get_finished(self, finished_req_ids):
    """返回异步传输完成的请求 ID"""
```

#### Scheduler 侧方法

```python
def has_cache_item(self, identifier: str) -> bool:
    """检查指定多模态数据的 Encoder Cache 是否存在于外部存储"""

def update_state_after_alloc(self, request, index):
    """Encoder Cache 分配后更新连接器状态"""

def build_connector_meta(self, scheduler_output):
    """为当前调度步骤构建 Worker 所需的元数据"""
```

### 4.2 核心文件结构

```
vllm/
├── config/
│   └── ec_transfer.py                    # ECTransferConfig 配置类
├── distributed/
│   └── ec_transfer/
│       ├── ec_transfer_state.py          # 全局单例状态管理
│       └── ec_connector/
│           ├── base.py                   # ECConnectorBase 抽象基类
│           ├── factory.py                # ECConnectorFactory 工厂
│           └── example_connector.py      # ECExampleConnector 参考实现
├── v1/
│   ├── worker/
│   │   ├── gpu_model_runner.py           # 集成 EC Mixin
│   │   └── ec_connector_model_runner_mixin.py  # EC 生命周期管理 Mixin
│   ├── core/
│   │   ├── sched/scheduler.py            # 调度器 EC 集成
│   │   └── encoder_cache_manager.py      # Encoder Cache 管理器
│   └── outputs.py                        # ECConnectorOutput
```

### 4.3 ECExampleConnector 参考实现

当前唯一的参考实现，使用磁盘 safetensors 作为共享存储：

```
{shared_storage_path}/
├── {mm_hash_1}/
│   └── encoder_cache.safetensors
├── {mm_hash_2}/
│   └── encoder_cache.safetensors
```

**save_caches（生产者保存）**：
```python
def save_caches(self, encoder_cache, mm_hash, **kwargs):
    if not self.is_producer:
        return
    filename = self._generate_filename_debug(mm_hash)
    ec_cache = encoder_cache[mm_hash]
    tensors = {"ec_cache": ec_cache.detach().cpu()}
    safetensors.torch.save_file(tensors, filename)
```

**start_load_caches（消费者加载）**：
```python
def start_load_caches(self, encoder_cache, **kwargs):
    metadata = self._get_connector_metadata()
    for mm_data in metadata.mm_datas:
        if mm_data.mm_hash in encoder_cache:  # 已存在则跳过
            continue
        filename = self._generate_filename_debug(mm_data.mm_hash)
        ec_cache = safetensors.torch.load_file(
            filename, device=current_platform.device_type
        )["ec_cache"]
        encoder_cache[mm_data.mm_hash] = ec_cache
```

---

## 5. 调度器集成

### 5.1 核心决策逻辑：_try_schedule_encoder_inputs

三路选择器：reuse local / load remote / recompute local。

```python
def _try_schedule_encoder_inputs(self, request, ...):
    for i, mm_feature in enumerate(request.mm_features):
        # 检查本地缓存
        if self.encoder_cache_manager.check_and_update_cache(request, i):
            continue  # 本地命中，跳过

        # EC 核心判断：远端是否存在
        if (self.ec_connector is not None and
            self.ec_connector.has_cache_item(item_identifier)):
            # 远端有缓存→外部加载路径，不消耗 encoder_compute_budget
            external_load_encoder_input.append(i)
            continue

        # 远端也没有→需要本地编码
        encoder_inputs_to_schedule.append(i)
```

**关键洞察**：外部加载的编码器输入不消耗编码预算，这使得 PD 实例即使没有 Encoder 能力也能处理多模态请求。

### 5.2 Encoder Cache Manager

```python
class EncoderCacheManager:
    cache_size: int
    num_free_slots: int
    cached: dict[str, set[str]]       # mm_hash → request_id 集合
    freeable: OrderedDict[str, int]   # 可回收项
```

EC Connector 管理外部存储中的 Encoder Cache 传输；Cache Manager 管理本地内存中的容量。两者在 `_try_schedule_encoder_inputs` 中协作。

---

## 6. Worker / GPU Model Runner 集成

### 6.1 Context Manager 生命周期

```python
@contextmanager
def _get_ec_connector_output(scheduler_output, encoder_cache, **kwargs):
    # 进入：bind_metadata + Consumer 的 start_load_caches
    ec_connector.bind_connector_metadata(scheduler_output.ec_connector_metadata)
    if ec_connector.is_consumer:
        ec_connector.start_load_caches(encoder_cache, **kwargs)

    yield output  # 期间：执行 Encoder + 收集嵌入

    # 退出：get_finished + clear_metadata
    output.finished_sending, output.finished_recving = (
        ec_connector.get_finished(scheduler_output.finished_req_ids))
    ec_connector.clear_connector_metadata()
```

### 6.2 Encoder-Only 运行时路径

```python
# execute_model 入口
if has_ec_transfer() and not get_ec_transfer().is_consumer:
    with self.maybe_get_ec_connector_output(...) as ec_connector_output:
        self._execute_mm_encoder(scheduler_output)
        return make_empty_encoder_model_runner_output(scheduler_output)
```

`encoder_cache` 是整个 runtime 的"合流点"——只要张量进入这个 dict，后续 prefill/decode 不在乎它是本地算出还是远端载入的。

---

## 7. EPD Proxy 路由代理

### 7.1 核心路由逻辑

Proxy 是 FastAPI 应用，将标准 OpenAI Chat Completions 请求拆分为多阶段处理：

```python
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    e_urls = app.state.e_urls      # Encoder 集群
    p_url = random.choice(app.state.p_urls)  # 可选 Prefill
    d_url = random.choice(app.state.d_urls)  # Decode/PD
```

### 7.2 Encoder 扇出机制

- 一个 MM 项 = 一个请求，每个 MM 项构造独立子请求
- `max_tokens=1` 确保只做编码
- Round-robin 分发到 Encoder 集群
- 全 barrier 语义：所有 primer 成功才继续

---

## 8. 部署实战

### E+PD 部署

```bash
# 共享存储
EC_SHARED_STORAGE_PATH="/tmp/ec_cache"
mkdir -p "$EC_SHARED_STORAGE_PATH"

# Encoder 实例 (GPU 0)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 19534 \
    --gpu-memory-utilization 0.01 \
    --enforce-eager \
    --no-enable-prefix-caching \
    --mm-encoder-only \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_producer",
        "ec_connector_extra_config": {"shared_storage_path": "/tmp/ec_cache"}
    }'

# PD 实例 (GPU 1)
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 19535 \
    --gpu-memory-utilization 0.7 \
    --enforce-eager \
    --ec-transfer-config '{
        "ec_connector": "ECExampleConnector",
        "ec_role": "ec_consumer",
        "ec_connector_extra_config": {"shared_storage_path": "/tmp/ec_cache"}
    }'

# Proxy
python disagg_epd_proxy.py \
    --port 10001 \
    --encode-servers-urls "http://localhost:19534" \
    --decode-servers-urls "http://localhost:19535"
```

### E+P+D 部署

```bash
# Prefill 实例 - 同时是 EC Consumer + KV Producer
vllm serve $MODEL --port 19535 \
    --ec-transfer-config '{"ec_connector": "ECExampleConnector", "ec_role": "ec_consumer", ...}' \
    --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_producer"}'

# Decode 实例 - 仅 KV Consumer，无需 EC 配置
vllm serve $MODEL --port 19536 \
    --kv-transfer-config '{"kv_connector": "NixlConnector", "kv_role": "kv_consumer"}'
```

---

## 9. EC 与 KV 传输的协作

在 E+P+D 模式中，EC Transfer 和 KV Transfer 形成串行管道：

| 特性 | EC Transfer | KV Transfer |
|------|-------------|-------------|
| 传输内容 | Encoder 输出嵌入 | 注意力层 KV Cache |
| 数据量 | 较小 | 较大 |
| 传输时机 | Encoder 执行后→Prefill 执行前 | Prefill 执行后→Decode 前 |
| 传输协议 | 磁盘 safetensors | NixlConnector |
| 生产消费者关系 | E=Producer, P=Consumer | P=Producer, D=Consumer |

---

## 10. 能力边界与局限性

### EPD 直接解决

- Encoder 与 generation 资源池解耦
- Encoder outputs 跨进程复用
- 纯文本请求天然 bypass encoder
- 可与 PD 组合成 E|P|D

### EPD 不直接解决

- 不替代 PD 的 tail ITL 治理
- 不消除重复多模态 preprocessing
- 不提供生产级 EC transport
- 不自动治理外部 cache 一致性

### 局限性

- 只有磁盘 safetensors 参考实现，缺乏高性能 transport
- 无 cache 版本校验、部分写入、回滚机制
- proxy 编排是外部脚本，非内建
- 未覆盖异步/容错/一致性场景
- metadata 过于简陋，不承载 schema/version 信息

---

## 11. 源码里的隐含不变量

1. `identifier` 是 cache key 的唯一真相来源
2. scheduler 每一步必须构造 `ec_connector_metadata`，哪怕为空
3. encoder-only 实例的 `execute_model` 入口直接走提前返回
4. `encoder_cache` 是合流点，本地计算和远端载入最终汇聚
5. EC load 当前是同步 step 内完成，无异步状态机

---

## 12. 核心数据流总结

```
多模态请求
    │
    ▼
Proxy 拆分 MM 项
    │
    ▼ (并发)
Encoder 实例执行视觉编码
    │
    ▼ save_caches
共享存储 (mm_hash 索引)
    │
    ▼ load_caches
PD/P 实例的 self.encoder_cache 合流
    │
    ▼ _gather_mm_embeddings
LM forward → 生成 token
```

与 PD 分离叠加时：PD 实例 prefill 后通过 KV Transfer 将 KV Cache 传给 Decode 实例，Decode 实例完成自回归生成。
