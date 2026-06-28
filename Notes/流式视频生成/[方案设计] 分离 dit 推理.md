# \[方案设计\] 分离 dit 推理 



# RFC/问题定义



这里主要是写：引入分离 dit 推理的背景、问题、实现方案 / 讨论、预期收益和实现计划；



## 背景



相关 pr：\#3208 disaggregated encoder helps most for short and mid\-step workloads, where it improves throughput by roughly 10%\-15%, and e2e latency gains in the same time in 8\-step, especially when Offered QPS = 2\.5\.

主要面对的是两个核心场景：

- 传统 dit 模型：主要关心的是生图、生视频场景，对于吞吐量要求高，主要计算耗时在于 对于小阶段只执行一次

- 流式生成模型：主要关心的是流式视频生成，对于延迟强要求，主要计算耗时需要考虑 encode \+ dit \+ decode、对于小阶段需要执行多次

ps：流式生成场景，让单次请求中的 dit 占比更少、因此一次请求要考虑的计算因素更多



## 问题







|**卡型**|**负载评估**|**预计收益**|
|---|---|---|
|消费卡型<br>4090/5090<br><br>|计算：中高；<br>访存：中，GDDR 带宽弱于 HBM；<br>容量：低到中，适合小模型或低并发。<br>成本：低|瓶颈：主要是容量<br>- 适合作为小角色的推理<br>- 做 dit 推理主要的提升在于 容量\-\-\-\>并发/减少h2d<br><br>|
|中端卡型<br>910b/L 卡型<br>|计算：中到高；<br>访存：中到高，HBM 相对均衡；<br>容量：中，适合多模态推理<br>成本：中|瓶颈：主要在于通信<br>|
|高端卡型<br>h100/b200 |计算：高；<br>访存：高；<br>容量：高，适合大模型、多并发和高吞吐 rollout。<br>成本：高|瓶颈主要在于成本|
|其他卡型<br>h20|计算：中低，计算能力受限；<br>访存：高；<br>容量：高，适合大 KV/cache/latent 状态驻留。<br>成本：中|瓶颈：主要在于计算<br>- 适合作为部分小角色的推理（如 理解/pe 等llm模块）|
|![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NzI5MGZmODQyODRjYTNlNTgwNjE5OGMxODg1YTE0ODBfODUzYTRiYjFlNDk0MzMzZTRhZDA1ZDFkNDgyNzQzYTNfSUQ6NzY1MjQwMDQyMjEyNDEyOTQ3MV8xNzgyMTUxNTU1OjE3ODIyMzc5NTVfVjM)|||







对于下面模型部署在此类卡型时，分离后的预期收益：

- 所有卡型 x 所有 dit：主要收益在于 pipeline，减少阻塞带来的 延迟/吞吐 提升

    - 现有方案：e/d/d 共同放置在一张卡上，当有新的请求进入的时候需要阻塞直到当前请求结束

        - 当前请求位于 e/d/d 任何一个 stage，后续请求都需要阻塞

        - 等待的请求延迟被拉高

        - 统一 bsz/ bsz 没有打高，导致系统吞吐受限

    - 提高逻辑：分离之后每个阶段独立调度和执行，请求不会被阻塞

    - 预期收益：提高吞吐、降低延迟

- 消费卡型的传统 dit ：主要收益在于提高吞吐

    - 现有方案：通过 layerwise offload 实现、带来 h2d/d2h 开销

    - 提高逻辑：分离后减少显存占用、在显存稀缺场景可以减少 h2d/d2h 开销和提高 bsz 

    - 预期收益：提高吞吐

- 中高端卡型的传统 dit：主要收益在于减少成本

    - 现有方案：encode/decode 小角色只执行一次却长期占用

    - 提高逻辑：相比于放置小角色到同类型卡上，放到低端卡型可以省更多的成本、并且异构带来的通信/计算 延迟十分少





> 相关数据：3 stage 对比，通过给出三个阶段大概的耗时，能够推断出流式大概能得到的性能提升
> 
> 

- 生图 from npu/gpu：dit 部署在单卡 int8 下面执行 4/8/10/40 等步数的时候、生成 512\*512 分辨率 and 1024\*1024 分辨率，encode/dit/decode 三个阶段的时间占比

- 生视频 from npu：dit 部署在 8 卡 int8 下面执行 4/8/10/40 等步数的时候、生成 480p 视频 and 720p 视频都是 5s ，encode/dit/decode 三个阶段的时间占比

- 流式 from npu/gpu：dit 默认实时的时候（比如生成 5s 视频 dit 生成 5s），此时 encode/dit/decode 三个阶段的时间占比；注意这里只要生成视频的latents数量一致、不管 dit 是几b模型多少计算量、decode 时间应该都是一致的





- Dag 引进：音视频模型测试



> 预期收益
> 
> 

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ODkzZjgwODg0ODU3MTViZTExNjFjMjhiM2FlMDk1MTFfZWNmMDNiMGVmZjEwMGE0MDM5NzgyMzQyZGZhNjFiMzBfSUQ6NzY1Mzg2NjA5NDQ4NjI5MzcwOV8xNzgyMTUxNTU1OjE3ODIyMzc5NTVfVjM)

**对比 pipeline 带来的预期收益**：

- 原始方案：原始共置方案里，一个 worker 同时持有三个阶段，请求进入之后，只要当前请求还在任意一个 stage，后续请求都要等。问题不只是单请求 E2E 变长，而是：小 stage 阻塞大 stage、不同请求无法稳定流水、batchsize 不容易打高

- 分离之后的理论上限收益：吞吐上限从 1 / \(T\_encode \+ T\_dit \+ T\_decode\) 便成为 1 / max\(T\_encode, T\_dit, T\_decode\)；收益大概是 speedup ≈ \(T\_encode \+ T\_dit \+ T\_decode\) / max\(T\_encode, T\_dit, T\_decode\)；根据现有的数据，对于稳态 pipeline 下吞吐量收益可以有 1\.2x \~ 2\.2x 的性能提升

- 实际性能收益：该收益主要来自在 DiT 主实例之外，额外挂载少量低成本小阶段实例。典型部署中，可以用 8 张 H100/A100/910B 运行 DiT，再用 1～2 张 4090/5090 承载 Encode/Decode。由于小阶段计算占比较低，一组小阶段实例通常可以服务多组 DiT 实例，额外成本较小。并且这些异构卡原本难以直接参与 DiT 的同构并行，分离后可以被有效利用，从而以较低成本减少阻塞并提升吞吐。



- **对比消费级卡型 预期收益**



消费级卡型的主要问题是显存容量和访存带宽。以 4090/5090 为例，这类卡的成本较低，但显存容量和 GDDR 带宽弱于 HBM 高端卡。对于大 DiT 模型，如果采用共置方案，每张卡都需要同时加载 `Encode + DiT + Decode`，显存压力会非常大，容易触发 layerwise offload，并引入 H2D/D2H 传输、PCIe 带宽竞争和 batch size 受限等问题。

LightX2V https://light\-ai\.top/LightX2V\-BLOG/posts/Disaggregation/ 的数据可以说明这一点。Qwen\-Image\-2512 BF16 共置部署时，完整 pipeline 需要约 58GB 以上显存，无法直接放入 24GB 的 RTX 4090，只能依赖 CPU offload。offload 虽然能让模型跑起来，但会显著拉高延迟。例如在 4090 上，Qwen\-2512 的 Text Encoder 在共置 \+ offload 下耗时约 12\.89s，而分离后 Encoder 单独部署、无需 offload，耗时降到约 0\.40s，提升约 32\.2×。DiT 阶段也会受益于减少 offload 和带宽竞争，per\-step latency 从 5\.75s 降到 3\.76s，约 1\.53×。

因此，在消费级卡型上，分离的收益不是单纯来自 stage pipeline，而是来自显存解耦。分离之后，每个实例只需要加载自己负责的 stage：

```Plain Text
不分离：每张卡持有 Encode + DiT + Decode
分离后：Encoder / DiT / Decoder 分别部署，每张卡只持有一个 stage
```

因此，消费级卡型下的收益可以概括为：在相同低成本卡池内，分离式部署通过显存解耦减少 offload，并通过合理的 DiT/Decoder 配比提高吞吐。对于 4090/5090 这类低成本卡，在相同卡数下获得更高 QPS。





## 实现方案



实现的挑战

|可能的问题|预期方案|
|---|---|
|工程化简洁，模型能够快速接入相关特性|设计成为 mixin 的形式，大部分模型可以通过继承 mixin 快速接入分离特性；|
|性能和高可用性：虽然单个请求的 embedding 大小只有 mb 级别，但是在大集群可能会面临到多个 dit 实例同时跨机拉取的问题|需要在跨节点传输 构建稳定的生命周期还有传输管理|
|对于gpu影响最小化：需要让 cpu 引入的开销尽可能少，并且保证gpu不会处于空转|类似于 vllm 的做法，在请求 waiting 的时候进行缓存拉取（kvcache/embedding）、等到缓存状态 ready 才会实际执行。|





Dag 图（适合音视频模型 提升）





## 实现计划



* [ ] 初步接入：参考 \#3208 分离出单个状态，做好第一步功能合入

* [ ] 性能优化：小角色上的缓存和传输

* [ ] 性能优化：减少 bubble 、scheduler 状态 

* [ ] 性能优化：兼容 encode/decode fuse多个模型、还有分离出 audio vae/ vae 的情况



## 相关资料



> 数据
> 
> 

[VLLM\-Omni](https://my.feishu.cn/docx/AOJDdiuuTopxiAxDJm9carCGnaf?from=from_copylink)  关于 gpu 的部分执行

[Profiling\.md](https://my.feishu.cn/wiki/VZEiwQr0IijZACk9eFncB2Mknjg)

```Python
这里的 8 是同一个 request 的 TP8，不是 8 requests，也没有 batch 并发。

   模型              LTX-2
   分辨率 / 帧数     832x480 / 121f
   Step              4
   Text Encode       0.339s
   Connectors        0.056s
   DiT               12.486s
   Video VAE Decode  0.547s
   Audio VAE Decode  0.012s
   Vocoder           0.045s
   Decode 合计       0.605s
   E2E               16.484s
```







## 初版 RFC



name：\[RFC\] disaggregate diffusion inference

Relative pr：\#3208 disaggregated encoder helps most for short and mid\-step workloads, where it improves throughput by roughly 10%\-15%, and e2e latency gains in the same time in 8\-step, especially when Offered QPS = 2\.5\.



This RFC discusses the background, problems, possible implementation, expected benefits, and implementation plan for disaggregated DiT inference\.

---

### Background

This RFC mainly targets two scenarios\.



> Traditional DiT Models
> 
> 

Traditional DiT models are mainly used for image and video generation\. These workloads usually have high throughput requirements\.

The main compute cost is in the DiT stage\. Smaller stages, such as encode and decode, are usually executed only once per request\.



> Streaming Generation Models
> 
> 

Streaming generation models are mainly used for streaming video generation\. These workloads have strong latency requirements\.

For streaming generation, the runtime needs to consider the full execution path:

```Plain Text
Encode -> DiT -> Decode
```

Different from the traditional generation, small stages may be executed multiple times in one request\. Therefore, the DiT stage may account for a smaller portion of the full request, and more execution factors need to be considered\.

### Problems



> Hardware Characteristics
> 
> 





> Expected Benefits by Deployment Type
> 
> 



**All Hardware Types × All DiT Models**

The main benefit is pipeline execution, which can reduce blocking and improve latency and throughput\.

Current design:

```Plain Text
Encode + DiT + Decode are colocated on one worker.
```

When a new request arrives, it must wait until the current request finishes\. If the current request is in any stage, later requests are blocked\.

This causes: higher waiting latency; small stages blocking large stages; unstable request pipeline; limited batch size; lower system throughput\.

After disaggregation, each stage can be scheduled and executed independently\. Requests do not need to wait for the full colocated pipeline to finish\.

Expected benefit: higher throughput; lower queueing latency\.

**Traditional DiT on Consumer GPUs**

The main benefit is throughput improvement\.

Current design: Large DiT models on consumer GPUs often rely on layerwise offload\. This introduces H2D/D2H overhead and limits batch size\.

After disaggregation, each worker only needs to hold one stage\. This reduces memory usage, reduces H2D/D2H overhead in memory\-constrained scenarios, and allows larger batch size or higher concurrency\.

Expected benefit:

- higher throughput;

- lower offload overhead;

- better cost efficiency on low\-cost GPUs\.

**Traditional DiT on Mid\-range or High\-end GPUs**

The main benefit is cost reduction\.

Current design: Encode and decode are small stages\. They are usually executed only once per request, but they still occupy the same high\-end device as DiT\.

After disaggregation, small stages can be moved to lower\-cost devices\. Compared with placing all stages on the same type of high\-end card, this can reduce cost\. The communication and compute overhead introduced by this heterogeneous deployment is expected to be small\.

Expected benefit:

- lower cost;

- better high\-end GPU utilization;

- high\-end GPUs focus on DiT compute\.



### Profiling Data Needed

We need three\-stage profiling data to estimate the expected benefit of disaggregation\.

#### Profiling Data

We compare the three main stages, `Encode`, `DiT / Diffuse`, and `Decode`, to estimate the potential benefit of disaggregated execution\.



#### Expected Benefit from Pipeline Disaggregation

In the original colocated design, one worker owns all three stages: `Encode`, `DiT`, and `Decode`\. Once a request enters the worker, later requests have to wait until the current request finishes the whole pipeline\. This does not only increase single\-request E2E latency\. It also causes small stages to block large stages, prevents requests from forming a stable pipeline, and makes it harder to increase batch size\.

After disaggregation, the three stages can be scheduled and executed independently\. In the ideal steady state, the throughput upper bound changes from:

```Plain Text
1 / (T_encode + T_dit + T_decode)
```

to:

```Plain Text
1 / max(T_encode, T_dit, T_decode)
```

The ideal speedup is approximately:

```Plain Text
speedup ≈ (T_encode + T_dit + T_decode) / max(T_encode, T_dit, T_decode)
```

Based on current profiling data, steady\-state pipeline throughput can improve by about `1.2x ~ 2.2x`\.

In practice, this benefit mainly comes from attaching a small number of low\-cost stage workers to the main DiT pool\. A typical deployment can use 8 H100/A100/910B devices for DiT, and 1–2 4090/5090 devices for `Encode` and `Decode`\. Since these small stages have much lower compute cost, one small\-stage pool can usually serve many DiT workers\. The extra hardware cost is relatively small\. Also, these heterogeneous GPUs are hard to use directly for homogeneous DiT parallelism, but they can be effectively used after disaggregation to reduce blocking and improve throughput\.



#### Expected Benefit on Consumer GPUs



Consumer GPUs mainly suffer from limited memory capacity and lower memory bandwidth\. For 4090/5090, the cost is low, but GDDR bandwidth and memory capacity are weaker than high\-end HBM GPUs\. If a large DiT model uses the colocated design, each GPU needs to load `Encode + DiT + Decode`, which creates high memory pressure and can trigger layerwise offload\. This further introduces H2D/D2H transfers, PCIe contention, and batch size limits\.

LightX2V（https://light\-ai\.top/LightX2V\-BLOG/posts/Disaggregation/） provides a useful reference\. For Qwen\-Image\-2512 BF16, the colocated pipeline needs more than 58GB of memory, so it cannot fit into a 24GB RTX 4090 without CPU offload\. Although offload makes the model runnable, it significantly increases latency\. On 4090, the Qwen\-2512 text encoder takes about 12\.89s with colocated execution and offload\. After disaggregation, the encoder can run independently without offload, and the latency drops to about 0\.40s, which is about a 32\.2x improvement\. The DiT stage also benefits from reduced offload and less bandwidth contention: per\-step latency drops from 5\.75s to 3\.76s, about a 1\.53x improvement\.

Therefore, on consumer GPUs, the benefit is not only from pipeline execution\. A major benefit comes from memory decoupling\. Before disaggregation, each worker holds `Encode + DiT + Decode`\. After disaggregation, each worker only holds one stage\.

This reduces memory pressure, avoids or reduces offload, and improves the available batch size or concurrency\. In the same low\-cost GPU pool, disaggregated deployment can use a better DiT/Decode ratio and achieve higher throughput\. For 4090/5090, the expected benefit is higher QPS under the same number of cards\.



#### Expected Benefit on streaming dit



Streaming DiT has a special property: different stages inside a single request can be overlapped\.

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MzU4Y2FjYWVlNWY4ZTNiYzQyM2U0ZjNjYjYwNWMzYTJfMDExZDRiMTQ3Y2ZjZjZlYTk3YTU1ODUwNjMzZTI5OGRfSUQ6NzY1Mzg4MTI4MzA3NTkwMjY4MV8xNzgyMTUxNTU1OjE3ODIyMzc5NTVfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZWI3YTQ3YmJlZGMwMzI5YWMyODA1MzcwNzgwNTY2ODlfYzg4MjU2MTVmMWVjZWNiMTVmMWEwMjYyZWVkMThhNjdfSUQ6NzY1Mzg4MTI3OTkzMDM1NDY3Nl8xNzgyMTUxNTU1OjE3ODIyMzc5NTVfVjM)

The two figures compare the serial execution and the best\-overlap execution in StreamWorld SP2 streaming inference\. After disaggregation, DiT and VAE decode can run in parallel, reducing single\-request latency from **9\.84s to 5\.99s**, with a theoretical speedup of about **1\.64×**\.

---

### Proposed Implementation



### Implementation Plan

* [ ] Initial integration: split a single request state into stage\-level state, following the direction of existing request lifecycle work\.

* [ ] Performance optimization: add caching and transfer support for small stages\.

* [ ] Performance optimization: reduce bubbles and add scheduler state for stage readiness\.

* [ ] （optional）Performance optimization: support vit in encode stage

* [ ] （optional）Performance optimization: support fused encode/decode across multiple models, and support separated audio VAE / video VAE cases\.



## RFC 更新



后面加了一下maybe 讨论：支持 音视频模型 \+ dag 图、支持 vit 作为小角色

后面加了一下 streaming dit 特殊之处，可以单个请求 overlap 

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=NjI5ZGI0ZjBiN2JiODQ4NzM2M2VlZGU1YWUwYjM2MTJfYmMzNzU0OGIzNGY5MzA1N2YwOWEzY2Y2NDk0MWVmMGVfSUQ6NzY1Mzg4MDkzOTc4MDY3MjQ3MV8xNzgyMTUxNTU1OjE3ODIyMzc5NTVfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=MmNhMmI4MmY0ODNlYzk0NjgxOGUwZmY2NDcxNTgzNzhfYTg5ZjE4Yzc0ZmE2YWViNmQ1MjE3ODc1YWM2MmYzMGVfSUQ6NzY1Mzg4MDg4ODM4MzcyMDY2OF8xNzgyMTUxNTU1OjE3ODIyMzc5NTVfVjM)





# 和项目其他组件关系



## Qwen\-image vae 分离（\#3208）



\#3208 是一个具体实现：Qwen\-Image disaggregated VAE，把原 diffusion pipeline 拆成：

encode \-\> denoise \-\> decode

但它仍然把这些 stage 放在 diffusion family 下面：

- StageType = diffusion

- worker\_type = submodule

\#4590 是更大的 RFC：想把 disaggregated DiT inference 做成通用能力，覆盖 encode / DiT / decode、DAG、stage readiness、跨节点 tensor transfer、异构设备部署。可能会有下面冲突，对应解决思路



||\#3208 的问题|\#4590|
|---|---|---|
|分类冲突|submodule 是 diffusion 的 worker variant。|top\-level: submodule是新的 stage family。|
|Stagepool / orch |复用 diffusion StagePool 路径和 get\_diffusion\_output\_nowait\(\)。|top\-level 如果独立成 family，就需要自己的 client/proc/pool/output polling 规则，或者明确复用哪部分 diffusion runtime。|
|输出契约冲突|用 DiffusionOutput\.multimodal\_output 传 intermediate tensors。|如果变成通用 top\-level，应该定义 generic artifact contract，比如 latent / prompt  embeds / decoded media / state update，而不是继续借 diffusion output 字段。|
|调度语义冲突|是线性三段拆分。|想表达 pipeline overlap、DAG、stage readiness、异构 pool ratio。|





## Config 重构（\#4021）



主要冲突点在于 config model

1. Stage kind 需要有一个扩展性，支持 stage kind config 可扩展而不是硬编码 ar/generation/diffusion；提供好stage\-kind registry / discriminated union。





# Design



## 初步设计



大概的需求（不是 finally 版本）

- 做一个角色分离，拆分出一个专门的 toplevel （大概可以和 llm/diffusion 同级）

- Mixin 接入：diffusion 模型可以通过一次 mixin 继承就能拥有对应功能，这样子对于原来的 pipeline 侵入尽可能小，同时适配于大部分模型（比如 wan2\.2、qwen\-image 是必须的）

- 传输可以复用现有的 rdma/tcp 传输 embedding，初步版本可以用最简单的 同步拉取，在 runner/engine 上拉取，通过 input/output processor 处理



可能需要考虑的兼容点：

1. 兼容大部分主流模型

2. 后向兼容、给后续兼容多 encode/decode 留有接口

3. Diffusion engine 需要兼容 denoise（纯dit） \+ diffusion 情况







## 缓存/传输优化



## Scheduler 感知







