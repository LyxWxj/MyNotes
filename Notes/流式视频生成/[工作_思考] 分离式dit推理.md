# \[工作/思考\] 分离式dit推理





这里主要是基于 \#3208 分离式 dit 推理，来继续考虑更进一步的设计

1. 小角色走 Dag 图，可以实现并行 encode/decode \+ fused encode/decode，按照不同模型选择

2. 数据面传输更好：push 还是 pull，频繁的 cache 数据请求拉取、更加 robust 的设计

3. 如果做两个层次的调度，orch 上：更好的调度、限流；instance 上做请求的优先级调度和组批次







# background



Related 模型：

- 生图：hunyuan\-image（后续主要）、qwen\-image（当前实现 pr\#3208）

- 生视频：moss\-音视频、wan2\.2

- 流式：lingbot\-world、krea、omni\-forcing（if 开源）



总结，主要收益：

- 生图：layerwise offload \+ 流式pipeline

- 视频：dag 图 \+ 流式pipeline

- 流式视频：延迟更加敏感





流式的意思就是可以减少请求的 waiting bubble，提前处理好 vae/vit 这种 encoder 部分



---



Pr \#3208 做的是 only 3 stage：encode \-\- dit \-\- decode



1. Stage 数量比较膨胀、任务 stage 不一样：抽象

2. 依赖：dit 要依赖 upstream ，get\_DAG\_REDAY

3. Tensor shape：做好抽象

4. 调度：vllm （wait forkv transfer 状态）为了做 async runtime，引进 wait for dag 



# Design 



几类可能的设计思路：

1. 传算 overlap：可以参考 kvcache 得做法，在 scheduler 侧加入 wait\_for\_transfer 类似的状态，让 engine rpc 调用对应的上层 stage、进行拉取，没有被拉取依赖完得请求不会进入 step forward

核心设计思路：io/cpu 移除出 runner 关键路径，消除 gpu bubble 

对于视频/生图是有收益得，对于流式需要考虑延迟



2. Embedding cache 

可能的传输设计：

- 统一数据平面（类似于一个 redis 这样的缓存？）

- p2p 传输：下游角色 pull 拉取

- 为什么不用预推送（因为多 stage replica 的时候、不一定能够确定下一个 stage 要发过去的实例）





- 测试 tobe done



3. 设计上需要考虑：

    - 抽象一致性（统一的接口、executor/runner 做一个多态）：用一个统一的 submodule 可以管理好大部分任务、比如 image encoder / text encoder / audio encoder 得执行和数据传输、保证数据传输语义最简；

    - Dit/llm 主 runner 侧 merge 多 stage result 





# 测试



> Dag 
> 
> 

找一个音视频：audio e/d、vae e/d、wvae e/d 

- 视频场景：5s 视频 480p，4 step \+ int8 ，encode 各个小角色 ： dit 8卡 sp8 ulysses ： decode 各个小角色  

\[Wan22Profiling\.md\]

- 流式场景：1/3 latents \-\-\> encode \+ dit \+ decode（dit 默认 2/4 step、int8），encode 各个小角色  ： dit 8sp ： decode 各个小角色 vae streamning vae ）

    - Casualwan 1\.3b/14b 

\[CausalWanProfiling\.md\]



[Profiling\.md](https://my.feishu.cn/wiki/VZEiwQr0IijZACk9eFncB2Mknjg)



可以测试，算一下 bubble，多个请求并行的时候；





> 传输的时间
> 
> 



Cpu mock 一下可以测算出大概的时间

- Embedding tensor 大小：生图、视频（wan2\.2 音视频）、流式视频

    - 生图 1024 \* 1024 输入得参考图、生成的图

    - 视频 480p 5s/8s/10s 

    - 流式：1 / 3  latents 

测量结果：[VLLM\-Omni](https://my.feishu.cn/docx/AOJDdiuuTopxiAxDJm9carCGnaf?from=from_copylink)



> 周末：rfc ，下周一 release 到社区
> 
> 

我同步写一下设计方案

[\[RFC / 问题\] 分离 dit 推理 ](https://my.feishu.cn/wiki/EIpkwGBBGiPENCkGt4bcm4VTnld?from=from_copylink)

