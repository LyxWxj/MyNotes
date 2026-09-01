4 段流水线，batch_size = 2, micro_batch_size=1, CFG=True, Step=2

理想情况：

Pipeline Parallelism: 12 * Tick

Serial: 32 * Tick

![[PP4vsPP4.svg]]

实际上：

![[Pasted image 20260830112600.png]]

2 段流水线，batch_size = 2, micro_batch_size=1, CFG=True, Step=4

理想情况：

Pipeline Parallelism: 18 * Tick

Serial:32 * Tick

A100 上的实际 Trace

![[pp2-rank-component-timeline.svg]]

问题：

- 主要瓶颈是两槽 edge ring 的 credit 背压，其次是 stage 负载不均（重新调整布局）。

```text
两个 slot 都被占用
-> 下游尚未处理完
-> 没有 credit 返回
-> 上游暂停注入
-> 下游释放 slot 后恢复发送
```

快 stage 会被慢 stage 限制，避免快 stage 无限制地产生数据。当前 stage 想继续发送，但下游还没有归还可用 slot。

- 通信协议，通信组销毁顺序。
