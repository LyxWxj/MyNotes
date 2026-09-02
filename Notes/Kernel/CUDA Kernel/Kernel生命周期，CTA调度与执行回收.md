一个 kenel 会被拆分成若干 Block（Aka. CTA, Cooperative Thread Array）每个 CTA 落到某个 SM 上执行。

![[CTA life cycle.png]]

- GigaThread 将 lauch 命令展开为 CTA list,按 SM 资源可用性逐个分发。
- SM 收到 CTA 后做资源检查：寄存器，shared memory,warp slot, block slot 这四类资源必须同时满足，否则 GigaThread 跳过这个 SM 寻找下一个
- CTA 在 SM 上驻留，所有 warp 由 SM 内的 Warp Scheduler 调度执行，CTA 一直驻留到所有 warp 执行完最后一条指令。第四段是 SM 回收 CTA 占用的寄存器，shared memory, warp slot, block slot,然后同志 GigaThread 可以接收新的 CTA。

## GigaThread

![[GigaThread.png]]GigaThread Engine 是 GPu 上一个独立的微码控制器，物理上位于芯片中央，与所有 SM 都有等距连线。它的核心职责是：接收 Front-End 来的 lauch command,把 grid 维度展开成 CTA 序列，根据每个 SM 上报的资源状态把 CTA 分发下去。

GigaThread 内部维护一张 SM 资源占用表：

- 已用 warp slot/ 总 warp slot（64 for A100/H100）
- 已用 block slot/总 block slot（32 for A100/H100）
- 已用寄存器/总寄存器（65536 个 32bit 寄存器 for A100/H100）
- 已用 shared memory/总 shared memory(192KB for A100, 228KB for H100)
每分发一个 CTA 到某个 SM,就在表中对应 SM 的资源占用减去这个 CTA 的需求。每收到一个 SM 的 CTA 完成通知就将对应的资源占用加回来。

### 分发顺序与负载均衡

每个 block 对应一个 CTA,那么一共会有 $|\text{grid}|=\text{grid.x} * \text{grid.y} * \text{{grid.z}}$ 个 block（CTA）

GigaThread 按 blockIdx 的顺序进行贪心分配，即将 CTA 分配到第一个合适的 SM 上，如果没有合适的 SM 就等待，伪代码：

```python
for CTA in CTA_list:
	accepted = False
	for SM in SM_pool: # 顺序检查或者从轮转检查
		if SM.can_accept(CTA):
			accepted=True
			SM.accept(CTA)
			update_sm_resource_table(SM,CTA)
			break
	if False==accepted:
		wait_for_any_sm_done()
		retry
```

1. CTA 分发按 blockIdx 顺序，但是落到哪个 SM 是不保证的。
2. SM 的选择是贪心的而不是最优的，对于每个 CTA，只要找到一个合适的 SM 就直接分配。
3. 资源不够时排队。

### Wave

一个 Wave 表示所有 SM 可以同时驻留的 CTA 总数: $$|\text{Wave}|=\text{SM}_{\text{counts}}*\text{resident CTA per SM}$$

那么 Wave 的数量就是:

$$
\text{Wave}_{\text{count}} =\left\lceil  \frac{|\text{grid}|}{|\text{Wave}|}  \right\rceil 
$$

A100 有 108 个 SM,假设每个 SM 能够驻留 4 个 CTA,那么一个 Wave 能够容纳 432 个 CTA。grid=500 时第一个 wave 可以填充所有 SM,但是第二个 wave 只有 $\frac{64}{432}\sim_{15}.7\%$ 的 SM 在干活。

```text
A100 (108 SM), resident CTA = 4 → wave size = 432 CTA

grid=432 (1 wave, 完美填充):
  SM0: CTA0, CTA1, CTA2, CTA3   ┐
  SM1: CTA4, CTA5, CTA6, CTA7   │
  ...                           │  所有 SM 同时满载
  SM107: CTA428..431            ┘
  → 1 wave 完成,所有 SM 一起退出

grid=500 (1 wave + 68 CTA 尾部):
  第一 wave (432 CTA): 所有 SM 满
  尾部 (68 CTA): 68/108 ≈ 63% SM 各跑 1 个 CTA,37% SM 空闲
  → 尾部这一段时间 SM 利用率从 100% 掉到 63%

grid=864 (2 wave, 完美填充):
  两个 wave 都满,SM 利用率全程 100%
```

ncu 提供了直接读 wave 数的指标 `launch__waves_per_multiprocessor`,它返回的就是这个值的浮点数形式 (例如 grid=500 时是 500⁄432 ≈ 1.157)。

### SM 间负载不均衡的影响

GigaThread 按 SM ID 顺序贪心扫描,会导致一个微妙的现象:SM ID 小的 SM 总是先收到 CTA,SM ID 大的 SM 在 grid 较小时可能完全空闲。

如果这个 kernel 是计算密集的就出现了大量的算力浪费，如果这个 kernel 是访存密集的，反而可能只用了部分 SM 而避免 L2/HBM 的带宽争抢。这是为什么有些小 GEMM 在小 grid 下反而比大 grid 更快——SM 数少了但是每个 SM 得到的 L2 带宽更大了。

SM 间不均衡的另一个来源是 CTA 内部工作量不均。如果一个 kernel 的某些 CTA 因为数据原因跑得比别人慢 (例如 reduction 的最后一个 CTA 处理尾部元素),这些 CTA 所在的 SM 会拖累整个 wave。GigaThread 不会重新分配 CTA,所以这种不均衡只能靠 kernel 设计来缓解。

> [!info] Q: 为什么 CTA 不能跨 SM 迁移
> - 寄存器和 shred memory 是 SM 本地资源。迁移意味着要把这些状态完整搬到另一个 SM。
> - barrier 状态是 SM 本地的。`__syncthreads()` 用到的 barrier slot, `cooperative_groups`D barrier 对象绑定在 SM 上。
> - 迁移开销远大于等待。

## Hopper 架构的演进

Hopper 在此基础上引入了 SM-local Scheduler,把部分调度权下放到 SM 内部,这是为 TMA 和 Warp Specialization 服务的。

Hopper 之前的调度模型有一个问题:GigaThread 串行分发 CTA,每分发一个 CTA 要查一次 SM 资源表。grid 数大时 (例如 4096 个 CTA),即使每个 CTA 只花 5 ns,GigaThread 也要 20+ µs 才能把所有 CTA 分发完。

Hopper 的做法是引入 SM-local Scheduler:GigaThread 仍然负责”把一组 CTA 分给某个 SM”,但分到 SM 后,SM 内部的 Scheduler 决定何时拉起这些 CTA、何时让它们让出 warp slot。这种”两级调度”让 SM 有更大自主权,支持:

- **CTA 内的 warp group 调度**。Hopper 引入 warp group 概念 (4 个 warp = 128 线程为一组),SM Scheduler 可以按 warp group 而不是 warp 为单位调度,减少调度开销。

- **TMA 异步搬运与计算重叠**。SM Scheduler 知道 TMA 操作的进度,可以在 TMA 搬运时让计算 warp 继续,在 TMA 完成时唤醒等待的 warp。这不需要 GigaThread 介入。
- **mbarrier 的硬件支持**。Hopper 的 mbarrier(memory barrier) 是 SM Scheduler 直接管理的硬件对象,比传统 `__syncthreads` 更灵活,支持异步等待、超时、计数。

## Occupancy 计算

### Case1: vector add, block=256, regs=16, shmem=0
```text
A100 vec_add<<<grid,256>>>(...)
T = 256, W = 8 warps, R = 16, S = 0
资源约束：
	warp slot: 
```