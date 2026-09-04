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

A100 有 108 个 SM,假设每个 SM 能够驻留 4 个 CTA,那么一个 Wave 能够容纳 432 个 CTA。grid=500 时第一个 wave 可以填充所有 SM,但是第二个 wave 只有 $\frac{64}{432}\sim 15.7\%$ 的 SM 在干活。

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

GigaThread 分发一个 CTA 到 SM 的时候需要做如下判定：

```text
设CTA配置：
	T = threads_per_block
	W = ceil(T / 32) # warp数量
	R = regs_per_thread # 编译器决定
	S = static_shmem + dynamic_shmem # 该CTA占的shared memory
	
SM 当前状态：
	used_W, used_R, used_S, used_B # 已用的warp/reg/shmem/block
判定 can_accept(CTA, SM):
	(used_W + W) <= 64 # warp slot
	(used_B + 1) <= 32 # block slot
	(used_R + W * R*32bit) <= 65536 # 寄存器
	(used_S + S) <= total_shmem # shared memory
以上条件需要同时成立
```

A100 和 H100 的 SM 资源上限如下:

```text
A100-SXM4 (GA100, CC 8.0) 每 SM 资源上限:
  寄存器:           65536 个 32-bit 寄存器
  Shared/L1 总量:   192 KB(可配置划分)
  Warp slot:        64(对应 2048 线程)
  Block slot:       32
  每 Block 最多:    1024 线程
  每 Block 最多:    99 KB 动态 shared memory(需 cudaFuncSetAttribute)

H100-SXM5 (GH100, CC 9.0) 每 SM 资源上限:
  寄存器:           65536 个 32-bit 寄存器
  Shared/L1 总量:   228 KB(可配置划分)
  Warp slot:        64(对应 2048 线程)
  Block slot:       32
  每 Block 最多:    1024 线程
  每 Block 最多:    227 KB 动态 shared memory
```

### Case1: vector add, block=256, regs=16, shmem=0

```text
A100 vec_add<<<grid,256>>>(...)
Threads = 256, W = Threads/32 = 8 warps, R = 16, S = 0
资源约束：
	warp slot: 64 / 8 = 8 CTA
	block slot: 32 / 1 = 32 CTA
	寄存器: 65536 / (8 x 32 x 16) = 16 CTA
	shared memory: 192KB/0 -
取最严格: min(8,32,16,-) = 8 CTA/SM
resident warp = 8 * 8 = 64 warps # 8个block,每个block 8warp,一共驻留64 warp
Occupancy = 64 / 64
```

Case2: GEMM tile, block=128, regs=128, shmem=64KB

```text
A100, gemm_kernel<<<grid, 128>>>(...)
	T = 128, W = 4 warps, R = 128, S = 64KB = 65535 B
资源约束:
	warp slot = 64/4 = 16 CTA
	block slot = 32 / 1 = 32 CTA
	寄存器: 65536 / (4 x 32 x 128) = 65536/16384 = 4 CTA
	share memory: 192KB / 64KB = 3 CTA
取最严格： 3 CTA / SM
resident warp = 3 x 4 = 12 warps
Occupancy = 12 / 64 = 18.75%

总wave size = 3 x 108 = 324 CTA
```

Case3 : 大 dynamic shared memory, block=1024, regs=24, dyn_shmem=120KB

```text
A100, big_shmem_kernel<<<grid, 1024, 120*1024>>>(...)
  需先 cudaFuncSetAttribute(MaxDynamicSharedMemorySize, 120*1024)
  T = 1024, W = 32 warps, R = 24, S = 120KB = 122880 B

资源约束:
  warp slot:    64 / 32 = 2 CTA
  block slot:   32 / 1 = 32 CTA
  寄存器:        65536 / (32 × 32 × 24) = 65536 / 24576 = 2 CTA
  shared memory: 164KB / 120KB = 1 CTA(默认 carveout)
                 192KB / 120KB = 1 CTA(最大 carveout)

取最严:min(2, 32, 2, 1) = 1 CTA/SM
resident warp = 1 × 32 = 32 warps
Occupancy = 32 / 64 = 50%

总 wave size = 1 × 108 = 108 CTA
```

这个例子展示了大 shared memory 的代价: 每 SM 只能放 1 个 CTA,wave size 从例 1 的 864 掉到 108。如果 grid 不是 108 的整数倍,tail effect 会非常严重。这就是为什么 Flash Attention 这类大 shared memory kernel 对 grid 数选择特别敏感。

### 延迟隐藏量化

```text
A100, HBM 访问延迟 ≈ 400 cycles
每 warp 每 4 cycle 能发射 1 条指令(因为 16 CUDA Core 服务 32 线程,2 cycle 一条)
所以一个 warp 等 HBM 时,需要其他 warp 填补 400 / 4 = 100 个发射槽

如果 resident warp = 64(满载):
  4 个分区 × 16 warp/分区,每分区 16 warp 轮流发射
  每 cycle 每分区选 1 个就绪 warp → 16 warp 足够覆盖 100 槽
  → HBM 延迟可被隐藏

如果 resident warp = 16(25% Occupancy):
  4 warp/分区,要覆盖 100 槽 → 每 warp 要发射 25 次才有 1 个就绪
  → 不够,HBM 延迟隐藏不掉,执行单元空转

如果 resident warp = 4(6% Occupancy):
  1 warp/分区,根本没得选 → 完全暴露 HBM 延迟
```

## 跨 CTA 的同步：Cooperative Groups

CTA 内的同步用 `__syncthreads`,但有些场景需要跨 CTA 同步: 例如 grid 级 reduce、跨 SM 的生产者 - 消费者。CUDA 9 引入的 Cooperative Groups 提供了跨 CTA 同步的接口,但代价高昂。这一节讲跨 CTA 同步的几种机制和它们的硬件成本。

### Cooperative Groups 简介

Cooperative Groups(CG) 的核心概念是 `thread_group`,常见的几种:

| thread_group 子类型 | 作用域 | 同步原语 |
|---|---|---|
| `thread_block` | CTA（Grid 内） | `group.sync()` = `__syncthreads()` |
| `warp` | Warp（32 线程） | `group.sync()` = `__syncwarp()` |
| `thread_block_tile<N>` | Warp 子组 | `group.sync()` = `__syncwarp(mask)` |
| `coalesced_group` | coalesced 线程 | `group.sync()` |
| `grid_group` | Grid（所有 CTA） | `group.sync()` = grid 同步 |
| `multi_grid_group` | 多 GPU 所有 CTA | `group.sync()` = 跨 GPU 同步 |

前四个是 CTA 内或更小的同步,本质上还是用 barrier slot 或 warp shuffle。后两个是跨 CTA、跨 GPU 的同步,需要专门的硬件支持。

### grid_group:Grid 级同步

`grid_group` 把整个 grid 作为一个同步组,`group.sync()` 让所有 CTA 同时到达一个点。这需要 kernel 用 `cudaLaunchCooperativeKernel` 启动,而不是普通 `<<<>>>`:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void grid_sync_kernel(float* data, int n) {
    cg::grid_group grid = cg::this_grid();
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 第一阶段:每个 CTA 处理自己的部分
    if (tid < n) {
        data[tid] = compute_phase1(data[tid]);
    }

    grid.sync();  // 所有 CTA 同步,等所有 phase1 完成

    // 第二阶段:跨 CTA 处理
    if (tid < n) {
        data[tid] = compute_phase2(data[tid]);
    }
}

// 启动:
cudaLaunchCooperativeKernel(
    (void*)grid_sync_kernel, grid, block,
    args, shmem, stream);
```

grid_group 的硬件实现是一个全局 barrier,通常用 GPU 全局内存里的一个计数器实现。每个 CTA 到达 `grid.sync()` 时,原子加一个全局计数器; 当计数器等于 grid_size 时,所有 CTA 同时继续。

grid 同步的约束很严格:**所有 CTA 必须同时驻留在 GPU 上**。这意味着 grid_size ≤ SM_count × resident_CTA_per_SM,也就是只能跑一个 wave。如果 grid 超过一个 wave,grid sync 会死锁 (后 wave 的 CTA 没法被调度,因为前 wave 还没释放资源)。

```text
A100, resident CTA = 4, SM = 108 → grid_group 最大 grid = 432
H100, resident CTA = 4, SM = 132 → grid_group 最大 grid = 528

如果 grid > 这个上限,cudaLaunchCooperativeKernel 返回错误
```

这个约束让 grid_group 在大 grid 场景几乎不可用。生产环境的 grid sync 通常只在”全局 reduce 的小 grid”场景用,例如多 GPU 训练的全局梯度同步前的预 reduce。

### multi_grid_group: 跨 GPU 同步

`multi_grid_group` 把多个 GPU 上的所有 CTA 作为一个同步组,需要 `cudaLaunchCooperativeKernelMultiDevice`。这个机制用 NVLink 或 PCIe 做跨 GPU barrier,延迟在微秒级,比 kernel 间同步快,但远慢于 CTA 内 barrier(纳秒级)。

multi_grid_group 的实际工程用途很少,因为:

- 跨 GPU barrier 延迟高,频繁 sync 会让 GPU 闲置;
- 跨 GPU 同步语义可以用 NCCL 通信 + stream event 更灵活地实现;
- multi_grid_group 要求所有 GPU 同时驻留 kernel,资源利用率低。

所以这个特性更多是”理论上有”的状态,生产环境几乎不用。

### 跨 SM 同步为何昂贵

CTA 内的 `__syncthreads` 开销几十到几百 cycle,跨 SM 的 grid sync 开销几千到几万 cycle,差异来自:

第一,**barrier 状态位置不同**。CTA 内 barrier 在 SM 本地的 barrier slot,访问是寄存器速度。跨 SM barrier 在全局内存,访问要走 L2,延迟 200+ cycle。

第二,**参与者数量不同**。CTA 内最多 32 warp,所有 warp 都在同一个 SM 的同一个 Scheduler 视野内,一次检查就能判断”是否全部到达”。跨 SM 涉及上千个 CTA,要原子操作全局计数器,竞争激烈。

第三,**资源约束不同**。CTA 内 barrier 不阻塞 GigaThread,SM 可以继续接收新 CTA。跨 SM barrier 要求所有 CTA 同时驻留,GigaThread 不能用 resident CTA 数来调度新 CTA,等于把 GPU 的并行调度能力废了。

| 同步类型 | 延迟 | 约束 | 典型场景 |
|---|---|---|---|
| `__syncwarp` | ~5 cycle | 无 | warp 内同步 |
| `__syncthreads` | 20-150 cycle | CTA 内 | CTA 内 barrier |
| `grid_group.sync` | 1-10 µs | 单 wave | 跨 CTA 全局 reduce |
| `multi_grid_group` | 10-100 µs | 单 wave + 多 GPU | 极少用 |
| kernel 间 sync | 5-20 µs | 无 | 普通跨 kernel 同步 |

最后一行”kernel 间 sync”指的是用 stream + event 实现的同步,虽然延迟和 grid_group 接近,但没有单 wave 约束,所以生产环境更常用。

> **适用边界**:Cooperative Groups 的 grid sync 适合”grid 小、必须显式同步”的场景,例如全局 reduce、跨 CTA 的扫描。普通 kernel 不要用 grid sync,改用多 kernel + stream event 更灵活。

## Persistent Kernel:CTA 不释放,循环消费任务

正常 kernel 是”launch → CTA 跑完 → 回收 → 退出”。Persistent Kernel 反过来:CTA 跑一个无限循环,反复从任务队列取新任务,直到收到退出信号。这种模式绕过了”反复 launch”的开销,在特定场景有数量级收益。这一节讲 Persistent Kernel 的概念、实现、适用场景。

### Persistent Kernel 的基本结构

```text
__global__ void persistent_kernel(TaskQueue* queue, int* should_exit) {
    // CTA 启动后不退出,循环消费任务
    while (true) {
        // 1. 从队列取任务(原子操作)
        Task task = queue->fetch_next();
        if (task.type == TASK_EXIT) break;
        if (task.type == TASK_INVALID) continue;  // 队列空,重试

        // 2. 处理任务
        process_task(task);

        // 3. 可选:标记任务完成
        queue->mark_done(task.id);
    }
}
```

启动时 grid 数固定为 SM_count × resident_CTA_per_SM(例如 A100 上 432),让所有 CTA 一次驻留到 SM 上,之后不再 launch。任务通过 global memory 的队列传递,CTA 原子地取任务、处理、标记完成。

### 与正常 launch 的对比

|维度|正常 launch|Persistent Kernel|
|---|---|---|
|launch 开销|每 kernel 一次|一次启动,长期运行|
|资源占用|每 kernel 重新分配|一次分配,长期占用|
|任务调度|GigaThread 调度|用户在 kernel 内调度|
|跨任务数据复用|难 (每 kernel 重新加载)|易 (shared memory 持久)|
|错误隔离|kernel 失败影响小|失败影响整个 persistent 实例|
|调试难度|易 (nsys 看 kernel 边界)|难 (一个长 kernel 内多任务)|
|Stream/Event 兼容|完全兼容|需要特殊设计|

最后两行是 Persistent Kernel 的主要代价。错误隔离差意味着一个任务崩溃会拖垮整个 persistent 实例; 调试难意味着 nsys 时间线上只有一个大 kernel,看不到内部任务边界。

### 8.3 Persistent Kernel 的适用场景

Persistent Kernel 不是万能药,适合的场景:

第一,**小 kernel 序列**。每个 kernel 自身 < 10 µs,launch 开销占比 > 30%。LLM 推理的 decode 阶段是典型——每 step 几十个 kernel,每个几微秒。

第二,**跨任务数据复用**。多个任务用同一份权重 (例如 attention 的 KV cache),Persistent Kernel 可以把权重常驻 shared memory,避免反复加载。

第三,**动态任务调度**。任务之间有依赖,无法预先编译成 CUDA Graph。Persistent Kernel 在 kernel 内做调度,比 host 端动态 launch 灵活。

不适合的场景:

第一,**大 kernel 序列**。每个 kernel 几十毫秒,launch 开销占比 < 1%,Persistent Kernel 没收益。

第二,**任务间无数据复用**。每个任务独立,Persistent Kernel 反而增加复杂度。

第三,**需要严格错误隔离**。一个任务失败不能影响其他任务。Persistent Kernel 的失败模式是整个实例退出。

第四,**简单部署**。Persistent Kernel 需要用户管理任务队列、退出信号、错误处理,工程复杂度高。如果 CUDA Graph 能解决问题,优先用 Graph。

### Hopper 的特邀特性:Tensor Memory Accelerator 与 Persistent Kernel

Hopper 引入的 TMA(Tensor Memory Accelerator) 和 `mbarrier` 让 Persistent Kernel 的实现更高效。TMA 可以异步搬运数据,不需要 warp 主动发指令;`mbarrier` 支持异步等待,不需要 warp 阻塞。这让 Persistent Kernel 内部可以实现 warp specialization: 一部分 warp 专门做搬运,一部分 warp 专门做计算,通过 `mbarrier` 协调。

```text
Hopper Persistent Kernel(Warp Specialization):
  Warp 0-3: TMA producer,循环 cp.async.bulk + mbarrier.arrive
  Warp 4-7: mma consumer,循环 mbarrier.wait + wgmma + mbarrier.arrive
  Warp 8-11: TMA consumer,循环 mbarrier.wait + cp.async.bulk(store)

  所有 warp 在一个 Persistent Kernel 内,长期驻留
  通过 mbarrier 协调,无 __syncthreads 开销
```

这种模式在 CUTLASS 3.x 的 Hopper GEMM/Attention kernel 里被广泛使用,TensorRT-LLM 的 attention kernel 也是类似设计。

## 用 ncu 和 nsys 观察 CTA 调度

### 用 ncu 观察 resident CTA 数

ncu(Nsight Compute) 是 per-kernel profiler,能给出 kernel 的资源占用、Occupancy、wave 数等指标。下面是一个典型 kernel 的 ncu 报告:

```text
$ ncu --set full --kernel-name regex:my_kernel ./my_app

  My Kernel (grid=500, block=256)

  Resource Usage:
    Registers per thread:              32
    Stack frame per thread:            0 B
    Static shared memory:              4.06 KB
    Dynamic shared memory:             0 B
    Threads per block:                 256

  Occupancy:
    Theoretical active warps per SM:   64 / 64  (100%)
    Theoretical active blocks per SM:  8 / 32   (25%)
    Achieved active warps per SM:      58.2 / 64  (91%)    ← 实际略低于理论
    Achieved occupancy:                91%

  Launch Statistics:
    Waves per multiprocessor:          1.157    ← 不是整数,tail effect
    Block limit per SM due to warps:   8
    Block limit per SM due to regs:    16
    Block limit per SM due to shmem:   32
    Block limit per SM:                8        ← 取最严:warp slot

  Scheduler Statistics:
    Warps eligible per cycle:          11.3
    Warps issued per cycle:            3.8
    Issue rate:                        47.5%
    Stall reason (barrier):            8.2%
    Stall reason (memory):             31.5%
    Stall reason (short scoreboard):   12.7%
```

- `Theoretical active blocks per SM: 8` 是理论 resident CTA 数,根据 kernel 配置和 SM 资源算出。这里 warp slot 是最严约束 (8 个 CTA × 8 warp = 64 warp 满载)。
- `Achieved occupancy: 91%` 是实测的 Occupancy,略低于理论值。差异来自 kernel 末尾的 warp 退出不同步——部分 warp 退出后,resident warp 数下降。
- `Waves per multiprocessor: 1.157` 反映 wave 数不是整数。grid=500, wave=432, 500⁄432=1.157。tail effect 占总时间的 0.157⁄1.157 ≈ 13.6%。
- `Stall reason (barrier): 8.2%` 反映 warp 等 `__syncthreads` 的时间。这个比例高,说明 barrier 是瓶颈。

### 用 nsys 观察 CTA 调度时间线

nsys(Nsight Systems) 是 system-wide profiler,能看到 kernel 在时间线上的分布。但 nsys 本身不直接显示 per-CTA 调度,要看 CTA 调度需要用 ncu 的 `--target-processes all` + 特定指标。

下面是一段 nsys 输出,展示 grid=500 的 kernel 在 A100 上的执行时间线:

```text
$ nsys profile --trace=cuda --stats=true ./my_app

  Time(%)  Total Time   Avg         Min         Max         Call Count   Name
  -------  -----------  ----------  ----------  ----------  -----------  --------
     78.3  15.66 ms     15.660 ms   15.660 ms   15.660 ms   1            [kernel] my_kernel
     21.7  4.34 ms      4.340 ms    4.340 ms    4.340 ms    1            cudaLaunchKernel
      0.0  0.02 ms      0.020 ms    0.020 ms    0.020 ms    1            cudaDeviceSynchronize

  CUDA Kernel Execution Statistics:
    kernel       grid        block      duration    sm occupancy
    my_kernel    (500,1,1)   (256,1,1)  15.66 ms    91%

  GPU Timeline:
    时间(ms)   0      4      8      12     16
              │      │      │      │      │
    my_kernel ├──────────────────────────────┤
              0.00                          15.66

    CTA 分发(简化,基于 ncu --ctp):
      wave 1 (CTA 0..431):   0.00 - 8.20 ms   (T = 8.20 ms, 100% SM)
      wave 2 (CTA 432..499): 8.20 - 15.66 ms  (T' = 7.46 ms, 16% SM)
      → tail effect: 第二 wave 只有 68/432 = 15.7% SM 在工作
      → tail 时间占比: 7.46 / 15.66 = 47.6%
```

nsys 时间线上只能看到一个长 kernel,但通过 ncu 的 `--ctp`(CTA timeline profile) 模式可以拿到 per-CTA 的调度时间。从上面的简化时间线看,wave 2 的 68 个 CTA 跑了 7.46 ms,几乎和 wave 1 的 432 个 CTA 一样久——这就是 tail effect 的直接观察。

### Occupancy 调优实例

下面用一个实际例子展示 Occupancy 调优的过程。kernel 是 reduction,grid=500, block=256,每线程 32 寄存器:

```text
__global__ void reduce_v1(const float* in, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // 树形 reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) out[blockIdx.x] = smem[0];
}
```

ncu 报告 (基线):

```text
reduce_v1 (grid=500, block=256)
  Registers per thread:              24
  Theoretical active blocks per SM:  8 (warp slot 限制)
  Achieved occupancy:                87%
  Waves per multiprocessor:          1.157
  Issue rate:                        32%
  Stall reason (barrier):            35%    ← __syncthreads 占了 35%!
  Stall reason (memory):             18%
  Kernel time:                       2.1 ms
```

观察:`__syncthreads` 占了 35% 的 stall,是主要瓶颈。原因是 reduction 的树形结构里,每轮 reduce 都要一次 barrier,8 轮就是 8 次 barrier。每次 barrier 在 block=256 下开销约 45 cycle,8 次就是 360 cycle。

调优方向 1: 把最后几轮 reduce 改用 warp shuffle(无 barrier):

```text
__global__ void reduce_v2(const float* in, float* out, int n) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    smem[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // 树形 reduce 到 32 个元素
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    // 最后 32 个元素用 warp shuffle(无 barrier)
    float v = smem[tid];
    for (int s = 16; s > 0; s >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, s);
    }

    if (tid == 0) out[blockIdx.x] = v;
}
```

ncu 报告 (优化后):

```text
reduce_v2 (grid=500, block=256)
  Registers per thread:              24
  Theoretical active blocks per SM:  8
  Achieved occupancy:                87%
  Waves per multiprocessor:          1.157
  Issue rate:                        41%    ← 提升
  Stall reason (barrier):            18%    ← 从 35% 降到 18%
  Stall reason (memory):             17%
  Kernel time:                       1.6 ms  ← 从 2.1 ms 降到 1.6 ms,提升 24%
```

调优方向 2: 增大 block size 提高 warp shuffle 占比,同时降低 resident CTA 数。但要注意 Occupancy 下降可能影响延迟隐藏:

```text
reduce_v3 (grid=250, block=512)  ← block 翻倍,grid 减半
  Registers per thread:              24
  Theoretical active blocks per SM:  4 (warp slot: 64/16=4)
  Achieved occupancy:                43%   ← 翻倍 block 后 Occupancy 减半
  Waves per multiprocessor:          0.579 ← 半个 wave!
  Issue rate:                        38%
  Stall reason (barrier):            22%
  Kernel time:                       1.8 ms ← 反而变慢
```

block=512 时 Occupancy 减半,wave 数变成 0.579(只用了半个 wave),反而变慢。这说明 Occupancy 不是越高越好,但要避免撞到极端值。

调优方向 3: 回到 block=256,但调整 grid 到 wave size 整数倍:

```text
reduce_v4 (grid=432, block=256)  ← grid 调到 wave size
  Waves per multiprocessor:          1.0   ← 整数 wave,无 tail
  Achieved occupancy:                87%
  Kernel time:                       1.4 ms ← 比 v2 又快 12%
```

虽然 grid=432 比 grid=500 少处理 13% 的数据,但因为没有 tail effect,反而更快。这种调优在生产环境很常见——把 grid 数调到 wave size 整数倍,即使损失少量并行度也划算。

这个例子展示了 CTA 调度的几个调优方向:

1. 减少 `__syncthreads` 调用 (warp shuffle 替代);
2. block size 调整影响 Occupancy 和 wave size;
3. grid 数调到 wave size 整数倍消除 tail effect;
4. 实测验证每个调优,不要凭直觉。

### ncu 指标速查表

把本章涉及的 ncu 指标整理成速查表:

| 指标                                                  | 含义                   | 关注阈值                       |
| --------------------------------------------------- | -------------------- | -------------------------- |
| sm__cycles_active.avg                               | SM 活跃周期比例            | < 60% 说明 SM 闲置多            |
| smsp__inst_executed.sum                             | 总指令数                 | 用于算 IPC                    |
| smsp__warps_active.avg.pct_of_peak_sustained_active | Achieved Occupancy   | 与 Theoretical 比,差异大说明调度问题  |
| launch__waves_per_multiprocessor                    | wave 数               | 小数部分 > 0.1 关注 tail effect  |
| launch__block_size                                  | block size           | 用于验证配置                     |
| launch__grid_size                                   | grid size            | 用于验证配置                     |
| smsp__pcsamp_warps_issue_stalled_barrier            | barrier stall 比例     | > 20% 说明 __syncthreads 是瓶颈 |
| smsp__pcsamp_warps_issue_stalled_membar             | membar stall         | 高说明 memory fence 多         |
| sm__sass_thread_inst_executed_op_local_ld           | register spill load  | > 0 说明寄存器压力                |
| sm__sass_thread_inst_executed_op_local_st           | register spill store | 同上                         |

## 思考问题题

- 假设 A100 上一个 kernel 配置为 block=256, regs=40, shmem=32KB,resident CTA per SM 是多少?如果要把 resident CTA 提到 8,可以调整哪些参数?各自的代价是什么?
> [!info] 解答
> `block=256` 意味着每个 CTA 有 `ceil(256/32)=8` 个 warp。只看 warp、block 和线程数上限时，分别可以容纳 `64/8=8`、`32/1=32`、`2048/256=8` 个 CTA。寄存器需求为 `40*256=10240` 个 32-bit 寄存器/CTA，`floor(65536/10240)=6`；32 KB shared memory 在 192 KB 配置下最多也是 6 个 CTA。因此理论上限是 `min(8,32,8,6,6)=6 CTA/SM`。实际寄存器和 shared memory 按硬件分配粒度向上取整，最终值应以编译器资源报告或 Occupancy API 为准。
> 
> 要达到 8 CTA/SM（仍使用 256 threads/block），至少要同时满足 `regs <= floor(65536/(8*256))=32` 个/线程，以及 `shmem <= 192/8=24 KB`/CTA，并确认分配粒度没有把结果再向下取整。降低寄存器可以重排代码、减少 live range 或使用 `__launch_bounds__`，代价是可能发生 local-memory spill，增加访存和延迟；降低 shared memory 可以缩小 tile 或改用重算，代价是全局内存流量增加、数据复用下降。减小 block（如 128 threads）也可能提高 CTA 数，但会改变并行归约、同步和每 CTA 工作量，且若 shared memory/其他资源仍是瓶颈，未必真的达到 8 CTA。
- GigaThread 按 blockIdx 顺序分发 CTA,但 CUDA 编程模型说”Block 之间无序执行”。这两个说法矛盾吗?从硬件实现和编程模型契约两个角度解释。
> [!info] 解答
> 不矛盾。硬件实现可以按 blockIdx 顺序生成和尝试分发 CTA，也可以采用轮转、分批或其他策略；这是当前 GPU 的调度细节，不是 CUDA 程序可依赖的顺序。编程模型只承诺 block 之间没有执行先后、进度或同一 SM 的保证，因此 kernel 不能依赖 block 0 先于 block 1 完成，也不能用普通 block 间同步表达这种依赖。只有显式的 cooperative launch/grid synchronization 等机制，才会增加相应的同步契约。
- CTA 一旦调度到 SM 就不能迁移,这是硬件限制还是设计选择?如果未来 GPU 支持 CTA 迁移,会带来什么收益和什么新问题?
> [!info] 解答
> 这是“本地资源绑定导致的硬件限制”，也是为了降低实现复杂度而作出的设计选择。CTA 的寄存器、shared memory、barrier/mbarrier 状态和部分本地缓存状态都绑定在原 SM；迁移必须暂停并检查所有 warp，再搬运或重建这些状态。若未来支持迁移，可以改善长尾负载、支持更强的抢占/容错，并提高动态工作负载的 SM 利用率；代价是状态检查点和搬运开销、迁移期间的缓存失效与同步复杂度，以及对 `__syncthreads`、指针可见性、性能可预测性和一致性语义提出新的约束。更现实的实现通常是可抢占的 CTA 或任务级重调度，而不是任意时刻透明迁移。
- `__syncthreads_count(pred)` 比手写”先 `__syncthreads` 再 atomicAdd shared memory 再 `__syncthreads`“快得多。从硬件指令和 barrier 实现的角度,解释为什么这个差异是结构性的,而不是常数因子的。
> [!info] 解答
> `__syncthreads_count` 的语义可以由硬件的 barrier reduction（PTX 中类似 `bar.red.popc`）一次完成：线程到达同一个 barrier，同时对 predicate 做 popcount，并把结果广播给参与线程。手写版本至少包含两次 barrier、shared-memory 写入/读取和一次 atomic 更新，还要处理 atomic 的串行化与内存顺序。前者把“同步+归约”融合为一个硬件原语，后者是多个阶段和共享状态之间的依赖，所以差异来自操作数量、同步轮次和争用结构，而不只是某条指令快几个周期。该优化仍要求 block 中所有线程以一致方式参与。
- 一个 kernel 的 grid=500, wave size=432,实测总时间 1.95 ms,而 grid=432 的总时间 1.00 ms。多 1 个 CTA 让时间翻倍。除了”调 grid 到 432”还有什么缓解方法?各自的代价是什么?
> [!info] 解答
> 这里实际多出的不是 1 个而是 68 个 CTA；它们形成第二个、只有 `68/108` 个 SM 有工作的尾部 wave。可选办法包括：
> - 降低寄存器或 shared memory、调整 block size，提高每 SM 的 resident CTA 数，从而增大 wave size；代价是 spill、较少的数据复用或更多 block/同步开销。
> - 让每个 CTA 处理更多元素，使用 grid-stride loop，把 CTA 数压到接近一个或整数个 wave；代价是单 CTA 运行时间变长，负载不均时尾部更明显。
> - 使用 persistent kernel 或设备端 work queue，让固定数量的 resident CTA 持续领取任务；代价是实现复杂度、队列同步开销和错误隔离变差。
> - 将独立的小任务融合，或与其他 kernel 做并发执行以填充尾部；代价是资源竞争、cache 干扰和调度复杂度。CUDA Graph 只能降低 launch 开销，不能消除尾部本身。
- grid_group.sync 要求所有 CTA 同时驻留 (grid ≤ wave size)。如果 grid 超过这个限制,为什么是死锁而不是排队?从 GigaThread 的资源分配角度解释。
> [!info] 解答
> 已驻留的第一批 CTA 到达 `grid_group.sync` 后不会结束，因此不会释放寄存器、shared memory、warp/block slot。剩余 CTA 仍在 GigaThread 的待分配队列中，因没有资源无法驻留；而 barrier 又要求它们也到达后才能放行，于是形成“已驻留 CTA 等待未驻留 CTA、未驻留 CTA 等待资源释放”的循环等待，不存在普通队列那样的阶段性退出点。实际使用 cooperative launch 时，runtime 通常会在启动前检查 grid 是否超过同时驻留容量并返回错误；若检查被绕过或条件在运行时发生，才表现为上述死锁。
- Persistent Kernel 在 LLM 推理 decode 阶段有显著收益,但在训练场景几乎不用。从 grid 数、kernel 大小、错误隔离、shape 变化四个角度分析为什么。
> [!info] 解答
> Decode 每次计算量小、迭代次数多，普通 kernel 的 launch 和调度开销占比高；persistent kernel 让固定 CTA 常驻并循环取 token/task，收益明显。训练通常是大矩阵/大 batch，grid 很大且单个 kernel 已运行较久，launch 开销可被摊薄，普通调度也能充分填满 SM。持久 kernel 还会把一个长期运行实例变成错误传播边界：一个任务的异常、非法访问或队列错误可能使整批推理失败；训练本来就常以整步/整作业失败，收益较小。最后，decode 的 shape 和请求数频繁变化，持久 kernel 可以通过队列吸收；训练 shape 通常规则但每层、每步资源需求变化大，固定 resident 配置会造成资源浪费，并增加与通信、检查点和调试工具的协同成本。
- ncu 报告里 `Achieved Occupancy` 通常比 `Theoretical Occupancy` 低 5-15%。从 CTA 完成、warp 退出、tail effect 三个角度解释这个差异的来源。
> [!info] 解答
> `Theoretical Occupancy` 是资源约束允许的峰值驻留数，假设整个执行区间都能维持该峰值；`Achieved Occupancy` 是时间平均的 active warps。CTA 完成后会立即释放资源，最后一个 wave 中只有部分 SM 仍有 CTA，平均 active warp 数随之下降。CTA 内某些 warp 可能因条件分支、边界检查或提前 return 而结束/变为空闲，即使 CTA 尚未整体完成，也会降低 active warp 数。grid 不能整除 wave size 时还会出现 tail effect：尾部 wave 只占用部分 SM，且长 CTA 会拖住结束时间。因此 5-15% 只是常见现象，不是固定规律；还应排查 divergence、内存等待和采样时间窗口。
- 一个 reduction kernel 在 block=256 时 barrier stall 占 35%,改用 warp shuffle 后降到 18%。如果继续把 block 从 256 改到 128,barrier stall 会进一步降低吗?Occupancy 会怎么变?综合来看哪个配置最优?
> [!info] 解答
> 不一定。128-thread block 的单次 block barrier 参与者更少，可能降低 barrier 延迟；但 block 数会增加，跨 block 的归约/atomic、边界处理和调度次数可能增加，stall 百分比也可能因分母变化而上升。若寄存器按线程使用且 shared memory 按 block 使用，减小 block 往往允许更多 resident block；不过在 warp 或线程上限成为瓶颈时，256（8 warps×8 blocks）和 128（4 warps×16 blocks）都可能达到相同的 64 resident warps，即 theoretical occupancy 不变。最终应比较端到端时间、`smsp__pcsamp_warps_issue_stalled_barrier`、Achieved Occupancy、指令/内存吞吐和归约结果；若 128 的 barrier 降幅不能抵消更多 block 和访存开销，256 仍然更优，不能仅凭 occupancy 判定。
- 设计一个实验,验证”tail effect 的严重程度与 wave 数小数部分成正比”。需要控制哪些变量,观察哪些指标,如何排除其他因素干扰?
> [!info] 解答
> 固定 GPU、时钟/功耗状态、编译选项、block size、寄存器数、shared memory、每个 CTA 的工作量和 kernel 代码，只改变 grid，使 `grid = k*wave_size + r`，覆盖多个 `r/wave_size`（例如 0、0.25、0.5、0.75、接近 1）。使用 warm-up 后重复运行，固定 stream，隔离其他 GPU 工作，并用足够大的 `k` 降低单次 launch 噪声。记录 kernel duration、`launch__waves_per_multiprocessor`、`sm__cycles_active.avg`、`smsp__warps_active.avg.pct_of_peak_sustained_active`、SM 利用率和最后一个 wave 的时间；用整 wave (`r=0`) 的时间作基线，比较额外 tail 时间与 `r/wave_size` 的回归关系。应同时用等时长的合成 CTA（例如固定迭代次数的依赖计算）验证，避免内存/cache 和 CTA 工作量差异混入；改变 `k`、重复次数并报告均值和置信区间。线性关系只是近似，因为 CTA 时长差异、调度顺序、频率变化和测量窗口都会使 tail 成本偏离严格正比。
