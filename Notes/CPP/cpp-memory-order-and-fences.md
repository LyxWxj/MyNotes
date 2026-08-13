---
type: Note
---

# C++ 内存序与内存屏障

`std::memory_order` 用来定义原子操作周围的内存访问如何排序，以及哪些写入能被其他线程观察到。它解决的不是单个原子变量是否原子，而是多个变量之间的跨线程可见性。

## 先区分三个概念

- **原子性**：对同一个 `std::atomic` 的读写不会被撕裂，并且该原子对象有自己的修改顺序。
- **排序**：编译器和 CPU 可以在不改变单线程结果的前提下重排操作；内存序限制这种重排的可观察效果。
- **可见性**：一个线程写入普通数据后，另一个线程何时可以安全地读取它。

若两个线程并发读写同一个非原子对象，且两者之间没有 happens-before 关系，则为 data race，行为未定义。原子状态变量经常用于建立这条关系。

## 发布与订阅：最重要的模式

```cpp
#include <atomic>

int payload;
std::atomic<bool> ready = false;

// 线程 A：发布者
payload = 42;
ready.store(true, std::memory_order_release);

// 线程 B：订阅者
while (!ready.load(std::memory_order_acquire)) {}
use(payload); // 保证读到 42
```

前提是 B 的 acquire load 确实读到了 A 的 release store 写入的 `true`。此时产生如下关系：

```text
A: payload = 42
       ↓
A: ready.store(true, release)
       ↓ synchronizes-with
B: ready.load(acquire) == true
       ↓
B: use(payload)
```

因此 `payload = 42` happens-before `use(payload)`。`payload` 本身不需要是原子变量，因为它的写入和读取被这条同步链保护；发布后 A 不能再与 B 并发、无同步地修改它。

### release/acquire 不是“阻塞等待”

`ready.load(std::memory_order_acquire)` 会立即返回：

- 返回 `false` 时，B 不等待；只有代码中的循环使 B 自旋。
- 返回来自对应 release store 的 `true` 时，B 保证能看到 A 在该 release 前发布的数据。

如果希望等待而非空转，C++20 可以使用：

```cpp
ready.wait(false, std::memory_order_acquire);
use(payload);
```

`wait` 允许实现阻塞或休眠等待；memory order 本身不会让线程等待。

## 为什么 relaxed 不够

```cpp
// 线程 A
payload = 42;
ready.store(true, std::memory_order_relaxed);

// 线程 B
if (ready.load(std::memory_order_relaxed)) {
    use(payload); // 错误：没有同步
}
```

`relaxed` 只保证 `ready` 的读写是原子的，并只约束 `ready` 自身的修改顺序。它不要求 B 在看到 `ready == true` 时也看到 `payload = 42`。

于是 A 对 `payload` 的写和 B 对 `payload` 的读之间没有 happens-before 关系，构成 data race；这在标准层面是未定义行为，而不只是“偶尔读到旧值”。

## 内存序速查

| 内存序 | 作用 | 常见用途 |
| --- | --- | --- |
| `relaxed` | 仅原子性与该原子对象的修改顺序 | 统计计数、指标 |
| `acquire` | 后续读写不能表现得发生在该读之前；接收发布的数据 | 读取状态、成功加锁 |
| `release` | 之前读写不能表现得发生在该写之后；发布此前数据 | 写入状态、解锁 |
| `acq_rel` | 同时 acquire 与 release；用于读改写操作 | CAS、`fetch_add`、无锁结构 |
| `seq_cst` | acquire/release 效果外加全局单一顺序 | 默认值；先验证正确性 |
| `consume` | 仅数据依赖上的较弱 acquire | C++26 已弃用，避免使用 |

`load` 通常使用 `relaxed`、`acquire` 或 `seq_cst`；`store` 使用 `relaxed`、`release` 或 `seq_cst`；读改写操作才适合 `acq_rel`。

## release 到底保证什么

可以把 release 看作一个**发布边界**：

```cpp
payload1 = 42;
payload2 = 99;
ready.store(true, std::memory_order_release);
```

它保证：对于任何通过 acquire 读取到这个 `true` 的线程，上述两个 payload 的写入不能被观察成发生在 `ready = true` 之后。

它不意味着 A 必须把所有缓存行预先同步到所有 CPU，也不意味着 B 的 acquire load 会等待。实际可能是：B 观察到 `ready` 后，在随后读取 `payload1` 时通过缓存一致性协议按需取得最新缓存行。B 从不读取 `payload2` 时，通常无需把 `payload2` 的缓存行拿到 B 的缓存中。

## 一次发布覆盖一批数据，性能意味着什么

```cpp
// A
payload1 = 42;
payload2 = 99;
ready.store(true, std::memory_order_release);
```

这里不是每个 `payload` 各有一次屏障。逻辑上是一个发布点覆盖此前的一批写入；硬件通常也会以一个带 release 语义的 store 或一个屏障实现该边界。x86 上 release store 常接近普通 store；ARM 等弱内存序架构可能需要更强的专用指令或屏障。

即使 B 不读取 `payload2`，写入 `payload2` 本身仍有执行、缓存和一致性成本；但通常不会因此多出一份 release 屏障，B 也不会因此被迫获取 `payload2` 的缓存行。

若 `payload2` 与本次发布无关，且没有其他线程依赖“看到 `ready == true` 时 `payload2` 已完成”，可将它放在发布后：

```cpp
// A
payload1 = 42;
ready.store(true, std::memory_order_release); // 只发布 payload1

payload2 = 99;
```

这样让发布协议更精确，也给编译器和 CPU 更多调度空间。若 `payload2` 影响 `payload1` 的计算、B 也会读取它，或另一个线程依赖它，则不能这样移动。

## 显式屏障

```cpp
std::atomic_thread_fence(std::memory_order_release);
std::atomic_signal_fence(std::memory_order_acquire);
```

- `atomic_thread_fence` 是线程间内存屏障。通常仍需要原子变量作为发布/接收的载体，才会与另一线程建立同步。
- `atomic_signal_fence` 主要限制编译器优化，用于信号处理或特殊底层场景，不是一般多线程同步工具。

下面的 fence 写法可以表达发布协议，但业务代码通常应优先使用直接的 release/acquire，后者更清晰：

```cpp
// A
payload = 42;
std::atomic_thread_fence(std::memory_order_release);
ready.store(true, std::memory_order_relaxed);

// B
while (!ready.load(std::memory_order_relaxed)) {}
std::atomic_thread_fence(std::memory_order_acquire);
use(payload);
```

## 实践准则

1. 先使用 `mutex`、条件变量、队列、future 等高层同步工具；无锁代码才直接设计内存序。
2. 为每个共享普通数据的“写入者 -> 读取者”画出一条 release/acquire 的 happens-before 链。
3. 不确定时先用默认的 `seq_cst` 写对，再依据证明降低内存序。
4. `volatile` 不是标准 C++ 的线程同步工具：它不保证原子性，也不建立 happens-before。

## 参考

- [cppreference: `std::memory_order`](https://en.cppreference.com/cpp/atomic/memory_order)
