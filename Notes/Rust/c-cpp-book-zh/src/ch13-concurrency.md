# Rust 并发

> **你将学到：** Rust 的并发模型 —— 线程、`Send`/`Sync` 标记特征、`Mutex<T>`、`Arc<T>`、通道，以及编译器如何在编译期防止数据竞争。不使用的线程安全特性不会带来运行时开销。

- Rust 内置了对并发的支持，类似于 C++ 中的 `std::thread`
    - 关键区别：Rust 通过 `Send` 和 `Sync` 标记特征**在编译期防止数据竞争**
    - 在 C++ 中，跨线程共享 `std::vector` 而不使用互斥锁是未定义行为，但可以正常编译。在 Rust 中，这无法通过编译
    - Rust 中的 `Mutex<T>` 包裹的是**数据**，而不仅仅是访问权 —— 你在不加锁的情况下根本无法读取数据
- `thread::spawn()` 可用于创建一个独立的线程，并行执行闭包 `||`
```rust
use std::thread;
use std::time::Duration;
fn main() {
    let handle = thread::spawn(|| {
        for i in 0..10 {
            println!("Count in thread: {i}!");
            thread::sleep(Duration::from_millis(5));
        }
    });

    for i in 0..5 {
        println!("Main thread: {i}");
        thread::sleep(Duration::from_millis(5));
    }

    handle.join().unwrap(); // The handle.join() ensures that the spawned thread exits
}
```

# Rust 并发
- `thread::scope()` 可用于需要从环境中借用的场景。这是可行的，因为 `thread::scope` 会等待内部线程返回
- 尝试在不使用 `thread::scope` 的情况下执行此练习，看看会出什么问题
```rust
use std::thread;
fn main() {
  let a = [0, 1, 2];
  thread::scope(|scope| {
      scope.spawn(|| {
          for x in &a {
            println!("{x}");
          }
      });
  });
}
```
----
# Rust 并发
- 我们也可以使用 `move` 将所有权转移给线程。对于像 `[i32; 3]` 这样的 `Copy` 类型，`move` 关键字会将数据复制到闭包中，原始数据仍然可用
```rust
use std::thread;
fn main() {
  let mut a = [0, 1, 2];
  let handle = thread::spawn(move || {
      for x in a {
        println!("{x}");
      }
  });
  a[0] = 42;    // Doesn't affect the copy sent to the thread
  handle.join().unwrap();
}
```

# Rust 并发
- `Arc<T>` 可用于在多个线程之间共享*只读*引用
    - `Arc` 代表原子引用计数。引用在引用计数降为 0 之前不会被释放
    - `Arc::clone()` 只是增加引用计数，不会克隆数据
```rust
use std::sync::Arc;
use std::thread;
fn main() {
    let a = Arc::new([0, 1, 2]);
    let mut handles = Vec::new();
    for i in 0..2 {
        let arc = Arc::clone(&a);
        handles.push(thread::spawn(move || {
            println!("Thread: {i} {arc:?}");
        }));
    }
    handles.into_iter().for_each(|h| h.join().unwrap());
}
```

# Rust 并发
- `Arc<T>` 可以与 `Mutex<T>` 结合使用来提供可变引用。
    - `Mutex` 守护受保护的数据，确保只有持有锁的线程才能访问。
    - `MutexGuard` 在离开作用域时自动释放（RAII）。注意：`std::mem::forget` 仍然可以泄漏守卫 —— 所以"不可能忘记解锁"比"不可能泄漏"更准确。
```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = Vec::new();

    for _ in 0..5 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            let mut num = counter.lock().unwrap();
            *num += 1;
            // MutexGuard dropped here — lock released automatically
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Final count: {}", *counter.lock().unwrap());
    // Output: Final count: 5
}
```

# Rust 并发：RwLock
- `RwLock<T>` 允许**多个并发读取者**或**一个独占写入者** —— 这是 C++ 中的读写锁模式（`std::shared_mutex`）
    - 当读操作远多于写操作时使用 `RwLock`（例如配置、缓存）
    - 当读写频率相近或临界区很短时使用 `Mutex`
```rust
use std::sync::{Arc, RwLock};
use std::thread;

fn main() {
    let config = Arc::new(RwLock::new(String::from("v1.0")));
    let mut handles = Vec::new();

    // Spawn 5 readers — all can run concurrently
    for i in 0..5 {
        let config = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            let val = config.read().unwrap();  // Multiple readers OK
            println!("Reader {i}: {val}");
        }));
    }

    // One writer — blocks until all readers finish
    {
        let config = Arc::clone(&config);
        handles.push(thread::spawn(move || {
            let mut val = config.write().unwrap();  // Exclusive access
            *val = String::from("v2.0");
            println!("Writer: updated to {val}");
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
```

# Rust 并发：Mutex 中毒
- 如果一个线程在持有 `Mutex` 或 `RwLock` 时**发生了 panic**，锁就会变为**中毒**状态
    - 后续调用 `.lock()` 会返回 `Err(PoisonError)` —— 数据可能处于不一致状态
    - 如果你确信数据仍然有效，可以使用 `.into_inner()` 来恢复
    - 这在 C++ 中没有等价概念 —— `std::mutex` 没有中毒机制；发生 panic 的线程只是让锁一直被持有
```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let data = Arc::new(Mutex::new(vec![1, 2, 3]));

    let data2 = Arc::clone(&data);
    let handle = thread::spawn(move || {
        let mut guard = data2.lock().unwrap();
        guard.push(4);
        panic!("oops!");  // Lock is now poisoned
    });

    let _ = handle.join();  // Thread panicked

    // Subsequent lock attempts return Err(PoisonError)
    match data.lock() {
        Ok(guard) => println!("Data: {guard:?}"),
        Err(poisoned) => {
            println!("Lock was poisoned! Recovering...");
            let guard = poisoned.into_inner();  // Access data anyway
            println!("Recovered data: {guard:?}");  // [1, 2, 3, 4] — push succeeded before panic
        }
    }
}
```

# Rust 并发：原子操作
- 对于简单的计数器和标志位，`std::sync::atomic` 类型可以避免 `Mutex` 的开销
    - `AtomicBool`、`AtomicI32`、`AtomicU64`、`AtomicUsize` 等
    - 等价于 C++ 的 `std::atomic<T>` —— 相同的内存顺序模型（`Relaxed`、`Acquire`、`Release`、`SeqCst`）
```rust
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::thread;

fn main() {
    let counter = Arc::new(AtomicU64::new(0));
    let mut handles = Vec::new();

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        handles.push(thread::spawn(move || {
            for _ in 0..1000 {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Counter: {}", counter.load(Ordering::SeqCst));
    // Output: Counter: 10000
}
```

| 原语 | 使用场景 | C++ 等价物 |
|-----------|-------------|----------------|
| `Mutex<T>` | 通用可变共享状态 | `std::mutex` + 手动数据关联 |
| `RwLock<T>` | 读密集型工作负载 | `std::shared_mutex` |
| `Atomic*` | 简单计数器、标志位、无锁模式 | `std::atomic<T>` |
| `Condvar` | 等待某个条件变为真 | `std::condition_variable` |

# Rust 并发：Condvar
- `Condvar`（条件变量）让一个线程**休眠直到另一个线程发出信号**表明条件已改变
    - 始终与 `Mutex` 配对使用 —— 模式是：加锁、检查条件、如果未就绪则等待、就绪时执行
    - 等价于 C++ 的 `std::condition_variable` / `std::condition_variable::wait`
    - 处理**虚假唤醒** —— 始终在循环中重新检查条件（或使用 `wait_while`/`wait_until`）
```rust
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

fn main() {
    let pair = Arc::new((Mutex::new(false), Condvar::new()));

    // Spawn a worker that waits for a signal
    let pair2 = Arc::clone(&pair);
    let worker = thread::spawn(move || {
        let (lock, cvar) = &*pair2;
        let mut ready = lock.lock().unwrap();
        // wait: sleeps until signaled (always re-check in a loop for spurious wakeups)
        while !*ready {
            ready = cvar.wait(ready).unwrap();
        }
        println!("Worker: condition met, proceeding!");
    });

    // Main thread does some work, then signals the worker
    thread::sleep(std::time::Duration::from_millis(100));
    {
        let (lock, cvar) = &*pair;
        let mut ready = lock.lock().unwrap();
        *ready = true;
        cvar.notify_one();  // Wake one waiting thread (notify_all() wakes all)
    }

    worker.join().unwrap();
}
```

> **何时使用 Condvar vs 通道：** 当线程共享可变状态并需要等待该状态上的某个条件时（例如"缓冲区非空"），使用 `Condvar`。当线程需要传递*消息*时，使用通道（`mpsc`）。通道通常更容易理解和推理。

# Rust 并发
- Rust 通道可用于在 `Sender` 和 `Receiver` 之间交换消息
    - 使用了一种叫做 `mpsc` 即 `Multi-producer, Single-Consumer`（多生产者、单消费者）的范式
    - `send()` 和 `recv()` 都可能阻塞线程
```rust
use std::sync::mpsc;

fn main() {
    let (tx, rx) = mpsc::channel();
    
    tx.send(10).unwrap();
    tx.send(20).unwrap();
    
    println!("Received: {:?}", rx.recv());
    println!("Received: {:?}", rx.recv());

    let tx2 = tx.clone();
    tx2.send(30).unwrap();
    println!("Received: {:?}", rx.recv());
}
```

# Rust 并发
- 通道可以与线程结合使用
```rust
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn main() {
    let (tx, rx) = mpsc::channel();
    for _ in 0..2 {
        let tx2 = tx.clone();
        thread::spawn(move || {
            let thread_id = thread::current().id();
            for i in 0..10 {
                tx2.send(format!("Message {i}")).unwrap();
                println!("{thread_id:?}: sent Message {i}");
            }
            println!("{thread_id:?}: done");
        });
    }

        // Drop the original sender so rx.iter() terminates when all cloned senders are dropped
    drop(tx);

    thread::sleep(Duration::from_millis(100));

    for msg in rx.iter() {
        println!("Main: got {msg}");
    }
}
```



## 为什么 Rust 能防止数据竞争：Send 和 Sync

- Rust 使用两个标记特征在编译期强制线程安全：
    - `Send`：如果一个类型可以安全地**转移**到另一个线程，它就是 `Send` 的
    - `Sync`：如果一个类型可以安全地通过 `&T` 在线程间**共享**，它就是 `Sync` 的
- 大多数类型自动是 `Send + Sync` 的。值得注意的例外：
    - `Rc<T>` **既不是** Send 也不是 Sync（在多线程环境中使用 `Arc<T>`）
    - `Cell<T>` 和 `RefCell<T>` **不是** Sync（使用 `Mutex<T>` 或 `RwLock<T>`）
    - 裸指针（`*const T`、`*mut T`）**既不是** Send 也不是 Sync
- 这就是为什么编译器会阻止你跨线程使用 `Rc<T>` —— 它根本没有实现 `Send`
- `Arc<Mutex<T>>` 是 `Rc<RefCell<T>>` 的线程安全等价物

> **直觉理解** *(Jon Gjengset)*：把值想象成玩具。
> **`Send`** = 你可以**把玩具送给**另一个小朋友（线程）—— 转移所有权是安全的。
> **`Sync`** = 你可以**让其他人同时玩你的玩具** —— 共享引用是安全的。
> `Rc<T>` 有一个脆弱的（非原子的）引用计数器；把它交出去或共享它会破坏计数，所以它既不是 `Send` 也不是 `Sync`。


# 练习：多线程单词计数

🔴 **挑战** —— 综合使用线程、Arc、Mutex 和 HashMap

- 给定一个 `Vec<String>` 的文本行，为每行启动一个线程来统计该行中的单词数
- 使用 `Arc<Mutex<HashMap<String, usize>>>` 来收集结果
- 打印所有行的总单词数
- **加分项**：尝试用通道（`mpsc`）代替共享状态来实现

<details><summary>解答（点击展开）</summary>

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let lines = vec![
        "the quick brown fox".to_string(),
        "jumps over the lazy dog".to_string(),
        "the fox is quick".to_string(),
    ];

    let word_counts: Arc<Mutex<HashMap<String, usize>>> =
        Arc::new(Mutex::new(HashMap::new()));

    let mut handles = vec![];
    for line in &lines {
        let line = line.clone();
        let counts = Arc::clone(&word_counts);
        handles.push(thread::spawn(move || {
            for word in line.split_whitespace() {
                let mut map = counts.lock().unwrap();
                *map.entry(word.to_lowercase()).or_insert(0) += 1;
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let counts = word_counts.lock().unwrap();
    let total: usize = counts.values().sum();
    println!("Word frequencies: {counts:#?}");
    println!("Total words: {total}");
}
// Output (order may vary):
// Word frequencies: {
//     "the": 3,
//     "quick": 2,
//     "brown": 1,
//     "fox": 2,
//     "jumps": 1,
//     "over": 1,
//     "lazy": 1,
//     "dog": 1,
//     "is": 1,
// }
// Total words: 13
```

</details>


