## 避免过度使用 clone()

> **你将学到：** 为什么 `.clone()` 在 Rust 中是一种代码异味，如何重构所有权来消除不必要的复制，以及哪些特定模式表明存在所有权设计问题。

- 来自 C++ 的开发者会觉得 `.clone()` 是一个安全的默认选择 —— "复制一份就好了"。但过度克隆会隐藏所有权问题并损害性能。
- **经验法则**：如果你是为了满足借用检查器而克隆，你可能需要重构所有权。

### 什么时候 clone() 是错误的

```rust
// BAD: Cloning a String just to pass it to a function that only reads it
fn log_message(msg: String) {  // Takes ownership unnecessarily
    println!("[LOG] {}", msg);
}
let message = String::from("GPU test passed");
log_message(message.clone());  // Wasteful: allocates a whole new String
log_message(message);           // Original consumed — clone was pointless
```

```rust
// GOOD: Accept a borrow — zero allocation
fn log_message(msg: &str) {    // Borrows, doesn't own
    println!("[LOG] {}", msg);
}
let message = String::from("GPU test passed");
log_message(&message);          // No clone, no allocation
log_message(&message);          // Can call again — message not consumed
```

### 真实示例：返回 `&str` 而不是克隆
```rust
// Example: healthcheck.rs — returns a borrowed view, zero allocation
pub fn serial_or_unknown(&self) -> &str {
    self.serial.as_deref().unwrap_or(UNKNOWN_VALUE)
}

pub fn model_or_unknown(&self) -> &str {
    self.model.as_deref().unwrap_or(UNKNOWN_VALUE)
}
```
C++ 等价写法会返回 `const std::string&` 或 `std::string_view` —— 但在 C++ 中两者都没有生命周期检查。在 Rust 中，借用检查器保证返回的 `&str` 不会比 `self` 活得更久。

### 真实示例：静态字符串切片 —— 完全不使用堆
```rust
// Example: healthcheck.rs — compile-time string tables
const HBM_SCREEN_RECIPES: &[&str] = &[
    "hbm_ds_ntd", "hbm_ds_ntd_gfx", "hbm_dt_ntd", "hbm_dt_ntd_gfx",
    "hbm_burnin_8h", "hbm_burnin_24h",
];
```
在 C++ 中这通常是 `std::vector<std::string>`（首次使用时在堆上分配）。Rust 的 `&'static [&'static str]` 存在于只读内存中 —— 零运行时开销。

### 什么时候 clone() 是合适的

| **场景** | **为什么 clone 是 OK 的** | **示例** |
|--------------|--------------------|-----------|
| 用于线程的 `Arc::clone()` | 只增加引用计数（约 1 ns），不复制数据 | `let flag = stop_flag.clone();` |
| 将数据移入新生成的线程 | 线程需要自己的副本 | `let ctx = ctx.clone(); thread::spawn(move \|\| { ... })` |
| 从 `&self` 字段中提取 | 不能从借用中移出 | 返回拥有的 `String` 时用 `self.name.clone()` |
| 包装在 `Option` 中的小型 `Copy` 类型 | `.copied()` 比 `.clone()` 更清晰 | `opt.get(0).copied()` 将 `Option<&u32>` 转为 `Option<u32>` |

### 真实示例：用于线程共享的 Arc::clone
```rust
// Example: workload.rs — Arc::clone is cheap (ref count bump)
let stop_flag = Arc::new(AtomicBool::new(false));
let stop_flag_clone = stop_flag.clone();   // ~1 ns, no data copied
let ctx_clone = ctx.clone();               // Clone context for move into thread

let sensor_handle = thread::spawn(move || {
    // ...uses stop_flag_clone and ctx_clone
});
```

### 检查清单：我应该克隆吗？
1. **我能接受 `&str` / `&T` 而不是 `String` / `T` 吗？** → 借用，不要克隆
2. **我能重构以避免需要两个所有者吗？** → 通过引用传递或使用作用域
3. **这是 `Arc::clone()` 吗？** → 没问题，它是 O(1) 的
4. **我是要将数据移入线程/闭包吗？** → 克隆是必要的
5. **我在热循环中克隆吗？** → 做性能分析，考虑借用或 `Cow<T>`

----

## `Cow<'a, T>`：写时克隆 —— 能借用时借用，必须时克隆

`Cow`（Clone on Write）是一个枚举，持有**借用引用**或**拥有所有权的值**。它是 Rust 中"尽可能避免分配，但在需要修改时分配"的等价物。C++ 没有直接的等价物 —— 最接近的是一个有时返回 `const std::string&` 有时返回 `std::string` 的函数。

### 为什么 `Cow` 存在

```rust
// Without Cow — you must choose: always borrow OR always clone
fn normalize(s: &str) -> String {          // Always allocates!
    if s.contains(' ') {
        s.replace(' ', "_")               // New String (allocation needed)
    } else {
        s.to_string()                     // Unnecessary allocation!
    }
}

// With Cow — borrow when unchanged, allocate only when modified
use std::borrow::Cow;

fn normalize(s: &str) -> Cow<'_, str> {
    if s.contains(' ') {
        Cow::Owned(s.replace(' ', "_"))    // Allocates (must modify)
    } else {
        Cow::Borrowed(s)                   // Zero allocation (passthrough)
    }
}
```

### `Cow` 的工作原理

```rust
use std::borrow::Cow;

// Cow<'a, str> is essentially:
// enum Cow<'a, str> {
//     Borrowed(&'a str),     // Zero-cost reference
//     Owned(String),          // Heap-allocated owned value
// }

fn greet(name: &str) -> Cow<'_, str> {
    if name.is_empty() {
        Cow::Borrowed("stranger")         // Static string — no allocation
    } else if name.starts_with(' ') {
        Cow::Owned(name.trim().to_string()) // Modified — allocation needed
    } else {
        Cow::Borrowed(name)               // Passthrough — no allocation
    }
}

fn main() {
    let g1 = greet("Alice");     // Cow::Borrowed("Alice")
    let g2 = greet("");          // Cow::Borrowed("stranger")
    let g3 = greet(" Bob ");     // Cow::Owned("Bob")
    
    // Cow<str> implements Deref<Target = str>, so you can use it as &str:
    println!("Hello, {g1}!");    // Works — Cow auto-derefs to &str
    println!("Hello, {g2}!");
    println!("Hello, {g3}!");
}
```

### 真实用例：配置值规范化

```rust
use std::borrow::Cow;

/// Normalize a SKU name: trim whitespace, lowercase.
/// Returns Cow::Borrowed if already normalized (zero allocation).
fn normalize_sku(sku: &str) -> Cow<'_, str> {
    let trimmed = sku.trim();
    if trimmed == sku && sku.chars().all(|c| c.is_lowercase() || !c.is_alphabetic()) {
        Cow::Borrowed(sku)   // Already normalized — no allocation
    } else {
        Cow::Owned(trimmed.to_lowercase())  // Needs modification — allocate
    }
}

fn main() {
    let s1 = normalize_sku("server-x1");   // Borrowed — zero alloc
    let s2 = normalize_sku("  Server-X1 "); // Owned — must allocate
    println!("{s1}, {s2}"); // "server-x1, server-x1"
}
```

### 何时使用 `Cow`

| **场景** | **使用 `Cow`？** |
|--------------|---------------|
| 函数大多数时候原样返回输入 | ✅ 是 —— 避免不必要的克隆 |
| 解析/规范化字符串（trim、小写、替换） | ✅ 是 —— 输入通常已经是有效的 |
| 总是修改 —— 每个代码路径都分配 | ❌ 否 —— 直接返回 `String` |
| 简单透传（从不修改） | ❌ 否 —— 直接返回 `&str` |
| 数据长期存储在结构体中 | ❌ 否 —— 使用 `String`（拥有所有权） |

> **C++ 对比**：`Cow<str>` 类似于一个返回 `std::variant<std::string_view, std::string>` 的函数 —— 但有自动解引用且无需样板代码来访问值。

----

## `Weak<T>`：打破引用循环 —— Rust 的 `weak_ptr`

`Weak<T>` 是 C++ `std::weak_ptr<T>` 的 Rust 等价物。它持有对 `Rc<T>` 或 `Arc<T>` 值的非拥有引用。值可以在 `Weak` 引用仍然存在时被释放 —— 调用 `upgrade()` 在值已消失时返回 `None`。

### 为什么 `Weak` 存在

`Rc<T>` 和 `Arc<T>` 在两个值互相指向时会创建引用循环 —— 两者的引用计数都永远不会降为 0，所以都不会被释放（内存泄漏）。`Weak` 打破了循环：

```rust
use std::rc::{Rc, Weak};
use std::cell::RefCell;

#[derive(Debug)]
struct Node {
    value: String,
    parent: RefCell<Weak<Node>>,      // Weak — doesn't prevent parent from dropping
    children: RefCell<Vec<Rc<Node>>>,  // Strong — parent owns children
}

impl Node {
    fn new(value: &str) -> Rc<Node> {
        Rc::new(Node {
            value: value.to_string(),
            parent: RefCell::new(Weak::new()),
            children: RefCell::new(Vec::new()),
        })
    }

    fn add_child(parent: &Rc<Node>, child: &Rc<Node>) {
        // Child gets a weak reference to parent (no cycle)
        *child.parent.borrow_mut() = Rc::downgrade(parent);
        // Parent gets a strong reference to child
        parent.children.borrow_mut().push(Rc::clone(child));
    }
}

fn main() {
    let root = Node::new("root");
    let child = Node::new("child");
    Node::add_child(&root, &child);

    // Access parent from child via upgrade()
    if let Some(parent) = child.parent.borrow().upgrade() {
        println!("Child's parent: {}", parent.value); // "root"
    }
    
    println!("Root strong count: {}", Rc::strong_count(&root));  // 1
    println!("Root weak count: {}", Rc::weak_count(&root));      // 1
}
```

### C++ 对比

```cpp
// C++ — weak_ptr to break shared_ptr cycle
struct Node {
    std::string value;
    std::weak_ptr<Node> parent;                  // Weak — no ownership
    std::vector<std::shared_ptr<Node>> children;  // Strong — owns children

    static auto create(const std::string& v) {
        return std::make_shared<Node>(Node{v, {}, {}});
    }
};

auto root = Node::create("root");
auto child = Node::create("child");
child->parent = root;          // weak_ptr assignment
root->children.push_back(child);

if (auto p = child->parent.lock()) {   // lock() → shared_ptr or null
    std::cout << "Parent: " << p->value << std::endl;
}
```

| C++ | Rust | 备注 |
|-----|------|-------|
| `shared_ptr<T>` | `Rc<T>`（单线程）/ `Arc<T>`（多线程） | 相同语义 |
| `weak_ptr<T>` | 来自 `Rc::downgrade()` / `Arc::downgrade()` 的 `Weak<T>` | 相同语义 |
| `weak_ptr::lock()` → `shared_ptr` 或 null | `Weak::upgrade()` → `Option<Rc<T>>` | 如果已释放则为 `None` |
| `shared_ptr::use_count()` | `Rc::strong_count()` | 相同含义 |

### 何时使用 `Weak`

| **场景** | **模式** |
|--------------|-----------|
| 父子树关系 | 父节点持有 `Rc<Child>`，子节点持有 `Weak<Parent>` |
| 观察者模式 / 事件监听器 | 事件源持有 `Weak<Observer>`，观察者持有 `Rc<Source>` |
| 不阻止释放的缓存 | `HashMap<Key, Weak<Value>>` —— 条目自然过期 |
| 打破图结构中的循环 | 交叉链接使用 `Weak`，树边使用 `Rc`/`Arc` |

> **优先使用 arena 模式**（案例研究 2）而不是 `Rc/Weak` 来处理新代码中的树结构。`Vec<T>` + 索引更简单、更快，且零引用计数开销。当你需要具有动态生命周期的共享所有权时，使用 `Rc/Weak`。

----

## Copy vs Clone，PartialEq vs Eq —— 何时派生什么

- **Copy 类似于 C++ 的平凡可复制类型（没有自定义拷贝构造函数/析构函数）。** 像 `int`、`enum` 和简单的 POD 结构体这样的类型 —— 编译器自动生成按位 `memcpy`。在 Rust 中，`Copy` 是同样的概念：赋值 `let b = a;` 做隐式的按位复制，两个变量都保持有效。
- **Clone 类似于 C++ 的拷贝构造函数 / `operator=` 深拷贝。** 当 C++ 类有自定义拷贝构造函数时（例如深拷贝 `std::vector` 成员），Rust 中的等价物是实现 `Clone`。你必须显式调用 `.clone()` —— Rust 永远不会在 `=` 背后隐藏昂贵的复制。
- **关键区别：** 在 C++ 中，平凡复制和深拷贝都通过相同的 `=` 语法隐式发生。Rust 强制你做出选择：`Copy` 类型静默复制（廉价），非 `Copy` 类型默认**移动**，你必须通过 `.clone()` 显式选择昂贵的复制。
- 类似地，C++ 的 `operator==` 不区分 `a == a` 总是成立的类型（如整数）和不总是成立的类型（如带 NaN 的 `float`）。Rust 通过 `PartialEq` vs `Eq` 来编码这一点。

### Copy vs Clone

| | **Copy** | **Clone** |
|---|---------|----------|
| **工作方式** | 按位 memcpy（隐式） | 自定义逻辑（显式 `.clone()`） |
| **何时发生** | 赋值时：`let b = a;` | 只在调用 `.clone()` 时 |
| **复制/克隆后** | `a` 和 `b` 都有效 | `a` 和 `b` 都有效 |
| **没有两者时** | `let b = a;` **移动** `a`（a 不再可用） | `let b = a;` **移动** `a`（a 不再可用） |
| **允许用于** | 没有堆数据的类型 | 任何类型 |
| **C++ 类比** | 平凡可复制 / POD 类型（无自定义拷贝构造函数） | 自定义拷贝构造函数（深拷贝） |

### 真实示例：Copy —— 简单枚举
```rust
// From fan_diag/src/sensor.rs — all unit variants, fits in 1 byte
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FanStatus {
    #[default]
    Normal,
    Low,
    High,
    Missing,
    Failed,
    Unknown,
}

let status = FanStatus::Normal;
let copy = status;   // Implicit copy — status is still valid
println!("{:?} {:?}", status, copy);  // Both work
```

### 真实示例：Copy —— 带整数负载的枚举
```rust
// Example: healthcheck.rs — u32 payloads are Copy, so the whole enum is too
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthcheckStatus {
    Pass,
    ProgramError(u32),
    DmesgError(u32),
    RasError(u32),
    OtherError(u32),
    Unknown,
}
```

### 真实示例：仅 Clone —— 包含堆数据的结构体
```rust
// Example: components.rs — String prevents Copy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FruData {
    pub technology: DeviceTechnology,
    pub physical_location: String,      // ← String: heap-allocated, can't Copy
    pub expected: bool,
    pub removable: bool,
}
// let a = fru_data;   → MOVES (a is gone)
// let a = fru_data.clone();  → CLONES (fru_data still valid, new heap allocation)
```

### 规则：它能是 Copy 吗？
```text
Does the type contain String, Vec, Box, HashMap,
Rc, Arc, or any other heap-owning type?
    YES → Clone only (cannot be Copy)
    NO  → You CAN derive Copy (and should, if the type is small)
```

### PartialEq vs Eq

| | **PartialEq** | **Eq** |
|---|--------------|-------|
| **给你什么** | `==` 和 `!=` 操作符 | 标记："等价性是自反的" |
| **自反？(a == a)** | 不保证 | **保证** |
| **为什么重要** | `f32::NAN != f32::NAN` | `HashMap` 键**要求** `Eq` |
| **何时派生** | 几乎总是 | 当类型没有 `f32`/`f64` 字段时 |
| **C++ 类比** | `operator==` | 没有直接等价物（C++ 不检查） |

### 真实示例：Eq —— 用作 HashMap 键
```rust
// From hms_trap/src/cpu_handler.rs — Hash requires Eq
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpuFaultType {
    InvalidFaultType,
    CpuCperFatalErr,
    CpuLpddr5UceErr,
    CpuC2CUceFatalErr,
    // ...
}
// Used as: HashMap<CpuFaultType, FaultHandler>
// HashMap keys must be Eq + Hash — PartialEq alone won't compile
```

### 真实示例：不能使用 Eq —— 类型包含 f32
```rust
// Example: types.rs — f32 prevents Eq
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemperatureSensors {
    pub warning_threshold: Option<f32>,   // ← f32 has NaN ≠ NaN
    pub critical_threshold: Option<f32>,  // ← can't derive Eq
    pub sensor_names: Vec<String>,
}
// Cannot be used as HashMap key. Cannot derive Eq.
// Because: f32::NAN == f32::NAN is false, violating reflexivity.
```

### PartialOrd vs Ord

| | **PartialOrd** | **Ord** |
|---|---------------|--------|
| **给你什么** | `<`、`>`、`<=`、`>=` | `.sort()`、`BTreeMap` 键 |
| **全序？** | 否（某些对可能不可比较） | **是**（每对都可比较） |
| **f32/f64？** | 仅 PartialOrd（NaN 破坏排序） | 不能派生 Ord |

### 真实示例：Ord —— 严重性排序
```rust
// From hms_trap/src/fault.rs — variant order defines severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FaultSeverity {
    Info,      // lowest  (discriminant 0)
    Warning,   //         (discriminant 1)
    Error,     //         (discriminant 2)
    Critical,  // highest (discriminant 3)
}
// FaultSeverity::Info < FaultSeverity::Critical → true
// Enables: if severity >= FaultSeverity::Error { escalate(); }
```

### 真实示例：Ord —— 用于比较的诊断级别
```rust
// Example: orchestration.rs
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum GpuDiagLevel {
    #[default]
    Quick,     // lowest
    Standard,
    Extended,
    Full,      // highest
}
// Enables: if requested_level >= GpuDiagLevel::Extended { run_extended_tests(); }
```

### 派生决策树

```text
                        Your new type
                            │
                   Contains String/Vec/Box?
                      /              \
                    YES                NO
                     │                  │
              Clone only          Clone + Copy
                     │                  │
              Contains f32/f64?    Contains f32/f64?
                /          \         /          \
              YES           NO     YES           NO
               │             │      │             │
         PartialEq       PartialEq  PartialEq  PartialEq
         only            + Eq       only       + Eq
                          │                      │
                    Need sorting?           Need sorting?
                      /       \               /       \
                    YES        NO            YES        NO
                     │          │              │          │
               PartialOrd    Done        PartialOrd    Done
               + Ord                     + Ord
                     │                        │
               Need as                  Need as
               map key?                 map key?
                  │                        │
                + Hash                   + Hash
```

### 快速参考：生产 Rust 代码中常见的派生组合

| **类型类别** | **典型派生** | **示例** |
|-------------------|--------------------|------------|
| 简单状态枚举 | `Copy, Clone, PartialEq, Eq, Default` | `FanStatus` |
| 用作 HashMap 键的枚举 | `Copy, Clone, PartialEq, Eq, Hash` | `CpuFaultType`、`SelComponent` |
| 可排序的严重性枚举 | `Copy, Clone, PartialEq, Eq, PartialOrd, Ord` | `FaultSeverity`、`GpuDiagLevel` |
| 包含 String 的数据结构体 | `Clone, Debug, Serialize, Deserialize` | `FruData`、`OverallSummary` |
| 可序列化的配置 | `Clone, Debug, Default, Serialize, Deserialize` | `DiagConfig` |

----

