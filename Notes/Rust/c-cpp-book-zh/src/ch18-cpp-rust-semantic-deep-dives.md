## C++ → Rust 语义深度对比

> **你将学到：** C++ 概念到 Rust 的详细映射，这些概念没有显而易见的 Rust 等价物 —— 四种命名类型转换、SFINAE vs 特征约束、CRTP vs 关联类型，以及翻译过程中的其他常见摩擦点。

以下各节映射了那些没有明显 1:1 Rust 等价物的 C++ 概念。这些差异在翻译工作中经常让 C++ 程序员感到困惑。

### 类型转换层次：C++ 的四种命名转换 → Rust 等价物

C++ 有四种命名类型转换。Rust 用不同的、更明确的机制来替代它们：

```cpp
// C++ 类型转换层次
int i = static_cast<int>(3.14);            // 1. 数值 / 向上转换
Derived* d = dynamic_cast<Derived*>(base); // 2. 运行时向下转换
int* p = const_cast<int*>(cp);              // 3. 去除 const
auto* raw = reinterpret_cast<char*>(&obj); // 4. 位级重新解释
```

| C++ 转换 | Rust 等价物 | 安全性 | 说明 |
|----------|----------------|--------|-------|
| `static_cast`（数值） | `as` 关键字 | 安全但可能截断/回绕 | `let i = 3.14_f64 as i32;` —— 截断为 3 |
| `static_cast`（数值，带检查） | `From`/`Into` | 安全，编译时验证 | `let i: i32 = 42_u8.into();` —— 仅扩展 |
| `static_cast`（数值，可失败） | `TryFrom`/`TryInto` | 安全，返回 `Result` | `let i: u8 = 300_u16.try_into()?;` —— 返回 Err |
| `dynamic_cast`（向下转换） | `match` 枚举 / `Any::downcast_ref` | 安全 | 枚举用模式匹配；特征对象用 `Any` |
| `const_cast` | 无等价物 | | Rust 在安全代码中无法将 `&` 转为 `&mut`。使用 `Cell`/`RefCell` 实现内部可变性 |
| `reinterpret_cast` | `std::mem::transmute` | **`unsafe`** | 重新解释位模式。几乎总是错误的 —— 优先使用 `from_le_bytes()` 等方法 |

```rust
// Rust 等价写法：

// 1. 数值转换 —— 优先使用 From/Into 而非 `as`
let widened: u32 = 42_u8.into();             // 不可失败的扩展 —— 始终优先
let truncated = 300_u16 as u8;                // ⚠ 回绕为 44！静默数据丢失
let checked: Result<u8, _> = 300_u16.try_into(); // Err —— 安全的可失败转换

// 2. 向下转换：枚举（首选）或 Any（需要类型擦除时使用）
use std::any::Any;

fn handle_any(val: &dyn Any) {
    if let Some(s) = val.downcast_ref::<String>() {
        println!("Got string: {s}");
    } else if let Some(n) = val.downcast_ref::<i32>() {
        println!("Got int: {n}");
    }
}

// 3. "const_cast" → 内部可变性（无需 unsafe）
use std::cell::Cell;
struct Sensor {
    read_count: Cell<u32>,  // 通过 &self 进行修改
}
impl Sensor {
    fn read(&self) -> f64 {
        self.read_count.set(self.read_count.get() + 1); // &self，而非 &mut self
        42.0
    }
}

// 4. reinterpret_cast → transmute（几乎永远不需要）
// 优先使用安全的替代方案：
let bytes: [u8; 4] = 0x12345678_u32.to_ne_bytes();  // ✅ 安全
let val = u32::from_ne_bytes(bytes);                   // ✅ 安全
// unsafe { std::mem::transmute::<u32, [u8; 4]>(val) } // ❌ 避免使用
```

> **指导原则**：在惯用 Rust 中，`as` 应该很少使用（扩展用 `From`/`Into`，
> 收窄用 `TryFrom`/`TryInto`），`transmute` 应该是例外情况，
> 而 `const_cast` 没有等价物，因为内部可变性类型使其变得不必要。

---

### 预处理器 → `cfg`、Feature Flag 和 `macro_rules!`

C++ 严重依赖预处理器来进行条件编译、常量定义和代码生成。
Rust 用一等语言特性替代了所有这些功能。

#### `#define` 常量 → `const` 或 `const fn`

```cpp
// C++
#define MAX_RETRIES 5
#define BUFFER_SIZE (1024 * 64)
#define SQUARE(x) ((x) * (x))  // 宏 —— 文本替换，无类型安全
```

```rust
// Rust —— 类型安全、有作用域、无文本替换
const MAX_RETRIES: u32 = 5;
const BUFFER_SIZE: usize = 1024 * 64;
const fn square(x: u32) -> u32 { x * x }  // 编译时求值

// 可在 const 上下文中使用：
const AREA: u32 = square(12);  // 编译时计算
static BUFFER: [u8; BUFFER_SIZE] = [0; BUFFER_SIZE];
```

#### `#ifdef` / `#if` → `#[cfg()]` 和 `cfg!()`

```cpp
// C++
#ifdef DEBUG
    log_verbose("Step 1 complete");
#endif

#if defined(LINUX) && !defined(ARM)
    use_x86_path();
#else
    use_generic_path();
#endif
```

```rust
// Rust —— 基于属性的条件编译
#[cfg(debug_assertions)]
fn log_verbose(msg: &str) { eprintln!("[VERBOSE] {msg}"); }

#[cfg(not(debug_assertions))]
fn log_verbose(_msg: &str) { /* release 模式下被编译移除 */ }

// 组合条件：
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
fn use_x86_path() { /* ... */ }

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
fn use_generic_path() { /* ... */ }

// 运行时检查（条件仍然是编译时的，但可在表达式中使用）：
if cfg!(target_os = "windows") {
    println!("Running on Windows");
}
```

#### `Cargo.toml` 中的 Feature Flag

```toml
# Cargo.toml —— 替代 #ifdef FEATURE_FOO
[features]
default = ["json"]
json = ["dep:serde_json"]       # 可选依赖
verbose-logging = []            # 无额外依赖的标志
gpu-support = ["dep:cuda-sys"]  # 可选 GPU 支持
```

```rust
// 基于 feature flag 的条件代码：
#[cfg(feature = "json")]
pub fn parse_config(data: &str) -> Result<Config, Error> {
    serde_json::from_str(data).map_err(Error::from)
}

#[cfg(feature = "verbose-logging")]
macro_rules! verbose {
    ($($arg:tt)*) => { eprintln!("[VERBOSE] {}", format!($($arg)*)); }
}
#[cfg(not(feature = "verbose-logging"))]
macro_rules! verbose {
    ($($arg:tt)*) => { }; // 编译为空
}
```

#### `#define MACRO(x)` → `macro_rules!`

```cpp
// C++ —— 文本替换，出了名的容易出错
#define DIAG_CHECK(cond, msg) \
    do { if (!(cond)) { log_error(msg); return false; } } while(0)
```

```rust
// Rust —— 卫生的、类型检查的、操作语法树
macro_rules! diag_check {
    ($cond:expr, $msg:expr) => {
        if !($cond) {
            log_error($msg);
            return Err(DiagError::CheckFailed($msg.to_string()));
        }
    };
}

fn run_test() -> Result<(), DiagError> {
    diag_check!(temperature < 85.0, "GPU too hot");
    diag_check!(voltage > 0.8, "Rail voltage too low");
    Ok(())
}
```

| C++ 预处理器 | Rust 等价物 | 优势 |
|-----------------|----------------|-----------|
| `#define PI 3.14` | `const PI: f64 = 3.14;` | 有类型、有作用域、调试器可见 |
| `#define MAX(a,b) ((a)>(b)?(a):(b))` | `macro_rules!` 或泛型 `fn max<T: Ord>` | 无重复求值 bug |
| `#ifdef DEBUG` | `#[cfg(debug_assertions)]` | 编译器检查，无拼写错误风险 |
| `#ifdef FEATURE_X` | `#[cfg(feature = "x")]` | Cargo 管理 feature；感知依赖 |
| `#include "header.h"` | `mod module;` + `use module::Item;` | 无 include 守卫，无循环包含 |
| `#pragma once` | 不需要 | 每个 `.rs` 文件都是模块 —— 只包含一次 |

---

### 头文件和 `#include` → 模块和 `use`

在 C++ 中，编译模型围绕文本包含展开：

```cpp
// widget.h —— 每个使用 Widget 的翻译单元都要包含这个
#pragma once
#include <string>
#include <vector>

class Widget {
public:
    Widget(std::string name);
    void activate();
private:
    std::string name_;
    std::vector<int> data_;
};
```

```cpp
// widget.cpp —— 单独的定义
#include "widget.h"
Widget::Widget(std::string name) : name_(std::move(name)) {}
void Widget::activate() { /* ... */ }
```

在 Rust 中，**没有头文件、没有前向声明、没有 include 守卫**：

```rust
// src/widget.rs —— 声明和定义在同一个文件中
pub struct Widget {
    name: String,         // 默认私有
    data: Vec<i32>,
}

impl Widget {
    pub fn new(name: String) -> Self {
        Widget { name, data: Vec::new() }
    }
    pub fn activate(&self) { /* ... */ }
}
```

```rust
// src/main.rs —— 通过模块路径导入
mod widget;  // 告诉编译器包含 src/widget.rs
use widget::Widget;

fn main() {
    let w = Widget::new("sensor".to_string());
    w.activate();
}
```

| C++ | Rust | 为什么更好 |
|-----|------|-----------------|
| `#include "foo.h"` | 在父模块中 `mod foo;` + `use foo::Item;` | 无文本包含，无 ODR 违规 |
| `#pragma once` / include 守卫 | 不需要 | 每个 `.rs` 文件都是模块 —— 只编译一次 |
| 前向声明 | 不需要 | 编译器看到整个 crate；顺序无关 |
| `class Foo;`（不完整类型） | 不需要 | 无声明/定义分离 |
| 每个类对应 `.h` + `.cpp` | 单个 `.rs` 文件 | 无声明/定义不匹配 bug |
| `using namespace std;` | `use std::collections::HashMap;` | 始终显式 —— 无全局命名空间污染 |
| 嵌套 `namespace a::b` | 嵌套 `mod a { mod b { } }` 或 `a/b.rs` | 文件系统映射模块树 |

---

### `friend` 和访问控制 → 模块可见性

C++ 使用 `friend` 来授予特定类或函数对私有成员的访问权限。
Rust 没有 `friend` 关键字 —— 取而代之的是，**隐私以模块为作用域**：

```cpp
// C++
class Engine {
    friend class Car;   // Car 可以访问私有成员
    int rpm_;
    void set_rpm(int r) { rpm_ = r; }
public:
    int rpm() const { return rpm_; }
}
```

```rust
// Rust —— 同一模块中的项可以访问所有字段，无需 `friend`
mod vehicle {
    pub struct Engine {
        rpm: u32,  // 对模块私有（不是对结构体私有！）
    }

    impl Engine {
        pub fn new() -> Self { Engine { rpm: 0 } }
        pub fn rpm(&self) -> u32 { self.rpm }
    }

    pub struct Car {
        engine: Engine,
    }

    impl Car {
        pub fn new() -> Self { Car { engine: Engine::new() } }
        pub fn accelerate(&mut self) {
            self.engine.rpm = 3000; // ✅ 同一模块 —— 直接访问字段
        }
        pub fn rpm(&self) -> u32 {
            self.engine.rpm  // ✅ 同一模块 —— 可以读取私有字段
        }
    }
}

fn main() {
    let mut car = vehicle::Car::new();
    car.accelerate();
    // car.engine.rpm = 9000;  // ❌ 编译错误：`engine` 是私有的
    println!("RPM: {}", car.rpm()); // ✅ Car 的公开方法
}
```

| C++ 访问控制 | Rust 等价物 | 作用域 |
|-----------|----------------|-------|
| `private` | （默认，无需关键字） | 仅在同一模块内可访问 |
| `protected` | 无直接等价物 | 使用 `pub(super)` 访问父模块 |
| `public` | `pub` | 任何地方都可访问 |
| `friend class Foo` | 将 `Foo` 放在同一模块中 | 模块级隐私替代 friend |
| — | `pub(crate)` | 在 crate 内可见，但对外部依赖者不可见 |
| — | `pub(super)` | 仅对父模块可见 |
| — | `pub(in crate::path)` | 在特定模块子树内可见 |

> **关键洞察**：C++ 的隐私是按类的。Rust 的隐私是按模块的。
> 这意味着你通过选择哪些类型放在同一模块中来控制访问 ——
> 同处一个模块的类型可以完全访问彼此的私有字段。

---

### `volatile` → 原子操作和 `read_volatile`/`write_volatile`

在 C++ 中，`volatile` 告诉编译器不要优化掉读写操作 —— 通常用于内存映射的硬件寄存器。**Rust 没有 `volatile` 关键字。**

```cpp
// C++：用 volatile 访问硬件寄存器
volatile uint32_t* const GPIO_REG = reinterpret_cast<volatile uint32_t*>(0x4002'0000);
*GPIO_REG = 0x01;              // 写操作不会被优化掉
uint32_t val = *GPIO_REG;     // 读操作不会被优化掉
```

```rust
// Rust：显式的 volatile 操作 —— 仅在 unsafe 代码中
use std::ptr;

const GPIO_REG: *mut u32 = 0x4002_0000 as *mut u32;

// SAFETY: GPIO_REG 是一个有效的内存映射 I/O 地址。
unsafe {
    ptr::write_volatile(GPIO_REG, 0x01);   // 写操作不会被优化掉
    let val = ptr::read_volatile(GPIO_REG); // 读操作不会被优化掉
}
```

对于**并发共享状态**（C++ `volatile` 的另一个常见用途），Rust 使用原子操作：

```cpp
// C++：volatile 不足以保证线程安全（常见错误！）
volatile bool stop_flag = false;  // ❌ 数据竞争 —— C++11+ 中的未定义行为

// 正确的 C++ 写法：
std::atomic<bool> stop_flag{false};
```

```rust
// Rust：原子操作是跨线程共享可变状态的唯一方式
use std::sync::atomic::{AtomicBool, Ordering};

static STOP_FLAG: AtomicBool = AtomicBool::new(false);

// 从另一个线程：
STOP_FLAG.store(true, Ordering::Release);

// 检查：
if STOP_FLAG.load(Ordering::Acquire) {
    println!("Stopping");
}
```

| C++ 用法 | Rust 等价物 | 说明 |
|-----------|----------------|-------|
| `volatile` 用于硬件寄存器 | `ptr::read_volatile` / `ptr::write_volatile` | 需要 `unsafe` —— 对 MMIO 是正确的 |
| `volatile` 用于线程信号 | `AtomicBool` / `AtomicU32` 等 | C++ 的 `volatile` 用于此场景也是错误的！ |
| `std::atomic<T>` | `std::sync::atomic::AtomicT` | 相同语义，相同内存序 |
| `std::atomic<T>::load(memory_order_acquire)` | `AtomicT::load(Ordering::Acquire)` | 1:1 映射 |

---

### `static` 变量 → `static`、`const`、`LazyLock`、`OnceLock`

#### 基本的 `static` 和 `const`

```cpp
// C++
const int MAX_RETRIES = 5;                    // 编译时常量
static std::string CONFIG_PATH = "/etc/app";  // 静态初始化 —— 顺序未定义！
```

```rust
// Rust
const MAX_RETRIES: u32 = 5;                   // 编译时常量，会被内联
static CONFIG_PATH: &str = "/etc/app";         // 'static 生命周期，固定地址
```

#### 静态初始化顺序问题

C++ 有一个众所周知的问题：不同翻译单元中的全局构造函数以**未指定的顺序**执行。
Rust 完全避免了这个问题 —— `static` 值必须是编译时常量（没有构造函数）。

对于需要运行时初始化的全局变量，使用 `LazyLock`（Rust 1.80+）或 `OnceLock`：

```rust
use std::sync::LazyLock;

// 等同于 C++ 的 `static std::regex` —— 首次访问时初始化，线程安全
static CONFIG_REGEX: LazyLock<regex::Regex> = LazyLock::new(|| {
    regex::Regex::new(r"^[a-z]+_diag$").expect("invalid regex")
});

fn is_valid_diag(name: &str) -> bool {
    CONFIG_REGEX.is_match(name)  // 首次调用初始化；后续调用很快
}
```

```rust
use std::sync::OnceLock;

// OnceLock：初始化一次，可以从运行时数据设置
static DB_CONN: OnceLock<String> = OnceLock::new();

fn init_db(connection_string: &str) {
    DB_CONN.set(connection_string.to_string())
        .expect("DB_CONN already initialized");
}

fn get_db() -> &'static str {
    DB_CONN.get().expect("DB not initialized")
}
```

| C++ | Rust | 说明 |
|-----|------|-------|
| `const int X = 5;` | `const X: i32 = 5;` | 都是编译时。Rust 需要类型标注 |
| `constexpr int X = 5;` | `const X: i32 = 5;` | Rust 的 `const` 始终是 constexpr |
| `static int count = 0;`（文件作用域） | `static COUNT: AtomicI32 = AtomicI32::new(0);` | 可变 static 需要 `unsafe` 或原子操作 |
| `static std::string s = "hi";` | `static S: &str = "hi";` 或 `LazyLock<String>` | 简单情况无运行时构造函数 |
| `static MyObj obj;`（复杂初始化） | `static OBJ: LazyLock<MyObj> = LazyLock::new(\|\| { ... });` | 线程安全、惰性、无初始化顺序问题 |
| `thread_local` | `thread_local! { static X: Cell<u32> = Cell::new(0); }` | 相同语义 |

---

### `constexpr` → `const fn`

C++ 的 `constexpr` 将函数和变量标记为可在编译时求值。
Rust 使用 `const fn` 和 `const` 实现相同目的：

```cpp
// C++
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}
constexpr int val = factorial(5);  // 编译时计算 → 120
```

```rust
// Rust
const fn factorial(n: u32) -> u32 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}
const VAL: u32 = factorial(5);  // 编译时计算 → 120

// 也可用于数组大小和 match 模式：
const LOOKUP: [u32; 5] = [factorial(1), factorial(2), factorial(3),
                           factorial(4), factorial(5)];
```

| C++ | Rust | 说明 |
|-----|------|-------|
| `constexpr int f()` | `const fn f() -> i32` | 相同意图 —— 可在编译时求值 |
| `constexpr` 变量 | `const` 变量 | Rust 的 `const` 始终是编译时 |
| `consteval`（C++20） | 无等价物 | `const fn` 也可以在运行时执行 |
| `if constexpr`（C++17） | 无等价物（使用 `cfg!` 或泛型） | 特征特化可以填补部分用例 |
| `constinit`（C++20） | 带 const 初始化器的 `static` | Rust 的 `static` 默认必须是 const 初始化的 |

> **`const fn` 的当前限制**（截至 Rust 1.82 已稳定）：
> - 不支持特征方法（不能在 const 上下文中对 `Vec` 调用 `.len()`）
> - 不支持堆分配（`Box::new`、`Vec::new` 不是 const）
> - ~~不支持浮点运算~~ —— **已在 Rust 1.82 中稳定**
> - 不能使用 `for` 循环（使用递归或带手动索引的 `while`）

---

### SFINAE 和 `enable_if` → 特征约束和 `where` 子句

在 C++ 中，SFINAE（替换失败不是错误）是条件泛型编程背后的机制。
它功能强大但出了名的难以阅读。Rust 用**特征约束**完全替代了它：

```cpp
// C++：基于 SFINAE 的条件函数（C++20 之前）
template<typename T,
         std::enable_if_t<std::is_integral_v<T>, int> = 0>
T double_it(T val) { return val * 2; }

template<typename T,
         std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
T double_it(T val) { return val * 2.0; }

// C++20 concepts —— 更清晰但仍然冗长：
template<std::integral T>
T double_it(T val) { return val * 2; }
```

```rust
// Rust：特征约束 —— 可读、可组合、错误信息优秀
use std::ops::Mul;

fn double_it<T: Mul<Output = T> + From<u8>>(val: T) -> T {
    val * T::from(2)
}

// 或用 where 子句处理复杂约束：
fn process<T>(val: T) -> String
where
    T: std::fmt::Display + Clone + Send,
{
    format!("Processing: {}", val)
}

// 通过独立 impl 实现条件行为（替代 SFINAE 重载）：
trait Describable {
    fn describe(&self) -> String;
}

impl Describable for u32 {
    fn describe(&self) -> String { format!("integer: {self}") }
}

impl Describable for f64 {
    fn describe(&self) -> String { format!("float: {self:.2}") }
}
```

| C++ 模板元编程 | Rust 等价物 | 可读性 |
|-----------------------------|----------------|-------------|
| `std::enable_if_t<cond>` | `where T: Trait` | 🟢 清晰的英语 |
| `std::is_integral_v<T>` | 对数值特征或特定类型的约束 | 🟢 无 `_v` / `_t` 后缀 |
| SFINAE 重载集 | 独立的 `impl Trait for ConcreteType` 块 | 🟢 每个 impl 独立存在 |
| `if constexpr (std::is_same_v<T, int>)` | 通过特征 impl 实现特化 | 🟢 编译时分发 |
| C++20 `concept` | `trait` | 🟢 几乎相同的意图 |
| `requires` 子句 | `where` 子句 | 🟢 相同位置，类似语法 |
| 编译失败深藏在模板内部 | 编译失败在调用点，提示特征不匹配 | 🟢 无 200 行错误级联 |

> **关键洞察**：C++20 的 concepts 是最接近 Rust 特征的东西。
> 如果你熟悉 C++20 concepts，可以把 Rust 特征看作是从 1.0 起就是
> 一等语言特性的 concepts，具有一致的实现模型（特征 impl）
> 而非鸭子类型。

---

### `std::function` → 函数指针、`impl Fn` 和 `Box<dyn Fn>`

C++ 的 `std::function<R(Args...)>` 是类型擦除的可调用对象。
Rust 有三种选择，各有不同的权衡：

```cpp
// C++：一刀切（堆分配、类型擦除）
#include <functional>
std::function<int(int)> make_adder(int n) {
    return [n](int x) { return x + n; };
}
```

```rust
// Rust 选项 1：fn 指针 —— 简单、无捕获、无分配
fn add_one(x: i32) -> i32 { x + 1 }
let f: fn(i32) -> i32 = add_one;
println!("{}", f(5)); // 6

// Rust 选项 2：impl Fn —— 单态化、零开销、可捕获
fn apply(val: i32, f: impl Fn(i32) -> i32) -> i32 { f(val) }
let n = 10;
let result = apply(5, |x| x + n);  // 闭包捕获 `n`

// Rust 选项 3：Box<dyn Fn> —— 类型擦除、堆分配（类似 std::function）
fn make_adder(n: i32) -> Box<dyn Fn(i32) -> i32> {
    Box::new(move |x| x + n)
}
let adder = make_adder(10);
println!("{}", adder(5));  // 15

// 存储异构可调用对象（类似 vector<function<int(int)>>）：
let callbacks: Vec<Box<dyn Fn(i32) -> i32>> = vec![
    Box::new(|x| x + 1),
    Box::new(|x| x * 2),
    Box::new(make_adder(100)),
];
for cb in &callbacks {
    println!("{}", cb(5));  // 6, 10, 105
}
```

| 使用场景 | C++ 等价物 | Rust 选择 |
|------------|---------------|-------------|
| 顶层函数，无捕获 | 函数指针 | `fn(Args) -> Ret` |
| 接受可调用对象的泛型函数 | 模板参数 | `impl Fn(Args) -> Ret`（静态分发） |
| 泛型中的特征约束 | `template<typename F>` | `F: Fn(Args) -> Ret` |
| 存储的可调用对象，类型擦除 | `std::function<R(Args)>` | `Box<dyn Fn(Args) -> Ret>` |
| 会修改状态的回调 | 带可变 lambda 的 `std::function` | `Box<dyn FnMut(Args) -> Ret>` |
| 一次性回调（被消耗） | `std::function`（移动） | `Box<dyn FnOnce(Args) -> Ret>` |

> **性能说明**：`impl Fn` 零开销（单态化，类似 C++ 模板）。
> `Box<dyn Fn>` 与 `std::function` 开销相同（虚表 + 堆分配）。
> 除非需要存储异构可调用对象，否则优先使用 `impl Fn`。

---

### 容器映射：C++ STL → Rust `std::collections`

| C++ STL 容器 | Rust 等价物 | 说明 |
|------------------|----------------|-------|
| `std::vector<T>` | `Vec<T>` | API 几乎相同。Rust 默认检查边界 |
| `std::array<T, N>` | `[T; N]` | 栈分配的固定大小数组 |
| `std::deque<T>` | `std::collections::VecDeque<T>` | 环形缓冲区。两端高效 push/pop |
| `std::list<T>` | `std::collections::LinkedList<T>` | 在 Rust 中很少使用 —— `Vec` 几乎总是更快 |
| `std::forward_list<T>` | 无等价物 | 使用 `Vec` 或 `VecDeque` |
| `std::unordered_map<K, V>` | `std::collections::HashMap<K, V>` | 默认使用 `SipHash`（抗 DoS） |
| `std::map<K, V>` | `std::collections::BTreeMap<K, V>` | B 树；键有序；需要 `K: Ord` |
| `std::unordered_set<T>` | `std::collections::HashSet<T>` | 需要 `T: Hash + Eq` |
| `std::set<T>` | `std::collections::BTreeSet<T>` | 有序集合；需要 `T: Ord` |
| `std::priority_queue<T>` | `std::collections::BinaryHeap<T>` | 默认最大堆（与 C++ 相同） |
| `std::stack<T>` | `Vec<T>` 配合 `.push()` / `.pop()` | 无需单独的栈类型 |
| `std::queue<T>` | `VecDeque<T>` 配合 `.push_back()` / `.pop_front()` | 无需单独的队列类型 |
| `std::string` | `String` | 保证 UTF-8，非空终止 |
| `std::string_view` | `&str` | 借用的 UTF-8 切片 |
| `std::span<T>`（C++20） | `&[T]` / `&mut [T]` | Rust 切片从 1.0 起就是一等类型 |
| `std::tuple<A, B, C>` | `(A, B, C)` | 一等语法，可解构 |
| `std::pair<A, B>` | `(A, B)` | 就是 2 元素元组 |
| `std::bitset<N>` | 标准库无等价物 | 使用 `bitvec` crate 或 `[u8; N/8]` |

**关键区别**：
- Rust 的 `HashMap`/`HashSet` 要求 `K: Hash + Eq` —— 编译器在类型层面强制执行，不像 C++ 中使用不可哈希的键会在 STL 深处给出模板错误
- `Vec` 索引（`v[i]`）默认在越界时 panic。使用 `.get(i)` 获取 `Option<&T>`，或使用迭代器完全避免边界检查
- 没有 `std::multimap` 或 `std::multiset` —— 使用 `HashMap<K, Vec<V>>` 或 `BTreeMap<K, Vec<V>>`

---

### 异常安全 → Panic 安全

C++ 定义了三个级别的异常安全（Abrahams 保证）：

| C++ 级别 | 含义 | Rust 等价物 |
|----------|---------|----------------|
| **不抛出** | 函数永不抛出异常 | 函数永不 panic（返回 `Result`） |
| **强保证**（提交或回滚） | 如果抛出异常，状态不变 | 所有权模型使这变得自然 —— 如果 `?` 提前返回，部分构建的值会被丢弃 |
| **基本保证** | 如果抛出异常，不变量得到保持 | Rust 的默认行为 —— `Drop` 运行，无泄漏 |

#### Rust 的所有权模型如何帮助

```rust
// 免费获得强保证 —— 如果 file.write() 失败，config 不变
fn update_config(config: &mut Config, path: &str) -> Result<(), Error> {
    let new_data = fetch_from_network()?; // Err → 提前返回，config 不变
    let validated = validate(new_data)?;   // Err → 提前返回，config 不变
    *config = validated;                   // 仅在成功时到达（提交）
    Ok(())
}
```

在 C++ 中，实现强保证需要手动回滚或复制并交换惯用法。在 Rust 中，`?` 传播默认为大多数代码提供了强保证。

#### `catch_unwind` —— Rust 的 `catch(...)` 等价物

```rust
use std::panic;

// 捕获 panic（类似 C++ 的 catch(...)）—— 很少需要
let result = panic::catch_unwind(|| {
    // 可能 panic 的代码
    let v = vec![1, 2, 3];
    v[10]  // Panic！（索引越界）
});

match result {
    Ok(val) => println!("Got: {val}"),
    Err(_) => eprintln!("Caught a panic — cleaned up"),
}
```

#### `UnwindSafe` —— 标记类型为 panic 安全

```rust
use std::panic::UnwindSafe;

// &mut 背后的类型默认不是 UnwindSafe 的 —— panic 可能使它们
// 处于部分修改的状态
fn safe_execute<F: FnOnce() + UnwindSafe>(f: F) {
    let _ = std::panic::catch_unwind(f);
}

// 当你已审计代码时，使用 AssertUnwindSafe 来覆盖：
use std::panic::AssertUnwindSafe;
let mut data = vec![1, 2, 3];
let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
    data.push(4);
}));
```

| C++ 异常模式 | Rust 等价物 |
|-----------------------|-----------------|
| `throw MyException()` | `return Err(MyError::...)`（首选）或 `panic!("...")` |
| `try { } catch (const E& e)` | `match result { Ok(v) => ..., Err(e) => ... }` 或 `?` |
| `catch (...)` | `std::panic::catch_unwind(...)` |
| `noexcept` | `-> Result<T, E>`（错误是值，不是异常） |
| 栈展开中的 RAII 清理 | `Drop::drop()` 在 panic 展开期间运行 |
| `std::uncaught_exceptions()` | `std::thread::panicking()` |
| `-fno-exceptions` 编译标志 | `Cargo.toml` [profile] 中的 `panic = "abort"` |

> **总结**：在 Rust 中，大多数代码使用 `Result<T, E>` 而非异常，
> 使错误路径显式且可组合。`panic!` 保留给 bug
>（如 `assert!` 失败），而非常规错误。这意味着"异常安全"
> 在很大程度上不是问题 —— 所有权系统自动处理清理工作。

---

## C++ 到 Rust 迁移模式

### 快速参考：C++ → Rust 惯用法映射

| **C++ 模式** | **Rust 惯用法** | **说明** |
|----------------|---------------|----------|
| `class Derived : public Base` | `enum Variant { A {...}, B {...} }` | 封闭集合优先使用枚举 |
| `virtual void method() = 0` | `trait MyTrait { fn method(&self); }` | 用于开放/可扩展接口 |
| `dynamic_cast<Derived*>(ptr)` | `match value { Variant::A(data) => ..., }` | 穷举匹配，无运行时失败 |
| `vector<unique_ptr<Base>>` | `Vec<Box<dyn Trait>>` | 仅在真正需要多态时使用 |
| `shared_ptr<T>` | `Rc<T>` 或 `Arc<T>` | 优先使用 `Box<T>` 或拥有的值 |
| `enable_shared_from_this<T>` | Arena 模式（`Vec<T>` + 索引） | 完全消除引用循环 |
| 每个类中的 `Base* m_pFramework` | `fn execute(&mut self, ctx: &mut Context)` | 传递上下文，不要存储指针 |
| `try { } catch (...) { }` | `match result { Ok(v) => ..., Err(e) => ... }` | 或使用 `?` 传播 |
| `std::optional<T>` | `Option<T>` | 必须 `match`，不会遗忘 None |
| `const std::string&` 参数 | `&str` 参数 | 同时接受 `String` 和 `&str` |
| `enum class Foo { A, B, C }` | `enum Foo { A, B, C }` | Rust 枚举还可以携带数据 |
| `auto x = std::move(obj)` | `let x = obj;` | 移动是默认行为，无需 `std::move` |
| CMake + make + lint | `cargo build / test / clippy / fmt` | 一个工具搞定一切 |

### 迁移策略
1. **从数据类型开始**：先翻译结构体和枚举 —— 这会迫使你思考所有权
2. **将工厂转换为枚举**：如果工厂创建不同的派生类型，它应该变成 `enum` + `match`
3. **将上帝对象转换为组合结构体**：将相关字段分组到专注的结构体中
4. **用借用替换指针**：将存储的 `Base*` 指针转换为带生命周期约束的 `&'a T` 借用
5. **谨慎使用 `Box<dyn Trait>`**：仅用于插件系统和测试模拟
6. **让编译器引导你**：Rust 的错误信息非常优秀 —— 仔细阅读它们

