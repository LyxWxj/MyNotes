## Rust 宏：从预处理器到元编程

> **你将学到：** Rust 宏的工作原理，何时使用宏而非函数或泛型，以及宏如何替代 C/C++ 预处理器。学完本章后，你将能够编写自己的 `macro_rules!` 宏，并理解 `#[derive(Debug)]` 底层做了什么。

宏是你在 Rust 中最早接触到的东西之一（第一行的 `println!("hello")`），却是大多数课程最后才讲解的内容。本章解决这个问题。

### 宏为什么存在

函数和泛型处理了 Rust 中大部分的代码复用。宏填补了类型系统无法触及的空白：

| 需求 | 函数/泛型？ | 宏？ | 原因 |
|------|-------------------|--------|-----|
| 计算一个值 | ✅ `fn max<T: Ord>(a: T, b: T) -> T` | — | 类型系统可以处理 |
| 接受可变数量的参数 | ❌ Rust 没有可变参数函数 | ✅ `println!("{} {}", a, b)` | 宏接受任意数量的 token |
| 为多个类型生成重复的 `impl` 块 | ❌ 仅用泛型无法实现 | ✅ `macro_rules!` | 宏在编译时生成代码 |
| 在编译时运行代码 | ❌ `const fn` 有限制 | ✅ 过程宏 | 完整的 Rust 代码在编译时运行 |
| 条件性地包含代码 | ❌ | ✅ `#[cfg(...)]` | 属性宏控制编译 |

如果你来自 C/C++，可以把宏看作是*预处理器唯一正确的替代品* —— 只不过它们操作的是语法树而非原始文本，所以它们是卫生的（不会有意外的名称冲突）且感知类型。

> **给 C 开发者：** Rust 宏完全替代了 `#define`。没有文本预处理器。完整的预处理器 → Rust 映射请参阅 [ch18](ch18-cpp-rust-semantic-deep-dives.md)。

---

## 使用 `macro_rules!` 的声明式宏

声明式宏（也称"按示例宏"）是 Rust 中最常见的宏形式。它们对语法进行模式匹配，类似于对值使用 `match`。

### 基本语法

```rust
macro_rules! say_hello {
    () => {
        println!("Hello!");
    };
}

fn main() {
    say_hello!();  // 展开为：println!("Hello!");
}
```

名称后的 `!` 告诉你（和编译器）这是一个宏调用。

### 带参数的模式匹配

宏使用片段说明符对*token 树*进行匹配：

```rust
macro_rules! greet {
    // 模式 1：无参数
    () => {
        println!("Hello, world!");
    };
    // 模式 2：一个表达式参数
    ($name:expr) => {
        println!("Hello, {}!", $name);
    };
}

fn main() {
    greet!();           // "Hello, world!"
    greet!("Rust");     // "Hello, Rust!"
}
```

#### 片段说明符参考

| 说明符 | 匹配内容 | 示例 |
|-----------|---------|---------|
| `$x:expr` | 任意表达式 | `42`、`a + b`、`foo()` |
| `$x:ty` | 一个类型 | `i32`、`Vec<String>`、`&str` |
| `$x:ident` | 一个标识符 | `foo`、`my_var` |
| `$x:pat` | 一个模式 | `Some(x)`、`_`、`(a, b)` |
| `$x:stmt` | 一条语句 | `let x = 5;` |
| `$x:block` | 一个块 | `{ println!("hi"); 42 }` |
| `$x:literal` | 一个字面量 | `42`、`"hello"`、`true` |
| `$x:tt` | 一个 token 树 | 任何东西 —— 通配符 |
| `$x:item` | 一个项（fn、struct、impl 等） | `fn foo() {}` |

### 重复 —— 杀手级特性

C/C++ 宏不能循环。Rust 宏可以重复模式：

```rust
macro_rules! make_vec {
    // 匹配零个或多个逗号分隔的表达式
    ( $( $element:expr ),* ) => {
        {
            let mut v = Vec::new();
            $( v.push($element); )*  // 对每个匹配的元素重复
            v
        }
    };
}

fn main() {
    let v = make_vec![1, 2, 3, 4, 5];
    println!("{v:?}");  // [1, 2, 3, 4, 5]
}
```

`$( ... ),*` 语法表示"匹配零个或多个此模式，以逗号分隔"。展开中的 `$( ... )*` 对每个匹配重复一次主体。

> **这正是标准库中 `vec![]` 的实现方式。** 实际源码是：
> ```rust
> macro_rules! vec {
>     () => { Vec::new() };
>     ($elem:expr; $n:expr) => { vec::from_elem($elem, $n) };
>     ($($x:expr),+ $(,)?) => { <[_]>::into_vec(Box::new([$($x),+])) };
> }
> ```
> 末尾的 `$(,)?` 允许可选的尾随逗号。

#### 重复运算符

| 运算符 | 含义 | 示例 |
|----------|---------|---------|
| `$( ... )*` | 零个或多个 | `vec![]`、`vec![1]`、`vec![1, 2, 3]` |
| `$( ... )+` | 一个或多个 | 至少需要一个元素 |
| `$( ... )?` | 零个或一个 | 可选元素 |

### 实际示例：`hashmap!` 构造器

标准库有 `vec![]` 但没有 `hashmap!{}`。让我们来构建一个：

```rust
macro_rules! hashmap {
    ( $( $key:expr => $value:expr ),* $(,)? ) => {
        {
            let mut map = std::collections::HashMap::new();
            $( map.insert($key, $value); )*
            map
        }
    };
}

fn main() {
    let scores = hashmap! {
        "Alice" => 95,
        "Bob" => 87,
        "Carol" => 92,  // 尾随逗号没问题，感谢 $(,)?
    };
    println!("{scores:?}");
}
```

### 实际示例：诊断检查宏

嵌入式/诊断代码中常见的模式 —— 检查条件并返回错误：

```rust
use thiserror::Error;

#[derive(Error, Debug)]
enum DiagError {
    #[error("Check failed: {0}")]
    CheckFailed(String),
}

macro_rules! diag_check {
    ($cond:expr, $msg:expr) => {
        if !($cond) {
            return Err(DiagError::CheckFailed($msg.to_string()));
        }
    };
}

fn run_diagnostics(temp: f64, voltage: f64) -> Result<(), DiagError> {
    diag_check!(temp < 85.0, "GPU too hot");
    diag_check!(voltage > 0.8, "Rail voltage too low");
    diag_check!(voltage < 1.5, "Rail voltage too high");
    println!("All checks passed");
    Ok(())
}
```

> **C/C++ 对比：**
> ```c
> // C 预处理器 —— 文本替换，无类型安全，无卫生性
> #define DIAG_CHECK(cond, msg) \
>     do { if (!(cond)) { log_error(msg); return -1; } } while(0)
> ```
> Rust 版本返回正确的 `Result` 类型，没有重复求值风险，且编译器会检查 `$cond` 确实是一个 `bool` 表达式。

### 卫生性：为什么 Rust 宏是安全的

C/C++ 宏的 bug 通常来自名称冲突：

```c
// C：危险 —— `x` 可能遮蔽调用者的 `x`
#define SQUARE(x) ((x) * (x))
int x = 5;
int result = SQUARE(x++);  // 未定义行为：x 被递增两次！
```

Rust 宏是**卫生的** —— 宏内部创建的变量不会泄漏到外部：

```rust
macro_rules! make_x {
    () => {
        let x = 42;  // 这个 `x` 的作用域限于宏展开
    };
}

fn main() {
    let x = 10;
    make_x!();
    println!("{x}");  // 打印 10，而非 42 —— 卫生性防止了冲突
}
```

宏的 `x` 和调用者的 `x` 被编译器视为不同的变量，即使它们同名。**这在 C 预处理器中是不可能的。**

---

## 常用标准库宏

你从第 1 章就一直在使用这些宏 —— 以下是它们实际做的事情：

| 宏 | 功能 | 展开为（简化） |
|-------|-------------|------------------------|
| `println!("{}", x)` | 格式化并打印到 stdout + 换行 | `std::io::_print(format_args!(...))` |
| `eprintln!("{}", x)` | 打印到 stderr + 换行 | 同上但输出到 stderr |
| `format!("{}", x)` | 格式化为 `String` | 分配并返回一个 `String` |
| `vec![1, 2, 3]` | 创建包含元素的 `Vec` | `Vec::from([1, 2, 3])`（近似） |
| `todo!()` | 标记未完成的代码 | `panic!("not yet implemented")` |
| `unimplemented!()` | 标记故意未实现的代码 | `panic!("not implemented")` |
| `unreachable!()` | 标记编译器无法证明不可达的代码 | `panic!("unreachable")` |
| `assert!(cond)` | 条件为 false 时 panic | `if !cond { panic!(...) }` |
| `assert_eq!(a, b)` | 值不相等时 panic | 失败时显示两个值 |
| `dbg!(expr)` | 将表达式 + 值打印到 stderr，返回值 | `eprintln!("[file:line] expr = {:#?}", &expr); expr` |
| `include_str!("file.txt")` | 编译时将文件内容嵌入为 `&str` | 编译期间读取文件 |
| `include_bytes!("data.bin")` | 编译时将文件内容嵌入为 `&[u8]` | 编译期间读取文件 |
| `cfg!(condition)` | 编译时条件作为 `bool` | 根据目标平台返回 `true` 或 `false` |
| `env!("VAR")` | 编译时读取环境变量 | 未设置时编译失败 |
| `concat!("a", "b")` | 编译时连接字面量 | `"ab"` |

### `dbg!` —— 你每天都会用到的调试宏

```rust
fn factorial(n: u32) -> u32 {
    if dbg!(n <= 1) {     // 打印：[src/main.rs:2] n <= 1 = false
        dbg!(1)           // 打印：[src/main.rs:3] 1 = 1
    } else {
        dbg!(n * factorial(n - 1))  // 打印中间值
    }
}

fn main() {
    dbg!(factorial(4));   // 打印所有递归调用及 file:line
}
```

`dbg!` 返回它包裹的值，所以你可以在任何地方插入它而不改变程序行为。它打印到 stderr（而非 stdout），所以不会干扰程序输出。**提交代码前请移除所有 `dbg!` 调用。**

### 格式化字符串语法

由于 `println!`、`format!`、`eprintln!` 和 `write!` 都使用相同的格式化机制，以下是快速参考：

```rust
let name = "sensor";
let value = 3.14159;
let count = 42;

println!("{name}");                    // 按变量名（Rust 1.58+）
println!("{}", name);                  // 按位置
println!("{value:.2}");                // 2 位小数："3.14"
println!("{count:>10}");               // 右对齐，宽度 10："        42"
println!("{count:0>10}");              // 零填充："0000000042"
println!("{count:#06x}");              // 带前缀的十六进制："0x002a"
println!("{count:#010b}");             // 带前缀的二进制："0b00101010"
println!("{value:?}");                 // Debug 格式
println!("{value:#?}");                // 美化的 Debug 格式
```

> **给 C 开发者：** 可以把它想象成类型安全的 `printf` —— 编译器会检查 `{:.2}` 是否应用于浮点数而非字符串。不会有 `%s`/`%d` 格式不匹配的 bug。
>
> **给 C++ 开发者：** 这用一个可读的格式化字符串替代了 `std::cout << std::fixed << std::setprecision(2) << value`。

---

## 派生宏

你在本书的几乎每个结构体上都见过 `#[derive(...)]`：

```rust
#[derive(Debug, Clone, PartialEq)]
struct Point {
    x: f64,
    y: f64,
}
```

`#[derive(Debug)]` 是一个**派生宏** —— 一种特殊的过程宏，可以自动生成特征实现。以下是它生成的内容（简化版）：

```rust
// #[derive(Debug)] 为 Point 生成的代码：
impl std::fmt::Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Point")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}
```

没有 `#[derive(Debug)]`，你需要为每个结构体手动编写那个 `impl` 块。

### 常用派生特征

| 派生 | 生成内容 | 适用场景 |
|--------|-------------------|-------------|
| `Debug` | `{:?}` 格式化 | 几乎总是需要 —— 启用调试打印 |
| `Clone` | `.clone()` 方法 | 需要复制值时 |
| `Copy` | 赋值时隐式复制 | 小型、仅栈类型（整数、`[f64; 3]`） |
| `PartialEq` / `Eq` | `==` 和 `!=` 运算符 | 需要相等比较时 |
| `PartialOrd` / `Ord` | `<`、`>`、`<=`、`>=` 运算符 | 需要排序时 |
| `Hash` | 用于 `HashMap`/`HashSet` 键的哈希 | 用作 map 键的类型 |
| `Default` | `Type::default()` 构造器 | 有合理零值/空值的类型 |
| `serde::Serialize` / `Deserialize` | JSON/TOML 等序列化 | 跨 API 边界的数据类型 |

### 派生决策树

```text
我应该派生它吗？
  │
  ├── 我的类型只包含已实现该特征的类型吗？
  │     ├── 是 → #[derive] 可以工作
  │     └── 否 → 手动编写 impl（或跳过）
  │
  └── 我的类型的使用者会合理地期望这种行为吗？
        ├── 是 → 派生它（Debug、Clone、PartialEq 几乎总是合理的）
        └── 否 → 不要派生（例如，不要为持有文件句柄的类型派生 Copy）
```

> **C++ 对比：** `#[derive(Clone)]` 类似于自动生成正确的拷贝构造函数。`#[derive(PartialEq)]` 类似于自动生成比较每个字段的 `operator==` —— 这是 C++20 的 `= default` 太空船运算符最终提供的功能。

---

## 属性宏

属性宏转换它们所附加的项。你已经用过几个了：

```rust
#[test]                    // 将函数标记为测试
fn test_addition() {
    assert_eq!(2 + 2, 4);
}

#[cfg(target_os = "linux")] // 条件性地包含此函数
fn linux_only() { /* ... */ }

#[derive(Debug)]            // 生成 Debug 实现
struct MyType { /* ... */ }

#[allow(dead_code)]         // 抑制编译器警告
fn unused_helper() { /* ... */ }

#[must_use]                 // 丢弃返回值时发出警告
fn compute_checksum(data: &[u8]) -> u32 { /* ... */ }
```

常用内置属性：

| 属性 | 用途 |
|-----------|---------|
| `#[test]` | 标记为测试函数 |
| `#[cfg(...)]` | 条件编译 |
| `#[derive(...)]` | 自动生成特征实现 |
| `#[allow(...)]` / `#[deny(...)]` / `#[warn(...)]` | 控制 lint 级别 |
| `#[must_use]` | 未使用的返回值发出警告 |
| `#[inline]` / `#[inline(always)]` | 提示内联函数 |
| `#[repr(C)]` | 使用 C 兼容的内存布局（用于 FFI） |
| `#[no_mangle]` | 不修饰符号名称（用于 FFI） |
| `#[deprecated]` | 标记为已弃用，可附带消息 |

> **给 C/C++ 开发者：** 属性替代了预处理器指令（`#pragma`、`__attribute__((...))`）和编译器特定扩展的混合体。它们是语言语法的一部分，而非外挂的扩展。

---

## 过程宏（概念概述）

过程宏（"proc macros"）是作为*独立 Rust 程序*编写的宏，在编译时运行并生成代码。它们比 `macro_rules!` 更强大，但也更复杂。

有三种类型：

| 类型 | 语法 | 示例 | 功能 |
|------|--------|---------|-------------|
| **类函数** | `my_macro!(...)` | `sql!(SELECT * FROM users)` | 解析自定义语法，生成 Rust 代码 |
| **派生** | `#[derive(MyTrait)]` | `#[derive(Serialize)]` | 从结构体定义生成特征实现 |
| **属性** | `#[my_attr]` | `#[tokio::main]`、`#[instrument]` | 转换被标注的项 |

### 你已经用过过程宏了

- `thiserror` 中的 `#[derive(Error)]` —— 为错误枚举生成 `Display` 和 `From` 实现
- `serde` 中的 `#[derive(Serialize, Deserialize)]` —— 生成序列化代码
- `#[tokio::main]` —— 将 `async fn main()` 转换为运行时设置 + block_on
- `#[test]` —— 由测试框架注册（内置过程宏）

### 何时编写自己的过程宏

在本课程中你可能不需要编写过程宏。它们在以下场景有用：
- 需要在编译时检查结构体字段/枚举变体（派生宏）
- 构建领域特定语言（类函数宏）
- 需要转换函数签名（属性宏）

对于大多数代码，`macro_rules!` 或普通函数就足够了。

> **C++ 对比：** 过程宏填补了 C++ 中代码生成器、模板元编程和外部工具（如 `protoc`）的角色。区别在于过程宏是 cargo 构建管道的一部分 —— 无需外部构建步骤，无需 CMake 自定义命令。

---

## 何时使用什么：宏 vs 函数 vs 泛型

```text
需要生成代码？
  │
  ├── 否 → 使用函数或泛型函数
  │         （更简单、更好的错误信息、IDE 支持）
  │
  └── 是 ─┬── 可变数量的参数？
            │     └── 是 → macro_rules!（例如 println!、vec!）
            │
            ├── 为多个类型生成重复的 impl 块？
            │     └── 是 → 带重复的 macro_rules!
            │
            ├── 需要检查结构体字段？
            │     └── 是 → 派生宏（过程宏）
            │
            ├── 需要自定义语法（DSL）？
            │     └── 是 → 类函数过程宏
            │
            └── 需要转换函数/结构体？
                  └── 是 → 属性过程宏
```

**一般准则：** 如果函数或泛型能做到，就不要使用宏。宏的错误信息更差，宏体内没有 IDE 自动补全，且更难调试。

---

## 练习

### 🟢 练习 1：`min!` 宏

编写一个 `min!` 宏：
- `min!(a, b)` 返回两个值中较小的
- `min!(a, b, c)` 返回三个值中最小的
- 适用于任何实现了 `PartialOrd` 的类型

**提示：** 你的 `macro_rules!` 需要两个匹配分支。

<details><summary>参考答案（点击展开）</summary>

```rust
macro_rules! min {
    ($a:expr, $b:expr) => {
        if $a < $b { $a } else { $b }
    };
    ($a:expr, $b:expr, $c:expr) => {
        min!(min!($a, $b), $c)
    };
}

fn main() {
    println!("{}", min!(3, 7));        // 3
    println!("{}", min!(9, 2, 5));     // 2
    println!("{}", min!(1.5, 0.3));    // 0.3
}
```

**注意：** 对于生产代码，优先使用 `std::cmp::min` 或 `a.min(b)`。本练习演示了多分支宏的机制。

</details>

### 🟡 练习 2：从零开始的 `hashmap!`

不参考上面的示例，编写一个 `hashmap!` 宏：
- 从 `key => value` 对创建 `HashMap`
- 支持尾随逗号
- 适用于任何可哈希的键类型

测试：
```rust
let m = hashmap! {
    "name" => "Alice",
    "role" => "Engineer",
};
assert_eq!(m["name"], "Alice");
assert_eq!(m.len(), 2);
```

<details><summary>参考答案（点击展开）</summary>

```rust
use std::collections::HashMap;

macro_rules! hashmap {
    ( $( $key:expr => $val:expr ),* $(,)? ) => {{
        let mut map = HashMap::new();
        $( map.insert($key, $val); )*
        map
    }};
}

fn main() {
    let m = hashmap! {
        "name" => "Alice",
        "role" => "Engineer",
    };
    assert_eq!(m["name"], "Alice");
    assert_eq!(m.len(), 2);
    println!("Tests passed!");
}
```

</details>

### 🟡 练习 3：浮点比较的 `assert_approx_eq!`

编写一个宏 `assert_approx_eq!(a, b, epsilon)`，当 `|a - b| > epsilon` 时 panic。这对于测试精确相等会失败的浮点计算很有用。

测试：
```rust
assert_approx_eq!(0.1 + 0.2, 0.3, 1e-10);        // 应该通过
assert_approx_eq!(3.14159, std::f64::consts::PI, 1e-4); // 应该通过
// assert_approx_eq!(1.0, 2.0, 0.5);              // 应该 panic
```

<details><summary>参考答案（点击展开）</summary>

```rust
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr, $eps:expr) => {
        let (a, b, eps) = ($a as f64, $b as f64, $eps as f64);
        let diff = (a - b).abs();
        if diff > eps {
            panic!(
                "assertion failed: |{} - {}| = {} > {} (epsilon)",
                a, b, diff, eps
            );
        }
    };
}

fn main() {
    assert_approx_eq!(0.1 + 0.2, 0.3, 1e-10);
    assert_approx_eq!(3.14159, std::f64::consts::PI, 1e-4);
    println!("All float comparisons passed!");
}
```

</details>

### 🔴 练习 4：`impl_display_for_enum!`

编写一个宏，为简单的 C 风格枚举生成 `Display` 实现。给定：

```rust
impl_display_for_enum! {
    enum Color {
        Red => "red",
        Green => "green",
        Blue => "blue",
    }
}
```

它应该同时生成 `enum Color { Red, Green, Blue }` 定义和将每个变体映射到其字符串的 `impl Display for Color`。

**提示：** 你需要 `$( ... ),*` 重复和多个片段说明符。

<details><summary>参考答案（点击展开）</summary>

```rust
use std::fmt;

macro_rules! impl_display_for_enum {
    (enum $name:ident { $( $variant:ident => $display:expr ),* $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq)]
        enum $name {
            $( $variant ),*
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                match self {
                    $( $name::$variant => write!(f, "{}", $display), )*
                }
            }
        }
    };
}

impl_display_for_enum! {
    enum Color {
        Red => "red",
        Green => "green",
        Blue => "blue",
    }
}

fn main() {
    let c = Color::Green;
    println!("Color: {c}");          // "Color: green"
    println!("Debug: {c:?}");        // "Debug: Green"
    assert_eq!(format!("{}", Color::Red), "red");
    println!("All tests passed!");
}
```

</details>
