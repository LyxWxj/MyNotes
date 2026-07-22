# Rust Option 和 Result 关键要点

> **你将学到：** 地道的错误处理模式 —— `unwrap()` 的安全替代方案、用于传播的 `?` 运算符、自定义错误类型，以及在生产代码中何时使用 `anyhow` vs `thiserror`。

- ```Option``` 和 ```Result``` 是地道 Rust 的核心组成部分
- **`unwrap()` 的安全替代方案**：
```rust
// Option<T> 的安全替代方案
let value = opt.unwrap_or(default);              // 提供回退值
let value = opt.unwrap_or_else(|| compute());    // 惰性计算回退值
let value = opt.unwrap_or_default();             // 使用 Default 特征实现
let value = opt.expect("descriptive message");   // 仅在 panic 可接受时使用

// Result<T, E> 的安全替代方案
let value = result.unwrap_or(fallback);          // 忽略错误，使用回退值
let value = result.unwrap_or_else(|e| handle(e)); // 处理错误，返回回退值
let value = result.unwrap_or_default();          // 使用 Default 特征
```
- **模式匹配实现显式控制**：
```rust
match some_option {
    Some(value) => println!("Got: {}", value),
    None => println!("No value found"),
}

match some_result {
    Ok(value) => process(value),
    Err(error) => log_error(error),
}
```
- **使用 `?` 运算符进行错误传播**：短路并向上冒泡错误
```rust
fn process_file(path: &str) -> Result<String, std::io::Error> {
    let content = std::fs::read_to_string(path)?; // 自动返回错误
    Ok(content.to_uppercase())
}
```
- **转换方法**：
    - `map()`：转换成功值 `Ok(T)` -> `Ok(U)` 或 `Some(T)` -> `Some(U)`
    - `map_err()`：转换错误类型 `Err(E)` -> `Err(F)`
    - `and_then()`：链接可能失败的操作
- **在你自己的 API 中使用**：优先使用 `Result<T, E>` 而非异常或错误码
- **参考**：[Option 文档](https://doc.rust-lang.org/std/option/enum.Option.html) | [Result 文档](https://doc.rust-lang.org/std/result/enum.Result.html)

# Rust 常见陷阱和调试技巧
- **借用问题**：最常见的新手错误
    - "cannot borrow as mutable" -> 同一时间只允许一个可变引用
    - "borrowed value does not live long enough" -> 引用比它指向的数据活得更久
    - **解决方法**：使用作用域 `{}` 限制引用的生命周期，或在需要时克隆数据
- **缺少特征实现**："method not found" 错误
    - **解决方法**：为常用特征添加 `#[derive(Debug, Clone, PartialEq)]`
    - 使用 `cargo check` 获取比 `cargo run` 更好的错误信息
- **调试模式下的整数溢出**：Rust 在溢出时 panic
    - **解决方法**：使用 `wrapping_add()`、`saturating_add()` 或 `checked_add()` 来显式处理行为
- **String 和 &str 的困惑**：不同用例使用不同类型
    - 对字符串切片（借用的）使用 `&str`，对拥有的字符串使用 `String`
    - **解决方法**：使用 `.to_string()` 或 `String::from()` 将 `&str` 转换为 `String`
- **与借用检查器对抗**：不要试图智胜它
    - **解决方法**：重构代码以配合所有权规则工作，而不是对抗它们
    - 对于复杂的共享场景，考虑使用 `Rc<RefCell<T>>`（谨慎使用）

## 错误处理示例：好的 vs 坏的
```rust
// [ERROR] 坏的：可能意外 panic
fn bad_config_reader() -> String {
    let config = std::env::var("CONFIG_FILE").unwrap(); // 未设置时 panic！
    std::fs::read_to_string(config).unwrap()           // 文件缺失时 panic！
}

// [OK] 好的：优雅地处理错误
fn good_config_reader() -> Result<String, ConfigError> {
    let config_path = std::env::var("CONFIG_FILE")
        .unwrap_or_else(|_| "default.conf".to_string()); // 回退到默认值
    
    let content = std::fs::read_to_string(config_path)
        .map_err(ConfigError::FileRead)?;                // 转换并传播错误
    
    Ok(content)
}

// [OK] 更好的：使用正确的错误类型
use thiserror::Error;

#[derive(Error, Debug)]
enum ConfigError {
    #[error("Failed to read config file: {0}")]
    FileRead(#[from] std::io::Error),
    
    #[error("Invalid configuration: {message}")]
    Invalid { message: String },
}
```

让我们分解一下这里发生了什么。`ConfigError` 只有**两个变体** —— 一个用于 I/O 错误，一个用于验证错误。这是大多数模块的正确起点：

| `ConfigError` 变体 | 包含 | 创建方式 |
|----------------------|-------|-----------|
| `FileRead(io::Error)` | 原始 I/O 错误 | `#[from]` 通过 `?` 自动转换 |
| `Invalid { message }` | 人类可读的解释 | 你的验证代码 |

现在你可以编写返回 `Result<T, ConfigError>` 的函数：

```rust
fn read_config(path: &str) -> Result<String, ConfigError> {
    let content = std::fs::read_to_string(path)?;  // io::Error → ConfigError::FileRead
    if content.is_empty() {
        return Err(ConfigError::Invalid {
            message: "config file is empty".to_string(),
        });
    }
    Ok(content)
}
```

> **🟢 自学检查点：** 在继续之前，确保你能回答：
> 1. 为什么 `read_to_string` 调用上的 `?` 能工作？（因为 `#[from]` 生成了 `impl From<io::Error> for ConfigError`）
> 2. 如果添加第三个变体 `MissingKey(String)` 会怎样 —— 哪些代码需要更改？（只需添加变体；现有代码仍然可以编译）

## Crate 级别的错误类型和 Result 别名

随着项目超出单个文件，你会将多个模块级错误组合成一个**crate 级别的错误类型**。这是生产 Rust 中的标准模式。让我们从上面的 `ConfigError` 开始构建。

在实际的 Rust 项目中，每个 crate（或重要模块）都定义自己的 `Error` 枚举和 `Result` 类型别名。这是地道的模式 —— 类似于在 C++ 中你会为每个库定义异常层次结构和 `using Result = std::expected<T, Error>`。

### 模式

```rust
// src/error.rs（或在 lib.rs 顶部）
use thiserror::Error;

/// 此 crate 可能产生的每种错误。
#[derive(Error, Debug)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),          // 通过 From 自动转换

    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),     // 通过 From 自动转换

    #[error("Invalid sensor id: {0}")]
    InvalidSensor(u32),                  // 领域特定的变体

    #[error("Timeout after {ms} ms")]
    Timeout { ms: u64 },
}

/// Crate 范围的 Result 别名 —— 在整个 crate 中节省输入。
pub type Result<T> = core::result::Result<T, Error>;
```

### 如何简化每个函数

没有别名时你需要写：

```rust
// 冗长 —— 错误类型到处重复
fn read_sensor(id: u32) -> Result<f64, crate::Error> { ... }
fn parse_config(path: &str) -> Result<Config, crate::Error> { ... }
```

有了别名：

```rust
// 简洁 —— 只需 `Result<T>`
use crate::{Error, Result};

fn read_sensor(id: u32) -> Result<f64> {
    if id > 128 {
        return Err(Error::InvalidSensor(id));
    }
    let raw = std::fs::read_to_string(format!("/dev/sensor/{id}"))?; // io::Error → Error::Io
    let value: f64 = raw.trim().parse()
        .map_err(|_| Error::InvalidSensor(id))?;
    Ok(value)
}
```

`Io` 上的 `#[from]` 属性免费生成了这个 `impl`：

```rust
// 由 thiserror 的 #[from] 自动生成
impl From<std::io::Error> for Error {
    fn from(source: std::io::Error) -> Self {
        Error::Io(source)
    }
}
```

这就是 `?` 工作的原理：当函数返回 `std::io::Error` 而你的函数返回 `Result<T>`（你的别名）时，编译器调用 `From::from()` 自动转换。

### 组合模块级错误

较大的 crate 按模块拆分错误，然后在 crate 根目录组合它们：

```rust
// src/config/error.rs
#[derive(thiserror::Error, Debug)]
pub enum ConfigError {
    #[error("Missing key: {0}")]
    MissingKey(String),
    #[error("Invalid value for '{key}': {reason}")]
    InvalidValue { key: String, reason: String },
}

// src/error.rs（crate 级别）
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error(transparent)]               // 将 Display 委托给内部错误
    Config(#[from] crate::config::ConfigError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}
pub type Result<T> = core::result::Result<T, Error>;
```

调用者仍然可以匹配特定的配置错误：

```rust
match result {
    Err(Error::Config(ConfigError::MissingKey(k))) => eprintln!("Add '{k}' to config"),
    Err(e) => eprintln!("Other error: {e}"),
    Ok(v) => use_value(v),
}
```

### C++ 对比

| 概念 | C++ | Rust |
|---------|-----|------|
| 错误层次结构 | `class AppError : public std::runtime_error` | `#[derive(thiserror::Error)] enum Error { ... }` |
| 返回错误 | `std::expected<T, Error>` 或 `throw` | `fn foo() -> Result<T>` |
| 转换错误 | 手动 `try/catch` + 重新抛出 | `#[from]` + `?` —— 零样板代码 |
| Result 别名 | `template<class T> using Result = std::expected<T, Error>;` | `pub type Result<T> = core::result::Result<T, Error>;` |
| 错误消息 | 重写 `what()` | `#[error("...")]` —— 编译为 `Display` 实现 |
