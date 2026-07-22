# Rust 最佳实践总结

> **你将学到：** 编写地道 Rust 的实用指南 —— 代码组织、命名规范、错误处理模式和文档编写。这是一个你会经常回来查阅的速查章节。

## 代码组织
- **优先使用小函数**：易于测试和理解
- **使用描述性名称**：`calculate_total_price()` 优于 `calc()`
- **将相关功能分组**：使用模块和独立文件
- **编写文档**：对公共 API 使用 `///`

## 错误处理
- **除非不可能失败，否则避免使用 `unwrap()`**：只在你 100% 确定不会 panic 时使用
```rust
// Bad: Can panic
let value = some_option.unwrap();

// Good: Handle the None case
let value = some_option.unwrap_or(default_value);
let value = some_option.unwrap_or_else(|| expensive_computation());
let value = some_option.unwrap_or_default(); // Uses Default trait

// For Result<T, E>
let value = some_result.unwrap_or(fallback_value);
let value = some_result.unwrap_or_else(|err| {
    eprintln!("Error occurred: {err}");
    default_value
});
```
- **使用带有描述性消息的 `expect()`**：当 unwrap 是合理的时候，解释原因
```rust
let config = std::env::var("CONFIG_PATH")
    .expect("CONFIG_PATH environment variable must be set");
```
- **对可能失败的操作返回 `Result<T, E>`**：让调用者决定如何处理错误
- **使用 `thiserror` 定义自定义错误类型**：比手动实现更符合人体工程学
```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum MyError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    
    #[error("Parse error: {message}")]
    Parse { message: String },
    
    #[error("Value {value} is out of range")]
    OutOfRange { value: i32 },
}
```
- **使用 `?` 操作符链式传播错误**：将错误向上传递到调用栈
- **优先使用 `thiserror` 而不是 `anyhow`**：我们团队的惯例是使用 `#[derive(thiserror::Error)]` 定义显式的错误枚举，这样调用者可以匹配特定的变体。`anyhow::Error` 对于快速原型开发很方便，但会擦除错误类型，使调用者难以处理特定的失败。在库和生产代码中使用 `thiserror`；将 `anyhow` 保留给一次性脚本或只需要打印错误的顶层二进制文件。
- **`unwrap()` 可接受的场景**：
  - **单元测试**：`assert_eq!(result.unwrap(), expected)`
  - **原型开发**：快速粗糙的代码，之后会替换
  - **不可能失败的操作**：当你能证明它不会失败时
```rust
let numbers = vec![1, 2, 3];
let first = numbers.get(0).unwrap(); // Safe: we just created the vec with elements

// Better: Use expect() with explanation
let first = numbers.get(0).expect("numbers vec is non-empty by construction");
```
- **快速失败**：尽早检查前置条件并立即返回错误

## 内存管理
- **优先借用而非克隆**：尽可能使用 `&T` 而不是克隆
- **谨慎使用 `Rc<T>`**：只在需要共享所有权时使用
- **限制生命周期**：使用作用域 `{}` 控制值何时被丢弃
- **避免在公共 API 中使用 `RefCell<T>`**：将内部可变性保持为内部实现

## 性能
- **优化前先做性能分析**：使用 `cargo bench` 和性能分析工具
- **优先使用迭代器而非循环**：更可读且通常更快
- **使用 `&str` 而非 `String`**：当不需要所有权时
- **对大型栈对象考虑使用 `Box<T>`**：如果需要，将它们移到堆上

## 应该实现的基本特征

### 每个类型都应考虑的核心特征

创建自定义类型时，考虑实现这些基本特征，使你的类型在 Rust 中感觉像原生的：

#### **Debug 和 Display**
```rust
use std::fmt;

#[derive(Debug)]  // Automatic implementation for debugging
struct Person {
    name: String,
    age: u32,
}

// Manual Display implementation for user-facing output
impl fmt::Display for Person {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (age {})", self.name, self.age)
    }
}

// Usage:
let person = Person { name: "Alice".to_string(), age: 30 };
println!("{:?}", person);  // Debug: Person { name: "Alice", age: 30 }
println!("{}", person);    // Display: Alice (age 30)
```

#### **Clone 和 Copy**
```rust
// Copy: Implicit duplication for small, simple types
#[derive(Debug, Clone, Copy)]
struct Point {
    x: i32,
    y: i32,
}

// Clone: Explicit duplication for complex types
#[derive(Debug, Clone)]
struct Person {
    name: String,  // String doesn't implement Copy
    age: u32,
}

let p1 = Point { x: 1, y: 2 };
let p2 = p1;  // Copy (implicit)

let person1 = Person { name: "Bob".to_string(), age: 25 };
let person2 = person1.clone();  // Clone (explicit)
```

#### **PartialEq 和 Eq**
```rust
#[derive(Debug, PartialEq, Eq)]
struct UserId(u64);

#[derive(Debug, PartialEq)]
struct Temperature {
    celsius: f64,  // f64 doesn't implement Eq (due to NaN)
}

let id1 = UserId(123);
let id2 = UserId(123);
assert_eq!(id1, id2);  // Works because of PartialEq

let temp1 = Temperature { celsius: 20.0 };
let temp2 = Temperature { celsius: 20.0 };
assert_eq!(temp1, temp2);  // Works with PartialEq
```

#### **PartialOrd 和 Ord**
```rust
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Priority(u8);

let high = Priority(1);
let low = Priority(10);
assert!(high < low);  // Lower numbers = higher priority

// Use in collections
let mut priorities = vec![Priority(5), Priority(1), Priority(8)];
priorities.sort();  // Works because Priority implements Ord
```

#### **Default**
```rust
#[derive(Debug, Default)]
struct Config {
    debug: bool,           // false (default)
    max_connections: u32,  // 0 (default)
    timeout: Option<u64>,  // None (default)
}

// Custom Default implementation
impl Default for Config {
    fn default() -> Self {
        Config {
            debug: false,
            max_connections: 100,  // Custom default
            timeout: Some(30),     // Custom default
        }
    }
}

let config = Config::default();
let config = Config { debug: true, ..Default::default() };  // Partial override
```

#### **From 和 Into**
```rust
struct UserId(u64);
struct UserName(String);

// Implement From, and Into comes for free
impl From<u64> for UserId {
    fn from(id: u64) -> Self {
        UserId(id)
    }
}

impl From<String> for UserName {
    fn from(name: String) -> Self {
        UserName(name)
    }
}

impl From<&str> for UserName {
    fn from(name: &str) -> Self {
        UserName(name.to_string())
    }
}

// Usage:
let user_id: UserId = 123u64.into();         // Using Into
let user_id = UserId::from(123u64);          // Using From
let username = UserName::from("alice");      // &str -> UserName
let username: UserName = "bob".into();       // Using Into
```

#### **TryFrom 和 TryInto**
```rust
use std::convert::TryFrom;

struct PositiveNumber(u32);

#[derive(Debug)]
struct NegativeNumberError;

impl TryFrom<i32> for PositiveNumber {
    type Error = NegativeNumberError;
    
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        if value >= 0 {
            Ok(PositiveNumber(value as u32))
        } else {
            Err(NegativeNumberError)
        }
    }
}

// Usage:
let positive = PositiveNumber::try_from(42)?;     // Ok(PositiveNumber(42))
let error = PositiveNumber::try_from(-5);         // Err(NegativeNumberError)
```

#### **Serde（用于序列化）**
```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    email: String,
}

// Automatic JSON serialization/deserialization
let user = User {
    id: 1,
    name: "Alice".to_string(),
    email: "alice@example.com".to_string(),
};

let json = serde_json::to_string(&user)?;
let deserialized: User = serde_json::from_str(&json)?;
```

### 特征实现检查清单

对于任何新类型，考虑以下检查清单：

```rust
#[derive(
    Debug,          // [OK] Always implement for debugging
    Clone,          // [OK] If the type should be duplicatable
    PartialEq,      // [OK] If the type should be comparable
    Eq,             // [OK] If comparison is reflexive/transitive
    PartialOrd,     // [OK] If the type has ordering
    Ord,            // [OK] If ordering is total
    Hash,           // [OK] If type will be used as HashMap key
    Default,        // [OK] If there's a sensible default value
)]
struct MyType {
    // fields...
}

// Manual implementations to consider:
impl Display for MyType { /* user-facing representation */ }
impl From<OtherType> for MyType { /* convenient conversion */ }
impl TryFrom<FallibleType> for MyType { /* fallible conversion */ }
```

### 何时不实现特征

- **不要为包含堆数据的类型实现 Copy**：`String`、`Vec`、`HashMap` 等
- **如果值可能是 NaN 则不要实现 Eq**：包含 `f32`/`f64` 的类型
- **如果没有合理的默认值则不要实现 Default**：文件句柄、网络连接
- **如果克隆开销很大则不要实现 Clone**：大型数据结构（考虑使用 `Rc<T>` 代替）

### 总结：特征的好处

| 特征 | 好处 | 何时使用 |
|-------|---------|-------------|
| `Debug` | `println!("{:?}", value)` | 始终（极少数例外） |
| `Display` | `println!("{}", value)` | 面向用户的类型 |
| `Clone` | `value.clone()` | 当显式复制有意义时 |
| `Copy` | 隐式复制 | 小型、简单的类型 |
| `PartialEq` | `==` 和 `!=` 操作符 | 大多数类型 |
| `Eq` | 自反等价性 | 当等价性在数学上是健全的 |
| `PartialOrd` | `<`、`>`、`<=`、`>=` | 具有自然排序的类型 |
| `Ord` | `sort()`、`BinaryHeap` | 当排序是全序的 |
| `Hash` | `HashMap` 键 | 用作 map 键的类型 |
| `Default` | `Default::default()` | 有明显默认值的类型 |
| `From/Into` | 便捷转换 | 常见的类型转换 |
| `TryFrom/TryInto` | 可能失败的转换 | 可能失败的转换 |

----

----

