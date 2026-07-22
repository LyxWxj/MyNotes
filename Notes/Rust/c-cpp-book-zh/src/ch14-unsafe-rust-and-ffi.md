### 不安全 Rust

> **你将学到：** 何时以及如何使用 `unsafe` —— 裸指针解引用、FFI（外部函数接口）用于从 Rust 调用 C 以及反过来、用于字符串互操作的 `CString`/`CStr`，以及如何为不安全代码编写安全包装。

- `unsafe` 解锁了 Rust 编译器通常不允许的功能访问
    - 解引用裸指针
    - 访问*可变*静态变量
    - https://doc.rust-lang.org/book/ch19-01-unsafe-rust.html
- 能力越大，责任越大
    - `unsafe` 告诉编译器"我，程序员，承担起维护编译器通常保证的不变量的责任"
    - 必须保证没有别名的可变和不可变引用、没有悬垂指针、没有无效引用……
    - `unsafe` 的使用应限制在尽可能小的作用域内
    - 所有使用 `unsafe` 的代码都应该有描述假设条件的 "safety" 注释

### 不安全 Rust 示例
```rust
unsafe fn harmless() {}
fn main() {
    // Safety: We are calling a harmless unsafe function
    unsafe {
        harmless();
    }
    let a = 42u32;
    let p = &a as *const u32;
    // Safety: p is a valid pointer to a variable that will remain in scope
    unsafe {
        println!("{}", *p);
    }
    // Safety: Not safe; for illustration purposes only
    let dangerous_buffer = 0xb8000 as *mut u32;
    unsafe {
        println!("About to go kaboom!!!");
        *dangerous_buffer = 0; // This will SEGV on most modern machines
    }
}
```

### 简单 FFI 示例（由 C 消费的 Rust 库函数）

## FFI 字符串：CString 和 CStr

FFI 代表*外部函数接口* —— Rust 用来调用其他语言（如 C）编写的函数以及反过来的机制。

与 C 代码交互时，Rust 的 `String` 和 `&str` 类型（UTF-8 编码且没有空终止符）与 C 字符串（以空字符结尾的字节数组）不直接兼容。Rust 提供了 `std::ffi` 中的 `CString`（拥有所有权）和 `CStr`（借用）来解决这个问题：

| 类型 | 类似于 | 使用场景 |
|------|-------------|----------|
| `CString` | `String`（拥有所有权） | 从 Rust 数据创建 C 字符串 |
| `&CStr` | `&str`（借用） | 从外部代码接收 C 字符串 |

```rust
use std::ffi::{CString, CStr};
use std::os::raw::c_char;

fn demo_ffi_strings() {
    // Creating a C-compatible string (adds null terminator)
    let c_string = CString::new("Hello from Rust").expect("CString::new failed");
    let ptr: *const c_char = c_string.as_ptr();

    // Converting a C string back to Rust (unsafe because we trust the pointer)
    // Safety: ptr is valid and null-terminated (we just created it above)
    let back_to_rust: &CStr = unsafe { CStr::from_ptr(ptr) };
    let rust_str: &str = back_to_rust.to_str().expect("Invalid UTF-8");
    println!("{}", rust_str);
}
```

> **警告**：如果输入包含内部空字节（`\0`），`CString::new()` 将返回错误。始终处理 `Result`。你将在下面的 FFI 示例中大量看到 `CStr` 的使用。

- FFI 方法必须标记 `#[no_mangle]` 以确保编译器不会修改函数名
- 我们将把 crate 编译为静态库
    ```
    #[no_mangle] 
    pub extern "C" fn add(left: u64, right: u64) -> u64 {
        left + right
    }
    ```
- 我们将编译以下 C 代码并将其链接到我们的静态库。
    ```
    #include <stdio.h>
    #include <stdint.h>
    extern uint64_t add(uint64_t, uint64_t);
    int main() {
        printf("Add returned %llu\n", add(21, 21));
    }
    ``` 

### 复杂 FFI 示例
- 在以下示例中，我们将创建一个 Rust 日志接口并将其暴露给
[PYTHON] 和 `C`
    - 我们将看到同一个接口如何从 Rust 和 C 原生使用
    - 我们将探索使用 `cbindgen` 等工具为 `C` 生成头文件
    - 我们将看到 `unsafe` 包装如何充当安全 Rust 代码的桥梁

## 日志辅助函数
```rust
fn create_or_open_log_file(log_file: &str, overwrite: bool) -> Result<File, String> {
    if overwrite {
        File::create(log_file).map_err(|e| e.to_string())
    } else {
        OpenOptions::new()
            .write(true)
            .append(true)
            .open(log_file)
            .map_err(|e| e.to_string())
    }
}

fn log_to_file(file_handle: &mut File, message: &str) -> Result<(), String> {
    file_handle
        .write_all(message.as_bytes())
        .map_err(|e| e.to_string())
}
```

## 日志结构体
```rust
struct SimpleLogger {
    log_level: LogLevel,
    file_handle: File,
}

impl SimpleLogger {
    fn new(log_file: &str, overwrite: bool, log_level: LogLevel) -> Result<Self, String> {
        let file_handle = create_or_open_log_file(log_file, overwrite)?;
        Ok(Self {
            file_handle,
            log_level,
        })
    }

    fn log_message(&mut self, log_level: LogLevel, message: &str) -> Result<(), String> {
        if log_level as u32 <= self.log_level as u32 {
            let timestamp = Local::now().format("%Y-%m-%d %H:%M:%S").to_string();
            let message = format!("Simple: {timestamp} {log_level} {message}\n");
            log_to_file(&mut self.file_handle, &message)
        } else {
            Ok(())
        }
    }
}
```

## 测试
- 用 Rust 测试功能非常简单
    - 测试方法用 `#[test]` 修饰，不会包含在编译后的二进制文件中
    - 创建用于测试的模拟方法很容易
```rust
#[test]
fn testfunc() -> Result<(), String> {
    let mut logger = SimpleLogger::new("test.log", false, LogLevel::INFO)?;
    logger.log_message(LogLevel::TRACELEVEL1, "Hello world")?;
    logger.log_message(LogLevel::CRITICAL, "Critical message")?;
    Ok(()) // The compiler automatically drops logger here
}
```
```bash
cargo test
```

## (C)-Rust FFI
- cbindgen 是为导出的 Rust 函数生成头文件的优秀工具
    - 可以使用 cargo 安装
```bash
cargo install cbindgen
cbindgen 
```
- 函数和结构体可以使用 `#[no_mangle]` 和 `#[repr(C)]` 导出
    - 我们假设常见的接口模式：传入 `**` 指向实际实现，成功返回 0，错误返回非零值
    - **不透明 vs 透明结构体**：我们的 `SimpleLogger` 作为*不透明指针*（`*mut SimpleLogger`）传递 —— C 端永远不会访问其字段，所以不需要 `#[repr(C)]`。当 C 代码需要直接读写结构体字段时，使用 `#[repr(C)]`：

```rust
// Opaque — C only holds a pointer, never inspects fields. No #[repr(C)] needed.
struct SimpleLogger { /* Rust-only fields */ }

// Transparent — C reads/writes fields directly. MUST use #[repr(C)].
#[repr(C)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}
```
```c
typedef struct SimpleLogger SimpleLogger;
uint32_t create_simple_logger(const char *file_name, struct SimpleLogger **out_logger);
uint32_t log_entry(struct SimpleLogger *logger, const char *message);
uint32_t drop_logger(struct SimpleLogger *logger);
```

- 注意我们需要做大量的健全性检查
- 我们必须显式泄漏内存以防止 Rust 自动释放
```rust
#[no_mangle] 
pub extern "C" fn create_simple_logger(file_name: *const std::os::raw::c_char, out_logger: *mut *mut SimpleLogger) -> u32 {
    use std::ffi::CStr;
    // Make sure pointer isn't NULL
    if file_name.is_null() || out_logger.is_null() {
        return 1;
    }
    // Safety: The passed in pointer is either NULL or 0-terminated by contract
    let file_name = unsafe {
        CStr::from_ptr(file_name)
    };
    let file_name = file_name.to_str();
    // Make sure that file_name doesn't have garbage characters
    if file_name.is_err() {
        return 1;
    }
    let file_name = file_name.unwrap();
    // Assume some defaults; we'll pass them in real life
    let new_logger = SimpleLogger::new(file_name, true, LogLevel::CRITICAL);
    // Check that we were able to construct the logger
    if new_logger.is_err() {
        return 1;
    }
    let new_logger = Box::new(new_logger.unwrap());
    // This prevents the Box from being dropped when if goes out of scope
    let logger_ptr: *mut SimpleLogger = Box::leak(new_logger);
    // Safety: logger is non-null and logger_ptr is valid
    unsafe {
        *out_logger = logger_ptr;
    }
    return 0;
}
```

- 在 `log_entry()` 中有类似的错误检查
```rust
#[no_mangle]
pub extern "C" fn log_entry(logger: *mut SimpleLogger, message: *const std::os::raw::c_char) -> u32 {
    use std::ffi::CStr;
    if message.is_null() || logger.is_null() {
        return 1;
    }
    // Safety: message is non-null
    let message = unsafe {
        CStr::from_ptr(message)
    };
    let message = message.to_str();
    // Make sure that file_name doesn't have garbage characters
    if message.is_err() {
        return 1;
    }
    // Safety: logger is valid pointer previously constructed by create_simple_logger()
    unsafe {
        (*logger).log_message(LogLevel::CRITICAL, message.unwrap()).is_err() as u32
    }
}

#[no_mangle]
pub extern "C" fn drop_logger(logger: *mut SimpleLogger) -> u32 {
    if logger.is_null() {
        return 1;
    }
    // Safety: logger is valid pointer previously constructed by create_simple_logger()
    unsafe {
        // This constructs a Box<SimpleLogger>, which is dropped when it goes out of scope
        let _ = Box::from_raw(logger);
    }
    0
}
```

- 我们可以使用 Rust 测试 (C)-FFI，也可以编写一个 (C) 程序来测试
```rust
#[test]
fn test_c_logger() {
    // The c".." creates a NULL terminated string
    let file_name = c"test.log".as_ptr() as *const std::os::raw::c_char;
    let mut c_logger: *mut SimpleLogger = std::ptr::null_mut();
    assert_eq!(create_simple_logger(file_name, &mut c_logger), 0);
    // This is the manual way to create c"..." strings
    let message = b"message from C\0".as_ptr() as *const std::os::raw::c_char;
    assert_eq!(log_entry(c_logger, message), 0);
    drop_logger(c_logger);
}
```
```c
#include "logger.h"
...
int main() {
    SimpleLogger *logger = NULL;
    if (create_simple_logger("test.log", &logger) == 0) {
        log_entry(logger, "Hello from C");
        drop_logger(logger); /*Needed to close handle, etc.*/
    } 
    ...
}
```

## 确保不安全代码的正确性
- 简而言之，使用 `unsafe` 需要深思熟虑
    - 始终记录代码所做的安全假设，并与专家一起审查
    - 使用 cbindgen、Miri、Valgrind 等工具帮助验证正确性
    - **永远不要让 panic 跨越 FFI 边界展开** —— 这是未定义行为。在 FFI 入口点使用 `std::panic::catch_unwind`，或在配置中设置 `panic = "abort"`
    - 如果结构体在 FFI 间共享，标记 `#[repr(C)]` 以保证 C 兼容的内存布局
    - 参阅 https://doc.rust-lang.org/nomicon/intro.html（"Rustonomicon" —— 不安全 Rust 的黑魔法）
    - 寻求内部专家的帮助

### 验证工具：Miri vs Valgrind

C++ 开发者熟悉 Valgrind 和 sanitizer。Rust 除了有这些工具外，还有 Miri，它对于 Rust 特有的未定义行为检测更加精确：

| | **Miri** | **Valgrind** | **C++ sanitizer (ASan/MSan/UBSan)** |
|---|---------|-------------|--------------------------------------|
| **能捕获什么** | Rust 特有的未定义行为：栈借用、无效 `enum` 判别值、未初始化读取、别名违规 | 内存泄漏、释放后使用、无效读写、未初始化内存 | 缓冲区溢出、释放后使用、数据竞争、未定义行为 |
| **工作原理** | 解释 MIR（Rust 的中间表示）—— 不进行原生执行 | 在运行时对编译后的二进制文件进行插桩 | 编译时插桩 |
| **FFI 支持** | ❌ 无法跨越 FFI 边界（跳过 C 调用） | ✅ 适用于任何编译后的二进制文件，包括 FFI | ✅ 如果 C 代码也用 sanitizer 编译则有效 |
| **速度** | 比原生慢约 100 倍 | 比原生慢约 10-50 倍 | 比原生慢约 2-5 倍 |
| **何时使用** | 纯 Rust `unsafe` 代码、数据结构不变量 | FFI 代码、完整二进制集成测试 | FFI 的 C/C++ 端、性能敏感的测试 |
| **能否捕获别名 bug** | ✅ 栈借用模型 | ❌ | 部分（TSan 用于数据竞争） |

**建议**：**两者都用** —— Miri 用于纯 Rust 不安全代码，Valgrind 用于 FFI 集成：

- **Miri** —— 捕获 Valgrind 无法发现的 Rust 特有未定义行为（别名违规、无效枚举值、栈借用）：
    ```
    rustup +nightly component add miri
    cargo +nightly miri test                    # Run all tests under Miri
    cargo +nightly miri test -- test_name       # Run a specific test
    ```
    > ⚠️ Miri 需要 nightly 版本且无法执行 FFI 调用。将不安全 Rust 逻辑隔离为可测试的单元。

- **Valgrind** —— 你已经熟悉的工具，适用于包含 FFI 的编译后二进制文件：
    ```
    sudo apt install valgrind
    cargo install cargo-valgrind
    cargo valgrind test                         # Run all tests under Valgrind
    ```
    > 能捕获 FFI 代码中常见的 `Box::leak` / `Box::from_raw` 模式中的泄漏。

- **cargo-careful** —— 启用额外运行时检查运行测试（介于常规测试和 Miri 之间）：
    ```
    cargo install cargo-careful
    cargo +nightly careful test
    ```

## 不安全 Rust 总结
- `cbindgen` 是 (C) FFI 到 Rust 的优秀工具
    - 对于反方向的 FFI 接口使用 `bindgen`（请参阅其详尽文档）
- **不要假设你的不安全代码是正确的，或者从安全 Rust 使用它没有问题。犯错非常容易，即使看似正常工作的代码也可能因为微妙的原因而有问题**
    - 使用工具验证正确性
    - 如果仍有疑问，请寻求专家建议
- 确保你的 `unsafe` 代码有关于假设条件和正确性原因的显式文档注释
    - `unsafe` 代码的调用者也应该有相应的安全注释，并遵守限制条件

# 练习：编写安全的 FFI 包装

🔴 **挑战** —— 需要理解不安全块、裸指针和安全 API 设计

- 为一个 `unsafe` 的 FFI 风格函数编写安全的 Rust 包装。本练习模拟调用一个将格式化字符串写入调用者提供的缓冲区的 C 函数。
- **第 1 步**：实现不安全函数 `unsafe_greet`，将问候语写入裸 `*mut u8` 缓冲区
- **第 2 步**：编写安全包装 `safe_greet`，分配一个 `Vec<u8>`，调用不安全函数，并返回 `String`
- **第 3 步**：为每个不安全块添加正确的 `// Safety:` 注释

**起始代码：**
```rust
use std::fmt::Write as _;

/// Simulates a C function: writes "Hello, <name>!" into buffer.
/// Returns the number of bytes written (excluding null terminator).
/// # Safety
/// - `buf` must point to at least `buf_len` writable bytes
/// - `name` must be a valid pointer to a null-terminated C string
unsafe fn unsafe_greet(buf: *mut u8, buf_len: usize, name: *const u8) -> isize {
    // TODO: Build greeting, copy bytes into buf, return length
    // Hint: use std::ffi::CStr::from_ptr or iterate bytes manually
    todo!()
}

/// Safe wrapper — no unsafe in the public API
fn safe_greet(name: &str) -> Result<String, String> {
    // TODO: Allocate a Vec<u8> buffer, create a null-terminated name,
    // call unsafe_greet inside an unsafe block with Safety comment,
    // convert the result back to a String
    todo!()
}

fn main() {
    match safe_greet("Rustacean") {
        Ok(msg) => println!("{msg}"),
        Err(e) => eprintln!("Error: {e}"),
    }
    // Expected output: Hello, Rustacean!
}
```

<details><summary>解答（点击展开）</summary>

```rust
use std::ffi::CStr;

/// Simulates a C function: writes "Hello, <name>!" into buffer.
/// Returns the number of bytes written, or -1 if buffer too small.
/// # Safety
/// - `buf` must point to at least `buf_len` writable bytes
/// - `name` must be a valid pointer to a null-terminated C string
unsafe fn unsafe_greet(buf: *mut u8, buf_len: usize, name: *const u8) -> isize {
    // Safety: caller guarantees name is a valid null-terminated string
    let name_cstr = unsafe { CStr::from_ptr(name as *const std::os::raw::c_char) };
    let name_str = match name_cstr.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let greeting = format!("Hello, {}!", name_str);
    if greeting.len() > buf_len {
        return -1;
    }
    // Safety: buf points to at least buf_len writable bytes (caller guarantee)
    unsafe {
        std::ptr::copy_nonoverlapping(greeting.as_ptr(), buf, greeting.len());
    }
    greeting.len() as isize
}

/// Safe wrapper — no unsafe in the public API
fn safe_greet(name: &str) -> Result<String, String> {
    let mut buffer = vec![0u8; 256];
    // Create a null-terminated version of name for the C API
    let name_with_null: Vec<u8> = name.bytes().chain(std::iter::once(0)).collect();

    // Safety: buffer has 256 writable bytes, name_with_null is null-terminated
    let bytes_written = unsafe {
        unsafe_greet(buffer.as_mut_ptr(), buffer.len(), name_with_null.as_ptr())
    };

    if bytes_written < 0 {
        return Err("Buffer too small or invalid name".to_string());
    }

    String::from_utf8(buffer[..bytes_written as usize].to_vec())
        .map_err(|e| format!("Invalid UTF-8: {e}"))
}

fn main() {
    match safe_greet("Rustacean") {
        Ok(msg) => println!("{msg}"),
        Err(e) => eprintln!("Error: {e}"),
    }
}
// Output:
// Hello, Rustacean!
```

</details>

----

