# Rust crate 和模块

> **你将学到：** Rust 如何将代码组织为模块和 crate —— 默认私有的可见性、`pub` 修饰符、工作空间以及 `crates.io` 生态系统。替代 C/C++ 的头文件、`#include` 和 CMake 依赖管理。

- 模块是 crate 内代码的基本组织单元
    - 每个源文件（.rs）都是自己的模块，可以使用 ```mod``` 关键字创建嵌套模块。
    - （子）模块中的所有类型默认是**私有的**，除非明确标记为 ```pub```（公共的），否则在同一 crate 内不可见。```pub``` 的范围可以进一步限制为 ```pub(crate)``` 等
    - 即使类型是公共的，除非使用 ```use``` 关键字导入，否则它不会自动在另一个模块的作用域中可见。子模块可以使用 ```use super::``` 引用父作用域中的类型
    - 源文件（.rs）不会自动包含在 crate 中，**除非**它们在 ```main.rs```（可执行文件）或 ```lib.rs``` 中被明确列出

# 练习：模块和函数
- 我们来看看如何修改 [hello world](https://play.rust-lang.org/?version=stable&mode=debug&edition=2021&gist=522d86dbb8c4af71ff2ec081fb76aee7) 来调用另一个函数
    - 如前所述，函数使用 ```fn``` 关键字定义。```->``` 关键字声明函数返回一个值（默认是 void），类型为 ```u32```（无符号 32 位整数）
    - 函数按模块作用域划分，即两个模块中同名的两个函数不会发生名称冲突
        - 模块作用域扩展到所有类型（例如，```mod a { struct foo; }``` 中的 ```struct foo``` 是一个不同于 ```mod b { struct foo; }```（```b::foo```）的独立类型（```a::foo```））

**起始代码** —— 完成函数：
```rust
mod math {
    // TODO: 实现 pub fn add(a: u32, b: u32) -> u32
}

fn greet(name: &str) -> String {
    // TODO: 返回 "Hello, <name>! The secret number is <math::add(21,21)>"
    todo!()
}

fn main() {
    println!("{}", greet("Rustacean"));
}
```

<details><summary>答案（点击展开）</summary>

```rust
mod math {
    pub fn add(a: u32, b: u32) -> u32 {
        a + b
    }
}

fn greet(name: &str) -> String {
    format!("Hello, {}! The secret number is {}", name, math::add(21, 21))
}

fn main() {
    println!("{}", greet("Rustacean"));
}
// 输出：Hello, Rustacean! The secret number is 42
```

</details>


## 工作空间和 crate（包）

- 任何重要的 Rust 项目都应该使用工作空间来组织组件 crate
    - 工作空间只是一组本地 crate 的集合，用于构建目标二进制文件。工作空间根目录的 `Cargo.toml` 应该有一个指向各组成包（crate）的指针

```toml
[workspace]
resolver = "2"
members = ["package1", "package2"]
```

```text
workspace_root/
|-- Cargo.toml      # 工作空间配置
|-- package1/
|   |-- Cargo.toml  # 包 1 配置
|   `-- src/
|       `-- lib.rs  # 包 1 源代码
|-- package2/
|   |-- Cargo.toml  # 包 2 配置
|   `-- src/
|       `-- main.rs # 包 2 源代码
```

---
## 练习：使用工作空间和包依赖
- 我们将创建一个简单的包，并从 ```hello world``` 程序中使用它
- 创建工作空间目录
```bash
mkdir workspace
cd workspace
```
- 创建一个名为 Cargo.toml 的文件并添加以下内容。这将创建一个空的工作空间
```toml
[workspace]
resolver = "2"
members = []
```
- 添加包（```cargo new --lib``` 指定创建库而非可执行文件）
```bash
cargo new hello
cargo new --lib hellolib
```

## 练习：使用工作空间和包依赖
- 查看 ```hello``` 和 ```hellolib``` 中生成的 Cargo.toml。注意它们都已被添加到上层的 ```Cargo.toml``` 中
- ```hellolib``` 中存在 ```lib.rs``` 意味着这是一个库包（自定义选项参见 https://doc.rust-lang.org/cargo/reference/cargo-targets.html）
- 在 ```hello``` 的 ```Cargo.toml``` 中添加对 ```hellolib``` 的依赖
```toml
[dependencies]
hellolib = {path = "../hellolib"}
```
- 使用 ```hellolib``` 中的 ```add()```
```rust
fn main() {
    println!("Hello, world! {}", hellolib::add(21, 21));
}
```

<details><summary>答案（点击展开）</summary>

完整的工作空间设置：

```bash
# 终端命令
mkdir workspace && cd workspace

# 创建工作空间 Cargo.toml
cat > Cargo.toml << 'EOF'
[workspace]
resolver = "2"
members = ["hello", "hellolib"]
EOF

cargo new hello
cargo new --lib hellolib
```

```toml
# hello/Cargo.toml —— 添加依赖
[dependencies]
hellolib = {path = "../hellolib"}
```

```rust
// hellolib/src/lib.rs —— cargo new --lib 已经有 add()
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
```

```rust,ignore
// hello/src/main.rs
fn main() {
    println!("Hello, world! {}", hellolib::add(21, 21));
}
// 输出：Hello, world! 42
```

</details>

# 使用 crates.io 上的社区 crate
- Rust 拥有一个充满活力的社区 crate 生态系统（参见 https://crates.io/）
    - Rust 的理念是保持标准库精简，将功能外包给社区 crate
    - 使用社区 crate 没有硬性规定，但经验法则是确保 crate 具有一定的成熟度（由版本号指示），并且正在被积极维护。如果对某个 crate 有疑问，请咨询内部资源
- 发布在 ```crates.io``` 上的每个 crate 都有主版本号和次版本号
    - crate 应遵守这里定义的主要和次要 ```SemVer``` 准则：https://doc.rust-lang.org/cargo/reference/semver.html
    - 简而言之，相同次版本号内不应有破坏性更改。例如，v0.11 必须与 v0.15 兼容（但 v0.20 可能有破坏性更改）

# Crate 依赖和 SemVer
- crate 可以定义对特定版本、特定次版本或主版本的依赖，或不关心版本。以下示例展示了在 ```Cargo.toml``` 中声明对 ```rand``` crate 依赖的条目
- 至少 ```0.10.0```，但 ```< 0.11.0``` 的任何版本都可以
```toml
[dependencies]
rand = { version = "0.10.0"}
```
- 仅限 ```0.10.0```，不接受其他版本
```toml
[dependencies]
rand = { version = "=0.10.0"}
```
- 不关心；```cargo``` 将选择最新版本
```toml
[dependencies]
rand = { version = "*"}
```
- 参考：https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html
----
# 练习：使用 rand crate
- 修改 ```helloworld``` 示例以打印随机数
- 使用 ```cargo add rand``` 添加依赖
- 使用 ```https://docs.rs/rand/latest/rand/``` 作为 API 参考

**起始代码** —— 运行 `cargo add rand` 后将此添加到 `main.rs`：
```rust,ignore
use rand::RngExt;

fn main() {
    let mut rng = rand::rng();
    // TODO: 生成并打印 1..=100 范围内的随机 u32
    // TODO: 生成并打印随机 bool
    // TODO: 生成并打印随机 f64
}
```

<details><summary>答案（点击展开）</summary>

```rust
use rand::RngExt;

fn main() {
    let mut rng = rand::rng();
    let n: u32 = rng.random_range(1..=100);
    println!("Random number (1-100): {n}");

    // 生成随机布尔值
    let b: bool = rng.random();
    println!("Random bool: {b}");

    // 生成 0.0 到 1.0 之间的随机浮点数
    let f: f64 = rng.random();
    println!("Random float: {f:.4}");
}
```

</details>

# Cargo.toml 和 Cargo.lock
- 如前所述，Cargo.lock 是从 Cargo.toml 自动生成的
    - Cargo.lock 的主要思想是确保可重现的构建。例如，如果 ```Cargo.toml``` 指定了版本 ```0.10.0```，cargo 可以自由选择 ```< 0.11.0``` 的任何版本
    - Cargo.lock 包含构建期间使用的 rand crate 的*具体*版本。
    - 建议将 ```Cargo.lock``` 包含在 git 仓库中以确保可重现的构建

## Cargo 测试功能
- Rust 单元测试驻留在同一源文件中（按惯例），通常分组到单独的模块中
    - 测试代码永远不会包含在实际的二进制文件中。这通过 ```cfg```（配置）功能实现。配置对于创建平台特定代码（例如 ```Linux``` 与 ```Windows```）很有用
    - 测试可以用 ```cargo test``` 执行。参考：https://doc.rust-lang.org/reference/conditional-compilation.html

```rust
pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
// 仅在测试期间包含
#[cfg(test)]
mod tests {
    use super::*; // 这使得父作用域中的所有类型可见
    #[test]
    fn it_works() {
        let result = add(2, 2); // 或者 super::add(2, 2);
        assert_eq!(result, 4);
    }
}
```

# 其他 Cargo 功能
- ```cargo``` 还有几个其他有用的功能，包括：
    - ```cargo clippy``` 是 lint Rust 代码的好方法。通常应该修复警告（或在确实有必要时很少抑制）
    - ```cargo format``` 执行 ```rustfmt``` 工具来格式化源代码。使用该工具可确保提交代码的标准格式化，并终结关于风格的争论
    - ```cargo doc``` 可用于从 ```///``` 风格的注释生成文档。```crates.io``` 上所有 crate 的文档都是用这种方法生成的

### 构建配置：控制优化

在 C 中，你向 `gcc`/`clang` 传递 `-O0`、`-O2`、`-Os`、`-flto`。在 Rust 中，你在 `Cargo.toml` 中配置构建配置：

```toml
# Cargo.toml —— 构建配置

[profile.dev]
opt-level = 0          # 无优化（快速编译，类似 -O0）
debug = true           # 完整调试符号（类似 -g）

[profile.release]
opt-level = 3          # 最大优化（类似 -O3）
lto = "fat"            # 链接时优化（类似 -flto）
strip = true           # 去除符号（类似 strip 命令）
codegen-units = 1      # 单个代码生成单元 —— 编译更慢，但优化更好
panic = "abort"        # 无展开表（更小的二进制文件）
```

| C/GCC 标志 | Cargo.toml 键 | 值 |
|------------|---------------|--------|
| `-O0` / `-O2` / `-O3` | `opt-level` | `0`, `1`, `2`, `3`, `"s"`, `"z"` |
| `-flto` | `lto` | `false`, `"thin"`, `"fat"` |
| `-g` / 无 `-g` | `debug` | `true`, `false`, `"line-tables-only"` |
| `strip` 命令 | `strip` | `"none"`, `"debuginfo"`, `"symbols"`, `true`/`false` |
| — | `codegen-units` | `1` = 最佳优化，最慢编译 |

```bash
cargo build              # 使用 [profile.dev]
cargo build --release    # 使用 [profile.release]
```

### 构建脚本（`build.rs`）：链接 C 库

在 C 中，你使用 Makefile 或 CMake 来链接库和运行代码生成。Rust 使用 crate 根目录下的 `build.rs` 文件：

```rust
// build.rs —— 在编译 crate 之前运行

fn main() {
    // 链接系统 C 库（类似 gcc 中的 -lbmc_ipmi）
    println!("cargo::rustc-link-lib=bmc_ipmi");

    // 在哪里查找库（类似 -L/usr/lib/bmc）
    println!("cargo::rustc-link-search=/usr/lib/bmc");

    // 如果 C 头文件更改则重新运行
    println!("cargo::rerun-if-changed=wrapper.h");
}
```

你甚至可以直接从 Rust crate 编译 C 源文件：

```toml
# Cargo.toml
[build-dependencies]
cc = "1"  # C 编译器集成
```

```rust
// build.rs
fn main() {
    cc::Build::new()
        .file("src/c_helpers/ipmi_raw.c")
        .include("/usr/include/bmc")
        .compile("ipmi_raw");   // 生成 libipmi_raw.a，自动链接
    println!("cargo::rerun-if-changed=src/c_helpers/ipmi_raw.c");
}
```

| C / Make / CMake | Rust `build.rs` |
|-----------------|-----------------|
| `-lfoo` | `println!("cargo::rustc-link-lib=foo")` |
| `-L/path` | `println!("cargo::rustc-link-search=/path")` |
| 编译 C 源文件 | `cc::Build::new().file("foo.c").compile("foo")` |
| 生成代码 | 将文件写入 `$OUT_DIR`，然后 `include!()` |

### 交叉编译

在 C 中，交叉编译需要安装单独的工具链（`arm-linux-gnueabihf-gcc`）并配置 Make/CMake。在 Rust 中：

```bash
# 安装交叉编译目标
rustup target add aarch64-unknown-linux-gnu

# 交叉编译
cargo build --target aarch64-unknown-linux-gnu --release
```

在 `.cargo/config.toml` 中指定链接器：

```toml
[target.aarch64-unknown-linux-gnu]
linker = "aarch64-linux-gnu-gcc"
```

| C 交叉编译 | Rust 等价物 |
|-----------------|-----------------|
| `apt install gcc-aarch64-linux-gnu` | `rustup target add aarch64-unknown-linux-gnu` + 安装链接器 |
| `CC=aarch64-linux-gnu-gcc make` | `.cargo/config.toml` `[target.X] linker = "..."` |
| `#ifdef __aarch64__` | `#[cfg(target_arch = "aarch64")]` |
| 单独的 Makefile 目标 | `cargo build --target ...` |

### 功能标志：条件编译

C 使用 `#ifdef` 和 `-DFOO` 进行条件编译。Rust 使用在 `Cargo.toml` 中定义的功能标志：

```toml
# Cargo.toml
[features]
default = ["json"]         # 默认启用
json = ["dep:serde_json"]  # 可选依赖
verbose = []               # 无依赖的标志
gpu = ["dep:cuda-sys"]     # 可选 GPU 支持
```

```rust
// 基于功能的条件代码：
#[cfg(feature = "json")]
pub fn parse_config(data: &str) -> Result<Config, Error> {
    serde_json::from_str(data).map_err(Error::from)
}

#[cfg(feature = "verbose")]
macro_rules! verbose {
    ($($arg:tt)*) => { eprintln!("[VERBOSE] {}", format!($($arg)*)); }
}
#[cfg(not(feature = "verbose"))]
macro_rules! verbose {
    ($($arg:tt)*) => {}; // 编译为空
}
```

| C 预处理器 | Rust 功能标志 |
|---------------|-------------------|
| `gcc -DDEBUG` | `cargo build --features verbose` |
| `#ifdef DEBUG` | `#[cfg(feature = "verbose")]` |
| `#define MAX 100` | `const MAX: u32 = 100;` |
| `#ifdef __linux__` | `#[cfg(target_os = "linux")]` |

### 集成测试与单元测试

单元测试与代码一起放在 `#[cfg(test)]` 中。**集成测试**位于 `tests/` 目录中，仅测试 crate 的**公共 API**：

```rust
// tests/smoke_test.rs —— 不需要 #[cfg(test)]
use my_crate::parse_config;

#[test]
fn parse_valid_config() {
    let config = parse_config("test_data/valid.json").unwrap();
    assert_eq!(config.max_retries, 5);
}
```

| 方面 | 单元测试（`#[cfg(test)]`） | 集成测试（`tests/`） |
|--------|----------------------------|------------------------------|
| 位置 | 与代码同一文件 | 单独的 `tests/` 目录 |
| 访问权限 | 私有 + 公共项 | **仅公共 API** |
| 运行命令 | `cargo test` | `cargo test --test smoke_test` |


### 测试模式和策略

C 固件团队通常使用 CUnit、CMocka 或自定义框架编写测试，需要大量样板代码。Rust 的内置测试框架功能要强大得多。本节涵盖了生产代码所需的模式。

#### `#[should_panic]` —— 测试预期失败

```rust
// 测试某些条件会导致 panic（类似 C 的 assert 失败）
#[test]
#[should_panic(expected = "index out of bounds")]
fn test_bounds_check() {
    let v = vec![1, 2, 3];
    let _ = v[10];  // 应该 panic
}

#[test]
#[should_panic(expected = "temperature exceeds safe limit")]
fn test_thermal_shutdown() {
    fn check_temperature(celsius: f64) {
        if celsius > 105.0 {
            panic!("temperature exceeds safe limit: {celsius}°C");
        }
    }
    check_temperature(110.0);
}
```

#### `#[ignore]` —— 慢速或依赖硬件的测试

```rust
// 标记需要特殊条件的测试（类似 C 的 #ifdef HARDWARE_TEST）
#[test]
#[ignore = "requires GPU hardware"]
fn test_gpu_ecc_scrub() {
    // 此测试仅在有 GPU 的机器上运行
    // 运行方式：cargo test -- --ignored
    // 运行方式：cargo test -- --include-ignored（运行所有测试）
}
```

#### 返回 Result 的测试（替代 `unwrap` 链）

```rust
// 不使用许多隐藏实际失败的 unwrap() 调用：
#[test]
fn test_config_parsing() -> Result<(), Box<dyn std::error::Error>> {
    let json = r#"{"hostname": "node-01", "port": 8080}"#;
    let config: ServerConfig = serde_json::from_str(json)?;  // 用 ? 代替 unwrap()
    assert_eq!(config.hostname, "node-01");
    assert_eq!(config.port, 8080);
    Ok(())  // 如果到达这里没有错误，测试通过
}
```

#### 使用构建器函数的测试夹具

C 使用 `setUp()`/`tearDown()` 函数。Rust 使用辅助函数和 `Drop`：

```rust
struct TestFixture {
    temp_dir: std::path::PathBuf,
    config: Config,
}

impl TestFixture {
    fn new() -> Self {
        let temp_dir = std::env::temp_dir().join(format!("test_{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).unwrap();
        let config = Config {
            log_dir: temp_dir.clone(),
            max_retries: 3,
            ..Default::default()
        };
        Self { temp_dir, config }
    }
}

impl Drop for TestFixture {
    fn drop(&mut self) {
        // 自动清理 —— 类似 C 的 tearDown() 但不会被遗忘
        let _ = std::fs::remove_dir_all(&self.temp_dir);
    }
}

#[test]
fn test_with_fixture() {
    let fixture = TestFixture::new();
    // 使用 fixture.config、fixture.temp_dir...
    assert!(fixture.temp_dir.exists());
    // fixture 在这里自动 drop → 清理运行
}
```

#### 使用特征模拟硬件接口

在 C 中，模拟硬件需要预处理器技巧或函数指针替换。在 Rust 中，特征使这变得自然：

```rust
// 用于 IPMI 通信的生产特征
trait IpmiTransport {
    fn send_command(&self, cmd: u8, data: &[u8]) -> Result<Vec<u8>, String>;
}

// 真实实现（用于生产）
struct RealIpmi { /* BMC 连接详情 */ }
impl IpmiTransport for RealIpmi {
    fn send_command(&self, cmd: u8, data: &[u8]) -> Result<Vec<u8>, String> {
        // 实际与 BMC 硬件通信
        todo!("Real IPMI call")
    }
}

// 模拟实现（用于测试）
struct MockIpmi {
    responses: std::collections::HashMap<u8, Vec<u8>>,
}
impl IpmiTransport for MockIpmi {
    fn send_command(&self, cmd: u8, _data: &[u8]) -> Result<Vec<u8>, String> {
        self.responses.get(&cmd)
            .cloned()
            .ok_or_else(|| format!("No mock response for cmd 0x{cmd:02x}"))
    }
}

// 同时适用于真实和模拟的泛型函数
fn read_sensor_temperature(transport: &dyn IpmiTransport) -> Result<f64, String> {
    let response = transport.send_command(0x2D, &[])?;
    if response.len() < 2 {
        return Err("Response too short".into());
    }
    Ok(response[0] as f64 + (response[1] as f64 / 256.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_reading() {
        let mut mock = MockIpmi { responses: std::collections::HashMap::new() };
        mock.responses.insert(0x2D, vec![72, 128]); // 72.5°C

        let temp = read_sensor_temperature(&mock).unwrap();
        assert!((temp - 72.5).abs() < 0.01);
    }

    #[test]
    fn test_short_response() {
        let mock = MockIpmi { responses: std::collections::HashMap::new() };
        // 未配置响应 → 错误
        assert!(read_sensor_temperature(&mock).is_err());
    }
}
```

#### 使用 `proptest` 进行基于属性的测试

不是测试特定值，而是测试必须始终成立的**属性**：

```rust
// Cargo.toml: [dev-dependencies] proptest = "1"
use proptest::prelude::*;

fn parse_sensor_id(s: &str) -> Option<u32> {
    s.strip_prefix("sensor_")?.parse().ok()
}

fn format_sensor_id(id: u32) -> String {
    format!("sensor_{id}")
}

proptest! {
    #[test]
    fn roundtrip_sensor_id(id in 0u32..10000) {
        // 属性：格式化然后解析应该返回原始值
        let formatted = format_sensor_id(id);
        let parsed = parse_sensor_id(&formatted);
        prop_assert_eq!(parsed, Some(id));
    }

    #[test]
    fn parse_rejects_garbage(s in "[^s].*") {
        // 属性：不以 's' 开头的字符串永远不应该被解析成功
        let result = parse_sensor_id(&s);
        prop_assert!(result.is_none());
    }
}
```

#### C 与 Rust 测试对比

| C 测试 | Rust 等价物 |
|-----------|----------------|
| `CUnit`、`CMocka`、自定义框架 | 内置 `#[test]` + `cargo test` |
| `setUp()` / `tearDown()` | 构建器函数 + `Drop` 特征 |
| `#ifdef TEST` 模拟函数 | 基于特征的依赖注入 |
| `assert(x == y)` | `assert_eq!(x, y)` 带自动差异输出 |
| 单独的测试可执行文件 | 同一二进制文件，使用 `#[cfg(test)]` 条件编译 |
| `valgrind --leak-check=full ./test` | `cargo test`（默认内存安全）+ `cargo miri test` |
| 代码覆盖率：`gcov` / `lcov` | `cargo tarpaulin` 或 `cargo llvm-cov` |
| 测试发现：手动注册 | 自动 —— 任何 `#[test]` 函数都会被发现 |


