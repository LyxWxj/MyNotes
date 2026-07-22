## 日志与追踪：syslog/printf → `log` + `tracing`

> **你将学到：** Rust 的双层日志架构（门面 + 后端）、`log` 和 `tracing` crate、带 span 的结构化日志，以及它如何替代 `printf`/`syslog` 调试方式。

C++ 诊断代码通常使用 `printf`、`syslog` 或自定义日志框架。
Rust 有一套标准化的双层日志架构：一个**门面** crate（`log` 或
`tracing`）和一个**后端**（实际的日志实现）。

### `log` 门面 —— Rust 的通用日志 API

`log` crate 提供了与 syslog 严重级别对应的宏。库使用
`log` 宏；二进制程序选择后端：

```rust
// Cargo.toml
// [dependencies]
// log = "0.4"
// env_logger = "0.11"    # 众多后端之一

use log::{info, warn, error, debug, trace};

fn check_sensor(id: u32, temp: f64) {
    trace!("Reading sensor {id}");           // 最细粒度
    debug!("Sensor {id} raw value: {temp}"); // 开发阶段的详细信息

    if temp > 85.0 {
        warn!("Sensor {id} high temperature: {temp}°C");
    }
    if temp > 95.0 {
        error!("Sensor {id} CRITICAL: {temp}°C — initiating shutdown");
    }
    info!("Sensor {id} check complete");     // 正常运行
}

fn main() {
    // 初始化后端 —— 通常在 main() 中只做一次
    env_logger::init();  // 通过 RUST_LOG 环境变量控制

    check_sensor(0, 72.5);
    check_sensor(1, 91.0);
}
```

```bash
# 通过环境变量控制日志级别
RUST_LOG=debug cargo run          # 显示 debug 及以上级别
RUST_LOG=warn cargo run           # 仅显示 warn 和 error
RUST_LOG=my_crate=trace cargo run # 按模块过滤
RUST_LOG=my_crate::gpu=debug,warn cargo run  # 混合级别
```

### C++ 对比

| C++ | Rust (`log`) | 说明 |
|-----|-------------|-------|
| `printf("DEBUG: %s\n", msg)` | `debug!("{msg}")` | 格式在编译时检查 |
| `syslog(LOG_ERR, "...")` | `error!("...")` | 后端决定输出位置 |
| 日志调用周围的 `#ifdef DEBUG` | `trace!` / `debug!` 在 max_level 时被编译移除 | 禁用时零开销 |
| 自定义 `Logger::log(level, msg)` | `log::info!("...")` —— 所有 crate 使用相同 API | 通用门面，可替换后端 |
| 按文件设置日志详细度 | `RUST_LOG=crate::module=level` | 基于环境变量，无需重新编译 |

### `tracing` crate —— 带 span 的结构化日志

`tracing` 在 `log` 的基础上扩展了**结构化字段**和 **span**（带时间的作用域）。
这在诊断代码中特别有用，因为你可以追踪上下文：

```rust
// Cargo.toml
// [dependencies]
// tracing = "0.1"
// tracing-subscriber = { version = "0.3", features = ["env-filter"] }

use tracing::{info, warn, error, instrument, info_span};

#[instrument(skip(data), fields(gpu_id = gpu_id, data_len = data.len()))]
fn run_gpu_test(gpu_id: u32, data: &[u8]) -> Result<(), String> {
    info!("Starting GPU test");

    let span = info_span!("ecc_check", gpu_id);
    let _guard = span.enter();  // 此作用域内的所有日志都包含 gpu_id

    if data.is_empty() {
        error!(gpu_id, "No test data provided");
        return Err("empty data".to_string());
    }

    // 结构化字段 —— 机器可解析，不仅仅是字符串插值
    info!(
        gpu_id,
        temp_celsius = 72.5,
        ecc_errors = 0,
        "ECC check passed"
    );

    Ok(())
}

fn main() {
    // 初始化 tracing 订阅者
    tracing_subscriber::fmt()
        .with_env_filter("debug")  // 或使用 RUST_LOG 环境变量
        .with_target(true)          // 显示模块路径
        .with_thread_ids(true)      // 显示线程 ID
        .init();

    let _ = run_gpu_test(0, &[1, 2, 3]);
}
```

使用 `tracing-subscriber` 的输出：
```rust
2026-02-15T10:30:00.123Z DEBUG ThreadId(01) run_gpu_test{gpu_id=0 data_len=3}: my_crate: Starting GPU test
2026-02-15T10:30:00.124Z  INFO ThreadId(01) run_gpu_test{gpu_id=0 data_len=3}:ecc_check{gpu_id=0}: my_crate: ECC check passed gpu_id=0 temp_celsius=72.5 ecc_errors=0
```

### `#[instrument]` —— 自动创建 span

`#[instrument]` 属性会自动用函数名和参数创建一个 span：

```rust
use tracing::instrument;

#[instrument]
fn parse_sel_record(record_id: u16, sensor_type: u8, data: &[u8]) -> Result<(), String> {
    // 此函数内的每条日志自动包含：
    // record_id、sensor_type 和 data（如果实现了 Debug）
    tracing::debug!("Parsing SEL record");
    Ok(())
}

// skip：从 span 中排除大型/敏感参数
// fields：添加计算字段
#[instrument(skip(raw_buffer), fields(buf_len = raw_buffer.len()))]
fn decode_ipmi_response(raw_buffer: &[u8]) -> Result<Vec<u8>, String> {
    tracing::trace!("Decoding {} bytes", raw_buffer.len());
    Ok(raw_buffer.to_vec())
}
```

### `log` vs `tracing` —— 如何选择

| 方面 | `log` | `tracing` |
|--------|-------|-----------|
| **复杂度** | 简单 —— 5 个宏 | 更丰富 —— span、字段、instrument |
| **结构化数据** | 仅字符串插值 | 键值字段：`info!(gpu_id = 0, "msg")` |
| **计时 / span** | 无 | 有 —— `#[instrument]`、`span.enter()` |
| **异步支持** | 基础 | 一等支持 —— span 可跨 `.await` 传播 |
| **兼容性** | 通用门面 | 与 `log` 兼容（有 `log` 桥接层） |
| **适用场景** | 简单应用、库 | 诊断工具、异步代码、可观测性 |

> **建议**：对于生产级诊断风格的项目（带结构化输出的诊断工具），使用 `tracing`。
> 对于依赖最小化的简单库，使用 `log`。`tracing` 包含兼容层，因此使用 `log`
> 宏的库仍然可以与 `tracing` 订阅者配合工作。

### 后端选项

| 后端 Crate | 输出 | 使用场景 |
|--------------|--------|----------|
| `env_logger` | stderr，带颜色 | 开发、简单 CLI 工具 |
| `tracing-subscriber` | stderr，格式化 | 配合 `tracing` 的生产环境 |
| `syslog` | 系统 syslog | Linux 系统服务 |
| `tracing-journald` | systemd journal | systemd 管理的服务 |
| `tracing-appender` | 滚动日志文件 | 长期运行的守护进程 |
| `tracing-opentelemetry` | OpenTelemetry 收集器 | 分布式追踪 |

----
