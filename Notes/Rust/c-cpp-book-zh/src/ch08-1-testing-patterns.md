## 面向 C++ 程序员的测试模式

> **你将学到：** Rust 的内置测试框架 —— `#[test]`、`#[should_panic]`、返回 `Result` 的测试、测试数据的构建器模式、基于特征的模拟、使用 `proptest` 的属性测试、使用 `insta` 的快照测试以及集成测试组织。零配置测试，替代 Google Test + CMake。

C++ 测试通常依赖外部框架（Google Test、Catch2、Boost.Test），需要复杂的构建集成。Rust 的测试框架**内置于语言和工具链中** —— 无需依赖、无需 CMake 集成、无需测试运行器配置。

### `#[test]` 之外的测试属性

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_pass() {
        assert_eq!(2 + 2, 4);
    }

    // 期望 panic —— 等价于 GTest 的 EXPECT_DEATH
    #[test]
    #[should_panic]
    fn out_of_bounds_panics() {
        let v = vec![1, 2, 3];
        let _ = v[10]; // Panic —— 测试通过
    }

    // 期望 panic 并带有特定消息子串
    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn specific_panic_message() {
        let v = vec![1, 2, 3];
        let _ = v[10];
    }

    // 返回 Result<(), E> 的测试 —— 用 ? 代替 unwrap()
    #[test]
    fn test_with_result() -> Result<(), String> {
        let value: u32 = "42".parse().map_err(|e| format!("{e}"))?;
        assert_eq!(value, 42);
        Ok(())
    }

    // 默认忽略慢速测试 —— 用 `cargo test -- --ignored` 运行
    #[test]
    #[ignore]
    fn slow_integration_test() {
        std::thread::sleep(std::time::Duration::from_secs(10));
    }
}
```

```bash
cargo test                          # 运行所有非忽略的测试
cargo test -- --ignored             # 仅运行被忽略的测试
cargo test -- --include-ignored     # 运行所有测试，包括被忽略的
cargo test test_name                # 运行匹配名称模式的测试
cargo test -- --nocapture           # 测试期间显示 println! 输出
cargo test -- --test-threads=1      # 串行运行测试（用于共享状态）
```

### 测试辅助工具：测试数据的构建器模式

在 C++ 中你会使用 Google Test 夹具（`class MyTest : public ::testing::Test`）。在 Rust 中，使用构建器函数或 `Default` 特征：

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // 构建器函数 —— 使用合理的默认值创建测试数据
    fn make_gpu_event(severity: Severity, fault_code: u32) -> DiagEvent {
        DiagEvent {
            source: "accel_diag".to_string(),
            severity,
            message: format!("Test event FC:{fault_code}"),
            fault_code,
        }
    }

    // 可复用的测试夹具 —— 一组预构建的事件
    fn sample_events() -> Vec<DiagEvent> {
        vec![
            make_gpu_event(Severity::Critical, 67956),
            make_gpu_event(Severity::Warning, 32709),
            make_gpu_event(Severity::Info, 10001),
        ]
    }

    #[test]
    fn filter_critical_events() {
        let events = sample_events();
        let critical: Vec<_> = events.iter()
            .filter(|e| e.severity == Severity::Critical)
            .collect();
        assert_eq!(critical.len(), 1);
        assert_eq!(critical[0].fault_code, 67956);
    }
}
```

### 使用特征进行模拟

在 C++ 中，模拟需要 Google Mock 等框架或手动虚函数重写。在 Rust 中，为依赖定义特征并在测试中替换实现：

```rust
// 生产特征
trait SensorReader {
    fn read_temperature(&self, sensor_id: u32) -> Result<f64, String>;
}

// 生产实现
struct HwSensorReader;
impl SensorReader for HwSensorReader {
    fn read_temperature(&self, sensor_id: u32) -> Result<f64, String> {
        // 真实硬件调用...
        Ok(72.5)
    }
}

// 测试模拟 —— 返回可预测的值
#[cfg(test)]
struct MockSensorReader {
    temperatures: std::collections::HashMap<u32, f64>,
}

#[cfg(test)]
impl SensorReader for MockSensorReader {
    fn read_temperature(&self, sensor_id: u32) -> Result<f64, String> {
        self.temperatures.get(&sensor_id)
            .copied()
            .ok_or_else(|| format!("Unknown sensor {sensor_id}"))
    }
}

// 被测函数 —— 对读取器使用泛型
fn check_overtemp(reader: &impl SensorReader, ids: &[u32], threshold: f64) -> Vec<u32> {
    ids.iter()
        .filter(|&&id| reader.read_temperature(id).unwrap_or(0.0) > threshold)
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_overtemp_sensors() {
        let mut mock = MockSensorReader { temperatures: Default::default() };
        mock.temperatures.insert(0, 72.5);
        mock.temperatures.insert(1, 91.0);  // 超过阈值
        mock.temperatures.insert(2, 65.0);

        let hot = check_overtemp(&mock, &[0, 1, 2], 80.0);
        assert_eq!(hot, vec![1]);
    }
}
```

### 测试中的临时文件和目录

C++ 测试通常使用平台特定的临时目录。Rust 有 `tempfile`：

```rust
// Cargo.toml: [dev-dependencies]
// tempfile = "3"

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    use std::io::Write;

    #[test]
    fn parse_config_from_file() -> Result<(), Box<dyn std::error::Error>> {
        // 创建一个 drop 时自动删除的临时文件
        let mut file = NamedTempFile::new()?;
        writeln!(file, r#"{{"sku": "ServerNode", "level": "Quick"}}"#)?;

        let config = load_config(file.path().to_str().unwrap())?;
        assert_eq!(config.sku, "ServerNode");
        Ok(())
        // file 在这里被删除 —— 不需要清理代码
    }
}
```

### 使用 `proptest` 进行基于属性的测试

不是编写特定的测试用例，而是描述对所有输入都应成立的**属性**。`proptest` 生成随机输入并找到最小的失败用例：

```rust
// Cargo.toml: [dev-dependencies]
// proptest = "1"

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    fn parse_and_format(n: u32) -> String {
        format!("{n}")
    }

    proptest! {
        #[test]
        fn roundtrip_u32(n: u32) {
            let formatted = parse_and_format(n);
            let parsed: u32 = formatted.parse().unwrap();
            prop_assert_eq!(n, parsed);
        }

        #[test]
        fn string_contains_no_null(s in "[a-zA-Z0-9 ]{0,100}") {
            prop_assert!(!s.contains('\0'));
        }
    }
}
```

### 使用 `insta` 进行快照测试

对于产生复杂输出（JSON、格式化字符串）的测试，`insta` 自动生成和管理参考快照：

```rust
// Cargo.toml: [dev-dependencies]
// insta = { version = "1", features = ["json"] }

#[cfg(test)]
mod tests {
    use insta::assert_json_snapshot;

    #[test]
    fn der_entry_format() {
        let entry = DerEntry {
            fault_code: 67956,
            component: "GPU".to_string(),
            message: "ECC error detected".to_string(),
        };
        // 首次运行：在 tests/snapshots/ 中创建快照文件
        // 后续运行：与保存的快照进行比较
        assert_json_snapshot!(entry);
    }
}
```

```bash
cargo insta test              # 运行测试并审查新的/更改的快照
cargo insta review            # 交互式审查快照更改
```

### C++ 与 Rust 测试对比

| **C++（Google Test）** | **Rust** | **说明** |
|----------------------|---------|----------|
| `TEST(Suite, Name) { }` | `#[test] fn name() { }` | 不需要套件/类层次结构 |
| `ASSERT_EQ(a, b)` | `assert_eq!(a, b)` | 内置宏，不需要框架 |
| `ASSERT_NEAR(a, b, eps)` | `assert!((a - b).abs() < eps)` | 或使用 `approx` crate |
| `EXPECT_THROW(expr, type)` | `#[should_panic(expected = "...")]` | 或用 `catch_unwind` 进行精细控制 |
| `EXPECT_DEATH(expr, "msg")` | `#[should_panic(expected = "msg")]` | |
| `class Fixture : public ::testing::Test` | 构建器函数 + `Default` | 不需要继承 |
| Google Mock `MOCK_METHOD` | 特征 + 测试实现 | 更明确，没有宏魔法 |
| `INSTANTIATE_TEST_SUITE_P`（参数化） | `proptest!` 或宏生成的测试 | |
| `SetUp()` / `TearDown()` | 通过 `Drop` 实现 RAII —— 清理是自动的 | 变量在测试结束时 drop |
| 单独的测试二进制文件 + CMake | `cargo test` —— 零配置 | |
| `ctest --output-on-failure` | `cargo test -- --nocapture` | |

----

### 集成测试：`tests/` 目录

单元测试与代码一起放在 `#[cfg(test)]` 模块中。**集成测试**位于 crate 根目录的单独 `tests/` 目录中，像外部使用者一样测试库的公共 API：

```
my_crate/
├── src/
│   └── lib.rs          # 你的库代码
├── tests/
│   ├── smoke.rs        # 每个 .rs 文件是一个单独的测试二进制文件
│   ├── regression.rs
│   └── common/
│       └── mod.rs      # 共享测试辅助工具（不是测试本身）
└── Cargo.toml
```

```rust
// tests/smoke.rs —— 像外部用户一样测试你的 crate
use my_crate::DiagEngine;  // 只能访问公共 API

#[test]
fn engine_starts_successfully() {
    let engine = DiagEngine::new("test_config.json");
    assert!(engine.is_ok());
}

#[test]
fn engine_rejects_invalid_config() {
    let engine = DiagEngine::new("nonexistent.json");
    assert!(engine.is_err());
}
```

```rust
// tests/common/mod.rs —— 共享辅助工具，不会被编译为测试二进制文件
pub fn setup_test_environment() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("config.json"), r#"{"log_level": "debug"}"#).unwrap();
    dir
}
```

```rust
// tests/regression.rs —— 可以使用共享辅助工具
mod common;

#[test]
fn regression_issue_42() {
    let env = common::setup_test_environment();
    let engine = my_crate::DiagEngine::new(
        env.path().join("config.json").to_str().unwrap()
    );
    assert!(engine.is_ok());
}
```

**运行集成测试：**
```bash
cargo test                          # 运行单元测试和集成测试
cargo test --test smoke             # 仅运行 tests/smoke.rs
cargo test --test regression        # 仅运行 tests/regression.rs
cargo test --lib                    # 仅运行单元测试（跳过集成测试）
```

> **与单元测试的关键区别**：集成测试无法访问私有函数或 `pub(crate)` 项。这迫使你验证公共 API 是否足够 —— 这是一个有价值的设计信号。用 C++ 的术语来说，就像只针对公共头文件进行测试，没有 `friend` 访问权限。

----
