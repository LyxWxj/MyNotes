## 用闭包消除赋值金字塔

> **你将学到：** Rust 的基于表达式的语法和闭包如何将 C++ 中深度嵌套的 `if/else` 验证链扁平化为简洁的线性代码。

- C++ 中经常需要多块 `if/else` 链来给变量赋值，尤其是在涉及验证或回退逻辑时。Rust 的基于表达式的语法和闭包可以将这些代码折叠成扁平的线性代码。

### 模式 1：用 `if` 表达式进行元组赋值
```cpp
// C++ —— 三个变量在多块 if/else 链中赋值
uint32_t fault_code;
const char* der_marker;
const char* action;
if (is_c44ad) {
    fault_code = 32709; der_marker = "CSI_WARN"; action = "No action";
} else if (error.is_hardware_error()) {
    fault_code = 67956; der_marker = "CSI_ERR"; action = "Replace GPU";
} else {
    fault_code = 32709; der_marker = "CSI_WARN"; action = "No action";
}
```

```rust
// Rust 等效写法：accel_fieldiag.rs
// 单个表达式同时赋值三个变量：
let (fault_code, der_marker, recommended_action) = if is_c44ad {
    (32709u32, "CSI_WARN", "No action")
} else if error.is_hardware_error() {
    (67956u32, "CSI_ERR", "Replace GPU")
} else {
    (32709u32, "CSI_WARN", "No action")
};
```

### 模式 2：用 IIFE（立即调用函数表达式）处理可失败链
```cpp
// C++ —— JSON 导航的"末日金字塔"
std::string get_part_number(const nlohmann::json& root) {
    if (root.contains("SystemInfo")) {
        auto& sys = root["SystemInfo"];
        if (sys.contains("BaseboardFru")) {
            auto& bb = sys["BaseboardFru"];
            if (bb.contains("ProductPartNumber")) {
                return bb["ProductPartNumber"].get<std::string>();
            }
        }
    }
    return "UNKNOWN";
}
```

```rust
// Rust 等效写法：framework.rs
// 闭包 + ? 运算符将金字塔折叠为线性代码：
let part_number = (|| -> Option<String> {
    let path = self.args.sysinfo.as_ref()?;
    let content = std::fs::read_to_string(path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    let ppn = json
        .get("SystemInfo")?
        .get("BaseboardFru")?
        .get("ProductPartNumber")?
        .as_str()?;
    Some(ppn.to_string())
})()
.unwrap_or_else(|| "UNKNOWN".to_string());
```
闭包创建了一个 `Option<String>` 作用域，其中 `?` 在任何步骤都可以提前退出。`.unwrap_or_else()` 在末尾提供一次性的回退值。

### 模式 3：迭代器链替代手动循环 + push_back
```cpp
// C++ —— 带中间变量的手动循环
std::vector<std::tuple<std::vector<std::string>, std::string, std::string>> gpu_info;
for (const auto& [key, info] : gpu_pcie_map) {
    std::vector<std::string> bdfs;
    // ... 将 bdf_path 解析为 bdfs
    std::string serial = info.serial_number.value_or("UNKNOWN");
    std::string model = info.model_number.value_or(model_name);
    gpu_info.push_back({bdfs, serial, model});
}
```

```rust
// Rust 等效写法：peripherals.rs
// 单条链：values() → map → collect
let gpu_info: Vec<(Vec<String>, String, String, String)> = self
    .gpu_pcie_map
    .values()
    .map(|info| {
        let bdfs: Vec<String> = info.bdf_path
            .split(')')
            .filter(|s| !s.is_empty())
            .map(|s| s.trim_start_matches('(').to_string())
            .collect();
        let serial = info.serial_number.clone()
            .unwrap_or_else(|| "UNKNOWN".to_string());
        let model = info.model_number.clone()
            .unwrap_or_else(|| model_name.to_string());
        let gpu_bdf = format!("{}:{}:{}.{}",
            info.bdf.segment, info.bdf.bus, info.bdf.device, info.bdf.function);
        (bdfs, serial, model, gpu_bdf)
    })
    .collect();
```

### 模式 4：`.filter().collect()` 替代循环 + `if (condition) continue`
```cpp
// C++
std::vector<TestResult*> failures;
for (auto& t : test_results) {
    if (!t.is_pass()) {
        failures.push_back(&t);
    }
}
```

```rust
// Rust —— 来自 accel_diag/src/healthcheck.rs
pub fn failed_tests(&self) -> Vec<&TestResult> {
    self.test_results.iter().filter(|t| !t.is_pass()).collect()
}
```

### 总结：何时使用哪种模式
| **C++ 模式** | **Rust 替代方案** | **核心优势** |
|----------------|---------------------|-----------------|
| 多块变量赋值 | `let (a, b) = if ... { } else { };` | 所有变量原子性绑定 |
| 嵌套 `if (contains)` 金字塔 | IIFE 闭包 + `?` 运算符 | 线性、扁平、提前退出 |
| `for` 循环 + `push_back` | `.iter().map(\|\|).collect()` | 无需中间可变 Vec |
| `for` + `if (cond) continue` | `.iter().filter(\|\|).collect()` | 声明式意图表达 |
| `for` + `if + break`（查找第一个） | `.iter().find_map(\|\|)` | 搜索 + 转换一步完成 |

----

# 综合练习：诊断事件处理管道

🔴 **挑战** —— 综合枚举、特征、迭代器、错误处理和泛型的集成练习

这个集成练习综合了枚举、特征、迭代器、错误处理和泛型。你将构建一个简化的诊断事件处理管道，类似于生产环境 Rust 代码中使用的模式。

**需求：**
1. 定义一个 `enum Severity { Info, Warning, Critical }` 并实现 `Display`，以及一个 `struct DiagEvent`，包含 `source: String`、`severity: Severity`、`message: String` 和 `fault_code: u32`
2. 定义一个 `trait EventFilter`，包含方法 `fn should_include(&self, event: &DiagEvent) -> bool`
3. 实现两个过滤器：`SeverityFilter`（仅包含 >= 给定严重级别的事件）和 `SourceFilter`（仅包含来自特定来源字符串的事件）
4. 编写函数 `fn process_events(events: &[DiagEvent], filters: &[&dyn EventFilter]) -> Vec<String>`，返回通过**所有**过滤器的事件的格式化报告行
5. 编写函数 `fn parse_event(line: &str) -> Result<DiagEvent, String>`，解析格式为 `"source:severity:fault_code:message"` 的行（输入无效时返回 `Err`）

**起始代码：**
```rust
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    Info,
    Warning,
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

#[derive(Debug, Clone)]
struct DiagEvent {
    source: String,
    severity: Severity,
    message: String,
    fault_code: u32,
}

trait EventFilter {
    fn should_include(&self, event: &DiagEvent) -> bool;
}

struct SeverityFilter {
    min_severity: Severity,
}
// TODO: impl EventFilter for SeverityFilter

struct SourceFilter {
    source: String,
}
// TODO: impl EventFilter for SourceFilter

fn process_events(events: &[DiagEvent], filters: &[&dyn EventFilter]) -> Vec<String> {
    // TODO: 过滤通过所有过滤器的事件，格式化为
    // "[SEVERITY] source (FC:fault_code): message"
    todo!()
}

fn parse_event(line: &str) -> Result<DiagEvent, String> {
    // 解析 "source:severity:fault_code:message"
    // 无效输入返回 Err
    todo!()
}

fn main() {
    let raw_lines = vec![
        "accel_diag:Critical:67956:ECC uncorrectable error detected",
        "nic_diag:Warning:32709:Link speed degraded",
        "accel_diag:Info:10001:Self-test passed",
        "cpu_diag:Critical:55012:Thermal throttling active",
        "accel_diag:Warning:32710:PCIe link width reduced",
    ];

    // 解析所有行，收集成功结果并报告错误
    let events: Vec<DiagEvent> = raw_lines.iter()
        .filter_map(|line| match parse_event(line) {
            Ok(e) => Some(e),
            Err(e) => { eprintln!("Parse error: {e}"); None }
        })
        .collect();

    // 应用过滤器：仅保留来自 accel_diag 的 Critical 和 Warning 事件
    let sev_filter = SeverityFilter { min_severity: Severity::Warning };
    let src_filter = SourceFilter { source: "accel_diag".to_string() };
    let filters: Vec<&dyn EventFilter> = vec![&sev_filter, &src_filter];

    let report = process_events(&events, &filters);
    for line in &report {
        println!("{line}");
    }
    println!("--- {} event(s) matched ---", report.len());
}
```

<details><summary>参考答案（点击展开）</summary>

```rust
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Severity {
    Info,
    Warning,
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Severity::Info => write!(f, "INFO"),
            Severity::Warning => write!(f, "WARNING"),
            Severity::Critical => write!(f, "CRITICAL"),
        }
    }
}

impl Severity {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "Info" => Ok(Severity::Info),
            "Warning" => Ok(Severity::Warning),
            "Critical" => Ok(Severity::Critical),
            other => Err(format!("Unknown severity: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
struct DiagEvent {
    source: String,
    severity: Severity,
    message: String,
    fault_code: u32,
}

trait EventFilter {
    fn should_include(&self, event: &DiagEvent) -> bool;
}

struct SeverityFilter {
    min_severity: Severity,
}

impl EventFilter for SeverityFilter {
    fn should_include(&self, event: &DiagEvent) -> bool {
        event.severity >= self.min_severity
    }
}

struct SourceFilter {
    source: String,
}

impl EventFilter for SourceFilter {
    fn should_include(&self, event: &DiagEvent) -> bool {
        event.source == self.source
    }
}

fn process_events(events: &[DiagEvent], filters: &[&dyn EventFilter]) -> Vec<String> {
    events.iter()
        .filter(|e| filters.iter().all(|f| f.should_include(e)))
        .map(|e| format!("[{}] {} (FC:{}): {}", e.severity, e.source, e.fault_code, e.message))
        .collect()
}

fn parse_event(line: &str) -> Result<DiagEvent, String> {
    let parts: Vec<&str> = line.splitn(4, ':').collect();
    if parts.len() != 4 {
        return Err(format!("Expected 4 colon-separated fields, got {}", parts.len()));
    }
    let fault_code = parts[2].parse::<u32>()
        .map_err(|e| format!("Invalid fault code '{}': {e}", parts[2]))?;
    Ok(DiagEvent {
        source: parts[0].to_string(),
        severity: Severity::from_str(parts[1])?,
        fault_code,
        message: parts[3].to_string(),
    })
}

fn main() {
    let raw_lines = vec![
        "accel_diag:Critical:67956:ECC uncorrectable error detected",
        "nic_diag:Warning:32709:Link speed degraded",
        "accel_diag:Info:10001:Self-test passed",
        "cpu_diag:Critical:55012:Thermal throttling active",
        "accel_diag:Warning:32710:PCIe link width reduced",
    ];

    let events: Vec<DiagEvent> = raw_lines.iter()
        .filter_map(|line| match parse_event(line) {
            Ok(e) => Some(e),
            Err(e) => { eprintln!("Parse error: {e}"); None }
        })
        .collect();

    let sev_filter = SeverityFilter { min_severity: Severity::Warning };
    let src_filter = SourceFilter { source: "accel_diag".to_string() };
    let filters: Vec<&dyn EventFilter> = vec![&sev_filter, &src_filter];

    let report = process_events(&events, &filters);
    for line in &report {
        println!("{line}");
    }
    println!("--- {} event(s) matched ---", report.len());
}
// 输出：
// [CRITICAL] accel_diag (FC:67956): ECC uncorrectable error detected
// [WARNING] accel_diag (FC:32710): PCIe link width reduced
// --- 2 event(s) matched ---
```

</details>

----
