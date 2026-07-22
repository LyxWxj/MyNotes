# 案例研究 3：框架通信 → 生命周期借用

> **你将学到：** 如何将 C++ 裸指针框架通信模式转换为 Rust 的基于生命周期的借用系统，在保持零成本抽象的同时消除悬垂指针风险。

## C++ 模式：指向框架的裸指针
```cpp
// C++ original: Every diagnostic module stores a raw pointer to the framework
class DiagBase {
protected:
    DiagFramework* m_pFramework;  // Raw pointer — who owns this?
public:
    DiagBase(DiagFramework* fw) : m_pFramework(fw) {}
    
    void LogEvent(uint32_t code, const std::string& msg) {
        m_pFramework->GetEventLog()->Record(code, msg);  // Hope it's still alive!
    }
};
// Problem: m_pFramework is a raw pointer with no lifetime guarantee
// If framework is destroyed while modules still reference it → UB
```

## Rust 解决方案：带生命周期借用的 DiagContext
```rust
// Example: module.rs — Borrow, don't store

/// Context passed to diagnostic modules during execution.
/// The lifetime 'a guarantees the framework outlives the context.
pub struct DiagContext<'a> {
    pub der_log: &'a mut EventLogManager,
    pub config: &'a ModuleConfig,
    pub framework_opts: &'a HashMap<String, String>,
}

/// Modules receive context as a parameter — never store framework pointers
pub trait DiagModule {
    fn id(&self) -> &str;
    fn execute(&mut self, ctx: &mut DiagContext) -> DiagResult<()>;
    fn pre_execute(&mut self, _ctx: &mut DiagContext) -> DiagResult<()> {
        Ok(())
    }
    fn post_execute(&mut self, _ctx: &mut DiagContext) -> DiagResult<()> {
        Ok(())
    }
}
```

### 关键洞察
- C++ 模块**存储**指向框架的指针（危险：如果框架先被销毁怎么办？）
- Rust 模块**接收**上下文作为函数参数 —— 借用检查器保证在调用期间框架是存活的
- 没有裸指针，没有生命周期歧义，没有"希望它还活着"

----

# 案例研究 4：上帝对象 → 可组合状态

## C++ 模式：单体框架类
```cpp
// C++ original: The framework is god object
class DiagFramework {
    // Health-monitor trap processing
    std::vector<AlertTriggerInfo> m_alertTriggers;
    std::vector<WarnTriggerInfo> m_warnTriggers;
    bool m_healthMonHasBootTimeError;
    uint32_t m_healthMonActionCounter;
    
    // GPU diagnostics
    std::map<uint32_t, GpuPcieInfo> m_gpuPcieMap;
    bool m_isRecoveryContext;
    bool m_healthcheckDetectedDevices;
    // ... 30+ more GPU-related fields
    
    // PCIe tree
    std::shared_ptr<CPcieTreeLinux> m_pPcieTree;
    
    // Event logging
    CEventLogMgr* m_pEventLogMgr;
    
    // ... several other methods
    void HandleGpuEvents();
    void HandleNicEvents();
    void RunGpuDiag();
    // Everything depends on everything
};
```

## Rust 解决方案：可组合的状态结构体
```rust
// Example: main.rs — State decomposed into focused structs

#[derive(Default)]
struct HealthMonitorState {
    alert_triggers: Vec<AlertTriggerInfo>,
    warn_triggers: Vec<WarnTriggerInfo>,
    health_monitor_action_counter: u32,
    health_monitor_has_boot_time_error: bool,
    // Only health-monitor-related fields
}

#[derive(Default)]
struct GpuDiagState {
    gpu_pcie_map: HashMap<u32, GpuPcieInfo>,
    is_recovery_context: bool,
    healthcheck_detected_devices: bool,
    // Only GPU-related fields
}

/// The framework composes these states rather than owning everything flat
struct DiagFramework {
    ctx: DiagContext,             // Execution context
    args: Args,                   // CLI arguments
    pcie_tree: Option<DeviceTree>,  // No shared_ptr needed
    event_log_mgr: EventLogManager,   // Owned, not raw pointer
    fc_manager: FcManager,        // Fault code management
    health: HealthMonitorState,   // Health-monitor state — its own struct
    gpu: GpuDiagState,           // GPU state — its own struct
}
```

### 关键洞察
- **可测试性**：每个状态结构体可以独立进行单元测试
- **可读性**：`self.health.alert_triggers` vs `m_alertTriggers` —— 所有权清晰
- **无畏重构**：修改 `GpuDiagState` 不会意外影响健康监控处理
- **没有方法泥潭**：只需要健康监控状态的函数接收 `&mut HealthMonitorState`，而不是整个框架

----

# 案例研究 5：特征对象 —— 什么时候它们是正确的选择

- 不是所有东西都应该用枚举！**诊断模块插件系统**是特征对象的真正用例
- 为什么？因为诊断模块是**开放扩展的** —— 可以在不修改框架的情况下添加新模块

```rust
// Example: framework.rs — Vec<Box<dyn DiagModule>> is correct here
pub struct DiagFramework {
    modules: Vec<Box<dyn DiagModule>>,        // Runtime polymorphism
    pre_diag_modules: Vec<Box<dyn DiagModule>>,
    event_log_mgr: EventLogManager,
    // ...
}

impl DiagFramework {
    /// Register a diagnostic module — any type implementing DiagModule
    pub fn register_module(&mut self, module: Box<dyn DiagModule>) {
        info!("Registering module: {}", module.id());
        self.modules.push(module);
    }
}
```

### 何时使用哪种模式

| **用例** | **模式** | **原因** |
|-------------|-----------|--------|
| 编译时已知的固定变体集 | `enum` + `match` | 穷尽检查，无虚表 |
| 硬件事件类型（Degrade、Fatal、Boot……） | `enum GpuEventKind` | 所有变体已知，性能很重要 |
| PCIe 设备类型（GPU、NIC、Switch……） | `enum PcieDeviceKind` | 固定集合，每个变体有不同的数据 |
| 插件/模块系统（开放扩展） | `Box<dyn Trait>` | 无需修改框架即可添加新模块 |
| 测试模拟 | `Box<dyn Trait>` | 注入测试替身 |

### 练习：翻译前先思考
给定这段 C++ 代码：
```cpp
class Shape { public: virtual double area() = 0; };
class Circle : public Shape { double r; double area() override { return 3.14*r*r; } };
class Rect : public Shape { double w, h; double area() override { return w*h; } };
std::vector<std::unique_ptr<Shape>> shapes;
```
**问题**：Rust 翻译应该使用 `enum Shape` 还是 `Vec<Box<dyn Shape>>`？

<details><summary>解答（点击展开）</summary>

**答案**：`enum Shape` —— 因为形状的集合是**封闭的**（编译时已知）。只有在用户可以在运行时添加新形状类型时，才使用 `Box<dyn Shape>`。

```rust
// Correct Rust translation:
enum Shape {
    Circle { r: f64 },
    Rect { w: f64, h: f64 },
}

impl Shape {
    fn area(&self) -> f64 {
        match self {
            Shape::Circle { r } => std::f64::consts::PI * r * r,
            Shape::Rect { w, h } => w * h,
        }
    }
}

fn main() {
    let shapes: Vec<Shape> = vec![
        Shape::Circle { r: 5.0 },
        Shape::Rect { w: 3.0, h: 4.0 },
    ];
    for shape in &shapes {
        println!("Area: {:.2}", shape.area());
    }
}
// Output:
// Area: 78.54
// Area: 12.00
```

</details>

----

# 翻译指标和经验教训

## 我们学到了什么
1. **默认使用枚举分发** —— 在约 10 万行 C++ 中，只有约 25 处使用 `Box<dyn Trait>` 是真正需要的（插件系统、测试模拟）。其余约 900 个虚方法变成了带 match 的枚举
2. **Arena 模式消除引用循环** —— `shared_ptr` 和 `enable_shared_from_this` 是所有权不清晰的症状。先想清楚谁**拥有**数据
3. **传递上下文，不要存储指针** —— 带生命周期约束的 `DiagContext<'a>` 比在每个模块中存储 `Framework*` 更安全、更清晰
4. **分解上帝对象** —— 如果一个结构体有 30 多个字段，它可能是 3-4 个结构体穿了件风衣
5. **编译器是你的结对编程伙伴** —— 约 400 个 `dynamic_cast` 调用意味着约 400 个潜在的运行时失败。Rust 中零个 `dynamic_cast` 等价物意味着零个运行时类型错误

## 最困难的部分
- **生命周期标注**：当你习惯了裸指针时，正确处理借用需要时间 —— 但一旦编译通过，它就是正确的
- **与借用检查器作斗争**：想在两个地方同时使用 `&mut self`。解决方案：将状态分解为独立的结构体
- **抵制字面翻译的诱惑**：到处写 `Vec<Box<dyn Base>>` 的冲动。问自己："这个变体集合是封闭的吗？" → 如果是，使用枚举

## 对 C++ 团队的建议
1. 从一个小的、自包含的模块开始（不要从上帝对象开始）
2. 先翻译数据结构，再翻译行为
3. 让编译器引导你 —— 它的错误消息非常出色
4. 优先使用 `enum` 而不是 `dyn Trait`
5. 使用 [Rust Playground](https://play.rust-lang.org/) 在集成前原型验证模式

----

