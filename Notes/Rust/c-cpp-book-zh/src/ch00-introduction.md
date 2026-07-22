# 面向 C/C++ 程序员的 Rust 入门课程

## 课程概述
- 课程概述
    - Rust 的优势（分别从 C 和 C++ 的角度阐述）
    - 本地安装
    - 类型、函数、控制流、模式匹配
    - 模块、cargo
    - 特征、泛型
    - 集合、错误处理
    - 闭包、内存管理、生命周期、智能指针
    - 并发
    - 不安全 Rust，包括外部函数接口（FFI）
    - `no_std` 和嵌入式 Rust 基础（面向固件团队）
    - 案例研究：真实世界的 C++ 到 Rust 转换模式
- 本课程不涉及 `async` Rust — 请参阅配套的 [Async Rust 培训](../async-book/)，其中全面介绍了 future、执行器、`Pin`、tokio 以及生产环境中的异步模式


---

# 自学指南

本教材既可用于讲师授课，也适合自学。如果你是自学，以下是一些建议，帮助你充分利用本教材：

**推荐学习进度：**

| 章节 | 主题 | 建议用时 | 检查点 |
|------|------|---------|--------|
| 1–4 | 环境搭建、类型、控制流 | 1 天 | 你能编写一个命令行温度转换器 |
| 5–7 | 数据结构、所有权 | 1–2 天 | 你能解释*为什么* `let s2 = s1` 会使 `s1` 失效 |
| 8–9 | 模块、错误处理 | 1 天 | 你能创建一个多文件项目，并使用 `?` 传播错误 |
| 10–12 | 特征、泛型、闭包 | 1–2 天 | 你能编写一个带有特征约束的泛型函数 |
| 13–14 | 并发、unsafe/FFI | 1 天 | 你能使用 `Arc<Mutex<T>>` 编写一个线程安全的计数器 |
| 15–16 | 深入探索 | 按自己的节奏 | 参考资料 — 需要时再阅读 |
| 17–19 | 最佳实践与参考 | 按自己的节奏 | 编写实际代码时查阅 |

**如何使用练习：**
- 每个章节都有标注难度的动手练习：🟢 入门、🟡 中级、🔴 挑战
- **在查看解答之前，一定要先自己尝试练习。** 与借用检查器的斗争是学习过程的一部分 — 编译器的错误信息就是你的老师
- 如果卡住超过 15 分钟，可以展开查看解答，研究它，然后关闭解答从头再试一次
- [Rust Playground](https://play.rust-lang.org/) 让你无需本地安装就能运行代码

**遇到困难时：**
- 仔细阅读编译器的错误信息 — Rust 的错误信息非常有帮助
- 重新阅读相关章节；像所有权（第 7 章）这样的概念，往往在第二遍阅读时才会恍然大悟
- [Rust 标准库文档](https://doc.rust-lang.org/std/) 非常优秀 — 可以搜索任何类型或方法
- 关于异步模式，请参阅配套的 [Async Rust 培训](../async-book/)

---

# 目录

## 第一部分 — 基础

### 1. 介绍与动机
- [讲师介绍和总体方法](ch01-introduction-and-motivation.md#speaker-intro-and-general-approach)
- [Rust 的优势](ch01-introduction-and-motivation.md#the-case-for-rust)
- [Rust 如何解决这些问题？](ch01-introduction-and-motivation.md#how-does-rust-address-these-issues)
- [Rust 的其他独特卖点和特性](ch01-introduction-and-motivation.md#other-rust-usps-and-features)
- [快速参考：Rust vs C/C++](ch01-introduction-and-motivation.md#quick-reference-rust-vs-cc)
- [为什么 C/C++ 开发者需要 Rust](ch01-1-why-c-cpp-developers-need-rust.md)
  - [Rust 消除了什么 — 完整列表](ch01-1-why-c-cpp-developers-need-rust.md#what-rust-eliminates--the-complete-list)
  - [C 和 C++ 共有的问题](ch01-1-why-c-cpp-developers-need-rust.md#the-problems-shared-by-c-and-c)
  - [C++ 带来了更多问题](ch01-1-why-c-cpp-developers-need-rust.md#c-adds-more-problems-on-top)
  - [Rust 如何解决所有这些问题](ch01-1-why-c-cpp-developers-need-rust.md#how-rust-addresses-all-of-this)

### 2. 快速入门
- [别说了：给我看代码](ch02-getting-started.md#enough-talk-already-show-me-some-code)
- [Rust 本地安装](ch02-getting-started.md#rust-local-installation)
- [Rust 包（crate）](ch02-getting-started.md#rust-packages-crates)
- [示例：cargo 和 crate](ch02-getting-started.md#example-cargo-and-crates)

### 3. 基本类型和变量
- [Rust 内置类型](ch03-built-in-types.md#built-in-rust-types)
- [Rust 类型声明和赋值](ch03-built-in-types.md#rust-type-specification-and-assignment)
- [Rust 类型声明和推断](ch03-built-in-types.md#rust-type-specification-and-inference)
- [Rust 变量和可变性](ch03-built-in-types.md#rust-variables-and-mutability)

### 4. 控制流
- [Rust if 关键字](ch04-control-flow.md#rust-if-keyword)
- [Rust 使用 while 和 for 循环](ch04-control-flow.md#rust-loops-using-while-and-for)
- [Rust 使用 loop 循环](ch04-control-flow.md#rust-loops-using-loop)
- [Rust 表达式块](ch04-control-flow.md#rust-expression-blocks)

### 5. 数据结构和集合
- [Rust 数组类型](ch05-data-structures.md#rust-array-type)
- [Rust 元组](ch05-data-structures.md#rust-tuples)
- [Rust 引用](ch05-data-structures.md#rust-references)
- [C++ 引用 vs Rust 引用 — 关键区别](ch05-data-structures.md#c-references-vs-rust-references--key-differences)
- [Rust 切片](ch05-data-structures.md#rust-slices)
- [Rust 常量和静态变量](ch05-data-structures.md#rust-constants-and-statics)
- [Rust 字符串：String vs &str](ch05-data-structures.md#rust-strings-string-vs-str)
- [Rust 结构体](ch05-data-structures.md#rust-structs)
- [Rust Vec\<T\>](ch05-data-structures.md#rust-vec-type)
- [Rust HashMap](ch05-data-structures.md#rust-hashmap-type)
- [练习：Vec 和 HashMap](ch05-data-structures.md#exercise-vec-and-hashmap)

### 6. 模式匹配和枚举
- [Rust 枚举类型](ch06-enums-and-pattern-matching.md#rust-enum-types)
- [Rust match 语句](ch06-enums-and-pattern-matching.md#rust-match-statement)
- [练习：使用 match 和枚举实现加法和减法](ch06-enums-and-pattern-matching.md#exercise-implement-add-and-subtract-using-match-and-enum)

### 7. 所有权和内存管理
- [Rust 内存管理](ch07-ownership-and-borrowing.md#rust-memory-management)
- [Rust 所有权、借用和生命周期](ch07-ownership-and-borrowing.md#rust-ownership-borrowing-and-lifetimes)
- [Rust 移动语义](ch07-ownership-and-borrowing.md#rust-move-semantics)
- [Rust Clone](ch07-ownership-and-borrowing.md#rust-clone)
- [Rust Copy 特征](ch07-ownership-and-borrowing.md#rust-copy-trait)
- [Rust Drop 特征](ch07-ownership-and-borrowing.md#rust-drop-trait)
- [练习：移动、Copy 和 Drop](ch07-ownership-and-borrowing.md#exercise-move-copy-and-drop)
- [Rust 生命周期和借用](ch07-1-lifetimes-and-borrowing-deep-dive.md#rust-lifetime-and-borrowing)
- [Rust 生命周期标注](ch07-1-lifetimes-and-borrowing-deep-dive.md#rust-lifetime-annotations)
- [练习：带生命周期的切片存储](ch07-1-lifetimes-and-borrowing-deep-dive.md#exercise-slice-storage-with-lifetimes)
- [生命周期省略规则深入探讨](ch07-1-lifetimes-and-borrowing-deep-dive.md#lifetime-elision-rules-deep-dive)
- [Rust Box\<T\>](ch07-2-smart-pointers-and-interior-mutability.md#rust-boxt)
- [内部可变性：Cell\<T\> 和 RefCell\<T\>](ch07-2-smart-pointers-and-interior-mutability.md#interior-mutability-cellt-and-refcellt)
- [共享所有权：Rc\<T\>](ch07-2-smart-pointers-and-interior-mutability.md#shared-ownership-rct)
- [练习：共享所有权和内部可变性](ch07-2-smart-pointers-and-interior-mutability.md#exercise-shared-ownership-and-interior-mutability)

### 8. 模块和 Crate
- [Rust crate 和模块](ch08-crates-and-modules.md#rust-crates-and-modules)
- [练习：模块和函数](ch08-crates-and-modules.md#exercise-modules-and-functions)
- [工作空间和 crate（包）](ch08-crates-and-modules.md#workspaces-and-crates-packages)
- [练习：使用工作空间和包依赖](ch08-crates-and-modules.md#exercise-using-workspaces-and-package-dependencies)
- [使用 crates.io 上的社区 crate](ch08-crates-and-modules.md#using-community-crates-from-cratesio)
- [Crate 依赖和语义化版本](ch08-crates-and-modules.md#crates-dependencies-and-semver)
- [练习：使用 rand crate](ch08-crates-and-modules.md#exercise-using-the-rand-crate)
- [Cargo.toml 和 Cargo.lock](ch08-crates-and-modules.md#cargotoml-and-cargolock)
- [Cargo 测试功能](ch08-crates-and-modules.md#cargo-test-feature)
- [Cargo 的其他功能](ch08-crates-and-modules.md#other-cargo-features)
- [测试模式](ch08-1-testing-patterns.md)

### 9. 错误处理
- [将枚举与 Option 和 Result 关联起来](ch09-error-handling.md#connecting-enums-to-option-and-result)
- [Rust Option 类型](ch09-error-handling.md#rust-option-type)
- [Rust Result 类型](ch09-error-handling.md#rust-result-type)
- [练习：使用 Option 实现 log() 函数](ch09-error-handling.md#exercise-log-function-implementation-with-option)
- [Rust 错误处理](ch09-error-handling.md#rust-error-handling)
- [练习：错误处理](ch09-error-handling.md#exercise-error-handling)
- [错误处理最佳实践](ch09-1-error-handling-best-practices.md)

### 10. 特征和泛型
- [Rust 特征](ch10-traits.md#rust-traits)
- [C++ 运算符重载 → Rust std::ops 特征](ch10-traits.md#c-operator-overloading--rust-stdops-traits)
- [练习：Logger 特征实现](ch10-traits.md#exercise-logger-trait-implementation)
- [何时使用枚举 vs dyn Trait](ch10-traits.md#when-to-use-enum-vs-dyn-trait)
- [练习：翻译之前先思考](ch10-traits.md#exercise-think-before-you-translate)
- [Rust 泛型](ch10-1-generics.md#rust-generics)
- [练习：泛型](ch10-1-generics.md#exercise-generics)
- [结合 Rust 特征和泛型](ch10-1-generics.md#combining-rust-traits-and-generics)
- [Rust 数据类型中的特征约束](ch10-1-generics.md#rust-traits-constraints-in-data-types)
- [练习：特征约束和泛型](ch10-1-generics.md#exercise-traits-constraints-and-generics)
- [Rust 类型状态模式和泛型](ch10-1-generics.md#rust-type-state-pattern-and-generics)
- [Rust 建造者模式](ch10-1-generics.md#rust-builder-pattern)

### 11. 类型系统高级特性
- [Rust From 和 Into 特征](ch11-from-and-into-traits.md#rust-from-and-into-traits)
- [练习：From 和 Into](ch11-from-and-into-traits.md#exercise-from-and-into)
- [Rust Default 特征](ch11-from-and-into-traits.md#rust-default-trait)
- [Rust 的其他类型转换](ch11-from-and-into-traits.md#other-rust-type-conversions)

### 12. 函数式编程
- [Rust 闭包](ch12-closures.md#rust-closures)
- [练习：闭包和捕获](ch12-closures.md#exercise-closures-and-capturing)
- [Rust 迭代器](ch12-closures.md#rust-iterators)
- [练习：Rust 迭代器](ch12-closures.md#exercise-rust-iterators)
- [迭代器高级技巧参考](ch12-1-iterator-power-tools.md#iterator-power-tools-reference)

### 13. 并发
- [Rust 并发](ch13-concurrency.md#rust-concurrency)
- [为什么 Rust 能防止数据竞争：Send 和 Sync](ch13-concurrency.md#why-rust-prevents-data-races-send-and-sync)
- [练习：多线程单词计数](ch13-concurrency.md#exercise-multi-threaded-word-count)

### 14. 不安全 Rust 和 FFI
- [不安全 Rust](ch14-unsafe-rust-and-ffi.md#unsafe-rust)
- [简单 FFI 示例](ch14-unsafe-rust-and-ffi.md#simple-ffi-example-rust-library-function-consumed-by-c)
- [复杂 FFI 示例](ch14-unsafe-rust-and-ffi.md#complex-ffi-example)
- [确保 unsafe 代码的正确性](ch14-unsafe-rust-and-ffi.md#ensuring-correctness-of-unsafe-code)
- [练习：编写安全的 FFI 包装器](ch14-unsafe-rust-and-ffi.md#exercise-writing-a-safe-ffi-wrapper)

## 第二部分 — 深入探索

### 15. no_std — 面向裸机的 Rust
- [什么是 no_std？](ch15-no_std-rust-without-the-standard-library.md#what-is-no_std)
- [何时使用 no_std vs std](ch15-no_std-rust-without-the-standard-library.md#when-to-use-no_std-vs-std)
- [练习：no_std 环形缓冲区](ch15-no_std-rust-without-the-standard-library.md#exercise-no_std-ring-buffer)
- [嵌入式深入探讨](ch15-1-embedded-deep-dive.md)

### 16. 案例研究：真实世界的 C++ 到 Rust 转换
- [案例研究 1：继承层次结构 → 枚举分发](ch16-case-studies.md#case-study-1-inheritance-hierarchy--enum-dispatch)
- [案例研究 2：shared_ptr 树 → Arena/索引模式](ch16-case-studies.md#case-study-2-shared_ptr-tree--arenaindex-pattern)
- [案例研究 3：框架通信 → 生命周期借用](ch16-1-case-study-lifetime-borrowing.md#case-study-3-framework-communication--lifetime-borrowing)
- [案例研究 4：上帝对象 → 可组合状态](ch16-1-case-study-lifetime-borrowing.md#case-study-4-god-object--composable-state)
- [案例研究 5：特征对象 — 何时该用](ch16-1-case-study-lifetime-borrowing.md#case-study-5-trait-objects--when-they-are-right)

## 第三部分 — 最佳实践与参考

### 17. 最佳实践
- [Rust 最佳实践总结](ch17-best-practices.md#rust-best-practices-summary)
- [避免过度使用 clone()](ch17-1-avoiding-excessive-clone.md#avoiding-excessive-clone)
- [避免未检查的索引访问](ch17-2-avoiding-unchecked-indexing.md#avoiding-unchecked-indexing)
- [消除赋值金字塔](ch17-3-collapsing-assignment-pyramids.md#collapsing-assignment-pyramids)
- [综合练习：诊断事件管道](ch17-3-collapsing-assignment-pyramids.md#capstone-exercise-diagnostic-event-pipeline)
- [日志与追踪生态系统](ch17-4-logging-and-tracing-ecosystem.md#logging-and-tracing-ecosystem)

### 18. C++ → Rust 语义深入对比
- [类型转换、预处理器、模块、volatile、static、constexpr、SFINAE 等](ch18-cpp-rust-semantic-deep-dives.md)

### 19. Rust 宏
- [声明式宏（`macro_rules!`）](ch19-macros.md#declarative-macros-with-macro_rules)
- [常用标准库宏](ch19-macros.md#common-standard-library-macros)
- [派生宏](ch19-macros.md#derive-macros)
- [属性宏](ch19-macros.md#attribute-macros)
- [过程宏](ch19-macros.md#procedural-macros-conceptual-overview)
- [何时使用什么：宏 vs 函数 vs 泛型](ch19-macros.md#when-to-use-what-macros-vs-functions-vs-generics)
- [练习](ch19-macros.md#exercises)
