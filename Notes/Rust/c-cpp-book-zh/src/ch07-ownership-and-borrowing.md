# Rust 内存管理

> **你将学到：** Rust 的所有权系统 — 这门语言中最重要的概念。学完本章后，你将理解移动语义、借用规则和 `Drop` 特征。如果你掌握了本章，Rust 的其余部分就水到渠成了。如果你感到困难，请重新阅读 — 对大多数 C/C++ 开发者来说，所有权在第二遍阅读时才会恍然大悟。

- C/C++ 中的内存管理是 bug 的来源：
    - 在 C 中：内存通过 `malloc()` 分配，通过 `free()` 释放。没有对悬垂指针、释放后使用或重复释放的检查
    - 在 C++ 中：RAII（资源获取即初始化）和智能指针有所帮助，但 `std::move(ptr)` 在移动后仍能编译 — 使用后移动是未定义行为
- Rust 使 RAII **万无一失**：
    - 移动是**破坏性的** — 编译器拒绝让你触碰被移动的变量
    - 不需要五法则（不需要拷贝构造函数、移动构造函数、拷贝赋值、移动赋值、析构函数）
    - Rust 提供对内存分配的完全控制，但在**编译期**强制保证安全
    - 这通过所有权、借用、可变性和生命周期等机制的组合来实现
    - Rust 运行时分配可以发生在栈和堆上

> **对于 C++ 开发者 — 智能指针对照表：**
>
> | **C++** | **Rust** | **安全改进** |
> |---------|----------|-------------|
> | `std::unique_ptr<T>` | `Box<T>` | 不可能发生使用后移动 |
> | `std::shared_ptr<T>` | `Rc<T>`（单线程） | 默认无引用循环 |
> | `std::shared_ptr<T>`（线程安全） | `Arc<T>` | 显式的线程安全 |
> | `std::weak_ptr<T>` | `Weak<T>` | 必须检查有效性 |
> | 原始指针 | `*const T` / `*mut T` | 仅在 `unsafe` 块中 |
>
> 对于 C 开发者：`Box<T>` 取代了 `malloc`/`free` 配对。`Rc<T>` 取代了手动引用计数。原始指针存在但仅限于 `unsafe` 块中。

# Rust 所有权、借用和生命周期
- 回忆一下，Rust 只允许对一个变量有一个可变引用和多个只读引用
    - 变量的初始声明建立了```所有权```
    - 后续的引用从原始所有者```借用```。规则是借用的作用域永远不能超过拥有者的作用域。换句话说，借用的```生命周期```不能超过拥有者的生命周期
```rust
fn main() {
    let a = 42; // Owner
    let b = &a; // First borrow
    {
        let aa = 42;
        let c = &a; // Second borrow; a is still in scope
        // Ok: c goes out of scope here
        // aa goes out of scope here
    }
    // let d = &aa; // Will not compile unless aa is moved to outside scope
    // b implicitly goes out of scope before a
    // a goes out of scope last
}
```

- Rust 可以通过几种不同的机制向方法传递参数
    - 按值（拷贝）：通常用于可以简单拷贝的类型（如：u8、u32、i8、i32）
    - 按引用：这等同于传递指向实际值的指针。这也通常被称为借用，引用可以是不可变的（```&```）或可变的（```&mut```）
    - 通过移动：这将值的"所有权"转移给函数。调用者不能再引用原始值
```rust
fn foo(x: &u32) {
    println!("{x}");
}
fn bar(x: u32) {
    println!("{x}");
}
fn main() {
    let a = 42;
    foo(&a);    // By reference
    bar(a);     // By value (copy)
}
```

- Rust 禁止方法返回悬垂引用
    - 方法返回的引用必须仍在作用域内
    - Rust 会在引用超出作用域时自动 ```drop``` 它
```rust
fn no_dangling() -> &u32 {
    // lifetime of a begins here
    let a = 42;
    // Won't compile. lifetime of a ends here
    &a
}

fn ok_reference(a: &u32) -> &u32 {
    // Ok because the lifetime of a always exceeds ok_reference()
    a
}
fn main() {
    let a = 42;     // lifetime of a begins here
    let b = ok_reference(&a);
    // lifetime of b ends here
    // lifetime of a ends here
}
```

# Rust 移动语义
- 默认情况下，Rust 赋值会转移所有权
```rust
fn main() {
    let s = String::from("Rust");    // Allocate a string from the heap
    let s1 = s; // Transfer ownership to s1. s is invalid at this point
    println!("{s1}");
    // This will not compile
    //println!("{s}");
    // s1 goes out of scope here and the memory is deallocated
    // s goes out of scope here, but nothing happens because it doesn't own anything
}
```
```mermaid
graph LR
    subgraph "Before: let s1 = s"
        S["s (stack)<br/>ptr"] -->|"owns"| H1["Heap: R u s t"]
    end

    subgraph "After: let s1 = s"
        S_MOVED["s (stack)<br/>⚠️ MOVED"] -.->|"invalid"| H2["Heap: R u s t"]
        S1["s1 (stack)<br/>ptr"] -->|"now owns"| H2
    end

    style S_MOVED fill:#ff6b6b,color:#000,stroke:#333
    style S1 fill:#51cf66,color:#000,stroke:#333
    style H2 fill:#91e5a3,color:#000,stroke:#333
```
*`let s1 = s` 之后，所有权转移到 `s1`。堆数据保持不动 — 只有栈指针移动了。`s` 现在无效了。*

----
# Rust 移动语义和借用
```rust
fn foo(s : String) {
    println!("{s}");
    // The heap memory pointed to by s will be deallocated here
}
fn bar(s : &String) {
    println!("{s}");
    // Nothing happens -- s is borrowed
}
fn main() {
    let s = String::from("Rust string move example");    // Allocate a string from the heap
    foo(s); // Transfers ownership; s is invalid now
    // println!("{s}");  // will not compile
    let t = String::from("Rust string borrow example");
    bar(&t);    // t continues to hold ownership
    println!("{t}"); 
}
```

# Rust 移动语义和所有权
- 可以通过移动来转移所有权
    - 在移动完成后引用未释放的引用是非法的
    - 如果不希望移动，请考虑使用借用
```rust
struct Point {
    x: u32,
    y: u32,
}
fn consume_point(p: Point) {
    println!("{} {}", p.x, p.y);
}
fn borrow_point(p: &Point) {
    println!("{} {}", p.x, p.y);
}
fn main() {
    let p = Point {x: 10, y: 20};
    // Try flipping the two lines
    borrow_point(&p);
    consume_point(p);
}
```

# Rust Clone
- ```clone()``` 方法可用于拷贝原始内存。原始引用继续有效（缺点是我们有了 2 倍的分配）
```rust
fn main() {
    let s = String::from("Rust");    // Allocate a string from the heap
    let s1 = s.clone(); // Copy the string; creates a new allocation on the heap
    println!("{s1}");  
    println!("{s}");
    // s1 goes out of scope here and the memory is deallocated
    // s goes out of scope here, and the memory is deallocated
}
```
```mermaid
graph LR
    subgraph "After: let s1 = s.clone()"
        S["s (stack)<br/>ptr"] -->|"owns"| H1["Heap: R u s t"]
        S1["s1 (stack)<br/>ptr"] -->|"owns (copy)"| H2["Heap: R u s t"]
    end

    style S fill:#51cf66,color:#000,stroke:#333
    style S1 fill:#51cf66,color:#000,stroke:#333
    style H1 fill:#91e5a3,color:#000,stroke:#333
    style H2 fill:#91e5a3,color:#000,stroke:#333
```
*`clone()` 创建一个**独立的**堆分配。`s` 和 `s1` 都是有效的 — 各自拥有自己的副本。*

# Rust Copy 特征
- Rust 使用 ```Copy``` 特征为内置类型实现拷贝语义
    - 示例包括 u8、u32、i8、i32 等。拷贝语义使用"按值传递"
    - 用户定义的数据类型可以选择性地通过 ```derive``` 宏自动实现 ```Copy``` 特征来启用```拷贝```语义
    - 编译器将在新赋值后为副本分配空间
```rust
// Try commenting this out to see the change in let p1 = p; below
#[derive(Copy, Clone, Debug)]   // We'll discuss this more later
struct Point{x: u32, y:u32}
fn main() {
    let p = Point {x: 42, y: 40};
    let p1 = p;     // This will perform a copy now instead of move
    println!("p: {p:?}");
    println!("p1: {p:?}");
    let p2 = p1.clone();    // Semantically the same as copy
}
```

# Rust Drop 特征

- Rust 在作用域结束时自动调用 `drop()` 方法
    - `drop` 是一个名为 `Drop` 的泛型特征的一部分。编译器为所有类型提供了一个空操作的默认实现，但类型可以覆盖它。例如，`String` 类型覆盖了它以释放堆分配的内存
    - 对于 C 开发者：这取代了手动 `free()` 调用的需要 — 资源在超出作用域时自动释放（RAII）
- **关键安全性：** 你不能直接调用 `.drop()`（编译器禁止）。相反，使用 `drop(obj)` 将值移入函数，运行其析构函数，并防止任何进一步的使用 — 消除了重复释放的 bug

> **对于 C++ 开发者：** `Drop` 直接对应 C++ 析构函数（`~ClassName()`）：
>
> | | **C++ 析构函数** | **Rust `Drop`** |
> |---|---|---|
> | **语法** | `~MyClass() { ... }` | `impl Drop for MyType { fn drop(&mut self) { ... } }` |
> | **何时调用** | 作用域结束（RAII） | 作用域结束（相同） |
> | **移动时调用** | 源对象留在"有效但未指定"状态 — 析构函数仍然在被移动的对象上运行 | 源对象**消失** — 不会在被移动的值上调用析构函数 |
> | **手动调用** | `obj.~MyClass()`（危险，很少使用） | `drop(obj)`（安全 — 获取所有权，调用 `drop`，防止进一步使用） |
> | **顺序** | 声明的逆序 | 声明的逆序（相同） |
> | **五法则** | 必须管理拷贝构造函数、移动构造函数、拷贝赋值、移动赋值、析构函数 | 只有 `Drop` — 编译器处理移动语义，`Clone` 是可选的 |
> | **需要虚析构函数？** | 是，如果通过基类指针删除 | 否 — 没有继承，所以没有切片问题 |

```rust
struct Point {x: u32, y:u32}

// Equivalent to: ~Point() { printf("Goodbye point x:%u, y:%u\n", x, y); }
impl Drop for Point {
    fn drop(&mut self) {
        println!("Goodbye point x:{}, y:{}", self.x, self.y);
    }
}
fn main() {
    let p = Point{x: 42, y: 42};
    {
        let p1 = Point{x:43, y: 43};
        println!("Exiting inner block");
        // p1.drop() called here — like C++ end-of-scope destructor
    }
    println!("Exiting main");
    // p.drop() called here
}
```

# 练习：移动、Copy 和 Drop

🟡 **中级** — 自由实验；编译器会引导你
- 使用带和不带 ```Copy``` 的 ```Point```（在 ```#[derive(Debug)]``` 中）创建你自己的实验，确保你理解区别。目的是深入理解移动 vs 拷贝的工作方式，所以务必提问
- 为 ```Point``` 实现一个自定义 ```Drop```，在 ```drop``` 中将 x 和 y 设置为 0。这是一种对释放锁和其他资源很有用的模式
```rust
struct Point{x: u32, y: u32}
fn main() {
    // Create Point, assign it to a different variable, create a new scope,
    // pass point to a function, etc.
}
```

<details><summary>解答（点击展开）</summary>

```rust
#[derive(Debug)]
struct Point { x: u32, y: u32 }

impl Drop for Point {
    fn drop(&mut self) {
        println!("Dropping Point({}, {})", self.x, self.y);
        self.x = 0;
        self.y = 0;
        // Note: setting to 0 in drop demonstrates the pattern,
        // but you can't observe these values after drop completes
    }
}

fn consume(p: Point) {
    println!("Consuming: {:?}", p);
    // p is dropped here
}

fn main() {
    let p1 = Point { x: 10, y: 20 };
    let p2 = p1;  // Move — p1 is no longer valid
    // println!("{:?}", p1);  // Won't compile: p1 was moved

    {
        let p3 = Point { x: 30, y: 40 };
        println!("p3 in inner scope: {:?}", p3);
        // p3 is dropped here (end of scope)
    }

    consume(p2);  // p2 is moved into consume and dropped there
    // println!("{:?}", p2);  // Won't compile: p2 was moved

    // Now try: add #[derive(Copy, Clone)] to Point (and remove the Drop impl)
    // and observe how p1 remains valid after let p2 = p1;
}
// Output:
// p3 in inner scope: Point { x: 30, y: 40 }
// Dropping Point(30, 40)
// Consuming: Point { x: 10, y: 20 }
// Dropping Point(10, 20)
```

</details>
