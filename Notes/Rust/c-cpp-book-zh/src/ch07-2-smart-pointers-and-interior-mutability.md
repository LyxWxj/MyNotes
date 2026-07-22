# Rust `Box<T>`

> **你将学到：** Rust 的智能指针类型 —— 用于堆分配的 `Box<T>`、用于共享所有权的 `Rc<T>`，以及用于内部可变性的 `Cell<T>` / `RefCell<T>`。这些内容建立在前面章节的所有权和生命周期概念之上。你还会简要了解用于打破引用循环的 `Weak<T>`。

**为什么需要 `Box<T>`？** 在 C 中，你使用 `malloc`/`free` 进行堆分配。在 C++ 中，`std::unique_ptr<T>` 封装了 `new`/`delete`。Rust 的 `Box<T>` 就是等价物 —— 一个堆分配的、单一所有者的指针，当它离开作用域时会自动释放。与 `malloc` 不同，没有可能忘记的配对 `free`。与 `unique_ptr` 不同，不存在移动后使用的问题 —— 编译器完全阻止了这种情况。

**何时使用 `Box` 而非栈分配：**
- 包含的类型很大，你不想在栈上复制它
- 你需要递归类型（例如，包含自身的链表节点）
- 你需要特征对象（`Box<dyn Trait>`）

- ```Box<T>``` 可用于创建指向堆分配类型的指针。无论 ```<T>``` 的类型是什么，指针始终是固定大小的
```rust
fn main() {
    // 创建一个指向堆上整数（值为 42）的指针
    let f = Box::new(42);
    println!("{} {}", *f, f);
    // 克隆一个 Box 会创建一个新的堆分配
    let mut g = f.clone();
    *g = 43;
    println!("{f} {g}");
    // g 和 f 在这里离开作用域并自动释放
}
```
```mermaid
graph LR
    subgraph "Stack"
        F["f: Box&lt;i32&gt;"]
        G["g: Box&lt;i32&gt;"]
    end

    subgraph "Heap"
        HF["42"]
        HG["43"]
    end

    F -->|"owns"| HF
    G -->|"owns (cloned)"| HG

    style F fill:#51cf66,color:#000,stroke:#333
    style G fill:#51cf66,color:#000,stroke:#333
    style HF fill:#91e5a3,color:#000,stroke:#333
    style HG fill:#91e5a3,color:#000,stroke:#333
```

## 所有权和借用可视化

### C/C++ 与 Rust：指针和所有权管理

```c
// C - 手动内存管理，存在潜在问题
void c_pointer_problems() {
    int* ptr1 = malloc(sizeof(int));
    *ptr1 = 42;
    
    int* ptr2 = ptr1;  // 两个指针指向同一内存
    int* ptr3 = ptr1;  // 三个指针指向同一内存
    
    free(ptr1);        // 释放内存
    
    *ptr2 = 43;        // 释放后使用 - 未定义行为！
    *ptr3 = 44;        // 释放后使用 - 未定义行为！
}
```

> **给 C++ 开发者：** 智能指针有所帮助，但不能防止所有问题：
>
> ```cpp
> // C++ - 智能指针有帮助，但不能防止所有问题
> void cpp_pointer_issues() {
>     auto ptr1 = std::make_unique<int>(42);
>     
>     // auto ptr2 = ptr1;  // 编译错误：unique_ptr 不可复制
>     auto ptr2 = std::move(ptr1);  // OK：所有权已转移
>     
>     // 但 C++ 仍然允许移动后使用：
>     // std::cout << *ptr1;  // 能编译！但属于未定义行为！
>     
>     // shared_ptr 别名问题：
>     auto shared1 = std::make_shared<int>(42);
>     auto shared2 = shared1;  // 两者都拥有数据
>     // 谁"真正"拥有它？都不是。到处都有引用计数的开销。
> }
> ```

```rust
// Rust - 所有权系统防止了这些问题
fn rust_ownership_safety() {
    let data = Box::new(42);  // data 拥有堆分配
    
    let moved_data = data;    // 所有权转移到 moved_data
    // data 不再可访问 - 如果使用会导致编译错误
    
    let borrowed = &moved_data;  // 不可变借用
    println!("{}", borrowed);    // 安全使用
    
    // moved_data 离开作用域时自动释放
}
```

```mermaid
graph TD
    subgraph "C/C++ Memory Management Issues"
        CP1["int* ptr1"] --> CM["Heap Memory<br/>value: 42"]
        CP2["int* ptr2"] --> CM
        CP3["int* ptr3"] --> CM
        CF["free(ptr1)"] --> CM_F["[ERROR] Freed Memory"]
        CP2 -.->|"Use after free<br/>Undefined Behavior"| CM_F
        CP3 -.->|"Use after free<br/>Undefined Behavior"| CM_F
    end
    
    subgraph "Rust Ownership System"
        RO1["data: Box<i32>"] --> RM["Heap Memory<br/>value: 42"]
        RO1 -.->|"Move ownership"| RO2["moved_data: Box<i32>"]
        RO2 --> RM
        RO1_X["data: [WARNING] MOVED<br/>Cannot access"]
        RB["&moved_data<br/>Immutable borrow"] -.->|"Safe reference"| RM
        RD["Drop automatically<br/>when out of scope"] --> RM
    end
    
    style CM_F fill:#ff6b6b,color:#000
    style CP2 fill:#ff6b6b,color:#000
    style CP3 fill:#ff6b6b,color:#000
    style RO1_X fill:#ffa07a,color:#000
    style RO2 fill:#51cf66,color:#000
    style RB fill:#91e5a3,color:#000
    style RD fill:#91e5a3,color:#000
```

### 借用规则可视化

```rust
fn borrowing_rules_example() {
    let mut data = vec![1, 2, 3, 4, 5];
    
    // 多个不可变借用 - OK
    let ref1 = &data;
    let ref2 = &data;
    println!("{:?} {:?}", ref1, ref2);  // 两个都可以使用
    
    // 可变借用 - 独占访问
    let ref_mut = &mut data;
    ref_mut.push(6);
    // ref_mut 活跃期间不能使用 ref1 和 ref2
    
    // ref_mut 使用完毕后，不可变借用又可以使用了
    let ref3 = &data;
    println!("{:?}", ref3);
}
```

```mermaid
graph TD
    subgraph "Rust Borrowing Rules"
        D["mut data: Vec<i32>"]
        
        subgraph "Phase 1: Multiple Immutable Borrows [OK]"
            IR1["&data (ref1)"]
            IR2["&data (ref2)"]
            D --> IR1
            D --> IR2
            IR1 -.->|"Read-only access"| MEM1["Memory: [1,2,3,4,5]"]
            IR2 -.->|"Read-only access"| MEM1
        end
        
        subgraph "Phase 2: Exclusive Mutable Borrow [OK]"
            MR["&mut data (ref_mut)"]
            D --> MR
            MR -.->|"Exclusive read/write"| MEM2["Memory: [1,2,3,4,5,6]"]
            BLOCK["[ERROR] Other borrows blocked"]
        end
        
        subgraph "Phase 3: Immutable Borrows Again [OK]"
            IR3["&data (ref3)"]
            D --> IR3
            IR3 -.->|"Read-only access"| MEM3["Memory: [1,2,3,4,5,6]"]
        end
    end
    
    subgraph "What C/C++ Allows (Dangerous)"
        CP["int* ptr"]
        CP2["int* ptr2"]
        CP3["int* ptr3"]
        CP --> CMEM["Same Memory"]
        CP2 --> CMEM
        CP3 --> CMEM
        RACE["[ERROR] Data races possible<br/>[ERROR] Use after free possible"]
    end
    
    style MEM1 fill:#91e5a3,color:#000
    style MEM2 fill:#91e5a3,color:#000
    style MEM3 fill:#91e5a3,color:#000
    style BLOCK fill:#ffa07a,color:#000
    style RACE fill:#ff6b6b,color:#000
    style CMEM fill:#ff6b6b,color:#000
```

---

## 内部可变性：`Cell<T>` 和 `RefCell<T>`

回忆一下，Rust 中变量默认是不可变的。有时候我们希望类型的大部分字段是只读的，同时允许对单个字段进行写访问。

```rust
struct Employee {
    employee_id : u64,   // 这个必须是不可变的
    on_vacation: bool,   // 如果我们想允许对这个字段的写访问，但让 employee_id 不可变呢？
}
```

- 回忆一下，Rust 允许对变量有*单个可变*引用和任意数量的*不可变*引用 —— 在*编译时*强制执行
- 如果我们想传递一个*不可变*的员工向量，*但*允许更新 `on_vacation` 字段，同时确保 `employee_id` 不能被修改呢？

### `Cell<T>` —— 用于 Copy 类型的内部可变性

- `Cell<T>` 提供**内部可变性**，即对原本只读的引用的特定元素进行写访问
- 通过复制值进出来工作（`.get()` 需要 `T: Copy`）

### `RefCell<T>` —— 带运行时借用检查的内部可变性

- `RefCell<T>` 提供了一种适用于引用的变体
    - 在**运行时**而非编译时强制执行 Rust 的借用检查
    - 允许单个*可变*借用，但如果存在任何其他未归还的引用则会**panic**
    - 使用 `.borrow()` 进行不可变访问，使用 `.borrow_mut()` 进行可变访问

### 何时选择 `Cell` 还是 `RefCell`

| 标准 | `Cell<T>` | `RefCell<T>` |
|-----------|-----------|-------------|
| 适用于 | `Copy` 类型（整数、布尔值、浮点数） | 任何类型（`String`、`Vec`、结构体） |
| 访问模式 | 复制值进出（`.get()`、`.set()`） | 就地借用（`.borrow()`、`.borrow_mut()`） |
| 失败模式 | 不会失败 —— 无运行时检查 | 如果在另一个借用活跃时进行可变借用会 **panic** |
| 开销 | 零 —— 只是复制字节 | 小 —— 在运行时跟踪借用状态 |
| 使用场景 | 需要在不可变结构体内部放置可变的标志、计数器或小值 | 需要在不可变结构体内部修改 `String`、`Vec` 或复杂类型 |

---

## 共享所有权：`Rc<T>`

`Rc<T>` 允许对*不可变*数据进行引用计数的共享所有权。如果我们想在多个地方存储同一个 `Employee` 而不复制呢？

```rust
#[derive(Debug)]
struct Employee {
    employee_id: u64,
}
fn main() {
    let mut us_employees = vec![];
    let mut all_global_employees = Vec::<Employee>::new();
    let employee = Employee { employee_id: 42 };
    us_employees.push(employee);
    // 无法编译 —— employee 已经被移动了
    //all_global_employees.push(employee);
}
```

`Rc<T>` 通过允许共享*不可变*访问来解决这个问题：
- 包含的类型会自动解引用
- 当引用计数归零时，类型会被释放

```rust
use std::rc::Rc;
#[derive(Debug)]
struct Employee {employee_id: u64}
fn main() {
    let mut us_employees = vec![];
    let mut all_global_employees = vec![];
    let employee = Employee { employee_id: 42 };
    let employee_rc = Rc::new(employee);
    us_employees.push(employee_rc.clone());
    all_global_employees.push(employee_rc.clone());
    let employee_one = all_global_employees.get(0); // 共享不可变引用
    for e in us_employees {
        println!("{}", e.employee_id);  // 共享不可变引用
    }
    println!("{employee_one:?}");
}
```

> **给 C++ 开发者：智能指针对应关系**
>
> | C++ 智能指针 | Rust 等价物 | 关键区别 |
> |---|---|---|
> | `std::unique_ptr<T>` | `Box<T>` | Rust 的版本是默认选择 —— 移动是语言级别的，不是可选的 |
> | `std::shared_ptr<T>` | `Rc<T>`（单线程）/ `Arc<T>`（多线程） | `Rc` 没有原子操作开销；仅在跨线程共享时使用 `Arc` |
> | `std::weak_ptr<T>` | `Weak<T>`（来自 `Rc::downgrade()` 或 `Arc::downgrade()`） | 相同用途：打破引用循环 |
>
> **关键区别**：在 C++ 中，你*选择*使用智能指针。在 Rust 中，拥有的值（`T`）和借用（`&T`）覆盖了大多数用例 —— 只有在需要堆分配或共享所有权时才使用 `Box`/`Rc`/`Arc`。

### 使用 `Weak<T>` 打破引用循环

`Rc<T>` 使用引用计数 —— 如果两个 `Rc` 值互相指向对方，它们都永远不会被释放（循环）。`Weak<T>` 解决了这个问题：

```rust
use std::rc::{Rc, Weak};

struct Node {
    value: i32,
    parent: Option<Weak<Node>>,  // 弱引用 —— 不会阻止释放
}

fn main() {
    let parent = Rc::new(Node { value: 1, parent: None });
    let child = Rc::new(Node {
        value: 2,
        parent: Some(Rc::downgrade(&parent)),  // 对 parent 的弱引用
    });

    // 要使用 Weak，尝试升级它 —— 返回 Option<Rc<T>>
    if let Some(parent_rc) = child.parent.as_ref().unwrap().upgrade() {
        println!("Parent value: {}", parent_rc.value);
    }
    println!("Parent strong count: {}", Rc::strong_count(&parent)); // 1，不是 2
}
```

> `Weak<T>` 在 [避免过度使用 clone()](ch17-1-avoiding-excessive-clone.md) 中有更深入的介绍。目前的关键要点：**在树/图结构中使用 `Weak` 作为"反向引用"以避免内存泄漏。**

---

## 组合 `Rc` 与内部可变性

真正的威力在于将 `Rc<T>`（共享所有权）与 `Cell<T>` 或 `RefCell<T>`（内部可变性）结合使用。这让多个所有者可以**读取和修改**共享数据：

| 模式 | 使用场景 |
|---------|----------|
| `Rc<RefCell<T>>` | 共享的可变数据（单线程） |
| `Arc<Mutex<T>>` | 共享的可变数据（多线程 —— 参见 [ch13](ch13-concurrency.md)） |
| `Rc<Cell<T>>` | 共享的可变 Copy 类型（简单标志、计数器） |

---

# 练习：共享所有权和内部可变性

🟡 **中级**

- **第 1 部分（Rc）**：创建一个包含 `employee_id: u64` 和 `name: String` 的 `Employee` 结构体。将其放入 `Rc<Employee>` 中，并克隆到两个独立的 `Vec` 中（`us_employees` 和 `global_employees`）。从两个向量中打印以展示它们共享相同的数据。
- **第 2 部分（Cell）**：为 `Employee` 添加一个 `on_vacation: Cell<bool>` 字段。将不可变的 `&Employee` 引用传递给一个函数，并在该函数内部切换 `on_vacation` —— 无需让引用变为可变的。
- **第 3 部分（RefCell）**：将 `name: String` 替换为 `name: RefCell<String>`，编写一个函数通过 `&Employee`（不可变引用）为员工姓名添加后缀。

**起始代码：**
```rust
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Debug)]
struct Employee {
    employee_id: u64,
    name: RefCell<String>,
    on_vacation: Cell<bool>,
}

fn toggle_vacation(emp: &Employee) {
    // TODO: 使用 Cell::set() 翻转 on_vacation
}

fn append_title(emp: &Employee, title: &str) {
    // TODO: 通过 RefCell 可变借用 name 并 push_str 标题
}

fn main() {
    // TODO: 创建一个 employee，用 Rc 包装，克隆到两个 Vec 中，
    // 调用 toggle_vacation 和 append_title，打印结果
}
```

<details><summary>答案（点击展开）</summary>

```rust
use std::cell::{Cell, RefCell};
use std::rc::Rc;

#[derive(Debug)]
struct Employee {
    employee_id: u64,
    name: RefCell<String>,
    on_vacation: Cell<bool>,
}

fn toggle_vacation(emp: &Employee) {
    emp.on_vacation.set(!emp.on_vacation.get());
}

fn append_title(emp: &Employee, title: &str) {
    emp.name.borrow_mut().push_str(title);
}

fn main() {
    let emp = Rc::new(Employee {
        employee_id: 42,
        name: RefCell::new("Alice".to_string()),
        on_vacation: Cell::new(false),
    });

    let mut us_employees = vec![];
    let mut global_employees = vec![];
    us_employees.push(Rc::clone(&emp));
    global_employees.push(Rc::clone(&emp));

    // 通过不可变引用切换休假状态
    toggle_vacation(&emp);
    println!("On vacation: {}", emp.on_vacation.get()); // true

    // 通过不可变引用追加标题
    append_title(&emp, ", Sr. Engineer");
    println!("Name: {}", emp.name.borrow()); // "Alice, Sr. Engineer"

    // 两个 Vec 看到的是相同的数据（Rc 共享所有权）
    println!("US: {:?}", us_employees[0].name.borrow());
    println!("Global: {:?}", global_employees[0].name.borrow());
    println!("Rc strong count: {}", Rc::strong_count(&emp));
}
// 输出：
// On vacation: true
// Name: Alice, Sr. Engineer
// US: "Alice, Sr. Engineer"
// Global: "Alice, Sr. Engineer"
// Rc strong count: 3
```

</details>
