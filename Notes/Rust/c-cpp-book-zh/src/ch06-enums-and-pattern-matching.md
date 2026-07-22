# Rust 枚举类型

> **你将学到：** Rust 枚举作为可辨识联合体（做对了的标签联合）、用于穷尽模式匹配的 `match`，以及枚举如何用编译器强制的安全性取代 C++ 类层次结构和 C 标签联合。

- 枚举类型是可辨识联合体，即它们是多种可能不同类型的和类型，带有标识特定变体的标签
    - 对于 C 开发者：Rust 中的枚举可以携带数据（做对了的标签联合 — 编译器跟踪哪个变体处于活动状态）
    - 对于 C++ 开发者：Rust 枚举类似于 `std::variant`，但具有穷尽模式匹配，没有 `std::get` 异常，也没有 `std::visit` 样板代码
    - `enum` 的大小等于最大可能类型的大小。各个变体彼此不相关，可以拥有完全不同的类型
    - `enum` 类型是该语言最强大的特性之一 — 它们取代了 C++ 中的整个类层次结构（更多内容见案例研究）
```rust
fn main() {
    enum Numbers {
        Zero,
        SmallNumber(u8),
        BiggerNumber(u32),
        EvenBiggerNumber(u64),
    }
    let a = Numbers::Zero;
    let b = Numbers::SmallNumber(42);
    let c : Numbers = a; // Ok -- the type of a is Numbers
    let d : Numbers = b; // Ok -- the type of b is Numbers
}
```
----
# Rust match 语句
- Rust 的 ```match``` 相当于 C 的 "switch" 的加强版
    - ```match``` 可用于对简单数据类型、```struct```、```enum``` 进行模式匹配
    - ```match``` 语句必须是穷尽的，即必须覆盖给定 ```type``` 的所有可能情况。```_``` 可用作"其他所有情况"的通配符
    - ```match``` 可以产生一个值，但所有分支（```=>```）必须返回相同类型的值

```rust
fn main() {
    let x = 42;
    // In this case, the _ covers all numbers except the ones explicitly listed
    let is_secret_of_life = match x {
        42 => true, // return type is boolean value
        _ => false, // return type boolean value
        // This won't compile because return type isn't boolean
        // _ => 0  
    };
    println!("{is_secret_of_life}");
}
```

# Rust match 语句
- ```match``` 支持范围、布尔过滤器和 ```if``` 守卫语句
```rust
fn main() {
    let x = 42;
    match x {
        // Note that the =41 ensures the inclusive range
        0..=41 => println!("Less than the secret of life"),
        42 => println!("Secret of life"),
        _ => println!("More than the secret of life"),
    }
    let y = 100;
    match y {
        100 if x == 43 => println!("y is 100% not secret of life"),
        100 if x == 42 => println!("y is 100% secret of life"),
        _ => (),    // Do nothing
    }
}
```

# Rust match 语句
- ```match``` 和 ```enum``` 经常组合使用
    - match 语句可以将包含的值"绑定"到一个变量。如果不关心该值，使用 ```_```
    - ```matches!``` 宏可用于匹配特定变体
```rust
fn main() {
    enum Numbers {
        Zero,
        SmallNumber(u8),
        BiggerNumber(u32),
        EvenBiggerNumber(u64),
    }
    let b = Numbers::SmallNumber(42);
    match b {
        Numbers::Zero => println!("Zero"),
        Numbers::SmallNumber(value) => println!("Small number {value}"),
        Numbers::BiggerNumber(_) | Numbers::EvenBiggerNumber(_) => println!("Some BiggerNumber or EvenBiggerNumber"),
    }
    
    // Boolean test for specific variants
    if matches!(b, Numbers::Zero | Numbers::SmallNumber(_)) {
        println!("Matched Zero or small number");
    }
}
```

# Rust match 语句
- ```match``` 还可以使用解构和切片进行匹配
```rust
fn main() {
    struct Foo {
        x: (u32, bool),
        y: u32
    }
    let f = Foo {x: (42, true), y: 100};
    match f {
        // Capture the value of x into a variable called tuple
        Foo{y: 100, x : tuple} => println!("Matched x: {tuple:?}"),
        _ => ()
    }
    let a = [40, 41, 42];
    match a {
        // Last element of slice must be 42. @ is used to bind the match
        [rest @ .., 42] => println!("{rest:?}"),
        // First element of the slice must be 42. @ is used to bind the match
        [42, rest @ ..] => println!("{rest:?}"),
        _ => (),
    }
}
```

# 练习：使用 match 和枚举实现加法和减法

🟢 **入门**

- 编写一个函数，对无符号 64 位数字实现算术运算
- **第 1 步**：定义一个操作枚举：
```rust
enum Operation {
    Add(u64, u64),
    Subtract(u64, u64),
}
```
- **第 2 步**：定义一个结果枚举：
```rust
enum CalcResult {
    Ok(u64),                    // Successful result
    Invalid(String),            // Error message for invalid operations
}
```
- **第 3 步**：实现 `calculate(op: Operation) -> CalcResult`
    - 对于 Add：返回 Ok(sum)
    - 对于 Subtract：如果第一个数 >= 第二个数则返回 Ok(difference)，否则返回 Invalid("Underflow")
- **提示**：在函数中使用模式匹配：
```rust
match op {
    Operation::Add(a, b) => { /* your code */ },
    Operation::Subtract(a, b) => { /* your code */ },
}
```

<details><summary>解答（点击展开）</summary>

```rust
enum Operation {
    Add(u64, u64),
    Subtract(u64, u64),
}

enum CalcResult {
    Ok(u64),
    Invalid(String),
}

fn calculate(op: Operation) -> CalcResult {
    match op {
        Operation::Add(a, b) => CalcResult::Ok(a + b),
        Operation::Subtract(a, b) => {
            if a >= b {
                CalcResult::Ok(a - b)
            } else {
                CalcResult::Invalid("Underflow".to_string())
            }
        }
    }
}

fn main() {
    match calculate(Operation::Add(10, 20)) {
        CalcResult::Ok(result) => println!("10 + 20 = {result}"),
        CalcResult::Invalid(msg) => println!("Error: {msg}"),
    }
    match calculate(Operation::Subtract(5, 10)) {
        CalcResult::Ok(result) => println!("5 - 10 = {result}"),
        CalcResult::Invalid(msg) => println!("Error: {msg}"),
    }
}
// Output:
// 10 + 20 = 30
// Error: Underflow
```

</details>

# Rust 关联方法
- ```impl``` 可以为 ```struct```、```enum``` 等类型定义关联方法
    - 方法可以选择性地将 ```self``` 作为参数。```self``` 在概念上类似于在 C 中将指向结构体的指针作为第一个参数传递，或 C++ 中的 ```this```
    - 对 ```self``` 的引用可以是不可变的（默认：```&self```）、可变的（```&mut self```）或 ```self```（转移所有权）
    - ```Self``` 关键字可用作类型的简写
```rust
struct Point {x: u32, y: u32}
impl Point {
    fn new(x: u32, y: u32) -> Self {
        Point {x, y}
    }
    fn increment_x(&mut self) {
        self.x += 1;
    }
}
fn main() {
    let mut p = Point::new(10, 20);
    p.increment_x();
}
```

# 练习：Point 的加法和变换

🟡 **中级** — 需要理解方法签名中的移动 vs 借用
- 为 ```Point``` 实现以下关联方法
    - ```add()``` 接受另一个 ```Point``` 并就地递增 x 和 y 值（提示：使用 ```&mut self```）
    - ```transform()``` 消耗一个现有的 ```Point```（提示：使用 ```self```）并通过对 x 和 y 求平方返回一个新的 ```Point```

<details><summary>解答（点击展开）</summary>

```rust
struct Point { x: u32, y: u32 }

impl Point {
    fn new(x: u32, y: u32) -> Self {
        Point { x, y }
    }
    fn add(&mut self, other: &Point) {
        self.x += other.x;
        self.y += other.y;
    }
    fn transform(self) -> Point {
        Point { x: self.x * self.x, y: self.y * self.y }
    }
}

fn main() {
    let mut p1 = Point::new(2, 3);
    let p2 = Point::new(10, 20);
    p1.add(&p2);
    println!("After add: x={}, y={}", p1.x, p1.y);           // x=12, y=23
    let p3 = p1.transform();
    println!("After transform: x={}, y={}", p3.x, p3.y);     // x=144, y=529
    // p1 is no longer accessible — transform() consumed it
}
```

</details>

----
