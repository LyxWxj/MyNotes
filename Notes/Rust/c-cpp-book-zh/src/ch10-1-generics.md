# Rust 泛型

> **你将学到：** 泛型类型参数、单态化（零开销泛型）、特征约束，以及 Rust 泛型与 C++ 模板的比较 —— 更好的错误信息且没有 SFINAE。

- 泛型允许相同的算法或数据结构在不同数据类型间复用
    - 泛型参数以标识符的形式出现在 ```<>``` 内，例如：```<T>```。参数可以使用任何合法的标识符名称，但通常为了简洁而保持简短
    - 编译器在编译时执行单态化，即为遇到的每种 ```T``` 变体生成一个新类型
```rust
// 返回由类型 <T> 的 left 和 right 组成的 <T> 类型元组
fn pick<T>(x: u32, left: T, right: T) -> (T, T) {
   if x == 42 {
    (left, right) 
   } else {
    (right, left)
   }
}
fn main() {
    let a = pick(42, true, false);
    let b = pick(42, "hello", "world");
    println!("{a:?}, {b:?}");
}
```

# Rust 泛型
- 泛型也可以应用于数据类型和关联方法。可以为特定的 ```<T>``` 特化实现（例如：```f32``` vs. ```u32```）
```rust
#[derive(Debug)] // 我们稍后会讨论这个
struct Point<T> {
    x : T,
    y : T,
}
impl<T> Point<T> {
    fn new(x: T, y: T) -> Self {
        Point {x, y}
    }
    fn set_x(&mut self, x: T) {
         self.x = x;       
    }
    fn set_y(&mut self, y: T) {
         self.y = y;       
    }
}
impl Point<f32> {
    fn is_secret(&self) -> bool {
        self.x == 42.0
    }    
}
fn main() {
    let mut p = Point::new(2, 4); // i32
    let q = Point::new(2.0, 4.0); // f32
    p.set_x(42);
    p.set_y(43);
    println!("{p:?} {q:?} {}", q.is_secret());
}
```

# 练习：泛型

🟢 **入门**
- 修改 ```Point``` 类型，为 x 和 y 使用两种不同的类型（```T``` 和 ```U```）

<details><summary>答案（点击展开）</summary>

```rust
#[derive(Debug)]
struct Point<T, U> {
    x: T,
    y: U,
}

impl<T, U> Point<T, U> {
    fn new(x: T, y: U) -> Self {
        Point { x, y }
    }
}

fn main() {
    let p1 = Point::new(42, 3.14);        // Point<i32, f64>
    let p2 = Point::new("hello", true);   // Point<&str, bool>
    let p3 = Point::new(1u8, 1000u64);    // Point<u8, u64>
    println!("{p1:?}");
    println!("{p2:?}");
    println!("{p3:?}");
}
// 输出：
// Point { x: 42, y: 3.14 }
// Point { x: "hello", y: true }
// Point { x: 1, y: 1000 }
```

</details>

### 组合 Rust 特征和泛型
- 特征可用于对泛型类型施加限制（约束）
- 约束可以使用泛型类型参数后的 ```:``` 指定，或使用 ```where```。以下定义了一个泛型函数 ```get_area```，它接受任何类型 ```T```，只要它实现了 ```ComputeArea``` ```trait```
```rust
    trait ComputeArea {
        fn area(&self) -> u64;
    }
    fn get_area<T: ComputeArea>(t: &T) -> u64 {
        t.area()
    }
```
- [▶ 在 Rust Playground 中试试](https://play.rust-lang.org/)

### 组合 Rust 特征和泛型
- 可以有多个特征约束
```rust
trait Fish {}
trait Mammal {}
struct Shark;
struct Whale;
impl Fish for Shark {}
impl Fish for Whale {}
impl Mammal for Whale {}
fn only_fish_and_mammals<T: Fish + Mammal>(_t: &T) {}
fn main() {
    let w = Whale {};
    only_fish_and_mammals(&w);
    let _s = Shark {};
    // 无法编译
    only_fish_and_mammals(&_s);
}
```

### Rust 数据类型中的特征约束
- 特征约束可以与数据类型中的泛型结合使用
- 在以下示例中，我们定义了 ```PrintDescription``` ```trait``` 和一个泛型 ```struct``` ```Shape```，其成员受特征约束
```rust
trait PrintDescription {
    fn print_description(&self);
}
struct Shape<S: PrintDescription> {
    shape: S,
}
// 对任何实现了 PrintDescription 的类型的泛型 Shape 实现
impl<S: PrintDescription> Shape<S> {
    fn print(&self) {
        self.shape.print_description();
    }
}
```
- [▶ 在 Rust Playground 中试试](https://play.rust-lang.org/)

# 练习：特征约束和泛型

🟡 **中级**
- 实现一个带有泛型成员 ```cipher``` 的 ```struct```，该成员实现 ```CipherText```
```rust
trait CipherText {
    fn encrypt(&self);
}
// 待完成
//struct Cipher<>

```
- 接下来，在 ```struct``` ```impl``` 上实现一个名为 ```encrypt``` 的方法，该方法调用 ```cipher``` 上的 ```encrypt```
```rust
// 待完成
impl for Cipher<> {}
```
- 接下来，在两个名为 ```CipherOne``` 和 ```CipherTwo``` 的结构体上实现 ```CipherText```（只用 ```println()``` 就行）。创建 ```CipherOne``` 和 ```CipherTwo```，并使用 ```Cipher``` 调用它们

<details><summary>答案（点击展开）</summary>

```rust
trait CipherText {
    fn encrypt(&self);
}

struct Cipher<T: CipherText> {
    cipher: T,
}

impl<T: CipherText> Cipher<T> {
    fn encrypt(&self) {
        self.cipher.encrypt();
    }
}

struct CipherOne;
struct CipherTwo;

impl CipherText for CipherOne {
    fn encrypt(&self) {
        println!("CipherOne encryption applied");
    }
}

impl CipherText for CipherTwo {
    fn encrypt(&self) {
        println!("CipherTwo encryption applied");
    }
}

fn main() {
    let c1 = Cipher { cipher: CipherOne };
    let c2 = Cipher { cipher: CipherTwo };
    c1.encrypt();
    c2.encrypt();
}
// 输出：
// CipherOne encryption applied
// CipherTwo encryption applied
```

</details>

### Rust 类型状态模式和泛型
- Rust 类型可用于在*编译时*强制执行状态机转换
    - 考虑一个有两个状态的 ```Drone```：```Idle``` 和 ```Flying```。在 ```Idle``` 状态下，唯一允许的方法是 ```takeoff()```。在 ```Flying``` 状态下，允许 ```land()```
    
- 一种方法是使用类似以下的方式建模状态机
```rust
enum DroneState {
    Idle,
    Flying
}
struct Drone {x: u64, y: u64, z: u64, state: DroneState}  // x、y、z 是坐标
```
- 这需要大量运行时检查来强制执行状态机语义 —— [▶ 试试看](https://play.rust-lang.org/) 了解原因

### Rust 类型状态模式泛型
- 泛型允许我们在*编译时*强制执行状态机。这需要使用一种叫做 ```PhantomData<T>``` 的特殊泛型
- ```PhantomData<T>``` 是一种 ```零大小``` 的标记数据类型。在本例中，我们用它来表示 ```Idle``` 和 ```Flying``` 状态，但它具有 ```零``` 运行时大小
- 注意 ```takeoff``` 和 ```land``` 方法接受 ```self``` 作为参数。这被称为 ```消耗```（与使用借用的 ```&self``` 对比）。基本上，一旦我们在 ```Drone<Idle>``` 上调用 ```takeoff()```，我们只能得到一个 ```Drone<Flying>```，反之亦然
```rust
struct Drone<T> {x: u64, y: u64, z: u64, state: PhantomData<T> }
impl Drone<Idle> {
    fn takeoff(self) -> Drone<Flying> {...}
}
impl Drone<Flying> {
    fn land(self) -> Drone<Idle> { ...}
}
```
    - [▶ 在 Rust Playground 中试试](https://play.rust-lang.org/)

### Rust 类型状态模式泛型
- 关键要点：
    - 状态可以使用结构体表示（零大小）
    - 我们可以将状态 ```T``` 与 ```PhantomData<T>```（零大小）结合
    - 为状态机的特定阶段实现方法现在只需要 ```impl State<T>```
    - 使用消耗 ```self``` 的方法从一个状态转换到另一个状态
    - 这给我们提供了 ```零开销``` 抽象。编译器可以在编译时强制执行状态机，除非状态正确否则不可能调用方法

### Rust 构建器模式
- 消耗 ```self``` 对构建器模式很有用
- 考虑一个有几十个引脚的 GPIO 配置。引脚可以配置为高电平或低电平（默认是低电平）
```rust
#[derive(default)]
enum PinState {
    #[default]
    Low,
    High,
} 
#[derive(default)]
struct GPIOConfig {
    pin0: PinState,
    pin1: PinState
    ... 
}
```
- 构建器模式可以通过链式调用来构建 GPIO 配置 —— [▶ 试试看](https://play.rust-lang.org/)
