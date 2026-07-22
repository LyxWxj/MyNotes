# Rust 内置类型

> **你将学到：** Rust 的基本类型（`i32`、`u64`、`f64`、`bool`、`char`）、类型推断、显式类型标注，以及它们与 C/C++ 基本类型的对比。没有隐式转换 — Rust 要求显式类型转换。

- Rust 有类型推断功能，但也允许显式指定类型

|  **描述**  |            **类型**            |          **示例**          |
|:---------:|:------------------------------:|:--------------------------:|
| 有符号整数 | i8, i16, i32, i64, i128, isize | -1, 42, 1_00_000, 1_00_000i64 |
| 无符号整数 | u8, u16, u32, u64, u128, usize | 0, 42, 42u32, 42u64           |
| 浮点数     | f32, f64                       | 0.0, 0.42                     |
| Unicode   | char                           | 'a', '$'                      |
| 布尔值     | bool                           | true, false                   |

- Rust 允许在数字之间任意使用 ```_``` 以便于阅读
----
### Rust 类型声明和赋值
- Rust 使用 ```let``` 关键字将值赋给变量。变量的类型可以在 ```:``` 之后可选地指定
```rust
fn main() {
    let x : i32 = 42;
    // These two assignments are logically equivalent
    let y : u32 = 42;
    let z = 42u32;
}
``` 
- 函数参数和返回值（如果有的话）需要显式类型。以下函数接受一个 u8 参数并返回 u32
```rust
fn foo(x : u8) -> u32
{
    return x as u32 * x as u32;
}
```
- 未使用的变量以 ```_``` 为前缀以避免编译器警告
----
# Rust 类型声明和推断
- Rust 可以根据上下文自动推断变量的类型
- [▶ 在 Rust Playground 中试试](https://play.rust-lang.org/)
```rust
fn secret_of_life_u32(x : u32) {
    println!("The u32 secret_of_life is {}", x);
}

fn secret_of_life_u8(x : u8) {
    println!("The u8 secret_of_life is {}", x);
}

fn main() {
    let a = 42; // The let keyword assigns a value; type of a is u32
    let b = 42; // The let keyword assigns a value; inferred type of b is u8
    secret_of_life_u32(a);
    secret_of_life_u8(b);
}
```

# Rust 变量和可变性
- Rust 变量默认是**不可变的**，除非使用 ```mut``` 关键字表示变量是可变的。例如，以下代码将无法编译，除非将 ```let a = 42``` 改为 ```let mut a = 42```
```rust
fn main() {
    let a = 42; // Must be changed to let mut a = 42 to permit the assignment below 
    a = 43;  // Will not compile unless the above is changed
}
```
- Rust 允许重用变量名（遮蔽）
```rust
fn main() {
    let a = 42;
    {
        let a = 43; //OK: Different variable with the same name
    }
    // a = 43; // Not permitted
    let a = 43; // Ok: New variable and assignment
}
```
