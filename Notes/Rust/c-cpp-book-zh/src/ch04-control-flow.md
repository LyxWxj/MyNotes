# Rust if 关键字

> **你将学到：** Rust 的控制流结构 — 作为表达式的 `if`/`else`、`loop`/`while`/`for`、`match`，以及它们与 C/C++ 对应结构的区别。关键洞察：大多数 Rust 控制流都返回值。

- 在 Rust 中，```if``` 实际上是一个表达式，即它可以用来赋值，但也可以像语句一样使用。[▶ 试试](https://play.rust-lang.org/)

```rust
fn main() {
    let x = 42;
    if x < 42 {
        println!("Smaller than the secret of life");
    } else if x == 42 {
        println!("Is equal to the secret of life");
    } else {
        println!("Larger than the secret of life");
    }
    let is_secret_of_life = if x == 42 {true} else {false};
    println!("{}", is_secret_of_life);
}
```

# Rust 使用 while 和 for 循环
- ```while``` 关键字可用于在表达式为真时循环
```rust
fn main() {
    let mut x = 40;
    while x != 42 {
        x += 1;
    }
}
```
- ```for``` 关键字可用于遍历范围
```rust
fn main() {
    // Will not print 43; use 40..=43 to include last element
    for x in 40..43 {
        println!("{}", x);
    } 
}
```

# Rust 使用 loop 循环
- ```loop``` 关键字创建一个无限循环，直到遇到 ```break```
```rust
fn main() {
    let mut x = 40;
    // Change the below to 'here: loop to specify optional label for the loop
    loop {
        if x == 42 {
            break; // Use break x; to return the value of x
        }
        x += 1;
    }
}
```
- ```break``` 语句可以包含一个可选的表达式，用于赋值给 ```loop``` 表达式的结果
- ```continue``` 关键字可用于返回到 ```loop``` 的顶部
- 循环标签可与 ```break``` 或 ```continue``` 一起使用，在处理嵌套循环时非常有用

# Rust 表达式块
- Rust 表达式块就是一系列用 ```{}``` 包围的表达式。求值结果就是块中最后一个表达式的值
```rust
fn main() {
    let x = {
        let y = 40;
        y + 2 // Note: ; must be omitted
    };
    // Notice the Python style printing
    println!("{x}");
}
```
- Rust 的风格是利用这一特性来省略函数中的 ```return``` 关键字
```rust
fn is_secret_of_life(x: u32) -> bool {
    // Same as if x == 42 {true} else {false}
    x == 42 // Note: ; must be omitted 
}
fn main() {
    println!("{}", is_secret_of_life(42));
}
```
