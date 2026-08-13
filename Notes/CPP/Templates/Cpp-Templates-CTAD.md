# C++ 类模板参数推导（CTAD）详解

## 概述

C++17 引入**类模板参数推导**（Class Template Argument Deduction, CTAD），允许从初始化式中自动推导类模板参数，无需显式指定。

```cpp
// C++17 之前
std::pair<int, double> p(1, 3.14);
std::vector<int> v{1, 2, 3};

// C++17 之后
std::pair p(1, 3.14);   // 自动推导为 pair<int, double>
std::vector v{1, 2, 3}; // 自动推导为 vector<int>
```

---

## 1. 基本用法

### 1.1 从构造函数推导

```cpp
template<typename T1, typename T2, typename T3 = T2>
class C {
public:
    C(T1 x, T2 y, T3 z);
};

C c1(22, 44.3, "hi");  // T1=int, T2=double, T3=char const*
C c2(22, 44.3);         // T1=int, T2=T3=double
C c3("hi", "guy");      // T1=T2=T3=char const*
```

### 1.2 不能部分指定再推导

```cpp
C<string> c10("hi", "my", 42);    // ERROR: T2 无法推导
C<> c11(22, 44.3, 42);             // ERROR: T1, T2 未指定
C<string, string> c12("hi", "my"); // OK: T3 有默认值
```

> [!warning] 规则
> 所有参数必须通过**推导**或**默认参数**确定。不能部分显式指定再推导其余参数。

---

## 2. 推导指引（Deduction Guides）

### 2.1 基本语法

```cpp
template<typename T>
class S {
    T a;
public:
    S(T b) : a(b) {}
};

template<typename T> S(T) -> S<T>;  // 推导指引

S x{12};           // OK: S<int>
S y(12);           // OK: S<int>
auto z = S{12};    // OK: S<int>
```

### 2.2 推导指引与函数模板的区别

```cpp
template<typename T> S(T) -> S<T>;
//                    ↑ 看起来像函数，但有重要区别：
```

| 特性 | 函数模板 | 推导指引 |
|------|---------|---------|
| `->` 后面 | 返回类型 | 引导类型（guide type） |
| `auto` 关键字 | 可以有 | 不能有 |
| 名称 | 任意 | 必须是类模板的非限定名 |
| 引导类型 | N/A | 必须是模板标识（template-id） |
| 可以 `explicit` | N/A | ✅ |
| 可以被调用 | ✅ | ❌ 仅为推导所用 |

### 2.3 用于聚合类型

```cpp
template<typename T>
struct A { T val; };

// 没有推导指引时：
A a4 = 42;  // ERROR

// 有了推导指引：
template<typename T> A(T) -> A<T>;
A a4 = {42};  // OK: A<int>
```

> [!warning] 聚合初始化的限制
> 对于聚合类型，初始化式必须是有效的聚合初始化（花括号列表）：

```cpp
A a5(42);  // ERROR: 不是聚合初始化
A a6 = 42; // ERROR: 不是聚合初始化
A a4 = {42}; // OK
```

---

## 3. 隐式推导指引（Implicit Deduction Guides）

### 3.1 自动生成

编译器为类模板的**每个构造函数**自动生成一个隐式推导指引：

```cpp
template<typename T>
class S {
public:
    S(T b) : a(b) {}
};

// 编译器自动生成：
// template<typename T> S(T) -> S<T>;

S x(12);  // OK: S<int>（不需要手动写推导指引）
```

### 3.2 隐式指引的构成

对于构造函数 `S(T b)`：
- 模板参数列表：`typename T`（类模板的参数）
- 函数参数：`(T b)`（从构造函数复制）
- 引导类型：`S<T>`（类模板名称 + 类模板参数）

### 3.3 花括号初始化的歧义

```cpp
std::vector v{1, 2, 3};  // vector<int>
std::vector w2{v, v};     // vector<vector<int>>
std::vector w1{v};        // vector<int>！不是 vector<vector<int>>！
```

> [!warning] 单元素花括号的歧义
> 委员会决定：单元素 `{v}` 认为是「用 v 初始化」而非「列表中只有 v」。这在泛型代码中容易出问题：

```cpp
template<typename T, typename... Ts>
auto f(T p, Ts... ps) {
    std::vector v{p, ps...}; // ps 空 → vector<T>
                             // ps 非空 → vector<T, Ts...> 类型不同！
}
```

### 3.4 隐式指引可能破坏库代码

```cpp
// 原始版本
template<typename T>
class S {
public:
    S(T b) : a(b) {}
};
// 隐式指引：template<typename T> S(T) -> S<T>;  ✅

// 库作者修改后
template<typename T>
class S {
public:
    using ArgType = typename ValueArg<T>::Type;
    S(ArgType b) : a(b) {}
};
// 隐式指引：template<typename T> S(typename ValueArg<T>::Type) -> S<T>;
// 嵌套依赖类型无法推导！❌
```

> [!danger] 库作者注意事项
> 修改构造函数参数类型时，可能意外破坏用户的 CTAD 代码。

---

## 4. 注入类名（Injected Class Name）

```cpp
template<typename T> struct X {
    template<typename Iter> X(Iter b, Iter e);
    template<typename Iter> auto f(Iter b, Iter e) {
        return X(b, e);  // X 是注入类名
    }
};
```

**问题：** `X` 在类内部既是注入类名（等价于 `X<T>`），又是 CTAD 占位符。

**解决方案：** 当模板名是注入类名时，**禁用 CTAD**，保持历史行为。

---

## 5. 转发引用的特殊处理

### 5.1 问题

```cpp
template<typename T>
struct Y {
    Y(T const&);
    Y(T&&);       // T 是类模板参数，不是转发引用
};

Y y = s;  // s 是 string 左值
```

隐式推导指引：

```cpp
template<typename T> Y(T const&) -> Y<T>;  // #1
template<typename T> Y(T&&) -> Y<T>;       // #2
```

指引 #2 中 `T` 是指引的模板参数，`T&&` 变成转发引用！按正常规则，s 是左值 → T = string& → Y<string&> → 悬空引用！

### 5.2 解决方案

当 `T` 是**类模板参数**时，隐式推导指引**禁用转发引用的特殊推导规则**。`T&&` 按普通右值引用处理。

---

## 6. explicit 推导指引

```cpp
template<typename T> Z(T const&) -> Z<T, T&>;       // #1
template<typename T> explicit Z(T&&) -> Z<T, T>;     // #2

Z z1 = 1;   // 复制初始化，不能用 #2 → Z<int, int&>
Z z2{2};    // 直接初始化，可以用 #2 → Z<int, int>
Z z3(3);    // 直接初始化，可以用 #2 → Z<int, int>
```

---

## 7. 复制构造与初始化列表

```cpp
template<typename... Ts> struct Tuple {
    Tuple(Ts...);
    Tuple(Tuple<Ts...> const&);
};

auto x = Tuple{1, 2};  // Tuple<int, int>

Tuple a = x;   // 复制构造 → Tuple<int, int>
Tuple b(x);    // 复制构造 → Tuple<int, int>
Tuple d{x};    // 复制构造 → Tuple<int, int>
```

> [!tip] 优先匹配
> 编译器优先选择更匹配的推导指引。复制构造的指引比用 `x` 作为普通参数的指引更匹配。

---

## 8. 推导指引仅为推导所用

```cpp
template<typename T> struct X { ... };
template<typename T> struct Y {
    Y(X<T> const&);
    Y(X<T>&&);
};
template<typename T> Y(X<T>) -> Y<T>;  // 按值传递
```

推导指引**不是函数模板**，不能被调用。参数是传值还是传引用**不影响推导结果**——它只用于推导 `T`，实际构造时再根据 `Y` 的构造函数选择重载。

---

## 9. auto 与 CTAD 的区别

| 特性 | auto | CTAD |
|------|------|------|
| 推导机制 | 模板参数推导规则 | 推导指引 |
| 适用范围 | 任何类型 | 类模板 |
| 衰变 | ✅ 按值衰变 | 取决于构造函数参数 |
| 多变量 | `auto a=1, b=2.0;` ❌ | `pair a(1, 2.0);` ✅ |

```cpp
auto x = {1, 2, 3};          // std::initializer_list<int>
auto x = std::vector{1, 2, 3}; // vector<int>（CTAD）

std::pair a(1, 2.0);  // pair<int, double>
// auto a = (1, 2.0);  // ERROR
```

---

## 10. 完整示例

```cpp
#include <iostream>
#include <string>

// 类模板
template<typename T>
class Box {
    T value;
public:
    Box(T v) : value(v) {}
    T get() const { return value; }
};

// 推导指引（可选，通常隐式生成就够了）
template<typename T> Box(T) -> Box<T>;

// 聚合类型
template<typename T>
struct Pair {
    T first, second;
};

template<typename T> Pair(T, T) -> Pair<T>;

int main() {
    Box b1(42);              // Box<int>
    Box b2("hello");         // Box<const char*>
    Box b3{std::string("x")}; // Box<std::string>

    Pair p1{1, 2};           // Pair<int>
    Pair p2(3.14, 2.71);     // Pair<double>

    std::cout << b1.get() << '\n';
    std::cout << p1.first << ", " << p1.second << '\n';
}
```

---

## 11. 速查表

| 我想要... | 使用... |
|----------|---------|
| 从构造函数自动推导 | CTAD（默认可用） |
| 为聚合类型启用 CTAD | 手写推导指引 |
| 限制只在直接初始化时使用 | `explicit` 推导指引 |
| 类内部使用自己的名字 | 注入类名（禁用 CTAD） |
| 避免 `T&&` 被推导为引用 | 类模板参数自动禁用特殊规则 |
| 推导指引 vs 函数模板 | 指引仅为推导，不可调用 |
