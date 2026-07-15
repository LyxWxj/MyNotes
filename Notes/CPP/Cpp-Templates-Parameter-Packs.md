# C++ 模板参数包（Template Parameter Packs）详解

## 概述

C++11 引入可变参数模板，允许模板接受**任意数量**的参数。参数包是这一特性的核心机制。

---

## 1. 基本语法

### 1.1 模板参数包

```cpp
template<typename... Types>  // Types 是模板参数包
class Tuple {};

Tuple<int, double, char> t;  // 3 个参数
Tuple<> empty;               // 0 个参数
```

### 1.2 函数参数包

```cpp
template<typename... Types>
void print(Types... args) {  // args 是函数参数包
    // ...
}
```

### 1.3 参数包展开

```cpp
template<typename... Types>
void print(Types... args) {
    // args... 展开参数包
    someFunction(args...);
}
```

---

## 2. 参数包的递归处理

### 2.1 经典递归模式

```cpp
// 终止函数
void print() {}

// 递归函数
template<typename T, typename... Types>
void print(T first, Types... rest) {
    std::cout << first << '\n';
    print(rest...);  // 递归调用，参数包缩小
}

// 调用
print(7.5, "hello", std::string("world"));
// 展开过程：
// print(7.5, "hello", "world")  → 打印 7.5，递归
// print("hello", "world")       → 打印 hello，递归
// print("world")                → 打印 world，递归
// print()                       → 终止
```

### 2.2 单参数重载模式

```cpp
// 单参数版本
template<typename T>
void print(T arg) {
    std::cout << arg << '\n';
}

// 多参数版本
template<typename T, typename... Types>
void print(T first, Types... rest) {
    print(first);
    print(rest...);
}
```

> [!tip] 重载解析规则
> 当两个函数模板仅在尾部参数包上不同时，**没有尾部参数包的版本优先**。

---

## 3. sizeof... 操作符

```cpp
template<typename... Types>
void print(Types... args) {
    std::cout << "参数数量: " << sizeof...(Types) << '\n';
    std::cout << "参数数量: " << sizeof...(args) << '\n';
}
```

> [!warning] sizeof... 不能用于避免递归
> `if (sizeof...(args) > 0)` 不能避免递归，因为 `if` 的两个分支在编译时都会实例化。C++17 的 `if constexpr` 可以解决：

```cpp
template<typename T, typename... Types>
void print(T first, Types... rest) {
    std::cout << first;
    if constexpr (sizeof...(rest) > 0) {
        std::cout << ", ";
        print(rest...);
    }
}
```

---

## 4. 折叠表达式（Fold Expressions，C++17）

### 4.1 四种形式

| 形式 | 名称 | 语法 | 展开 |
|------|------|------|------|
| 一元左折叠 | Unary left fold | `(... op pack)` | `((p1 op p2) op p3) ...` |
| 一元右折叠 | Unary right fold | `(pack op ...)` | `p1 op (p2 op (... op pN))` |
| 二元左折叠 | Binary left fold | `(init op ... op pack)` | `(((init op p1) op p2) op p3) ...` |
| 二元右折叠 | Binary right fold | `(pack op ... op init)` | `p1 op (p2 op (... op (pN op init)))` |

### 4.2 基本示例

```cpp
// 求和
template<typename... T>
auto sum(T... s) {
    return (s + ...);  // 一元右折叠
}

// 带初始值
template<typename... T>
auto sum(T... s) {
    return (0 + ... + s);  // 二元左折叠
}

// 逻辑与
template<typename... T>
bool all(T... args) {
    return (args && ... && true);  // 二元左折叠
}

// 打印
template<typename... Args>
void print(Args&&... args) {
    (std::cout << ... << args) << '\n';
}
```

### 4.3 空参数包的特殊情况

```cpp
// 以下折叠空参数包是合法的：
(... &&)   // true
(... ||)   // false
(... ,)    // void()（空表达式）

// 其他运算符对空参数包是错误的：
// (... +)  // ERROR
```

### 4.4 带分隔符的打印

```cpp
template<typename T, char Sep>
class AddSeparator {
    T const& ref;
public:
    AddSeparator(T const& r) : ref(r) {}
    friend std::ostream& operator<<(std::ostream& os, AddSeparator<T, Sep> s) {
        return os << s.ref << Sep;
    }
};

template<typename... Args>
void print(Args&&... args) {
    (std::cout << ... << AddSeparator<Args, ','>(args)) << '\n';
}
```

---

## 5. 包扩展的位置

包扩展可以在以下位置使用：

### 5.1 表达式

```cpp
template<typename... Args>
void printDouble(Args&&... args) {
    print(args + args...);  // 每个参数加倍
}
```

### 5.2 初始化列表

```cpp
template<typename... Args>
auto makeVector(Args&&... args) {
    return std::vector<std::common_type_t<Args...>>{args...};
}
```

### 5.3 基类列表

```cpp
template<typename... Bases>
struct Overloader : Bases... {
    using Bases::operator()...;  // C++17
};
```

### 5.4 模板参数列表

```cpp
template<typename... Types>
class Tuple {};

template<typename... Types>
class MyTuple : public Tuple<Types*...> {};  // 每个类型变为指针
```

### 5.5 using 声明

```cpp
template<typename... Bases>
struct Overloader : Bases... {
    using Bases::operator()...;
};
```

---

## 6. 完美转发

### 6.1 基本模式

```cpp
template<typename... Args>
auto makeWidget(Args&&... args) {
    return Widget(std::forward<Args>(args)...);
}
```

### 6.2 转发引用的特殊行为

```cpp
template<typename... Args>
void wrapper(Args&&... args) {
    // Args 可能是左值引用或右值
    target(std::forward<Args>(args)...);
}
```

---

## 7. 非类型参数包

```cpp
template<std::size_t... Indices>
struct IndexSequence {};

template<typename T, std::size_t... Idx>
void printElems(T const& t, IndexSequence<Idx...>) {
    print(std::get<Idx>(t)...);
}
```

---

## 8. 参数包的高级用法

### 8.1 折叠表达式 + 类型特征

```cpp
template<typename T1, typename... TN>
constexpr bool isSameAll = (std::is_same_v<T1, TN> && ...);
```

### 8.2 参数包继承

```cpp
template<typename... Bases>
struct MultiInherit : Bases... {
    using Bases::operator()...;
};
```

### 8.3 成员指针遍历

```cpp
struct Node {
    int v;
    Node* left;
    Node* right;
};

template<typename T, typename... TP>
Node* traverse(T np, TP... paths) {
    return (np ->* ... ->* paths);
}

// 使用
Node* result = traverse(root, &Node::left, &Node::right);
// 等价于 root->*(&Node::left)->*(&Node::right)
```

### 8.4 推导指引中的参数包

```cpp
template<typename T, typename... U>
array(T, U...) -> array<
    std::enable_if_t<(std::is_same_v<T, U> && ...), T>,
    1 + sizeof...(U)>;
```

---

## 9. 速查表

| 我想要... | 使用... |
|----------|---------|
| 获取参数包大小 | `sizeof...(pack)` |
| 对所有参数应用运算符 | 折叠表达式 `(args op ...)` |
| 递归处理参数包 | 首参数 + 递归 `rest...` |
| 完美转发所有参数 | `std::forward<Args>(args)...` |
| 编译时避免递归 | `if constexpr (sizeof...(rest) > 0)` |
| 展开为基类列表 | `Base<Types*>...` |
| 展开为 using 声明 | `using Bases::operator()...` |
