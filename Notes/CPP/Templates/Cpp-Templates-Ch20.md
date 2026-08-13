# C++ Templates 第 20 章：基于类型属性的重载

## 章节概述

本章讨论如何基于模板参数的**类型属性**（如是否是随机访问迭代器、是否支持 `<` 操作符等）来选择不同的算法实现。C++ 没有直接的语法支持，但有多种技术可以模拟这种重载。

---

## 20.1 算法特化的动机

> [!info] 核心思想
> 为通用算法提供针对特定类型属性的**更高效版本**，调用者无需了解这些特化版本的存在，编译器会自动选择最优实现。

**示例：`swap()` 的特化**

```cpp
// 通用版本：3 次复制
template<typename T>
void swap(T& x, T& y) {
    T tmp(x);
    x = y;
    y = tmp;
}

// Array<T> 特化版本：只交换指针和长度
template<typename T>
void swap(Array<T>& x, Array<T>& y) {
    swap(x.ptr, y.ptr);
    swap(x.len, y.len);
}
```

**示例：`advanceIter()` 的问题**

```cpp
// 通用版本：线性时间
template<typename InputIterator, typename Distance>
void advanceIter(InputIterator& x, Distance n) {
    while (n > 0) { ++x; --n; }
}

// 随机访问版本：常量时间
template<typename RandomAccessIterator, typename Distance>
void advanceIter(RandomAccessIterator& x, Distance n) {
    x += n;
}
// ❌ 编译错误！两个模板参数名不同但签名等价，不能重载
```

本章的其余部分讨论解决这个问题的各种技术。

---

## 20.2 标签调度（Tag Dispatching）

> [!info] 核心思想
> 通过额外的**标签参数**区分不同的实现，利用标签之间的继承关系和重载解析选择最优版本。

**实现：**

```cpp
// 两个实现变体，通过标签区分
template<typename Iterator, typename Distance>
void advanceIterImpl(Iterator& x, Distance n, std::input_iterator_tag) {
    while (n > 0) { ++x; --n; }  // 线性时间
}

template<typename Iterator, typename Distance>
void advanceIterImpl(Iterator& x, Distance n, std::random_access_iterator_tag) {
    x += n;  // 常量时间
}

// 主函数：提取迭代器类别标签，转发给实现
template<typename Iterator, typename Distance>
void advanceIter(Iterator& x, Distance n) {
    advanceIterImpl(x, n,
        typename std::iterator_traits<Iterator>::iterator_category());
}
```

**标签继承体系：**

```cpp
namespace std {
    struct input_iterator_tag { };
    struct output_iterator_tag { };
    struct forward_iterator_tag : public input_iterator_tag { };
    struct bidirectional_iterator_tag : public forward_iterator_tag { };
    struct random_access_iterator_tag : public bidirectional_iterator_tag { };
}
```

> [!tip] 关键
> `random_access_iterator_tag` 继承自 `input_iterator_tag`，所以对随机访问迭代器调用 `advanceIterImpl` 时，两个重载都匹配，但更特化的 `random_access_iterator_tag` 版本优先。

**优缺点：**
- ✅ 简单、高效，标签有继承体系时很好用
- ❌ 不适合基于非体系化属性（如"是否有普通的复制赋值操作符"）的特化

---

## 20.3 EnableIf

> [!info] 核心思想
> 通过 `EnableIf` 在编译时条件为 false 时**禁用**函数模板，使其不参与重载解析。

**EnableIf 实现：**

```cpp
template<bool, typename T = void>
struct EnableIfT {};

template<typename T>
struct EnableIfT<true, T> {
    using Type = T;
};

template<bool Cond, typename T = void>
using EnableIf = typename EnableIfT<Cond, T>::Type;
```

**用于 `advanceIter()`：**

```cpp
template<typename Iterator>
constexpr bool IsRandomAccessIterator =
    IsConvertible<
        typename std::iterator_traits<Iterator>::iterator_category,
        std::random_access_iterator_tag>;

// 随机访问版本：启用条件为 IsRandomAccessIterator
template<typename Iterator, typename Distance>
EnableIf<IsRandomAccessIterator<Iterator>>
advanceIter(Iterator& x, Distance n) {
    x += n;
}

// 通用版本：启用条件为 NOT IsRandomAccessIterator
template<typename Iterator, typename Distance>
EnableIf<!IsRandomAccessIterator<Iterator>>
advanceIter(Iterator& x, Distance n) {
    while (n > 0) { ++x; --n; }
}
```

> [!warning] 条件必须互斥
> 每个重载的 `EnableIf` 条件必须与其他所有重载的条件互斥，否则两个模板都匹配会产生歧义错误。

### 20.3.1 多个特化版本

当需要更多实现时（如输入、双向、随机访问），每个版本的条件需要更精确：

```cpp
template<typename Iterator>
constexpr bool IsRandomAccessIterator = /* ... */;

template<typename Iterator>
constexpr bool IsBidirectionalIterator = /* ... */;

// 随机访问
template<typename Iterator, typename Distance>
EnableIf<IsRandomAccessIterator<Iterator>>
advanceIter(Iterator& x, Distance n) { x += n; }

// 双向但非随机访问
template<typename Iterator, typename Distance>
EnableIf<IsBidirectionalIterator<Iterator> && !IsRandomAccessIterator<Iterator>>
advanceIter(Iterator& x, Distance n) { /* 双向线性 */ }

// 其他（输入迭代器）
template<typename Iterator, typename Distance>
EnableIf<!IsBidirectionalIterator<Iterator>>
advanceIter(Iterator& x, Distance n) { /* 仅向前 */ }
```

> [!warning] EnableIf 的缺点
> 每次引入新实现时，都需要**重新审视所有条件**确保互斥。标签调度则只需添加一个新的 `advanceIterImpl` 重载。

### 20.3.2 EnableIf 的位置

`EnableIf` 可以放在**返回类型**或**默认模板参数**中：

```cpp
// 返回类型位置（不适用于构造函数）
template<typename Iterator, typename Distance>
EnableIf<IsRandomAccessIterator<Iterator>>
advanceIter(Iterator& x, Distance n);

// 默认模板参数位置（适用于构造函数）
template<typename T>
class Container {
    template<typename Iterator,
             typename = EnableIf<IsInputIterator<Iterator>>>
    Container(Iterator first, Iterator last);
};
```

**构造函数重载的陷阱：**

```cpp
// ❌ 错误：两个构造函数签名相同（默认参数不算在签名内）
template<typename Iterator, typename = EnableIf<A>>
Container(Iterator, Iterator);

template<typename Iterator, typename = EnableIf<B>>
Container(Iterator, Iterator);  // ERROR: redeclaration

// ✅ 修复：添加虚拟参数使签名不同
template<typename Iterator, typename = EnableIf<A>>
Container(Iterator, Iterator);

template<typename Iterator, typename = EnableIf<B>, typename = int>
Container(Iterator, Iterator);  // OK
```

### 20.3.3 编译时 if（C++17）

`if constexpr` 可以替代 `EnableIf`，更简洁：

```cpp
template<typename Iterator, typename Distance>
void advanceIter(Iterator& x, Distance n) {
    if constexpr (IsRandomAccessIterator<Iterator>) {
        x += n;
    } else if constexpr (IsBidirectionalIterator<Iterator>) {
        if (n > 0) { for (; n > 0; ++x, --n) {} }
        else { for (; n < 0; --x, ++n) {} }
    } else {
        if (n < 0) throw "invalid";
        while (n > 0) { ++x; --n; }
    }
}
```

> [!warning] 局限性
> `if constexpr` 不适用于：
> - 涉及不同接口的情况
> - 需要不同类定义的情况
> - 某些模板参数不存在有效实例化的情况
>
> 这些情况仍需 `EnableIf`，因为 `if constexpr` 不会从候选列表中移除函数。

### 20.3.4 概念（C++20）

`requires` 子句是 `EnableIf` 的更直接表达：

```cpp
template<typename Iterator>
requires IsRandomAccessIterator<Iterator>
void advanceIter(Iterator& x, Distance n) { x += n; }

template<typename Iterator>
requires (!IsRandomAccessIterator<Iterator>)
void advanceIter(Iterator& x, Distance n) { /* 线性 */ }
```

**优势：**
- 约束包含（subsumption）提供自动排序，无需手动互斥
- 可以附加到非模板成员函数上
- 错误信息更清晰

---

## 20.4 类模板偏特化

### 20.4.1 启用/禁用类模板

用 `EnableIf` + 偏特化为不同属性的类型提供不同的类实现：

```cpp
// 主模板：默认实现
template<typename Key, typename Value, typename = void>
class Dictionary {
    vector<pair<Key const, Value>> data;  // 线性查找
    // ...
};

// 偏特化：有 < 操作符时用 map
template<typename Key, typename Value>
class Dictionary<Key, Value, EnableIf<HasLess<Key> && !HasHash<Key>>> {
    map<Key, Value> data;  // O(log n) 查找
    // ...
};

// 偏特化：有哈希时用 unordered_map
template<typename Key, typename Value>
class Dictionary<Key, Value, EnableIf<HasHash<Key>>> {
    unordered_map<Key, Value> data;  // O(1) 查找
    // ...
};
```

> [!tip] 偏特化优先于主模板
> 不需要在主模板上添加禁用条件，偏特化天然优先。但多个偏特化之间条件必须互斥。

### 20.4.2 类模板的标签调度

用函数对象 + 偏特化实现标签调度：

```cpp
// 主模板（未定义）
template<typename Iterator,
         typename Tag = BestMatchInSet<
             typename std::iterator_traits<Iterator>::iterator_category,
             std::input_iterator_tag,
             std::bidirectional_iterator_tag,
             std::random_access_iterator_tag>>
class Advance;

// 输入迭代器版本
template<typename Iterator>
class Advance<Iterator, std::input_iterator_tag> {
public:
    void operator()(Iterator& x, DifferenceType n) const {
        while (n > 0) { ++x; --n; }
    }
};

// 随机访问迭代器版本
template<typename Iterator>
class Advance<Iterator, std::random_access_iterator_tag> {
public:
    void operator()(Iterator& x, DifferenceType n) const { x += n; }
};
```

**`BestMatchInSet` 特征的实现（利用重载解析）：**

```cpp
template<typename... Types>
struct MatchOverloads;

template<>
struct MatchOverloads<> {
    static void match(...);  // 回退
};

template<typename T1, typename... Rest>
struct MatchOverloads<T1, Rest...> : public MatchOverloads<Rest...> {
    static T1 match(T1);  // 引入 T1 的重载
    using MatchOverloads<Rest...>::match;  // 收集基类的重载
};

template<typename T, typename... Types>
struct BestMatchInSetT {
    using Type = decltype(MatchOverloads<Types...>::match(std::declval<T>()));
};

template<typename T, typename... Types>
using BestMatchInSet = typename BestMatchInSetT<T, Types...>::Type;
```

> [!info] 原理
> `MatchOverloads` 通过递归继承为每个类型引入一个 `match()` 重载。调用 `match(std::declval<T>())` 时，重载解析选择最佳匹配的版本，`decltype` 拿到返回类型。

---

## 20.5 实例化安全模板

> [!info] 核心思想
> 将模板对参数的**每个操作**都编码为 `EnableIf` 条件，确保模板**永远不会因为实例化失败而报错**——不满足条件的参数直接被 SFINAE 排除。

**示例：实例化安全的 `min()`**

```cpp
// 普通版本：如果 T 没有 < 操作符，实例化时报错
template<typename T>
T const& min(T const& x, T const& y) {
    if (y < x) return y;
    return x;
}

// 实例化安全版本：条件不满足时 SFINAE 排除
template<typename T>
EnableIf<IsContextualBool<LessResult<T const&, T const&>>,
         T const&>
min(T const& x, T const& y) {
    if (y < x) return y;
    return x;
}
```

**`LessResult` 特征（检测 `<` 操作符）：**

```cpp
template<typename T1, typename T2>
class HasLess {
    template<typename T> struct Identity;
    template<typename U1, typename U2> static std::true_type
        test(Identity<decltype(std::declval<U1>() < std::declval<U2>())>*);
    template<typename U1, typename U2> static std::false_type
        test(...);
public:
    static constexpr bool value = decltype(test<T1, T2>(nullptr))::value;
};
```

**`IsContextualBool` 特征（检测上下文转换为 bool）：**

```cpp
// 利用三元操作符测试：条件表达式必须可转换为 bool
template<typename T>
class IsContextualBoolT {
    template<typename U> struct Identity;
    template<typename U> static std::true_type
        test(Identity<decltype(std::declval<U>() ? 0 : 1)>*);
    template<typename U> static std::false_type
        test(...);
public:
    static constexpr bool value = decltype(test<T>(nullptr))::value;
};
```

> [!warning] 过度约束 vs 约束不足
> - **过度约束**：要求隐式 `bool` 转换，但实际只需上下文转换（如 `explicit operator bool` 在 `if` 中可用）
> - **约束不足**：忘记检查某个操作，导致实例化失败
>
> 准确编码模板需求是一项困难但重要的任务。

---

## 20.6 标准库特化技术

> [!info] 标准库的做法
> C++ 标准库在多处使用这些技术：
> - `std::advance()` 和 `std::distance()` 使用标签调度
> - `std::copy()` 在连续内存 + 普通复制赋值时优化为 `memcpy`
> - `std::fill()` 在普通类型时优化为 `memset`
> - `std::vector` 的迭代器构造函数使用 `enable_if` 确保参数是输入迭代器

---

## 技术对比总结

| 技术 | 适用场景 | 优点 | 缺点 |
|------|---------|------|------|
| **标签调度** | 有层次化标签体系 | 简单、易扩展 | 需要标签继承体系 |
| **EnableIf** | 基于任意类型属性 | 灵活 | 条件需手动互斥，难维护 |
| **if constexpr** | 差异在函数体内 | 最简洁 | 不适用于不同接口/类定义 |
| **Concepts** | C++20+ | 最清晰，自动排序 | 需要 C++20 |
| **偏特化 + EnableIf** | 类模板的不同实现 | 适用于类模板 | 条件需互斥 |
| **实例化安全模板** | 需要严格类型检查 | 永不实例化失败 | 编码复杂 |

---

## 速查表

| 我想要... | 使用... |
|----------|---------|
| 按迭代器类别分派算法 | 标签调度 |
| 按任意类型属性启用/禁用函数 | EnableIf |
| 函数体内按类型选择不同代码路径 | if constexpr |
| 清晰的约束和错误信息（C++20） | requires 子句 |
| 为类模板提供不同实现 | 偏特化 + EnableIf |
| 确保模板永不因实例化失败报错 | 实例化安全模板（EnableIf + 完整需求检查） |
