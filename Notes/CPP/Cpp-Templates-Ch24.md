# C++ Templates 第 24 章：类型列表

## 章节概述

类型列表是类型元编程的**核心数据结构**。本章介绍类型列表的基本操作和算法。

---

## 24.1 基本操作

```cpp
template<typename... Elements>
class Typelist {};
```

**Front、PopFront、PushFront：**

```cpp
// Front：取第一个元素
template<typename Head, typename... Tail>
class FrontT<Typelist<Head, Tail...>> {
public:
    using Type = Head;
};
template<typename List> using Front = typename FrontT<List>::Type;

// PopFront：移除第一个元素
template<typename Head, typename... Tail>
class PopFrontT<Typelist<Head, Tail...>> {
public:
    using Type = Typelist<Tail...>;
};
template<typename List> using PopFront = typename PopFrontT<List>::Type;

// PushFront：在头部添加元素
template<typename... Elements, typename NewElement>
class PushFrontT<Typelist<Elements...>, NewElement> {
public:
    using Type = Typelist<NewElement, Elements...>;
};
template<typename List, typename Elem>
using PushFront = typename PushFrontT<List, Elem>::Type;
```

**IsEmpty：**

```cpp
template<typename List> class IsEmpty { public: static constexpr bool value = false; };
template<> class IsEmpty<Typelist<>> { public: static constexpr bool value = true; };
```

---

## 24.2 类型列表算法

### 24.2.1 索引（NthElement）

```cpp
template<typename List, unsigned N>
class NthElementT : public NthElementT<PopFront<List>, N-1> {};

template<typename List>
class NthElementT<List, 0> : public FrontT<List> {};

template<typename List, unsigned N>
using NthElement = typename NthElementT<List, N>::Type;
```

### 24.2.2 寻找最大类型

```cpp
template<typename List>
class LargestTypeT {
    using First = Front<List>;
    using Rest = typename LargestTypeT<PopFront<List>>::Type;
public:
    using Type = IfThenElse<(sizeof(First) >= sizeof(Rest)), First, Rest>;
};

template<> class LargestTypeT<Typelist<>> { public: using Type = char; };
```

### 24.2.3 PushBack

```cpp
// 变参模板版本（简单）
template<typename... Elements, typename NewElement>
class PushBackT<Typelist<Elements...>, NewElement> {
public:
    using Type = Typelist<Elements..., NewElement>;
};

// 递归版本（通用）
template<typename List, typename NewElement, bool = IsEmpty<List>::value>
class PushBackRecT;

template<typename List, typename NewElement>
class PushBackRecT<List, NewElement, false> {
    using Head = Front<List>;
    using Tail = PopFront<List>;
    using NewTail = typename PushBackRecT<Tail, NewElement>::Type;
public:
    using Type = PushFront<Head, NewTail>;
};

template<typename List, typename NewElement>
class PushBackRecT<List, NewElement, true> {
public:
    using Type = PushFront<List, NewElement>;
};
```

### 24.2.4 反转

```cpp
template<typename List, bool Empty = IsEmpty<List>::value>
class ReverseT;

template<typename List>
class ReverseT<List, false>
    : public PushBackT<Reverse<PopFront<List>>, Front<List>> {};

template<typename List>
class ReverseT<List, true> { public: using Type = List; };

template<typename List> using Reverse = typename ReverseT<List>::Type;

// PopBack：利用反转实现
template<typename List>
class PopBackT { public: using Type = Reverse<PopFront<Reverse<List>>>; };
```

### 24.2.5 变换（Transform）

对列表中每个类型应用元函数：

```cpp
template<typename List, template<typename T> class MetaFun,
         bool Empty = IsEmpty<List>::value>
class TransformT;

template<typename List, template<typename T> class MetaFun>
class TransformT<List, MetaFun, false>
    : public PushFrontT<
        typename TransformT<PopFront<List>, MetaFun>::Type,
        typename MetaFun<Front<List>>::Type> {};

template<typename List, template<typename T> class MetaFun>
class TransformT<List, MetaFun, true> { public: using Type = List; };
```

### 24.2.6 累加（Accumulate）

```cpp
template<typename List,
         template<typename X, typename Y> class F,
         typename I,
         bool = IsEmpty<List>::value>
class AccumulateT;

template<typename List, template<typename X, typename Y> class F, typename I>
class AccumulateT<List, F, I, false>
    : public AccumulateT<PopFront<List>, F, typename F<I, Front<List>>::Type> {};

template<typename List, template<typename X, typename Y> class F, typename I>
class AccumulateT<List, F, I, true> { public: using Type = I; };
```

### 24.2.7 插入排序

```cpp
template<typename List,
         template<typename T, typename U> class Compare,
         bool = IsEmpty<List>::value>
class InsertionSortT;

template<typename List, template<typename T, typename U> class Compare>
class InsertionSortT<List, Compare, false>
    : public InsertSortedT<InsertionSort<PopFront<List>, Compare>,
                           Front<List>, Compare> {};

template<typename List, template<typename T, typename U> class Compare>
class InsertionSortT<List, Compare, true> { public: using Type = List; };
```

---

## 24.3 值类型列表

```cpp
template<typename T, T Value>
struct CTValue { static constexpr T value = Value; };

template<typename T, T... Values>
using CTTypelist = Typelist<CTValue<T, Values>...>;

// 使用
using Primes = CTTypelist<int, 2, 3, 5, 7, 11, 13>;
```

**C++17 简化：**

```cpp
template<auto Value>
struct CTValue { static constexpr auto value = Value; };
```

---

## 24.4 包扩展优化

用包扩展替代递归：

```cpp
// Transform 的包扩展版本
template<typename... Elements, template<typename T> class MetaFun>
class TransformT<Typelist<Elements...>, MetaFun, false> {
public:
    using Type = Typelist<typename MetaFun<Elements>::Type...>;
};
```

---

## 24.5 Cons 风格类型列表（预可变参数模板时代）

```cpp
class Nil {};
template<typename HeadT, typename TailT = Nil>
class Cons {
public:
    using Head = HeadT;
    using Tail = TailT;
};

// 使用
using MyTypes = Cons<int, Cons<double, Cons<char, Nil>>>;
```

> [!info] 对比
> Cons 风格是递归链表，Typelist 风格是变参模板扁平列表。现代 C++ 优先使用 Typelist。
