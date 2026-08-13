# C++ Templates 第 21 章：模板与继承的高级交互

## 章节概述

本章介绍模板与继承结合的各种技术，包括空基类优化（EBCO）、奇异递归模板模式（CRTP）、混合类（Mixins）和命名模板参数。

---

## 21.1 空基类优化（EBCO）

> [!info] 核心问题
> 空类（无非静态数据成员、无虚函数）的 `sizeof` 不为零（通常为 1），否则指针算术会出问题。但当空类作为**基类**时，编译器可以不为其分配空间——这就是 EBCO。

**EBCO 的限制：相同类型的子对象不能共享地址**

```cpp
class Empty {};
class EmptyToo : public Empty {};
class NonEmpty : public Empty, public EmptyToo {};
// sizeof(NonEmpty) > 0！
// Empty 和 EmptyToo 中的 Empty 子对象不能在同一地址
```

**模板中的应用：将成员变量改为基类**

```cpp
// 普通写法：如果 T1 或 T2 是空类，浪费空间
template<typename T1, typename T2>
class MyClass {
    T1 a;
    T2 b;
};

// 优化写法：利用 EBCO
template<typename T1, typename T2>
class MyClass : private T1, private T2 {
};
```

**BaseMemberPair 工具：合并可能为空的类型与普通成员**

```cpp
template<typename Base, typename Member>
class BaseMemberPair : private Base {
    Member mem;
public:
    BaseMemberPair(Base const& b, Member const& m)
        : Base(b), mem(m) {}
    Base const& base() const { return static_cast<Base const&>(*this); }
    Member const& member() const { return mem; }
};
```

---

## 21.2 奇异递归模板模式（CRTP）

> [!info] 核心思想
> 派生类将**自己**作为模板参数传递给基类，基类通过模板参数知道派生类的类型。

```cpp
template<typename Derived>
class CuriousBase { ... };

class Curious : public CuriousBase<Curious> { ... };
```

### 21.2.1 对象计数器

```cpp
template<typename CountedType>
class ObjectCounter {
    inline static std::size_t count = 0;
protected:
    ObjectCounter() { ++count; }
    ObjectCounter(ObjectCounter const&) { ++count; }
    ObjectCounter(ObjectCounter&&) { ++count; }
    ~ObjectCounter() { --count; }
public:
    static std::size_t live() { return count; }
};

class MyWidget : public ObjectCounter<MyWidget> {};
class MyButton : public ObjectCounter<MyButton> {};

// 每个类有独立的计数器
MyWidget w1, w2;
MyButton b1;
MyWidget::live();  // 2
MyButton::live();  // 1
```

### 21.2.2 Barton-Nackman 技巧（友元工厂）

在类模板内定义友元函数，使非模板函数通过 ADL 可见：

```cpp
template<typename T>
class Array {
    static bool areEqual(Array<T> const& a, Array<T> const& b);
public:
    friend bool operator==(Array<T> const& a, Array<T> const& b) {
        return areEqual(a, b);
    }
};
```

### 21.2.3 实现操作符（CRTP + Barton-Nackman）

```cpp
template<typename Derived>
class EqualityComparable {
public:
    friend bool operator!=(Derived const& x1, Derived const& x2) {
        return !(x1 == x2);  // 基于 Derived 的 operator== 生成 operator!=
    }
};

class X : public EqualityComparable<X> {
public:
    friend bool operator==(X const& x1, X const& x2) {
        // 实现比较逻辑
    }
    // 自动生成 operator!=
};
```

### 21.2.4 门面模式（Iterator Facade）

CRTP 基类用少量核心操作定义完整的迭代器接口：

```cpp
template<typename Derived, typename Value, typename Category,
         typename Reference = Value&, typename Distance = std::ptrdiff_t>
class IteratorFacade {
public:
    using value_type = std::remove_const_t<Value>;
    using reference = Reference;
    using pointer = Value*;
    using difference_type = Distance;
    using iterator_category = Category;

    // 由核心操作派生
    reference operator*() const { return derived().dereference(); }
    Derived& operator++() { derived().increment(); return derived(); }
    bool operator==(IteratorFacade const& o) const {
        return derived().equals(o);
    }
    // ...
private:
    Derived& derived() { return static_cast<Derived&>(*this); }
};

// 使用：只需实现 3 个核心方法
template<typename T>
class ListNodeIterator
    : public IteratorFacade<ListNodeIterator<T>, T, std::forward_iterator_tag>
{
    ListNode<T>* current = nullptr;
public:
    T& dereference() const { return current->value; }
    void increment() { current = current->next; }
    bool equals(ListNodeIterator const& other) const {
        return current == other.current;
    }
};
```

---

## 21.3 混合类（Mixins）

> [!info] 核心思想
> 混合类**颠倒继承方向**——基类接受派生类作为模板参数，通过参数包支持多个混合类。

```cpp
template<typename... Mixins>
class Point : public Mixins... {
    double x, y;
public:
    Point() : Mixins()..., x(0.0), y(0.0) {}
    Point(double x, double y) : Mixins()..., x(x), y(y) {}
};

// 使用
class Color { public: int r, g, b; };
class Label { public: std::string name; };
Point<Color, Label> p;  // 同时具有颜色和标签
```

**奇怪的混合类（CRTP + Mixins）：**

```cpp
template<template<typename>... Mixins>
class Point : public Mixins<Point>... {
    double x, y;
public:
    Point() : Mixins<Point>()..., x(0.0), y(0.0) {}
};
```

**参数化的虚拟性：**

```cpp
class NotVirtual {};
class Virtual { public: virtual void foo() {} };

template<typename... Mixins>
class Base : public Mixins... {
public:
    void foo() { std::cout << "Base::foo()" << '\n'; }
};

template<typename... Mixins>
class Derived : public Base<Mixins...> {
public:
    void foo() { std::cout << "Derived::foo()" << '\n'; }
};

Base<NotVirtual>* p1 = new Derived<NotVirtual>;
p1->foo();  // Base::foo() — 非虚调用

Base<Virtual>* p2 = new Derived<Virtual>;
p2->foo();  // Derived::foo() — 虚调用
```

---

## 21.4 命名模板参数

> [!info] 问题
> 类模板有大量默认参数时，想指定第 4 个参数必须先指定前 3 个。

**解决方案：** 用策略类 + 判别器 + 虚继承

```cpp
// 默认策略
class DefaultPolicies {
public:
    using P1 = DefaultPolicy1;
    using P2 = DefaultPolicy2;
    using P3 = DefaultPolicy3;
    using P4 = DefaultPolicy4;
};

class DefaultPolicyArgs : virtual public DefaultPolicies {};

// 命名参数
template<typename Policy>
class Policy3_is : virtual public DefaultPolicies {
public:
    using P3 = Policy;  // 只覆盖 P3
};

// 判别器：允许多个相同类型的基类
template<typename Base, int D>
class Discriminator : public Base {};

template<typename S1, typename S2, typename S3, typename S4>
class PolicySelector : public Discriminator<S1,1>,
                        public Discriminator<S2,2>,
                        public Discriminator<S3,3>,
                        public Discriminator<S4,4> {};

// 使用
template<typename PS1 = DefaultPolicyArgs,
         typename PS2 = DefaultPolicyArgs,
         typename PS3 = DefaultPolicyArgs,
         typename PS4 = DefaultPolicyArgs>
class BreadSlicer {
    using Policies = PolicySelector<PS1, PS2, PS3, PS4>;
    // 使用 Policies::P1, Policies::P2, ...
};

// 只指定第 3 个策略
BreadSlicer<Policy3_is<CustomPolicy>> bc;
```

---

## 技术对比

| 技术 | 用途 | 关键机制 |
|------|------|---------|
| **EBCO** | 消除空基类的空间浪费 | 空基类不占空间 |
| **CRTP** | 基类知道派生类类型 | 派生类作为模板参数传递 |
| **Barton-Nackman** | 非模板友元函数通过 ADL 可见 | 类内定义 friend 函数 |
| **Iterator Facade** | 用少量核心操作定义完整接口 | CRTP + 派生方法调用 |
| **Mixins** | 多个功能组合到一个类 | 参数包继承 |
| **命名模板参数** | 任意位置指定模板参数 | 虚继承 + 判别器 |
