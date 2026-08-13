# C++ Templates 第 22 章：桥接静态和动态多态

## 章节概述

本章构建 `std::function<>` 的简化版本，展示如何用**类型擦除**技术桥接静态多态（模板）和动态多态（虚函数）。

---

## 22.1 函数对象与类型擦除

> [!info] 问题
> 模板函数 `forUpTo(int n, F f)` 接受任意可调用对象，但每次调用都会实例化新版本。我们想要一个**非模板**的函数，能接受任意可调用对象。

```cpp
// 模板版本：每种 F 生成一份代码
template<typename F>
void forUpTo(int n, F f) { for (int i = 0; i != n; ++i) f(i); }

// std::function 版本：只有一份代码
void forUpTo(int n, std::function<void(int)> f) {
    for (int i = 0; i != n; ++i) f(i);
}
```

---

## 22.2 FunctionPtr 接口

我们构建 `FunctionPtr<R(Args...)>`，类似于 `std::function`：

```cpp
template<typename Signature> class FunctionPtr;

template<typename R, typename... Args>
class FunctionPtr<R(Args...)> {
    FunctorBridge<R, Args...>* bridge;  // 指向实现的指针
public:
    FunctionPtr() : bridge(nullptr) {}
    FunctionPtr(FunctionPtr const& other);      // 拷贝构造
    FunctionPtr(FunctionPtr&& other);           // 移动构造
    template<typename F> FunctionPtr(F&& f);    // 从任意可调用对象构造
    ~FunctionPtr() { delete bridge; }

    R operator()(Args... args) const;           // 调用
    explicit operator bool() const { return bridge != nullptr; }
};
```

---

## 22.3 FunctorBridge（抽象接口）

```cpp
template<typename R, typename... Args>
class FunctorBridge {
public:
    virtual ~FunctorBridge() {}
    virtual FunctorBridge* clone() const = 0;       // 复制
    virtual R invoke(Args... args) const = 0;       // 调用
};
```

---

## 22.4 SpecificFunctorBridge（具体实现）

```cpp
template<typename Functor, typename R, typename... Args>
class SpecificFunctorBridge : public FunctorBridge<R, Args...> {
    Functor functor;
public:
    template<typename FunctorFwd>
    SpecificFunctorBridge(FunctorFwd&& functor)
        : functor(std::forward<FunctorFwd>(functor)) {}

    SpecificFunctorBridge* clone() const override {
        return new SpecificFunctorBridge(functor);
    }
    R invoke(Args... args) const override {
        return functor(std::forward<Args>(args)...);
    }
};
```

**FunctionPtr 的构造和调用：**

```cpp
// 构造：创建 SpecificFunctorBridge
template<typename R, typename... Args>
template<typename F>
FunctionPtr<R(Args...)>::FunctionPtr(F&& f) : bridge(nullptr) {
    using Functor = std::decay_t<F>;
    using Bridge = SpecificFunctorBridge<Functor, R, Args...>;
    bridge = new Bridge(std::forward<F>(f));
}

// 调用：委托给 bridge
template<typename R, typename... Args>
R FunctionPtr<R(Args...)>::operator()(Args... args) const {
    return bridge->invoke(std::forward<Args>(args)...);
}

// 拷贝构造：通过 clone
template<typename R, typename... Args>
FunctionPtr<R(Args...)>::FunctionPtr(FunctionPtr const& other)
    : bridge(nullptr) {
    if (other.bridge) bridge = other.bridge->clone();
}
```

> [!info] 类型擦除的含义
> 具体类型 `F` 在构造时被封装进 `SpecificFunctorBridge<F, R, Args...>`，然后通过基类指针 `FunctorBridge<R, Args...>*` 存储。`F` 的类型信息从此丢失——这就是"类型擦除"。

---

## 22.5 相等比较

添加 `equals` 到 `FunctorBridge`：

```cpp
// FunctorBridge 中添加
virtual bool equals(FunctorBridge const* fb) const = 0;

// SpecificFunctorBridge 中实现
bool equals(FunctorBridge const* fb) const override {
    if (auto p = dynamic_cast<SpecificFunctorBridge const*>(fb)) {
        return functor == p->functor;  // 如果 Functor 支持 ==
    }
    return false;
}
```

**`IsEqualityComparable` 特征：**

```cpp
template<typename T>
class IsEqualityComparable {
    static void* conv(bool);
    template<typename U>
    static std::true_type test(
        decltype(conv(std::declval<U const&>() == std::declval<U const&>())),
        decltype(conv(!(std::declval<U const&>() == std::declval<U const&>()))));
    template<typename U>
    static std::false_type test(...);
public:
    static constexpr bool value = decltype(test<T>(nullptr, nullptr))::value;
};
```

**`TryEquals`：根据是否可比较选择不同策略**

```cpp
template<typename T, bool EqComparable = IsEqualityComparable<T>::value>
struct TryEquals {
    static bool equals(T const& x1, T const& x2) { return x1 == x2; }
};

template<typename T>
struct TryEquals<T, false> {
    static bool equals(T const&, T const&) { throw NotEqualityComparable(); }
};
```

---

## 22.6 性能注意事项

| | 静态多态（模板） | 动态多态（虚函数） | 类型擦除 |
|---|---|---|---|
| **调用开销** | 可内联，零开销 | 虚函数表间接调用 | 虚函数调用 |
| **代码体积** | 每种类型一份 | 共享 | 共享 |
| **灵活性** | 编译时确定 | 运行时确定 | 运行时确定 |

类型擦除的性能更接近动态多态，但提供了类似静态多态的灵活接口。

---

## 22.7 历史

- Kevlin Henney 推广了类型擦除，引入 `any` 类型（后成为 C++17 `std::any`）
- Boost.Function → `std::function<>`
- Boost.TypeErasure 和 Adobe 的 Poly 库进一步发展了类型擦除技术
