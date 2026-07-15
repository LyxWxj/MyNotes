# C++ Templates 第 26 章：可辨别联合（Variant）

## 章节概述

本章实现 `std::variant` 的简化版本，展示如何用模板构建类型安全的可辨别联合。

---

## 26.1 Variant 存储

```cpp
template<typename... Types>
class VariantStorage {
    using LargestT = LargestType<Typelist<Types...>>;
    alignas(Types...) unsigned char buffer[sizeof(LargestT)];
    unsigned char discriminator = 0;
public:
    unsigned char getDiscriminator() const { return discriminator; }
    void setDiscriminator(unsigned char d) { discriminator = d; }
    void* getRawBuffer() { return buffer; }

    template<typename T>
    T* getBufferAs() { return std::launder(reinterpret_cast<T*>(buffer)); }
};
```

> [!info] 关键设计
> - `buffer`：足够大的对齐存储，能容纳任何类型
> - `discriminator`：记录当前存储的是哪种类型（0 表示空）
> - `alignas(Types...)`：确保对齐满足所有类型的要求

---

## 26.2 Variant 设计（CRTP）

### 26.2.1 FindIndexOfT

查找类型在类型列表中的索引：

```cpp
template<typename List, typename T, unsigned N = 0,
         bool Empty = IsEmpty<List>::value>
struct FindIndexOfT;

template<typename List, typename T, unsigned N>
struct FindIndexOfT<List, T, N, false>
    : public IfThenElse<std::is_same<Front<List>, T>::value,
                        std::integral_constant<unsigned, N>,
                        FindIndexOfT<PopFront<List>, T, N+1>> {};
```

### 26.2.2 VariantChoice

每个类型的选择器（CRTP 混入）：

```cpp
template<typename T, typename... Types>
class VariantChoice {
    using Derived = Variant<Types...>;
    Derived& getDerived() { return *static_cast<Derived*>(this); }
protected:
    constexpr static unsigned Discriminator =
        FindIndexOfT<Typelist<Types...>, T>::value + 1;
public:
    VariantChoice() {}
    VariantChoice(T const& value);
    VariantChoice(T&& value);
    bool destroy();
    Derived& operator=(T const& value);
    Derived& operator=(T&& value);
};
```

### 26.2.3 Variant 框架

```cpp
template<typename... Types>
class Variant
    : private VariantStorage<Types...>,
      private VariantChoice<Types, Types...>...  // 每个类型一个基类
{
    template<typename T, typename... OtherTypes>
    friend class VariantChoice;
public:
    using VariantChoice<Types, Types...>::VariantChoice...;  // 继承构造函数
    using VariantChoice<Types, Types...>::operator=...;      // 继承赋值

    template<typename T> bool is() const;
    template<typename T> T& get();
    template<typename R, typename Visitor> auto visit(Visitor&& vis);

    Variant();
    ~Variant() { destroy(); }
    void destroy();
};
```

---

## 26.3 查询（is 和 get）

```cpp
template<typename... Types>
template<typename T>
bool Variant<Types...>::is() const {
    return this->getDiscriminator() ==
           VariantChoice<T, Types...>::Discriminator;
}

template<typename... Types>
template<typename T>
T& Variant<Types...>::get() & {
    if (empty()) throw EmptyVariant();
    assert(is<T>());
    return *this->template getBufferAs<T>();
}
```

---

## 26.4 初始化、销毁和赋值

### 26.4.1 初始化（placement new）

```cpp
template<typename T, typename... Types>
VariantChoice<T, Types...>::VariantChoice(T const& value) {
    new(getDerived().getRawBuffer()) T(value);        // placement new
    getDerived().setDiscriminator(Discriminator);
}
```

### 26.4.2 销毁

```cpp
template<typename T, typename... Types>
bool VariantChoice<T, Types...>::destroy() {
    if (getDerived().getDiscriminator() == Discriminator) {
        getDerived().template getBufferAs<T>()->~T();  // 显式析构
        return true;
    }
    return false;
}

template<typename... Types>
void Variant<Types...>::destroy() {
    (VariantChoice<Types, Types...>::destroy() , ...);  // C++17 折叠表达式
    this->setDiscriminator(0);
}
```

### 26.4.3 赋值

```cpp
template<typename T, typename... Types>
auto VariantChoice<T, Types...>::operator=(T const& value) -> Derived& {
    if (getDerived().getDiscriminator() == Discriminator) {
        *getDerived().template getBufferAs<T>() = value;  // 同类型：直接赋值
    } else {
        getDerived().destroy();
        new(getDerived().getRawBuffer()) T(value);         // 不同类型：先析构再构造
        getDerived().setDiscriminator(Discriminator);
    }
    return getDerived();
}
```

> [!warning] std::launder()
> C++17 引入 `std::launder()` 告诉编译器：这块内存中的对象可能已经换了，不要用旧的缓存值。

---

## 26.5 访问者（Visitor）

```cpp
template<typename R, typename V, typename Visitor, typename Head, typename... Tail>
R variantVisitImpl(V&& variant, Visitor&& vis, Typelist<Head, Tail...>) {
    if (variant.template is<Head>()) {
        return static_cast<R>(
            std::forward<Visitor>(vis)(
                std::forward<V>(variant).template get<Head>()));
    } else if constexpr (sizeof...(Tail) > 0) {
        return variantVisitImpl<R>(std::forward<V>(variant),
                                   std::forward<Visitor>(vis),
                                   Typelist<Tail...>());
    } else {
        throw EmptyVariant();
    }
}
```

**结果类型推导：**

```cpp
// 用 Accumulate + CommonType 推导所有访问结果的公共类型
template<typename Visitor, typename T>
using VisitElementResult = decltype(std::declval<Visitor>()(std::declval<T>()));

template<typename Visitor, typename... ElementTypes>
class VisitResultT<ComputedResultType, Visitor, ElementTypes...> {
    using ResultTypes = Typelist<VisitElementResult<Visitor, ElementTypes>...>;
public:
    using Type = Accumulate<PopFront<ResultTypes>, CommonTypeT, Front<ResultTypes>>;
};
```

---

## 26.6 构造与赋值细节

```cpp
// 默认构造：初始化为第一个类型
template<typename... Types>
Variant<Types...>::Variant() { *this = Front<Typelist<Types...>>(); }

// 复制构造：通过 visit
template<typename... Types>
Variant<Types...>::Variant(Variant const& source) {
    if (!source.empty()) {
        source.visit([&](auto const& value) { *this = value; });
    }
}
```

---

## 26.7 历史

- Andrei Alexandrescu 详细介绍了可辨别联合
- Boost.Variant → C++17 `std::variant`
- `std::variant` 放弃了"永不空"的约定（允许 `valueless_by_exception`）
