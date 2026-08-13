# C++ Templates 第 25 章：元组

## 章节概述

本章实现 `std::tuple` 的简化版本，展示元组的存储、构造、访问、算法和优化。

---

## 25.1 元组存储与构造

### 25.1.1 递归存储

```cpp
template<typename... Types> class Tuple;

template<typename Head, typename... Tail>
class Tuple<Head, Tail...> {
    Head head;
    Tuple<Tail...> tail;
public:
    Tuple() {}
    Tuple(Head const& head, Tuple<Tail...> const& tail)
        : head(head), tail(tail) {}

    Head& getHead() { return head; }
    Tuple<Tail...>& getTail() { return tail; }
};

template<> class Tuple<> {};
```

**get 函数：**

```cpp
template<unsigned N>
struct TupleGet {
    template<typename Head, typename... Tail>
    static auto apply(Tuple<Head, Tail...> const& t) {
        return TupleGet<N-1>::apply(t.getTail());
    }
};

template<>
struct TupleGet<0> {
    template<typename Head, typename... Tail>
    static Head const& apply(Tuple<Head, Tail...> const& t) {
        return t.getHead();
    }
};

template<unsigned N, typename... Types>
auto get(Tuple<Types...> const& t) {
    return TupleGet<N>::apply(t);
}
```

### 25.1.2 完美转发构造

```cpp
template<typename VHead, typename... VTail,
         typename = std::enable_if_t<sizeof...(VTail) == sizeof...(Tail)>>
Tuple(VHead&& vhead, VTail&&... vtail)
    : head(std::forward<VHead>(vhead)),
      tail(std::forward<VTail>(vtail)...) {}
```

**makeTuple：**

```cpp
template<typename... Types>
auto makeTuple(Types&&... elems) {
    return Tuple<std::decay_t<Types>...>(std::forward<Types>(elems)...);
}
```

---

## 25.2 元组 I/O 与比较

```cpp
// 比较
bool operator==(Tuple<> const&, Tuple<> const&) { return true; }

template<typename Head1, typename... Tail1, typename Head2, typename... Tail2>
bool operator==(Tuple<Head1, Tail1...> const& t1, Tuple<Head2, Tail2...> const& t2) {
    return t1.getHead() == t2.getHead()
        && t1.getTail() == t2.getTail();
}

// 输出
template<typename Head, typename... Tail>
void printTuple(std::ostream& strm, Tuple<Head, Tail...> const& t, bool isFirst = true) {
    strm << (isFirst ? "(" : ", ");
    strm << t.getHead();
    printTuple(strm, t.getTail(), false);
}

void printTuple(std::ostream& strm, Tuple<> const&, bool isFirst) {
    strm << (isFirst ? "()" : ")");
}
```

---

## 25.3 元组算法

### 25.3.1 元组作为类型列表

```cpp
template<> struct IsEmpty<Tuple<>> { static constexpr bool value = true; };
template<typename Head, typename... Tail>
class FrontT<Tuple<Head, Tail...>> { public: using Type = Head; };
template<typename Head, typename... Tail>
class PopFrontT<Tuple<Head, Tail...>> { public: using Type = Tuple<Tail...>; };
```

### 25.3.2 添加和删除元素

```cpp
template<typename... Types, typename V>
PushFront<Tuple<Types...>, V> pushFront(Tuple<Types...> const& tuple, V const& value) {
    return PushFront<Tuple<Types...>, V>(value, tuple);
}

template<typename... Types>
PopFront<Tuple<Types...>> popFront(Tuple<Types...> const& tuple) {
    return tuple.getTail();
}
```

### 25.3.3 索引列表

```cpp
template<unsigned N, typename Result = Valuelist<unsigned>>
struct MakeIndexListT
    : MakeIndexListT<N-1, PushFront<Result, CTValue<unsigned, N-1>>> {};

template<typename Result>
struct MakeIndexListT<0, Result> { using Type = Result; };

template<unsigned N>
using MakeIndexList = typename MakeIndexListT<N>::Type;
```

### 25.3.4 用索引列表反转

```cpp
template<typename... Elements, unsigned... Indices>
auto reverseImpl(Tuple<Elements...> const& t, Valuelist<unsigned, Indices...>) {
    return makeTuple(get<Indices>(t)...);
}

template<typename... Elements>
auto reverse(Tuple<Elements...> const& t) {
    return reverseImpl(t, Reverse<MakeIndexList<sizeof...(Elements)>>());
}
```

### 25.3.5 apply（元组解包）

```cpp
template<typename F, typename... Elements, unsigned... Indices>
auto applyImpl(F f, Tuple<Elements...> const& t, Valuelist<unsigned, Indices...>)
    -> decltype(f(get<Indices>(t)...)) {
    return f(get<Indices>(t)...);
}

template<typename F, typename... Elements, unsigned N = sizeof...(Elements)>
auto apply(F f, Tuple<Elements...> const& t)
    -> decltype(applyImpl(f, t, MakeIndexList<N>())) {
    return applyImpl(f, t, MakeIndexList<N>());
}
```

---

## 25.4 元组优化

### 25.4.1 EBCO 优化

```cpp
// 用 TupleElt 存储每个元素，支持空类优化
template<unsigned Height, typename T,
         bool = std::is_class<T>::value && !std::is_final<T>::value>
class TupleElt;

template<unsigned Height, typename T>
class TupleElt<Height, T, false> {
    T value;
public:
    T& get() { return value; }
};

template<unsigned Height, typename T>
class TupleElt<Height, T, true> : private T {  // 继承空类
public:
    T& get() { return *this; }
};
```

### 25.4.2 常量时间 get

```cpp
template<unsigned H, typename T>
T& getHeight(TupleElt<H, T>& te) { return te.get(); }

template<unsigned I, typename... Elements>
auto get(Tuple<Elements...>& t)
    -> decltype(getHeight<sizeof...(Elements)-I-1>(t)) {
    return getHeight<sizeof...(Elements)-I-1>(t);
}
```

---

## 25.5 用户定义字面量

```cpp
template<char... cs>
constexpr auto operator"" _c() {
    return CTValue<int, parseInt<sizeof...(cs)>({cs...})>{};
}

// 使用
auto idx = 3_c;  // CTValue<int, 3>
tuple[idx];      // 等价于 get<3>(tuple)
```
