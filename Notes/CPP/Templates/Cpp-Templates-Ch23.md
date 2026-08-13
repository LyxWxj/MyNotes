# C++ Templates 第 23 章：元编程

## 章节概述

元编程是编写在**编译时**执行的代码，生成运行时代码。本章介绍值元编程、类型元编程、混合元编程和编译时递归。

---

## 23.1 元编程的分类

### 23.1.1 值元编程

在编译时计算值（C++14+ `constexpr` 函数）：

```cpp
template<typename T>
constexpr T sqrt(T x) {
    if (x <= 1) return x;
    T lo = 0, hi = x;
    for (;;) {
        auto mid = (hi+lo)/2, midSquared = mid*mid;
        if (lo+1 >= hi || midSquared == x) return mid;
        if (midSquared < x) lo = mid;
        else hi = mid;
    }
}
```

### 23.1.2 类型元编程

元函数接受类型，返回类型：

```cpp
// 去除所有数组维度
template<typename T>
struct RemoveAllExtentsT { using Type = T; };

template<typename T, std::size_t SZ>
struct RemoveAllExtentsT<T[SZ]> {
    using Type = typename RemoveAllExtentsT<T>::Type;
};

template<typename T>
using RemoveAllExtents = typename RemoveAllExtentsT<T>::Type;

// array<array<int, 3>, 4> → int
using Inner = RemoveAllExtents<std::array<std::array<int, 3>, 4>>;
```

### 23.1.3 混合元编程

编译时循环展开 + 运行时计算：

```cpp
// 递归展开的点积
template<typename T, std::size_t N>
struct DotProductT {
    static inline T result(T* a, T* b) {
        return *a * *b + DotProductT<T, N-1>::result(a+1, b+1);
    }
};

template<typename T>
struct DotProductT<T, 0> {
    static inline T result(T*, T*) { return T{}; }
};

double a[] = {1, 2, 3};
double b[] = {4, 5, 6};
double r = DotProductT<double, 3>::result(a, b);  // 1*4 + 2*5 + 3*6 = 32
```

### 23.1.4 单位类型的混合元编程

编译时比率计算（`std::chrono` 的基础）：

```cpp
template<unsigned N, unsigned D = 1>
struct Ratio {
    static constexpr unsigned num = N;
    static constexpr unsigned den = D;
};

template<typename R1, typename R2>
struct RatioAddImpl {
    static constexpr unsigned den = R1::den * R2::den;
    static constexpr unsigned num = R1::num * R2::den + R2::num * R1::den;
    using Type = Ratio<num, den>;
};

template<typename R1, typename R2>
using RatioAdd = typename RatioAddImpl<R1, R2>::Type;

// Duration 类型
template<typename T, typename U = Ratio<1>>
class Duration {
    T val;
public:
    constexpr Duration(T v = 0) : val(v) {}
    constexpr T value() const { return val; }
};
```

---

## 23.2 递归模板实例化计算平方根

```cpp
template<int N, int LO = 1, int HI = N>
struct Sqrt {
    static constexpr auto mid = (LO + HI + 1) / 2;
    static constexpr auto value =
        (N < mid * mid) ? Sqrt<N, LO, mid-1>::value
                        : Sqrt<N, mid, HI>::value;
};

template<int N, int M>
struct Sqrt<N, M, M> {
    static constexpr auto value = M;
};

Sqrt<16>::value  // 4
```

**展开过程：**
```
Sqrt<16,1,16> → mid=9 → Sqrt<16,1,8>
Sqrt<16,1,8>  → mid=5 → Sqrt<16,1,4>
Sqrt<16,1,4>  → mid=3 → Sqrt<16,3,4>
Sqrt<16,3,4>  → mid=4 → Sqrt<16,4,4> = 4
```

**优化：用 `IfThenElse` 减少实例化**

```cpp
template<int N, int LO = 1, int HI = N>
struct Sqrt {
    static constexpr auto mid = (LO + HI + 1) / 2;
    using SubT = IfThenElse<(N < mid*mid),
                            Sqrt<N, LO, mid-1>,
                            Sqrt<N, mid, HI>>;
    static constexpr auto value = SubT::value;
};
```

> [!tip] 实例化数量
> 优化后实例化数量与 `log2(N)` 成正比。

---

## 23.3 元编程的能力与限制

模板元程序可以有：
- **状态变量**：模板参数
- **循环构造**：递归实例化
- **执行路径选择**：条件表达式或特化
- **整数运算**：编译时常量表达式

**限制：**
- C++ 标准建议至少允许 **1024 级**递归实例化
- 类型描述的复杂性可能指数增长

```cpp
// 类型复杂性指数增长示例
template<typename T, typename U>
struct Doublify {};

template<int N>
struct Trouble {
    using LongType = Doublify<typename Trouble<N-1>::LongType,
                              typename Trouble<N-1>::LongType>;
};

template<>
struct Trouble<0> {
    using LongType = double;
};
// Trouble<10>::LongType 有 2^10 = 1024 个 double 嵌套
```

---

## 元编程范式总结

| 类型 | 输入 | 输出 | 示例 |
|------|------|------|------|
| **值元编程** | 值 | 值 | `sqrt(16)` → `4` |
| **类型元编程** | 类型 | 类型 | `RemoveAllExtents<T>` |
| **混合元编程** | 值+类型 | 值+类型 | `DotProductT<N>::result(a,b)` |
