# C++ Templates 第 27 章：表达式模板

## 章节概述

表达式模板消除数值计算中的**临时数组**，将表达式编码到类型中，赋值时逐元素一次性计算。

---

## 27.1 动机

```cpp
x = 1.2 * x + x * y;  // x, y 是大数组
```

**朴素实现的问题：**
1. `tmp1 = 1.2 * x` — 创建临时数组，遍历一次
2. `tmp2 = x * y` — 创建临时数组，遍历一次
3. `x = tmp1 + tmp2` — 创建临时数组，遍历一次

共 **3 个临时数组，3 次遍历**。表达式模板将其变为 **0 个临时数组，1 次遍历**。

---

## 27.2 SArray（简单数组）

```cpp
template<typename T>
class SArray {
    T* storage;
    std::size_t storage_size;
public:
    explicit SArray(std::size_t s) : storage(new T[s]), storage_size(s) { init(); }
    SArray(SArray const& orig) : storage(new T[orig.size()]), storage_size(orig.size()) { copy(orig); }
    ~SArray() { delete[] storage; }
    std::size_t size() const { return storage_size; }
    T const& operator[](std::size_t idx) const { return storage[idx]; }
    T& operator[](std::size_t idx) { return storage[idx]; }
};
```

朴素操作符（低效）：

```cpp
template<typename T>
SArray<T> operator+(SArray<T> const& a, SArray<T> const& b) {
    SArray<T> result(a.size());
    for (std::size_t k = 0; k < a.size(); ++k)
        result[k] = a[k] + b[k];
    return result;  // 返回临时数组！
}
```

---

## 27.3 表达式模板实现

### 27.3.1 表达式操作数

```cpp
// 加法表达式
template<typename T, typename OP1, typename OP2>
class A_Add {
    typename A_Traits<OP1>::ExprRef op1;  // 可能是引用或值
    typename A_Traits<OP2>::ExprRef op2;
public:
    A_Add(OP1 const& a, OP2 const& b) : op1(a), op2(b) {}
    T operator[](std::size_t idx) const { return op1[idx] + op2[idx]; }
    std::size_t size() const { return op1.size() != 0 ? op1.size() : op2.size(); }
};

// 乘法表达式
template<typename T, typename OP1, typename OP2>
class A_Mult {
    typename A_Traits<OP1>::ExprRef op1;
    typename A_Traits<OP2>::ExprRef op2;
public:
    A_Mult(OP1 const& a, OP2 const& b) : op1(a), op2(b) {}
    T operator[](std::size_t idx) const { return op1[idx] * op2[idx]; }
    std::size_t size() const { return op1.size() != 0 ? op1.size() : op2.size(); }
};

// 标量包装器
template<typename T>
class A_Scalar {
    T const& s;
public:
    constexpr A_Scalar(T const& v) : s(v) {}
    constexpr T const& operator[](std::size_t) const { return s; }
    constexpr std::size_t size() const { return 0; }
};
```

**A_Traits：控制存储方式**

```cpp
// 默认：存储为 const 引用（避免拷贝）
template<typename T>
class A_Traits { public: using ExprRef = T const&; };

// 标量：存储为值（标量很小，且避免悬空引用）
template<typename T>
class A_Traits<A_Scalar<T>> { public: using ExprRef = A_Scalar<T>; };
```

### 27.3.2 Array 模板

```cpp
template<typename T, typename Rep = SArray<T>>
class Array {
    Rep expr_rep;  // 可以是实际存储，也可以是表达式
public:
    explicit Array(std::size_t s) : expr_rep(s) {}
    Array(Rep const& rb) : expr_rep(rb) {}

    // 赋值：逐元素计算表达式
    template<typename T2, typename Rep2>
    Array& operator=(Array<T2, Rep2> const& b) {
        assert(size() == b.size());
        for (std::size_t idx = 0; idx < b.size(); ++idx)
            expr_rep[idx] = b[idx];
        return *this;
    }

    std::size_t size() const { return expr_rep.size(); }
    decltype(auto) operator[](std::size_t idx) const { return expr_rep[idx]; }
    Rep const& rep() const { return expr_rep; }
    Rep& rep() { return expr_rep; }
};
```

### 27.3.3 运算符重载

```cpp
// 加法：返回表达式类型，不计算
template<typename T, typename R1, typename R2>
Array<T, A_Add<T, R1, R2>>
operator+(Array<T, R1> const& a, Array<T, R2> const& b) {
    return Array<T, A_Add<T, R1, R2>>(A_Add<T, R1, R2>(a.rep(), b.rep()));
}

// 标量乘法
template<typename T, typename R2>
Array<T, A_Mult<T, A_Scalar<T>, R2>>
operator*(T const& s, Array<T, R2> const& b) {
    return Array<T, A_Mult<T, A_Scalar<T>, R2>>(
        A_Mult<T, A_Scalar<T>, R2>(A_Scalar<T>(s), b.rep()));
}
```

---

## 27.4 工作原理

对于 `x = 1.2 * x + x * y`，编译器构建的类型：

```
Array<double,
    A_Add<double,
        A_Mult<double, A_Scalar<double>, SArray<double>>,
        A_Mult<double, SArray<double>, SArray<double>>>>
```

赋值时，`operator[]` 递归展开表达式树：

```
x[idx] = (1.2 * x[idx]) + (x[idx] * y[idx])
```

**零临时数组，一次遍历。**

---

## 27.5 性能与限制

| | 朴素实现 | 表达式模板 |
|---|---|---|
| **临时数组** | N-1 个（N 是操作符数） | 0 个 |
| **遍历次数** | N 次 | 1 次 |
| **编译时间** | 快 | 慢（深度嵌套模板实例化） |
| **代码体积** | 小 | 可能膨胀 |

**限制：**
- 不适用于 `x = A * x`（矩阵-向量乘法，结果元素依赖于输入的所有元素，需要临时存储）
- 深度嵌套的模板类型可能导致编译器错误信息难以阅读
- 需要小心处理别名问题

---

## 27.6 历史

- Todd Veldhuizen 和 David Vandevoorde 独立开发
- `std::valarray` 可以使用类似技术
- Boost.Lambda 和 Boost.Proto 进一步发展了表达式模板思想
- Eric Niebler 的 Ranges 库也使用了表达式模板技术
