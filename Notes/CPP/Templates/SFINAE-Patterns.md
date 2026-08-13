# SFINAE 模式总结

## 什么是 SFINAE

**SFINAE**（Substitution Failure Is Not An Error）：模板参数替换失败不是错误，只是该候选被忽略。

```cpp
template<typename T> RT1 test(typename T::X const*);  // T 没有 X？→ SFINAE，丢弃
template<typename T> RT2 test(...);                     // 回退

test<int>(nullptr);  // 第一个被丢弃，选第二个
```

> [!danger] 限制：SFINAE 仅在**立即上下文**中生效
> 函数体内的错误、默认参数、异常规格中的错误是**真正的编译错误**，不会被 SFINAE 捕获。

```cpp
template<typename T> auto f(T p) { return p->m; }  // 函数体内错误，不是 SFINAE！
int f(...);
template<typename T> auto g(T p) -> decltype(f(p)); // ERROR: 实例化 f 时报错

// 正确做法：用尾返回类型
template<typename T> auto len(T const& t) -> decltype(t.size(), T::size_type()) {
    return t.size();  // 如果 t 没有 size()，SFINAE 在声明处就生效
}
```

---

## 模式一：`enable_if`（C++11）

最基础的 SFINAE 工具，通过返回类型或默认模板参数控制函数是否参与重载。

```cpp
// 写法 1：返回类型
template<typename T>
typename std::enable_if<(sizeof(T) > 4)>::type foo(T t) { }

// 写法 2：默认模板参数（更常用）
template<typename T, typename = std::enable_if_t<(sizeof(T) > 4)>>
void foo(T t) {}

// 写法 3：类型别名简化
template<typename T>
using EnableIfLarge = std::enable_if_t<(sizeof(T) > 4)>;

template<typename T, typename = EnableIfLarge<T>>
void foo(T t) {}
```

**适用场景：** 简单的类型约束（大小、是否算术类型等）。

---

## 模式二：返回类型 SFINAE（尾返回类型 + `decltype`）

通过返回类型中的 `decltype` 表达式触发 SFINAE。

```cpp
// 经典：begin() 同时支持容器和数组
template<typename T, unsigned N>
T* begin(T (&array)[N]) { return array; }

template<typename Container>
typename Container::iterator begin(Container& c) { return c.begin(); }
```

**更安全的写法（用尾返回类型）：**

```cpp
template<typename T>
auto len(T const& t) -> decltype((void)(t.size()), T::size_type()) {
    return t.size();
}
```

> [!tip] 为什么用尾返回类型？
> `auto f(T) { return expr; }` 需要实例化函数体，不属于立即上下文，SFINAE 不生效。`-> decltype(expr)` 在声明处就能判断，属于立即上下文。

---

## 模式三：双 `test()` 重载（类型特征的标准套路）

实现类型特征（如 `IsConvertibleT`、`IsSameT`）的标准模式。

```cpp
template<typename FROM, typename TO>
struct IsConvertibleHelper {
private:
    static void aux(TO);

    // 成功路径：SFINAE 通过
    template<typename F, typename,
             typename = decltype(aux(std::declval<F>()))>
    static std::true_type test(void*);

    // 回退路径：总是可用
    template<typename, typename>
    static std::false_type test(...);

public:
    using Type = decltype(test<FROM>(nullptr));
};

template<typename FROM, typename TO>
struct IsConvertibleT : IsConvertibleHelper<FROM, TO>::Type {};
```

**关键技巧：**
- `std::declval<F>()`：不构造对象，生成假值
- 模板参数 `F` 而非直接用 `FROM`：延迟替换到 SFINAE 上下文
- `void*` vs `...`：`void*` 精确匹配优先，`...` 是最低优先级回退

**为什么需要模板参数 `F`？**

```cpp
// ❌ 错误写法：FROM 在类模板解析时已确定，替换失败不是 SFINAE
template<typename = decltype(aux(std::declval<FROM>()))>
static std::true_type test(void*);

// ✅ 正确写法：F 是函数模板参数，延迟到调用时替换
template<typename F, typename,
         typename = decltype(aux(std::declval<F>()))>
static std::true_type test(void*);
```

---

## 模式四：偏特化 + SFINAE（类模板特征）

用偏特化在类模板上做 SFINAE。

```cpp
// IsSameT：偏特化匹配时继承 true_type
template<typename T1, typename T2>
struct IsSameT : std::false_type {};

template<typename T>
struct IsSameT<T, T> : std::true_type {};

// IsConvertibleT 的特殊情况处理：布尔参数 + 偏特化
template<typename FROM, typename TO,
         bool = IsVoidT<TO>::value || IsArrayT<TO>::value>
struct IsConvertibleHelper {
    using Type = std::integral_constant<bool,
        IsVoidT<TO>::value && IsVoidT<FROM>::value>;
};

template<typename FROM, typename TO>
struct IsConvertibleHelper<FROM, TO, false> {
    // 正常的 SFINAE 实现
};
```

---

## 模式五：`void_t`（C++17，检测表达式合法性）

`std::void_t<T...>` 将任意类型映射为 `void`，配合偏特化检测表达式是否合法。

```cpp
// 检测类型是否有 size() 成员
template<typename, typename = void>
struct HasSize : std::false_type {};

template<typename T>
struct HasSize<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

HasSize<std::vector<int>>::value  // true
HasSize<int>::value               // false

// 检测是否有 value_type 成员
template<typename, typename = void>
struct HasValueType : std::false_type {};

template<typename T>
struct HasValueType<T, std::void_t<typename T::value_type>>
    : std::true_type {};
```

> [!tip] void_t 的原理
> `std::void_t<T...>` 定义为 `template<typename...> using void_t = void;`。当 `decltype(...)` 表达式合法时，偏特化匹配成功；否则 SFINAE 丢弃偏特化，回退到主模板。

---

## 模式六：Concepts（C++20，SFINAE 的终极替代）

C++20 用 `concept` 和 `requires` 彻底取代复杂的 SFINAE。

```cpp
// 定义概念
template<typename T>
concept Hashable = requires(T a) {
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept Sortable = requires(T a) {
    typename T::value_type;
    { a.begin() } -> std::input_or_output_iterator;
    { a.end() } -> std::input_or_output_iterator;
};

// 使用方式 1：requires 子句
template<typename T>
requires Hashable<T>
void process(T val) { }

// 使用方式 2：简写语法
void process(Hashable auto val) { }

// 使用方式 3：概念约束模板参数
template<Sortable T>
void mySort(T& container) { std::sort(container.begin(), container.end()); }
```

**Concepts vs SFINAE 对比：**

| | SFINAE | Concepts |
|---|---|---|
| **可读性** | 嵌套模板参数，难以理解 | 语义清晰 |
| **错误信息** | 冗长、难懂 | 简洁、明确 |
| **偏序规则** | 手动控制优先级 | 自动子化（subsumption） |
| **适用版本** | C++11 起 | C++20 起 |

---

## 速查表：该用哪种模式？

| 场景 | 推荐模式 |
|------|---------|
| 简单条件启用/禁用函数 | `enable_if` |
| 检测表达式是否合法（C++17） | `void_t` |
| 实现类型特征（如 `is_convertible`） | 双 `test()` 重载 |
| 类模板的条件特化 | 偏特化 + SFINAE |
| 需要干净的错误信息（C++20） | Concepts |
| 函数重载的条件约束（C++20） | `requires` 子句 |

---

## 完整示例：手写 `std::is_convertible`

```cpp
#include <type_traits>
#include <utility>

// 基础实现
template<typename FROM, typename TO>
struct IsConvertibleHelper {
private:
    static void aux(TO);

    template<typename F, typename,
             typename = decltype(aux(std::declval<F>()))>
    static std::true_type test(void*);

    template<typename, typename>
    static std::false_type test(...);

public:
    using Type = decltype(test<FROM>(nullptr));
};

// 处理特殊情况：void、数组、函数
template<typename FROM, typename TO,
         bool = std::is_void_v<TO>
             || std::is_array_v<TO>
             || std::is_function_v<TO>>
struct IsConvertibleHelper2 {
    using Type = std::integral_constant<bool,
        std::is_void_v<TO> && std::is_void_v<FROM>>;
};

template<typename FROM, typename TO>
struct IsConvertibleHelper2<FROM, TO, false>
    : IsConvertibleHelper<FROM, TO> {};

// 最终特征
template<typename FROM, typename TO>
struct IsConvertibleT : IsConvertibleHelper2<FROM, TO>::Type {};

template<typename FROM, typename TO>
constexpr bool isConvertible = IsConvertibleT<FROM, TO>::value;

// 测试
static_assert(isConvertible<int, double>);
static_assert(isConvertible<char const*, std::string>);
static_assert(!isConvertible<std::string, char const*>);
static_assert(isConvertible<void, void>);
static_assert(!isConvertible<int, void>);
```
