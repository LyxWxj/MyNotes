# C++ Templates 完全指南

## 基础部分

### 多模板参数

> [!tip] 返回类型推导
> 当希望自动推导返回类型时，有以下几种写法：

**方法一：通过构造表达式推导返回类型**

```cpp
template <typename T1, typename T2>
auto f0(T1 A, T2 B) -> decltype(A > B ? A : B) {
  return A > B ? A : B;
}
```

**方法二：使用 `std::decay` 处理引用类型**

```cpp
template <typename T1, typename T2>
auto f1(T1 A, T2 B) -> typename std::decay<decltype(A > B ? A : B)>::type{
  return A > B ? A : B;
}
```

**方法三：使用 `std::common_type` 推导共同类型**

```cpp
template <typename T1, typename T2>
auto f2(T1 A, T2 B) -> typename std::common_type<T1, T2>::type {
  return A > B ? A : B;
}
```

**方法四：使用默认模板参数**

```cpp
template <typename T1, typename T2,
  typename RT = std::decay_t<decltype(A > B ? A : B)>
>
auto f3(T1 A, T2 B) -> RT {
  return A > B ? A : B;
}
```

> [!warning] typename 关键字
> 由于在编译时不能确定 `common_type::type` 是否是一个类型，我们需要显式地加上 `typename` 来告诉编译器它是一个类型。

---

### 模板的重载

> [!info] 重载规则
> 不同模板参数个数视为重载

```cpp
template<typename T1, typename T2>
auto max(T1 a, T2 b) {
  return a > b ? a : b;
}

template<typename RT, typename T1, typename T2>
RT max(T1 a, T2 b) {
  return a > b ? a : b;
}

auto a = ::max(4, 7.2); // 提供两个参数，可以推断两个参数，使用第一个函数模板
auto b = ::max<long double>(4.2, 7); // 提供一个模板参数，提供两个实参可以推导出两个参数，使用第二个模板
auto c = ::max<int>(4, 7.2); // ERROR: 两个重载都匹配
```

> [!danger] 重要注意事项
> 需要从左到右按顺序推断的模板参数，如果自己显式提供的第一个模板参数和需要推断的实参类型相同，那么他会认为这两个都对应到第一个模板参数上，即实际上只提供了一个模板参数，导致两个重载都匹配。

---

### 内联与编译时计算

```cpp
template<typename T1, typename T2>
auto g1(T1 a, T2 b) -> decltype(a > b ? a : b) {
  return a > b ? a : b;
}

template<typename T1, typename T2>
constexpr auto g2(T1 a, T2 b) -> decltype(a > b ? a : b) {
  return a > b ? a : b;
}

int a[g1(sizeof(int), 1000u)]; // error
int b[g2(sizeof(int), 1000u)]; // ok
```

> [!tip] constexpr 的优势
> `constexpr` 函数可以尽量在编译期计算结果（当参数都在编译期可知时），而普通函数只能在运行时计算。

---

### 类模板

```cpp
template<typename T>
class Stack {
  // ...
};

template<class T>
class Stack{
  // ...
};

template<typename T>
void Stack<T>::f(/*...*/) {
  // ...
}
```

> [!info] 类模板中的类型推断
> 在类模板中使用不带模板参数的类名，表示这个内部类的模板参数和模板类相同

**示例：**

```cpp
template<typename T>
class Stack {
public:
  // ...
  Stack (Stack const&);
  Stack& operator=(Stack const&);
  // ...
}
```

等同于：

```cpp
template<typename T>
class Stack {
public:
  // ...
  Stack (Stack<T> const&);
  Stack& operator=(Stack<T> const&);
  // ...
}
```

---

### 模板的特化

```cpp
template<typename T>
class Stack {
  // ...
};  

template<>
class Stack<bool> {
  // ... 
};
```

> [!warning] 特化限制
> 特化的模板参数列表必须和原模板参数列表完全匹配，不能有默认参数，不能有模板参数包。

---

### 偏特化

> [!example] 偏特化示例
> 部分特化参数或者特化部分参数

```cpp
template<typename T>
class Stack<T*>{
  // ...
};

template<typename T1, typename T2>
class Myclass{};

template<typename T>
class Myclass<T, T>{ // 偏特化
  // ...
};

template<typename T1, typename T2>
class Myclass<T1*, T2*>{ // 偏特化
  // ...
};

template<typename T1>
class Myclass<T1, int>{
  // ...
};
```

---

### 类型别名

```cpp
typedef std::stack<int> intstack;
using intstack = std::stack<int>;
```

> [!tip] 别名模板
> `using` 可以模板化，这种方式被称为别名模板：

```cpp
template<typename T>
using DequeStack = std::stack<T, std::deque<T>>;
```

可以用一个参数来绑定两个模板参数：

```cpp
template<typename T> struct MyType {
  using iterator=...;
};

template<typename T>
using MyTypeIter = typename MyType<T>::iterator;
```

> [!warning] typename 的必要性
> 因为不能确定 `MyType<T>::iterator` 是否是一个类型，所以需要加上 `typename` 来告诉编译器它是一个类型。

---

### 类模板的参数推导

> [!info] C++17 改进
> C++17 前，必须将所有模板参数类型传递给类模板（除非有默认值）。C++17 后，指定模板参数的约束放宽了。相反，若构造函数能够推导出所有模板参数（没有默认值），则可以不用显式定义模板参数。

```cpp
stack<int> s1; // stack of ints
stack<int> s2 = s1; // copy constructor
stack s3 = s1; // OK since C++17
auto s4 = s1; // OK 自动推断整个类型
vector v1{1, 2, 3}; // 自动推导类型
```

**推导的结果和模板类的构造函数接受的参数有关：**

```cpp
stack(T const& elem) // reference
stack stringstack="stringstack"; // 推导结果：stack<char const[]>

stack(T elem) // by value
stack stringstack="stringstack"; // 推导结果：stack<char const*>
```

**推导指引（Deduction Guides）：**

```cpp
stack(char const*)->stack<std::string>;
stack stringStack{"bottom"}; // stack<string> deduced and valid;
stack stringStack={"bottom"}; // stack<string> deduced and valid;
stack stringStack2 = "bottom"; // Error
```

---

### 模板聚合

> [!note] 聚合类定义
> 聚合类（不由用户提供、显式或继承的构造函数的类/结构，没有 private 或 protected 的非静态数据成员，没有虚函数，也没有 virtual、private 或 protected 基类）也可以是模板

```cpp
template<typename T>
struct S {
  T value;
  string comment;
};

S(const char*, const char*)->S<std::string>;
```

---

### 非类型模板参数

> [!info] 非类型模板参数
> 对于函数和类模板来说，模板参数可以是类型，也可以是普通值。与使用类型参数的模板一样，定义在使用之前。使用这样的模板时，必须显式地指定值。

**限制条件：**
- 只能是整型常量值（包括枚举）
- 指向对象/函数/成员的指针
- 指向对象或函数的左值引用
- 或者 `std::nullptr_t`（nullptr 的类型）

> [!danger] 不允许的类型
> 浮点数和类型对象不允许作为非类型模板参数

```cpp
template<double VAT> // ERROR: floating-point values are not allowed as template parameters
double process (double v) {
  return v*VAT;
}

template<std::string name> // ERROR: class-type objects are not allowed as template parameters
class Myclass {
};
```

**C++17 自动推断非类型模板参数：**

```cpp
template<typename T, auto Maxsize>
class stack{
  using size_type = decltype(Maxsize);
  std::array<T,Maxsize> elems;
}

stack<int, 100> s1; // ints20
stack<int, 100u> s2; // ints20
```

---

### 可变参数模板

> [!example] 基本用法
> 最简单的例子，递归调用 `print()` 来打印所有参数：

```cpp
void print() {}
template<typename T, typename... Args>
void print(T&& first, Args&&... args) {
  std::cout << first << '\n';
  print(args...);
}
```

**参数包语法说明：**
- `typename...` 中的 `...` 表示这是一个模板参数包
- `Args&&...` 和 `args...` 表示对参数包解包

**使用 `sizeof...` 获取参数包中参数的数量：**

```cpp
template<typename T, typename... Types>
void print(T&& first, Types... args) {
  cout << first << '\n';
  if( sizeof... (args) > 0) {
    print(args...);
  }
}
```

**C++17 折叠表达式：**

```cpp
template<typename ... T>
auto foldsum(T... s) {
  return (s + ...);
}
```

> [!warning] 空参数包
> 如果参数包为空，表达式是错误格式的。

**折叠表达式类型：**
- 一元左折叠: `( … op pack )`
- 一元右折叠: `( pack op … )`
- 二元左折叠: `( init op … op pack )`
- 二元右折叠: `( pack op … op init )`

几乎所有的二元运算符都可以用折叠表达式。

**成员指针示例：**

```cpp
struct Node{
  int v;
  Node* left, *right;
  Node(int i = 0):v(i), left(nullptr), right(nullptr) {};
};

template<typename T, typename... TP>
Node* traverse(T np, TP... paths) {
  return (np->*...->*paths); // np->*paths1->*paths2
}
```

> [!tip] 成员指针理解
> `traverse` 能够传入 `&Node::left`、`&Node::right` 这样的参数，是因为它们并不是**某个具体对象的成员变量的地址**，而是**成员指针（pointer to member）**。

**关键点：**
- **成员指针**（如 `&Node::left`）在编译时就已经确定，它表示该成员在类 `Node` 中的**偏移量**（或称为"成员描述符"），而不是内存中某个实际存在的变量的地址。
- 成员指针可以独立于任何对象存在，因此即使没有创建任何 `Node` 对象，`&Node::left` 本身也是合法且有效的编译时常量。
- 在 `traverse` 函数中，通过 `np->*...` 这样的语法，将成员指针与具体的对象 `np` 相结合，才能得到该对象中对应成员的实际地址或引用。

**简单类比：**
- 普通指针：`int x = 5; int* p = &x;` —— `p` 指向**具体的变量** `x` 的地址，如果 `x` 不存在，`&x` 就是非法的。
- 成员指针：`int Node::* p = &Node::v;` —— `p` 只是记录了 `v` 在 `Node` 类中的位置，它不需要有任何 `Node` 对象就已经存在。

因此，`&Node::left` 并不是"不存在的变量的地址"，而是一个**类型安全的偏移量**，它本身是合法的常量，当然可以传递给函数模板。

**使用初始化器的折叠表达式简化打印：**

```cpp
template<typename... Args>
void print(Args&&... args) {
  (std::cout << ... << args) << '\n';
}
```

**带分隔符的打印：**

```cpp
template<typename T, char Sep>
class AddSeparator {
private:
  T const& ref;
public:
  AddSeparator(T const& r): ref(r) {}
  friend std::ostream& operator<<(std::ostream& os, AddSeparator<T, Sep> s){
    return os << s.ref << Sep;
  }
}; 

template<typename... Args>
void print(Args&&... args) {
  (std::cout << ... << AddSeparator<Args, ','>(args)) << '\n';
}
```

---

### 类模板和表达式

参数包还可以出现在其他地方，例如表达式、类模板、using 声明，甚至推导策略。

**将每个参数加倍后打印：**

```cpp
template<typename... Args>
void printdouble(Args&&... args){
  print(args + args...);
}
```

**返回所有参数类型是否相同：**

```cpp
template<typename T1, typename... TN>
bool is_sameall(T1, TN...) {
  return (std::is_same<T1,TN>::value && ...);
}

// 或者
template<typename T1, typename... TN>
constexpr bool is_sameall(T1, TN...) {
  return (std::is_same_v<T1, TN> && ...);
}
```

**打印多个索引：**

```cpp
template<typename C, typename... Idx>
void printElems(C const& coll, Idx... idx) {
  print(coll[idx]...);
}

template<size_t... idx, typename C>
void printElems(C const& coll) {
  print(coll[idx]...);
}
```

> [!warning] 模板参数顺序
> 如果 `idx` 作为模板参数，则需要将其提到前面来，因为模板参数必须从左到右依次推断，而形参的类型最后推断，因此作为形参类型的容器应该放在 `idx` 的后面

**使用类型记录索引参数包：**

```cpp
template<std::size_t...>
struct Indices {};

template<typename T, std::size_t... Idx>
void printElems(T const& t, Indices<Idx...>) {
  print(t[Idx]...);
}
```

**推导策略用于可变参数包：**

```cpp
template<typename T, typename... U>
array(T, U...) -> array<enable_if_t<(std::is_same_v<T, U> && ...), T>, (1 + sizeof...(U))>
```

**参数包继承：**

```cpp
template<typename... Bases>
struct overloader: Bases... {
  using Bases::operator()...; // 继承参数包中所有基类的operator()函数
};
```

---

### 基础技巧

#### typename 关键字

> [!warning] typename 的必要性
> 使用某个类内部的类型时需要加上 `typename`，否则编译器会优先假定这是一个静态成员变量

```cpp
template<typename T>
void f() {
  typename T::value_type x; // value_type是T的一个类型成员
}
```

---

#### 零初始化

> [!tip] 内置类型初始化
> 简单的定义对内置类型并没有进行初始化。因此，可以显式调用内置类型的默认构造函数，该构造函数用 0 初始化内置类型（bool 为 false，指针为 nullptr）。

```cpp
template<typename T>
void foo(){
  T x{}; // 零初始化
}
```

**类模板成员初始化：**

```cpp
template<typename T>
class SomeClass {
  private: 
    T x;
  public:
    SomeClass(): x{} {} // 使用大括号初始化成员x
};
```

或者：

```cpp
template<typename T>
class SomeClass {
  private: 
    T x{};
  public:
    SomeClass() {} // 使用大括号初始化成员x
};
```

> [!danger] 默认参数限制
> 默认参数不可以这样写：

```cpp
template<typename T>
void foo(T x{}) {}// False
template<typename T>
void foo(T x = T{}) {} // Correct
```

---

#### .template 成员函数调用

```cpp
template <unsigned long N>
void printBitset(std::bitset<N> const& Bs) {
  std::cout << Bs.template to_string<char, std::char_traits<char>, std::allocator<char>>();
}
```

> [!info] 语法说明
> 对于 `bitset bs`，使用 `to_string()` 的成员函数模板，同时显式指定字符串类型的信息。如果没有使用 `.template`，编译器就不知道后面的小于标记 (`<`) 是模板参数列表的开头。注意，只有在句点之前的构造依赖于模板参数时才会出现问题。在例子中，参数 `bs` 依赖于模板参数 `N`。

---

#### 变量模板

> [!note] 术语区分
> - **变量模板**：是一个变量，它是一个模板（变量在这里是名词）
> - **可变参数模板**：是用于可变数量模板参数的模板（可变参数在这里是形容词）

```cpp
template<typename T>
constexpr T pi = T(3.1415926535897932385);

std::cout << pi<double> << '\n'; // 3.14159
std::cout << pi<int> << '\n'; // 3
```

**变量模板可以有默认模板参数：**

```cpp
template<typename T=long double>
constexpr T pi = T{3.1415926535897932385};
pi<> // pi <long double>
pi<int> //
pi // Error 必须要有<>
```

**变量模板可以用非类型参数进行参数化：**

```cpp
template<int N>
std::array<int,N> arr{};

template<auto N>
constexpr decltype(N) dval=N;
```

**等价写法：**

```cpp
template<auto N>
constexpr decltype(N) dval = N;

template <typename T, T N>
constexpr T dval2 = N;

std::cout << dval<1> << std::endl;
std::cout << dval2<int, 2> << std::endl;
```

---

#### 数据成员的变量模板

> [!example] 静态成员访问
> 如果有某一个模板类中特化不同的静态成员，可以用变量模板取到其中的成员

```cpp
template<typename T>
class MyClass {
  public:
    static constexpr int max=1000;
};

template<>
class MyClass<float> {
  public:
    static constexpr int max = 10;
};

template<typename T>
int myMax = MyClass<T>::max;

auto i = myMax<int>;
auto f = myMax<float>;
```

---

#### 双重模板参数

> [!tip] 类模板作为模板参数
> 允许模板参数本身是类模板

```cpp
stack<int, vector<int>> stk1;
stack<int, vector> stk2;
```

**省略第二个模板参数的写法：**

```cpp
template<typename T,
      template<class E> class Container = std::deque>
class Stack{}; // OK

template<typename T,
      template<typename E> typename Container = std::deque>
class Stack{}; // OK;

template<typename T,
      template<typename E> class Container = std::deque>
class Stack{}; // OK;

template<typename T,
      template<typename> class Container = std::deque>
class Stack{}; // OK;
```

---

### 移动语义与 `enable_if<>`

**完美转发：**

```cpp
template<typename T>
void foo(T&& t) {
  g(std::forward<T>(t));
}
```

**`std::enable_if<cond, T=void>`：在编译时条件下忽略函数模板**

```cpp
template<typename T>
typename std::enable_if<(sizeof (T) > 4)>::type foo(T t) {
  // ...
}
```

> [!info] 工作原理
> 如果 `sizeof(T)>4` 生成 false，则忽略 `foo<>` 的定义。如果结果为 true，函数模板实例展开为：

```cpp
void foo(T t) {};
```

**简便写法：**

```cpp
template<typename T, typename = std::enable_if_t<(sizeof(T) > 4)>>
void foo(){}

// 或者
template<typename T>
using EnableIfSizeGreater4 = std::enable_if_t<(sizeof(T) > 4)>;
template<typename T, typename = EnableIfSizeGreater4>
void foo(){};
```

---

#### 用概念简化 `enable_if<>`

**`requires` 关键字：**

```cpp
template<typename STR>
requires std::is_convertible_v<STR,std::string>
Person(STR&& n):name(std::forward<STR>(n)) {
  // ...
}
```

**`concept` 关键字：**

```cpp
template<typename T>
concept ConvertibleToString = std::is_convertible_v<T,std::string>;

template<typename STR>
requires ConvertibleToString<STR> 
Person(STR&& n): name(std::forward<STR>(n)) {
  // ...
}
```

---

### 模板元编程

> [!example] 素数判断示例
> 使用模板进行素数判断的例子

```cpp
template<unsigned p, unsigned d>
struct DoIsPrime {
  static constexpr bool value = (p%d != 0) && DoIsPrime<p, d-1>::value;
};

template<unsigned p>
struct DoIsPrime<p,2>{
  static constexpr bool value = (p%2 != 0);
};

template<typename p>
struct IsPrime {
  static constexpr bool value = DoIsPrime<p, p/2>::value;
};

template<>
struct IsPrime<0> {static constexpr bool value = false;};
template<>
struct IsPrime<1> {static constexpr bool value = false;};
template<>
struct IsPrime<2> {static constexpr bool value = true;};
template<>
struct IsPrime<3> {static constexpr bool value = true;};
```

> [!tip] C++14 改进
> C++14 中，`constexpr` 函数可以使用通用 C++ 代码中的控制结构。因此，不用编写笨拙的模板代码或有些"奇怪的"单行程序，现在只使用普通的 for 循环：

```cpp
constexpr bool isPrime(unsigned int p) {
  for(unsigned int d=2; d <=p/2; ++ d) {
    if(p%d == 0) {
      return false;
    }
  }
  return p > 1;
}
```

**直接调用：**

```cpp
isPrime(9);
```

**编译时测试应用：**

```cpp
template<int SZ, bool =isPrime(SZ)>
struct Helper;

template<int SZ>
struct Helper<SZ, false> {
  // ...
};

template<int SZ>
struct Helper<SZ,true> {
  // ...
};

template <typename T, std::size_t SZ>
long foo (std::array<T,SZ> const& coll) {
  Helper<SZ> h;
}
```

---

#### SFINAE（替换失败不是错误）

> [!info] SFINAE 原则
> C++ 中，以各种参数类型重载的函数很常见。因此，当编译器看到对重载函数的调用时，必须考虑每个候选函数，评估调用参数，并选择最匹配的候选函数。
> 
> 候选集包括函数模板的情况下，编译器首先必须确定为该候选对象使用哪些模板参数，然后在函数参数列表及其返回类型中替换这些参数，然后评估匹配程度。但替换过程可能会遇到问题：可能产生毫无意义的构造。语言规则并不认为这种无意义的替换会导致错误，而具有这种问题的候选则会直接忽略。
> 
> 这就是所谓的 SFINAE（Substitution Failure Is Not An Error）原则。

**示例：**

```cpp
template<typename T, unsigned N>
std::size_t len(T(&)[N]) {
  return N;
}

template<typename T>
typename T::size_type len(T const& t) {
  return t.size();
}
```

**使用场景：**
1. 第一个函数模板将参数声明为 `T[&](N)`，从而参数必须是由 N 个 T 类型元素组成的数组。
2. 第二个函数模板将参数声明为 `T`，没有对参数施加任何约束，而是返回类型 `T::size_type`，这要求传递的参数类型具有 `size_type` 成员变量。

```cpp
int a[10];
std::cout << len(a); // OK : only len() for array matches
std::cout << len("temp"); // OK : only len() for array matches
std::cout << len(vector<int>{0}); // OK
std::cout << len(allocator<int>{}); // ERROR: len() selected, but x has no size();
```

> [!tip] SFINAE 模式
> 有一种常见的模式或习语可以用来处理这种情况：
> - 用尾部返回类型语法指定返回类型（前面使用 auto，在末尾返回类型之前使用 ->）
> - 使用 `decltype` 和逗号操作符定义返回类型
> - 给出以逗号操作符开头的表达式（在重载逗号操作符时转换为 void）
> - 在逗号操作符的末尾定义一个实际返回类型的对象

```cpp
template<typename T>
auto len( T const& t) -> decltype((void)(t.size()), T::size_type()) {
  return t.size();
}
```

---

### 通用库

**`std::invoke()` 的应用：**

```cpp
#include <utility>
#include <functional>

template<typename Callable, typename... Args>
decltype(auto) call(Callable&& op, Args&&... args) {
  return std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...);
}
```

> [!info] decltype(auto) 的作用
> 为了支持返回引用（比如 `std::ostream&`），必须使用 `decltype(auto)` 而不是 `auto`

**处理 void 返回类型：**

```cpp
#include <utility>
#include <functional>
#include <type_traits>

template<typename Callable, typename... Args>
decltype(auto) call(Callable&& op, Args&&... args) {
  if constexpr(std::is_same_v<std::invoke_result_t<Callable, Args...>, void>) {
    std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...);
    return;
  } else {
    decltype(auto) ret{std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...)};
    return ret;
  }
}
```

**类型特征注意事项：**

```cpp
std::remove_const_t<int const&> // 注意：引用不是const
std::remove_const_t<std::remove_reference_t<int const&>> // int
std::remove_reference_t<int const&> // int const
std::decay_t<int const&> // yields int
```

**`std::addressof<>()` 函数模板：**

```cpp
template<typename T>
void f(T&& x) {
  auto p = &x; // might fail with overloaded operator &
  auto q = std::addressof(x); // works even with overloaded operator &
}
```

**`std::declval<>()` 函数模板：**

```cpp
#include <utility>

template<typename T1, typename T2, 
      typename RT = std::decay_t<decltype(true ? std::declval<T1>() : std::declval<T2>())>
>
RT max(T1 a, T2 b) {
  return b < a ? a : b;
}
```

> [!warning] 引用类型陷阱
> 容易自动产生引用的地方：

```cpp
#include <iostream>

template<typename T>
void tmplParamIsReference(T) {
  std::cout << "T is reference: " << std::is_reference_v<T> << '\n';
}

int main(){
  std::cout << std::boolalpha;
  int i;
  int& r=i;
  tmplParamIsReference(i); // false;
  tmplParamIsReference(r); // false; pass by value
  tmplParamIsReference<int&>(i); // true;
  tmplParamIsReference<int&> (r); // true;
  return 0;
}
```

**引用类型作为模板参数的问题：**

```cpp
template<typename T, T Z = T{}>
class RefMem {
  private: 
    T zero;
  public:
    RefMem():zero(Z) {
    }
};

int null= 0;

int main(){
  RefMem<int> rm1, rm2;
  rm1 = rm2; // OK;
  
  RefMem<int&> rm3; // ERROR
  RefMem<int&, 0> rm4; // ERROR: invalid default value for N
  
  extern int null;
  RefMem<int&, null> rm5,rm6;
  rm5 = rm6;// ERROR: operator= is deleted due to reference member;
}
```

> [!danger] decltype(auto) 的风险
> 使用 `decltype(auto)` 可以很容易地产生引用类型，因此在上下文中最好不要使用（默认使用 `auto`）。

**处理不完整类型：**

```cpp
template<typename T>
class Cont {
  private:
    T* elems;
  public:
    // ...
};

struct Node {
  std::string value;
  Cont<Node> next; // only use Pointer
};
```

> [!warning] 特性限制
> 仅通过使用一些特性，就会失去处理不完整类型的能力。例如：

```cpp
template<typename T>
class Cont {
  private:
    T* elems;
  public:
    typename std::conditional<std::is_move_constructible<T>::value, T&&, T&>::type foo();
}
```

**解决方案：延迟特性计算**

```cpp
template<typename T>
class Cont {
  private:
    T* elems;
  public:
    template<typename D = T> 
    typename std::conditional<std::is_move_constructible<T>::value, T&&, T&>::type foo();
}
```

---

#### 编写泛型库注意事项

> [!tip] 最佳实践
> 实现泛型库时需要记住的事情：
> - 模板中使用转发引用来转发值。如果值不依赖于模板参数，使用 `auto&&`
> - 当参数声明为转发引用时，模板参数在传递左值时要有引用类型
> - 当需要依赖于模板形参的对象地址时，使用 `std::addressof()`
> - 对于成员函数模板，确保不会比预定义的复制/移动构造函数或赋值操作符更好地匹配
> - 模板参数可能是字符串字面值，且不通过值传递时，请考虑使用 `std::decay`
> - 如果模板参数有 `out` 或 `inout`，请准备好处理参数可能指定为 `const` 类型的情况
> - 准备好处理模板参数引用的副作用。特别是，要确保返回类型不能是引用
> - 准备好处理不完全类型，从而进行以支持，例如：递归数据结构
> - 重载所有数组类型，而不仅仅是 `T[SZ]`

---

## 深入了解模板

### 参数化声明

> [!info] 模板类型
> C++ 目前支持四种基本模板：
> 1. 类模板
> 2. 函数模板
> 3. 变量模板
> 4. 别名模板
> 
> 这些模板类型都可以出现在名称空间中，也可以出现在类中。在类作用域中，作为嵌套的类模板、成员函数模板、静态数据成员模板和成员别名模板。

**基本模板声明：**

```cpp
template<typename T> // a namespace scope class template
class Data {
public:
    static constexpr bool copyable = true;
    // ...
};

template<typename T>
void log(T x) {
    // ...
}

template<typename T>
T zero = 0;

template<typename T>
bool dataCopyable = Data<T>::copyable;

template<typename T>
using DataList = Data<T*>;
```

**类模板中的成员模板：**

```cpp
template<typename T> // a namespace scope class template
class List {
public:
    List() = default; // because a template constructor is defined
    template<typename U>
    class Handle; // without its definition
    template<typename U>
    List(List<U> const&);
    template<typename U>
    static U zero;
};

template<typename T> // out-of-class member template definition
    template<typename U>
class List<T>::Handle {
    // ...
};

template<typename T>
    template<typename T2>
List<T>::List (List<T2> const& b) {
    // ...
}

template<typename T>
    template<typename U>
U List<T>::zero = 0;
```

> [!warning] 类外成员模板定义
> 在类外的成员模板需要多个 `template<...>` 参数化子句；每个外围作用域的类模板一个，成员模板本身也需要一个，子句从类模板最外层开始逐行展示

**联合模板：**

```cpp
template<typename T>
union AllocChunk {
    T object;
    unsigned char bytes[sizeof(T)];
};
```

**函数模板声明：**

```cpp
template<typename T>
void report_top(Stack<T> const&, int number=10);

template<typename T>
void fill(Array<T>&, T const& = T{});
```

> [!danger] 虚成员函数限制
> 成员函数模板不能进行虚声明。这个约束是因为虚函数的调用机制实现的，使用一个固定大小的表，每一个虚函数只有一个条目。但是成员函数模板的实例化直到整个程序翻译后才固定。因此支持虚成员函数模板需要在 C++ 编译器和链接器中支持一套全新的机制。

**链接模板：**

```cpp
int C;
class C; // OK: class names and nonclass names are in a different "space"

int X;
template<typename T>
class X; // ERROR: conflict with variable X

struct S;
template<typename T>
class S; // ERROR: conflict with struct S
```

> [!info] 命名规则
> 每个模板都必须有一个名字，并且这个名字在其作用域内必须唯一（除了函数模板可以重载）。与类类型不同，类模板不能与其他类型的实体共享名称。

---

### 模板形参

> [!info] 模板参数的三种基本类型
> 1. **类型参数**（最常见的）
> 2. **非类型参数**
> 3. **双重模板参数**
>
> 这些基本类型的模板参数都可以作为模板参数包的基础。

模板参数在模板声明的介绍性参数化子句中进行声明。声明不一定需要命名：

```cpp
template<typename, int>
class X; // X<> 由一个类型和一个整数参数化
```

若参数在模板中引用，则需要参数名。模板参数名称可以在后续的参数声明中引用（但不能在前面引用）：

```cpp
template<typename T, // 第一个参数被用于
        T Root, // 第二个参数的声明中，以及
        template<T> class Buf> // 第三个参数的声明中
class Structure;
```

---

#### 类型参数

> [!info] typename 与 class 等价
> 类型参数通过关键字 `typename` 或 `class` 引入，两者等价。关键字 `class` 并不意味着替换参数应该是类类型，还可以是一个可访问的类型。

关键字后面必须跟着一个简单的标识符，该标识符后面必须跟着：
- 逗号：表示下一个参数声明的开始
- 结束尖括号（`>`）：表示参数化子句的结束
- 等号（`=`）：表示默认模板参数的开始

> [!warning] 类型参数的使用限制
> 在模板声明中，类型形参的作用类似于类型别名。当 `T` 是模板参数时，**不能**在参数名前再加 `typename` 或 `class` 关键字：

```cpp
template<typename Allocator>
class List {
    class Allocator* allocptr; // ERROR: 应使用 "Allocator* allocptr"
    friend class Allocator;    // ERROR: 应使用 "friend Allocator"
    // ...
};
```

---

#### 非类型参数

> [!info] 非类型模板参数
> 非类型模板参数表示可在编译或链接时确定的**常量**。

**允许的非类型参数类型：**

- 整数类型或枚举类型
- 指针类型（指向对象的指针和指向函数的指针）
- 成员指针类型
- 左值引用类型（对对象的引用和对函数的引用都可以）
- `std::nullptr_t`
- 包含 `auto` 或 `decltype(auto)` 的类型（仅 C++17）

> [!danger] 不允许的类型
> 所有其他类型目前都排除在外（浮点类型可能会在将来添加）。

**非类型参数声明示例：**

```cpp
template<typename T,                    // 类型参数
        typename T::Allocator* Allocator> // 非类型参数
class List;
```

**函数和数组类型的隐式衰变：**

```cpp
template<int buf[5]> class Lexer;     // buf 实际上是 int*
template<int* buf> class Lexer;       // OK: 这是一个重新声明
template<int fun()> struct FuncWrap;  // fun 实际上是函数指针类型
template<int (*)()> struct FuncWrap;  // OK: 这是一个重新声明
```

> [!warning] 非类型参数的修饰符限制
> 非类型模板参数的声明很像变量，但它们**不能有** `static`、`mutable` 等非类型修饰符。可以有 `const` 和 `volatile` 限定符，但若出现在参数类型的最外层则会被忽略：

```cpp
template<int const length> class Buffer; // const 无效
template<int length> class Buffer;       // 与上一行相同
```

> [!tip] 值类别
> 非引用非类型参数在表达式中使用时，始终是 **prvalue**（纯右值）。地址不能取走，也不能赋值。左值引用类型的非类型参数可用于表示左值：

```cpp
template<int& Counter>
struct LocalIncrement {
    LocalIncrement() { Counter = Counter + 1; } // OK: 引用一个整数
    ~LocalIncrement() { Counter = Counter - 1; }
};
```

---

#### 双重模板参数

> [!info] 双重模板参数
> 双重参数是类模板或别名模板的占位符。声明很像类模板，但关键字 `struct` 和 `union` **不能使用**：

```cpp
template<template<typename X> class C>   // OK
void f(C<int>* p);

template<template<typename X> struct C>  // ERROR: struct 不合法
void f(C<int>* p);

template<template<typename X> union C>   // ERROR: union 不合法
void f(C<int>* p);
```

> [!tip] C++17 改进
> C++17 允许使用 `typename` 替代 `class`，因为双重模板参数不仅可以使用类模板替换，还可以使用别名模板替换：

```cpp
template<template<typename X> typename C> // OK since C++17
void f(C<int>* p);
```

**双重模板参数的默认值：**

```cpp
template<template<typename T,
                  typename A = MyAllocator> class Container>
class Adaptation {
    Container<int> storage; // 隐式等价于 Container<int, MyAllocator>
    // ...
};
```

> [!warning] 参数名作用域限制
> `T` 和 `A` 是模板参数 `Container` 的模板参数名，只能在该模板参数的其他参数声明中使用：

```cpp
template<template<typename T, T*> class Buf> // OK
class Lexer {
    static T* storage; // ERROR: 双重模板参数不能在这里使用
    // ...
};
```

---

#### 模板参数包

> [!info] 模板参数包（C++11）
> 任何类型的模板参数都可以通过在模板参数名之前引入省略号（`...`）转换为模板参数包。普通模板参数只能匹配一个模板参数，而模板参数包可以匹配**任意数量**的模板参数。

```cpp
template<typename... Types> // 声明一个名为 Types 的模板参数包
class Tuple;

using IntTuple = Tuple<int>;              // OK: 一个模板参数
using IntCharTuple = Tuple<int, char>;    // OK: 两个模板参数
using IntTriple = Tuple<int, int, int>;   // OK: 三个模板参数
using EmptyTuple = Tuple<>;               // OK: 零个模板参数
```

**非类型和双重模板参数包：**

```cpp
template<typename T, unsigned... Dimensions>
class MultiArray; // OK: 声明一个非类型模板参数包

using TransformMatrix = MultiArray<double, 3, 3>; // OK: 3x3 矩阵

template<typename T, template<typename,typename>... Containers>
void testContainers(); // OK: 声明一个双重模板参数包
```

> [!warning] 模板参数包的位置限制
> 主模板、变量模板和别名模板最多可以有一个模板参数包。函数模板有一个较弱的限制：若模板参数包是最后一个参数，则允许多个模板参数包，只要后续参数有默认值或可以推导：

```cpp
template<typename... Types, typename Last>
class LastType; // ERROR: 参数包不是最后一个参数

template<typename... TestTypes, typename T>
void runTests(T value); // OK: 参数包后面是可推导的参数

template<unsigned... Dims1, unsigned... Dims2>
auto compose(Tensor<Dims1...>, Tensor<Dims2...>); // OK: 可推导
```

> [!tip] 偏特化中的多个参数包
> 类和变量模板的偏特化声明可以有多个参数包，因为偏特化通过推导选择，推导过程与函数模板相同：

```cpp
template<typename...> struct Typelist;
template<typename X, typename Y> struct Zip;
template<typename... Xs, typename... Ys>
struct Zip<Typelist<Xs...>, Typelist<Ys...>>; // OK
```

**参数包不能在自身参数子句中展开：**

```cpp
template<typename... Ts, Ts... vals> struct StaticValues {};
// ERROR: Ts 不能在自己的参数列表中展开
```

**解决方案：使用嵌套模板：**

```cpp
template<typename... Ts> struct ArgList {
    template<Ts... vals> struct Vals {};
};
ArgList<int, char, char>::Vals<3, 'x', 'y'> tada; // OK
```

---

#### 默认模板参数

> [!info] 默认模板参数规则
> 非模板参数包中的模板参数都可以配备一个默认参数，必须在类型上与相应的参数匹配（例如，类型参数不能有非类型的默认参数）。默认参数**不能依赖于自己的参数**，因为参数名直到默认参数之后才在作用域中，但**可以依赖于之前的参数**：

```cpp
template<typename T, typename Allocator = allocator<T>>
class List;
```

**类模板的默认参数必须从右到左提供：**

```cpp
template<typename T1, typename T2, typename T3,
    typename T4 = char, typename T5 = char>
class Quintuple; // OK

template<typename T1, typename T2, typename T3 = char,
    typename T4, typename T5>
class Quintuple; // OK: T4 和 T5 已经有默认值

template<typename T1 = char, typename T2, typename T3,
    typename T4, typename T5>
class Quintuple; // ERROR: T1 不能有默认值，因为 T2 没有
```

> [!tip] 函数模板的特殊性
> 函数模板的参数默认模板参数**不需要**后续模板参数有默认值，因为后续参数可以通过推导确定：

```cpp
template<typename R = void, typename T>
R* addressof(T& value); // OK: 如果未显式指定，R 将为 void
```

**默认模板参数不能重复：**

```cpp
template<typename T = void>
class Value;

template<typename T = void>
class Value; // ERROR: 重复的默认参数
```

**不允许默认模板参数的上下文：**

1. **偏特化：**
```cpp
template<typename T> class C;
template<typename T = int> class C<T*>; // ERROR
```

2. **参数包：**
```cpp
template<typename... Ts = int> struct X; // ERROR
```

3. **类模板成员类外定义：**
```cpp
template<typename T> struct X { T f(); };
template<typename T = int> T X<T>::f() { /*...*/ } // ERROR
```

4. **友元类模板声明：**
```cpp
struct S {
    template<typename = void> friend struct F; // ERROR
};
```

5. **友元函数模板声明**（除非是定义，且在编译单元其他地方没有声明）：
```cpp
struct S {
    template<typename = void> friend void f(); // ERROR: 不是定义
    template<typename = void> friend void g() {} // OK
};
template<typename> void g(); // ERROR: g() 在定义时已有默认参数
```

---

### 模板实参

> [!info] 模板实参的确定机制
> 实例化模板时，需要确定模板参数。参数可以通过以下几种机制确定：
> 1. **显式模板参数**：模板名称后面跟着用尖括号括起来的显式模板参数，产生的名称称为 template-id
> 2. **注入类名**：类模板 `X` 的作用域内，模板参数 P1, P2, …，该模板的名称 `X` 等价于 `X<P1, P2, …>`
> 3. **默认模板参数**：若默认模板参数可用，则可以省略。但对于类模板或别名模板，即使所有参数都有默认值，也必须提供（可能为空的）尖括号
> 4. **参数类型推导**：未显式指定的函数模板参数可以从调用中的函数参数类型推导出来。C++17 还引入了类模板参数的推导

---

#### 函数模板参数

函数模板参数可以显式指定，可以根据使用方式推导，也可以作为默认模板参数提供：

```cpp
template<typename T>
T max(T a, T b) {
    return b < a ? a : b;
}

int main() {
    ::max<double>(1.0, -3.0); // 显式指定模板参数
    ::max(1.0, -3.0);         // 模板参数隐式推导为 double
    ::max<int>(1.0, 3.0);     // 显式 <int> 抑制推导；结果类型为 int
}
```

> [!warning] 不可推导参数的位置
> 有些模板参数永远无法推导（对应的参数没有出现在函数参数类型中，或其他原因）。相应的参数通常放在模板参数列表的**开头**，以便在推导其他参数的同时显式指定：

```cpp
template<typename DstT, typename SrcT>
DstT implicit_cast(SrcT const& x) // SrcT 可推导，DstT 不可推导
{
    return x;
}

int main() {
    double value = implicit_cast<double>(-1); // OK
}
```

> [!danger] 不可推导参数的限制
> 若颠倒参数顺序（`template<typename SrcT, typename DstT>`），则调用必须显式指定**两个**参数。此外，不可推导参数不能放在参数包之后，也不能出现在偏特化中：

```cpp
template<typename... Ts, int N>
void f(double (&)[N+1], Ts... ps); // 无用声明：N 无法指定或推导
```

**函数模板重载与参数匹配：**

```cpp
template<typename Func, typename T>
void apply(Func funcPtr, T x) { funcPtr(x); }

template<typename T> void single(T);
template<typename T> void multi(T);
template<typename T> void multi(T*);

int main() {
    apply(&single<int>, 3); // OK: single<int> 类型明确
    apply(&multi<int>, 7);  // ERROR: multi<int> 不是唯一的函数
}
```

> [!tip] SFINAE 原则
> 在函数模板中替换模板参数可能导致无效的构造。SFINAE（替换失败不为过）原则确保这种失败不会导致错误，而是简单地忽略该候选：

```cpp
template<typename T> RT1 test(typename T::X const*);
template<typename T> RT2 test(...);

// &test<int> 有效：第一个模板因 int 没有成员类型 X 而被忽略
// 但第二个模板没有问题，表达式仍然有效
```

---

#### 类型模板实参

> [!info] 类型模板实参
> 类型模板实参为类型形参指定的"值"。**任何类型**（包括 `void`、函数类型、引用类型等）都可以用作模板实参，但对形参的替换必须产生有效的构造：

```cpp
template<typename T>
void clear(T p) {
    *p = 0; // 要求一元 * 对 T 可用
}

int main() {
    int a;
    clear(a); // ERROR: int 不支持一元 *
}
```

---

#### 非类型模板实参

> [!info] 非类型模板实参
> 非类型模板实参是替代非类型形参的值。这样的值必须是以下条件之一：
> - 具有正确类型的另一个非类型模板参数
> - 整型（或枚举）的编译时常量值（类型必须匹配或可隐式转换）
> - 外部变量或函数名称前加 `&` 操作符（函数和数组可省略 `&`）
> - 对于引用类型的形参，不带 `&` 的变量名
> - 成员指针常数 `&C::m`
> - 空指针常量（用于指针或指针成员类型）

**示例：**

```cpp
template<typename T, T nontypeParam>
class C;

C<int, 33>* c1;                    // 整型

int a;
C<int*, &a>* c2;                   // 外部变量的地址

void f();
void f(int);
C<void(*)(int), f>* c3;            // 函数名：重载解析选择 f(int)

template<typename T> void templ_func();
C<void(), &templ_func<double>>* c4; // 函数模板实例化

struct X {
    static bool b;
    int n;
    constexpr operator int() const { return 42; }
};

C<bool&, X::b>* c5;               // 静态类成员
C<int X::*, &X::n>* c6;           // 成员指针常量
C<long, X{}>* c7;                  // OK: X 先通过 constexpr 转换为 int，再转为 long
```

> [!danger] 无效的非类型实参
> 目前以下类型的值仍然无效：
> - 浮点数
> - 字符串字面值
>
> 字符串字面值的问题是两个相同的字面值可以存储在不同的地址。

**字符串字面值的变通方法：**

```cpp
template<char const* str>
class Message { /*...*/ };

extern char const hello[] = "Hello World!";
char const hello11[] = "Hello World!";

void foo() {
    static char const hello17[] = "Hello World!";

    Message<hello> msg03;   // 所有版本都 OK
    Message<hello11> msg11; // C++11 起 OK
    Message<hello17> msg17; // C++17 起 OK
}
```

> [!info] 链接要求的演变
> - C++17 之前：要求外部链接
> - C++11 起：允许内部链接
> - C++17 起：允许任意链接方式

**无效示例：**

```cpp
template<typename T, T nontypeParam>
class C;

struct Base { int i; } base;
struct Derived : public Base {} derived;

C<Base*, &derived>* err1;   // ERROR: 不考虑派生类到基类的转换
C<int&, base.i>* err2;      // ERROR: 变量的字段不被视为变量
int arr[10];
C<int*, &arr[0]>* err3;     // ERROR: 数组元素的地址也不可接受
```

---

#### 双重模板实参

> [!info] 双重模板实参匹配规则
> 双重模板实参通常必须是类模板或别名模板，其形参必须与它所替换的双重模板参数的形参精确匹配。**C++17 放宽了匹配规则**，只要求模板形参至少与对应的形参一样特化。

**C++17 之前的问题：**

```cpp
#include <list>
// std::list 声明: template<typename T, typename Allocator = allocator<T>> class list;

template<typename T1, typename T2,
        template<typename> class Cont> // Cont 期望一个参数
class Rel { /*...*/ };

Rel<int, double, std::list> rel; // C++17 前 ERROR: std::list 有多个参数
```

> [!tip] 解决方案：可变参数双重模板参数包
> 可变参数模板参数包可以匹配零个或多个同类模板形参，是"精确匹配"规则的例外：

```cpp
template<typename T1, typename T2,
        template<typename...> class Cont> // Cont 接受任意数量的类型参数
class Rel { /*...*/ };

Rel<int, double, std::list> rel; // OK
```

**`AlmostAnyTmpl` 模式：**

```cpp
template<template<typename...> class TT>
class AlmostAnyTmpl {};

AlmostAnyTmpl<std::vector> withVector; // OK: 两个类型参数
AlmostAnyTmpl<std::map> withMap;       // OK: 四个类型参数
AlmostAnyTmpl<std::array> withArray;   // ERROR: 类型参数包不匹配非类型参数
```

---

#### 等价模板实参

> [!info] 等价性判断
> 当参数的值一一相同时，两组模板实参是等价的。对于类型参数，类型别名不重要，比较的是最终类型；对于整型非类型实参，比较的是值，表达方式不重要：

```cpp
template<typename T, int I>
class Mix;

using Int = int;

Mix<int, 3*3>* p1;
Mix<Int, 4+5>* p2; // p2 与 p1 具有相同类型
```

**表达式等价的复杂性：**

```cpp
template<int N> struct I {};

template<int M, int N> void f(I<M+N>); // #1
template<int N, int M> void f(I<N+M>); // #2 — 等价于 #1（重命名参数）

template<int M, int N> void f(I<N+M>); // #3 ERROR — 操作数顺序不同，不等价
```

> [!warning] 模板生成的函数与普通函数
> 从函数模板生成的函数**永远不等同于**普通函数，即使具有相同的类型和名称。这对类成员有两个重要影响：
> 1. 从成员函数模板生成的函数**不重写**虚函数
> 2. 从构造函数模板生成的构造函数**不是**复制或移动构造函数
>
> 类似地，从赋值模板生成的赋值操作不是复制/移动赋值操作符。

---

### 可变参数模板

> [!info] 可变参数模板
> 可变参数模板是包含至少一个**模板参数包**的模板。当模板的行为可泛化为任意数量的参数时，可变参数模板是有用的。

当为可变参数模板确定模板参数时，每个模板参数包将匹配一个由零个或多个模板参数组成的序列：

```cpp
template<typename... Types>
class Tuple {
public:
    static constexpr std::size_t length = sizeof...(Types);
};

int a1[Tuple<int>::length];              // 一个整数的数组
int a3[Tuple<short, int, long>::length]; // 三个整数的数组
```

---

#### 包扩展

> [!info] 包扩展
> 包扩展是将参数包展开为独立参数的构造。`sizeof...` 是一种包扩展，用于计算参数数量。其他形式的包扩展可以展开为逗号分隔列表中的多个元素，由省略号（`...`）标识。

**基本示例：**

```cpp
template<typename... Types>
class MyTuple : public Tuple<Types...> {
    // MyTuple 额外的操作
};

MyTuple<int, float> t2; // 继承自 Tuple<int, float>
```

> [!tip] 语法理解
> 可以将包扩展视为语法扩展，其中模板参数包替换为相应数量的（非包）模板参数。例如 `MyTuple<int, float>` 展开为：

```cpp
template<typename T1, typename T2>
class MyTuple : public Tuple<T1, T2> { /*...*/ };
```

> [!warning] 无法直接访问单个元素
> 不能直接通过名称访问参数包中的单个元素（如 T1、T2），若需要这些类型，唯一能做的就是将它们递归地传递给另一个类或函数。

**带模式的包扩展：**

```cpp
template<typename... Types>
class PtrTuple : public Tuple<Types*...> { /*...*/ };

PtrTuple<int, float> t3; // 继承自 Tuple<int*, float*>
```

模式 `Types*...` 对每个参数重复，产生指针类型的序列。

---

#### 包扩展的使用位置

> [!info] 包扩展上下文
> 包扩展可以在语法中提供逗号分隔列表的地方使用：
> - 基类列表
> - 构造函数的基类初始化列表
> - 调用参数列表
> - 初始化列表（如带括号的初始化列表）
> - 类、函数或别名模板的模板参数列表
> - 指定声明的对齐时
> - Lambda 的捕获列表
> - 函数类型的参数列表
> - 使用声明（C++17）

**Mixin 示例（基类列表 + 初始化列表 + 调用参数）：**

```cpp
template<typename... Mixins>
class Point : public Mixins... {           // 基类列表中的包扩展
    double x, y, z;
public:
    Point() : Mixins()... { }              // 基类初始化列表中的包扩展

    template<typename Visitor>
    void visitMixins(Visitor visitor) {
        visitor(static_cast<Mixins&>(*this)...); // 调用参数中的包扩展
    }
};

struct Color { char red, green, blue; };
struct Label { std::string name; };
Point<Color, Label> p; // 同时继承 Color 和 Label
```

**模板参数列表中的包扩展：**

```cpp
template<typename... Ts>
struct Values {
    template<Ts... Vs>
    struct Holder {};
};

int i;
Values<char, int, int*>::Holder<'a', 17, &i> valueHolder;
```

---

#### 函数参数包

> [!info] 函数参数包
> 函数参数包匹配零个或多个函数调用参数。与模板参数包一样，使用省略号（`...`）声明，使用时必须通过包扩展来展开。

```cpp
template<typename... Mixins>
class Point : public Mixins... {
    double x, y, z;
public:
    Point(Mixins... mixins)                // 函数参数包
        : Mixins(mixins)... { }            // 用提供的值初始化每个基类
};

Point<Color, Label> p({0x7F, 0, 0x7F}, {"center"});
```

**函数模板中的函数参数包：**

```cpp
template<typename... Types>
void print(Types... values);

int main() {
    std::string welcome("Welcome to ");
    print(welcome, "C++ ", 2011, '\n');
    // 调用 print<std::string, char const*, int, char>
}
```

> [!warning] 语法歧义
> 参数列表末尾的未命名函数参数包和 C 风格的 "vararg" 参数之间存在歧义：

```cpp
template<typename T> void c_style(int, T...);   // T... = T, ...（C 风格可变参数）
template<typename... T> void pack(int, T...);   // T... 是函数参数包
```

消除歧义：在省略号前加逗号（确保是 C 风格），或在 `...` 后加标识符（使其成为命名参数包）。

---

#### 多重和嵌套包扩展

> [!info] 多重包扩展
> 包扩展的模式可以包含多个不同的参数包。实例化时，**所有参数包必须具有相同的长度**。通过将每个参数包的第 i 个参数代入模式，形成结果序列。

```cpp
template<typename F, typename... Types>
void forwardCopy(F f, Types const&... values) {
    f(Types(values)...); // 复制每个值并转发
}
```

语法展开（三个参数）：

```cpp
template<typename F, typename T1, typename T2, typename T3>
void forwardCopy(F f, T1 const& v1, T2 const& v2, T3 const& v3) {
    f(T1(v1), T2(v2), T3(v3));
}
```

**嵌套包扩展：**

```cpp
template<typename... OuterTypes>
class Nested {
    template<typename... InnerTypes>
    void f(InnerTypes const&... innerValues) {
        g(OuterTypes(InnerTypes(innerValues)...)...);
    }
};
```

语法展开（OuterTypes 两个参数，InnerTypes 三个参数）：

```cpp
template<typename O1, typename O2>
class Nested {
    template<typename I1, typename I2, typename I3>
    void f(I1 const& iv1, I2 const& iv2, I3 const& iv3) {
        g(O1(I1(iv1), I2(iv2), I3(iv3)),
          O2(I1(iv1), I2(iv2), I3(iv3)));
    }
};
```

---

#### 扩展空参数包

> [!warning] 零长度参数包
> 当参数包长度为零时，语法解析通常会失败。但包扩展是**语义结构**，替换参数包不影响解析。当包展开为空列表时，行为就好像列表不存在。

```cpp
template<typename T, typename... Types>
void g(Types... values) {
    T v(values...); // 当 values 为空时，v 进行值初始化
}
```

> [!info] 语义解释
> 即使 `T v()` 在语法上看起来像函数声明，由于是在语义上对包扩展进行替换，所以不影响解析。可以用零参数初始化 `v`，即值初始化。

---

#### 折叠表达式

> [!info] 折叠表达式（C++17）
> 折叠表达式适用于除 `.`、`->` 和 `[]` 之外的所有二元操作符，是对一系列值进行操作的简洁方式。

**四种折叠形式：**

| 形式 | 名称 | 语法 |
|------|------|------|
| 二元右折叠 | Binary right fold | `(pack op ... op value)` |
| 二元左折叠 | Binary left fold | `(value op ... op pack)` |
| 一元右折叠 | Unary right fold | `(pack op ...)` |
| 一元左折叠 | Unary left fold | `(... op pack)` |

**示例：用折叠表达式替代递归**

```cpp
// 递归方式
bool and_all() { return true; }
template<typename T>
bool and_all(T cond) { return cond; }
template<typename T, typename... Ts>
bool and_all(T cond, Ts... conds) {
    return cond && and_all(conds...);
}

// C++17 折叠表达式
template<typename... T>
bool g() {
    return (trait<T>() && ... && true); // 二元左折叠
}
```

> [!danger] 空参数包的特殊情况
> 一元折叠的空展开通常是错误的，但有三个例外：
> - `&&` 的空展开产生 `true`
> - `||` 的空展开产生 `false`
> - `,` 的空展开产生空表达式

**潜在问题：重载操作符与空展开**

```cpp
struct BooleanSymbol { /*...*/ };
BooleanSymbol operator||(BooleanSymbol, BooleanSymbol);

template<typename... BTs>
void symbolic(BTs... ps) {
    BooleanSymbol result = (ps || ...); // 空展开产生 bool，而非 BooleanSymbol！
}
```

> [!tip] 最佳实践
> 通常建议使用**二元折叠表达式**（显式指定空展开的值），而不是一元折叠表达式，以避免意外行为。

### 友元
```cpp
template<typename T>
class Node;

template<typename T>
class Tree{ 
  friend class Node; // Error if node is a template
  friend class Node<T>; // OK
  template<typename U>
  friend class Node<U>; // OK 
  template<typename>
  friend class Node; // OK
}
```
函数模板的实例可以成为友元函数，方法是确保友元函数的名称后面加上尖括号。尖括号可以
包含模板参数，若参数可以推导，则尖括号可以为空

```cpp
template<typename T1, typename T2>
void combine(T1, T2);

class Mixer {
  friend void combine<>(int&, int&); // OK
  friend void combine<int,int>(int,int); //OK
  friend void combine<char>(char ,int); // OK
  friend void combine<char>(char&, int); // ERROR
  friend void combine<>(long, long) {...} // ERROR: definition not allowed here 
}
```
不能定义模板实例（最多只能定义特化），因此命名实例的友元声明不能成为定义


模板参数可以参与识别友元函数:
```cpp
template<typename T>
class Node {
  Node<T>* allocate();
  ...
};

template<typename T>
class List {
  friend Node<T>* Node<T>::allocate(); // OK: 友元函数模板实例
};
```

### 模板参数推导

#### 推导过程

> [!info] 核心机制
> 编译器将函数调用的**实参类型（A）**与函数模板的**形参类型（P）**逐一比较。每对参数独立分析，如果不同参数对推导出的模板参数不一致，推导失败。

```cpp
template<typename T>
T max(T a, T b) { return b < a ? a : b; }

auto g = max(1, 1.0); // 第一个参数推导 T=int，第二个推导 T=double → 失败！
```

> [!warning] 推导失败不等于错误（SFINAE）
> 即使推导出的参数替换到函数声明的其余部分产生了无效构造，也只是推导失败——其他候选函数仍可能成功。

**按值传递 vs 按引用传递——衰变规则：**

```cpp
template<typename T> void f(T);   // 按值：会衰变
template<typename T> void g(T&);  // 按引用：不衰变

double arr[20];
int const seven = 7;

f(arr);   // T = double*（数组衰变为指针）
g(arr);   // T = double[20]（引用参数不衰变）
f(seven); // T = int（按值去掉 const）
g(seven); // T = int const（引用保留 const）
f(7);     // T = int
g(7);     // ERROR: 右值 7 不能绑定到 int&
```

> [!tip] 衰变规则总结
> - **按值传递**：数组→指针、函数→函数指针、去掉顶层 const/volatile
> - **按引用传递**：保留数组/函数类型、保留 const/volatile

**字符串字面值陷阱：**

```cpp
template<typename T> T const& max(T const& a, T const& b);
max("Apple", "Pie"); // "Apple" 是 char[6]，"Pie" 是 char[4]
                     // T 无法同时满足两个不同大小的数组 → 失败！
// 修复：max<std::string>("Apple", "Pie")
```

---

#### 推导上下文与不可推导上下文

> [!info] 推导上下文
> 复杂参数化类型可以匹配给定的实参类型，匹配过程从顶层开始递归遍历。大部分类型构造都可以参与推导：

```cpp
template<typename T> void f1(T*);
template<typename E, int N> void f2(E(&)[N]);
template<typename T1, typename T2, typename T3>
void f3(T1 (T2::*)(T3*));

class S { public: void f(double*); };

void g(int*** ppp) {
  bool b[42];
  f1(ppp);        // T = int**
  f2(b);          // E = bool, N = 42
  f3(&S::f);      // T1 = void, T2 = S, T3 = double
}
```

**不可推导上下文（Non-deduced Contexts）：**

> [!warning] 以下类型无法用于推导模板参数
> - **限定类型名**：`Q<T>::X` 永远不能推导 `T`
> - **含模板参数的非类型表达式**：`S<I+1>` 不能推导 `I`，`sizeof(T)` 不能推导 `T`

```cpp
// 不可推导上下文不一定是错误——同一参数若能从别处推导即可
template<int N> class X { public: using I = int; void f(int){} };

template<int N>
void fppm(void (X<N>::*p)(typename X<N>::I));  // X<N>::I 不可推导

int main() {
  fppm(&X<33>::f); // OK: N 从成员指针中的 X<N> 推导，X<N>::I 随之确定
}
```

---

#### 右值引用与完美转发

**引用折叠规则：**

| | T& | T&& |
|---|---|---|
| **U&** | U& | U& |
| **U&&** | U& | U&& |

> [!tip] 简记
> 只要有一个是左值引用，结果就是左值引用；否则才是右值引用。外层的 const/volatile 会被丢弃。

**转发引用（Forwarding Reference）的特殊推导：**

```cpp
template<typename T> void f(T&& p);

int i;
int const j = 0;
f(i); // T = int&，       p 类型为 int&       （& + && → &）
f(j); // T = int const&， p 类型为 int const&
f(2); // T = int，         p 类型为 int&&
```

> [!warning] 转发引用的陷阱
> 当 `T` 被推导为引用类型时，局部变量 `T x;` 也是引用类型，需要初始化：

```cpp
template<typename T> void f(T&&) {
  T x;                          // 当 T 是引用时，ERROR！
  std::remove_reference_t<T> x; // OK: 保证不是引用
}
```

> [!info] 转发引用仅适用于
> - `T&&` 形式（`T` 是**该函数模板**的模板参数）
> - **不适用于**：类模板构造函数中的 `T&&`、非模板参数的 `X<T>&&`

**完美转发：**

```cpp
template<typename T>
void forwardToG(T&& x) {
  g(std::forward<T>(x));  // 保留原始值类别
}
```

- 传入左值：`T` 是左值引用，`T&&` 折叠为左值引用
- 传入右值：`T` 是普通类型，`T&&` 是右值引用
- 具名的右值引用是左值（防止意外移动），所以需要 `std::forward` 或 `static_cast<T&&>`

> [!danger] 完美转发的不完美之处
> - 不能区分位域左值
> - 不能捕获常量值（如空指针常量 0）

```cpp
void g(int*);
void g(...);

template<typename T> void fwd(T&& x) { g(std::forward<T>(x)); }

g(0);           // 调用 g(int*)——0 是空指针常量
fwd(0);         // 调用 g(...)——常量值丢失了！
g(nullptr);     // 调用 g(int*)
fwd(nullptr);   // 调用 g(int*)——nullptr 不是常量表达式，没问题
```

---

#### 右值引用推导的意外行为

```cpp
void int_rvalues(int&&);              // 只接受 int 右值
template<typename T> void anything(T&&); // 惊人：同时接受左值和右值！
```

> [!warning] 转发引用的限制
> `T&&` 的特殊推导规则**只在以下情况**生效：
> 1. 参数形式为 `T&&`（`T` 是模板参数）
> 2. 是函数模板的参数
> 3. `T` 由该函数模板声明

```cpp
template<typename T> class X {
  X(X&&);                    // X 不是模板参数，不适用
  X(T&&);                    // 不是函数模板（是构造函数），不适用
  template<typename U> X(X<U>&&);  // X<U> 不是模板参数，不适用
};
```

---

#### 初始化列表的推导

```cpp
template<typename T> void f(T p);
f({1, 2, 3});  // ERROR: 花括号列表没有具体类型，无法推导 T
```

> [!tip] 例外：参数是 `std::initializer_list<T>` 时可以推导

```cpp
template<typename T> void f(std::initializer_list<T>);
f({2, 3, 5, 7, 9});           // OK: T = int
f({'a', 'e', 'i', 'o', 'u', 42}); // ERROR: T = char 和 int 冲突
```

---

#### 参数包的推导

> [!info] 非包参数必须一致，包参数收集每个实参的值

```cpp
template<typename T, typename U> class pair {};
template<typename T, typename... Rest>
void h1(pair<T, Rest> const&...);

void foo(pair<int, float> pif, pair<int, double> pid, pair<double, double> pdd) {
  h1(pif, pid); // OK: T = int, Rest = {float, double}
  h1(pif, pdd); // ERROR: T 推导为 int 和 double，冲突！
}
```

**包展开的不可推导上下文：**

```cpp
template<typename... Types> class Tuple {};
template<typename... Types> bool f1(Tuple<Types...>, Tuple<Types...>);
template<typename... Ts, typename... Ys> bool f2(Tuple<Ts...>, Tuple<Ys...>);

f1(sv, sv); // OK: Types = {short, int, long}
f1(sv, uv); // ERROR: Types 从两个参数推导不一致
f2(sv, uv); // OK: Ts 和 Ys 分别独立推导
```

---

#### 推导的限制

**推导中不允许的隐式转换：**

```cpp
std::string s;
::max(s, "hello"); // ERROR: T = string 和 T = char[6] 冲突
// 修复：::max<std::string>(s, "hello")  显式指定后允许隐式转换
```

> [!danger] 推导期间不做隐式转换
> 推导要求精确匹配（允许的少量例外：cv 限定、指针限定转换、派生类到基类）。两个参数推导出不同类型时直接失败，不会寻找共同基类。

**默认函数参数不参与推导：**

```cpp
template<typename T> void f(T x = 42) {}
f();      // ERROR: 不能从默认参数推导 T
f<int>(); // OK: 显式指定
```

---

#### 显式指定模板参数

```cpp
template<typename T> T default_value() { return T{}; }
default_value<int>(); // T = int

// 部分显式指定（从左到右）
template<typename Out, typename In>
Out convert(In p) {}
auto x = convert<double>(42); // Out = double（显式），In = int（推导）
```

**空尖括号强制选择模板版本：**

```cpp
int f(int);            // 普通函数
template<typename T> T f(T); // 函数模板

auto x = f(42);   // 调用普通函数（更匹配）
auto y = f<>(42); // 调用函数模板
```

> [!warning] 不可推导参数必须放在参数列表开头
> 不可推导的参数放在前面，可以与可推导的参数同时使用——前面的显式指定，后面的推导得出。如果颠倒顺序，则所有参数都需要显式指定。

---

#### 别名模板与推导

> [!info] 别名模板对推导是透明的
> 别名模板在推导前展开为其定义，因为别名模板不能被特化，展开总是无歧义的：

```cpp
template<typename T> using DequeStack = Stack<T, std::deque<T>>;

template<typename T, typename Cont> void f1(Stack<T, Cont>);
template<typename T> void f2(DequeStack<T>);

void test(DequeStack<int> s) {
  f1(s); // T = int, Cont = deque<int>
  f2(s); // T = int（DequeStack 展开为 Stack<T, deque<T>>）
}
```

---

#### 类模板参数推导（CTAD，C++17）

```cpp
template<typename T1, typename T2, typename T3 = T2>
class C { public: C(T1 x, T2 y, T3 z); };

C c1(22, 44.3, "hi"); // T1=int, T2=double, T3=char const*
C c2(22, 44.3);        // T1=int, T2=T3=double
```

> [!warning] 不能部分指定再推导
> `C<string> c("hi", "my", 42);` 是错误的——只指定了 T1，T2 无法推导。

**推导指引（Deduction Guides）：**

```cpp
template<typename T> class S {
  T a;
public:
  S(T b) : a(b) {}
};

template<typename T> S(T) -> S<T>;  // 显式推导指引

S x{12};  // OK: S<int>
```

> [!tip] 隐式推导指引
> 每个构造函数自动生成一个隐式推导指引，大多数情况下不需要显式指引。但当构造函数参数类型涉及非推导上下文时，需要显式指引。

```cpp
// 隐式指引可能导致歧义
std::vector v{1, 2, 3};  // vector<int>
std::vector w{v};         // vector<int>（拷贝构造），不是 vector<vector<int>>
std::vector w2{v, v};     // vector<vector<int>>
```

---

#### auto 与 decltype

**`auto` 使用与模板参数相同的推导机制（按值传递，会衰变）：**

```cpp
auto x = 42;          // int
auto& r = x;          // int&
auto&& rr = 42;       // int&&（右值）
auto&& lr = x;        // int&（左值，折叠）
auto const N = 400u;  // unsigned const
```

**`decltype` 保留值类别（更精确）：**

```cpp
int i = 42;
int const& ref = i;
decltype(ref) r1 = i;     // int const&（声明类型）
decltype((ref)) r2 = i;   // int const&（表达式是左值）
decltype(42) n = 0;       // int（纯右值）
decltype((42)) n2 = 0;    // int（纯右值，括号不改变纯右值）
```

> [!danger] 括号很重要
> `decltype(s)` 返回声明类型，`decltype((s))` 返回表达式的值类别类型。对引用变量，`decltype(s)` 是引用，`decltype((s))` 也是引用（因为具名变量是左值）。但 `decltype(auto)` 中括号可能导致悬挂引用：

```cpp
decltype(auto) f() {
  int r = g();
  return (r);  // ERROR: 返回 int& 指向局部变量！
  return r;    // OK: 返回 int
}
```

**`decltype(auto)`（C++14）——结合两者优点：**

```cpp
int i = 42;
int const& ref = i;
auto x = ref;              // int（按值推导，去掉引用和 const）
decltype(auto) y = ref;    // int const&（保留一切）
```

---

#### SFINAE 与立即上下文

> [!info] SFINAE：替换失败不为过
> 模板参数替换失败不是错误，只是该候选被忽略。SFINAE 仅在**立即上下文**中生效。

```cpp
// 经典例子：begin() 同时支持容器和数组
template<typename T, unsigned N>
T* begin(T (&array)[N]) { return array; }

template<typename Container>
typename Container::iterator begin(Container& c) { return c.begin(); }

std::vector<int> v;
int a[10];
::begin(v); // 数组版失败（SFINAE），容器版成功
::begin(a); // 容器版失败（SFINAE），数组版成功
```

> [!danger] 非立即上下文的错误不是 SFINAE
> 以下情况的错误是**真正的错误**，不会被 SFINAE 捕获：
> - 类模板的定义体和基类列表
> - 函数模板的函数体
> - 默认参数、默认成员初始化器
> - 异常规格

```cpp
template<typename T> auto f(T p) { return p->m; }  // #1
int f(...);                                          // #2
template<typename T> auto g(T p) -> decltype(f(p));  // #3

g(42); // ERROR: #1 被实例化，p->m 对 int 无效
       // 但这是在函数体中（非立即上下文），SFINAE 不适用！
```

> [!tip] 最佳实践
> 需要与 SFINAE 交互时，使用尾返回类型 `-> decltype(expr)` 而不是 `auto` 推导返回类型。后者需要实例化函数体，不属于立即上下文。

---

#### 结构化绑定（C++17）

```cpp
// 简单类类型
struct MaybeInt { bool valid; int value; };
auto [b, N] = MaybeInt{true, 42};  // b = true, N = 42

// 数组
double pt[3] = {1.0, 2.0, 3.0};
auto& [x, y, z] = pt;  // x, y, z 是数组元素的引用

// 元组类
std::tuple<bool, int> bi{true, 42};
auto [b, i] = bi;
```

---

## 模板高级应用

### 显式特化与重载

#### 函数模板的显式特化

```cpp
// 主模板
template<typename T>
std::string f(T) { return "generic"; }

// 显式特化（针对 const char*）
template<>
std::string f<const char*>(const char* p) { return "cstring"; }
```

> [!warning] 特化不参与重载
> 函数模板的**显式特化不会被视为独立的函数**——它只是主模板的一个特化版本。重载解析发生在主模板之间，选中主模板后才查看是否有特化匹配。

```cpp
template<typename T> void f(T);          // #1
template<typename T> void f(T*);         // #2
template<> void f<int>(int);             // 特化 #1
template<> void f<int>(int*);            // 特化 #2

f(new int{42}); // 重载解析选 #2（T* 匹配更好），实例化 #2 的特化
```

> [!danger] 陷阱：显式特化 vs 重载
> 以下写法意图是特化 `f(T*)`，但实际上特化了 `f(T)`：

```cpp
template<typename T> void f(T);   // 主模板 #1
template<typename T> void f(T*);  // 主模板 #2

template<> void f<int*>(int*);    // 这是特化 #1（T=int*），不是 #2！
```

#### 类模板偏特化

类模板没有重载，但有偏特化。偏特化可以针对部分参数或参数的某种模式：

```cpp
template<typename T1, typename T2> class Pair;

// 偏特化：两个类型相同
template<typename T>
class Pair<T, T> { /* T x, y; */ };

// 偏特化：第二个参数是 int
template<typename T1>
class Pair<T1, int> { /* T1 first; int second; */ };

// 偏特化：两个都是指针
template<typename T1, typename T2>
class Pair<T1*, T2*> { /* T1* first; T2* second; */ };
```

> [!tip] 偏特化选择规则
> 编译器选择**最特化**的匹配版本。如果有多个候选，选约束最强的那个；如果无法区分，则报歧义错误。

---

### 模板实例化

#### 按需实例化（On-demand Instantiation）

> [!info] 核心机制
> 编译器遇到模板特化时，自动通过替换模板参数来创建该特化。这称为**按需实例化**（也叫隐式或自动实例化）。编译器必须在使用点能访问模板的**完整定义**（不仅是声明）。

```cpp
template<typename T> class C;       // #1 仅声明
C<int>* p = 0;                      // #2 OK: 只需要指针大小
template<typename T>
class C {
public:
  void f();                         // #3 成员声明
};                                  // #4 类模板定义完成

void g(C<int>& c) {                 // #5 仅使用声明
  c.f();                            // #6 访问成员 → 需要完整定义和 f() 的定义
}

template<typename T>
void C<T>::f() { }                  // 因为 #6 而需要此定义
```

> [!warning] `new` 也会触发实例化
> `C<void>* p = new C<void>;` 需要实例化以确定 `C<void>` 的大小，即使类是空的。编译器不会分析模板定义来避免实例化。

---

#### 惰性实例化（Lazy Instantiation）

> [!info] 编译器只实例化到需要的程度
> - **部分实例化**：仅替换声明，不替换函数体
> - **完全实例化**：替换完整的定义（当我们说"实例化"时默认指这个）

```cpp
template<typename T> T f(T p) { return 2*p; }
decltype(f(2)) x = 2;  // 仅实例化 f 的声明，不实例化函数体

template<typename T> class Q {
  using Type = typename T::Type;
};
Q<int>* p = 0;  // OK: Q<int> 的函数体不被替换（否则 int::Type 会报错）
```

> [!tip] 别名模板没有部分/完全之分
> 别名模板不存在这种区分。

**类模板实例化的组成：**

当类模板被隐式完全实例化时：
- 成员声明被实例化
- 成员定义**不被实例化**（惰性！）

**例外：**
- **匿名联合体**：定义会被实例化（被视为外围类的成员）
- **虚成员函数**：定义可能被实例化（虚函数调用机制需要可链接的实体）
- **默认函数参数**：仅在实际使用默认值时才实例化

```cpp
template<typename T> class Safe {};
template<int N> class Danger { int arr[N]; };

template<typename T, int N> class Tricky {
public:
  void inclass() { Danger<N> noBoomYet; }  // OK 直到 inclass() 被使用且 N<=0
  struct Nested { Danger<N> pfew; };        // OK 直到 Nested 被使用且 N<=0
  union { Danger<N> anonymous; int align; }; // 匿名联合：实例化时立即检查
  void unsafe(T (*p)[N]);                    // 实例化时检查（N<=0 则数组大小非法）
  void error() { Danger<-1> boom; }          // 总是错误，但编译器可以不报（如果从未使用）
};

Tricky<int, -1> inst; // ERROR: 匿名联合实例化 → Danger<-1> 无效
                       // unsafe 的声明中 T(*p)[-1] 也无效
                       // 但 inclass() 和 Nested 不会被实例化
```

---

#### 两阶段名称查找

> [!note] 两个阶段
> C++ 模板的名称查找分为两个阶段：
> 1. **解析模板时**：查找非依赖名称；对未限定的依赖名称做初步查找（仅用于判断是否是模板）
> 2. **实例化时**：对未限定的依赖名称通过 ADL 重新查找

```cpp
namespace N {
  template<typename> void g() {}
  enum E { e };
}
template<typename T> void h(T p) {
  f<int>(p);  // OK: 第一阶段普通查找找到模板 f，< 被解析为模板参数
  g<int>(p);  // ERROR: 第一阶段普通查找找不到 g，< 被解析为小于号！
}
int main() { h(N::e); }
```

> [!danger] ADL 找不到非关联命名空间的函数
> 依赖名称在 POI 处只通过 ADL 查找。如果参数类型没有关联命名空间（如 `int`），则普通查找可见但 ADL 不可见的函数会找不到。

```cpp
void g1(int) { }
template<typename T>
void f1(T x) { g1(x); }  // g1 依赖于 x → 仅 ADL 查找

f1(7); // ERROR: int 没有关联命名空间，ADL 找不到 g1
```

---

#### 实例化点（POI）

> [!info] 实例化点的位置
> - **函数模板**的 POI：在包含该特化引用的最近命名空间作用域声明**之后**
> - **类模板**的 POI：在包含该特化引用的最近命名空间作用域声明**之前**（因为 `sizeof` 等需要大小已知）

```cpp
template<typename T> class S { public: T m; };
// #1
unsigned long h() {
  return (unsigned long)sizeof(S<int>);  // 需要 S<int> 的大小
  // #2
}
// #3
// S<int> 的 POI 在 #1（h() 之前），而不是 #3
```

> [!tip] 实际行为
> 大多数编译器将函数模板的实例化延迟到翻译单元末尾。例外：需要确定 `auto` 返回类型时、`constexpr` 函数需要编译时求值时。

---

#### 显式实例化

**显式实例化定义：**

```cpp
template<typename T> void f(T) { }

// 四种有效写法
template void f<int>(int);
template void f<>(float);
template void f(long);      // 模板参数可推导
template void f(char);      // 模板参数可推导

// 类模板：实例化所有成员
template class Stack<int>;

// 实例化单个成员
template void Stack<int>::push(int const&);
```

**显式实例化声明（`extern template`）：**

```cpp
// header.hpp
template<typename T> void f() { /* 实现 */ }
extern template void f<int>();    // 声明但不定义
extern template void f<float>();

// source.cpp
template void f<int>();           // 定义
template void f<float>();         // 定义
```

> [!warning] extern template 的例外
> 以下情况仍然会触发自动实例化：
> - `inline` 函数（内联展开需要定义）
> - `auto`/`decltype(auto)` 类型变量（需要确定类型）
> - `constexpr` 变量（需要确定值）
> - 引用类型变量（需要确定引用的实体）
> - 类模板和别名模板本身（需要检查结果类型）

> [!danger] 声明必须配对定义
> 每个显式实例化声明必须有对应的显式实例化定义，否则链接器报错。

---

#### 手动实例化与 .tpp 模式

> [!tip] 编译优化策略
> 将模板定义放在 `.tpp` 文件中，手动控制实例化：

```cpp
// f.hpp: 仅声明
template<typename T> void f();

// f.tpp: 定义
#include "f.hpp"
template<typename T> void f() { /* 实现 */ }

// f.cpp: 显式实例化
#include "f.tpp"
template void f<int>();
template void f<double>();

// 使用者只包含 f.hpp，不包含 f.tpp → 避免重复实例化
```

> [!info] 灵活性
> 如果手动管理太麻烦，可以改为包含 `f.tpp` 来启用自动实例化。`extern template` 是更轻量的替代方案——不需要隐藏模板定义，但编译时收益可能较小（模板定义仍需解析）。

---

#### 编译时 if（C++17）

```cpp
template<typename T>
bool f(T p) {
  if constexpr (sizeof(T) <= sizeof(long long))
    return p > 0;        // 仅当 T 较小时实例化
  else
    return p.compare(0) > 0;  // 仅当 T 较大时实例化
}

f(42);  // else 分支被丢弃，不实例化 → p.compare(0) 不会报错
```

> [!info] 关键区别
> 普通 `if` 两个分支**都会被实例化**（只是运行时选一个）。`if constexpr` 的丢弃分支**根本不会被实例化**。

**简化可变参数模板递归：**

```cpp
template<typename Head, typename... Remainder>
void print(Head&& h, Remainder&&... r) {
  std::cout << h;
  if constexpr (sizeof...(r) > 0) {
    std::cout << ", ";
    print(std::forward<Remainder>(r)...);  // 参数包非空时才递归
  }
  // 不需要单独的 print() {} 终止重载
}
```

**非模板上下文中也有效：**

```cpp
void h();
void g() {
  if constexpr (sizeof(int) == 1) { h(); }  // 条件为 false
}
// h() 不需要定义——被丢弃的分支不要求定义存在
```

---

#### 实例化方案（编译器实现）

| 方案 | 原理 | 代表 | 现状 |
|------|------|------|------|
| **贪婪实例化** | 每个翻译单元都实例化，链接器去重 | Borland | **主流** |
| **查询实例化** | 共享数据库跟踪已实例化的特化 | Sun | 已废弃 |
| **迭代实例化** | 预链接器迭代直到所有实例化完成 | Cfront | 罕见 |

> [!tip] 贪婪实例化
> 最常用的方案。每个翻译单元独立实例化，编译器标记模板实例化，链接器发现重复时保留一个丢弃其余。
> - 优点：保持传统构建模型、支持内联
> - 缺点：编译时间浪费（N 个实例化只保留 1 个）、目标文件膨胀

---

#### 标准库中的显式实例化

```cpp
namespace std {
  template<typename charT,
           typename traits = char_traits<charT>,
           typename Allocator = allocator<charT>>
  class basic_string { /* ... */ };

  extern template class basic_string<char>;
  extern template class basic_string<wchar_t>;
}

// 类似的还有 basic_iostream、basic_istream、basic_ostream 等
```

> [!info] 标准库的做法
> 标准库实现对常用类型（`char`、`wchar_t`）使用 `extern template`，在库的源文件中提供显式实例化定义。所有用户共享同一份实例化，避免每个翻译单元重复生成。

---

### 模板中的名称

#### 依赖名称与 `typename`

> [!warning] 何时需要 `typename`
> 当一个**限定名称**（含 `::`）依赖于模板参数且引用的是类型时，必须加 `typename`：

```cpp
template<typename T>
void f(T x) {
  typename T::iterator it;       // OK: 告诉编译器这是类型
  T::iterator * p;               // 错误！编译器认为是乘法
  typename std::vector<T>::value_type val;
}
```

**不需要 `typename` 的场景：**
1. 基类列表中：`class C : public T::Base { };`
2. 成员初始化列表中：`C() : T::Base(0) {}`
3. 详细类型说明符中：`class T::Nested x;`

#### 依赖模板名称与 `template`

```cpp
template<typename T>
void f(T& p) {
  p.template g<int>();   // 必须加 template
  T::template Iterator<int>* it;  // 必须加 template
}

// 具体例子
template<unsigned N>
void printBitset(std::bitset<N> const& bs) {
  std::cout << bs.template to_string<char>();  // 必须用 .template
}
```

> [!info] 原因
> 编译器在解析模板时，遇到依赖名称后面跟 `<`，无法判断它是模板参数列表的开始还是小于运算符。`template` 关键字明确告诉编译器后面是模板参数。

#### 两阶段查找

> [!note] 两阶段名称查找（详见"模板实例化"章节）
> C++ 模板的名称查找分为两个阶段：
> 1. **模板定义阶段**：查找非依赖名称，对未限定的依赖名称做初步查找
> 2. **模板实例化阶段**：对依赖名称通过 ADL 重新查找

```cpp
void f(int) { std::cout << "int"; }

template<typename T>
void g(T x) {
  f(1);       // 非依赖名称：在定义阶段查找 → 找到全局 f(int)
  f(x);       // 依赖名称：在实例化阶段通过 ADL 查找
}

struct S {};
void f(S) { std::cout << "S"; }

g(S{});       // f(x) 通过 ADL 找到 f(S)
g(42);        // f(x) 通过 ADL 找不到 f(int)，但定义阶段已有
```

> [!danger] 关键陷阱
> - 第一阶段的初步查找决定 `<` 是模板参数还是小于号——如果找不到模板，`<` 被解析为运算符，导致语法错误（即使 ADL 在第二阶段能找到）
> - ADL 只查找参数类型的关联命名空间。`int` 没有关联命名空间，所以 `g1(int)` 对 `int` 参数不可见（见"模板实例化"章节的 ADL 陷阱示例）

> [!danger] 依赖名称的 ADL 陷阱
> 如果依赖名称调用的函数只有 ADL 可见（不是非限定查找可见的），则只通过 ADL 查找。这可能导致找不到本应可见的函数。

---

### 类型特征（Type Traits）

#### 类型函数

> [!info] 类型函数的概念
> 类型函数：接受类型作为参数，返回类型或常量值。`sizeof` 是内置的类型函数。C++ 标准库的 `<type_traits>` 提供了丰富的类型函数。

**元素类型提取：**

```cpp
template<typename T>
struct ElementT;  // 主模板

template<typename T>
struct ElementT<std::vector<T>> {
  using Type = T;
};

template<typename T, std::size_t N>
struct ElementT<std::array<T, N>> {
  using Type = T;
};

template<typename T>
struct ElementT<T[]> {
  using Type = T;
};

// 使用
using Elem = typename ElementT<std::vector<int>>::Type;  // int
```

#### 常用类型变换

```cpp
// 去除引用
std::remove_reference_t<int const&>        // int const
std::remove_reference_t<int&&>             // int

// 去除 const
std::remove_const_t<int const>             // int
std::remove_const_t<int const&>            // int const&（注意！引用不是const）

// 去除所有 cv 和引用
std::decay_t<int const&>                   // int
std::decay_t<char const[6]>               // char const*（数组衰变）
std::decay_t<void(int)>                   // void(*)(int)（函数衰变）
```

> [!warning] remove_const 与引用
> `std::remove_const_t<int const&>` 结果是 `int const&`，不是 `int&`。因为引用本身不是 const 的，它引用的对象是 const 的。要同时去除引用和 const，用 `std::decay_t` 或链式调用 `std::remove_const_t<std::remove_reference_t<T>>`。

#### 谓词特征

```cpp
// 判断两个类型是否相同
template<typename T1, typename T2>
struct IsSameT : std::false_type {};

template<typename T>
struct IsSameT<T, T> : std::true_type {};

// 变量模板简化
template<typename T1, typename T2>
constexpr bool isSame = IsSameT<T1, T2>::value;

static_assert(isSame<int, int>);
static_assert(!isSame<int, double>);
```

**标准库谓词：**

```cpp
std::is_same_v<int, int>              // true
std::is_convertible_v<int, double>    // true
std::is_base_of_v<Base, Derived>      // true
std::is_arithmetic_v<int>             // true
std::is_void_v<void>                  // true
std::is_pointer_v<int*>              // true
std::is_reference_v<int&>            // true
std::is_const_v<int const>           // true
```

#### 标签调度（Tag Dispatch）

> [!example] 标签调度模式
> 通过标签类型选择最优实现，是 STL 迭代器分类的基础：

```cpp
// 标签类型（标准库已定义继承链）
struct input_iterator_tag {};
struct forward_iterator_tag : input_iterator_tag {};
struct bidirectional_iterator_tag : forward_iterator_tag {};
struct random_access_iterator_tag : bidirectional_iterator_tag {};

// 通用版本
template<typename Iter>
void advance_impl(Iter& it, int n, input_iterator_tag) {
  while (n-- > 0) ++it;  // 逐个前进
}

// 随机访问版本（O(1)）
template<typename Iter>
void advance_impl(Iter& it, int n, random_access_iterator_tag) {
  it += n;  // 直接跳转
}

// 入口函数
template<typename Iter>
void advance(Iter& it, int n) {
  advance_impl(it, n, typename std::iterator_traits<Iter>::iterator_category{});
}
```

> [!tip] 工作原理
> `random_access_iterator_tag` 继承自 `forward_iterator_tag`。对 `vector<int>::iterator`（随机访问），重载解析选 `random_access_iterator_tag` 版本（更特化）。对 `list<int>::iterator`（双向），选 `bidirectional_iterator_tag` 版本。标签的继承关系确保了自动选择最优实现。

---

### 模板与继承

#### 奇异递归模板模式（CRTP）

```cpp
template<typename Derived>
class Counter {
  static inline int count = 0;
public:
  Counter() { ++count; }
  Counter(Counter const&) { ++count; }
  Counter(Counter&&) { ++count; }
  ~Counter() { --count; }
  static int getCount() { return count; }
};

class MyWidget : public Counter<MyWidget> {
  // 自动获得计数功能
};

class MyButton : public Counter<MyButton> {
  // 每个派生类有独立的计数器
};

MyWidget w1, w2;
MyButton b1;
std::cout << MyWidget::getCount();  // 2
std::cout << MyButton::getCount();  // 1
```

> [!info] CRTP 的核心思想
> 基类通过模板参数知道派生类的类型，可以用 `static_cast<Derived*>(this)` 安全地向下转换，调用派生类的方法——实现**编译时多态**。

**CRTP 实现运算符：**

```cpp
template<typename Derived>
class EqualityComparable {
public:
  friend bool operator!=(Derived const& a, Derived const& b) {
    return !(a == b);  // 基于 Derived 的 operator== 生成 operator!=
  }
};

class Point : public EqualityComparable<Point> {
  int x, y;
public:
  bool operator==(Point const& o) const { return x == o.x && y == o.y; }
  // 自动生成 operator!=
};
```

> [!tip] Barton-Nackman 技巧（友元工厂）
> 在类模板内定义 `friend` 函数，使该函数成为非模板函数，通过 ADL 可见。这避免了全局模板运算符过于泛化的问题。

**CRTP 实现迭代器外观（Iterator Facade）：**

```cpp
template<typename Derived, typename Value, typename Category>
class IteratorFacade {
  Derived& derived() { return static_cast<Derived&>(*this); }
public:
  // 由核心操作派生出完整迭代器接口
  Value operator*() const { return derived().dereference(); }
  Derived& operator++() { derived().increment(); return derived(); }
  bool operator==(IteratorFacade const& o) const {
    return derived().equals(o);
  }
  // ... operator--, operator+= 等
};

// 使用：只需实现 3 个核心方法
class ListIter : public IteratorFacade<ListIter, int, forward_iterator_tag> {
  friend class IteratorFacadeAccess;
  Node* node;
  int dereference() const { return node->value; }
  void increment() { node = node->next; }
  bool equals(ListIter const& o) const { return node == o.node; }
};
```

#### 空基类优化（EBCO）

```cpp
class Empty {};
class EmptyToo {};

// 通常 sizeof(MyClass) > sizeof(int)
class MyClass {
  Empty e;
  int x;
};

// 利用空基类优化
class MyClass : private Empty {
  int x;
};
// sizeof(MyClass) == sizeof(int) 在大多数编译器上
```

> [!warning] 模板中的 EBCO
> 当基类依赖于模板参数时，编译器不知道两个不同的模板实例化是否可能产生相同的基类。C++ 标准不允许对相同类型的基类进行空基类优化（即使它们在不同偏移上）。

---

### 类型擦除

> [!example] 类型擦除的核心思想
> 类型擦除 = **静态多态的接口 + 动态多态的实现**。标准库的 `std::function` 就是类型擦除的典型例子。

```cpp
// 抽象桥接基类
template<typename R, typename... Args>
class FunctorBridge {
public:
  virtual ~FunctorBridge() = default;
  virtual R invoke(Args... args) = 0;
  virtual FunctorBridge* clone() const = 0;
};

// 具体实现：存储实际的可调用对象
template<typename Functor, typename R, typename... Args>
class SpecificFunctorBridge : public FunctorBridge<R, Args...> {
  Functor functor;
public:
  explicit SpecificFunctorBridge(Functor f) : functor(std::move(f)) {}

  R invoke(Args... args) override {
    return functor(std::forward<Args>(args)...);
  }

  FunctorBridge<R, Args...>* clone() const override {
    return new SpecificFunctorBridge(functor);
  }
};

// 简化的 std::function 实现
template<typename Signature> class FunctionPtr;

template<typename R, typename... Args>
class FunctionPtr<R(Args...)> {
  FunctorBridge<R, Args...>* bridge;
public:
  // 构造函数模板：接受任意可调用对象
  template<typename F>
  FunctionPtr(F&& f)
    : bridge(new SpecificFunctorBridge<std::decay_t<F>, R, Args...>(
        std::forward<F>(f))) {}

  R operator()(Args... args) {
    return bridge->invoke(std::forward<Args>(args)...);
  }
};
```

> [!info] 类型如何被"擦除"
> `FunctionPtr<void(int)>` 可以存储 lambda、函数指针、仿函数等任意可调用对象。具体类型 `F` 在构造时被封装进 `SpecificFunctorBridge<F, ...>`，然后通过基类指针 `FunctorBridge<...>*` 存储。具体的 `F` 类型信息从此丢失——这就是"类型擦除"的含义。

**使用示例：**

```cpp
FunctionPtr<void(int)> f1 = [](int x) { std::cout << x; };
FunctionPtr<void(int)> f2 = [](int x) { std::cout << x * 2; };

f1(42);  // prints 42
f2(42);  // prints 84

f1 = f2;  // OK: 可以重新赋值
f1(42);  // prints 84
```

---

### 模板元编程

#### 值元编程（constexpr）

```cpp
// C++14: constexpr 函数可以用循环
constexpr bool isPrime(unsigned n) {
  if (n < 2) return false;
  for (unsigned d = 2; d * d <= n; ++d)
    if (n % d == 0) return false;
  return true;
}

// 编译时使用
constexpr bool b = isPrime(17);  // true
static_assert(isPrime(7));

// 编译时数组大小
int primes[isPrime(7) + isPrime(8)];  // int primes[1+0]
```

#### 类型元编程

```cpp
// 递归去除所有数组维度
template<typename T>
struct RemoveAllExtentsT { using Type = T; };

template<typename T, std::size_t N>
struct RemoveAllExtentsT<std::array<T, N>> {
  using Type = typename RemoveAllExtentsT<T>::Type;
};

// std::array<std::array<int, 3>, 4> → int
using Inner = RemoveAllExtentsT<std::array<std::array<int, 3>, 4>>::Type;
```

**类型列表操作：**

```cpp
// 类型列表
template<typename... Types> struct Typelist;

// 获取第 N 个类型
template<std::size_t N, typename List> struct TypeAt;

template<typename Head, typename... Tail>
struct TypeAt<0, Typelist<Head, Tail...>> {
  using Type = Head;
};

template<std::size_t N, typename Head, typename... Tail>
struct TypeAt<N, Typelist<Head, Tail...>> {
  static_assert(N < sizeof...(Tail) + 1, "Index out of bounds");
  using Type = typename TypeAt<N - 1, Typelist<Tail...>>::Type;
};

using MyTypes = Typelist<int, double, char>;
using Second = TypeAt<1, MyTypes>::Type;  // double
```

#### 混合元编程

> [!example] 编译时循环展开
> 用递归模板实例化实现循环展开，运行时无循环开销：

```cpp
// 编译时点积
template<std::size_t N>
struct DotProduct {
  template<typename T>
  static T result(T* a, T* b) {
    return *a * *b + DotProduct<N-1>::result(a+1, b+1);
  }
};

template<>
struct DotProduct<0> {
  template<typename T>
  static T result(T*, T*) { return T{}; }
};

double a[] = {1, 2, 3};
double b[] = {4, 5, 6};
double r = DotProduct<3>::result(a, b);  // 1*4 + 2*5 + 3*6 = 32
```

> [!tip] std::tuple 与元编程
> `std::tuple` 是混合元编程的"英雄容器"——它在编译时管理异构类型集合，运行时存储实际值。

#### 单位类型元编程

```cpp
// 编译时比率计算（std::chrono 的基础）
template<std::size_t N, std::size_t D = 1>
struct Ratio {
  static constexpr std::size_t num = N;
  static constexpr std::size_t den = D;
};

// 两个比率相加
template<typename R1, typename R2>
struct RatioAdd {
private:
  static constexpr std::size_t gcd = /* 最大公约数计算 */;
public:
  using Type = Ratio<
    (R1::num * R2::den + R2::num * R1::den) / gcd,
    (R1::den * R2::den) / gcd
  >;
};

// 带单位的量
template<typename Value, typename Unit>
class Duration {
  Value val;
public:
  constexpr Duration(Value v) : val(v) {}
  constexpr Value value() const { return val; }
};

// 加法：编译时确定结果单位
template<typename V, typename U1, typename U2>
auto operator+(Duration<V, U1> a, Duration<V, U2> b) {
  using NewUnit = typename RatioAdd<U1, U2>::Type;
  return Duration<V, NewUnit>(a.value() + b.value());
}
```

---

### 表达式模板

> [!example] 问题：临时数组开销
> 对于 `result = 1.2*x + x*y`（x, y, result 都是大数组），传统实现会创建两个临时数组：
> 1. `tmp1 = 1.2 * x`（逐元素）
> 2. `tmp2 = x * y`（逐元素）
> 3. `result = tmp1 + tmp2`（逐元素）
>
> 表达式模板将操作编码到类型中，赋值时逐元素一次性计算，零临时数组。

```cpp
// 标量包装器
template<typename T>
class A_Scalar {
  T const& s;
public:
  A_Scalar(T const& v) : s(v) {}
  T const& operator[](std::size_t) const { return s; }
  static constexpr std::size_t size() { return 0; }
};

// 加法表达式
template<typename T, typename OP1, typename OP2>
class A_Add {
  OP1 const& op1;
  OP2 const& op2;
public:
  A_Add(OP1 const& a, OP2 const& b) : op1(a), op2(b) {}
  T operator[](std::size_t i) const { return op1[i] + op2[i]; }
  std::size_t size() const { return op1.size(); }
};

// 乘法表达式
template<typename T, typename OP1, typename OP2>
class A_Mult {
  OP1 const& op1;
  OP2 const& op2;
public:
  A_Mult(OP1 const& a, OP2 const& b) : op1(a), op2(b) {}
  T operator[](std::size_t i) const { return op1[i] * op2[i]; }
  std::size_t size() const { return op1.size(); }
};

// 数组模板：Rep 可以是实际存储或表达式
template<typename T, typename Rep = std::vector<T>>
class Array {
  Rep rep;
public:
  // 运算符返回表达式类型（不计算！）
  friend Array operator+(Array const& a, Array const& b) {
    return Array(A_Add<T, Rep, Rep>(a.rep, b.rep));
  }
  friend Array operator*(T const& s, Array const& a) {
    return Array(A_Mult<T, A_Scalar<T>, Rep>(A_Scalar<T>(s), a.rep));
  }

  // 赋值触发实际计算
  Array& operator=(Array const& other) {
    for (std::size_t i = 0; i < rep.size(); ++i)
      rep[i] = other.rep[i];  // 逐元素计算表达式树
    return *this;
  }
};
```

> [!info] 工作原理
> `1.2 * x` 返回 `Array<double, A_Mult<double, A_Scalar<double>, vector<double>>>`——类型编码了操作。`+` 和 `*` 不计算任何值，只构建类型树。赋值 `=` 时，循环遍历每个元素，通过 `operator[]` 递归展开表达式树计算结果。**零临时数组，一次遍历。**

> [!warning] 与元编程的互补
> - **元编程**：适合编译时已知大小的小数组（如固定维度向量），直接展开循环
> - **表达式模板**：适合运行时大小的中大型数组，避免临时对象

---

### 概念（Concepts，C++20）

#### 定义概念

```cpp
// 基本语法
template<typename T>
concept Hashable = requires(T a) {
  { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

// 复合要求
template<typename T>
concept EqualityComparable = requires(T a, T b) {
  { a == b } -> std::convertible_to<bool>;  // 可比较，结果转为 bool
  { a != b } -> std::convertible_to<bool>;
};

// 类型要求
template<typename T>
concept HasValueType = requires {
  typename T::value_type;  // T 必须有 value_type 成员类型
};

// 嵌套要求
template<typename T>
concept Sortable = requires(T a) {
  typename T::value_type;
  requires std::totally_ordered<typename T::value_type>;  // 值类型可全序
  { a.begin() } -> std::input_or_output_iterator;
  { a.end() } -> std::input_or_output_iterator;
};
```

> [!info] requires 表达式的四种要求
> 1. **简单要求**：`swap(a, b);` — 表达式有效即可
> 2. **复合要求**：`{ expr } -> type;` — 表达式有效且返回类型可转换为指定类型
> 3. **类型要求**：`typename T::foo;` — 类型存在
> 4. **嵌套要求**：`requires Concept<Args...>;` — 满足另一个概念

#### 使用概念

```cpp
// 方式一：requires 子句
template<typename T>
requires std::sortable<T>
void mySort(T& container) { std::sort(container.begin(), container.end()); }

// 方式二：简写语法（单参数概念）
template<std::sortable T>
void mySort(T& container) { std::sort(container.begin(), container.end()); }

// 方式三：auto + concept
void mySort(std::sortable auto& container) {
  std::sort(container.begin(), container.end());
}
```

#### 约束的偏序（Subsumption）

> [!tip] 概念子化
> 当多个约束模板匹配时，更"受约束"的版本被优先选择：

```cpp
template<typename T>
concept Integral = std::is_integral_v<T>;

template<typename T>
concept SignedIntegral = Integral<T> && std::is_signed_v<T>;

// SignedIntegral 子化（subsume）Integral
void f(Integral auto x) { /* 通用版本 */ }
void f(SignedIntegral auto x) { /* 更特化的版本 */ }

f(42);     // 选 SignedIntegral 版本
f(42u);    // 选 Integral 版本（unsigned 不满足 SignedIntegral）
```

> [!info] 子化规则
> `C2` 子化 `C1` 当且仅当 `C2` 的规范化约束蕴含 `C1`。简单来说，如果 `C2 = C1 && 额外约束`，则 `C2` 子化 `C1`。这比 SFINAE + `enable_if` 的重载更清晰、更可靠。

**概念 vs enable_if vs 标签调度：**

| 技术 | 优先级 | 适用场景 |
|------|--------|---------|
| 概念 | 最佳选择 | C++20+，清晰的错误信息，自动子化偏序 |
| `enable_if` | C++11/14/17 | 复杂的 SFINAE 条件 |
| 标签调度 | 有层次结构时 | 迭代器分类等已有标签体系的场景 |

