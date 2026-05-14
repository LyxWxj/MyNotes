# C++-Templates

## Basic

### 多模板参数

我们希望自动推导返回类型的时候有下面几种写法：

- 通过构造一个表达式来推导返回类型
- 通过`std::common_type`来推导返回类型

```cpp
template <typename T1, typename T2>
auto f0(T1 A, T2 B) -> decltype(A > B ? A : B) {
  return A > B ? A : B;
}

template <typename T1, typename T2>
auto f1(T1 A, T2 B) -> typename std::decay<decltype(A > B ? A : B)>::type{
  return A > B ? A : B;
}

template <typename T1, typename T2>
auto f2(T1 A, T2 B) -> typename std::common_type<T1, T2>::type {
  return A > B ? A : B;
}

template <typename T1, typename T2,
  typename RT = std::decay_t<decltype(A > B ? A : B)>
>
auto f3(T1 A, T2 B) -> RT {
  return A > B ? A : B;
}
```

由于在编译时不能确定`common_type::type`是否是一个类型，我们需要显式地加上typename来告诉编译器它是一个类型。

### 模板的重载

- 不同模板参数个数视为重载

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

重要的是，需要从左到右按顺序推断的模板参数，如果自己显式提供的第一个模板参数和需要推断的实参类型相同那么他会认为这两个都对应到第一个模板参数上，即实际上只提供了一个模板参数，导致两个重载都匹配。

### 内联与编译时计算

```cpp
template<typename T1, typename T2>
auto g1(T1 a, T2 b) -> decltype(a > b ? a :
  b) {
  return a > b ? a : b;
}

template<typename T1, typename T2>
constexpr auto g2(T1 a, T2 b) -> decltype(a > b ? a : b) {
  return a > b ? a : b;
}

int a[g1(sizeof(int), 1000u)]; // error
int b[g2(sizeof(int), 1000u)]; // ok
```

constexpr函数可以尽量在编译期计算结果（当参数都在编译期可知时），而普通函数只能在运行时计算。

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

在类模板中使用不带模板参数的类名，表示这个内部类的模板参数和模板类相同
例如：

```cpp
template<typename T>
class Stack {
public:
  ...
  Stack (Stack const&);
  Stack& operator=(Stack const&);
  ...
}
```

等同于

```cpp
template<typename T>
class Stack {
public:
  ...
  Stack (Stack<T> const&);
  Stack& operator=(Stack<T> const&);
  ...
}
```

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

特化的模板参数列表必须和原模板参数列表完全匹配，不能有默认参数，不能有模板参数包。

### 偏特化

部分特化参数或者特化部分参数

```cpp
template<typename T>
class Stack<T*>{

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

### 类型别名

```cpp
typedef std::stack<int> intstack;
using intstack = std::stack<int>;
```

此外，`using`可以模板化，这种方式被称为别名模板：

```cpp
template<typaname T>
using DequeStack = 
std::stack<T, std::deque<T>>;
```

可以用一个参数来绑定两个模板参数。

```cpp
template<typnname T> struct MyType {
  using iterator=...;
};

template<typename T>
using MyTypeIter = typename MyType<T>::iterator;
```

因为不能确定MyType<T>::iterator是否是一个类型，所以需要加上typename来告诉编译器它是一个类型。

### 类模板的参数推导

C++17 前，必须将所有模板参数类型传递给类模板 (除非有默认值)。C++17 后，指定模板参数
的约束放宽了。相反，若构造函数能够推导出所有模板参数 (没有默认值)，则可以不用显式定义模
板参数。

```cpp
stack<int> s1; // stack of ints
stack<int> s2 = s1; // copy constructor
stack s3 = s1 // OK since C++17
auto s4 = s1; // OK 自动推断整个类型
vector v1{1, 2, 3}; //自动推导类型
```

推导的结果和模板类的构造函数接受的参数有关

```cpp
stack(T const& elem) // reference
stack stringstack="stringstack; // 推导结果：stack<char const[]>
```

```cpp
stack(T elem) // by value
stack stringstack="stringstack; // 推导结果：stack<char const*>
```

也可以用推导指引（deduction guides）

```cpp
stack(char const*)->stack<std::string>;
stack stringStack{"bottom"}; // stack<string> deduced and valid;
stack stringStack={"bottom"}; // stack<string> deduced and valid;
stack stringStack2 = "bottom"; // Error
```

### 模板聚合

聚合类 (不由用户提供、显式或继承的构造函数的类/结构，没有 private 或 protected 的非静态
数据成员，没有虚函数，也没有 virtual、private 或 protected 基类) 也可以是模板

```cpp
template<typename T>
struct S {
  T value;
  string conmment;
};

S(const char*, const char*)->S<std::string>;
```

### 非类型模板参数

对于函数和类模板来说，模板参数可以是类型，也可以是普通值。与使用类型参数的模板一样，
定义在使用之前。使用这样的模板时，必须显式地指定值。

非类型模板参数有一些限制，只能是整型常量值 (包括枚举)，指向对象/函数/成员的指针，指
向对象或函数的左值引用，或者 std::nullptr_t(nullptr 的类型)。
浮点数和类型对象不允许作为非类型模板参数:

```cpp
template<double VAT> // ERROR: floating-point values are
double process (double v) // not allowed as template parameters
{
  return v*VAT;
}
template<std::string name> // ERROR: class-type objects are not
class Myclass { // alowed as template parameters
};
```

C++17 后，可以自动推断非类型模板参数。

```cpp
template<typename T, auto Maxsize>
class stack{
  using size_type = decltype(Maxsize);
  std::array<T,Maxsize> elems;
}

stack<int, 100> s1; // ints20
stack<int, 100u> s2; // ints20
```

### 可变参数模板

最简单的例子，递归调用print()来打印所有参数:

```cpp
void print() {}
template<typename T, typename... Args>
void print(T&& first, Args&&... args) {
  std::cout << first << '\n>>
  print(args...);
}
```

在`typename...`中的`...`表示这是一个模板参数包，
`Args&&...`和`args...`表示对参数包解包。

或者可以使用操作符`sizeof...`来获取参数包中参数的数量:

```cpp
template<tupename T, typanem... Types>
void print(T** first, Types... args) {
  cout << first << '\n';
  if( sizeof... (args) > 0) {
    print(args...);
  }
}
```

C++17后，有一个特性可以对参数包的所有参数使用二元运算符计算结果

```cpp
template<typanem ... T>
auto foldsum(T... s) {
  return (s + ...);
}
```

如果参数包为空，表达式是错误格式的。
折叠表达式分一下几种类型：

- 一元左折叠: `( ... op pack )`
- 一元右折叠: `( pack op ... )`
- 二元左折叠: `( init op ... op pack )`
- 二元右折叠: `( pack op ... op init )`
几乎所有的二元运算符都可以用折叠表达式。

```cpp
struct Node{
  int v;
  Node* left, *right;
  Node(int i = 0):v(i), left(nullptr), right(nullptr) {};
};

template<typename T, typename... TP>
Node* traverse(T np, Tp... paths) {
  return (np->*...->*paths); // np->*paths1->*paths2
}z
```

这种使用初始化器的折叠表达式，可以简化可变参数模板来打印上面的所有参数

```cpp
template<typename... Args>
void print(Args&&... args) {
  (std::cout << ... << args) << '\n';
}
```

但无法在参数中添加打印分隔符，需要一个模板类来实现

```cpp
template<typename T, char Sep>
class AddSeparator {
private:
  T const& ref;
  public:
  AddSeparator(T const& r): ref(r) {}
  friend std::ostream& operator<<(std::ostream& os, AddSeparator<T, Sep>s){
    return os << s.ref << Sep;
  }
}; 
template<typename... Args>
void print(Args&&... args) {
  (std::cout << ... << AddSeparator<Args, ','>(args)) << '\n';
}
```

`traverse` 能够传入 `&Node::left`、`&Node::right` 这样的参数，是因为它们并不是**某个具体对象的成员变量的地址**，而是**成员指针（pointer to member）**。

#### 关键点

- **成员指针**（如 `&Node::left`）在编译时就已经确定，它表示该成员在类 `Node` 中的**偏移量**（或称为“成员描述符”），而不是内存中某个实际存在的变量的地址。
- 成员指针可以独立于任何对象存在，因此即使没有创建任何 `Node` 对象，`&Node::left` 本身也是合法且有效的编译时常量。
- 在 `traverse` 函数中，通过 `np->*...` 这样的语法，将成员指针与具体的对象 `np` 相结合，才能得到该对象中对应成员的实际地址或引用。

#### 简单类比

- 普通指针：`int x = 5; int* p = &x;` —— `p` 指向**具体的变量** `x` 的地址，如果 `x` 不存在，`&x` 就是非法的。
- 成员指针：`int Node::* p = &Node::v;` —— `p` 只是记录了 `v` 在 `Node` 类中的位置，它不需要有任何 `Node` 对象就已经存在。

因此，`&Node::left` 并不是“不存在的变量的地址”，而是一个**类型安全的偏移量**，它本身是合法的常量，当然可以传递给函数模板。

### 类模板和表达式

参数包还可以出现在其他地方，例如表达式、类模板、using 声明，甚至推导策略。

```cpp
// 将每个参数加倍后打印
template<typename... Args>
void printdouble(Args&&... args){
  print(args + args...);
}
```

返回所有参数类型是否相同

```cpp
template<typename T1, typename... TN>
bool is_sameall(T1, TN...) {
  return (std::is_same<T1,TN>::value && ...);
}

// or 
template<typename T1, typename... TN>
constexpr bool is_sameall(T1, TN...) {
  return (std::is_same_v<T1, TN>&& ...);
}
```

打印多个索引

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

如果idx作为模板参数，则需要将其提到前面来，因为模板参数必须从左到右依次推断，而形参的类型最后推断，因此作为形参类型的容器应该放在idx的后面

也可以用一个类型来记录索引参数包

```cpp
template<std::size_t...>
struct Indices {};
// 类型本身是空的，我们使用模板来记录索引参数包
template<typename T, std::size_t... Idx>
void printElems(T const& t, Indices <Idx...>) {
  print(t[Idx]...);
}
```

推导策略也可以用于可变参数包，例如：

```cpp
template<typename T, typename... U>
array(T, U...) -> array<enable_if_t<(std::is_same_v<T, U> && ...), T >, (1 + sizeof... (U))>
```

参数包是可以继承的

```cpp
template<typename... Bases>
struct overloader: Base... {
  using Bases::operator()...; // 继承参数包中所有基类的operator()函数
}
```

### 基础技巧

#### typename

使用某个类内部的类型时需要加上typename，否则编译器会优先假定这是一个静态成员变量

```cpp
template<typename T>
void f() {
  typename T::value_type x; // value_type是T的一个类型成员
}
```

#### 零初始化

简单的定义对内置类型并没有进行初始化

因此，可以显式调用内置类型的默认构造函数，该构造函数用 0 初始化内置类型 (bool 为 false，
指针为 nullptr)。因此，即使是内置类型，也可以通过编写以下代码来确保正确的初始化

```cpp
template<typename T>
void foo(){
  T x{}; //
}
```

为了确保将类型参数化的类模板成员初始化，可以定义默认构造函数，使用带大括号的初始化式来初始化成员

```cpp
template<typename T>
class SomeClass {
  private: 
    T x;
  public:
    SomeClass(): x{} {} // 使用大括号初始化成员x
}
```

或者

```cpp
template<typename T>
class SomeClass {
  private: 
    T x{};
  public:
    SomeClass(): {} // 使用大括号初始化成员x
}
```

但是默认参数不可以这样

```cpp
template<typename T>
void foo(T x{}) {}// False
template<typename T>
void foo(T x = T{}) {} // Correct
```

#### .template成员函数调用

```cpp
template <unsigned long N>
void printBitset(std::bitset<N> const& Bs) {
  std::cout << Bs.template to_string<char, std::char_traits<char>, std::allocator<char>>();
}
```

对于 bitset bs，使用 to_string() 的成员函数模板，同时显式指定字符串类型的信息。如果没有使
用.template，编译器就不知道后面的小于标记 (<) 是模板参数列表的开头。注意，只有在句点之前的
构造依赖于模板参数时才会出现问题。在例子中，参数 bs 依赖于模板参数 N。

#### 变量模板

我们有非常相似的术语来描述非常不同的事情: 变量模板是一个变量，它是一个模板 (变
量在这里是一个名词)。可变参数模板是用于可变数量模板参数的模板 (可变参数在这里是形
容词)。

```cpp
template<typename T>
constexpr T pi = T(3.1415926535897932385);

std::cout << pi<double> << '\n'; // 3.14159
std::cout << pi<int> << '\n'; // 3
```

变量模板可以有默认模板参数

```cpp
template<typename T=long double>
constexpr T pi = T{3.1415926535897932385};
pi<> // pi <long double>
pi<int> //
pi // Error 必须要有<>
```

变量模板可以用非类型参数进行参数化

```cpp
template<int N>
std::array<int,N> arr{};

template<auto N>
constexpr decltype(N) dval=N;
```

第二行有一种等价写法

```cpp
template<auto N>
constexpr decltype(N) dval = N;

template <typename T, T N>
constexpr T dval2 = N;

std::cout << dval<1> << std::endl;
std::cout << dval2<int, 2> << std::endl;
```

#### 数据成员的变量模板

如果有某一个模板类中特化不同的静态成员，可以用变量模板取到其中的成员

```cpp
template<typename T>
class MyClass {
  public:
    static constexpr int max=1000;
};
template<>
class Myclass<float> {
  public:
  static constexpr int max = 10;
};
template<typename T>
int myMax = MyClass<T>::max;
auto i = myMax<int>;
auto f = myMax<float>;
```

#### 双重模板参数

允许模板参数本身是类模板

```cpp
stack<int, vector<int>> stk1;
stack<int, vector> stk2;
```

可以省略第二个模板参数，无需重新指定容器元素的
类型

```cpp
template<typename T,
      template<class E> class Container = std::deque>
class Stack{}; // OK
template<typename T,
      template<typename E> typename Conainer = std::deque>
class Stack{}; // OK;
template<typename T,
      template<typename E> class Conainer = std::deque>
class Stack{}; // OK;
template<typename T,
      template<typename> class Conainer = std::deque>
class Stack{}; // OK;
```

### 移动语义与`enable_if<>`

以下代码表示完美转发：

```cpp
template<typename T>
void foo(T&& t) {
  g(std::forward<T>(t));
}
```

`std::enable_if<cond, T=void>`,在编译时条件下忽略函数模板

```cpp
template<typename T>
typename std::enable_if<(sizeof (T) > 4)>::type foo(T t) {
  // ...
}
```

如果sizeof(T)>4生成false，则忽略foo<>的定义
如果结果为true，函数模板实例展开为

```cpp
void foo(T t) {};
```

或者`std::enable_if<>::type`被实例化为第二个模板参数

简便写法：

```cpp
template<typename T, typename = std::enable_if_t<(sizeof(T) > 4)>>
void foo(){}
// OR
template<typename T>
using EnableIfSizeGreater4 = std::enable_if_t<(sizeof(T) > 4)>;
template<typename T, typename = EnableIfSizeGreater4>
void foo(){};
```

#### 用概念简化`enable_if<>`

`requires`关键字

```cpp
template<typanem STR>
requires std::is_convertible_v<STR,std::string>
Person(STR&& n):name(std::forward<STR>(n)) {
  ...
}
```

`concept`关键字

```cpp
template<typanem T>
concept ConvertibleToString = std::is_convertible_v<T,std::string>;

template<typename STR>
requires ConvertibleToString<STR> Person(STR&& n): name(std::forward<STR>(n)) {
  ...
}
```

### 模板元编程

使用模板进行素数判断的例子

```cpp
template<unsigned p, unsigned d>
struct DoIsPrime {
  static constexpr bool value = (p$d != 0) && DoIsPrime<p, d-1>::value;
}
template<unsigned p>
struct DoIsPrime<P,2>{
  static constexpr bool value = (p%2 != 0);
};
template<typename p>
struct IsPrime {
  static constexpr bool value = DoIsPrime<p, p/2>::value;
};

template<>
struct IsPrime<0> {static constepxr bool value = false;};
template<>
struct IsPrime<1> {static constexpr bool value = false;};
template<>
struct IsPrime<2>
{static constexpr bool value = true;};
template<>
struct IsPrime<3> {static constexpr bool value = true;};
```

C++14 中，constexpr 函数可以使用通用 C++ 代码中的控制结构。因此，不用编写笨拙的模板
代码或有些“奇怪的”单行程序，现在只使用普通的 for 循环:

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

可以直接调用

```cpp
isPrime(9);
```

isPrime() 等编译时测试的一种应用是，在编译时使用偏特化
在不同实现之间进行选择。

```cpp
template<int SZ, bool =isPrime(SZ)>
struct Helper;
template<int SZ>
struct Helper<SZ, false>
{
  
};
template<int SZ>
struct Helper<SZ,true>
{

};
template <typename T, std::size_t SZ>
long foo (std::array<T,SZ> cosnt& coll) {
  Helper<SZ> h;
}
```

#### SFINAE (替换失败不是错误)

C++ 中，以各种参数类型重载的函数很常见。因此，当编译器看到对重载函数的调用时，必须
考虑每个候选函数，评估调用参数，并选择最匹配的候选函数。
候选集包括函数模板的情况下，编译器首先必须确定为该候选对象使用哪些模板参数，然后在
函数参数列表及其返回类型中替换这些参数，然后评估匹配程度。但替换过程可能会遇到问题：可能产生毫无意义的构造。语言规则并不认为这种无意义的替换会导致错误，而具有这种问题的候选则会直接忽略。
这就是所谓的 SFINAE（Substitution Failure Is Not An Error）原则。

```cpp
template<typename T, unsigned N>
std::size_t len(T(&)[N]) {
  return N;
}
```

```cpp
template<typename T>
typename T::size_type len(T const& t) {
  return t.size();
}
```

1. 第一个函数模板将参数声明为 T[&](N)，从而参数必须是由 N 个 T 类型元素组成的数组。
2. 第二个函数模板将参数声明为 T，没有对参数施加任何约束，而是返回类型 T::size_type，这
要求传递的参数类型具有 size_type 成员变量。
当传递数组或字符串字面量时，只有数组的函数模板匹配:

```cpp
int a[10];
std::cout << len(a)； // OK : only len() for array matches
std::cout << len("temp"); // OK : only len() for array matches
std::cout << len(vector<int>{0}); // OK
std::cout << len(allocator<int>{}); //ERROR: len() selected, but x has no size();
```

例如，想确保函数模板 len() 对于具有 size_type 成员，但没有 size() 成员函数的参数类型就会
忽略。函数声明中没有对 size() 成员函数的要求，最终会在实例化时出错:

有一种常见的模式或习语可以用来处理这种情况:

- 用尾部返回类型语法指定返回类型 (前面使用 auto，在末尾返回类型之前使用->)。
- 使用 decltype 和逗号操作符定义返回类型。
- 给出以逗号操作符开头的表达式 (在重载逗号操作符时转换为 void)。
- 在逗号操作符的末尾定义一个实际返回类型的对象。

```cpp
template<typename T>
auto len( T const& t) -> decltpe((void)(t.size()), T::size_type()) {
  return t.size();
}
```

返回类型是`decltype( (void)(t.size()), T::size_type())`

### 通用库
`std::invoke()` 的一个常见应用是封装单个函数调用 (例如，记录调用，测量持续时间，或准备一
些上下文，例如启动一个新线程)。现在，可以通过完美转发可调用参数和传递参数来支持移动语
义:
```cpp
#include <utility>
#include <functional>

template<typename Callable, typename... Args>
decltype(auto) call(Callable&& op, Args&&... args) {
  return std::invoke(std::forawrd<Callable>(op), std::forward<Args>(args)...);
}
```
以便将其“完美地”转发回调用者,为了支持返回引用 (比如 std::ostream&)，必须使用 decltype(auto) 而不是 auto:
decltype(auto)(C++14) 是一个占位符类型，根据相关表达式的类型 (初始化器、返回值或模板参
数) 确定变量、返回类型或模板参数的类型。详见 15.10.3 节。
若将 std::invoke() 返回的值临时存储在变量中，以便在执行其他操作 (例如，处理返回值或记录
调用结束) 后返回，还必须使用 decltype(auto) 声明临时变量:
```cpp
decltype(auto) ret{std::invoke(std::forward<Callable>(op),std::forward<Args>(args)...)};
return ret;
```

注意，用 auto&& 声明 ret 并不正确。作为一个引用，auto&& 扩展返回值的生命周期直到作用
域结束 (参见第 11.3 节)，但不超出函数调用者的返回语句。
使用 decltype(auto) 也有一个问题: 若可调用对象的返回类型为 void，则不允许将 ret 初始化为
decltype(auto)，因为 void 是一个不完整的类型。现在，有以下选择:
- 在语句的前一行声明一个对象，其析构函数执行希望实现的可观察行为。例如:

```cpp
struct cleanip{
  ~cleanup(){}
} dummy;
return std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...);
```
- 以不同的方式实现 void 和非 void 的情况:
```cpp
#include <utility>
#include <functional>
#include <type_traits>

template<typename Callable, typename... Args>
decltype(auto) call(Callable&& op, Args&&... args) {
  if constexpr(std::is_same_v<std::invoke_result_t<Callable, Args...>, void>) {
    std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...);
    return;
  }else {
    decltype(auto) ret{std::invoke(std::forward<Callable>(op), std::forward<Args>(args)...)};
    return ret;
  }
}
```

必须特别小心地使用类型特征: 其行为可能与 (菜鸟期) 开发者所期望有所不同。例如
```cpp
std::remove_const_t<int const&> 
```
这里因为引用不是const（尽管不能修改），因此删除引用和const的顺序很重要
```cpp
std::remove_const_t<std::remove_reference_t<int const&>> // int
std::remove_reference_t<int const&> // int const
std::decay_t<int const&> // yields int
```

std::addressof<>() 函数模板生成对象或函数的实际地址。即使对象类型有重载操作符 &，也能
工作。尽管后者很少使用，但可能会发生 (例如，在智能指针中)。因此，如果需要任意类型对象的
地址，建议使用 addressof():
```cpp
template<typename T>
void f(T&& x) {
  auto p = &x; // might fail with overloaded operator &
  auto q = std::addressof(x); // works even with overloaded operator &
}
```

std::declval<>() 函数模板可以用作特定类型的对象引用的占位符。该函数没有定义，因此不能
调用 (也不创建对象)。因此，只能用于未求值的操作数 (decltype 和 sizeof 构造的操作数)。因此，与
其尝试创建一个对象，可以假设有一个相应类型的对象。
```cpp
#include <utility>
template<typename T1, typename T2, 
      typename RT = std::decay_t<decltype(true ? std::declval<T1>() : std::declval<T2>())>
>
RT max(T1 a, T2 b) {
  return b < a ? a : b;
}

```

容易自动产生引用的地方：
```cpp
#include <iostream>

template<typename T>
void tmplParamIsReference(T) {
  std:: cout << "T is referecne: " << std::is_reference_v<T> << '\n';
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
调用时指定模板参数可以解决问题，但是在模板设计的时候可能没有考虑过这种情况从而触发错误或意外行为

```cpp
template<typename T, T Z = T{}>
class RefMem {
  private: 
    T zero;
  public:
    RefMem():zero(Z) {

    }
};
```
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
使用 decltype(auto) 可以很容易地产生引用类型，因此在上下文中最好不要使用 (默认使用 auto)。
详见 15.10.3 节。
出于这个原因，标准库有时会有奇怪的规范和约束。例如:
• 为了在模板参数为引用实例化时仍然具有赋值操作符，std::pair<> 和 std::tuple<> 类实现了赋
值操作符，而不是使用默认行为。例如:
```cpp
template<typename T1, typename T2>
struct pair {
  T1 first; T2 second;
  pair(pair const&) = default;
  pair(pair&& ) = default;

  pair& operator=(pair const& p);
  pair& operator=(pair&& p) noexcept(...);
}
```

实现模板时，有时会出现这样的问题: 代码是否能够处理不完整的类型 (参见 10.3.1 节)。来看
看下面的类模板:
```cpp
template<typename T>
class Cont {
  private:
  T* elems;
  public:
  ...
};
struct Note {
  std::string value;
  Cont<Node> next;// only use Pointer
}
```

然而，仅通过使用一些特性，就会失去处理不完整类型的能力。例如:
```cpp
template<typename T>
class Cont {
  private:
    T* elems;
  public:
    typename std::conditional<std::is_move_constructible<T>::value, T&& T&>::type foo();
}
```
这里，使用特征 std::conditional(参见 D.5 节) 来决定成员函数 foo() 的返回类型是 T&& 还是 T&。
这取决于模板参数类型 T 是否支持移动语义。
问题是特性 std::is_move_constructible 要求参数是一个完整的类型 (不是 void 或未知边界的数
组; 参见的 D.3.2 节)。在 foo() 的这个声明中，struct Node 的声明失败了
> 如果 std::is_move_constructible 是一个完整的类型，并不是所有的编译器都会产生错误。
> 因为对于这种错误，不需要进行诊断。所以，在需要平台移植时需要考虑这个问题。
可以将 foo() 替换为成员模板来解决这个问题，这样 std::is_move_constructible 的计算就会延迟
到 foo() 的实例化点:
```cpp
template<typename T>
class Cont {
  private:
    T* elems;
  public:
    template<typename D = T> std::conditional<std::is_move_constructible<T>::value, T&&, T&>::type foo();
}
```
现在，特性依赖于模板参数 D(默认为 T，我们想要的值)，编译器必须等到 foo() 调，如 Node 之
前，再评估特性 (那时 Node 是一个完整的类型，只是在定义时不完整)。

#### 编写泛型库
让我们列出一些在实现泛型库时需要记住的事情 (注意，其中一些可能会在后面会介绍到):
• 模板中使用转发引用来转发值 (参见第 91 页 6.1 节)。如果值不依赖于模板参数，使用
auto&&(参见 11.3 节)。
• 当参数声明为转发引用时，模板参数在传递左值时要有引用类型 (参见 15.6.2 节)。
• 当需要依赖于模板形参的对象地址时，使用 std::addressof()，以避免当对象绑定到带有重载操
作符 & 的类型时出现意外 (11.2.2 节)
• 对于成员函数模板，确保不会比预定义的复制/移动构造函数或赋值操作符更好地匹配 (6.4
节)。
• 模板参数可能是字符串字面值，且不通过值传递时 (7.4 节和 D.4 节)，请考虑使用 std::decay。
• 如果模板参数有 out 或 inout，请准备好处理参数可能指定为 const 类型的情况 (参见 7.2.2 节)。
• 准备好处理模板参数引用的副作用 (参见 11.4 节了解详细信息，19.6.1 节为示例)。特别是，要
确保返回类型不能是引用 (参见 7.5 节)。
• 准备好处理不完全类型，从而进行以支持，例如：递归数据结构 (参见 11.5 节)。
• 重载所有数组类型，而不仅仅是 T[SZ](参见 5.4 节)。