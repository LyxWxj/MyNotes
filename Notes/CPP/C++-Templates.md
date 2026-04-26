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

### 移动语义与enable_if<>
