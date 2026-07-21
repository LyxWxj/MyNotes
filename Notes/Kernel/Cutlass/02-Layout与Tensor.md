# Layout 与 Tensor：CuTe 的核心抽象

## 1. 什么是 Layout？

Layout 是 CuTe 最基础的概念，它回答一个问题：**逻辑坐标 (i, j) 对应内存中的哪个位置？**

```
Layout = Shape + Stride

Shape:  矩阵有多大？  → (行数, 列数)
Stride: 相邻元素在内存中隔多远？ → (行步长, 列步长)
```

### 1.1 用你已经熟悉的概念理解

```cpp
// 你熟悉的 C 语言二维数组访问：
float A[rows][cols];
A[i][j]  →  地址 = base + i * cols + j
//                         ↑          ↑
//                       stride_0   stride_1
//                       (行步长=cols) (列步长=1)

// CuTe 的 Layout 就是把这个关系抽象出来：
auto layout = make_layout(make_shape(rows, cols),    // Shape: (行数, 列数)
                          make_stride(cols, 1));      // Stride: (行步长, 列步长)
```

> [!tip] 行主序 vs 列主序
> - **行主序 (RowMajor)**: 同一行的元素在内存中连续。`stride = (cols, 1)`
> - **列主序 (ColMajor)**: 同一列的元素在内存中连续。`stride = (1, rows)`
>
> CuTe 提供了快捷方式：`GenRowMajor{}` 和 `GenColMajor{}`

### 1.2 Shape 和 Stride 可以嵌套

这是 CuTe 的强大之处 — Shape 和 Stride 可以是 **多层嵌套** 的：

```cpp
// 简单 shape：(4, 8) → 4 行 8 列
auto s1 = make_shape(4, 8);

// 嵌套 shape：((2, 2), 8) → 等价于 (4, 8)，但分成 2 组每组 2 行
auto s2 = make_shape(make_shape(2, 2), 8);

// 为什么嵌套？因为硬件就是这么组织的！
// 比如一个 warp 32 个线程，分成 4 组每组 8 个：
auto warp_layout = make_shape(make_shape(4, 8));  // (4, 8) = 32 线程
```

> [!note] 嵌套 Shape 的意义
> 嵌套 Shape 不改变总元素数，但改变了"如何分组"。这在映射线程到数据时非常有用：
> - 外层 = warp 级别分组
> - 内层 = warp 内的线程分组

### 1.3 Layout 的代数运算

Layout 支持数学运算，这是 CuTe 的核心优势：

```cpp
auto L = make_layout(make_shape(4, 8), make_stride(8, 1));

// 访问元素：L(i, j) = i * 8 + j
int offset = L(2, 3);  // = 2 * 8 + 3 = 19

// 取 shape
auto s = shape(L);      // (4, 8)

// 取 stride
auto d = stride(L);     // (8, 1)

// 取总元素数
int n = size(L);        // 32

// 取 cosize（需要的最大偏移）
int cs = cosize(L);     // = L(3, 7) + 1 = 32
```

## 2. 什么是 Tensor？

Tensor = **指针 + Layout** = 可以实际访问数据的张量视图。

```cpp
// 创建一个 tensor
auto tensor = make_tensor(ptr, layout);

// 访问元素
float val = tensor(i, j);  // = *(ptr + layout(i, j))

// tensor 本质上就是一个"知道怎么索引的指针"
```

### 2.1 不同内存空间的 Tensor

```cpp
// Global memory tensor（指向 HBM）
auto gA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, N), GenRowMajor{}));

// Shared memory tensor（指向 SMEM）
__shared__ float smem[128 * 64];
auto sA = make_tensor(make_smem_ptr(smem), make_layout(make_shape(128, 64), GenRowMajor{}));

// 寄存器 tensor（每个线程自己的数据）
auto rA = make_tensor<float>(make_shape(4, 4));  // 4×4 的寄存器 tensor
```

> [!important] Tensor 不拥有数据
> Tensor 只是一个"视图" — 它不分配内存，只是告诉你"怎么访问已有的内存"。就像 C++ 的 `std::string_view` 不拥有字符串一样。

## 3. 切 Tile：local_tile

GEMM 的核心思想是 **分块计算**：把大矩阵切成小块（tile），每个 block 处理一个 tile。

```cpp
// 原始矩阵 A: M×K
auto mA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, K), GenRowMajor{}));

// 切出第 (bx, by) 个 tile，大小为 BM×BK
auto gA = local_tile(mA,                        // 原始 tensor
                     make_shape(Int<BM>{}, Int<BK>{}),  // tile 大小
                     make_coord(bx, by));        // 第几个 tile
// gA 现在是一个 BM×BK 的 tensor 视图，指向 A 矩阵中第 (bx, by) 块
```

```
原始矩阵 A (M×K):
┌────────┬────────┬────────┐
│ tile   │ tile   │ tile   │  ← by=0
│ (0,0)  │ (0,1)  │ (0,2)  │
├────────┼────────┼────────┤
│ tile   │ [tile] │ tile   │  ← by=1, bx=1 → 这就是 gA
│ (1,0)  │ (1,1)  │ (1,1)  │
├────────┼────────┼────────┤
│ tile   │ tile   │ tile   │  ← by=2
│ (2,0)  │ (2,1)  │ (2,2)  │
└────────┴────────┴────────┘
```

## 4. 分配给线程：local_partition

有了 tile 后，需要把它分配给 block 内的每个线程。

```cpp
// 定义线程布局：32 个线程排成 4×8 的网格
auto thread_layout = make_layout(make_shape(4, 8), GenRowMajor{});

// 把 tile 按线程布局分配给线程 tx
auto tA = local_partition(gA,          // tile (BM×BK)
                          thread_layout, // 线程布局 (4×8)
                          tx);          // 当前线程 ID
// tA 是当前线程负责的那部分数据
```

```
tile (BM×BK = 16×16)，线程布局 (4×8 = 32 线程)：

线程 0  线程 1  线程 2  ... 线程 7
┌───┬───┬───┬───┬───┬───┬───┬───┐
│t0 │t1 │t2 │t3 │t4 │t5 │t6 │t7 │  row 0~3
├───┼───┼───┼───┼───┼───┼───┼───┤
│t8 │t9 │t10│t11│t12│t13│t14│t15│  row 4~7
├───┼───┼───┼───┼───┼───┼───┼───┤
│t16│t17│t18│t19│t20│t21│t22│t23│  row 8~11
├───┼───┼───┼───┼───┼───┼───┼───┤
│t24│t25│t26│t27│t28│t29│t30│t31│  row 12~15
└───┴───┴───┴───┴───┴───┴───┴───┘

每个线程负责 4 行 × 2 列 = 8 个元素
```

## 5. Identity Tensor 与边界谓词

处理边界（矩阵大小不是 tile 大小的整数倍）时，CuTe 用 **identity tensor** 来判断：

```cpp
// identity tensor：每个元素的值就是它自己的坐标
auto cA = local_tile(make_identity_tensor(mA.shape()),  // (M, K) 的坐标 tensor
                     make_shape(Int<BM>{}, Int<BK>{}),
                     make_coord(bx, by));

// cA(i, j) 返回的是原始矩阵中的 (行号, 列号)

// 生成谓词：坐标在范围内才为 true
auto predicate = make_tensor<bool>(tAcA.shape());
for (int i = 0; i < size<0>(predicate); i++)
  for (int j = 0; j < size<1>(predicate); j++)
    predicate(i, j) = get<0>(cA(i, j)) < M && get<1>(cA(i, j)) < N;

// 带谓词的拷贝：越界的位置不拷贝
copy_if(predicate, src, dst);
```

> [!tip] 为什么需要 Identity Tensor？
> 因为 `local_tile` 切出的 tile 可能超出矩阵边界（比如矩阵 100×100，tile 128×128）。identity tensor 记录了每个元素在原始矩阵中的坐标，这样就能判断哪些位置是合法的。

## 6. 完整示例：矩阵转置的 CuTe 写法

```cpp
template <typename T, int BM, int BN>
__global__ void transpose_cute(const T* A, T* B, int M, int N) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  // 1. 创建全局 tensor
  auto mA = make_tensor(make_gmem_ptr(A),
                        make_layout(make_shape(M, N), GenRowMajor{}));
  auto mB = make_tensor(make_gmem_ptr(B),
                        make_layout(make_shape(N, M), GenRowMajor{}));

  // 2. 切 tile（注意坐标交换 = 转置）
  auto gA = local_tile(mA, make_shape(Int<BM>{}, Int<BN>{}), make_coord(bx, by));
  auto gB = local_tile(mB, make_shape(Int<BN>{}, Int<BM>{}), make_coord(by, bx));

  // 3. 按线程分配
  auto tA = make_layout(make_shape(Int<BM>{}, Int<BN>{}), GenColMajor{});
  auto tB = make_layout(make_shape(Int<BN>{}, Int<BM>{}), GenRowMajor{});
  auto tAgA = local_partition(gA, tA, tx);
  auto tBgB = local_partition(gB, tB, tx);

  // 4. 边界谓词 + 拷贝
  auto cA = local_tile(make_identity_tensor(mA.shape()),
                       make_shape(Int<BM>{}, Int<BN>{}), make_coord(bx, by));
  auto tAcA = local_partition(cA, tA, tx);
  auto pred = make_tensor<bool>(tAcA.shape());
  for (int i = 0; i < size<0>(pred); i++)
    for (int j = 0; j < size<1>(pred); j++)
      pred(i, j) = get<0>(tAcA(i, j)) < M && get<1>(tAcA(i, j)) < N;

  copy_if(pred, tAgA, tBgB);  // 一行完成转置！
}
```

> [!note] 对比手写版本
> 手写版本需要：
> - 手算 `global_idx = blockIdx.x * blockDim.x + threadIdx.x`
> - 手算 `global_row = global_idx / col`
> - 手算 `y[col * row + row] = x[global_idx]`
> - 手写边界检查
>
> CuTe 版本只需要：声明 Layout → 切 Tile → 分配线程 → 拷贝。
