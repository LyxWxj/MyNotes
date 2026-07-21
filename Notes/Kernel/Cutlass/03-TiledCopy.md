# TiledCopy：数据搬运

## 1. 为什么需要 TiledCopy？

GPU 计算的核心模式是：
```
GMEM → SMEM → Register → 计算 → Register → SMEM → GMEM
```

每一步都需要 **拷贝** 数据。TiledCopy 就是 CuTe 对"分块拷贝"的抽象。

> [!tip] TiledCopy 解决什么问题？
> 手动写拷贝需要：
> - 算每个线程拷贝哪些元素
> - 处理向量化（float4 = 128-bit load）
> - 处理边界（矩阵不是 tile 整数倍时）
> - 处理 bank conflict（SMEM 访问冲突）
>
> TiledCopy 把这些全部封装了。

## 2. Copy_Atom：最小拷贝单元

Copy_Atom 是 CuTe 中最小的拷贝操作，对应一条或几条机器指令：

```cpp
// 自动向量化拷贝（编译器自动选择最宽的指令）
Copy_Atom<AutoVectorizingCopy, float>   // float: 可能选 LD.E.128 (128-bit)

// 显式指定拷贝宽度
Copy_Atom<SM80_16x8_LDSM_T, half>       // Ampere ldmatrix 指令

// Hopper TMA 拷贝
Copy_Atom<SM90_TMA_LOAD, half>          // Hopper TMA 指令
```

> [!note] Copy_Atom vs 普通拷贝
> 普通拷贝：`dst[i] = src[i]` — 一条指令拷一个元素
> Copy_Atom：描述"用什么指令拷贝" — 可能是一条 `ldmatrix` 拷 8×8 个元素

## 3. make_tiled_copy：构建分块拷贝

`make_tiled_copy` 把 Copy_Atom 组织成一个"分块拷贝描述符"：

```cpp
auto tiled_copy = make_tiled_copy(
    Copy_Atom<AutoVectorizingCopy, float>{},  // 原子拷贝操作
    make_layout(make_shape(16, 16), GenRowMajor{}),  // 线程布局：16×16 = 256 线程
    make_layout(make_shape(4, 1), GenRowMajor{})     // 每线程处理 4×1 = 4 个元素
);
```

**三个参数的含义**：

```
参数 1: Copy_Atom — 每次拷贝用什么指令
参数 2: 线程布局 — 256 个线程怎么排列（16 行 × 16 列）
参数 3: 值布局 — 每个线程处理多少元素（4 行 × 1 列）

总拷贝量 = 线程数 × 每线程元素数 = 256 × 4 = 1024 个 float
         = 16×16 线程 × 4×1 元素/线程 = 64×16 的 tile
```

### 3.1 图解

```
TiledCopy: 线程布局 (4×8), 值布局 (2×2)

源 tile (8×16):
┌──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┐
│T0│T0│T1│T1│T2│T2│T3│T3│T4│T4│T5│T5│T6│T6│T7│T7│  row 0~1
│  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │  │
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│T8│T8│T9│T9│..│..│..│..│..│..│..│..│..│..│T15│  row 2~3
├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
│  ...以此类推...                                  │  row 4~7
└──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┘

每个线程负责 2×2 = 4 个元素（值布局）
32 个线程覆盖 8×16 的 tile（线程布局 4×8 × 值布局 2×2）
```

## 4. get_slice 与 partition

构建好 TiledCopy 后，需要把它"切片"给每个线程：

```cpp
// 获取当前线程的拷贝视图
auto thr_copy = tiled_copy.get_slice(threadIdx.x);

// 从源 tensor 和目标 tensor 中分出当前线程负责的部分
Tensor tSrc = thr_copy.partition_S(src);  // 源：当前线程要读的部分
Tensor tDst = thr_copy.partition_D(dst);  // 目标：当前线程要写的部分

// 执行拷贝
copy(tiled_copy, tSrc, tDst);
```

> [!tip] partition_S vs partition_D
> - `partition_S` (Source): 从源 tensor 中分出当前线程读的部分
> - `partition_D` (Destination): 从目标 tensor 中分出当前线程写的部分
> - 两者 shape 相同（同一个线程读和写的数据量一样）

## 5. TiledCopy 的两种用法

### 5.1 GMEM → SMEM（Block 级搬运）

```cpp
// 构建分块拷贝
auto copy_g2s = make_tiled_copy(
    Copy_Atom<AutoVectorizingCopy, float>{},
    make_layout(make_shape(Int<BM/4>{}, Int<BN>{}), GenRowMajor{}),
    make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

// 获取线程视图
auto thr_copy = copy_g2s.get_slice(threadIdx.x);
auto tAgA = thr_copy.partition_S(gA);  // 源：GMEM tile
auto tAsA = thr_copy.partition_D(sA);  // 目标：SMEM tile

// 执行拷贝
copy(copy_g2s, tAgA, tAsA);
__syncthreads();
```

### 5.2 SMEM → Register（MMA 前的数据加载）

```cpp
// 从 TiledMma 构建 S2R 拷贝（和 MMA 的数据布局匹配）
auto copy_s2r = make_tiled_copy_A(Copy_Atom<SM80_16x8_LDSM_T, half>{}, tiled_mma);

auto thr_copy = copy_s2r.get_slice(threadIdx.x);
auto tAsA = thr_copy.partition_S(sA);  // 源：SMEM
auto tArA = thr_copy.partition_D(rA);  // 目标：寄存器

copy(copy_s2r, tAsA, tArA);
```

> [!important] make_tiled_copy_A / _B / _C
> `make_tiled_copy_A` 是专门为 MMA 的 A 矩阵构建的拷贝。它保证拷贝出来的数据布局和 MMA 指令期望的 fragment 布局一致。
>
> 如果用普通的 `make_tiled_copy`，拷贝出来的数据布局可能和 MMA 不匹配，需要额外的 `retile` 操作。

## 6. AutoVectorizingCopy

最常用的 Copy_Atom。它会自动选择最宽的拷贝指令：

```cpp
Copy_Atom<AutoVectorizingCopy, float>
// 如果 4 个 float 连续且对齐 → 生成 LD.E.128 (128-bit, 一条指令)
// 如果 2 个 float 连续 → 生成 LD.E.64 (64-bit)
// 如果只有 1 个 float → 生成 LD.E.32 (32-bit)
```

> [!tip] 什么时候用 AutoVectorizingCopy？
> 大多数情况都用它。只有在需要显式控制拷贝行为时才用其他 Copy_Atom（比如 `SM80_16x8_LDSM_T` 对应 `ldmatrix` 指令）。

## 7. retile：重排数据布局

有时候拷贝出来的数据布局和目标不匹配，需要 `retile`：

```cpp
// 拷贝到寄存器
auto tArA = thr_copy.partition_S(sA);  // 从 SMEM 分出的数据
auto tArA_reg = make_tensor_like(tArA);  // 创建同 shape 的寄存器 tensor
auto tArA_view = thr_copy.retile_D(tArA_reg);  // 重排为目标布局

copy(copy_s2r, tAsA, tArA_view);  // 拷贝时自动重排
```

> [!note] retile 不拷贝数据
> `retile` 只是改变 tensor 的"视图"（shape/stride 的解释方式），不移动数据。它让同一个寄存器数据可以被不同方式访问。

## 8. 完整示例：GMEM → SMEM → RF → 计算

```cpp
__global__ void gemm_kernel(float* A, float* B, float* C, int M, int N, int K) {
  // 1. 创建 tensor
  auto gA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, K), GenRowMajor{}));
  auto gB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(N, K), GenRowMajor{}));

  // 2. 切 tile
  auto tileA = local_tile(gA, make_shape(BM, BK), make_coord(bx, by));
  auto tileB = local_tile(gB, make_shape(BN, BK), make_coord(by, bx));

  // 3. SMEM
  __shared__ float smemA[BM * BK], smemB[BN * BK];
  auto sA = make_tensor(make_smem_ptr(smemA), make_layout(make_shape(BM, BK), GenRowMajor{}));
  auto sB = make_tensor(make_smem_ptr(smemB), make_layout(make_shape(BN, BK), GenRowMajor{}));

  // 4. GMEM → SMEM
  auto copy_g2s = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, float>{}, ...);
  copy(copy_g2s.get_slice(tx).partition_S(tileA),
       copy_g2s.get_slice(tx).partition_D(sA));
  __syncthreads();

  // 5. SMEM → Register
  auto copy_s2r_A = make_tiled_copy_A(Copy_Atom<AutoVectorizingCopy, float>{}, tiled_mma);
  copy(copy_s2r_A.get_slice(tx).partition_S(sA),
       copy_s2r_A.get_slice(tx).partition_D(rA));

  // 6. MMA 计算
  cute::gemm(tiled_mma, rA, rB, rC);
}
```
