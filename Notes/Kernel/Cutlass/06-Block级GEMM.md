# Block 级 GEMM：把所有组件组合起来

## 1. GEMM 的整体流程

一个完整的 GEMM kernel `C[M×N] = A[M×K] × B[K×N]` 的流程：

```
for 每个 K 方向的 tile:
    1. GMEM → SMEM: 把 A 的一个 BM×BK tile 和 B 的一个 BK×BN tile 搬到 SMEM
    2. SMEM → RF: 从 SMEM 加载数据到寄存器
    3. MMA: 用 Tensor Core 计算 C += A × B
4. RF → SMEM → GMEM: 把结果 C 写回
```

## 2. Block MMA 与 Block Copy

在 CuTe 中，"Block 级"的操作由 **TiledMma** 和 **TiledCopy** 描述：

```cpp
// Block MMA：整个 block 协作的矩阵乘法
auto tiled_mma = make_tiled_mma(
    MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>{},
    make_layout(make_shape(Int<4>{}, Int<2>{}, Int<1>{})));

// Block Copy（GMEM → SMEM）：整个 block 协作的数据搬运
auto tiled_copy_g2s = make_tiled_copy(
    Copy_Atom<AutoVectorizingCopy, half>{},
    make_layout(make_shape(Int<BM/4>{}, Int<BK>{}), GenRowMajor{}),
    make_layout(make_shape(Int<4>{}, Int<1>{}), GenRowMajor{}));

// Block Copy（SMEM → RF）：从 TiledMma 自动构建
auto tiled_copy_s2r_A = make_tiled_copy_A(
    Copy_Atom<SM80_16x8_LDSM_T, half>{}, tiled_mma);
```

> [!tip] Block MMA 和 Block Copy 的关系
> - **Block Copy** 负责把数据搬到正确的位置
> - **Block MMA** 负责在数据上执行计算
> - 两者共享同一个线程集合（整个 block 的所有线程）

## 3. Shared Memory 的布局

SMEM 的布局需要同时满足两个需求：

```
1. GMEM → SMEM 的拷贝要 coalesced（连续访问 GMEM）
2. SMEM → RF 的加载要无 bank conflict（ldmatrix 需要 swizzle 布局）
```

```cpp
// SMEM 布局：Swizzle 后的布局
auto smem_layout = composition(
    Swizzle<3, 3, 3>{},
    make_layout(make_shape(Int<BM>{}, Int<BK>{}), GenRowMajor{}));

__shared__ half smem[BM * BK];
auto sA = make_tensor(make_smem_ptr(smem), smem_layout);
```

> [!note] 同一块 SMEM，两种视图
> 写入时用 `TiledCopy_G2S` 的 partition 视图（保证 coalesced 写入）
> 读出时用 `TiledCopy_S2R` 的 partition 视图（保证无 bank conflict）
> 两者看到的是同一块物理 SMEM，只是"解释方式"不同

## 4. K 方向的循环

GEMM 需要沿 K 方向循环处理多个 tile：

```cpp
// K 方向的 tile 数量
int num_tiles = (K + BK - 1) / BK;

for (int k = 0; k < num_tiles; k++) {
  // 1. GMEM → SMEM
  auto gA_k = local_tile(gA, make_shape(BM, BK), make_coord(bx, k));
  copy(copy_g2s, gA_k, sA);
  __syncthreads();

  // 2. SMEM → RF
  copy(copy_s2r_A, sA, rA);
  copy(copy_s2r_B, sB, rB);

  // 3. MMA
  cute::gemm(tiled_mma, rA, rB, rC);

  __syncthreads();
}
```

> [!important] syncthreads 的位置
> - GMEM → SMEM 之后必须 syncthreads（确保所有线程都写完 SMEM）
> - MMA 之前不需要 syncthreads（每个线程读自己的 RF，不冲突）
> - 下一次迭代的 GMEM → SMEM 之前必须 syncthreads（确保所有线程都读完 SMEM）

## 5. 完整 GEMM Kernel 示例

```cpp
template <int BM, int BN, int BK>
__global__ void gemm_cute(const half* A, const half* B, float* C, int M, int N, int K) {
  int tx = threadIdx.x;
  int bx = blockIdx.x, by = blockIdx.y;

  // === 1. 创建 tensor ===
  auto gA = make_tensor(make_gmem_ptr(A), make_layout(make_shape(M, K), GenRowMajor{}));
  auto gB = make_tensor(make_gmem_ptr(B), make_layout(make_shape(N, K), GenRowMajor{}));

  // 切 tile
  auto tileA = local_tile(gA, make_shape(Int<BM>{}, Int<BK>{}), make_coord(bx, by));
  auto tileB = local_tile(gB, make_shape(Int<BN>{}, Int<BK>{}), make_coord(by, bx));

  // === 2. SMEM 布局（Swizzle） ===
  __shared__ half smemA[BM * BK], smemB[BN * BK];
  auto sA = make_tensor(make_smem_ptr(smemA),
      composition(Swizzle<3,3,3>{}, make_layout(make_shape(Int<BM>{}, Int<BK>{}), GenRowMajor{})));
  auto sB = make_tensor(make_smem_ptr(smemB),
      composition(Swizzle<3,3,3>{}, make_layout(make_shape(Int<BN>{}, Int<BK>{}), GenRowMajor{})));

  // === 3. TiledMma ===
  auto tiled_mma = make_tiled_mma(
      MMA_Atom<SM80_16x8x8_F32BF16BF16F32_TN>{},
      make_layout(make_shape(Int<4>{}, Int<2>{}, Int<1>{})));

  // === 4. 寄存器分配 ===
  auto rA = partition_fragment_A(tiled_mma, make_shape(Int<BM>{}, Int<BK>{}));
  auto rB = partition_fragment_B(tiled_mma, make_shape(Int<BN>{}, Int<BK>{}));
  auto rC = partition_fragment_C(tiled_mma, make_shape(Int<BM>{}, Int<BN>{}));
  clear(rC);

  // === 5. TiledCopy ===
  auto copy_g2s = make_tiled_copy(Copy_Atom<AutoVectorizingCopy, half>{}, ...);
  auto copy_s2r_A = make_tiled_copy_A(Copy_Atom<SM80_16x8_LDSM_T, half>{}, tiled_mma);
  auto copy_s2r_B = make_tiled_copy_B(Copy_Atom<SM80_16x8_LDSM_T, half>{}, tiled_mma);

  // === 6. K 方向循环 ===
  int num_tiles = (K + BK - 1) / BK;
  for (int k = 0; k < num_tiles; k++) {
    // GMEM → SMEM
    auto gA_k = local_tile(gA, make_shape(Int<BM>{}, Int<BK>{}), make_coord(bx, k));
    copy(copy_g2s.get_slice(tx).partition_S(gA_k),
         copy_g2s.get_slice(tx).partition_D(sA));
    __syncthreads();

    // SMEM → RF
    copy(copy_s2r_A.get_slice(tx).partition_S(sA),
         copy_s2r_A.get_slice(tx).partition_D(rA));
    copy(copy_s2r_B.get_slice(tx).partition_S(sB),
         copy_s2r_B.get_slice(tx).partition_D(rB));

    // MMA
    cute::gemm(tiled_mma, rA, rB, rC);
    __syncthreads();
  }

  // === 7. 写回结果 ===
  // ...
}
```

## 6. Grid 配置

```cpp
dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
dim3 block(size(tiled_mma));  // = 256（4×2×32 = 256 线程）
```

```
Grid: (M/BM) × (N/BN) 个 block
每个 block 计算 C 的一个 BM×BN tile
每个 block 内所有线程协作完成这个 tile 的计算
```

## 7. 性能优化要点

```
1. 向量化 GMEM 访问 → 用 float4/half4 加载
2. Swizzle SMEM 布局 → 消除 bank conflict
3. 寄存器预取 → 在计算当前 tile 的同时加载下一个 tile
4. 多 stage SMEM → 用双缓冲或三缓冲减少等待
5. Warp Specialization → 专门的 warp 做搬运，其他 warp 做计算
```

> [!tip] 从简单到优化的路线
> 1. 先写一个能跑通的 kernel（用 AutoVectorizingCopy）
> 2. 加 Swizzle 消除 bank conflict
> 3. 加寄存器预取（K 循环内重叠计算和搬运）
> 4. 加多 stage SMEM（双缓冲）
> 5. 用 TMA 替代手动 GMEM→SMEM 搬运
> 6. 用 Warp Specialization 分离搬运和计算
