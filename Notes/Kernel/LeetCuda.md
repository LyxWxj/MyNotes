# LeetCUDA 学习笔记

## 矩阵转置 (Matrix Transpose)

矩阵转置 `x[row][col] → y[col][row]`，本质是 `x[r][c]` 写到 `y[c][r]`。

核心矛盾：**读行写列 or 读列写行，总有一端的内存访问不连续（non-coalesced）**。所有优化都围绕解决这个矛盾展开。

---

### 1. Naive 实现：标量 1D

**思路**：把 `row×col` 个元素线性展开，每线程处理 1 个元素。

**Grid 配置**：

```
<<<ceil(row * col / 256), 256>>>
       ↑ block 数              ↑ 每 block 线程数

gridDim.x  = ceil(row * col / 256)
blockDim.x = 256
```

**坐标计算**：

```cpp
int global_idx = blockIdx.x * blockDim.x + threadIdx.x;  // 全局线程 ID [0, row*col)
int global_row = global_idx / col;  // 源矩阵行号
int global_col = global_idx % col;  // 源矩阵列号
```

**col2row — 按源矩阵列主序遍历**：

```cpp
__global__ void mat_transpose_f32_col2row_kernel(float* x, float* y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_row = global_idx / col;
  const int global_col = global_idx % col;
  if (global_idx < row * col)
    y[global_col * row + global_row] = x[global_idx];
    //  ↑ 写 y[col][row]，步长 = row，不连续
}
```

- **读 x**：`x[global_idx]` 连续读 → coalesced ✓
- **写 y**：`y[col * row + row]` 跨 `row` 步长跳 → non-coalesced ✗

**row2col — 按源矩阵行主序遍历**：

```cpp
__global__ void mat_transpose_f32_row2col_kernel(float* x, float* y,
                                                 const int row, const int col) {
  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_col = global_idx / row;
  const int global_row = global_idx % row;
  if (global_idx < row * col)
    y[global_idx] = x[global_row * col + global_col];
    //  ↑ 连续写 → coalesced ✓
}
```

- **读 x**：`x[row * col + col]` 跨 `col` 步长跳 → non-coalesced ✗
- **写 y**：`y[global_idx]` 连续写 → coalesced ✓

> **结论**：标量版本无法同时做到读写都 coalesced。

---

### 2. 2D Grid 版本

把 1D 线性展开换成 2D 网格，坐标计算更自然。

**Grid 配置**：

```
<<<dim3(col / 32, row / 32), dim3(32, 32)>>>
       ↑ gridDim.x  gridDim.y     ↑ blockDim.x blockDim.y

gridDim.x  = col / 32   (x 方向覆盖所有列)
gridDim.y  = row / 32   (y 方向覆盖所有行)
blockDim.x = 32         (每 block 32 列)
blockDim.y = 32         (每 block 32 行)
每 block: 32×32 = 1024 线程
```

**坐标计算**：

```cpp
int global_x = blockIdx.x * blockDim.x + threadIdx.x;  // 列号
int global_y = blockIdx.y * blockDim.y + threadIdx.y;  // 行号
```

**col2row2d**：

```cpp
__global__ void mat_transpose_f32_col2row2d_kernel(float* x, float* y,
                                                   const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;  // col
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;  // row
  if (global_x < col && global_y < row)
    y[global_x * row + global_y] = x[global_y * col + global_x];
    //  x[row][col] → y[col][row]
}
```

**row2col2d**：

```cpp
__global__ void mat_transpose_f32_row2col2d_kernel(float* x, float* y,
                                                   const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;  // row
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;  // col
  if (global_y < col && global_x < row)
    y[global_y * row + global_x] = x[global_x * col + global_y];
}
```

> 2D 版本和 1D 版本质相同，只是坐标计算更直观。读写仍然只有一端 coalesced。

---

### 3. Diagonal Scheduling（对角线调度）

**问题**：普通 2D 调度中，相邻 block 同时访问相邻内存区域 → L2 cache thrashing。

**解决**：用对角线偏移打乱 block 的 x 坐标。

```cpp
const int block_y = blockIdx.x;
const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
```

**调度对比**（假设 gridDim = 4×4）：

```
普通调度:                  对角线调度:
(0,0) (1,0) (2,0) (3,0)   (0,0) (1,0) (2,0) (3,0)
(0,1) (1,1) (2,1) (3,1)   (1,1) (2,1) (3,1) (0,1)
(0,2) (1,2) (2,2) (3,2)   (2,2) (3,2) (0,2) (1,2)
(0,3) (1,3) (2,3) (3,3)   (3,3) (0,3) (1,3) (2,3)
                           ↑ 每行右移一格，形成对角线
```

**效果**：相邻 block 访问的内存区域不再连续，L2 cache 竞争降低。

```cpp
__global__ void mat_transpose_f32_diagonal2d_kernel(float* x, float* y,
                                                    int row, int col) {
  const int block_y = blockIdx.x;
  const int block_x = (blockIdx.x + blockIdx.y) % gridDim.x;
  const int global_col = threadIdx.x + blockDim.x * block_x;
  const int global_row = threadIdx.y + blockDim.y * block_y;
  if (global_col < col && global_row < row)
    y[global_row * col + global_col] = x[global_col * row + global_row];
}
```

> 对角线调度只是打乱了 block 的访问顺序，不改变每个线程的计算逻辑。读写仍然只有一端 coalesced。

---

### 4. float4 向量化

**思路**：每线程处理 4 个连续元素（`float4` = 128-bit），用一条 `LD.E.128` / `ST.E.128` 指令完成。

**Grid 配置**（1D）：

```
标量:  <<<ceil(row * col / 256), 256>>>
float4: <<<ceil(row * col / (256 * 4)), 256>>>
                       ↑ 每线程 4 个元素，block 数 /4
```

**Grid 配置**（2D）：

```
标量:  <<<dim3(col/32, row/32), dim3(32, 32)>>>
float4: <<<dim3(col/(32*4), row/32), dim3(32, 32)>>>
               ↑ x 方向 /4        ↑ y 方向不变
           = <<<dim3(col/128, row/32), dim3(32, 32)>>>
```

**col2row float4**：

```cpp
__global__ void mat_transpose_f32x4_col2row_kernel(float* x, float* y,
                                                    const int row, const int col) {
  int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int global_col = (global_idx * 4) % col;  // 4 个元素的起始列
  int global_row = (global_idx * 4) / col;

  if (global_row < row && global_col + 3 < col) {
    float4 x_val = reinterpret_cast<float4*>(x)[global_idx];  // 128-bit 连续读 ✓

    y[global_col * row + global_row]         = x_val.x;  // 4 个散写 ✗
    y[(global_col + 1) * row + global_row]   = x_val.y;
    y[(global_col + 2) * row + global_row]   = x_val.z;
    y[(global_col + 3) * row + global_row]   = x_val.w;
  }
}
```

- **读**：`float4` 一次读 128-bit，coalesced ✓
- **写**：4 个元素分别写到 4 个不同列，散写 ✗

> 向量化优化了读端，但写端仍然散。反过来 `row2col` 则优化写端、读端散。

---

### 5. Shared Memory 中转 — 核心优化

**核心思想**：用 shared memory 做中转，把一次 " 散读/散写 " 拆成两次 " 连续访问 "。

```
x[HBM] --coalesced read--> tile[smem] --coalesced write--> y[HBM]
```

**Grid 配置**：

```
<<<dim3(col / (32*4), row / 32), dim3(32, 32)>>>
每 block: 32×32 = 1024 线程
每线程: 处理 4 个 float (float4)
每 block 覆盖: 32 行 × 128 列
tile 大小: tile[32][128] = 4KB shared memory
```

**完整流程**：

```
步骤 1: x → tile（coalesced 读 x，coalesced 写 tile）
┌─────────────────────────────────┐
│ x[row][col]                     │
│   ↓ float4 连续读               │
│ tile[32][128] (shared memory)   │
│   布局: tile[local_y][local_x*4]│
│   同一 warp 写同一行的不同列     │
└─────────────────────────────────┘

__syncthreads()

步骤 2: tile → y（coalesced 读 tile，coalesced 写 y）
┌─────────────────────────────────┐
│ tile[32][128]                   │
│   ↓ 按转置后的坐标读            │
│ y[col][row]                     │
│   ↓ float4 连续写               │
└─────────────────────────────────┘
```

**关键：STRIDE 的含义**

```cpp
constexpr int STRIDE = WARP_SIZE_S / 4;  // = 32 / 4 = 8
```

一个 warp 32 个线程（`local_y` 0~31），每线程处理 4 个元素。`STRIDE = 8` 把 32 个线程分成 8 组：

```
local_y:    0  1  2  3 │ 4  5  6  7 │ 8  9 10 11 │ ... │ 28 29 30 31
local_y/STRIDE:  0     │     1      │     2      │     │      7
local_y%STRIDE:  0 1 2 3│  0 1 2 3  │  0 1 2 3  │     │  0  1  2  3
```

读 tile 时的坐标变换：

```cpp
// 线程 ly 读 tile 的第 (ly%8)*4 行，列号 = lx*4 + ly/8
smem_val.x = tile[(ly % STRIDE) * 4    ][lx * 4 + ly / STRIDE];
smem_val.y = tile[(ly % STRIDE) * 4 + 1][lx * 4 + ly / STRIDE];
smem_val.z = tile[(ly % STRIDE) * 4 + 2][lx * 4 + ly / STRIDE];
smem_val.w = tile[(ly % STRIDE) * 4 + 3][lx * 4 + ly / STRIDE];
```

**图解**（STRIDE=8，一个 warp 的 32 个线程读 tile）：

```
tile 行:    0  1  2  3  4  5  6  7  ... 28 29 30 31
            ├──┤  ├──┤  ├──┤  ├──┤     ├──┤  ├──┤
线程 0~3:   读行 0~3 (组0)              读行 28~31 (组7)
线程 4~7:   读行 0~3 (组0)              读行 28~31 (组7)
...
线程 28~31: 读行 0~3 (组0)              读行 28~31 (组7)

每 4 个连续线程读同一组的 4 行，列号由 lx*4 + 组内编号 决定
→ 4 个连续线程读同一列的不同行 → 可以合并成一个 float4 读
```

```cpp
__global__ void mat_transpose_f32x4_shared_col2row2d_kernel(float* x, float* y,
                                                             const int row, const int col) {
  const int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  const int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  const int local_x = threadIdx.x;
  const int local_y = threadIdx.y;
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4];

  if (global_x * 4 + 3 < col + 3 && global_y < row) {
    // 步骤 1: x → tile (coalesced)
    float4 x_val = reinterpret_cast<float4*>(x)[global_y * col / 4 + global_x];
    FLOAT4(tile[local_y][local_x * 4]) = FLOAT4(x_val);
    __syncthreads();

    // 步骤 2: tile → y (coalesced)
    constexpr int STRIDE = WARP_SIZE_S / 4;
    float4 smem_val;
    smem_val.x = tile[(local_y % STRIDE) * 4    ][local_x * 4 + local_y / STRIDE];
    smem_val.y = tile[(local_y % STRIDE) * 4 + 1][local_x * 4 + local_y / STRIDE];
    smem_val.z = tile[(local_y % STRIDE) * 4 + 2][local_x * 4 + local_y / STRIDE];
    smem_val.w = tile[(local_y % STRIDE) * 4 + 3][local_x * 4 + local_y / STRIDE];

    const int bid_y = blockIdx.y * blockDim.y;
    const int out_y = global_x * 4 + local_y / STRIDE;
    const int out_x = (local_y % STRIDE) * 4 + bid_y;
    reinterpret_cast<float4*>(y)[(out_y * row + out_x) / 4] = FLOAT4(smem_val);
  }
}
```

---

### 6. Bank Conflict Free (BCF) 版本

**问题**：shared memory 按 4 字节分 32 个 bank。如果 tile 每行 128 个 float = 512 字节，512 / 4 = 128 个 bank 访问，128 % 32 = 0 → **所有行的同一列映射到同一个 bank** → 32-way bank conflict。

```
tile[0][0] 和 tile[1][0] 和 ... tile[31][0] 都在 bank 0 → 冲突！
```

**解决**：加 `PAD = 1`，每行多 1 个 float，错开 bank 映射。

```cpp
__shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];
//                                       ↑ 128 + 1 = 129 个 float
// 129 % 32 = 1 → 每行的同一列错开 1 个 bank → 无冲突
```

```
无 PAD:                      有 PAD (+1):
tile[0][c] → bank (c % 32)  tile[0][c] → bank (c % 32)
tile[1][c] → bank (c % 32)  tile[1][c] → bank ((c+1) % 32)  ← 错开
tile[2][c] → bank (c % 32)  tile[2][c] → bank ((c+2) % 32)  ← 错开
...全部冲突!                  ...无冲突!
```

```cpp
__global__ void mat_transpose_f32x4_shared_bcf_col2row2d_kernel(float* x, float* y,
                                                                 const int row, const int col) {
  // 和 shared 版本完全一样，唯一区别:
  __shared__ float tile[WARP_SIZE_S][WARP_SIZE_S * 4 + PAD];  // +PAD
  // ... 其余代码相同 ...
}
```

---

### 优化路线总结

```
标量 1D            → 一端 coalesced，一端散
标量 2D            → 同上，坐标更直观
Diagonal 2D        → 打乱 block 调度，减少 L2 thrashing
float4 向量化      → 读端或写端优化为 128-bit
Shared Memory      → smem 中转，两端都 coalesced
Shared + BCF       → padding 消除 bank conflict
Shared + Merge     → 写端也合并成 float4（128-bit write）
```

**一句话：转置的矛盾是 " 读行写列 "，shared memory 的作用是把一次不连续拆成两次连续。**

## All Reduce

首先是 warp 内规约的函数：

```cpp
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}
```

```cpp
__shfl_xor_sync(0xffffffff, val, mask)
//              ↑ 参与的线程掩码（全 1 = 32 个线程全参与）
//                           ↑ 要交换的值
//                                ↑ 偏移量
```

**行为**：线程 `i` 读取线程 `i XOR mask` 的 `val`。

```
例: mask = 1
  线程 0 读线程 (0^1)=1 的值
  线程 1 读线程 (1^1)=0 的值
  线程 2 读线程 (2^1)=3 的值
  线程 3 读线程 (3^1)=2 的值
  → 相邻线程交换数据

例: mask = 2
  线程 0 读线程 (0^2)=2 的值
  线程 2 读线程 (2^2)=0 的值
  → 间隔 2 的线程交换数据
```

## 循环展开过程

```cpp
for (int mask = 16; mask >= 1; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask);
}
```

以 32 个线程为例（`kWarpSize=32`）：

```
初始:  T0=a0  T1=a1  T2=a2  T3=a3  T4=a4  T5=a5  T6=a6  T7=a7 ... T31=a31
━━━ mask = 16 (32>>1) ━━━
T0 读 T16: T0 = a0+a16    T4 读 T20: T4 = a4+a20
T1 读 T17: T1 = a1+a17    T5 读 T21: T5 = a5+a21
T2 读 T18: T2 = a2+a18    T6 读 T22: T6 = a6+a22
T3 读 T19: T3 = a3+a19    T7 读 T23: T7 = a7+a23
...
T16读 T0 : T16 = a0 + a16
T17读 T1 : T17 = a1 + a17
...
━━━ mask = 8 (16>>1) ━━━
T0 读 T8: T0 = a0+a16+a8+a24    T4 读 T12: T4 = a4+a20+a12+a28
T1 读 T9: T1 = a1+a17+a9+a25    T5 读 T13: T5 = a5+a21+a13+a29
T2 读 T10: T2 = a2+a18+a10+a26  T6 读 T14: T6 = a6+a22+a14+a30
T3 读 T11: T3 = a3+a19+a11+a27  T7 读 T15: T7 = a7+a23+a15+a31
...
━━━ mask = 4 (8>>1) ━━━
T0 读 T4: T0 = a0+a16+a8+a24+a4+a20+a12+a28 
             = a0+a4+a8+a12+a16+a20+a24+a28   
T1 读 T5: T1 = a1+a5+a9+a13+a17+a21+a25+a29
T2 读 T6: T2 = a2+a18+a10+a26+a6+a22+a14+a30
             = a2+a6+a10+a14+a18+a22+a26+a30
T3 = a3 + a7 + .. + a31
T4 读 T8: T4 = a4+a20+a12+a28+a8+a24+a16+a0
			 = a0+a4+a8+a12+a16+a20+a24+a28 = T1
...
━━━ mask = 2 ━━━
T0 读 T2: T0 = a0+a2+a4+...+a30
T1 读 T3: T1 = a1+a3+a5+...+a31
━━━ mask = 1 ━━━
T0 读 T1: T0 = a0+a1+..+a31  ← 最终结果
```

## 图解：Butterfly 拓扑

```
T0 ──┬── a0+a4 ──┬── a0+a4+a2+a6 ──┬── 全部之和
     │           │                 │
T2 ──┼── a2+a6 ──┘                 │
     │                             │
T1 ──┬── a1+a5 ──┬── a1+a5+a3+a7 ──┘
     │           │
T3 ──┼── a3+a7 ──┘

每一跳: mask 减半 → 交换距离加倍
log2(32) = 5 轮 → 32 个线程的值全部归约到 lane 0
```

## 为什么用 shuffle 而不是 smem？

```
Shared Memory 方案:
  32 个线程写 smem → __syncwarp → 1 个线程读 32 次
  = 32 次 smem 写 + 32 次 smem 读 = 64 次 smem 访问

Shuffle 方案:
  5 轮 × 每轮 32 个线程各读 1 次 = 160 次寄存器间传输
  但 shuffle 是寄存器→寄存器，延迟 ~1 cycle
  smem 延迟 ~20-30 cycles

Shuffle 更快: 零 smem 开销，纯寄存器操作
```

> [!tip] `0xffffffff` 的含义 这是 32-bit 掩码，每一位对应一个 lane。全 1 表示 32 个 lane 全参与。如果某些 lane 不活跃（比如 warp 内只有部分线程有效），需要动态获取活跃掩码：
>
> ```cpp
> __activemask()  // 获取当前活跃的 lane 掩码
> ```
>
> 但 reduce 场景通常所有 lane 都活跃，直接用 `0xffffffff` 即可。

Naive 实现：

```cpp
// lauch: <<<(N/256,1,1),(256,1,1)>>>
template<const int NUM_THREADS = 256>
__global__ void block_all_reduce_sum_f32_f32_kernel(float* a, float* y, const int N) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;
  int idx = bid * NUM_THREADS + tid;
  constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
  __shared__ float reduce_smem[NUM_WARPS];
  float sum = (idx < N) ? a[idx] : 0.f;
  int warp = tid / WARP_SIZE;
  int lane = tid % WARP_SIZE;
  sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
  if (lane == 0)
    reduce_smem[warp] = sum;
  __syncthreads();
  sum = (lane < NUM_WARPS) ? reduce_smem[lane] : 0.f;
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
}
```

计算基础索引

```cpp
int tid = threadIdx.x;                    // block 内线程 ID [0, 255]
int bid = blockIdx.x;                     // block ID
int idx = bid * NUM_THREADS + tid;        // 全局索引
constexpr int NUM_WARPS = 256 / 32 = 8;  // block 内有 8 个 warp
__shared__ float reduce_smem[8];          // shared memory: 每个 warp 存一个部分和
```

```cpp
float sum = (idx < N) ? a[idx] : 0.f;  // 每个线程加载 1 个元素，越界填 0
```

Warp 内 Reduce

```cpp
int warp = tid / 32;  // 当前线程属于第几个 warp
int lane = tid % 32;  // warp 内的 lane 编号
sum = warp_reduce_sum_f32<32>(sum);  // warp 内 32 个线程的 sum 全部加起来 
```

`warp_reduce_sum_f32` 的蝶形规约算法上面已经解释过了。

然后每个 Warp 内的 0 号线程（lane=0）把结果写入共享内存。

```cpp
if (lane == 0)
    reduce_smem[warp] = sum;  // reduce_smem[0~7] = 8 个 warp 的部分和
__syncthreads();  // 等所有线程写完
```

warp 内的前 NUM_WARPS 个 lane 读取每个 Warp 内的和，超出的填 0,这样现在 Warp 内的和就是整个 block 的和

```cpp
sum = (lane < 8) ? reduce_smem[lane] : 0.f;
```

0 号 warp 再规约一次，该 block 的 0 号线程返回答案

```cpp
  if (warp == 0)
    sum = warp_reduce_sum_f32<NUM_WARPS>(sum);
  if (tid == 0) atomicAdd(y, sum);
```

