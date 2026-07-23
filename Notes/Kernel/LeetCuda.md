# LeetCUDA 学习笔记

## 前置知识：Grid、Block、Warp、Thread 的关系

### 层级结构

```
Grid（整个 kernel 启动）
├── Block (0,0)        Block (1,0)        Block (2,0)  ...
│   ├── Warp 0 (线程 0~31)
│   ├── Warp 1 (线程 32~63)
│   ├── Warp 2 (线程 64~95)
│   └── ...
├── Block (0,1)        Block (1,1)        ...
└── ...
```

| 概念 | 包含 | 共享内存？ | 可同步？ | 最大数量 |
|---|---|---|---|---|
| **Grid** | 多个 Block | ✗ block 间不共享 | ✗ 不能跨 block 同步 | 2³¹-1 个 block |
| **Block** | 多个 Warp | ✓ shared memory | ✓ `__syncthreads()` | 1024 线程/block |
| **Warp** | 32 个 Thread | ✓ 隐式（同周期执行） | ✓ 隐式（lockstep） | 32 线程/warp |
| **Thread** | 1 个执行单元 | 寄存器 + 局部变量 | — | — |

### Grid 和 Block 都可以是 1D、2D 或 3D

```cpp
// 1D：<<<gridDim.x, blockDim.x>>>
kernel<<<256, 256>>>();  // 256 个 block，每 block 256 线程

// 2D：<<<(gridDim.x, gridDim.y), (blockDim.x, blockDim.y)>>>
kernel<<<dim3(16, 16), dim3(32, 32)>>>();  // 16×16 个 block，每 block 32×32 线程

// 3D：<<<(gx, gy, gz), (bx, by, bz)>>>
kernel<<<dim3(8, 8, 8), dim3(4, 4, 4)>>>();  // 8×8×8 个 block，每 block 4×4×4 线程
```

### 线程索引计算

```cpp
// 全局线程 ID（1D）
int idx = blockIdx.x * blockDim.x + threadIdx.x;

// 全局线程 ID（2D）
int gx = blockIdx.x * blockDim.x + threadIdx.x;  // 列
int gy = blockIdx.y * blockDim.y + threadIdx.y;  // 行

// Warp 内的 lane 编号
int lane = threadIdx.x % 32;  // 当 blockDim.x ≥ 32 时，lane = threadIdx.x

// Warp 编号（block 内）
int warp = threadIdx.x / 32;
```

### Warp：硬件调度的基本单位

```
一个 Warp = 32 个线程，SIMT（单指令多线程）执行

所有 32 个线程在同一条指令上同步执行：
  ┌────────────────────────────────────────┐
  │ T0: add r1, r2, r3                     │
  │ T1: add r1, r2, r3     ← 同一条指令    │
  │ T2: add r1, r2, r3                     │
  │ ...                                    │
  │ T31: add r1, r2, r3                    │
  └────────────────────────────────────────┘
  → 1 个周期发射 1 条指令，32 个线程同时执行
```

> [!important] Warp Divergence（分支分歧）
> 如果 warp 内的线程走了不同的分支：
> ```cpp
> if (lane < 16)  // T0~T15 走 if，T16~T31 走 else
>     do_A();
> else
>     do_B();
> ```
> GPU 会 **串行执行两条路径**：先执行 T0~T15 的 `do_A()`（T16~31 空转），再执行 T16~T31 的 `do_B()`（T0~15 空转）。性能减半。

### SM（Streaming Multiprocessor）与 Occupancy

```
GPU 芯片
├── SM 0
│   ├── Warp Scheduler 0 ──→ 从 active warps 池中选一个 warp 发射指令
│   ├── Warp Scheduler 1
│   ├── CUDA Cores (FP32/INT32)
│   ├── Special Function Units (sin/cos/exp)
│   ├── Register File（所有 active warp 共享）
│   └── Shared Memory / L1 Cache
├── SM 1
└── ...
```

**Occupancy = SM 上 active warp 数 / SM 最大 warp 数**

影响 occupancy 的因素：

| 因素 | 影响 |
|---|---|
| 每线程寄存器数 | 用越多 → SM 能容纳的 warp 越少 |
| Shared Memory 用量 | 用越多 → SM 能容纳的 block 越少 |
| Block 大小 | 太小（<64 线程）浪费 SM 资源；太大（1024）限制并发 block 数 |

> [!tip] Occupancy 不是越高越好
> Occupancy 低不代表性能差。计算密集型 kernel 只需 2~3 个 warp 就能隐藏计算延迟。
> 但 occupancy 太低（<25%）通常意味着 latency hiding 不足。

### Latency Hiding（延迟隐藏）

GPU 的核心优化策略：当一个 warp 等待内存时，调度器切换到另一个 ready warp 执行。

```
只有 1 个 warp：
  warp 0: [加载] → [等待内存 400 cycles] → [计算] → [等待内存] → [计算]
                    ↑ SM 空转 400 周期

4 个 warp：
  warp 0: [加载] → [等待] → [计算] → [等待] → [计算]
  warp 1:          [加载] → [等待] → [计算] → [等待]
  warp 2:                   [加载] → [等待] → [计算]
  warp 3:                            [加载] → [等待]
  → SM 始终有 warp 在执行，内存延迟被"隐藏"
```

> [!note] 延迟类型与所需 warp 数
> - **计算延迟** ~4-8 cycles → 2~3 个 warp 即可隐藏
> - **内存延迟** ~200-400 cycles → 需要 10+ 个 active warp 才能完全隐藏
> - **公式**：所需 warp 数 ≥ `latency × issue_rate / throughput`

### 常见 Block 大小选择

| Block 大小 | 适用场景 |
|---|---|
| 32 | 极简单 kernel（每线程工作量大，如 SGEMV 每 warp 一行） |
| 128 | 通用，occupancy 好 |
| 256 | 最常用，平衡 occupancy 和每线程寄存器数 |
| 512~1024 | 需要 block 内大量 reduce 时（如 softmax 每行归约） |

> [!tip] 实用规则
> 1. Block 大小通常是 32 的倍数（warp 对齐）
> 2. 256 是安全的默认值
> 3. 用 `__launch_bounds__(256, 4)` 告诉编译器"最多 256 线程/block，至少 4 个 block/SM"，编译器会优化寄存器分配

---

## 向量化加载：x4、x2、x8、x8_pack 是什么？

CUDA kernel 名字里的 `f32x4`、`f16x2`、`f16x8_pack` 表示 **每线程处理多少个元素**，核心目的是最大化内存带宽。

### 为什么需要向量化？

```
标量加载 (f32):
  每线程 1 条 LD.32 指令 → 4 bytes
  32 线程 × 4B = 128B = 1 个 cache line ✓

标量加载 (f16):
  每线程 1 条 LD.32 指令 → 2 bytes (浪费一半)
  32 线程 × 2B = 64B = 半个 cache line ✗

f16x2 加载:
  每线程 1 条 LD.32 指令 → 4 bytes (2 个 half)
  32 线程 × 4B = 128B = 1 个 cache line ✓

f16x8_pack 加载:
  每线程 1 条 LD.128 指令 → 16 bytes (8 个 half)
  32 线程 × 16B = 512B = 4 个 cache lines ✓
```

### 各种向量类型的含义

| 名称 | 类型 | 每线程字节 | 加载指令 | 32 线程总访问 |
|---|---|---|---|---|
| f32 | `float` | 4B | `LD.32` | 128B |
| f32x4 | `float4` | 16B | `LD.128` | 512B |
| f16 | `half` | 2B | `LD.32`（浪费） | 64B |
| f16x2 | `half2` | 4B | `LD.32` | 128B |
| f16x8 | `half2×4` | 16B | 4×`LD.32` | 512B |
| f16x8_pack | `half[8]` | 16B | `LD.128` | 512B |

### f16x8 vs f16x8_pack 的区别

```
f16x8:
  half2 r0 = HALF2(x[idx+0]);  // 2 个 half，LD.32
  half2 r1 = HALF2(x[idx+2]);  // 2 个 half，LD.32
  half2 r2 = HALF2(x[idx+4]);  // 2 个 half，LD.32
  half2 r3 = HALF2(x[idx+6]);  // 2 个 half，LD.32
  → 4 条独立的 32-bit 加载指令

f16x8_pack:
  half pack[8];
  LDST128BITS(pack[0]) = LDST128BITS(x[idx]);  // 1 条 128-bit 加载
  → 1 条 LD.E.128 指令，把 16 字节一次搬进寄存器
```

> [!tip] pack 版本更快
> `f16x8` 用 4 条 32-bit 指令，`f16x8_pack` 用 1 条 128-bit 指令。
> 同样搬运 16 字节，指令数少 4 倍 → 指令调度开销更低。
> 但要求地址 **16 字节对齐**，否则会段错误。

### 向量化不影响计算语义

```cpp
// 标量：每线程算 1 个元素
float val = x[idx];
float result = op(val);
y[idx] = result;

// f32x4：每线程算 4 个元素，但每个元素独立处理
float4 v = FLOAT4(x[idx]);
float4 r;
r.x = op(v.x);
r.y = op(v.y);
r.z = op(v.z);
r.w = op(v.w);
FLOAT4(y[idx]) = r;
```

> [!note] Grid 配置要跟着变
> 如果每线程处理 4 个元素，grid 大小要除以 4：
> ```
> 标量:  <<<ceil(N/256), 256>>>
> f32x4: <<<ceil(N/(256*4)), 256>>>  → 每线程处理 4 个，block 数少 4 倍
> ```

---

## 矩阵转置 (Matrix Transpose)

矩阵转置 `x[row][col] → y[col][row]`，本质是 `x[r][c]` 写到 `y[c][r]`。

核心矛盾：**读行写列 or 读列写行，总有一端的内存访问不连续（non-coalesced）**。所有优化都围绕解决这个矛盾展开。

---

### 1. Naive 实现：标量 1D

**思路**：把 `row×col` 个元素线性展开，每线程处理 1 个元素。

**Grid 配置**：

```
<<<ceil(row * col / 256), 256>>>
         ↑ block 数         ↑ 每 block 线程数

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

---

## Softmax

### 数学公式

```
softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
```

对每一行（一个 token 的所有维度）独立做归一化：

```
输入 x: (seq_len, head_dim)
输出 y: (seq_len, head_dim)

block 0 → softmax(x[0][0..head_dim-1]) → y[0]
block 1 → softmax(x[1][0..head_dim-1]) → y[1]
...
```

### Warp Reduce / Block Reduce / All Reduce 的关系

```
┌─────────────────────────────────────────────────────┐
│  All Reduce（跨 block）                              │
│  ┌───────────────┐ ┌───────────────┐                │
│  │ Block 0       │ │ Block 1       │  ...           │
│  │ ┌────┐ ┌────┐ │ │ ┌────┐ ┌────┐ │                │
│  │ │Warp│ │Warp│ │ │ │Warp│ │Warp│ │                │
│  │ │ 0  │ │ 1  │ │ │ │ 0  │ │ 1  │ │                │
│  │ └────┘ └────┘ │ │ └────┘ └────┘ │                │
│  │   Block Reduce│ │   Block Reduce│                │
│  └───────┬───────┘ └───────┬───────┘                │
│          └──── atomicAdd / 第二次 kernel ────→ 最终结果│
└─────────────────────────────────────────────────────┘
```

| 级别 | 范围 | 通信方式 | 延迟 |
|---|---|---|---|
| **Warp Reduce** | 32 线程（同一 warp） | `__shfl_xor_sync`（寄存器） | ~5 cycles |
| **Block Reduce** | 1 block 内所有线程 | warp reduce + smem + warp 0 reduce | ~50 cycles |
| **All Reduce** | 所有 block | block reduce + atomicAdd / 第二次 kernel | ~100+ cycles |

**Warp Reduce** — 32 个线程，纯寄存器 butterfly：

```cpp
for (int mask = 16; mask >= 1; mask >>= 1)
    val += __shfl_xor_sync(0xffffffff, val, mask);
// 5 轮 → lane 0 得到 32 个线程的总和
```

**Block Reduce** — warp reduce + smem + warp 0 二次 reduce：

```cpp
float warp_sum = warp_reduce_sum(val);       // warp 内 reduce
if (lane == 0) smem[warp] = warp_sum;        // 写 smem
__syncthreads();
val = smem[lane];                             // warp 0 读所有部分和
if (warp == 0) final = warp_reduce_sum(val);  // warp 0 二次 reduce
final = __shfl_sync(0xffffffff, final, 0);    // broadcast 给所有线程
```

> [!tip] 选择原则
> - 数据量 ≤ 32 → 只用 Warp Reduce
> - 32 < 数据量 ≤ 1024 → Block Reduce（softmax 每行归约）
> - 数据量 > 1024 → All Reduce（block reduce + atomicAdd）

### Softmax Per Token Kernel

```cpp
// launch: <<<(N/K, 1, 1), (K, 1, 1)>>>
template <const int NUM_THREADS = 256>
__global__ void softmax_f32_per_token_kernel(float* x, float* y, int N) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float exp_val = (idx < N) ? expf(x[idx]) : 0.f;     // 1. 每线程算 exp
  float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val); // 2. block reduce 求和
  if (idx < N) y[idx] = exp_val / exp_sum;             // 3. 归一化
}
```

### 数值稳定性：Safe Softmax

朴素实现在 `x[i]` 很大时 `exp(x)` 溢出 → inf。

```
safe softmax: exp(x_i - x_max) / Σ exp(x_j - x_max)
```

需要先求 `x_max`（一次 block reduce max），再求 `sum(exp(x - x_max))`（一次 block reduce sum），两遍扫描。

### Online Softmax（一遍扫描）

维护 `(max, denominator)` 对，边扫描边合并：

```
新来元素 x:
  new_max = max(old_max, x)
  new_denom = old_denom * exp(old_max - new_max) + exp(x - new_max)
                     ↑ 旧 sum 乘修正因子              ↑ 新元素贡献
```

用 `MD` 结构体在 warp 内合并：

```cpp
struct MD { float m; float d; };  // max, denominator

// 合并两个 (m, d) 对
greater = m 较大的那个
smaller = m 较小的那个
merged.d = greater.d + smaller.d * exp(smaller.m - greater.m)
merged.m = greater.m
```

> [!important] Online Softmax 是 FlashAttention 的基础
> FlashAttention 中每个线程处理一小块 Q@K^T，需要在线合并 max 和 sum。
> 用 MD reduce 只需一遍扫描，不需要先算全局 max 再算 sum（两遍）。

---

## RoPE (Rotary Position Embedding)

### 核心思想

对 Q 和 K 的每一对相邻维度 `(x1, x2)` 做 **旋转**，旋转角度取决于 token 的位置 `pos` 和维度索引 `i`：

```
RoPE(x, pos) = R(pos, θ_i) · x

其中 R 是 2D 旋转矩阵：
┌ out1 ┐   ┌ cos(θ)  -sin(θ) ┐ ┌ x1 ┐
│      │ = │                  │ │    │
└ out2 ┘   └ sin(θ)   cos(θ) ┘ └ x2 ┘

即：
  out1 = x1 * cos(θ) - x2 * sin(θ)
  out2 = x1 * sin(θ) + x2 * cos(θ)
```

角度 `θ` 的计算：

```
θ = pos * freq_i
freq_i = 1 / (base^(2i / d))

其中：
  pos    = token 在序列中的位置 (0, 1, 2, ...)
  i      = 维度索引 (0, 1, ..., d/2-1)
  base   = 10000 (超参数)
  d      = head dimension
```

> [!tip] 直觉理解
> 每个维度对 `(x1, x2)` 像一个 2D 平面上的点，RoPE 把它绕原点旋转一个角度。
> 不同位置的 token 旋转不同角度，所以两个 token 的点积会反映它们的相对距离。

### 为什么用旋转而不是加法？

传统位置编码（如 sinusoidal PE）是 **加法**：`Q' = Q + PE(pos)`。

RoPE 是 **乘法**（旋转）：`Q' = RoPE(Q, pos)`。

优势：两个 token 的点积自然包含 **相对位置** 信息：

```
dot(RoPE(Q, pos_q), RoPE(K, pos_k))
  = dot(Q, K) 的函数，只依赖于 (pos_q - pos_k)
```

> [!important] RoPE 是 LLaMA / GPT-NeoX / Mistral 等模型的标配
> 几乎所有现代 LLM 都用 RoPE 替代了传统加法位置编码。

### Kernel 实现

#### v1: 扁平索引（每线程处理一对元素）

```
launch: <<<(N/256, 1, 1), (256, 1, 1)>>>  N = seq_len * head_dim/2
```

```cpp
__global__ void rope_f32_kernel(float* x, float* out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // 每线程处理一对 (x1, x2)
  float x1 = x[idx * 2];
  float x2 = x[idx * 2 + 1];

  int token_pos = idx / N;    // token 位置
  int token_idx = idx % N;    // 维度索引 (0 ~ N-1)

  // freq_i = 1 / (10000^(2i / 2N))
  float exp_v = 1.0f / powf(10000.0f, 2 * token_idx / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);

  // 旋转
  out[idx * 2]     = x1 * cos_v - x2 * sin_v;
  out[idx * 2 + 1] = x1 * sin_v + x2 * cos_v;
}
```

内存布局：`x[seq_len][N * 2]`，每个 token 有 `N` 对元素。

```
idx:  0    1    2    3    4    5    6    7
x:   x0   x1   x2   x3   x4   x5   x6   x7
      ─pair─    ─pair─    ─pair─    ─pair─
pos:  0     0     1     1     2     2     3     3
dim:  0     0     0     0     1     1     1     1
```

#### v2: block per token（更自然的索引）

```
launch: <<<(seq_len, 1, 1), (N=head_dim/2, 1, 1)>>>
```

```cpp
__global__ void rope_f32_v2_kernel(float* x, float* out, int seq_len, int N) { // N = head_dim / 2
  int token_pos = blockIdx.x;   // 每个 block 处理一个 token
  int tid = threadIdx.x;        // 每个线程处理该 token 的一对元素

  float x1 = x[token_pos * N * 2 + tid * 2];
  float x2 = x[token_pos * N * 2 + tid * 2 + 1];

  float exp_v = 1.0f / powf(10000.0f, 2 * tid / (N * 2.0f));
  float sin_v = sinf(token_pos * exp_v);
  float cos_v = cosf(token_pos * exp_v);

  out[token_pos * N * 2 + tid * 2]     = x1 * cos_v - x2 * sin_v;
  out[token_pos * N * 2 + tid * 2 + 1] = x1 * sin_v + x2 * cos_v;
}
```

Grid: `<<<seq_len, N>>>` — 每个 block 处理一个 token 的 `N` 对元素。

> [!note] v1 vs v2
> - v1: 用 `idx / N` 算 token 位置 → 需要除法（慢）
> - v2: `blockIdx.x` 直接就是 token 位置 → 无需除法，coalesced 更好

#### v3: float4 向量化（每线程处理 4 个元素 = 2 对）

```
launch: <<<(N/256, 1, 1), (256, 1, 1)>>>  N = seq_len * head_dim/8
```

```cpp
__global__ void rope_f32x4_pack_kernel(float* x, float* out, int seq_len, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  float4 x_v = FLOAT4(x[idx * 4]);  // 128-bit 加载

  int token_pos = idx / N;
  int token_idx = idx % N;

  // 两对元素需要不同的 freq（相邻维度的 freq 不同）
  float exp_f_v = 1.0f / powf(10000.0f, 2 * token_idx * 2 / (N * 4.0f));
  float exp_s_v = 1.0f / powf(10000.0f, 2 * (token_idx * 2 + 1) / (N * 4.0f));

  float sin_f_v = sinf(token_pos * exp_f_v);
  float cos_f_v = cosf(token_pos * exp_f_v);
  float sin_s_v = sinf(token_pos * exp_s_v);
  float cos_s_v = cosf(token_pos * exp_s_v);

  // 第一对 (x, y)
  float4 out_v;
  out_v.x = x_v.x * cos_f_v - x_v.y * sin_f_v;
  out_v.y = x_v.x * sin_f_v + x_v.y * cos_f_v;
  // 第二对 (z, w)
  out_v.z = x_v.z * cos_s_v - x_v.w * sin_s_v;
  out_v.w = x_v.z * sin_s_v + x_v.w * cos_s_v;

  FLOAT4(out[idx * 4]) = out_v;  // 128-bit 存储
}
```

> [!tip] float4 版本的要点
> 每线程处理 4 个元素 = 2 对 `(x1,x2)` 和 `(x3,x4)`。
> 两对的 freq 不同（维度索引不同），所以需要分别计算 sin/cos。
> 用 `FLOAT4` 一次读 128-bit，一次写 128-bit → 最大化带宽。

### RoPE 的计算复杂度

```
每元素: 1 次 sin + 1 次 cos + 2 次乘 + 1 次加 = 5 FLOPs
每 token: d * 5 FLOPs (d = head dimension)

对比 attention 的 FLOPs: O(seq_len * d²) → RoPE 的开销可以忽略
```

> [!note] RoPE 不是性能瓶颈
> RoPE 的计算量远小于 attention 和 FFN，通常不需要特别优化。
> 实际部署中更关心的是 RoPE 的 **缓存**（KV cache 中的 cos/sin 值是否需要重算）。

---

## Layer Normalization

### 数学公式

```
LayerNorm(x) = (x - mean) / sqrt(var + eps) * g + b

其中（对每一行独立计算）：
  mean = sum(x) / K
  var  = sum((x - mean)^2) / K
  g, b = 可学习的 scale 和 bias
  eps  = 1e-5（防止除零）
```

### 输入输出

```
x: (N, K)  — N = batch_size × seq_len, K = hidden_size
y: (N, K)
g: scalar 或 (K,) — scale
b: scalar 或 (K,) — bias

每个 block 处理一行（一个 token 的 K 个维度）
```

### Kernel 流程

```
launch: f32      → <<<(N, 1, 1), (K, 1, 1)>>>
        f32x4    → <<<(N, 1, 1), (K/4, 1, 1)>>>
        f16      → <<<(N, 1, 1), (K, 1, 1)>>>
        f16x8    → <<<(N, 1, 1), (K/8, 1, 1)>>>
```

```cpp
// 每个 block 处理一行，每线程处理一个元素
float value = x[idx];

// Step 1: block reduce 求 mean
float sum = block_reduce_sum(value);
s_mean = sum / K;

// Step 2: block reduce 求 variance
float variance = (value - s_mean) * (value - s_mean);
variance = block_reduce_sum(variance);
s_variance = rsqrtf(variance / K + eps);

// Step 3: 归一化
y[idx] = (value - s_mean) * s_variance * g + b;
```

> [!important] 两次 block reduce
> Layer Norm 需要两次 block reduce：一次求 mean，一次求 variance。
> 两次之间需要 `__syncthreads()` 确保 mean 已写入 smem。

### Layer Norm vs RMS Norm

| | Layer Norm | RMS Norm |
|---|---|---|
| 公式 | (x - mean) / sqrt(var + eps) | x / sqrt(mean(x²) + eps) |
| 减均值 | ✓ 需要 | ✗ 不需要 |
| bias | ✓ 有 g 和 b | 只有 g |
| reduce 次数 | 2 次（mean + variance） | 1 次（sum of squares） |
| 用于 | BERT, GPT-2 | LLaMA, Mistral |

> [!tip] RMS Norm 更简单
> RMS Norm 不需要减均值，只需要一次 block reduce（求 `sum(x²)`）。
> 计算量约为 Layer Norm 的一半，是现代 LLM 的首选。

### 向量化版本

```cpp
// F32x4: 每线程加载 4 个 float，先在寄存器内求和
float4 reg_x = FLOAT4(x[idx]);
float value = reg_x.x + reg_x.y + reg_x.z + reg_x.w;
float sum = block_reduce_sum(value);  // 每线程贡献 4 个元素的和
s_mean = sum / K;

// 归一化时对 4 个元素分别处理
float4 reg_y;
reg_y.x = (reg_x.x - s_mean) * s_variance * g + b;
// ... y, z, w 同理
FLOAT4(y[idx]) = reg_y;
```

> [!note] 向量化不影响结果
> 向量化只是让每线程多处理几个元素，reduce 的结果和标量版本完全一致。
> 因为 reduce 求的是所有元素的总和，每线程贡献 1 个还是 4 个不影响最终结果。

---

## RMS Normalization

### 数学公式

```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * g

其中：
  mean(x²) = sum(x²) / K
  g = 可学习的 scale（无 bias）
  eps = 1e-5
```

### Kernel 流程

```
launch: f32      → <<<(N, 1, 1), (K, 1, 1)>>>
        f32x4    → <<<(N, 1, 1), (K/4, 1, 1)>>>
        f16      → <<<(N, 1, 1), (K, 1, 1)>>>
        f16x8    → <<<(N, 1, 1), (K/8, 1, 1)>>>
```

```cpp
float value = x[idx];

// 只需一次 reduce：求 sum(x²)
float variance = value * value;
variance = block_reduce_sum(variance);
s_variance = rsqrtf(variance / K + eps);

// 归一化（无数值稳定性问题，不需要减均值）
y[idx] = value * s_variance * g;
```

> [!tip] 为什么 RMS Norm 更受欢迎？
> 1. 少一次 reduce → 更快
> 2. 不减均值 → 数值更稳定（mean 的估计在低精度下误差大）
> 3. LLaMA/Mistral/Gemma 等主流模型都用 RMS Norm

### 计算量对比

```
Layer Norm: 2 次 block reduce + 2 次 __syncthreads
RMS Norm:   1 次 block reduce + 1 次 __syncthreads

对 K=4096, 256 threads/block:
  Layer Norm: 2 × (4096/256 × 5 shuffle + smem) ≈ 160 + 40 = ~200 cycles
  RMS Norm:   1 × (4096/256 × 5 shuffle + smem) ≈ 80 + 20  = ~100 cycles
```

---

## NMS (Non-Maximum Suppression)

### 用途

目标检测模型（YOLO, Faster R-CNN）输出大量候选框，NMS 去除重叠度过高的冗余框，只保留得分最高的。

### 算法流程

```
输入: boxes[N][4] (x1,y1,x2,y2), scores[N], iou_threshold
输出: keep[N] (1=保留, 0=抑制)

1. 按 score 降序排列所有框
2. 对每个框 i:
   检查所有得分更高的框 j < i:
     如果 keep[j] == 0 (已被抑制) → 跳过
     计算 IoU(box_i, box_j)
     如果 IoU > threshold → 抑制 box_i (keep[i] = 0)
```

### IoU 计算

```
IoU = Intersection / Union

Intersection = 交集面积
  inter_x1 = max(x1_i, x1_j)
  inter_y1 = max(y1_i, y1_j)
  inter_x2 = min(x2_i, x2_j)
  inter_y2 = min(y2_i, y2_j)
  inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

Union = 面积_i + 面积_j - Intersection
  area_i = (x2_i - x1_i) * (y2_i - y1_i)
  area_j = (x2_j - x1_j) * (y2_j - y1_j)

IoU = inter_area / (area_i + area_j - inter_area)
```

### Kernel 实现

```
launch: <<<(num_boxes/256, 1, 1), (256, 1, 1)>>>
```

```cpp
__global__ void nms_kernel(const float* boxes, const float* scores, int* keep,
                           int num_boxes, float iou_threshold) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_boxes) return;

  // 当前框的坐标
  float x1 = boxes[idx * 4 + 0], y1 = boxes[idx * 4 + 1];
  float x2 = boxes[idx * 4 + 2], y2 = boxes[idx * 4 + 3];

  // 检查所有得分更高的框 (i < idx)
  for (int i = 0; i < idx; ++i) {
    if (keep[i] == 0) continue;  // 已被抑制，跳过

    // 计算 IoU
    float inter_area = ...;
    float iou = inter_area / (area + area_i - inter_area);

    if (iou > iou_threshold) {
      keep[idx] = 0;  // 抑制
      return;
    }
  }
  keep[idx] = 1;  // 保留
}
```

> [!note] NMS 的局限性
> - 每个线程串行遍历所有更高分的框 → O(N²) 复杂度
> - `keep[i]` 的读取有数据依赖（前面的框是否被抑制会影响后面的判断）
> - 实际部署中通常在 CPU 上做 NMS（框数量通常 < 1000），GPU NMS 主要用于训练

---

## SGEMV (Single-precision General Matrix-Vector multiply)

### 数学公式

```
y = A * x

A: (M, K)  矩阵
x: (K, 1)  向量
y: (M, 1)  向量

y[m] = Σ(k=0..K-1) A[m][k] * x[k]
```

每行的点积是独立的 → 每行分配给一组线程，组内用 warp reduce 求和。

### Kernel 1: sgemv_k32 — K≥32，每行一个 warp

```
launch: <<<(M/4, 1, 1), (32, 4, 1)>>>
```

```
Block: (32, 4) = 128 线程 = 4 个 warp
  tx (0~31) = lane = 列索引 k
  ty (0~3)  = warp 编号 = 行偏移
Grid: M/4 个 block → 每 block 处理 4 行

每 warp 处理 1 行，32 线程各算 1 个乘积，warp reduce 求和：
```

```
A 的一行 (K=32):
┌────────────────────────────────────────┐
│ T0   T1   T2   ...           T31      │  ← 32 线程各算 A[m][k]*x[k]
└────────────────────────────────────────┘
         ↓ warp reduce sum
         y[m]
```

```cpp
int m = bx * blockDim.y + ty;  // 每个 ty 处理一行
float sum = 0.f;
for (int w = 0; w < NUM_WARPS; ++w) {  // K>32 时循环多轮
    int k = w * WARP_SIZE + lane;
    sum += a[m * K + k] * x[k];
}
sum = warp_reduce_sum_f32<WARP_SIZE>(sum);
if (lane == 0) y[m] = sum;
```

> [!tip] 为什么 blockDim.y=4？
> 一个 warp 32 线程只处理 1 行太浪费（每线程 1 个乘加）。
> 用 `blockDim.y=4` 让 4 个 warp 各处理 1 行，提高利用率。

> [!note] lane = tx 的原因
> `blockDim.x = 32 = WARP_SIZE`，所以 `lane = (tx + ty*32) % 32 = tx`。
> ty 决定 warp 编号（哪一行），tx 决定 lane（哪一列）。

### Kernel 2: sgemv_k128_f32x4 — K≥128，float4 向量化

```
launch: <<<(M/4, 1, 1), (32, 4, 1)>>>
```

每行 K=128 个元素，用 float4 向量化：每线程加载 4 个 float → 每 warp 一轮处理 32×4=128 个元素。

```
A 的一行 (K=128):
┌────────────────────────────────────────────────────────┐
│ T0×4   T1×4   T2×4   ... T31×4                        │
│ [f f f f] [f f f f] [f f f f]     [f f f f]           │
└────────────────────────────────────────────────────────┘
  32 线程 × 4 元素/线程 = 128 元素 = 1 行
```

```cpp
int k = (w * WARP_SIZE + lane) * 4;  // 每线程起始列
float4 regx = FLOAT4(x[k]);          // 加载 x 的 4 个元素
float4 rega = FLOAT4(a[m * K + k]);  // 加载 A 的 4 个元素
sum += rega.x * regx.x + rega.y * regx.y +
       rega.z * regx.z + rega.w * regx.w;  // 4 次乘加
```

> [!tip] 向量化的好处
> 一次 `FLOAT4` 加载 = 1 条 `LD.E.128` 指令搬运 16 字节。
> 标量需要 4 条 `LD.E.32` 指令。指令数少 4 倍。

### Kernel 3: sgemv_k16 — K 很小（< 32），一个 warp 处理多行

```
launch: <<<(M/NUM_ROWS, 1, 1), (32, NUM_WARPS, 1)>>>
```

K=16 < 32，如果每 warp 只处理 1 行，有 16 个线程闲置。解决方案：每个 warp 处理 2 行，每行 16 个线程。

```
Block: (32, NUM_WARPS)
一个 warp (32 线程) 处理 2 行：

lane 0~15  → 处理行 m 的 16 个元素
lane 16~31 → 处理行 m+1 的 16 个元素

┌──────────────────┬──────────────────┐
│  lane 0~15       │  lane 16~31      │
│  行 m, k=0..15   │  行 m+1, k=0..15 │
└──────────────────┴──────────────────┘
```

```cpp
constexpr int K_WARP_SIZE = 16;  // 32 / 2 行
int k = lane % K_WARP_SIZE;      // 0~15
int m = ... + lane / K_WARP_SIZE; // lane<16 → 行m, lane>=16 → 行m+1

float sum = A[m * K + k] * x[k];
sum = warp_reduce_sum_f32<K_WARP_SIZE>(sum);  // 只在 16 个线程内 reduce
if (k == 0) y[m] = sum;  // 注意：k==0，不是 lane==0
```

> [!important] reduce 宽度的选择
> `warp_reduce_sum_f32<16>` 只做 4 轮 shuffle（log2(16)=4），不是 5 轮。
> 因为每行只有 16 个有效线程，不需要和另外 16 个线程通信。
> 写结果的条件是 `k == 0`（每行的第 0 列），不是 `lane == 0`（warp 的第 0 线程）。

---

## HGEMV (Half-precision General Matrix-Vector multiply)

HGEMV 和 SGEMV 结构完全相同，只是数据类型从 `float` 换成 `half`。

### 区别

| | SGEMV | HGEMV |
|---|---|---|
| 数据类型 | `float` (32-bit) | `half` (16-bit) |
| 向量化 | `float4` (128-bit, 4 元素) | `half2×2` (64-bit, 4 元素) |
| reduce | `warp_reduce_sum_f32` (float 累加) | `warp_reduce_sum_f16` (half 累加) |
| 精度 | 无损 | 有精度损失（half 只有 ~3 位十进制精度） |

> [!note] 为什么 HGEMV 用 half 累加而不是 float？
> 这里的实现用 half 累加（精度较低）。更好的做法是加载 half、累加用 float、最后转回 half。
> 但 half 累加的优势是：reduce 时 `__shfl_xor_sync` 传输的数据量减半（16-bit vs 32-bit）。

### Kernel 1: hgemv_k32 — K≥32，每行一个 warp

```
launch: <<<(M/4, 1, 1), (32, 4, 1)>>>
```

和 SGEMV 的 k32 版本完全同构，只是类型换成 half：

```cpp
half sum = 0.0f;
for (int w = 0; w < NUM_WARPS; ++w) {
    int k = w * WARP_SIZE + lane;
    sum += a[m * K + k] * x[k];  // half × half = half
}
sum = warp_reduce_sum_f16<WARP_SIZE>(sum);  // half 累加
if (lane == 0) y[m] = sum;
```

### Kernel 2: hgemv_k128_f16x4 — K≥128，half2 向量化

```
launch: <<<(M/4, 1, 1), (32, 4, 1)>>>
```

每线程处理 4 个 half 元素，用 `half2` 加载（2 个 half = 32-bit）：

```cpp
int k = (w * WARP_SIZE + lane) * 4;
half2 reg_x_0 = HALF2(x[k + 0]);      // 加载 2 个 half
half2 reg_x_1 = HALF2(x[k + 2]);      // 再加载 2 个 half
half2 reg_a_0 = HALF2(a[m * K + k + 0]);
half2 reg_a_1 = HALF2(a[m * K + k + 2]);

// 4 次乘加
sum += reg_x_0.x * reg_a_0.x + reg_x_0.y * reg_a_0.y +
       reg_x_1.x * reg_a_1.x + reg_x_1.y * reg_a_1.y;
```

> [!tip] half2 向量化
> `HALF2(ptr)` 一次加载 32-bit（2 个 half），等价于 SGEMV 的 `FLOAT4` 思路。
> 但 half2 只有 32-bit，不是 128-bit。要达到 128-bit 需要 `LDST128BITS` 加载 8 个 half。

### Kernel 3: hgemv_k16 — K<16，一个 warp 处理多行

```
launch: <<<(M/NUM_ROWS, 1, 1), (32, NUM_WARPS, 1)>>>
```

和 SGEMV 的 k16 版本同构，reduce 宽度 = 16，写条件 `k == 0`：

```cpp
half sum = A[m * K + k] * x[k];
sum = warp_reduce_sum_f16<K_WARP_SIZE>(sum);  // 16 个线程内 reduce
if (k == 0) y[m] = sum;
```

> [!note] SGEMV vs HGEMV 的代码复用
> 三个 kernel 的逻辑和 SGEMV 完全一致，只是：
> - `float` → `half`
> - `float4` → `half2 × 2`
> - `warp_reduce_sum_f32` → `warp_reduce_sum_f16`
>
> 可以用模板统一（`template<typename T>`），但分开写更清晰、更容易针对 half 做特殊优化。

---

## SGEMM (Single-precision General Matrix Multiply)

SGEMM 是 CUDA 优化的经典课题：`C[M×N] = A[M×K] × B[K×N]`。

### 优化路线总览

```
Naive         → Block Tile       → Thread Tile      → Double Buffer    → cp.async
每线程1元素    → smem缓存tile     → 每线程TM×TN元素   → 计算/搬运重叠     → 异步拷贝
O(MNK)条指令   → 减少GMEM访问     → 提高计算密度      → 隐藏延迟         → 释放warp
```

### Level 0: Naive SGEMM

```
launch: <<<(N/16, M/16, 1), (16, 16, 1)>>>
```

每线程计算 C 的一个元素：`c[m][n] = Σ_k a[m][k] * b[k][n]`

```cpp
int n = blockIdx.x * blockDim.x + threadIdx.x;
int m = blockIdx.y * blockDim.y + threadIdx.y;
float psum = 0.0;
for (int k = 0; k < K; k++) {
    psum += a[m * K + k] * b[k * N + n];  // 每次循环访问 GMEM 两次
}
c[m * N + n] = psum;
```

> [!warning] Naive 的问题
> 每个元素做 K 次乘加，每次都要访问 GMEM（~400 cycles 延迟）。
> 计算/访存比极低 → 完全被内存延迟瓶颈限制。

### Level 1: Block Tile + K Tile + Shared Memory

```
launch: <<<(N/BN, M/BM, 1), (BN, BM, 1)>>>
BM=BN=32, BK=32
```

**核心思想**：把 A 和 B 的 tile 加载到 shared memory，block 内所有线程复用。

```
C 的一个 BM×BN tile 需要 A 的 BM×K 行 和 B 的 K×BN 列

K 方向分块（K Tile）：
  for bk in 0..K/BK:
    加载 A[BM×BK] → s_a (smem)
    加载 B[BK×BN] → s_b (smem)
    __syncthreads()
    每线程计算: sum += s_a[m][k] * s_b[k][n]  for k in 0..BK
    __syncthreads()
```

```
A[M×K]                          B[K×N]
┌──────────────────┐            ┌──────────────────┐
│                  │            │                  │
│  ┌────┐          │            │  ┌────────────┐  │
│  │s_a │ BM×BK    │            │  │   s_b      │  │
│  │    │ → smem   │            │  │  BK×BN     │  │
│  └────┘          │            │  │  → smem    │  │
│                  │            │  └────────────┘  │
└──────────────────┘            └──────────────────┘
         ↓ K方向循环 BK 步
    C[BM×BN] tile 累加完成
```

> [!tip] Shared Memory 的作用
> GMEM 访问 ~400 cycles，SMEM 访问 ~20 cycles。
> 把数据搬到 SMEM 后，block 内 1024 个线程复用 → GMEM 访问次数减少 1024 倍。

### Level 2: Thread Tile (TM×TN)

```
launch: <<<(N/BN, M/BM, 1), (BN/TN, BM/TM, 1)>>>
BM=BN=128, BK=8, TM=TN=8, blockDim=(16,16)=256 threads
```

**核心思想**：每线程不只算 1 个元素，而是算 TM×TN = 8×8 = 64 个元素。

```
C 的一个 BM×BN = 128×128 tile：
┌────────────────────────────────┐
│  T(0,0)  │  T(1,0)  │ ... │15,0│   每个 T = 8×8 子块
│  8×8     │  8×8     │     │8×8 │
├──────────┼──────────┤     │    │
│  T(0,1)  │  T(1,1)  │     │    │
│  8×8     │  8×8     │     │    │
├──────────┼──────────┤     │    │
│  ...     │  ...     │     │    │
│  T(0,15) │  T(1,15) │     │15,15│
└──────────┴──────────┴─────┴────┘
  16×16 = 256 线程，每线程 8×8 = 64 元素
```

**Thread Tile 的索引映射**：

```cpp
// blockDim = (BN/TN, BM/TM) = (16, 16)
// tx = threadIdx.x (0~15) → N 方向
// ty = threadIdx.y (0~15) → M 方向

// 线程 (tx, ty) 负责 C 的第 (ty*TM ~ ty*TM+TM-1) 行, (tx*TN ~ tx*TN+TN-1) 列
// 即 C[ty*8 .. ty*8+7][tx*8 .. tx*8+7]

// 从 smem 读取 A 的一行和 B 的一列，做外积：
for (int k = 0; k < BK; k++) {
    // A 的第 (ty*TM + m) 行, 第 k 列 → r_comp_a[m]
    for (int m = 0; m < TM; m++)
        r_comp_a[m] = s_a[ty * TM + m][k];

    // B 的第 k 行, 第 (tx*TN + n) 列 → r_comp_b[n]
    for (int n = 0; n < TN; n++)
        r_comp_b[n] = s_b[k][tx * TN + n];

    // 外积：TM×1 × 1×TN = TM×TN
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++)
            r_c[m][n] += r_comp_a[m] * r_comp_b[n];
}
```

**Shared Memory 加载索引**（256 线程加载 128×8 + 8×128 的 tile）：

```cpp
// s_a[BM][BK] = s_a[128][8]：128 行 × 8 列
// 每行 8 个 float = 2 个 float4，需要 2 个线程加载
// 128 行 × 2 线程/行 = 256 线程 → 刚好
int load_smem_a_m = tid / 2;              // 行号 0~127
int load_smem_a_k = (tid % 2 == 0) ? 0 : 4;  // 列号 0 或 4
// tid 为偶数加载 s_a[m][0..3]，tid 为奇数加载 s_a[m][4..7]

// s_b[BK][BN] = s_b[8][128]：8 行 × 128 列
// 每行 128 个 float = 32 个 float4，需要 32 个线程加载
// 8 行 × 32 线程/行 = 256 线程 → 刚好
int load_smem_b_k = tid / 32;       // 行号 0~7
int load_smem_b_n = (tid % 32) * 4; // 列号 0,4,8,...,124
// 每线程加载 4 个连续 float (float4)
```

**结果写回**（float4 向量化存储）：

```cpp
// 每线程写 TM=8 行，每行 TN=8 列 → 每行 2 个 float4
for (int m = 0; m < TM; m++) {
    int store_c_m = by * BM + ty * TM + m;       // 全局行号
    for (int n = 0; n < TN; n += 4) {
        int store_c_n = bx * BN + tx * TN + n;    // 全局列号
        FLOAT4(c[store_c_m * N + store_c_n]) = FLOAT4(r_c[m][n]);
    }
}
```

> [!important] Thread Tile 的意义
> Naive 版本：每线程做 K 次乘加，2K 次 GMEM 访问 → 计算/访存比 = K/2K = 0.5
> Thread Tile 版本：每线程做 BK×TM×TN 次乘加，TM+TN 次 SMEM 访问 → 比值大幅提升
>
> 更大的 TM×TN → 更高的计算密度 → 更好的性能（直到寄存器溢出）。
> TM=TN=8 时每线程 64 个 float 累加器 = 256B 寄存器，通常不会溢出。

### Level 3: Bank Conflict Free (BCF)

```
smem 布局: s_a[BK][BM + OFFSET]  ← 转置了！原来是 s_a[BM][BK]
```

**原始布局 `s_a[BM][BK] = s_a[128][8]` 的 bank conflict 分析**：

```
smem 有 32 个 bank，每个 bank 4 字节（1 个 float）。
s_a[128][8] 的内存布局：

行 0:  [b0] [b1] [b2] [b3] [b4] [b5] [b6] [b7]     ← bank 0~7
行 1:  [b8] [b9] [b10][b11][b12][b13][b14][b15]      ← bank 8~15
行 2:  [b16][b17][b18][b19][b20][b21][b22][b23]       ← bank 16~23
行 3:  [b24][b25][b26][b27][b28][b29][b30][b31]       ← bank 24~31
行 4:  [b0] [b1] [b2] [b3] [b4] [b5] [b6] [b7]       ← bank 0~7（重复！）
...
行 31: [b24][b25][b26][b27][b28][b29][b30][b31]       ← bank 24~31

读同一列 k 时（比如 k=0）：
  线程 0 读 s_a[0][0]  → bank 0
  线程 1 读 s_a[1][0]  → bank 8
  线程 2 读 s_a[2][0]  → bank 16
  线程 3 读 s_a[3][0]  → bank 24
  线程 4 读 s_a[4][0]  → bank 0  ← 和线程 0 冲突！
  线程 5 读 s_a[5][0]  → bank 8  ← 和线程 1 冲突！
  ...
  → 4-way bank conflict
```

**转置布局 `s_a[BK][BM + OFFSET] = s_a[8][129]`**：

```
加 OFFSET=1 后每行 129 个 float，129 % 32 = 1 → 每行起始 bank 错开 1 个。

行 0:  [b0] [b1] ... [b31] [b0] ...     ← 从 bank 0 开始
行 1:  [b1] [b2] ... [b0]  [b1] ...     ← 从 bank 1 开始（错开！）
行 2:  [b2] [b3] ... [b1]  [b2] ...     ← 从 bank 2 开始
...

读同一行的不同列时 → 每个线程访问不同 bank → 无冲突。
```

> [!note] BCF 的代价
> A 矩阵是行主序，但 smem 按列存储 → 写入时需要"在线转置"：
> `s_a[k][m] = a[m][k]` 而不是 `s_a[m][k] = a[m][k]`
> 即：从 GMEM 读取一行 A，逐元素写入 smem 的不同行。

### Level 4: Double Buffering

```
smem 布局: s_a[2][BK][BM]  ← 2 个 buffer
```

**单缓冲的问题**：

```
bk=0: [加载 s_a/s_b] → [sync] → [计算] → [sync]
bk=1: [加载 s_a/s_b] → [sync] → [计算] → [sync]
      ↑ 加载时计算单元空闲            ↑ 计算时加载单元空闲
```

**双缓冲：加载和计算重叠**：

```
bk=0: [加载 buf0] → [sync] → [计算 buf0 + 加载 buf1] → [sync]
bk=1:                                [计算 buf1 + 加载 buf0] → [sync]
bk=2:                                [计算 buf0 + 加载 buf1] → [sync]
...
最后一次:                              [计算 buf_last]
```

**双缓冲的主循环结构**：

```cpp
// 预加载第一个 tile 到 buf[0]
FLOAT4(r_load_a) = FLOAT4(a[...]);
s_a[0][...] = r_load_a;
FLOAT4(s_b[0][...]) = FLOAT4(b[...]);
__syncthreads();

// 主循环从 bk=1 开始
for (int bk = 1; bk < num_tiles; bk++) {
    int sel = (bk - 1) & 1;      // 当前计算的 buffer
    int sel_next = bk & 1;       // 下一个加载的 buffer

    // ① 从 GMEM 加载到寄存器（和下面的计算并行）
    FLOAT4(r_load_a) = FLOAT4(a[...]);
    FLOAT4(r_load_b) = FLOAT4(b[...]);

    // ② 从 buf[sel] 计算（和上面的加载并行）
    for (int tk = 0; tk < BK; tk++) {
        FLOAT4(r_comp_a) = FLOAT4(s_a[sel][tk][...]);
        FLOAT4(r_comp_b) = FLOAT4(s_b[sel][tk][...]);
        // ... TM×TN 外积累加到 r_c
    }

    // ③ 把寄存器中的数据写入 buf[sel_next]
    s_a[sel_next][...] = r_load_a;
    FLOAT4(s_b[sel_next][...]) = FLOAT4(r_load_b);

    __syncthreads();  // 确保 buf[sel_next] 写完
}

// 计算最后一个 tile（buf[1]）
for (int tk = 0; tk < BK; tk++) {
    FLOAT4(r_comp_a) = FLOAT4(s_a[1][tk][...]);
    FLOAT4(r_comp_b) = FLOAT4(s_b[1][tk][...]);
    // ... TM×TN 外积
}
```

> [!tip] 为什么 Double Buffering 能减少 __syncthreads__？
> 单缓冲版本：每次迭代需要 2 次 sync（加载后 + 计算后）
> 双缓冲版本：每次迭代只需 1 次 sync（因为加载和计算用不同的 buffer，无依赖）
>
> 总共节省 `(K/BK) - 1` 次 sync。对于 K=1024, BK=8 → 节省 127 次 sync。

### Level 5: cp.async（异步拷贝）

```cpp
// 普通加载：warp 发出 load 指令后等待数据到达
FLOAT4(s_b[...]) = FLOAT4(b[...]);  // warp 阻塞直到数据到达

// cp.async：warp 发出异步请求后立即去做其他事
CP_ASYNC_CA(smem_ptr, &b[...], 16);  // 发出请求，不等待
CP_ASYNC_COMMIT_GROUP();               // 提交一组请求

// ... 在等待期间可以做计算 ...

CP_ASYNC_WAIT_GROUP(0);  // 等待所有请求完成
```

```
cp.async 指令:
  cp.async.ca.shared.global  → 从 GMEM 异步拷贝到 SMEM
  .L2::128B                  → 经过 L2 cache，128B 对齐
  支持 4/8/16 字节

优势：
  - warp 不需要等待数据到达 → 可以去做计算
  - 数据直接从 GMEM → SMEM，不经过寄存器 → 节省 RF 带宽
  - 和 Double Buffering 配合 → 计算和搬运完全重叠
```

> [!important] cp.async 的架构要求
> `cp.async` 需要 SM80+（Ampere 及以上）。
> 在 SM75（Turing）上不可用，需要退回到普通加载。

### SGEMM 优化路线总结

| Level | 技术 | 关键改进 | 性能 |
|---|---|---|---|
| 0 | Naive | — | ~5% cuBLAS |
| 1 | Block Tile + smem | 减少 GMEM 访问 | ~20% |
| 2 | Thread Tile (TM×TN) | 提高计算密度 | ~50% |
| 3 | BCF (转置 smem) | 消除 bank conflict | ~60% |
| 4 | Double Buffering | 计算/搬运重叠 | ~80% |
| 5 | cp.async | 异步搬运，释放 warp | ~90% |

> [!note] SGEMM kernel 的命名规则
> `sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async` 含义：
> - `t_8x8`：Thread Tile = 8×8
> - `sliced_k16`：K 方向分块 BK=16
> - `f32x4`：float4 向量化加载
> - `bcf`：Bank Conflict Free（转置 smem）
> - `dbuf`：Double Buffering
> - `async`：cp.async 异步拷贝

---

## HGEMM (Half-precision General Matrix Multiply)

HGEMM 和 SGEMM 结构完全相同，只是数据类型从 `float` 换成 `half`。优化路线一致：Naive → Thread Tile → BCF → Double Buffer → cp.async。

### HGEMM vs SGEMM 的关键区别

| | SGEMM | HGEMM |
|---|---|---|
| 数据类型 | `float` (32-bit) | `half` (16-bit) |
| 向量化 | `float4` (128-bit, 4 元素) | `half2` (32-bit, 2 元素) / `LDST128BITS` (128-bit, 8 元素) |
| smem 占用 | 128×8×4 = 4KB | 128×8×2 = 2KB（减半） |
| 计算指令 | `__fmaf_rn` (FMA) | `__hfma2` (half2 FMA) 或 `__hmul` + `__hadd` |
| smem bank | 4B/bank, 32 banks | 2B/bank, 32 banks → bank conflict 更严重 |

> [!important] half 的 bank conflict 更严重
> half 只有 2 字节，而 smem bank 是 4 字节宽。所以：
> - 2 个 half 占 1 个 bank → 同一 bank 内有 2 个 half
> - 读同一个 half2 的两个元素 → 可能冲突
> - 需要更仔细的 padding 或转置来避免冲突

### 向量化策略对比

```
SGEMM:
  float4 加载 = 4 × 4B = 128-bit → 1 条 LD.E.128

HGEMM:
  half2 加载  = 2 × 2B = 32-bit  → 1 条 LD.E.32   (f16x2)
  half4 加载  = 4 × 2B = 64-bit  → 1 条 LD.E.64   (f16x4)
  half8 加载  = 8 × 2B = 128-bit → 1 条 LD.E.128  (f16x8_pack)
```

> [!tip] f16x8_pack 的含义
> 8 个 half = 16 字节 = 128 bit，正好一条 `LD.E.128` 指令。
> 用 `LDST128BITS(pack[0]) = LDST128BITS(a[idx])` 实现。
> 这是 HGEMM 最高效的加载方式，和 SGEMM 的 `float4` 等价。

### HGEMM Kernel 清单

```
hgemm.cu:
  hgemm_naive_f16_kernel                    → Level 0: Naive
  hgemm_sliced_k_f16_kernel                 → Level 1: Block Tile + smem
  hgemm_t_8x8_sliced_k_f16x4_kernel         → Level 2: Thread Tile + half2
  hgemm_t_8x8_sliced_k_f16x4_pack_kernel    → Level 2: Thread Tile + half4 pack
  hgemm_t_8x8_sliced_k_f16x4_bcf_kernel     → Level 3: BCF
  hgemm_t_8x8_sliced_k_f16x4_pack_bcf_kernel→ Level 3: BCF + pack
  hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel→ Level 3: BCF + 128-bit pack
  hgemm_t_8x8_sliced_k_f16x8_pack_bcf_dbuf_kernel → Level 4: Double Buffer

hgemm_async.cu:
  hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_kernel   → Level 4: BK=16 + dbuf
  hgemm_t_8x8_sliced_k16_f16x8_pack_dbuf_async_kernel → Level 5: cp.async
  hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_kernel   → Level 4: BK=32 + dbuf
  hgemm_t_8x8_sliced_k32_f16x8_pack_dbuf_async_kernel → Level 5: BK=32 + async
  hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_kernel   → Level 4: TM=16,TN=8
  hgemm_t_16x8_sliced_k32_f16x8_pack_dbuf_async_kernel → Level 5: TM=16,TN=8 + async
```

### Thread Tile 的选择：8×8 vs 16×8

```
8×8:  TM=8, TN=8 → 每线程 64 个 half 累加器 = 128B 寄存器
16×8: TM=16, TN=8 → 每线程 128 个 half 累加器 = 256B 寄存器

16×8 的优势：更大的 tile → 更高的计算密度
16×8 的劣势：寄存器压力更大 → occupancy 可能下降
```

> [!note] BK 的选择对 HGEMM 的影响
> SGEMM 通常用 BK=8 或 BK=16。
> HGEMM 可以用更大的 BK（如 BK=32），因为 half 只占 2B：
> - BK=32, BM=128 → s_a = 32×128×2 = 8KB
> - BK=16, BM=128 → s_a = 16×128×2 = 4KB
>
> 更大的 BK → 更少的 K 方向迭代 → 更少的 __syncthreads__
> 但 smem 占用更大 → 可能降低 occupancy。

### cp.async 在 HGEMM 中的使用

和 SGEMM 完全相同，只是数据量减半：

```cpp
// SGEMM: 16 字节 = 4 个 float
CP_ASYNC_CA(smem_ptr, &b[addr], 16);

// HGEMM: 16 字节 = 8 个 half（同样 128-bit）
CP_ASYNC_CA(smem_ptr, &b[addr], 16);
```

> [!tip] cp.async 对 HGEMM 的收益更大
> half 的 GMEM 带宽需求是 float 的一半，但延迟相同（~400 cycles）。
> 所以 HGEMM 比 SGEMM 更容易受内存延迟限制 → cp.async 的异步特性收益更大。

---

## SGEMM with Tensor Cores (WMMA TF32)

前面的 SGEMM 全部用 **CUDA Core**（标量 FMA）。Tensor Core 可以一条指令算一个小矩阵乘，吞吐量高得多。

### TF32 是什么？

```
FP32:  1 sign + 8 exponent + 23 mantissa = 32 bit
TF32:  1 sign + 8 exponent + 10 mantissa = 19 bit（截断尾数）
FP16:  1 sign + 5 exponent + 10 mantissa = 16 bit

TF32 = FP32 的范围 + FP16 的精度
→ 可以直接从 FP32 截断得到，不需要重新训练
→ Ampere Tensor Core 原生支持
```

> [!tip] TF32 的优势
> - 输入输出都是 FP32 指针 → 不需要类型转换
> - 计算用 TF32（19-bit）→ Tensor Core 加速
> - 累加用 FP32 → 精度损失可控
> - PyTorch 默认的 `torch.matmul` 在 Ampere 上就用 TF32

### WMMA API

WMMA (Warp Matrix Multiply Accumulate) 是 NVIDIA 提供的高级 Tensor Core API：

```cpp
#include <mma.h>
using namespace nvcuda;

// 定义 fragment（每个 warp 线程持有的数据）
wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> A_frag;
wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> B_frag;
wmma::fragment<wmma::accumulator, 16, 16, 8, float> C_frag;

// 初始化累加器
wmma::fill_fragment(C_frag, 0.0f);

// 从 smem 加载到 fragment
wmma::load_matrix_sync(A_frag, &s_a[row][0], smem_stride);
wmma::load_matrix_sync(B_frag, &s_b[row][0], smem_stride);

// 矩阵乘累加：C += A × B
wmma::mma_sync(C_frag, A_frag, B_frag, C_frag);

// 从 fragment 写回 smem
wmma::store_matrix_sync(&s_c[row][0], C_frag, smem_stride);
```

> [!note] WMMA 的 tile 大小
> `mma.sync.m16n16k8.tf32`：一次计算 16×16×8 的矩阵乘。
> - A fragment: 16×8 = 128 个 TF32 元素，32 线程 → 每线程 4 个
> - B fragment: 8×16 = 128 个 TF32 元素，32 线程 → 每线程 4 个
> - C fragment: 16×16 = 256 个 FP32 元素，32 线程 → 每线程 8 个

### WMMA SGEMM Kernel 结构

```
launch: <<<(N/BN, M/BM, 1), (32, WMMA_TILE_M * WMMA_TILE_N, 1)>>>
BM=128, BN=128, BK=8, WMMA_M=16, WMMA_N=16, WMMA_K=8
```

```cpp
// 每个 warp 负责一个 WMMA tile (16×16)
// block 内有 WMMA_TILE_M × WMMA_TILE_N 个 warp
// 每个 warp 做 WARP_TILE_M × WARP_TILE_N 次 WMMA

wmma::fragment<wmma::accumulator, M, N, K, float> C_frag[TILE_M][TILE_N];
for (int i = 0; i < TILE_M; i++)
    for (int j = 0; j < TILE_N; j++)
        wmma::fill_fragment(C_frag[i][j], 0.0f);

for (int k = 0; k < K_TILES; k++) {
    // 加载 A 和 B 的 fragment
    for (int i = 0; i < TILE_M; i++)
        wmma::load_matrix_sync(A_frag[i], &s_a[...], stride);
    for (int j = 0; j < TILE_N; j++)
        wmma::load_matrix_sync(B_frag[j], &s_b[...], stride);

    // WMMA: C[i][j] += A[i] × B[j]
    for (int i = 0; i < TILE_M; i++)
        for (int j = 0; j < TILE_N; j++)
            wmma::mma_sync(C_frag[i][j], A_frag[i], B_frag[j], C_frag[i][j]);
}

// 写回结果
for (int i = 0; i < TILE_M; i++)
    for (int j = 0; j < TILE_N; j++)
        wmma::store_matrix_sync(&s_c[...], C_frag[i][j], stride);
```

> [!important] WMMA 的 smem 布局要求
> `wmma::load_matrix_sync` 要求 smem 的 stride 满足对齐要求。
> 对于 TF32（4B 元素），stride 必须是 4 的倍数（16 字节对齐）。
> 不需要手动处理 swizzle — WMMA 内部处理了 bank conflict。

---

## HGEMM with MMA PTX Instructions

WMMA 是高级 API，MMA PTX 是底层指令。手写 MMA PTX 可以更精细地控制寄存器分配和指令调度。

### MMA PTX 指令

```cpp
// mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16
// D[16×8] = A[16×16] × B[16×8] + C[16×8]
// A: row_major, B: col_major, 所有类型 f16
asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
    "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
    : "=r"(RD0), "=r"(RD1)           // 输出 D (2 个 32-bit reg = 4 个 half)
    : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3),  // 输入 A (4 个 32-bit reg = 8 个 half)
      "r"(RB0), "r"(RB1),                       // 输入 B (2 个 32-bit reg = 4 个 half)
      "r"(RC0), "r"(RC1)                         // 输入 C (2 个 32-bit reg = 4 个 half)
);
```

> [!note] MMA m16n8k16 的寄存器布局
> - A: 4 个 32-bit 寄存器 = 8 个 half → 16×16 的 A 矩阵
> - B: 2 个 32-bit 寄存器 = 4 个 half → 16×8 的 B 矩阵
> - C/D: 2 个 32-bit 寄存器 = 4 个 half → 16×8 的输出
> - 32 个线程协作完成一个 16×8×16 的矩阵乘

### ldmatrix 指令

`ldmatrix` 是专门为 Tensor Core 设计的 SMEM→RF 加载指令：

```cpp
// ldmatrix.sync.aligned.x4.m8n8.shared.b16
// 从 smem 加载 4 个 8×8 矩阵到寄存器（按 MMA fragment 布局排列）
asm volatile(
    "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
    : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)
    : "r"(smem_addr)
);
```

**ldmatrix 的加载语义**：

```
ldmatrix.x4 加载 4 个 8×8 的 half 子矩阵：

smem 中的布局（以 A 矩阵为例）：
  地址 = &s_a[lane_id % 16][(lane_id / 16) * 8]
  → lane 0~15 读 s_a[0..15][0..7]   (第一个 8×8)
  → lane 16~31 读 s_a[0..15][8..15]  (第二个 8×8)
  → 但实际布局由 MMA fragment 决定

每线程获得 4 个 32-bit 寄存器 = 8 个 half = 该线程负责的 A fragment
→ 直接可以喂给 mma.sync 指令
```

**实际代码中的 ldmatrix 调用**：

```cpp
// 计算 smem 地址
uint32_t load_smem_a_ptr =
    __cvta_generic_to_shared(&s_a[lane_id % 16][(lane_id / 16) * 8]);
// lane_id % 16 → 行号 (0~15)
// (lane_id / 16) * 8 → 列号 (0 或 8)

// 加载 A fragment（4 个寄存器 = 8 个 half）
LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], load_smem_a_ptr);

// 加载 B fragment（2 个寄存器 = 4 个 half）
uint32_t load_smem_b_ptr = __cvta_generic_to_shared(&s_b[lane_id % 16][0]);
LDMATRIX_X2_T(RB[0], RB[1], load_smem_b_ptr);
//                 ↑ .trans = 加载时转置
```

> [!tip] ldmatrix vs 普通加载
> 普通加载：连续读 smem → 寄存器中的数据是线性排列的
> ldmatrix：按 MMA fragment 布局读 smem → 寄存器中的数据直接可以喂给 MMA 指令
>
> 如果用普通加载，需要额外的 shuffle 指令来重排数据 → 浪费指令和寄存器。

### MMA m16n8k16 的寄存器布局

```
mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {d0,d1}, {a0,a1,a2,a3}, {b0,b1}, {c0,c1}

A fragment (16×16, row_major):
  每线程 4 个 32-bit 寄存器 = 8 个 half
  线程 0:  A[0][0:1], A[0][2:3], A[8][0:1], A[8][2:3]
  线程 1:  A[0][4:5], A[0][6:7], A[8][4:5], A[8][6:7]
  ...
  线程 15: A[7][4:5], A[7][6:7], A[15][4:5], A[15][6:7]
  线程 16: A[0][8:9], A[0][10:11], A[8][8:9], A[8][10:11]
  ...

B fragment (16×8, col_major):
  每线程 2 个 32-bit 寄存器 = 4 个 half
  线程 0:  B[0][0:1], B[8][0:1]
  线程 1:  B[0][2:3], B[8][2:3]
  ...

C/D fragment (16×8):
  每线程 2 个 32-bit 寄存器 = 4 个 half
  线程 0:  C[0][0:1], C[8][0:1]
  线程 1:  C[0][2:3], C[8][2:3]
  ...
```

> [!important] 为什么 B 要用 ldmatrix.trans？
> MMA 指令要求 B 是 **col_major**（列主序）。
> 但我们的 B 矩阵在 GMEM 和 smem 中都是 **row_major**（行主序）。
> `ldmatrix.trans` 在加载时自动转置，一步完成 row_major → col_major 的转换。
> 如果 B 已经是 col_major 存储，直接用普通 `ldmatrix`。

### ldmatrix.trans（转置加载）

```cpp
// ldmatrix.trans: 加载时同时转置
// B 矩阵是 col_major 存储，但 MMA 要求 row_major 输入
// 用 ldmatrix.trans 一步完成加载+转置
asm volatile(
    "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
    : "=r"(R0), "=r"(R1)
    : "r"(smem_addr)
);
```

> [!note] 为什么 B 需要 trans？
> MMA 指令的 B 矩阵是 `col_major`（列主序）。
> 如果 smem 中 B 是 `row_major`（行主序），就需要 `ldmatrix.trans` 来转置。
> 如果 smem 中 B 已经是 `col_major`，直接用普通 `ldmatrix`。

### MMA HGEMM Kernel 结构

```cpp
// 每个 warp 处理一个 MMA tile (16×8)
// block 内有 (BM/16) × (BN/8) 个 warp

for (int k = 0; k < K_TILES; k++) {
    // GMEM → SMEM (cp.async)
    CP_ASYNC_CA(smem_a_ptr, &a[...], 16);
    CP_ASYNC_CA(smem_b_ptr, &b[...], 16);

    // SMEM → RF (ldmatrix)
    LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], smem_a_ptr);  // 加载 A fragment
    LDMATRIX_X2_T(RB[0], RB[1], smem_b_ptr);               // 加载 B fragment (转置)

    // MMA: D = A × B + C
    HMMA16816(RD[0], RD[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
}
```

### Multi-Stage Pipelining with MMA

```
smem 布局: s_a[NUM_STAGE][BK][BM], s_b[NUM_STAGE][BK][BN]

Stage 0: [加载 stage 0] [计算 stage 0] [加载 stage 1] [计算 stage 1] ...
         ↑ cp.async                 ↑ ldmatrix + MMA
```

```cpp
// 预加载前几个 stage
for (int s = 0; s < NUM_STAGE - 1; s++) {
    CP_ASYNC_CA(s_a[s]..., &a[...], 16);
    CP_ASYNC_CA(s_b[s]..., &b[...], 16);
    CP_ASYNC_COMMIT_GROUP();
}

// 主循环：计算当前 stage + 预加载下一个 stage
for (int k = NUM_STAGE - 1; k < K_TILES; k++) {
    int stage = k % NUM_STAGE;
    int stage_next = (k + 1) % NUM_STAGE;

    // 预加载下一个 stage
    CP_ASYNC_CA(s_a[stage_next]..., &a[...], 16);
    CP_ASYNC_COMMIT_GROUP();

    // 计算当前 stage
    for (int bk = 0; bk < BK; bk++) {
        LDMATRIX_X4(RA..., &s_a[stage][bk][...]);
        LDMATRIX_X2_T(RB..., &s_b[stage][bk][...]);
        HMMA16816(RD..., RA..., RB..., RC...);
    }

    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
}
```

> [!important] WMMA vs MMA PTX 的选择
> | | WMMA API | MMA PTX |
> |---|---|---|
> | 易用性 | 高（C++ API） | 低（内联汇编） |
> | 灵活性 | 低（固定 tile 大小） | 高（可自由组合） |
> | 性能 | 好 | 更好（精细控制） |
> | 适用场景 | 快速原型 | 极致优化 |
>
> WMMA 适合入门和快速验证。MMA PTX 适合生产环境的极致优化。

### Tensor Core GEMM vs CUDA Core GEMM 性能对比

```
SGEMM (CUDA Core, 256 threads, 128×128 tile):
  ~500 GFLOPS (RTX 3080)

SGEMM WMMA TF32 (Tensor Core):
  ~150 GFLOPS (TF32 精度较低，但吞吐量高)

HGEMM MMA PTX (Tensor Core):
  ~300 GFLOPS (half 精度，2x 吞吐量 vs TF32)

cuBLAS:
  ~350 GFLOPS (HGEMM)
```

> [!note] Tensor Core 不是万能的
> - 小矩阵（M,N < 64）：CUDA Core 可能更快（Tensor Core 的启动开销大）
> - 矩阵形状不规则：Tensor Core 的 tile 对齐要求导致浪费
> - 需要高精度：TF32 有精度损失，FP32 CUDA Core 更准
