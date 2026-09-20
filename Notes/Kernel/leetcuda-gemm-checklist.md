---
type: Note
related_to: "[[LeetCuda]]"
status: Active
---

# LeetCUDA GEMM 学习 Checklist

> [!info] 这份清单解决什么问题
> SGEMM 的 CUDA Core 路线已经走通 5 个变体（naive → sliced_k → t8x8 → bcf → dbuf）。
> 剩下的 GEMM 变体看起来很多（async / wmma / mma / swizzle / wgmma / hgemm 各种 pack），
> 但**真正的新概念只有 6 个**，其余都是同一份代码换常量（tile 尺寸、stage 数、layout）。
> 本清单按「概念增量 ÷ 代码量」排序，每一步给出：**读哪段代码（含行号）→ 自检问题 → 动手验证 → 验收标准**，
> 并明确列出**该跳过什么**。

仓库根目录：`/media/lyxwxj/Data/common/Workspace/Omni-infra/kernels/Basic/LeetCUDA`（下文所有行号均指该仓库内文件）。

---

## 0. 起点：已完成 / 待学

### 已完成（SIMT / CUDA Core 路线）

- [x] `sgemm_naive_f32_kernel` — 每线程 1 元素，无 smem
- [x] `sgemm_sliced_k_f32_kernel` — Block Tile + K-slice + smem
- [x] `sgemm_t_8x8_sliced_k_f32x4_kernel` — Thread Tile 8×8 + float4
- [x] `sgemm_t_8x8_sliced_k_f32x4_bcf_kernel` — smem padding 消除 bank conflict
- [x] `sgemm_t_8x8_sliced_k_f32x4_bcf_dbuf_kernel` — double buffer 重叠访存与计算

### 待学：6 个真正的新概念

| # | 新概念 | 一句话 | 重要度 |
|---|---|---|---|
| ① | `cp.async` 异步拷贝 | 绕过「LDG→寄存器→STS」，把访存交给硬件流水线 | 必学（后续全部依赖） |
| ② | multi-stage 流水（≥2 buffer） | 参数化 stage 数，用 `cp.async.wait_group(N)` 控制飞行中批次 | 必学 |
| ③ | Tensor Core fragment 布局 + `ldmatrix` | 白盒理解 MMA 的寄存器映射，是一切上层优化的地基 | **最高** |
| ④ | SMEM swizzle（XOR 置换） | 替代 padding，用异或重排地址消除 bank conflict | **性价比最高** |
| ⑤ | Block swizzle | 改写 grid 发射顺序，提高 L2 复用 | 便宜，顺带学 |
| ⑥ | WGMMA + TMA + warp specialization | Hopper 级异步执行模型 | 仅 sm90a 需要 |

---

## 1. 路径总览（按性价比排序）

| 步骤 | 概念 | 主要文件（行号） | 新增代码量 | 预计 | 状态 |
|---|---|---|---|---|---|
| Step 1 | `cp.async` | `kernels/sgemm/sgemm_async.cu:399-537` | ~140 行 | 0.5 天 | [ ] |
| Step 2 | WMMA 入门 | `kernels/hgemm/wmma/hgemm_wmma.cu:47-78, 161-272` | ~110 行 | 1 天 | [ ] |
| Step 3 | **MMA PTX m16n8k16** | `kernels/hgemm/mma/basic/hgemm_mma.cu:80-161, 164-370` | 370 行 | 1–2 天 | [ ] |
| Step 4 | multi-stage + TN + block swizzle | `kernels/hgemm/mma/basic/hgemm_mma_stage_tn.cu:124-386` | 629 行 | 1–2 天 | [ ] |
| Step 5 | **SMEM swizzle** | `kernels/swizzle/mat_trans_swizzle.cu`、`hgemm_mma_swizzle.cu` | 112 + 706 行 | 1 天 | [ ] |
| Step 6 | 工程版 / CuTe | `.../swizzle/hgemm_mma_stage_tn_swizzle_x2.cu`、`cutlass/hgemm_mma_stage_tn_cute.cu` | 894 / 546 行 | 1–2 天 | [ ] |
| 可选 | WGMMA + TMA | `kernels/hgemm/wgmma/hgemm_wgmma_fp32acc_stages_tn.cu` | 559 行 | 2 天 | [ ] |

> [!tip] 阅读加速器
> 仓库自带一份**浓缩注释版**：`kernels/interview/sgemm.cuh`（SGEMM Level 1–5）与
> `kernels/interview/hgemm.cuh`（Phase 7b-2 / 7b-3 / 7c，中文「面试要点」注释）。
> 同一实现的工程版 + 注释版对照读，能省掉大量猜意图的时间。

---

## 2. 逐步 Checklist

### Step 1 — `cp.async` 异步拷贝

**目标**：理解为什么异步拷贝能让「访存」与「计算」真正重叠，而不是靠寄存器中转制造伪重叠。

**读什么**

| 文件:行 | 内容 | 为什么读 |
|---|---|---|
| `kernels/sgemm/sgemm_async.cu:25-45` | `CP_ASYNC_CA / _CG / COMMIT_GROUP / WAIT_GROUP` 宏 | 只有这几个宏是新东西 |
| `kernels/sgemm/sgemm_async.cu:399-537` | `sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async_kernel` | 主读目标，和你学过的 dbuf 版逐行 diff |
| `kernels/sgemm/sgemm_async.cu:276-398` | 同文件 dbuf（无 async）版 | 对照组 |

**新增概念要点**

- `cp.async.cg.shared.global` 一次最多搬 16 bytes，**只支持 16B**（`ca` 支持 4/8/16B）
- `commit_group` 把这一批发出的拷贝打包成一个 group；`wait_group(N)` 表示「允许最多 N 个 group 还没完成」
- 因此 stage 数由 `wait_group(K_STAGE - 2)` 参数化，这是 ② multi-stage 的入口
- 目标地址要先 `__cvta_generic_to_shared()` 转成 32-bit smem 地址

**自检问题**

- [ ] dbuf 版为什么要先把数据 load 到 `r_load_a[]` 寄存器再写 smem？async 版省掉了哪几条指令？
- [ ] `wait_group(0)` 和 `wait_group(1)` 分别在什么语义下使用？
- [ ] 为什么 async 版通常把 `BK` 从 8 提到 16（或 32）？16B 对齐与单线程搬运量的关系是什么？
- [ ] `cp.async` 之后仍然需要 `__syncthreads()` 吗？为什么？

**动手验证**

```bash
cd kernels/sgemm
export TORCH_CUDA_ARCH_LIST=Ada   # 或 Ampere
python3 sgemm.py
```

> [!warning] `sgemm.py` 默认不测 async 变体
> `sgemm_async.cu` 里的 6 个 kernel 已绑定（`sgemm_t_8x{4,8,16}_sliced_k16_f32x4_bcf_dbuf[_async]`，见 844/872/902/930/960/988 行），
> 但没进 `sgemm.py` 的 benchmark 列表。想对比就在 `sgemm.py` 的 run_benchmark 段落补两行：
> ```python
> run_benchmark(lib.sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf,       a, b, "f32x4(t8x8k16dbuf)", c)
> run_benchmark(lib.sgemm_t_8x8_sliced_k16_f32x4_bcf_dbuf_async, a, b, "f32x4(t8x8k16async)", c)
> ```

**验收标准**

- [ ] 数值与 cuBLAS 一致（相对误差 ~1e-6）
- [ ] SASS 中出现 `LDGSTS`（`cuobjdump -sass` 搜一下），说明没退化成 LDG+STS
- [ ] TFLOPS 相对 dbuf 版有提升（K 越大越明显）

> [!note] 跳过
> 同文件 `t_8x4` / `t_8x16` 只是 tile 尺寸扫描（thread tile 8×4 / 8×16），代码结构完全相同，扫一眼参数即可。

---

### Step 2 — WMMA 进 Tensor Core（最平滑的入口）

**目标**：先用「黑盒 API」建立 Tensor Core 的直觉：fragment 是什么、atom / tile / warp 三层 tiling 怎么叠。

**读什么**

| 文件:行 | 内容 | 为什么读 |
|---|---|---|
| `kernels/hgemm/wmma/hgemm_wmma.cu:47-78` | `hgemm_wmma_m16n16k16_naive_kernel` | 仅 30 行，最小可运行 WMMA |
| `kernels/hgemm/wmma/hgemm_wmma.cu:81-157` | `..._mma4x2_kernel` | 一个 warp 算 4×2 个 WMMA atom |
| `kernels/hgemm/wmma/hgemm_wmma.cu:161-272` | `..._mma4x2_warp2x4_kernel` | block = 2×4 warp → 128×128 tile |
| `kernels/sgemm/sgemm_wmma_tf32_stage.cu:276` | `sgemm_wmma_m16n16k8_mma4x2_warp2x4_stages_dsmem_kernel` | 想留在 f32 就看这条：TF32 + stages + block swizzle |

**新增概念要点**

- `wmma::fragment<matrix_a/matrix_b/accumulator, M, N, K, T>`：fragment 是 **per-warp** 的寄存器片段，长度由实现决定，不要假设布局
- `load_matrix_sync` / `mma_sync` / `store_matrix_sync` 三步曲
- **三层 tiling**：MMA atom（16×16×16）→ MMA tile（mma4x2 = 8 atom）→ warp tile（warp2x4 = 8 warp，block 128×128）
- TF32 路径注意：`sgemm_wmma_tf32_stage.cu:48` 的 `f32x4_tf32x4_kernel` 是一次额外的 FP32→TF32 转换，会摊薄收益

**自检问题**

- [ ] fragment 为什么不能跨线程共享？一个 warp 内 32 个 lane 各持有什么？
- [ ] `mma4x2` 与 `warp2x4` 分别沿哪个维度扩张？为什么 block tile 变成 128×128？
- [ ] WMMA API 为什么做不了 SMEM swizzle？（提示：它不暴露 smem 地址计算）
- [ ] WMMA 与 MMA PTX 的取舍是什么？（仓库 README 记录：4090 上大矩阵 WMMA 更好，小矩阵 MMA 更好）

**动手验证**

```bash
cd kernels/hgemm
export TORCH_CUDA_ARCH_LIST=Ada
python3 hgemm.py --M 4096 --N 4096 --K 4096 --i 10 --wmma-all --plot
```

**验收标准**

- [ ] `--wmma-all` 全绿（数值正确）
- [ ] 观察 `mma4x2` → `mma4x2+warp2x4` 的 TFLOPS 提升，能解释来源（数据复用 + 访存/计算比）
- [ ] 若跑 TF32 路径，能解释它为什么只有 cuBLAS TF32 的 ~80%（额外转换 kernel + API 限制）

> [!note] 跳过
> `hgemm_wmma.cu:277-448`（`_dbuf_async`）和 `453+`（`m32n8k16_mma2x4`）先不看；前者是 Step 1 的 f16 版，后者是 atom 形状的另一种切法。

---

### Step 3 — MMA PTX m16n8k16（★收益最高的一步）

**目标**：把 Tensor Core 从黑盒变白盒——掌握 fragment 的**精确寄存器布局**，这是后面所有内容（swizzle、CuTe、WGMMA、flash-attn）的地基。

**读什么**

| 文件:行 | 内容 | 为什么读 |
|---|---|---|
| `kernels/hgemm/mma/basic/hgemm_mma.cu:40-70` | `LDMATRIX_X1/X2/X4`、`LDMATRIX_X4_T`、`MMA_F16F16F16F16` 宏 | 全部内联汇编，只有 5 个宏 |
| `kernels/hgemm/mma/basic/hgemm_mma.cu:80-161` | `hgemm_mma_m16n8k16_naive_kernel` | 单 warp、单 atom，最小完整闭环 |
| `kernels/hgemm/mma/basic/hgemm_mma.cu:164-370` | `hgemm_mma_m16n8k16_mma2x4_warp4x4_kernel` | 128×128 tile + padding |
| `kernels/interview/hgemm.cuh:52-387` | Phase 7b-2 统一注释版 | 中文注释讲清 TN 布局与流水 |

**新增概念要点**

- `ldmatrix.sync.aligned.x4.m8n8.shared.b16`：一条指令让 32 个 lane 各取 4 个 32-bit 寄存器，直接喂给 MMA
- `ldmatrix` 默认按 **col-major** 语义加载，所以 B 需要 `.trans` 变体
- `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16`：`row.col` 指 A row-major / B col-major 的输入布局
- 寄存器 fragment 布局（务必背下来）：A = 4×32bit（8 个 half），B = 2×32bit（4 个 half），C = 4×float（4 个元素）
- 每个 lane 在 C 中持有 `c[0..3]`，对应 (row, col) 的具体映射决定 epilogue 怎么写回

**自检问题**

- [ ] 为什么 `ldmatrix` 的地址只需要 lane 0–15 提供？（16 行 × 每行 16B）
- [ ] `.trans` 在 TN 布局里出现在 A 还是 B？NN 布局里呢？
- [ ] m16n8k16 里 N=8，为什么不是 16？（与 ldmatrix.x2 / 寄存器数量的关系）
- [ ] 一个 lane 的 C fragment 4 个元素在 C[m][n] 中的坐标是什么？（写 epilogue 前必须答得出）
- [ ] 为什么 `mma2x4_warp4x4` 的 padding 取 8？（代码 355-356 行有注释：`trust the profile`）

**动手验证**

```bash
cd kernels/hgemm
export TORCH_CUDA_ARCH_LIST=Ada
python3 hgemm.py --M 4096 --N 4096 --K 4096 --i 10 --mma-all
```

> [!warning] naive MMA 未进默认 benchmark
> `hgemm_mma_m16n8k16_naive` 在 `hgemm_mma.cu:307` 有绑定，但 `hgemm.py` 的列表里只有 `..._mma2x4_warp4x4`（647 行起）。
> 想单独跑 naive 版，在 `hgemm.py` 里补一行 `run_benchmark(hgemm.hgemm_mma_m16n8k16_naive, a, b, "(mma_naive)", c)`。

**验收标准**

- [ ] 能手写画出 A/B/C 三个 fragment 的 lane → 元素映射表
- [ ] 数值与 cuBLAS 一致
- [ ] 能解释 naive 版为什么慢（每 warp 只算 16×8，数据复用率极低）

> [!note] 跳过
> `hgemm/mma/basic/hgemm_mma_stage.cu`（2677 行、11 个变体）**不要通读**，只用来 diff 差异。

---

### Step 4 — multi-stage 流水 + TN 布局 + block swizzle

**目标**：把 Step 1 的 `cp.async` 和 Step 3 的 MMA 合起来，形成可参数化 stage 数的流水线。

**读什么（只读 1 个文件）**

| 文件:行 | 内容 | 为什么读 |
|---|---|---|
| `kernels/hgemm/mma/basic/hgemm_mma_stage_tn.cu:25-47` | `cp.async` / bulk 宏 | 与 Step 1 相同 |
| `kernels/hgemm/mma/basic/hgemm_mma_stage_tn.cu:124-386` | `hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel` | **主读目标**：629 行文件里唯一的 kernel |
| `kernels/hgemm/mma/basic/hgemm_mma_stage_tn.cu:388-416` | launch wrapper | 看 stage / block swizzle 怎么传 |
| `kernels/interview/hgemm.cuh:52-387` | Phase 7b-2 注释版 | 同实现的统一循环 + 中文讲解 |
| `kernels/interview/hgemm.cuh:399-716` | Phase 7b-3 注释版 | 带 XOR swizzle + 寄存器双缓冲 |

**新增概念要点**

- `K_STAGE` 模板参数：`CP_ASYNC_WAIT_GROUP(K_STAGE - 2)`，s2→0、s3→1、s4→2
- **dsmem**（double smem）：计算与搬运使用不同 smem 分区，主循环只需一次 `__syncthreads()`
- **TN 布局**：A row-major `[M,K]`、B 以 `B^T [N,K]` row-major 存储；`ldmatrix` 无需 `.trans`，天然匹配 `row.col`
- **Block swizzle**：把 `(bx, by)` 重排成 3D grid，让同一 z slice 内的 block 共享 L2 上的 A/B tile
- 尾部 k tile 的处理：条件化 `cp.async`（仅当 `k + K_STAGE - 1 < NUM_K_TILES`）

**自检问题**

- [ ] stage 从 2 加到 3/4，收益来自哪里？为什么不是越多越好（smem 占用 vs occupancy）？
- [ ] dsmem 与 multi-stage 是一回事吗？各自解决什么？
- [ ] TN 布局下 B 的 smem 索引怎么算？为什么内维连续的是 K？
- [ ] block swizzle 的 `swizzle_stride`（如 2048）如何影响 L2 命中？
- [ ] 主循环里为什么只需要一次 `__syncthreads()`？

**动手验证**

```bash
cd kernels/hgemm
make mma_tn_89            # 或 mma_tn_89_debug 打开 HGEMM_MMA_DEBUG
./hgemm_mma_stage_tn.89.bin 4096 4096 4096 1 10   # M N K warmup repeat
```

**验收标准**

- [ ] 数值正确；TFLOPS 相比 Step 3 的 naive/pad 版明显提升
- [ ] 能对比 `STAGES=2/3/4` 的实测曲线并解释拐点
- [ ] 能用 ncu 看到 L2 hit rate 因 block swizzle 改善

> [!note] 跳过
> `hgemm_mma_stage.cu`（NN 版）只在需要 diff `_dsmem` / `_x2` / `_rr` 时局部查看，不逐变体读。

---

### Step 5 — SMEM swizzle（★性价比最高）

**目标**：把「padding」这个权宜手段换成工业界的 XOR 置换，并用 ncu 量化到 0 bank conflict。

**读什么（从最小对照开始）**

| 文件:行 | 内容 | 为什么读 |
|---|---|---|
| `kernels/swizzle/mat_trans_swizzle.cu:15-77` | naive / padding / swizzle 三个 kernel（共 112 行） | **最小完整对照**，`s_data[x][x ^ y]` 就是全部魔法 |
| `kernels/swizzle/hgemm_mma_swizzle.cu:80-306` vs `307-...` | `hgemm_mma_m16n8k16_naive_kernel` vs `..._naive_smem_swizzle_kernel` | 真实 GEMM 上的 A/B 对照 |
| `kernels/swizzle/mma_simple_swizzle.cu:79-92` | `swizzle_j(i, j)` 与 `mma_simple_swizzle_kernel` | 241 行，最短的「如何给 MMA 加 swizzle」 |
| `kernels/swizzle/print_swizzle_layout.py` | 布局可视化 | 把 swizzle 结果画成 bank 图 |
| `kernels/swizzle/README.md` | ncu 命令与实测数据 | 直接给出验收指标 |

**新增概念要点**

- XOR swizzle：逻辑列 `col` 映射到物理列 `col ^ (row % 8) * 8`（`kColStride=16`、`kStep=8`）
- swizzle 是**无损置换**：不浪费 smem、不影响 occupancy，这是它优于 padding 的根本原因
- 对 `ldmatrix` 尤其关键：LDSM 的 8×8 分块访问最容易撞 bank
- `ldmatrix` 的地址也要跟着做同一个置换，否则读出来的是错位的

**自检问题**

- [ ] 为什么 `x ^ y` 能消掉 conflict？对 32 bank / 4B 宽的具体推演是什么？
- [ ] 为什么 padding 会影响 occupancy，而 swizzle 不会？
- [ ] swizzle 的 XOR 位宽（`kStep=8`）由什么决定？（与 MMA atom 的 K=16、ldmatrix 的 16B 行宽有关）
- [ ] 存储端和读取端必须使用同一个 swizzle 函数，否则会怎样？

**动手验证**

```bash
cd kernels/swizzle
make                      # 或 make mat_89 / hgemm_89 / mma_89
./mat_trans_swizzle.bin 1024 1024
./hgemm_mma_swizzle.bin 4096 4096 4096 1 10

# 关键验收指标：LDSM 的 bank conflict 必须为 0
ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm \
    ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld \
    ./mat_trans_swizzle.bin
```

**验收标准**

- [ ] `..._op_ldsm` 从 ~2.3e6 降到 **0**（README 第 26-71 行有完整对照日志）
- [ ] RTX 3080 上 `hgemm_mma_swizzle.bin 4096 4096 4096` 从 34.98 → 58.61 TFLOPS（README 第 78-85 行）
- [ ] 能用 `print_swizzle_layout.py` 解释置换后的 bank 分布

> [!note] 跳过
> `kernels/hgemm/mma/swizzle/*` 是工程版，先看 `kernels/swizzle/*` 教学版。

---

### Step 6 — 工程版收尾：TN + swizzle + 寄存器双缓冲 / CuTe

**目标**：看清「接近 cuBLAS」的完整组合，以及 CuTe 是如何把这一堆手工细节自动化的。

**读什么（二选一或都看）**

| 文件:行 | 内容 | 说明 |
|---|---|---|
| `kernels/hgemm/mma/swizzle/hgemm_mma_stage_tn_swizzle_x2.cu` | TN + swizzle + 寄存器双缓冲（x2） | 894 行；x4（966 行）只是倍数不同，**读一个即可** |
| `kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu:8-208` | CuTe kernel 本体（~200 行） | 用 Layout/Swizzle/Pipeline 自动生成，达到 cuBLAS 98–100% |
| `kernels/hgemm/cutlass/hgemm_mma_stage_tn_cute.cu:211-397` | launch wrapper | TiledMMA / SmemLayout / Pipeline 的组装 |
| `kernels/interview/hgemm.cuh:717-1195` | Phase 7c CuTe 注释版 | 中文讲解「手写 MMA vs CuTe 的分工」 |

**新增概念要点**

- 寄存器双缓冲：把 C 的累加与 A/B 的 ldmatrix 结果再做一层软件流水
- CuTe 的四层抽象：Layout（形状+步长）→ TiledCopy（线程-数据映射）→ TiledMma（MMA atom 组合）→ Pipeline（cp.async + stage）
- CuTe 的 swizzle 用 `<3,3,3>` 这类 Swizzle 模板参数表达，等价于手写 XOR
- 结论性认识：手写 MMA 让你知道「发生了什么」，CuTe 让你不必重复写它

**自检问题**

- [ ] 手写版里哪些代码在 CuTe 版里「消失了」？各自对应 CuTe 的哪个抽象？
- [ ] CuTe 的 `SmemLayout` + Swizzle 与手写 `swizzle_j()` 是否等价？怎么验证？
- [ ] 为什么 CuTe 版能超过手写 MMA 版？（pipeline 调度 + 更细的 epilogue）
- [ ] x2 / x4 的寄存器双缓冲分别适用什么 tile 尺寸？

**动手验证**

```bash
cd kernels/hgemm
make                                  # 产出 hgemm_mma_stage_tn_swizzle_x2.bin / hgemm_cute.bin
./hgemm_mma_stage_tn_swizzle_x2.89.bin 16384 16384 16384 1 10
./hgemm_cute.bin

# Python 侧对照（含 CuTe TN 版）
python3 hgemm.py --cute-tn --mma --plot
```

**验收标准**

- [ ] CuTe TN 版达到 cuBLAS 的 98–100%（L20 上约 116–117 TFLOPS，README 第 167-175 行）
- [ ] 能逐条说出「手写代码 → CuTe 抽象」的对应关系

---

### 可选 — WGMMA + TMA + warp specialization（仅 sm90a / Hopper）

**读什么**

| 文件 | 说明 |
|---|---|
| `kernels/hgemm/wgmma/hgemm_wgmma_fp32acc_stages_tn.cu`（559 行） | fp32 累加；**与 fp16acc 版二选一** |
| `kernels/hgemm/wgmma/hgemm_wgmma_fp16acc_stages_tn.cu`（545 行） | fp16 累加 |
| `kernels/ws-hgemm/naive_ws_hgemm_sm8x.cu`（486 行） | sm80 上的 warp specialization（不用 TMA），想理解 WS 又不换卡时看这个 |
| `kernels/interview/hgemm.cuh:1429-1862` | WGMMA + TMA + mbarrier 注释版 |

**新增概念**：warpgroup（128 线程）级 MMA、TMA（硬件 DMA，~零寄存器开销）、mbarrier 生产者-消费者同步、warp specialization（producer 搬运 / consumer 计算）。

> [!warning] 前置条件
> WGMMA 与 TMA 需要 **sm90a**，本地卡跑不了（见第 5 节）。

---

## 3. 明确跳过清单

| 跳过对象 | 原因 |
|---|---|
| `kernels/hgemm/naive/hgemm.cu`、`hgemm_async.cu` | f16 的 SIMT 版，思路与 SGEMM 完全同构；新东西只有 `_pack`（half2 打包 128-bit），扫一眼 `hgemm_t_8x8_sliced_k_f16x8_pack_bcf_kernel` 即可 |
| 参数扫描型变体：`t_8x4/8x8/8x16`、`k16/k32`、`mma4x2/mma4x4`、`warp2x4/warp4x4`、`x2/x4` | 同一份代码换常量，看 README 的 ALGO 表就够，不要逐文件读 |
| `_rr`（reduce registers）变体 | 为 flash-attn 大 head_dim 做的寄存器压缩微优化，对 GEMM 理解零增益 |
| `kernels/hgemm/cublas/*`、`kernels/sgemm/sgemm_cublas.cu` | 只是 baseline |
| `kernels/cutlass/cutlass-3.x/`、`cute_dsl/` | 可选；CuTe 的 `.cu` 版本更短更易读 |
| `kernels/hgemm/mma/basic/hgemm_mma_stage.cu` 全文 | 2677 行 / 11 变体，只用于 diff |

---

## 4. 时间紧时的最小闭环（3 步）

如果只有 2–3 天，按这个顺序，覆盖全部**真正的新概念**：

- [ ] **1. `cp.async`** — `kernels/sgemm/sgemm_async.cu:399-537`（一个 kernel）
- [ ] **2. MMA 白盒** — `kernels/hgemm/mma/basic/hgemm_mma.cu:80-161`（naive 一个 kernel）
- [ ] **3. swizzle** — `kernels/swizzle/mat_trans_swizzle.cu` + `hgemm_mma_swizzle.cu`

这三步之后，`_stages` / `_dsmem` / `_tn` / `_x2` / CuTe 都只是「参数与组合」，可以按需查阅。

---

## 5. 环境与架构矩阵

> [!warning] 本机环境
> 本机 `nvidia-smi` 不通；`lspci` 显示为 **GTX 1650 Mobile（TU117, sm_75）**。
> 该卡**只能**跑 WMMA fp16（m16n16k16）与 CUDA Core 版 SGEMM/HGEMM；
> `cp.async`、MMA m16n8k16、TF32、WGMMA 全部跑不了。
> 因此 Step 1/3/4/5/6 的验证需要在 sm80/86/89 的机器上做（仓库 bench 数据来自 L20 / RTX 4090 / RTX 3080 Laptop）。

| 概念 / 指令 | 最低架构 | 备注 |
|---|---|---|
| WMMA fp16 m16n16k16 | sm70（Volta） | 本机可跑 |
| `ldmatrix` | sm75 | 本机可跑（但喂不了 m16n8k16） |
| `cp.async` | sm80 | LDGSTS |
| MMA m16n8k16 | sm80 | 本清单核心 |
| WMMA TF32 m16n16k8 | sm80 | `sgemm_wmma_tf32_stage.cu` |
| WGMMA m64n128k16 | sm90a | 需 `-arch=compute_90a` |
| TMA（`cp.async.bulk`） | sm90 | makefile 里 `ARCHS_90` 用 `-std=c++20 -lcuda` |
| tcgen05（Blackwell） | sm100/120a | 本仓库 GEMM 未覆盖，见 `kernels/interview` |

---

## 6. 速查：构建与 benchmark 命令

```bash
# 仓库根
cd /media/lyxwxj/Data/common/Workspace/Omni-infra/kernels/Basic/LeetCUDA

# SGEMM（CUDA Core + TF32 WMMA），需 PyTorch
cd kernels/sgemm
export TORCH_CUDA_ARCH_LIST=Ada    # 或 Ampere
python3 sgemm.py

# HGEMM（WMMA / MMA / CuTe），需 PyTorch + CUTLASS submodule
cd kernels/hgemm
git submodule update --init --recursive --force
python3 hgemm.py --wmma-all --plot
python3 hgemm.py --mma-all --plot --topk 8
python3 hgemm.py --cute-tn --mma --plot
make && ./hgemm_mma_stage.bin
./hgemm_cute.bin
./hgemm_cublas.bin

# SM89 指定构建
make mma_tn_89 && ./hgemm_mma_stage_tn.89.bin 4096 4096 4096 1 10
make mma_tn_89_swizzle_x2 && ./hgemm_mma_stage_tn_swizzle_x2.89.bin 4096 4096 4096 1 10

# swizzle 教学
cd kernels/swizzle
make && ./mat_trans_swizzle.bin 1024 1024 && ./hgemm_mma_swizzle.bin 4096 4096 4096 1 10
ncu --metrics sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm ./hgemm_mma_swizzle.bin 1024 1024 1024 0 1

# 浓缩注释版（interview）
cd kernels/interview
./build.sh --arch sm_89
./notes_v2_sm89.bin --bench --mnk 4096,4096,4096
```

---

## 7. 参考

- 仓库笔记：[[LeetCuda]]
- CUTLASS/CuTe 笔记：[[01-概述与架构]]、[[05-Swizzle]]
- 仓库内文档：`kernels/sgemm/README.md`、`kernels/hgemm/README.md`、`kernels/swizzle/README.md`、`kernels/ws-hgemm/README.md`
- 外部：CUDA C Programming Guide（`cp.async` / `ldmatrix` / `mma.sync`）、[CUTLASS](https://github.com/NVIDIA/cutlass)、[toy-hgemm](https://github.com/xlite-dev/HGEMM)
