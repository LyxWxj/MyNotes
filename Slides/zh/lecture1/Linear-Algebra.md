## 第二部分：线性代数 (Linear Algebra)
### 基本数据结构
**标量 (Scalar)**：单个数字
$$x = 5, \quad x \in \mathbb{R}$$
**向量 (Vector)**：有序的数字列表
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}, \quad \mathbf{v} \in \mathbb{R}^3$$
**矩阵 (Matrix)**：二维数组
$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}, \quad \mathbf{A} \in \mathbb{R}^{3 \times 2}$$
**张量 (Tensor)**：向 $n$ 维的推广
- 0阶张量 = 标量 (Scalar)
- 1阶张量 = 向量 (Vector)
- 2阶张量 = 矩阵 (Matrix)
- 3阶及以上 = 高阶张量 (Higher-order Tensor)
**维度记法**：$\mathbb{R}^{m \times n}$ 表示一个具有 $m$ 行 $n$ 列的实数矩阵。
在机器学习框架（PyTorch、TensorFlow）中，所有数据都以张量的形式存储和计算。

---

### 张量维度：可视化示例
**0维 — 标量 (Scalar)**：单个值
$$x = 5$$
**1维 — 向量 (Vector)**：值的列表
$$\mathbf{v} = [3, 1, 4, 1, 5]$$
**2维 — 矩阵 (Matrix)**：灰度图像（高度 × 宽度）
<GrayscaleTensor :rows="8" :cols="8" :cell-size="20" />
形状 (Shape)：$8 \times 8$（H × W）
**3维张量 (3D Tensor)**：彩色图像（通道数 × 高度 × 宽度）
形状 (Shape)：$3 \times 8 \times 8$（C × H × W）
**4维张量 (4D Tensor)**：批量彩色图像
$$\text{Shape: } N \times C \times H \times W $$
- $N$：批量大小（Batch Size，图像数量）
- $H \times W$：空间维度
- $C$：通道数（Channels）（RGB为3，RGBA为4）
在 PyTorch 中：`torch.Size([32, 4, 224, 224])` = 32张 224×224 的 RGBA 图像

---

### 向量的几何表示
<VectorChart />
一个向量可以表示为从原点出发的**有向线段**：
$$\mathbf{v} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
**范数 (Norm)**（长度）：
$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$
**单位向量 (Unit Vector)**：范数为1的向量，$\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$
在机器学习中，一个数据样本（如一张图像、一个用户档案）通常被表示为高维向量。

---

### 向量运算
**加法 (Addition)**：逐分量求和
$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \end{bmatrix}$$
**标量乘法 (Scalar Multiplication)**：将每个分量乘以标量
$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \end{bmatrix}$$
**标量加法 (Scalar Addition)**：
$$c+\mathbf{v} = \begin{bmatrix} c \\ c \\ \vdots \end{bmatrix}+\begin{bmatrix} v_1 \\ v_2 \\ \vdots \end{bmatrix}=\begin{bmatrix} c+v_1 \\ c+v_2 \\ \vdots \end{bmatrix}$$
**点积 (Dot Product)**（内积 (Inner Product)）：
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$
例如，$\mathbf{u} = [u_1, u_2, u_3,u_4]$ 和 $\mathbf{v} = [v_1, v_2, v_3,v_4]$：
$$\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + u_3 v_3 + u_4 v_4$$
几何解释：
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$
点积的几何解释：
- $\mathbf{u} \cdot \mathbf{v} > 0$：夹角 < 90°（大致同向）
- $\mathbf{u} \cdot \mathbf{v} = 0$：**正交 (Orthogonal)**（垂直）
- $\mathbf{u} \cdot \mathbf{v} < 0$：夹角 > 90°（大致反向）
在机器学习中，神经网络的单层本质上就是输入向量和权重向量的**点积**。

---

### 向量转置
**列向量 (Column Vector)** 通过转置变为**行向量 (Row Vector)**（反之亦然）：
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} \quad \Rightarrow \quad \mathbf{v}^T = \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}$$
**性质**：
- $(\mathbf{v}^T)^T = \mathbf{v}$
- $(c\mathbf{v})^T = c\mathbf{v}^T$
- $(\mathbf{u} + \mathbf{v})^T = \mathbf{u}^T + \mathbf{v}^T$
**重要性**：转置在行向量和列向量之间转换，这对矩阵乘法以及定义内积/外积至关重要。

---

### 向量乘法：行 × 列
**行向量 × 列向量 → 标量（点积 (Dot Product)）**
给定 $\mathbf{a}, \mathbf{b} \in \mathbb{R}^k$（均为 $k$ 维向量）：
$$\mathbf{a}^T \mathbf{b} = \begin{bmatrix} a_1 & a_2 & \cdots & a_k \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_k \end{bmatrix} = a_1 b_1 + a_2 b_2 + \cdots + a_k b_k$$
形状：$(1 \times k) \cdot (k \times 1) = 1 \times 1$
**示例**：
$$\begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1 \times 4 + 2 \times 5 + 3 \times 6 = 32$$
结果：$1 \times 1$ 标量

---

### 向量乘法：列 × 行
**列向量 × 行向量 → 矩阵（外积 (Outer Product)）**
给定 $\mathbf{a} \in \mathbb{R}^m$（$m$ 维）和 $\mathbf{b} \in \mathbb{R}^n$（$n$ 维）：
$$\mathbf{a} \mathbf{b}^T = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_m \end{bmatrix} \begin{bmatrix} b_1 & b_2 & \cdots & b_n \end{bmatrix} = \begin{bmatrix} a_1 b_1 & a_1 b_2 & \cdots & a_1 b_n \\ a_2 b_1 & a_2 b_2 & \cdots & a_2 b_n \\ \vdots & \vdots & \ddots & \vdots \\ a_m b_1 & a_m b_2 & \cdots & a_m b_n \end{bmatrix}$$
形状：$(m \times 1) \cdot (1 \times n) = m \times n$
**示例**：
$$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \begin{bmatrix} 4 & 5 & 6 \end{bmatrix} = \begin{bmatrix} 4 & 5 & 6 \\ 8 & 10 & 12 \\ 12 & 15 & 18 \end{bmatrix}$$
结果：$m \times n$ 矩阵

---

### 矩阵-向量乘法
对于 $\mathbf{A} \in \mathbb{R}^{m \times n}$ 和 $\mathbf{x} \in \mathbb{R}^n$：
$$\mathbf{A}\mathbf{x} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n \end{bmatrix}=\mathbf{y}\in \mathbb{R}^m$$
形状：$(m \times n) \cdot (n \times 1) = m \times 1$
**示例**：
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 1 \times 5 + 2 \times 6 \\ 3 \times 5 + 4 \times 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### 矩阵-向量乘法：行视角
**A 按行划分，y 的每个元素是一个点积**
$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1^T \\ \mathbf{a}_2^T \\ \vdots \\ \mathbf{a}_m^T \end{bmatrix}, \quad \mathbf{y} = \mathbf{A}\mathbf{x} = \begin{bmatrix} \mathbf{a}_1^T \mathbf{x} \\ \mathbf{a}_2^T \mathbf{x} \\ \vdots \\ \mathbf{a}_m^T \mathbf{x} \end{bmatrix}$$
每个 $y_i = \mathbf{a}_i^T \mathbf{x}$ 是 $\mathbf{A}$ 的第 $i$ 行与 $\mathbf{x}$ 的点积。
**示例**：
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} \\ \begin{bmatrix} 3 & 4 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### 矩阵-向量乘法：列视角
**A 按列划分，y 是列的加权求和**
$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n \end{bmatrix}, x= \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}\quad \mathbf{y} = \mathbf{A}\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$
$\mathbf{y}$ 是 $\mathbf{A}$ 的列的线性组合 (Linear Combination)，权重由 $\mathbf{x}$ 给出。
**示例**：
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 3 \end{bmatrix} + 6 \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 5 \\ 15 \end{bmatrix} + \begin{bmatrix} 12 \\ 24 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### 矩阵-向量乘法：几何视角
矩阵-向量乘法 $\mathbf{y} = \mathbf{A}\mathbf{x}$ 表示向量 $\mathbf{x}$ 的**几何变换 (Geometric Transformation)**：
- **缩放 (Scaling)**：沿坐标轴拉伸或收缩
- **旋转 (Rotation)**：绕原点旋转向量
- **剪切 (Shearing)**：扭曲空间
- **反射 (Reflection)**：沿某轴翻转
尝试不同的预设，观察矩阵如何将蓝色向量变换为琥珀色向量！
<LinearTransform />

---

### 矩阵乘法
对于 $\mathbf{A} \in \mathbb{R}^{m \times k}$ 和 $\mathbf{B} \in \mathbb{R}^{k \times n}$，乘积 $\mathbf{C} = \mathbf{A}_{m\times k}\mathbf{B}_{k\times n} \in \mathbb{R}^{m \times n}$ 为：
$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$
**视角 1：行 × 列（点积 (Dot Product)）**
$$
\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 \\ \mathbf{a}_2 \\ \vdots \\ \mathbf{a}_m \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} \mathbf{b}_1 & \mathbf{b}_2 & \cdots & \mathbf{b}_n \end{bmatrix}
$$
$$
C = \begin{bmatrix}
\mathbf{a}_1\cdot \mathbf{b}_1 & \mathbf{a}_1\cdot \mathbf{b}_2 & \cdots & \mathbf{a}_1\cdot \mathbf{b}_n \\
\mathbf{a}_2\cdot \mathbf{b}_1 & \mathbf{a}_2\cdot \mathbf{b}_2 & \cdots & \mathbf{a}_2\cdot \mathbf{b}_n \\
\vdots & \vdots & \ddots & \vdots \\
\mathbf{a}_m\cdot \mathbf{b}_1 & \mathbf{a}_m\cdot \mathbf{b}_2 & \cdots & \mathbf{a}_m \cdot \mathbf{b}_n
\end{bmatrix}=\begin{bmatrix}Ab_1&Ab_2&\cdots&Ab_n\end{bmatrix}
$$

---

### 矩阵乘法：视角 2
**视角 2：列 × 行（外积 (Outer Product)）**
$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_k \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \\ \vdots \\ \mathbf{b}_k \end{bmatrix}$$
$$\mathbf{C} = a_1 b_1+a_2 b_2 + \cdots + a_k b_k=\sum_{p=1}^{k} \mathbf{a}_p \mathbf{b}_p^T$$
**示例**：
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 7 & 9 \\ 6 & 8 & 10 \end{bmatrix} = \begin{bmatrix} 1{\times}5+2{\times}6 & 1{\times}7+2{\times}8 & 1{\times}9+2{\times}10 \\ 3{\times}5+4{\times}6 & 3{\times}7+4{\times}8 & 3{\times}9+4{\times}10 \end{bmatrix} = \begin{bmatrix} 17 & 23 & 29 \\ 39 & 53 & 67 \end{bmatrix}$$

---

### 矩阵乘法可视化
$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$
<MatrixMultiply :cell-size="50" />

---

### 矩阵转置
矩阵的**转置 (Transpose)** 交换行和列：
$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^T = \begin{bmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \\ a_{13} & a_{23} \end{bmatrix}$$
**性质**：
- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$
- $(c\mathbf{A})^T = c\mathbf{A}^T$
- $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$ ← 注意顺序反转！
**示例**：
$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

---

### 其他重要的矩阵运算
设 $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$，$\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$，$c = 2$
| 运算                | 定义                             | 示例                                               |
| ------------------- | ------------------------------ | ------------------------------------------------ |
| **转置 (Transpose)**       | $(A^T)_{ij} = A_{ji}$          | $\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$   |
| **矩阵加法 (Matrix Addition)** | $(A+B)_{ij} = A_{ij} + B_{ij}$ | $\begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$ |
| **标量加法 (Scalar Addition)** | $(A+c)_{ij} = A_{ij} + c$      | $\begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix}$   |

---

### 矩阵运算（续）
| 运算                    | 定义                               | 示例                                             |
| --------------------------- | ---------------------------------------- | -------------------------------------------------- |
| **逐元素乘法 (Hadamard)** | $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$ | $\begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$  |
| **矩阵乘法 (Matrix Multiplication)**   | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$        | $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ |
**重要**：矩阵乘法一般**不满足交换律 (Not Commutative)**：$\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$
**示例**：
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix} \neq \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

---

### 特殊矩阵
**单位矩阵 (Identity Matrix)** $\mathbf{I}$：$\mathbf{I}\mathbf{A} = \mathbf{A}\mathbf{I} = \mathbf{A}$
$$\mathbf{I}_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
**对角矩阵 (Diagonal Matrix)**：仅主对角线上有非零元素
$$\mathbf{D} = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}, \quad \mathbf{D}\mathbf{x} = \begin{bmatrix} d_1 x_1 \\ d_2 x_2 \\ d_3 x_3 \end{bmatrix}$$
**对称矩阵 (Symmetric Matrix)**：$\mathbf{A} = \mathbf{A}^T$（如协方差矩阵 (Covariance Matrix)、海森矩阵 (Hessian)）
$$\mathbf{S} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 4 \\ 3 & 4 & 6 \end{bmatrix}$$

---

**上三角矩阵 (Upper Triangular Matrix)**：对角线以下所有元素为零
$$\mathbf{U} = \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}$$
**下三角矩阵 (Lower Triangular Matrix)**：对角线以上所有元素为零
$$\mathbf{L} = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix}$$

---

**正交矩阵 (Orthogonal Matrix)**：$\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$，即 $\mathbf{Q}^{-1} = \mathbf{Q}^T$
- 列（和行）构成**标准正交基 (Orthonormal Basis)**
- 保持长度和角度不变：$\|\mathbf{Q}\mathbf{x}\| = \|\mathbf{x}\|$
- $\det(\mathbf{Q}) = \pm 1$
**示例**（2D 旋转）：
$$\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
在机器学习中，正交矩阵用于**主成分分析 (PCA)**、**QR 分解 (QR Decomposition)** 和神经网络的**正交初始化 (Orthogonal Initialization)**。

---

**正定矩阵 (Positive Definite Matrix)**：对所有 $\mathbf{x} \neq \mathbf{0}$，$\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$
**等价条件**：
- 所有特征值 (Eigenvalue) $\lambda_i > 0$
- 所有顺序主子式 (Leading Principal Minor) $> 0$
- 存在 Cholesky 分解 (Cholesky Decomposition)：$\mathbf{A} = \mathbf{L}\mathbf{L}^T$
**示例**：
$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad \mathbf{x}^T\mathbf{A}\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2 > 0$$
**半正定 (Semi-definite)**：$\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0$（特征值 $\geq 0$，如协方差矩阵 (Covariance Matrix)）
在机器学习中：正定海森矩阵 (Hessian) 保证凸性；正定核函数 (Kernel)（格拉姆矩阵 (Gram Matrix)）确保有效的相似性度量。

---

### 线性方程组
$m$ 个 $n$ 元线性方程组可以写成 $\mathbf{A}\mathbf{x} = \mathbf{b}$：
$$\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m \end{cases}$$
**示例**：
$$\begin{cases} 2x + 3y = 8 \\ x - y = 1 \end{cases} \quad \Leftrightarrow \quad \begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 \\ 1 \end{bmatrix}$$
解：$x = \frac{11}{5}, \; y = \frac{6}{5}$
**三种情况**：
- **唯一解 (Unique Solution)**：$\mathbf{A}$ 可逆（满秩 (Full Rank)）
- **无解 (No Solution)**：不相容方程组（超定 (Overdetermined)）
- **无穷多解 (Infinitely Many Solutions)**：欠定方程组 (Underdetermined)

---

### 矩阵逆
对于方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$，**逆矩阵 (Inverse)** $\mathbf{A}^{-1}$ 满足：
$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$
其中 $\mathbf{I}$ 是 $n \times n$ 单位矩阵。
**求解线性方程组**：如果 $\mathbf{A}$ 可逆，则 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 有唯一解：
$$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$
**性质**：
- $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
- $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
- $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$
**示例**（2×2 矩阵）：
$$\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

---

### 行列式
方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$ 的**行列式 (Determinant)** 是一个标量，用于判断 $\mathbf{A}$ 是否可逆：
$$\det(\mathbf{A}) \neq 0 \quad \Leftrightarrow \quad \mathbf{A} \text{ 可逆}$$
**2×2 矩阵**：
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$
**几何解释**：$|\det(\mathbf{A})|$ 是变换的缩放因子；符号表示方向。
- $\det(\mathbf{A}) > 0$：保持方向
- $\det(\mathbf{A}) < 0$：反转方向
- $\det(\mathbf{A}) = 0$：空间坍缩（奇异 (Singular)）

---

### 矩阵的秩
矩阵 $\mathbf{A}$ 的**秩 (Rank)** 是列空间（或行空间）的维度，即线性无关列（或行）的最大数量。
$$\text{rank}(\mathbf{A}) \leq \min(m, n)$$
**满秩 (Full Rank)**：$\text{rank}(\mathbf{A}) = \min(m, n)$
- 满秩方阵 ⟹ 可逆
- 满秩矩形矩阵 ⟹ 列/行线性无关
**示例**：
$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix} \quad \Rightarrow \quad \text{rank}(\mathbf{A}) = 1$$
第二行是第一行的 2 倍，因此只有 1 个线性无关行。
在机器学习中，秩揭示了数据的**有效维度 (Effective Dimensionality)**。低秩近似 (Low-rank Approximation) 用于主成分分析 (PCA) 和降维 (Dimensionality Reduction)。

---

### 特征值与特征向量
对于方阵 $\mathbf{A} \in \mathbb{R}^{n \times n}$，标量 $\lambda$ 是**特征值 (Eigenvalue)**，非零向量 $\mathbf{v}$ 是**特征向量 (Eigenvector)**，如果：
$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$
> [!tip] 直觉理解
> 矩阵 $\mathbf{A}$ 仅将特征向量 $\mathbf{v}$ 缩放 $\lambda$ 倍，而不改变其方向。
**求特征值**：求解特征方程 (Characteristic Equation)：
$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$
**示例**：
$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad \det(\mathbf{A} - \lambda\mathbf{I}) = (2-\lambda)^2 - 1 = 0$$
特征值：$\lambda_1 = 3, \; \lambda_2 = 1$
在机器学习中，特征值是**主成分分析 (PCA)**、**谱聚类 (Spectral Clustering)** 以及理解优化算法**稳定性 (Stability)** 的基础。

---

### 雅可比矩阵
函数 $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ 的**雅可比矩阵 (Jacobian Matrix)** 包含所有一阶偏导数：
$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$
**示例**：对于 $\mathbf{f}(x, y) = \begin{bmatrix} x^2 + y \\ xy \end{bmatrix}$：
$$\mathbf{J} = \begin{bmatrix} 2x & 1 \\ y & x \end{bmatrix}$$
在机器学习中，雅可比矩阵用于**反向传播 (Backpropagation)**、**标准化流 (Normalizing Flows)** 和**隐式微分 (Implicit Differentiation)**。

---

### 海森矩阵
标量函数 $f: \mathbb{R}^n \to \mathbb{R}$ 的**海森矩阵 (Hessian Matrix)** 包含所有二阶偏导数：
$$\mathbf{H} = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$
**示例**：对于 $f(x, y) = x^2 + 3xy + y^2$：
$$\mathbf{H} = \begin{bmatrix} 2 & 3 \\ 3 & 2 \end{bmatrix}$$
**性质**：
- $\mathbf{H}$ 是**对称的 (Symmetric)**（施瓦茨定理 (Schwarz's Theorem)）
- **正定 (Positive Definite)** ⟹ 局部最小值 (Local Minimum)
- **负定 (Negative Definite)** ⟹ 局部最大值 (Local Maximum)
- **不定 (Indefinite)** ⟹ 鞍点 (Saddle Point)
在机器学习中，海森矩阵用于**二阶优化 (Second-order Optimization)**（牛顿法 (Newton's Method)）、**自然梯度 (Natural Gradient)** 和分析**损失曲面曲率 (Loss Landscape Curvature)**。