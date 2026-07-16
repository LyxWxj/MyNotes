## Part 2: Linear Algebra
### Basic Data Structures
**Scalar**: a single number
$$x = 5, \quad x \in \mathbb{R}$$
**Vector**: an ordered list of numbers
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix}, \quad \mathbf{v} \in \mathbb{R}^3$$
**Matrix**: a 2D array
$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \\ a_{31} & a_{32} \end{bmatrix}, \quad \mathbf{A} \in \mathbb{R}^{3 \times 2}$$
**Tensor**: a generalization to $n$ dimensions
- 0th-order tensor = scalar
- 1st-order tensor = vector
- 2nd-order tensor = matrix
- 3rd-order and above = higher-order tensors
**Dimension notation**: $\mathbb{R}^{m \times n}$ denotes a real matrix with $m$ rows and $n$ columns.
In ML frameworks (PyTorch, TensorFlow), all data is stored and computed as tensors.

---

### Tensor Dimensions: Visual Examples
**0D — Scalar**: a single value
$$x = 5$$
**1D — Vector**: a list of values
$$\mathbf{v} = [3, 1, 4, 1, 5]$$
**2D — Matrix**: grayscale image (height × width)
<GrayscaleTensor :rows="8" :cols="8" :cell-size="20" />
Shape: $8 \times 8$ (H × W)
**3D Tensor**: color image (channels x height × width)
Shape: $3 \times 8 \times 8$ (C × H x W)
**4D Tensor**: batch of color images
$$\text{Shape: } N \times C \times H \times W $$
- $N$: batch size (number of images)
- $H \times W$: spatial dimensions
- $C$: channels (3 for RGB, 4 for RGBA)
In PyTorch: `torch.Size([32, 4, 224, 224])` = 32 RGBA images of 224×224

---

### Geometric Representation of Vectors
<VectorChart />
A vector can be represented as a **directed line segment** from the origin:
$$\mathbf{v} = \begin{bmatrix} 3 \\ 1 \end{bmatrix}, \quad \mathbf{u} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$$
**Norm (length)**:
$$\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$$
**Unit vector**: a vector with norm 1, $\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|}$
In ML, a data sample (e.g. an image, a user profile) is typically represented as a high-dimensional vector.

---

### Vector Operations
**Addition**: component-wise sum
$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \end{bmatrix}$$
**Scalar multiplication**: multiply each component by the scalar
$$c\mathbf{v} = \begin{bmatrix} cv_1 \\ cv_2 \\ \vdots \end{bmatrix}$$
**Scalar addition**:
$$c+\mathbf{v} = \begin{bmatrix} c \\ c \\ \vdots \end{bmatrix}+\begin{bmatrix} v_1 \\ v_2 \\ \vdots \end{bmatrix}=\begin{bmatrix} c+v_1 \\ c+v_2 \\ \vdots \end{bmatrix}$$
**Dot product (inner product)**:
$$\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{n} u_i v_i = u_1 v_1 + u_2 v_2 + \cdots + u_n v_n$$
For example, with $\mathbf{u} = [u_1, u_2, u_3,u_4]$ and $\mathbf{v} = [v_1, v_2, v_3,v_4]$:
$$\mathbf{u} \cdot \mathbf{v} = u_1 v_1 + u_2 v_2 + u_3 v_3 + u_4 v_4$$
Geometric interpretation:
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta$$
Geometric interpretation of the dot product:
- $\mathbf{u} \cdot \mathbf{v} > 0$: angle < 90° (roughly same direction)
- $\mathbf{u} \cdot \mathbf{v} = 0$: **orthogonal** (perpendicular)
- $\mathbf{u} \cdot \mathbf{v} < 0$: angle > 90° (roughly opposite)
In ML, a single layer of a neural network is essentially a **dot product** of the input and weight vectors.

---

### Vector Transpose
A **column vector** becomes a **row vector** (and vice versa) by transposing:
$$\mathbf{v} = \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} \quad \Rightarrow \quad \mathbf{v}^T = \begin{bmatrix} v_1 & v_2 & v_3 \end{bmatrix}$$
**Properties**:
- $(\mathbf{v}^T)^T = \mathbf{v}$
- $(c\mathbf{v})^T = c\mathbf{v}^T$
- $(\mathbf{u} + \mathbf{v})^T = \mathbf{u}^T + \mathbf{v}^T$
**Why it matters**: Transpose converts between row and column vectors, which is essential for matrix multiplication and defining inner/outer products.

---

### Vector Multiplication: Row × Column
**Row vector × Column vector → Scalar (Dot Product)**
Given $\mathbf{a}, \mathbf{b} \in \mathbb{R}^k$ (both $k$-dimensional vectors):
$$\mathbf{a}^T \mathbf{b} = \begin{bmatrix} a_1 & a_2 & \cdots & a_k \end{bmatrix} \begin{bmatrix} b_1 \\ b_2 \\ \vdots \\ b_k \end{bmatrix} = a_1 b_1 + a_2 b_2 + \cdots + a_k b_k$$
Shape: $(1 \times k) \cdot (k \times 1) = 1 \times 1$
**Example**:
$$\begin{bmatrix} 1 & 2 & 3 \end{bmatrix} \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix} = 1 \times 4 + 2 \times 5 + 3 \times 6 = 32$$
Result: $1 \times 1$ scalar

---

### Vector Multiplication: Column × Row
**Column vector × Row vector → Matrix (Outer Product)**
Given $\mathbf{a} \in \mathbb{R}^m$ ($m$-dimensional) and $\mathbf{b} \in \mathbb{R}^n$ ($n$-dimensional):
$$\mathbf{a} \mathbf{b}^T = \begin{bmatrix} a_1 \\ a_2 \\ \vdots \\ a_m \end{bmatrix} \begin{bmatrix} b_1 & b_2 & \cdots & b_n \end{bmatrix} = \begin{bmatrix} a_1 b_1 & a_1 b_2 & \cdots & a_1 b_n \\ a_2 b_1 & a_2 b_2 & \cdots & a_2 b_n \\ \vdots & \vdots & \ddots & \vdots \\ a_m b_1 & a_m b_2 & \cdots & a_m b_n \end{bmatrix}$$
Shape: $(m \times 1) \cdot (1 \times n) = m \times n$
**Example**:
$$\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix} \begin{bmatrix} 4 & 5 & 6 \end{bmatrix} = \begin{bmatrix} 4 & 5 & 6 \\ 8 & 10 & 12 \\ 12 & 15 & 18 \end{bmatrix}$$
Result: $m \times n$ matrix

---

### Matrix-Vector Multiplication
For $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$:
$$\mathbf{A}\mathbf{x} = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} = \begin{bmatrix} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n \end{bmatrix}=\mathbf{y}\in \mathbb{R}^m$$
Shape: $(m \times n) \cdot (n \times 1) = m \times 1$
**Example**:
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} 1 \times 5 + 2 \times 6 \\ 3 \times 5 + 4 \times 6 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### Matrix-Vector Multiplication: Row Perspective
**A is partitioned by rows, each element of y is a dot product**
$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1^T \\ \mathbf{a}_2^T \\ \vdots \\ \mathbf{a}_m^T \end{bmatrix}, \quad \mathbf{y} = \mathbf{A}\mathbf{x} = \begin{bmatrix} \mathbf{a}_1^T \mathbf{x} \\ \mathbf{a}_2^T \mathbf{x} \\ \vdots \\ \mathbf{a}_m^T \mathbf{x} \end{bmatrix}$$
Each $y_i = \mathbf{a}_i^T \mathbf{x}$ is the dot product of the $i$-th row of $\mathbf{A}$ with $\mathbf{x}$.
**Example**:
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = \begin{bmatrix} \begin{bmatrix} 1 & 2 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} \\ \begin{bmatrix} 3 & 4 \end{bmatrix} \cdot \begin{bmatrix} 5 \\ 6 \end{bmatrix} \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### Matrix-Vector Multiplication: Column Perspective
**A is partitioned by columns, y is a weighted sum of columns**
$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_n \end{bmatrix}, x= \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}\quad \mathbf{y} = \mathbf{A}\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$
$\mathbf{y}$ is a linear combination of the columns of $\mathbf{A}$, with weights given by $\mathbf{x}$.
**Example**:
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 \\ 6 \end{bmatrix} = 5 \begin{bmatrix} 1 \\ 3 \end{bmatrix} + 6 \begin{bmatrix} 2 \\ 4 \end{bmatrix} = \begin{bmatrix} 5 \\ 15 \end{bmatrix} + \begin{bmatrix} 12 \\ 24 \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$$

---

### Matrix-Vector Multiplication: Geometry Perspective
Matrix-vector multiplication $\mathbf{y} = \mathbf{A}\mathbf{x}$ represents a **geometric transformation** of the vector $\mathbf{x}$:
- **Scaling**: stretch or shrink along axes
- **Rotation**: rotate vectors around the origin
- **Shearing**: skew the space
- **Reflection**: flip across an axis
Try different presets to see how the matrix transforms the blue vector into the amber one!
<LinearTransform />

---

### Matrix Multiplication
For $\mathbf{A} \in \mathbb{R}^{m \times k}$ and $\mathbf{B} \in \mathbb{R}^{k \times n}$, the product $\mathbf{C} = \mathbf{A}_{m\times k}\mathbf{B}_{k\times n} \in \mathbb{R}^{m \times n}$ is:
$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$
**View 1: Row × Column (Dot Product)**
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

### Matrix Multiplication: View 2
**View 2: Column × Row (Outer Product)**
$$\mathbf{A} = \begin{bmatrix} \mathbf{a}_1 & \mathbf{a}_2 & \cdots & \mathbf{a}_k \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} \mathbf{b}_1 \\ \mathbf{b}_2 \\ \vdots \\ \mathbf{b}_k \end{bmatrix}$$
$$\mathbf{C} = a_1 b_1+a_2 b_2 + \cdots + a_k b_k=\sum_{p=1}^{k} \mathbf{a}_p \mathbf{b}_p^T$$
**Example**:
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 7 & 9 \\ 6 & 8 & 10 \end{bmatrix} = \begin{bmatrix} 1{\times}5+2{\times}6 & 1{\times}7+2{\times}8 & 1{\times}9+2{\times}10 \\ 3{\times}5+4{\times}6 & 3{\times}7+4{\times}8 & 3{\times}9+4{\times}10 \end{bmatrix} = \begin{bmatrix} 17 & 23 & 29 \\ 39 & 53 & 67 \end{bmatrix}$$

---

### Matrix Multiplication Visualization
$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$
<MatrixMultiply :cell-size="50" />

---

### Matrix Transpose
The **transpose** of a matrix swaps rows and columns:
$$\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^T = \begin{bmatrix} a_{11} & a_{21} \\ a_{12} & a_{22} \\ a_{13} & a_{23} \end{bmatrix}$$
**Properties**:
- $(\mathbf{A}^T)^T = \mathbf{A}$
- $(\mathbf{A} + \mathbf{B})^T = \mathbf{A}^T + \mathbf{B}^T$
- $(c\mathbf{A})^T = c\mathbf{A}^T$
- $(\mathbf{A}\mathbf{B})^T = \mathbf{B}^T \mathbf{A}^T$ ← note the order reversal!
**Example**:
$$\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^T = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}$$

---

### Other Important Matrix Operations
Let $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$, $c = 2$
| Operation           | Definition                     | Example                                          |
| ------------------- | ------------------------------ | ------------------------------------------------ |
| **Transpose**       | $(A^T)_{ij} = A_{ji}$          | $\begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}$   |
| **Matrix Addition** | $(A+B)_{ij} = A_{ij} + B_{ij}$ | $\begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}$ |
| **Scalar Addition** | $(A+c)_{ij} = A_{ij} + c$      | $\begin{bmatrix} 3 & 4 \\ 5 & 6 \end{bmatrix}$   |

---

### Matrix Operations (cont.)
| Operation                   | Definition                               | Example                                            |
| --------------------------- | ---------------------------------------- | -------------------------------------------------- |
| **Element-wise (Hadamard)** | $(A \odot B)_{ij} = A_{ij} \cdot B_{ij}$ | $\begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$  |
| **Matrix Multiplication**   | $(AB)_{ij} = \sum_k A_{ik}B_{kj}$        | $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ |
**Important**: Matrix multiplication is **not commutative** in general: $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$
**Example**:
$$\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix} \neq \begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

---

### Special Matrices
**Identity Matrix** $\mathbf{I}$: $\mathbf{I}\mathbf{A} = \mathbf{A}\mathbf{I} = \mathbf{A}$
$$\mathbf{I}_3 = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$
**Diagonal Matrix**: nonzero entries only on the main diagonal
$$\mathbf{D} = \begin{bmatrix} d_1 & 0 & 0 \\ 0 & d_2 & 0 \\ 0 & 0 & d_3 \end{bmatrix}, \quad \mathbf{D}\mathbf{x} = \begin{bmatrix} d_1 x_1 \\ d_2 x_2 \\ d_3 x_3 \end{bmatrix}$$
**Symmetric Matrix**: $\mathbf{A} = \mathbf{A}^T$ (e.g. covariance matrix, Hessian)
$$\mathbf{S} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 5 & 4 \\ 3 & 4 & 6 \end{bmatrix}$$

---

**Upper Triangular Matrix**: all entries below the diagonal are zero
$$\mathbf{U} = \begin{bmatrix} u_{11} & u_{12} & u_{13} \\ 0 & u_{22} & u_{23} \\ 0 & 0 & u_{33} \end{bmatrix}$$
**Lower Triangular Matrix**: all entries above the diagonal are zero
$$\mathbf{L} = \begin{bmatrix} l_{11} & 0 & 0 \\ l_{21} & l_{22} & 0 \\ l_{31} & l_{32} & l_{33} \end{bmatrix}$$

---

**Orthogonal Matrix**: $\mathbf{Q}^T\mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}$, i.e. $\mathbf{Q}^{-1} = \mathbf{Q}^T$
- Columns (and rows) form an **orthonormal basis**
- Preserves lengths and angles: $\|\mathbf{Q}\mathbf{x}\| = \|\mathbf{x}\|$
- $\det(\mathbf{Q}) = \pm 1$
**Example** (2D rotation):
$$\mathbf{R} = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$
In ML, orthogonal matrices are used in **PCA**, **QR decomposition**, and **orthogonal initialization** of neural networks.

---

**Positive Definite Matrix**: $\mathbf{x}^T\mathbf{A}\mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$
**Equivalent conditions**:
- All eigenvalues $\lambda_i > 0$
- All leading principal minors $> 0$
- Cholesky decomposition exists: $\mathbf{A} = \mathbf{L}\mathbf{L}^T$
**Example**:
$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad \mathbf{x}^T\mathbf{A}\mathbf{x} = 2x_1^2 + 2x_1x_2 + 2x_2^2 > 0$$
**Semi-definite**: $\mathbf{x}^T\mathbf{A}\mathbf{x} \geq 0$ (eigenvalues $\geq 0$, e.g. covariance matrix)
In ML: positive definite Hessians guarantee convexity; positive definite kernels (Gram matrices) ensure valid similarity measures.

---

### Systems of Linear Equations
A system of $m$ linear equations in $n$ unknowns can be written as $\mathbf{A}\mathbf{x} = \mathbf{b}$:
$$\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\ \vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m \end{cases}$$
**Example**:
$$\begin{cases} 2x + 3y = 8 \\ x - y = 1 \end{cases} \quad \Leftrightarrow \quad \begin{bmatrix} 2 & 3 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 8 \\ 1 \end{bmatrix}$$
Solution: $x = \frac{11}{5}, \; y = \frac{6}{5}$
**Three cases**:
- **Unique solution**: $\mathbf{A}$ is invertible (full rank)
- **No solution**: inconsistent system (overdetermined)
- **Infinitely many solutions**: underdetermined system

---

### Matrix Inverse
For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, the **inverse** $\mathbf{A}^{-1}$ satisfies:
$$\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$$
where $\mathbf{I}$ is the $n \times n$ identity matrix.
**Solving linear systems**: If $\mathbf{A}$ is invertible, then $\mathbf{A}\mathbf{x} = \mathbf{b}$ has the unique solution:
$$\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$$
**Properties**:
- $(\mathbf{A}^{-1})^{-1} = \mathbf{A}$
- $(\mathbf{A}\mathbf{B})^{-1} = \mathbf{B}^{-1}\mathbf{A}^{-1}$
- $(\mathbf{A}^T)^{-1} = (\mathbf{A}^{-1})^T$
**Example** (2×2 matrix):
$$\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \Rightarrow \quad \mathbf{A}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

---

### Determinant
The **determinant** of a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ is a scalar that indicates whether $\mathbf{A}$ is invertible:
$$\det(\mathbf{A}) \neq 0 \quad \Leftrightarrow \quad \mathbf{A} \text{ is invertible}$$
**2×2 matrix**:
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$
**Geometric interpretation**: $|\det(\mathbf{A})|$ is the scaling factor of the transformation; the sign indicates orientation.
- $\det(\mathbf{A}) > 0$: preserves orientation
- $\det(\mathbf{A}) < 0$: reverses orientation
- $\det(\mathbf{A}) = 0$: collapses space (singular)

---

### Matrix Rank
The **rank** of a matrix $\mathbf{A}$ is the dimension of the column space (or row space), i.e., the maximum number of linearly independent columns (or rows).
$$\text{rank}(\mathbf{A}) \leq \min(m, n)$$
**Full rank**: $\text{rank}(\mathbf{A}) = \min(m, n)$
- Square matrix with full rank ⟹ invertible
- Rectangular matrix with full rank ⟹ columns/rows are linearly independent
**Example**:
$$\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix} \quad \Rightarrow \quad \text{rank}(\mathbf{A}) = 1$$
The second row is $2 \times$ the first row, so only 1 linearly independent row.
In ML, rank reveals the **effective dimensionality** of data. Low-rank approximations are used in PCA and dimensionality reduction.

---

### Eigenvalues and Eigenvectors
For a square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$, a scalar $\lambda$ is an **eigenvalue** and a nonzero vector $\mathbf{v}$ is an **eigenvector** if:
$$\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$$
> [!tip] Intuition
> The matrix $\mathbf{A}$ only **scales** the eigenvector $\mathbf{v}$ by $\lambda$, without changing its direction.
**Finding eigenvalues**: Solve the characteristic equation:
$$\det(\mathbf{A} - \lambda\mathbf{I}) = 0$$
**Example**:
$$\mathbf{A} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}, \quad \det(\mathbf{A} - \lambda\mathbf{I}) = (2-\lambda)^2 - 1 = 0$$
Eigenvalues: $\lambda_1 = 3, \; \lambda_2 = 1$
In ML, eigenvalues are fundamental to **PCA**, **spectral clustering**, and understanding the **stability** of optimization algorithms.

---

### Jacobian Matrix
The **Jacobian matrix** of a function $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ contains all first-order partial derivatives:
$$\mathbf{J} = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \frac{\partial f_1}{\partial x_2} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \frac{\partial f_2}{\partial x_1} & \frac{\partial f_2}{\partial x_2} & \cdots & \frac{\partial f_2}{\partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \frac{\partial f_m}{\partial x_2} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$
**Example**: For $\mathbf{f}(x, y) = \begin{bmatrix} x^2 + y \\ xy \end{bmatrix}$:
$$\mathbf{J} = \begin{bmatrix} 2x & 1 \\ y & x \end{bmatrix}$$
In ML, the Jacobian is used in **backpropagation**, **normalizing flows**, and **implicit differentiation**.

---

### Hessian Matrix
The **Hessian matrix** of a scalar function $f: \mathbb{R}^n \to \mathbb{R}$ contains all second-order partial derivatives:
$$\mathbf{H} = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots & \frac{\partial^2 f}{\partial x_2 \partial x_n} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \frac{\partial^2 f}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix}$$
**Example**: For $f(x, y) = x^2 + 3xy + y^2$:
$$\mathbf{H} = \begin{bmatrix} 2 & 3 \\ 3 & 2 \end{bmatrix}$$
**Properties**:
- $\mathbf{H}$ is **symmetric** (Schwarz's theorem)
- **Positive definite** ⟹ local minimum
- **Negative definite** ⟹ local maximum
- **Indefinite** ⟹ saddle point
In ML, the Hessian is used in **second-order optimization** (Newton's method), **natural gradient**, and analyzing **loss landscape curvature**.