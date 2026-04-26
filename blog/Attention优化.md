# FlashAttention极简教程与实现

## 背景

FlashAttention是一种优化Transformer中注意力机制的算法，旨在提高计算效率和减少内存使用。与其他优化Attention算法相比，FlashAttention是一种精确计算Attention的算法，而不是近似计算。它通过重新组织计算流程，进而减少了GPU中全局内存的访问次数，从而提高计算效率。
本文试图以最简单的方式介绍，实现一个最简单的Flashattention算法(不考虑多头注意力)，并且只涉及简单的`Python`代码，得益于triton，基于Pytorch/Numpy的实现可以很快地使用triton重写，以便真正地在GPU上并行运行，你可以选择先看本文，再去看FlashAttention的原始论文，我会假设读者没有读过FlashAttention的论文，在符号的选择上尽可能贴近原论文，以便读者在两者之间不会产生混淆。

## 符号约定：

- `Q, K, V, O`: `Query`, `Key`, `Value`, `Ouput`矩阵，形状为`(batch_size, seq_len, d_model)` 实际上K的`seq_len`可以和`Q`的`seq_len`不一致，但是为了简化，我们假设它们一致。

- $m_{k,i}$: 第`k`行前`i`个元素的最大值。 
- $l_{k,i}$: 第`k`行前`i`个元素的指数的和。
- $S_{k,j}$: 第`k`个查询和第`j`个键乘积得到的分数。
- $o_{k}$: 第`k`个查询得到的输出。

## (Safe) Softmax:3-pass

我们以`Softmax`作为一个引子来介绍FlashAttention。
$$
Softmax(x) = [\frac{e^{x_1}}{\sum_{j}e^{x_j}}, \frac{e^{x_2}}{\sum_{j}e^{x_j}} , ... , \frac{e^{x_n}}{\sum_{j}e^{x_j}}]
$$
实际上为了数值稳定，我们通常会在Softmax中引入一个偏移量，通常是输入的最大值，这样可以避免指数函数的溢出问题，并且在数学的计算上是等价的，因为：
$$
\frac{e^{x_i}}{\sum_{j}e^{x_j}} = \frac{e^{x_i - m}}{\sum_{j}e^{x_j - m}}， m = max_{j}\{x_j\}
$$
所以最简单的实现方式就是：

```python
def softmax(x: nn.Tensor, o: nn.Tensor):
  ## x: (seq_len, d_k)
  seq_len, d_k = x.shape
  for b in range(seq_len): # 对于每个token
    m = -torch.inf
    for i in range(d_k):
      m = max(m, x[b, i])
    l = torch.zeros_like(m) # 初始化指数和
    exp_x = torch.zeros_like(x[b])
    for i in range(d_k):
      exp_x[b, i] = torch.exp(x[b, i] - m) 
      l += exp_x[b, i] # 更新指数和
    for i in range(d_k):
      a[b, i] = exp_x[b, i] / l # 计算softmax输出
  return a
```

上面用了三个循环来实现Softmax，因此叫做3-pass Softmax，使用torch或者numpy的向量化操作也不会改变三次循环的本质，我们将这个过程写成公式：
$$
  \begin{aligned}
    m = -\infty, l = 0, a = 0 \\
    \text{for } i=1,\ldots,d_k:& \\
    &\quad m \leftarrow \max(m, x_i) \\
    &\quad l \leftarrow l + e^{x_i - m} \\
    \text{for } i=1,\ldots,d_k:& \\
    &\quad a_i \leftarrow \frac{e^{x_i - m}}{l}
  \end{aligned}
$$
`m`和`l`都有递推的性质，我们定义:
$$
\begin{aligned}
m_i = \max(m_{i-1}, x_i)\\
m = \max(m_1, m_2, \ldots, m_{d_k}) = m_{d_k} \\
l_i = \sum_{j=1}^{i} e^{x_j - m} = l_{i-1} + e^{x_i - m}\\
l = \sum_{j=1}^{d_k} e^{x_j - m} = l_{d_k}
\end{aligned}
$$
必须遍历两次行向量是因为$l_i$的值必须依赖$m$,如果$l_i$的值不依赖于$m$而是依赖于$m_i$，那么我们就可以在一次循环中同时更新$m_i$和$l_i$，从而实现在线Softmax。

## Online Softmax: 2-pass

在线Softmax的核心思想是融合前两步计算，刚才说$m$和$l$都有递推的性质，每一项$m_i$都可以通过$m_{i-1}$计算出来，每一项$l_i$也一样，如果我们要压缩前两步计算，那我们就需要找到一种方法既可以保证结果正确，并且让$l_i$只依赖$m_i$，又可以保证不破坏$l_i$的递推性，不妨先把$l_i$的定义改成如下形式：
$$
l_i = \sum_{j=1}^{i} e^{x_j - m_i}
$$
那么当$i=d_k$时，$l_i=l_{d_k}$仍然表示整个序列的指数和，结果正确，$l_i$依附于$m_i$的计算，就像动态规划中的状态转移方程一样，现在我们要解决$l_i$如何用$l_{i-1}$表示的问题。
$$
\begin{aligned}
l_i &= \sum_{j=1}^{i} e^{x_j - m_i} \\
l_{i-1} &= \sum_{j=1}^{i-1} e^{x_j - m_{i-1}} \\
l_{i-1} \times e^{m_{i-1}} &= \sum_{j=1}^{i-1} e^{x_j} \\
l_{i-1} \times e^{m_{i-1}} + e^{x_i} &= \sum_{j=1}^{i} e^{x_j} \\
(l_{i-1} \times e^{m_{i-1}} + e^{x_i}) e^{-m_i} &= \sum_{j=1}^{i} e^{x_j - m_i} = l_i\\
l_{i-1}\times e^{m_{i-1} - m_i} + e^{x_i - m_i} &=l_i
\end{aligned}
$$
因此我们就得到了在线Softmax的递推公式：
$$
\begin{aligned}
m_i &= \max(m_{i-1}, x_i)\\
l_i &= l_{i-1}\times e^{m_{i-1} - m_i} + e^{x_i - m_i}
\end{aligned}
$$
代码是这样写的：

```python
def online_softmax(x: nn.Tensor, a: nn.Tensor):
  ## x: (seq_len, d_k)
  seq_len, d_k = x.shape
  for b in range(seq_len): # 对于每个token
    m, l = -torch.inf, 0
    for i in range(d_k):
      m_i = max(m, x[b, i])
      l_i = l * torch.exp(m - m_i) + torch.exp(x[b, i] - m_i)
      m, l = m_i, l_i
    for i in range(d_k):
      a[b, i] = torch.exp(x[b, i] - m) / l
  return a
```

很遗憾不能最后的O不能与上面的循环合并，因为在计算$o_i$时需要用到最终的$m$和$l$，所以我们只能在第二次循环中计算$o_i$，因此在线Softmax是2-pass的。你可能会疑惑，虽然少一次循环，但是实际上多计算了一次exp函数，速度也许会更慢，我的理解是，在CPU串行执行的时候确实可能会更慢，但是在GPU中一次性将一整行的数据读入高速的共享内存非常占空间，所以多一次循环往往意味着要多一次从全局内存（HBM，速度慢）中读取数据，而在一个循环中只需要暂存两个变量m和l,所有计算都在高速缓存或共享内存中完成，多计算一次exp函数的开销对于一次HBM的访问开销来说是微不足道的。

## FlashAttention: 1-pass Single-query

为什么FlashAttention可以做1-pass呢，因为Attention的目标不是求Softmax，而是求最终的O。我们先来看看Attention的计算公式：
$$
\begin{aligned}
S &= softmax(\frac{QK^T}{\sqrt{d_k}}) \\
O &= SV \\
O &= softmax(\frac{QK^T}{\sqrt{d_k}})V
\end{aligned}
$$
对于Q中的每一个查询向量$q_k=Q[k,:] \in R^{d_k}$，我们最终计算得到对应的输出向量$o_k=O[k,:] \in R^{d_k}$。
计算过程如下：
$$
o_k = softmax(\frac{q_k K^T}{\sqrt{d_k}})V
$$
拆解每一步：
$$
\begin{aligned}
for \ i=1,\ldots,d_k:& \\
 & x_i \leftarrow \frac{q_k K^T[:, i]}{\sqrt{d_k}} \\
 & m_i \leftarrow \max(m_{i-1}, x_i)\\
 & l_i \leftarrow l_{i-1}\times e^{m_{i-1} - m_i} + e^{x_i - m_i} \\
end&\\
for i=1,\ldots,d_k:& \\
  & a_i \leftarrow \frac{e^{x_i - m}}{l} \\
  & o_i \leftarrow o_{i-1} + a_i \times V[i,:] \\
  end&\\
O[k, :] = o_{d_k}
\end{aligned}
$$
这是2-pass Attention的实现方式，这里的最后一步我觉得可能不太好理解，要说明的是实际上$a_i$是$S \in R^{seq_len \times seq_len}$ 中的第$k$行$i$列元素，而$o = O[k, ：] = S[k, :] \times V = \sum_{i=1}^{seq_len} S[k,i] \times V[i,:] = \sum_{i=1}^{seq_len} a_i \times V[i,:]$,因此可以用`o_i`表示$a \times V[i,:]$的前缀和，迭代更新。
$$
\begin{aligned}
o_i &= o_{i-1} + a_i \times V[i,:]
 = o_{i-1} + \frac{e^x_i - m}{l} \times V[i,:] \\
 &= \sum_{j=1}^{i} \frac{e^{x_j} - m}{l} \times V[j,:]
\end{aligned}
$$
我们的目标是将$o_i$的计算也融合到前面去，因此不能依赖于最终的$m$和$l$，我们需要将$o_i$的计算改成依赖于$m_i$和$l_i$，可以先将$o_i$的计算式子进行变形：
如果$o_i$的计算式子是：
$$
\begin{aligned}
o_i = \sum_{j=1}^{i} \frac{e^{x_j} - m_i}{l_i} \times V[j,:]\\
\end{aligned}
$$
那么
$$
\begin{aligned}
o_{i-1} &= \sum_{j=1}^{i-1} \frac{e^{x_j} - m_{i-1}}{l_{i-1}} \times V[j,:] \\
l_{i-1} \times o_{i-1} &= \sum_{j=1}^{i-1} e^{x_j - m_{i-1}} \times V[j,:] \\
l_{i-1} \times o_{i-1} \times e^{m_{i-1}} &= \sum_{j=1}^{i-1} e^{x_j} \times V[j,:] \\
l_{i-1} \times o_{i-1} \times e^{m_{i-1}} + e^{x_i} \times V[i,:] &= \sum_{j=1}^{i} e^{x_j} \times V[j,:] \\
&= l_i \times o_i \times e^{m_i} \\
\end{aligned}
$$
所以我们可以得到如下的递推公式：
$$
\begin{aligned}
o_i &= \frac{1}{l_i} \times e^{-m_i} \times (l_{i-1} \times o_{i-1} \times e^{m_{i-1}} + e^{x_i} \times V[i,:]) \\
&= \frac{1}{l_i}\times(e^{m_{i-1} - m_i} \times l_{i-1} \times o_{i-1} + e^{x_i - m_i} \times V[i,:])
\end{aligned}
$$
这样我们就可以将$o_i$的计算合并到第一个循环中去，最终的流程如下：
$$
\begin{aligned}
for \ i=1,\ldots,d_k:& \\
 & x_i \leftarrow \frac{q_k K^T[:, i]}{\sqrt{d_k}} \\
 & m_i \leftarrow \max(m_{i-1}, x_i)\\
 & l_i \leftarrow l_{i-1}\times e^{m_{i-1} - m_i} + e^{x_i - m_i} \\
  &o_i = \frac{1}{l_i}\times(e^{m_{i-1} - m_i} \times l_{i-1} \times o_{i-1} + e^{x_i - m_i} \times V[i,:])\\
end&\\
\end  {aligned}
$$
python代码如下：

```python
def flashattn_naiveQKV(Q, K, V):
  batch_size, seq_len, d_k = Q.shape
  # K: (batch_size, seq_len, d_k)
  # V: (batch_size, seq_len, d_k)
  scale = 1 / math.sqrt(d_k)
  O = torch.zeros(batch_size, seq_len, d_k, device=Q.device)
  for batch in range(batch_size):
    Q_, K_, V_ = Q[batch], K[batch], V[batch] # q, k, v: (seq_len, d_k)
    for k in range(seq_len): # for each query
      q = Q_[k, None] # (1,d_k)
      m0 = torch.Tensor([-torch.inf])
      l0 = torch.zeros(1)
      o0 = torch.zeros(1, d_k)
      for i in range(seq_len):
        k_i = K_[i,None].transpose(0,1)
        x_i = torch.matmul(q, k_i) * scale #  
        m_i = torch.maximum(m0, x_i)
        expmm = torch.exp(m0-m_i)
        expxm = torch.exp(x_i-m_i)
        l_i = l0*expmm + expxm
        o0 = (o0*(l0*expmm) + expxm * V_[i,None]) / l_i
        m0, l0 = m_i, l_i
      O[batch, k,None] = o0
  return O
```

在这里有一个问题，假设GPU会自动并行，让每一个执行单元处理一个查询（至于执行单元具体如何分配和执行这个查询任务我们暂时忽略），那么我们先看最内层循环:

```python
# for every block, do:
q = Q_[k, None] # (1,d_k) load one query
m0 = torch.Tensor([-torch.inf])
l0 = torch.zeros(1)
o0 = torch.zeros(1, d_k)
for i in range(seq_len):
  k_i = K_[i,None].transpose(0,1) # load one key
  x_i = torch.matmul(q, k_i) * scale #  
  m_i = torch.maximum(m0, x_i)
  expmm = torch.exp(m0-m_i)
  expxm = torch.exp(x_i-m_i)
  l_i = l0*expmm + expxm
  o0 = (o0*(l0*expmm) + expxm * V_[i,None]) / l_i # load one value
  m0, l0 = m_i, l_i
O[batch, k,None] = o0
```

假设`Q,K,V`都在全局内存（HBM）中，那么上面的代码，每个执行单元会读取一个查询，然后按批次读取所有的K和V矩阵的行向量，然后将输出写入，这个执行单元的HBM IO次数是：`sizeof(q) + sizeof(K) + sizeof(V) + sizeof(o)`, 其中`sizeof(q) = d_k`, `sizeof(K) = seq_len * d_k`, `sizeof(V) = seq_len * d_k`, `sizeof(o) = d_k`, 因此总的IO次数是`2*seq_len*d_k + 2*d_k`。
我们完全可以将里外层调换位置，如果我们先load `K[i,:]`和`V[i,:]`,然后让这个执行单元处理所有的查询，就像这样：

```python
def flashattn_naiveKVQ(Q, K, V):
  batch_size, seq_len, d_k = Q.shape
  # K: (batch_size, seq_len, d_k)
  # V: (batch_size, seq_len, d_k)
  scale = 1 / math.sqrt(d_k)
  O = torch.zeros(batch_size, seq_len, d_k, device=Q.device)
  for batch in range(batch_size):
    Q_, K_, V_ = Q[batch], K[batch], V[batch] # q, k, v: (seq_len, d_k)
    # 每个查询位置一个m值
    m0 = torch.full((seq_len,), -torch.inf)  
    # 每个查询位置一个l值
    l0 = torch.zeros(seq_len)      
    o0 = torch.zeros(seq_len, d_k)           # 每个查询位置一个输出向量
    for i in range(seq_len): # for each query
      k_i = K_[i,:]
      v_i = V_[i,:]
      for k in range(seq_len):
        q = Q_[k, None] # (1,d_k)
        x_i = torch.matmul(q, k_i) * scale #  
        m_i = torch.maximum(m0[k], x_i)
        expmm = torch.exp(m0[k]-m_i)
        expxm = torch.exp(x_i-m_i)
        l_i = l0[k]*expmm + expxm
        o_i = (l0[k]*(l0[k]*expmm) + expxm* v_i)/l_i
        m0[k], l0[k], o0[k] = m_i, l_i, o_i
    O[batch] = o0
  return O
```

问题在于每一次对`Q`的迭代我们只会更新整个`l0,m0,o0`一次，因此我们需要暂存所有的`l0,m0,o0`在全局内存(HBM),每个执行单元执行以下代码：

```python
k_i = K_[i,:] # load one key
v_i = V_[i,:] # load one value
for k in range(seq_len):
  q = Q_[k, None] # (1,d_k)
  x_i = torch.matmul(q, k_i) * scale #  
  m_i = torch.maximum(m0[k], x_i)
  expmm = torch.exp(m0[k]-m_i)
  expxm = torch.exp(x_i-m_i)
  l_i = l0[k]*expmm + expxm
  o_i = (l0[k]*(l0[k]*expmm) + expxm* v_i)/l_i
  m0[k], l0[k], o0[k] = m_i, l_i, o_i
```

总的IO次数是`sizeof(k_i) + sizeof(v_i) + sizeof(Q) + sizeof(l0) + sizeof(m0) = 2*d_k + seq_len*d_k + 2 * seq_len`,因此在`seq_len`和`d_k`较大时，这种方法可能会获得更好的性能，FlashAttentionV1也采用这种顺序来计算。
接下来我们再加入tiling的技巧，将`Q,K,V`分块处理，简单来说我们一次性处理`Br`个查询与`Bc`个键，值的计算，在论文中，`Br`,`Bc`的选取方法为：$Br = {\frac{SRAM}{4d_k}}$,$Bc = min(Br, d_k)$,其中SRAM是显卡中每个SM的L1缓存大小。

## FlashAttention: 1-pass Tiled

在之前的计算中我们每次迭代的元素跨度是1个，也就是每计算一个元素就更新一次`m,l,o`, 现在我们每次计算`Bc`个元素再更新一次`m,l,o`,这样的情况会让o的状态转移公式变得很复杂，但是好在在FlashAttentionV2中简化了这个状态转移的计算，因此我们直接跳入FlashAttentionV2。
仍然先考虑一个查询q,多个查询只是在以下的基础上多个行同时计算。
我们记$S^{(i)} \in R^{1 \times Bc}$是$q\in R^{1 \times d_k}$与$ks \in R^{Bc \times d_k}$的转置的乘积，$m_i$是$S^{(i)}$中的最大值，$l_i$是$e^{S^{(i)}-m_i}$的元素和，$o_i$是$e^{S^{(i)}-m_i}$的乘积，$o_i$的维度是$1 \times d_k$。查询$q$的完整的分数是$S = \{S^{(1)},S^{(2)},...,S^{(Bc)}\}$, $V \in R^{seq_len \times d_k}$被分为$V^{(i)} \in R^{Bc \times d_k}$
$$
\begin{aligned}
m^{(1)} &= max(S^{(1)}) \\
l^{(1)} &= \sum e^{S^{(1)}-m^{(1)}} \\
\hat O^{(1)} &= e^{S^{(1)} - m^{1}} V^{(1)T} \\
m^{(i)} &= max(m^{(i-1)}, max(S^{(i)}))  \\
l^{(i)} &= e^{m^{(i-1)} - m^{(i)}} + \sum e^{S^{(i)} - m^{(i)}}\\
\hat O^{(i)} &= e^{S^{(i-1)} - m^{(i)}} V^{(i-1)T} + e^{S^{(i)} - m^{(i)}} V^{(i)T}\\
&= O^{(i-1)} e^{m^{(i-1)} - m^{(i)}} + e^{S^{(i)} - m^{(i)}} V^{(i)T} \\
O &= \frac{1}{l}{\hat O}
\end{aligned}
$$
其中$l$, $\hat O$分别是最后一次迭代的$l^{(i)}$和$\hat O^{(i)}$, $m^{i}$, $l^{i}$,均为标量 $O \in R^{d_k}$是最终的输出。
现在我们考虑多个查询的情况：
$$
\begin{aligned}
m^{(1)} &= max(S^{(1)}) \in R^{Br} \\
l^{(1)} &= \sum e^{S^{(1)}-m^{(1)}} \in R^{Br} \\
\hat O^{(1)} &= e^{S^{(1)} - m^{1}} V^{(1)T} \in R^{Br \times d_k}\\
m^{(i)} &= max(m^{(i-1)}, max(S^{(i)}))  \\
l^{(i)} &= e^{m^{(i-1)} - m^{(i)}} + \sum e^{S^{(i)} - m^{(i)}}\\
\hat O^{(i)} &= e^{S^{(i-1)} - m^{(i)}} V^{(i-1)T} + e^{S^{(i)} - m^{(i)}} V^{(i)T}\\
&= O^{(i-1)} e^{m^{(i-1)} - m^{(i)}} + e^{S^{(i)} - m^{(i)}} V^{(i)T} \\
O &= \frac{1}{l}{\hat O}
\end{aligned}
$$
其中$l$, $\hat O$分别是最后一次迭代的$l^{(i)}$和$\hat O^{(i)}$, $m^{i}, l^{i} \in R^{Br}$, $O \in R^{Br \times d_k}$是最终的输出。
代码如下：

```python
SRAM_SIZE = 1024
def flashattnV1_block(Q, K, V):
  batch_size, seq_len, d_k = Q.shape
  scale = 1 / math.sqrt(d_k)
  # 计算块大小：Br基于SRAM大小和d_k计算
  Br = (SRAM_SIZE + 4 * d_k - 1) // (4 * d_k)
  Bc = min(Br, d_k)  # Bc应该基于序列长度而不是d_k
  O = torch.zeros(batch_size, seq_len, d_k, device=Q.device)

  for batch in range(batch_size):
    Q_, K_, V_ = Q[batch], K[batch], V[batch]
        
    # 处理每个查询块
    for q_start in range(0, seq_len, Br):
      q_end = min(q_start + Br, seq_len)
      q = Q_[q_start:q_end]  # (Br, d_k)
      Br_actual = q.shape[0]
            
        # 初始化累加器
      m_prev = torch.full((Br_actual,), -float('inf'), device=Q.device)
      l_prev = torch.zeros(Br_actual, device=Q.device)
      o_prev = torch.zeros(Br_actual, d_k, device=Q.device)
            
      # 处理每个KV块
      for kv_start in range(0, seq_len, Bc):
        kv_end = min(kv_start + Bc, seq_len)
        k_block = K_[kv_start:kv_end]  # (Bc_actual, d_k)
        v_block = V_[kv_start:kv_end]  # (Bc_actual, d_k)

        # 计算注意力分数: (Br_actual, Bc_actual)
        S = torch.matmul(q, k_block.transpose(-2, -1)) * scale
                
        # 找到当前块的最大值
        m_block = torch.max(S, dim=-1).values
                
        # 计算新的全局最大值
        m_new = torch.maximum(m_prev, m_block)
        
        exp_m_prev_minus_m_new = torch.exp(m_prev - m_new)
        exp_m_block_minus_m_new = torch.exp(S - m_new.unsqueeze(-1))
        
        # 更新累加器
        l_block = torch.sum(exp_m_block_minus_m_new, dim=-1)
        l_new = l_prev * exp_m_prev_minus_m_new + l_block
                
        # 更新输出累加器
        o_block = torch.matmul(exp_m_block_minus_m_new, v_block)
        o_new = o_prev * exp_m_prev_minus_m_new.unsqueeze(-1) + o_block
                
        # 更新状态
        m_prev, l_prev, o_prev = m_new, l_new, o_new
            
      O[batch, q_start:q_end] = o_prev / l_prev.unsqueeze(-1)
  return O
```

在flashAttentionV2中，状态转移的计算被简化了，并且由于批量计算Q,K,V的分块，不适合保存过大的中间状态，因此先迭代Q,然后迭代K,V。

这里也提供先迭代K,V再迭代Q的实现：

```python
def flashattnV2_blockKVQ(Q, K, V):
  batch_size, seq_len, d_k = Q.shape
  scale = 1 / math.sqrt(d_k)
  Br = (SRAM_SIZE + 4 * d_k - 1) // (4 * d_k)
  Bc = min(Br, d_k)
  O = torch.zeros(batch_size, seq_len, d_k, device=Q.device)
  for batch in range(batch_size):
    Q_, K_, V_ = Q[batch], K[batch], V[batch]
    
    # 初始化所有查询位置的累加器
    m_prev = torch.full((seq_len,), -float('inf'), device=Q.device)  # (seq_len,)
    l_prev = torch.zeros(seq_len, device=Q.device)                   # (seq_len,)
    o_prev = torch.zeros(seq_len, d_k, device=Q.device)             # (seq_len, d_k)
    
    # 外层循环：遍历KV块
    for kv_start in range(0, seq_len, Bc):
      kv_end = min(kv_start + Bc, seq_len)
      k_block = K_[kv_start:kv_end]  # (Bc_actual, d_k)
      v_block = V_[kv_start:kv_end]  # (Bc_actual, d_k)
      Bc_actual = k_block.shape[0]
      
      # 内层循环：遍历查询块
      for q_start in range(0, seq_len, Br):
        q_end = min(q_start + Br, seq_len)
        q = Q_[q_start:q_end]  # (Br_actual, d_k)
        Br_actual = q.shape[0]
        
        # 计算注意力分数: (Br_actual, Bc_actual)
        attn_scores = torch.matmul(q, k_block.transpose(-2, -1)) * scale
        
        # 获取当前查询块对应的累加器状态
        m_prev_block = m_prev[q_start:q_end]  # (Br_actual,)
        l_prev_block = l_prev[q_start:q_end]  # (Br_actual,)
        o_prev_block = o_prev[q_start:q_end]  # (Br_actual, d_k)
        
        # 找到当前块的最大值
        m_block = torch.max(attn_scores, dim=-1).values  # (Br_actual,)
        
        # 计算新的全局最大值
        m_new = torch.maximum(m_prev_block, m_block)
        
        # 计算指数项
        exp_m_prev_minus_m_new = torch.exp(m_prev_block - m_new)
        exp_m_block_minus_m_new = torch.exp(attn_scores - m_new.unsqueeze(-1))
        
        # 更新累加器
        l_block = torch.sum(exp_m_block_minus_m_new, dim=-1)  # (Br_actual,)
        l_new = l_prev_block * exp_m_prev_minus_m_new + l_block
        
        # 更新输出累加器
        o_block = torch.matmul(exp_m_block_minus_m_new, v_block)  # (Br_actual, d_k)
        o_new = o_prev_block * exp_m_prev_minus_m_new.unsqueeze(-1) + o_block
        
        # 更新累加器状态
        m_prev[q_start:q_end] = m_new
        l_prev[q_start:q_end] = l_new
        o_prev[q_start:q_end] = o_new
    
    # 所有KV块处理完后，进行最终归一化
    O[batch] = o_prev / l_prev.unsqueeze(-1)
  return O
```

## 总结
以上就是pytorch串行实现的FlashAttentionV2算法，虽然在CPU上效率不高，但是足以提供思路，使其很容易地可以重写成triton代码然后在GPU上执行。
FlashAttention最重要的思想就是从降低内存访问次数来提升速度出发，将需要依赖全局状态的计算通过类似于动态规划的方式进行迭代计算，从而减少迭代的次数，减少了对于全局内存的访问次数，提升了计算效率，这是对GPU性质的深刻洞察。