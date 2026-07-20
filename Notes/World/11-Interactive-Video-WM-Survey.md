# Towards Interactive Video World Modeling: Survey (2026)

> **论文**: Towards Interactive Video World Modeling: Frontiers, Challenges, Benchmarks, and Future Trends
> **作者**: Cambridge, Peking, UC Berkeley, ETH Zurich, Oxford 等
> **来源**: arXiv 2606.01164
> **分类**: cs.CV

---

## 一、核心定义

交互式世界模型 = 用户通过动作控制世界状态演化的系统

**POMDP 形式化**：$(S, A, O, T, R)$，转移函数描述 $p(s_{t+1}|s_t, a_t)$

### 三个定义性特征

1. **多轮细粒度可控性**：帧级或区域级人在回路控制
2. **长水平一致世界转移**：跨交互轮次保持连贯场景演化
3. **实时交互与反馈**：即时在线视觉反馈

---

## 二、五种交互接口

| 类型 | 示例 |
|------|------|
| **视觉** | 图像、草图、键盘/鼠标 |
| **文本** | T5 编码器 |
| **音频** | 语音命令 |
| **物理** | 力、重力场 |
| **其他** | 遥操作 |

---

## 三、研究趋势

### 从专用到通用

- 早期方法针对单一领域
- 近期使用大规模预训练视频骨干（如 Wan 2.2）
- UniSim 混合多样数据集
- AdaWorld 自监督提取潜在动作
- Astra 使用动作混合专家

### 从静态到动态、自演化、多智能体世界

**静态 → 动态**：
- WonderWorld：时间冻结环境
- WorldCanvas/NeoVerse/VerseCrafter：每对象动力学 + 4D 建模
- LiveWorld：全局自演化状态处理"视线外动态"

**单智能体 → 多智能体**：
- Solaris：共享自注意力跨智能体交换
- MultiWorld/AgentParty/Combo：部分自我中心观察的多智能体合作

### 从单感官到多感官接口

- SonoWorld：整合音频输出
- 物理信息工作：嵌入力/重力场
- 自动驾驶：结构化接口（自车速度、HD 地图、多摄像机观察）

---

## 四、动作可控性

### 一次性控制（传统视频生成）

- 标准 T2V/I2V：一次性条件化所有帧
- 弱化时间因果性
- 摄像机轨迹方法：预定义视点采样

### 多次细粒度控制（世界模型）

- 每帧动作条件生成
- iVideoGPT：AR Transformer 交互视频预测
- UniSim：重叠块生成
- WorldCanvas：拖放
- NeoVerse：4D 建模
- VerseCrafter：每对象高斯轨迹

### 动作注入方式

#### 摄像机姿态/轨迹注入

| 类别 | 方法 |
|------|------|
| 与视觉 token 拼接 | Genie, iVideoGPT, AdaWorld, GameFactory |
| 缩放和偏移 | GameGen-X, HY-WorldPlay |
| 摄像机控制渲染/模拟 | Vmem, LiveWorld, SWM |
| 矩阵变换 | IaaW, GenEx |

#### 文本指令注入

- T5 编码器 + 交叉注意力
- 挑战：一致性 vs 动作跟随平衡

---

## 五、长水平交互与记忆

### 历史帧作为条件

- 大多数方法自回归使用近期帧
- UniSim：重叠块生成（最后 4 帧 + 噪声样本拼接）
- 问题：固定长度历史帧的有限时间窗口

### 记忆构建

#### 隐式 3D 记忆（视频潜在 token）

| 方法 | 特点 |
|------|------|
| WorldMem | token 级记忆库 + 状态感知交叉注意力检索 |
| VRAG | 基于相似性的位置感知状态/动作/帧三元组检索 |
| RELIC | 未压缩近期 KV Cache + 压缩长水平空间记忆 |
| HY-WorldPlay | 混合短期时间 + 长期空间记忆 + 时间重帧 |
| WorldCam | 姿态锚定长期记忆 |
| HM-World | 混合记忆：档案管理员（静态背景）+ 警觉跟踪器（动态主体） |

#### 显式 3D 记忆

| 方法 | 特点 |
|------|------|
| Vmem | Surfel 索引视图记忆 |
| DeepVerse | 几何相似过去状态检索 |
| Spmem | 空间记忆（增量更新静态点图）+ 情景记忆 |
| MosaicMem | 混合显式和隐式优势 |

### 噪声增强与强迫训练

| 方法 | 特点 |
|------|------|
| Oasis | 动态噪声（多到少调度） |
| Diffusion Forcing | 用不同高斯噪声损坏历史帧 |
| Self Forcing | 从自生成展开蒸馏 |
| Geometry Forcing | 与预训练几何模型的 3D 特征对齐 |
| Context Forcing | 记忆增强自展开 |
| LIVE | 前向展开 + 反向恢复 + 循环一致性 |

---

## 六、实时交互性

### 连贯性 vs 动作跟随权衡

- 增加历史帧 → 更好一致性，但弱化动作响应
- Astra：向条件帧注入随机噪声
- HY-WorldPlay：Context Forcing + 记忆增强自展开

### 实时展开优化

#### 模型蒸馏

| 方法 | FPS |
|------|-----|
| GameNGen | 50 |
| HY-GameCraft | - |
| Matrix-Game 3.0 | 40 |

#### 缓存加速

- Yume：跨去噪步骤复用中间残差特征
- HY-World 1.0：缓存 + 多 GPU 并行化
- Matrix-Game 2.0, SSM-WM, RELIC

---

## 七、基准与评估

### 开放世界探索

| 基准 | 特点 |
|------|------|
| WorldScore | 统一评估可控性、保真度、时间动态 |
| OmniWorldBench | 4D 交互中心指标 |
| MIND | 记忆一致性 |
| iWorldBench | 混合数据源（机器人、驾驶、3D 重建、无人机） |

### 游戏引擎

- Atari, Minecraft, AAA 游戏, UE 环境
- GameWorld：标准化 MLLM 游戏智能体评估

### 具身 AI

- 数据引擎：GigaWorld-0, MVISTA-4D
- 平台：LIBERO, CALVIN, VLABench, RoboTwin 2.0
- RoboChallenge：首个真实世界评估基准

### 自动驾驶

- nuScenes：开环评估
- NAVSIM：伪模拟规划
- Bench2Drive：闭环驾驶

---

## 八、主导架构

**DiT 是主导架构**：~40 种方法使用

典型帧率：20-50 FPS（蒸馏后）

主要动作类型：摄像机姿态、键盘、文本、潜在动作

主要一致性策略：历史帧、记忆库、3D 重建、强迫训练

---

## 九、关键挑战

1. **误差复合**：长展开中误差累积
2. **连贯性-动作跟随权衡**：增加一致性削弱响应
3. **实时效率**：高质量下的效率
4. **物理感知**：真实物理动态
5. **多智能体合作**
6. **4D 一致性**：动态对象和视线外演化
