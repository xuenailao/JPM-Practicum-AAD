# Edge-Pushing理论 vs PDE实践：为什么Gamma=0

## 您的理论解释（完全正确）

### 理想情况

```
输入: x = [S, K, σ, r, T]
      ↓
  [PDE Solver]  ← 作为黑盒函数
      ↓
输出: V(期权价格)

Edge-Pushing应该能计算:
H = [[∂²V/∂S²,    ∂²V/∂S∂K,   ∂²V/∂S∂σ,   ...],
     [∂²V/∂K∂S,   ∂²V/∂K²,    ...         ...],
     [∂²V/∂σ∂S,   ...         ∂²V/∂σ²,    ...],
     [...                                   ]]

Γ = H[0,0] = ∂²V/∂S²  ✅
```

**这在理论上完美无缺！**

---

## 实际问题：S不在计算图中

### 我们的实现细节

```python
def solve_pde_with_aad(S0_val: float, sigma_val: float):
    """
    实际的PDE求解器

    关键问题：S0_val 只是一个float常数！
    """

    # ❌ 问题1: S0不是ADVar
    # S0_val = 100.0  # 只是普通float

    # ✅ σ是ADVar（在计算图中）
    sigma_var = ADVar(sigma_val, requires_grad=True)

    # PDE在固定的空间网格上求解
    S_grid = [0, 50, 100, 150, 200, 250]  # 固定网格

    # 初始条件（不依赖S0！）
    V_terminal = max(S_grid - K, 0)

    # 时间步进（所有操作都在网格上）
    for n in range(N):
        V = cn_step(V, sigma_var, ...)  # σ在这里！

    # ❌ 问题2: S0只在最后用于插值
    # 插值使用常数权重
    idx = find_nearest(S_grid, S0_val)
    w = (S0_val - S_grid[idx]) / (S_grid[idx+1] - S_grid[idx])  # 常数！

    price = V[idx] * (1-w) + V[idx+1] * w  # 线性插值

    return price
```

### 计算图的实际结构

```
理论上应该有的图：
┌─────────────────────────────────────┐
│  S ──┐                              │
│  K ──┼─> [PDE Solver] ─> V         │
│  σ ──┤                              │
│  r ──┤                              │
│  T ──┘                              │
└─────────────────────────────────────┘

实际的图：
┌─────────────────────────────────────┐
│  σ ──> [PDE on fixed grid] ─> V_grid│
│                                 │    │
│  S0 ───X (不在图中)             │    │
│         只用于最后的线性插值 ───┘    │
└─────────────────────────────────────┘
```

---

## 为什么S不在计算图中？

### 原因1: PDE数值方法的本质

**PDE求解是在固定网格上进行的**：

```python
# 网格是预先定义的，不依赖S0
S_grid = linspace(0, S_max, M)  # M个点，固定

# PDE求解在这些固定点上
for i in range(M):
    V[i] = solve_at_point(S_grid[i], sigma)  # S_grid[i]是常数

# S0不影响PDE求解过程
# S0只影响我们如何"读取"结果
```

**类比**：
- 理想情况：函数V(S)是可微的连续函数
- PDE实际：V是在M个离散点{V[0], V[1], ..., V[M-1]}的数组
- S0只是"在哪个索引处读取"，不是计算的一部分

### 原因2: 插值破坏了可微性

```python
# 线性插值
price = V[idx] * (1-w) + V[idx+1] * w

# 其中 w = (S0 - S[idx]) / (S[idx+1] - S[idx])
```

**问题**：
```
∂price/∂S0 = ∂w/∂S0 * (V[idx+1] - V[idx])
           = 1/(S[idx+1] - S[idx]) * (V[idx+1] - V[idx])
           = 常数

∂²price/∂S0² = ∂/∂S0(常数) = 0  ❌
```

### 原因3: 计算图只追踪ADVar之间的操作

```python
# Edge-Pushing追踪的是ADVar之间的操作

# 这会被追踪：
result = sigma_var * sigma_var  # ADVar * ADVar ✅

# 这不会被追踪：
w = (S0_val - grid_point) / spacing  # float运算 ❌

# 最终插值：
price = V[idx] * (1-w) + V[idx+1] * w
#       ADVar   * float + ADVar   * float
#       ↓
#       计算图只看到 ADVar * 常数
```

---

## 具体案例分析

### 场景：计算V(S0=105)

```python
# Step 1: PDE求解（在固定网格上）
S_grid = [0, 50, 100, 150, 200, 250]
sigma_var = ADVar(0.2, requires_grad=True)

# PDE求解，得到：
V_grid = [ADVar(0), ADVar(4.2), ADVar(10.5),
          ADVar(18.3), ADVar(27.1), ADVar(36.5)]

# 计算图此时：
#   sigma_var ─> V_grid[0]
#   sigma_var ─> V_grid[1]
#   ...
#   sigma_var ─> V_grid[5]

# Step 2: 获取V(S0=105)
S0_val = 105  # float，不是ADVar！
idx = 2  # 因为 100 < 105 < 150

w = (105 - 100) / (150 - 100) = 0.1  # 常数！

price = V_grid[2] * 0.9 + V_grid[3] * 0.1
#       ADVar     * const + ADVar     * const

# 计算图现在：
#   sigma_var ─> V_grid[2] ─> price
#   sigma_var ─> V_grid[3] ─> price
#
#   S0_val ───X  (完全不在图中！)
```

### Edge-Pushing会计算什么？

```python
# 调用Edge-Pushing
hessian = algo4_edge_pushing(price, [sigma_var])

# 结果：
H = [[∂²V/∂σ²]]  # 1×1矩阵

# 为什么不是5×5？
# 因为只有1个ADVar输入（sigma_var）！
# S0, K, r, T 都不在计算图中
```

**如果我们想要理论中的5×5矩阵**：

```python
# 需要这样：
S_var = ADVar(S0, requires_grad=True)
K_var = ADVar(K, requires_grad=True)
sigma_var = ADVar(sigma, requires_grad=True)
r_var = ADVar(r, requires_grad=True)
T_var = ADVar(T, requires_grad=True)

price_var = solve_pde_with_all_advar(S_var, K_var, sigma_var, r_var, T_var)

hessian = algo4_edge_pushing(price_var, [S_var, K_var, sigma_var, r_var, T_var])

# 现在会得到5×5矩阵
H = [[∂²V/∂S²,   ∂²V/∂S∂K, ...],
     [∂²V/∂K∂S,  ∂²V/∂K²,  ...],
     ...]

Γ = H[0,0]  # 这才是真正的Gamma
```

**但是！这在PDE实现中极其困难！**

---

## 为什么让S成为ADVar如此困难？

### 挑战1: 网格必须动态依赖S

```python
# 当前实现（固定网格）
def solve_pde(S0: float, sigma: ADVar):
    S_grid = linspace(0, 300, M)  # 固定
    V = solve_on_grid(S_grid, sigma)
    return V

# 需要的实现（动态网格）
def solve_pde(S0: ADVar, sigma: ADVar):
    # 网格中心必须依赖S0
    S_grid = linspace(S0 - 3*S0, S0 + 3*S0, M)  # 每个都是ADVar！

    # 问题：S_grid中的每个点都是ADVar
    # PDE系数计算会产生巨大的计算图
    for i in range(M):
        S_i = S_grid[i]  # ADVar
        alpha = sigma**2 * S_i**2 / 2  # ADVar
        # ...网格上的每个操作都被追踪
```

### 挑战2: 索引操作不可微

```python
# 线性插值需要找到索引
idx = np.searchsorted(S_grid, S0)  # 这是离散操作！

# 如果S0是ADVar：
# S0从104.9变到105.1时，idx从2跳到3
# 这是不连续的！∂idx/∂S0 未定义
```

### 挑战3: 计算图爆炸

```python
# 如果S是ADVar
# M=50个网格点，N=150个时间步

# 每个网格点S_grid[i]都是S的函数
# 每个时间步的每个V[i][n]都依赖S

# 计算图节点数：O(M × N) = 7,500个节点
# 每个节点可能有多个父节点
# Edge-Pushing复杂度变为O(M² × N²) 甚至更高
```

---

## 我们的解决方案：混合方法

### 放弃让S成为ADVar

```python
def solve_pde_with_aad_fixed(S0_val: float, sigma_val: float):
    """
    混合方法：
    - σ 作为ADVar（用于Vega, Volga）
    - S0 作为float（用网格FD计算Gamma）
    """

    # σ在计算图中
    sigma_var = ADVar(sigma_val, requires_grad=True)

    # PDE求解
    V_grid = solve_pde_on_fixed_grid(sigma_var)

    # 提取数值
    V_vals = [v.val for v in V_grid]

    # 方法分离：
    # 1. Vega via AAD
    price.adj = 1.0
    backward_pass()
    vega = sigma_var.adj  # ✅ 通过Edge-Pushing

    # 2. Gamma via 网格FD
    idx = find_nearest(S_grid, S0_val)
    gamma = (V_vals[idx+1] - 2*V_vals[idx] + V_vals[idx-1]) / dS²  # ✅

    return price, vega, gamma
```

### 为什么这样work？

**关键洞察**：V_vals本身包含S依赖性

```
V_vals = [V(S=0), V(S=50), V(S=100), V(S=150), ...]

不同的索引 i 对应不同的股价 S[i]

因此：
V_vals[i+1] - V_vals[i] 反映 ∂V/∂S
(V_vals[i+1] - V_vals[i]) - (V_vals[i] - V_vals[i-1]) 反映 ∂²V/∂S²
```

**这不是在插值空间求导（会得到0）**
**而是在价格网格空间求导（正确！）**

---

## 理论 vs 实践对比表

| 方面 | 理论（您的描述） | PDE实践（我们的实现） |
|------|-----------------|---------------------|
| **输入变量** | x = [S, K, σ, r, T] 都是ADVar | 只有σ是ADVar |
| **计算图** | 所有输入→输出 | 只有σ→V |
| **Hessian维度** | 5×5 矩阵 | 1×1 矩阵 |
| **Γ来源** | H[0,0] via Edge-Pushing | 网格FD |
| **Vega来源** | H[2,2] via Edge-Pushing | H[0,0] via Edge-Pushing ✅ |
| **计算图大小** | O(操作数) | O(M×N) 非常大 |
| **为何不同** | 函数是黑盒 | PDE求解器结构特殊 |

---

## 为什么不能简单修复？

### 尝试1: 让S成为ADVar（失败）

```python
S_var = ADVar(S0, requires_grad=True)

# 问题：网格构建
S_grid = linspace(0, 3*K, M)  # 这不依赖S_var！

# 如果强制依赖：
S_grid = [S_var + i*dS for i in range(M)]  # 现在依赖了

# 但是：
# 1. 边界条件怎么办？仍然需要0和3*K
# 2. 网格中心应该在S_var附近，但现在从S_var开始？
# 3. 计算图爆炸：M×N个ADVar节点
```

### 尝试2: 使用自适应网格（仍然困难）

```python
def adaptive_grid(S_center: ADVar):
    # 以S_center为中心构建网格
    S_grid = [S_center + (i - M/2)*dS for i in range(M)]

    # 问题：
    # 1. 如何保证网格覆盖[0, 3*K]？
    # 2. 边界条件V(0)=0, V(3*K)=... 如何施加？
    # 3. S_center从100→101时，整个网格平移
    #    → V的值完全变化
    #    → 数值不稳定
```

### 尝试3: 使用可微插值（理论上可行，但...）

```python
# 三次样条插值（理论上二阶可微）
from scipy.interpolate import CubicSpline

V_interp = CubicSpline(S_grid, V_vals)
price = V_interp(S0_val)  # 可微

# 问题：
# 1. CubicSpline不支持ADVar
# 2. 需要手动实现AD版本的CubicSpline
# 3. 计算代价非常高
# 4. 仍然需要V_vals是准确的（网格问题未解决）
```

---

## 最终结论

### 理论正确性

您的描述**完全正确**：Edge-Pushing应该计算完整Hessian矩阵，Γ应该是H[0,0]。

### 实践障碍

但在PDE实现中：
1. **网格离散化**：S不是连续变量，是离散索引
2. **插值不可微**：线性插值的二阶导数为0
3. **计算图结构**：S不在图中，只有σ在图中

### 实用解决方案

**混合方法**：
- **参数导数**（∂²V/∂σ², ∂²V/∂σ∂r）：用Edge-Pushing ✅
- **空间导数**（∂²V/∂S²）：用网格FD ✅

**这不是Edge-Pushing的问题**，而是：
- PDE数值方法的固有限制
- 离散化与自动微分的冲突
- 工程实现的权衡

### 如果要完全遵循理论

需要：
1. 实现完全可微的PDE求解器（所有变量都是ADVar）
2. 使用可微插值（如可微样条）
3. 接受巨大的计算代价（10-100×慢）
4. 可能的数值不稳定性

**这在学术研究中有探索（如JAX-based PDE solvers），但工程上不实用。**

---

## 您的理解和我们的实现都是正确的

- **您的理解**：Edge-Pushing理论，应用于黑盒函数
- **我们的实现**：PDE求解器不是黑盒，有特殊结构

两者都正确，只是应用场景不同！
