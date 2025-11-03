# Edge-Pushing中Gamma计算的完整解释

## 目录
1. [问题背景](#问题背景)
2. [原始错误实现](#原始错误实现)
3. [修复后的正确实现](#修复后的正确实现)
4. [数学原理](#数学原理)
5. [计算图分析](#计算图分析)
6. [代码实现细节](#代码实现细节)
7. [与Bumping方法的对比](#与bumping方法的对比)

---

## 问题背景

### 目标
计算期权价格V对初始股价S0的二阶导数（Gamma）：
```
Gamma = ∂²V/∂S0²
```

### 挑战
Edge-Pushing方法使用AAD (Automatic Algorithmic Differentiation)，但面临以下问题：
1. **S0不在计算图中**：S0仅用于最终插值，不是ADVar
2. **线性插值破坏二阶导数**：`price = interp(S0, S_grid, V_grid)` 是线性的
3. **结果**：直接用Edge-Pushing算法计算 ∂²V/∂S0² 会得到 **Gamma = 0**

---

## 原始错误实现

### 错误的计算流程

```python
# Step 1: PDE求解 (Forward pass)
def solve_pde_with_aad(S0_val, sigma_val):
    sigma_var = ADVar(sigma_val, requires_grad=True)  # σ在计算图中
    # S0_val 只是float，不在计算图中！

    # ... PDE时间步进 ...
    # 得到价格网格 V[i] for i=0..M-1

    # Step 2: 线性插值获取V(S0)
    idx = argmin(|S_grid - S0_val|)
    if S_grid[idx] < S0_val:
        i1, i2 = idx, idx+1
    else:
        i1, i2 = idx-1, idx

    S1, S2 = S_grid[i1], S_grid[i2]
    w = (S0_val - S1) / (S2 - S1)  # 权重（常数）

    # ❌ 问题核心：线性插值
    price_var = V[i1] * (1 - w) + V[i2] * w

    return price_var
```

### 为什么Gamma = 0？

**数学分析**：

线性插值函数：
```
V(S0) = V[i1] * (1 - w) + V[i2] * w
其中 w = (S0 - S1) / (S2 - S1)
```

一阶导数（Delta）：
```
∂V/∂S0 = ∂V/∂w * ∂w/∂S0
       = (V[i2] - V[i1]) * 1/(S2 - S1)
       = 常数
```

二阶导数（Gamma）：
```
∂²V/∂S0² = ∂/∂S0 (常数) = 0  ❌
```

**计算图视角**：

```
计算图结构：

σ (ADVar) ──────┐
                │
                ▼
       [PDE Solver]
                │
                ▼
           V[0], V[1], ..., V[M-1]  (所有都是ADVar)
                │
                │  Linear Interpolation
                │  (使用常数权重 w)
                ▼
            price_var (ADVar)
                │
                │  只有σ的路径
                │
                ▼
         Hessian ∂²V/∂σ²  ✅ (Volga可计算)

但是：
S0 ───────X───> 没有路径到price_var！
         (不在图中)

所以：∂²V/∂S0² = 0  ❌
```

---

## 修复后的正确实现

### 核心思想

**不依赖计算图求Gamma，而是在价格网格上用有限差分直接计算**

### 修复后的计算流程

```python
def solve_pde_with_aad_fixed(S0_val, sigma_val):
    # Step 1: PDE求解 (不变)
    sigma_var = ADVar(sigma_val, requires_grad=True)
    # ... PDE时间步进 ...
    # 得到 V = [V[0], V[1], ..., V[M-1]]  (ADVar列表)

    # Step 2: 提取数值，保存完整价格网格
    V_vals = np.array([v.val for v in V])  # 转为numpy数组

    # Step 3: 在网格上计算Gamma (关键修复！)
    gamma = _compute_gamma_on_grid(V_vals, S0_val)

    # Step 4: Vega仍然通过AAD计算
    price_var.adj = 1.0
    # ... 反向传播 ...
    vega = sigma_var.adj

    return price, delta, gamma, vega
```

### Gamma计算细节

```python
def _compute_gamma_on_grid(V_grid: np.ndarray, S0: float) -> float:
    """
    在价格网格上计算Gamma

    使用三点中心差分公式：
    ∂²V/∂S² ≈ [V(S+h) - 2V(S) + V(S-h)] / h²
    """
    # Step 1: 找到S0在网格中的位置
    idx = np.searchsorted(S_grid[1:-1], S0)

    # Step 2: 边界处理
    if idx == 0:
        idx = 1  # 避免越界
    elif idx >= len(V_grid) - 1:
        idx = len(V_grid) - 2

    # Step 3: 三点中心差分
    dS = S_grid[1] - S_grid[0]  # 网格步长

    V_plus = V_grid[idx + 1]   # V(S + dS)
    V_center = V_grid[idx]     # V(S)
    V_minus = V_grid[idx - 1]  # V(S - dS)

    # 二阶中心差分公式
    gamma = (V_plus - 2.0 * V_center + V_minus) / (dS**2)

    return gamma
```

---

## 数学原理

### 有限差分法推导

**泰勒展开**：

向前一步：
```
V(S + h) = V(S) + h·V'(S) + (h²/2)·V''(S) + O(h³)
```

向后一步：
```
V(S - h) = V(S) - h·V'(S) + (h²/2)·V''(S) + O(h³)
```

两式相加：
```
V(S+h) + V(S-h) = 2V(S) + h²·V''(S) + O(h⁴)
```

求解V''(S)：
```
V''(S) = [V(S+h) - 2V(S) + V(S-h)] / h² + O(h²)
```

**精度分析**：

- 截断误差：O(h²) = O(dS²)
- 对于M=50, dS=6/50=0.12，误差 ~ 0.12² = 1.44%
- 对于M=100, dS=0.06，误差 ~ 0.36%

### 为什么网格FD有效？

**关键洞察**：

```
PDE求解器输出：V_grid = [V(S₀), V(S₁), ..., V(Sₘ)]

这些值是在不同股价点的期权价格
```

**V_grid本身就包含了S依赖性**：
- V[i] = V(S_grid[i], t=0, σ)
- 不同的i对应不同的S值
- 因此V[i+1] - V[i] 反映了 ∂V/∂S
- 而 (V[i+1] - 2V[i] + V[i-1]) 反映了 ∂²V/∂S²

**与插值的区别**：
```
线性插值：
  V(S0) = (1-w)·V[i] + w·V[i+1]
  关于S0是线性的 → Gamma = 0

网格FD：
  Gamma = (V[i+1] - 2V[i] + V[i-1]) / dS²
  直接使用网格点的值，不依赖插值 → Gamma ≠ 0
```

---

## 计算图分析

### 修复前：计算图中没有S0路径

```
                    σ (ADVar)
                        │
                        ▼
                   [PDE Solver]
                        │
                        ▼
    ┌──────────────────────────────────┐
    │  Computational Graph             │
    │                                  │
    │  V[0] ──┐                       │
    │  V[1] ──┼─── 线性插值 ─── price │
    │  V[2] ──┘   (权重=常数)         │
    │                                  │
    └──────────────────────────────────┘
                        │
                        ▼
              price.adj = 1.0
                        │
                        ▼
              [Backward Pass]
                        │
                        ▼
              σ.adj = ∂V/∂σ  ✅

              但是：S0不在图中 ❌
              无法计算 ∂²V/∂S0²
```

### 修复后：绕过计算图，直接用数值FD

```
                    σ (ADVar)              S0 (float)
                        │                      │
                        ▼                      │
                   [PDE Solver]                │
                        │                      │
                        ▼                      │
                  V[0], V[1], ..., V[M]       │
                        │                      │
                        ├──── AAD 路径 ────────┤
                        │                      │
                        │                      ▼
                        │           [Grid FD: 不在图中]
                        │                      │
                        ▼                      ▼
              Vega via AAD              Gamma via FD
              (∂V/∂σ) ✅                (∂²V/∂S²) ✅
```

**关键点**：
1. **Vega**: 通过AAD计算，因为σ在计算图中
2. **Gamma**: 不通过AAD，而是在V网格上用FD
3. **两者互补**：AAD擅长参数导数，FD擅长空间导数

---

## 代码实现细节

### 完整实现（带注释）

```python
def solve_pde_with_aad(self, S0_val: float, sigma_val: float,
                      compute_hessian: bool = False):
    """
    求解PDE with AAD (修复版)

    关键修复：Gamma不通过AAD计算
    """
    t_start = time.perf_counter()
    global_tape.reset()

    # ============================================================
    # Part 1: Forward Pass - PDE求解
    # ============================================================

    # σ作为ADVar（在计算图中）
    sigma_var = ADVar(sigma_val, requires_grad=True, name="sigma")

    # S0只是float（不在计算图中）
    self.S0 = S0_val

    # 自适应时间步
    t_grid, N = self.compute_adaptive_timesteps(sigma_val)
    dt_val = t_grid[1] - t_grid[0]
    dt = ADVar(dt_val, requires_grad=False)

    # 构建CN系统（系数依赖σ）
    a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_cn(sigma_var, dt)

    # 初始条件
    V_terminal = self._terminal_condition()
    V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

    # 时间步进（所有V[i]都是ADVar）
    for n in range(N):
        t_current = t_grid[n+1]
        V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_current)

    # ============================================================
    # Part 2: 获取V(S0)的值（用于报告价格）
    # ============================================================

    # 找到S0附近的网格点
    idx = np.argmin(np.abs(self.S_grid - S0_val))

    if abs(self.S_grid[idx] - S0_val) < 1e-10:
        # S0恰好在网格点上
        price_var = V[idx - 1]
    else:
        # 需要插值
        if self.S_grid[idx] < S0_val:
            i1, i2 = idx, idx + 1
        else:
            i1, i2 = idx - 1, idx

        S1, S2 = self.S_grid[i1], self.S_grid[i2]
        w_const = (S0_val - S1) / (S2 - S1)

        # 线性插值（用于价格，但不用于Gamma！）
        price_var = V[i1-1] * ADVar(1.0 - w_const) + V[i2-1] * ADVar(w_const)

    price = price_var.val

    # ============================================================
    # Part 3: Backward Pass - 计算Vega (通过AAD)
    # ============================================================

    price_var.adj = 1.0
    for node in reversed(global_tape.nodes):
        for parent, deriv in node.parents:
            if parent.requires_grad:
                parent.adj += node.out.adj * float(deriv)

    vega = sigma_var.adj  # ∂V/∂σ via AAD ✅

    # ============================================================
    # Part 4: 计算Delta和Gamma (通过网格FD，不通过AAD)
    # ============================================================

    # 提取V网格的数值
    V_vals = np.array([v.val for v in V])

    # Delta: 一阶中心差分
    delta = self._compute_delta_on_grid(V_vals, S0_val)

    # Gamma: 二阶中心差分（关键修复！）
    gamma = self._compute_gamma_on_grid(V_vals, S0_val)

    t_end = time.perf_counter()

    return {
        'price': price,
        'delta': delta,      # via 网格FD
        'gamma': gamma,      # via 网格FD (修复！)
        'vega': vega,        # via AAD
        'time_ms': (t_end - t_start) * 1000.0
    }


def _compute_delta_on_grid(self, V_grid: np.ndarray, S0: float) -> float:
    """
    Delta: ∂V/∂S via 一阶中心差分

    公式: V'(S) ≈ [V(S+h) - V(S-h)] / (2h)
    """
    idx = np.searchsorted(self.S_grid[1:-1], S0)

    # 边界处理
    if idx == 0:
        idx = 1
    elif idx >= len(V_grid) - 1:
        idx = len(V_grid) - 2

    dS = self.dS

    # 一阶中心差分
    delta = (V_grid[idx+1] - V_grid[idx-1]) / (2.0 * dS)

    return delta


def _compute_gamma_on_grid(self, V_grid: np.ndarray, S0: float) -> float:
    """
    Gamma: ∂²V/∂S² via 二阶中心差分

    公式: V''(S) ≈ [V(S+h) - 2V(S) + V(S-h)] / h²

    这是整个修复的核心！
    """
    idx = np.searchsorted(self.S_grid[1:-1], S0)

    # 边界处理
    if idx == 0:
        idx = 1
    elif idx >= len(V_grid) - 1:
        idx = len(V_grid) - 2

    dS = self.dS

    # 二阶中心差分
    #
    # 直观理解：
    #   V_grid[idx+1]  →  未来更高的股价 S+dS
    #   V_grid[idx]    →  当前股价 S
    #   V_grid[idx-1]  →  过去更低的股价 S-dS
    #
    # Gamma衡量Delta的变化率：
    #   Delta向上 = (V[idx+1] - V[idx]) / dS
    #   Delta向下 = (V[idx] - V[idx-1]) / dS
    #   Gamma = (Delta向上 - Delta向下) / dS
    #         = [(V[idx+1] - V[idx]) - (V[idx] - V[idx-1])] / dS²
    #         = [V[idx+1] - 2V[idx] + V[idx-1]] / dS²

    gamma = (V_grid[idx+1] - 2.0 * V_grid[idx] + V_grid[idx-1]) / (dS**2)

    return gamma
```

### 网格索引详解

```python
# PDE网格结构
S_grid = [S_0, S_1, S_2, ..., S_M]
         └─────────────────────────┘
         M+1 个点

# V网格（内点）
V = [V_1, V_2, ..., V_{M-1}]
     └────────────────────┘
     M-1 个点（去掉边界）

# searchsorted(S_grid[1:-1], S0)
#   在内点中搜索 S0 的位置
#   返回 idx ∈ [0, M-1)

# 例子：M=5, S_grid=[0, 50, 100, 150, 200, 250]
#       V = [V_50, V_100, V_150, V_200]
#       S0 = 105
#
#       searchsorted([50,100,150,200], 105) = 2
#       → 使用 V[1]=V_100, V[2]=V_150, V[3]=V_200
#
#       gamma = (V_150 - 2*V_100 + V_50) / (50²)
```

---

## 与Bumping方法的对比

### Method 2 (Bumping)

```python
# 思路：对输入参数S0进行扰动
eps_S = 1.0

# 3次PDE求解
V_0 = solve_pde(S0, sigma)
V_plus = solve_pde(S0 + eps_S, sigma)
V_minus = solve_pde(S0 - eps_S, sigma)

# Gamma via 二阶有限差分
gamma = (V_plus - 2*V_0 + V_minus) / eps_S²
```

**特点**：
- ✅ 简单直观
- ✅ Gamma计算正确（扰动参数空间）
- ❌ 需要3次PDE求解（较慢）
- ✅ 结果：Gamma ≠ 0

### Method 4 (Edge-Pushing 修复前)

```python
# 思路：1次PDE + AAD计算所有导数
V_grid = solve_pde_with_aad(S0, sigma)

# 线性插值获取价格
price = interp(S0, S_grid, V_grid)

# 尝试用AAD计算Gamma
gamma = hessian(price, S0)  # ❌ 失败！
```

**问题**：
- ❌ S0不在计算图中
- ❌ 线性插值 → Gamma = 0
- ✅ 只需1次PDE
- ❌ 结果：Gamma = 0（错误）

### Method 4 (Edge-Pushing 修复后)

```python
# 思路：1次PDE + 混合方法
V_grid = solve_pde_with_aad(S0, sigma)

# Vega: 通过AAD（σ在图中）
vega = aad_gradient(price, sigma)  # ✅

# Gamma: 通过网格FD（S0不在图中）
idx = find_index(S_grid, S0)
gamma = (V_grid[idx+1] - 2*V_grid[idx] + V_grid[idx-1]) / dS²  # ✅
```

**特点**：
- ✅ Gamma计算正确（网格FD）
- ✅ Vega通过AAD（高效）
- ✅ 只需1次PDE
- ⚠️ 混合方法（AAD + FD）
- ✅ 结果：Gamma ≠ 0

### 三种方法对比表

| 方面 | Method 2 (Bumping) | Method 4 (修复前) | Method 4 (修复后) |
|------|-------------------|-------------------|-------------------|
| **Gamma原理** | 参数空间FD | AAD（失败） | 网格空间FD |
| **S0处理** | 扰动输入 | 线性插值 | 网格查找 |
| **PDE次数** | 3 | 1 | 1 |
| **Gamma值** | 0.0177 ✅ | 0.0000 ❌ | 0.0165 ✅ |
| **Gamma误差** | 5.3% | 100% | 11.9% |
| **计算时间** | 93 ms | - | 1689 ms |
| **Vega计算** | 参数空间FD | AAD ✅ | AAD ✅ |
| **实现复杂度** | 简单 | 简单但错 | 中等 |

---

## 结论

### 为什么Edge-Pushing中Gamma=0？

1. **根本原因**：线性插值的二阶导数为0
2. **技术原因**：S0不在AAD计算图中
3. **数学原因**：`V(S0) = (1-w)·V[i] + w·V[i+1]` 关于S0是线性的

### 如何修复？

1. **放弃AAD计算Gamma**：AAD无法追踪S0
2. **改用网格FD**：在V网格上直接计算二阶导数
3. **保留AAD优势**：Vega等参数导数仍用AAD

### 修复后的性能

**M=50, N=150:**
- Gamma = 0.0165336643（解析解=0.0187620173）
- 误差 = 11.9%（可接受）
- 时间 = 1.69秒（比Bumping慢，但只需1次PDE）

### 实践建议

1. **生产环境**：使用Method 2 (Bumping)
   - 简单可靠
   - Gamma误差5.3%
   - 速度快（93 ms）

2. **研究环境**：使用Method 4 (Edge-Pushing Fixed)
   - 展示AAD+FD混合方法
   - 1次PDE求解
   - 适合参数敏感性分析

3. **避免使用**：Method 4原版（未修复）
   - Gamma = 0（完全错误）

---

## 参考代码位置

- 修复版实现：`aad_edge_pushing/pde/AADgraph/original_pde_aad_hessian_fixed.py`
- Gamma计算函数：`_compute_gamma_on_grid()` (第277-287行)
- 完整测试：`aad_edge_pushing/pde/benchmark_complete.py`
