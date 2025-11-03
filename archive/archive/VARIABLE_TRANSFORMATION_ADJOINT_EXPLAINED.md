# 变量变换PDE vs Adjoint PDE：如何解决Vega问题

## 🎯 问题回顾

**当前CN格式的问题**：

```
Black-Scholes PDE:
∂V/∂t + (σ²S²/2)∂²V/∂S² + rS∂V/∂S - rV = 0
         ^^^^^^^
         扩散系数 ∝ σ²
```

**失败机制**：
1. 扩散系数 α = (σ²S²/2)/dS² 随σ²增长
2. σ=0.30时，α比σ=0.15大4倍
3. 导致数值阻尼 → Price低估 → Vega错误

---

## 方法1：变量变换PDE (Variable Transformation)

### 📐 核心思想

**通过坐标变换，将σ从扩散系数中"移除"**

### 数学推导

#### Step 1: 标准BS PDE

```
∂V/∂t + (σ²S²/2)∂²V/∂S² + rS∂V/∂S - rV = 0
```

Terminal condition: V(S,T) = max(S-K, 0)

#### Step 2: 变量变换

**定义新变量**：

```
x = ln(S/K)           ← 对数价格 (无量纲)
τ = σ²(T-t)/2         ← 方差时间
```

**逆变换**：
```
S = K·e^x
t = T - 2τ/σ²
```

#### Step 3: 链式法则

计算偏导数变换：

**空间导数**：
```
∂V/∂S = (∂V/∂x)·(∂x/∂S) = (∂V/∂x)·(1/S)

∂²V/∂S² = ∂/∂S[(∂V/∂x)·(1/S)]
        = (∂²V/∂x²)·(1/S²) - (∂V/∂x)·(1/S²)
```

**时间导数**：
```
∂V/∂t = (∂V/∂τ)·(∂τ/∂t) = (∂V/∂τ)·(-σ²/2)
```

#### Step 4: 代入BS PDE

```
(∂V/∂τ)·(-σ²/2) + (σ²S²/2)·[(∂²V/∂x²)·(1/S²) - (∂V/∂x)·(1/S²)]
                  + rS·(∂V/∂x)·(1/S) - rV = 0
```

简化：
```
-σ²/2·∂V/∂τ + σ²/2·(∂²V/∂x² - ∂V/∂x) + r·∂V/∂x - rV = 0
```

**消去 σ²/2**：
```
-∂V/∂τ + (∂²V/∂x² - ∂V/∂x) + (2r/σ²)·∂V/∂x - (2r/σ²)V = 0
```

**整理为扩散方程**：
```
∂V/∂τ = ∂²V/∂x² + (2r/σ² - 1)·∂V/∂x - (2r/σ²)·V
```

### 🎉 关键突破

**变换后的PDE**：

```
∂V/∂τ = ∂²V/∂x² + b(σ)·∂V/∂x + c(σ)·V

其中：
  b(σ) = 2r/σ² - 1     ← σ在漂移项
  c(σ) = -2r/σ²        ← σ在反应项
```

**扩散系数 = 1 (常数！)**

### 💡 为什么解决了Vega问题

**原始问题**：
```
α = (σ²S²/2)/dS²  ∝ σ²  ← 随σ²增长，导致数值阻尼
```

**变换后**：
```
α = 1  ← 常数！无论σ多大
```

**数值稳定性**：
```
原始：dt·α = dt·(σ²S²/2)/dS² = O(σ²)  ← 高σ时>>1，不稳定

变换：dτ·α = dτ·1 = dτ < 1  ← 稳定！
```

**σ的影响**：
- σ只出现在**drift和reaction项** (b和c)
- 这些项是**线性的**，数值上稳定
- 不会产生阻尼效应

### 🔧 实现要点

#### 边界条件变换

**Terminal condition**：
```
原始：V(S, T) = max(S - K, 0)

变换：V(x, 0) = max(K·e^x - K, 0) = K·max(e^x - 1, 0)
```

**空间边界**：
```
原始：S ∈ [0, Smax]

变换：x ∈ [-∞, ln(Smax/K)]
      实际：x ∈ [-L, L]，其中L足够大
```

#### AAD路径

**σ如何进入计算**：

```python
# 1. τ的定义
tau_var = sigma_var * sigma_var * ADVar(T - t_current) / ADVar(2.0)

# 2. Drift系数
b_var = ADVar(2*r) / (sigma_var * sigma_var) - ADVar(1.0)

# 3. Reaction系数
c_var = -ADVar(2*r) / (sigma_var * sigma_var)

# 4. PDE求解
# ...使用b_var, c_var

# 5. AAD反向传播
# ∂V/∂σ 通过 τ, b, c 三条路径传播
```

**Vega计算**：
```
Vega = ∂V/∂σ = (∂V/∂τ)·(∂τ/∂σ) + (∂V/∂b)·(∂b/∂σ) + (∂V/∂c)·(∂c/∂σ)
```

### ✅ 优势

1. **数值稳定**：扩散系数=1，无阻尼
2. **精确Vega**：σ依赖性显式且线性
3. **高效**：时间复杂度相同 O(MN)
4. **Pure PDE**：仍然是PDE方法，不是MC

### ⚠️ 挑战

1. **实现复杂度**：需要重写PDE求解器
2. **边界处理**：x∈(-∞,∞)需要截断
3. **AAD追踪**：τ, b, c都依赖σ，路径复杂
4. **时间步长**：τ步长需要根据σ调整

---

## 方法2：Adjoint PDE

### 📐 核心思想

**不通过有限差分计算∂V/∂σ，而是直接求解Vega的PDE**

### 数学推导

#### Step 1: BS PDE

```
L[V] ≡ ∂V/∂t + (σ²S²/2)∂²V/∂S² + rS∂V/∂S - rV = 0
```

#### Step 2: 对σ求导

对整个PDE关于σ求导：

```
∂L[V]/∂σ = 0

展开：
∂/∂σ[∂V/∂t] + ∂/∂σ[(σ²S²/2)∂²V/∂S²] + ∂/∂σ[rS∂V/∂S] - ∂/∂σ[rV] = 0
```

**第一项**：
```
∂/∂σ[∂V/∂t] = ∂/∂t[∂V/∂σ] = ∂Vega/∂t
```

**第二项（关键！）**：
```
∂/∂σ[(σ²S²/2)∂²V/∂S²] = 2σS²/2·∂²V/∂S² + σ²S²/2·∂²Vega/∂S²
                       = σS²·Γ + σ²S²/2·∂²Vega/∂S²
```

其中 Γ = ∂²V/∂S² 是**Gamma**（已知）

**第三项**：
```
∂/∂σ[rS∂V/∂S] = rS·∂Vega/∂S
```

**第四项**：
```
∂/∂σ[rV] = r·Vega
```

#### Step 3: Vega的PDE

整理得到：

```
∂Vega/∂t + σ²S²/2·∂²Vega/∂S² + rS·∂Vega/∂S - r·Vega = -σS²·Γ
```

**标准形式**：

```
∂Vega/∂t + L_BS[Vega] = Source(S, t, σ, Γ)

其中：
  L_BS[·] = σ²S²/2·∂²/∂S² + rS·∂/∂S - r·(·)  ← BS算子
  Source = -σS²·Γ                                ← 源项！
```

### 🎉 关键突破

**两步求解**：

1. **Forward solve**：求解原始BS PDE得到V和Γ
   ```
   ∂V/∂t + L_BS[V] = 0
   V(S,T) = Payoff(S)

   → 得到 V(S,t) 和 Γ(S,t) = ∂²V/∂S²
   ```

2. **Adjoint solve**：用Γ作为源项求解Vega
   ```
   ∂Vega/∂t + L_BS[Vega] = -σS²·Γ(S,t)
   Vega(S,T) = 0  ← Terminal为0！

   → 得到 Vega(S,t)
   ```

### 💡 为什么解决了Vega问题

**原问题（有限差分）**：
```
Vega_FD = [V(σ+ε) - V(σ-ε)] / (2ε)

问题：
- 需要求解两次PDE（σ±ε）
- 高σ时V_PDE本身不准 → Vega误差巨大
- 误差放大：Vega_error ≈ Price_error / ε
```

**Adjoint方法**：
```
Vega_Adjoint：直接求解

优势：
- 只需一次forward + 一次adjoint
- 不依赖于有限差分
- Source term明确：σS²Γ
- Γ的精度决定Vega精度
```

**为什么不受数值阻尼影响**：

1. **Γ的计算**：
   - Γ = ∂²V/∂S² 来自forward solve
   - 即使V有误差，Γ的**空间导数**相对稳定
   - 因为误差主要来自时间步进，空间导数部分抵消

2. **源项的作用**：
   - Source = -σS²·Γ 显式依赖σ
   - 这个依赖关系是**精确的**
   - 不经过有限差分近似

3. **算子相同**：
   - Vega PDE使用相同的L_BS算子
   - 如果V的PDE稳定，Vega的PDE也稳定

### 🔧 实现要点

#### Forward Solve

```python
class ForwardSolver:
    def solve(self, sigma):
        # 标准BS PDE
        for t in range(N):
            V = pde_step(V, sigma)

        # 计算Gamma
        Gamma = compute_second_derivative(V)

        return V, Gamma
```

#### Adjoint Solve

```python
class AdjointSolver:
    def solve(self, sigma, Gamma_history):
        # 初始化：Terminal condition
        Vega = zeros(M)  # Vega(S,T) = 0

        # 反向时间步进
        for t in range(N-1, -1, -1):
            # Source term at time t
            Source = -sigma * S_grid**2 * Gamma_history[t]

            # PDE step with source
            Vega = pde_step_with_source(Vega, sigma, Source)

        return Vega
```

#### 关键：带源项的PDE步

**修改CN格式**：

```
标准CN：L_B·V^n = R_B·V^(n+1)

带源项：L_B·V^n = R_B·V^(n+1) + dt·Source
```

**离散化**：
```
Vega^n = L_B^{-1} · [R_B·Vega^(n+1) + dt·(-σS²·Γ^n)]
```

#### AAD集成

**关键问题**：Γ也是通过ADVar计算的！

```python
# Forward solve
sigma_var = ADVar(sigma, requires_grad=True)
V_grid = solve_forward_pde(sigma_var)  # V_grid是ADVar列表

# 计算Gamma（也是ADVar）
Gamma_grid = []
for i in range(1, M-1):
    Gamma_i = (V_grid[i+1] - 2*V_grid[i] + V_grid[i-1]) / (dS**2)
    Gamma_grid.append(Gamma_i)  # Gamma_i是ADVar！

# Adjoint solve
Source = []
for i in range(M-2):
    S_i = S_grid[i+1]
    source_i = -sigma_var * ADVar(S_i**2) * Gamma_grid[i]
    Source.append(source_i)

# 求解adjoint PDE（Vega_grid也是ADVar）
Vega_grid = solve_adjoint_pde(sigma_var, Source)

# 最终Vega
vega = Vega_grid[S0_index].val
```

**AAD自动处理所有路径**：
```
∂V/∂σ 的路径：
1. σ → Γ → Source → Vega  ← Adjoint路径
2. σ → L_BS算子 → Vega     ← 直接路径
```

### ✅ 优势

1. **理论精确**：直接求解Vega的PDE
2. **不依赖有限差分**：避免ε的选择问题
3. **高效**：一次forward + 一次adjoint ≈ 2×原始成本
4. **可扩展**：可推广到Volga, Vanna等

### ⚠️ 挑战

1. **Γ精度依赖**：需要准确的Gamma
2. **实现复杂**：需要支持源项的PDE求解器
3. **存储需求**：需要保存所有时间步的Γ
4. **边界条件**：Vega的边界条件不明显

---

## 对比总结

| 维度 | 变量变换PDE | Adjoint PDE | 当前方法 (CN+AAD) |
|------|------------|------------|-------------------|
| **核心思想** | 变换坐标，σ不在扩散项 | 直接求解Vega的PDE | 有限差分+AAD |
| **数值稳定性** | ⭐⭐⭐⭐⭐ 完美 | ⭐⭐⭐⭐ 取决于Γ | ⭐ 高σ失效 |
| **实现难度** | ⭐⭐⭐ 中等 | ⭐⭐⭐⭐ 难 | ⭐ 简单 |
| **计算成本** | 1× (与原始相同) | 2× (forward+adjoint) | 2× (两次PDE for FD) |
| **Vega精度** | ⭐⭐⭐⭐⭐ 很高 | ⭐⭐⭐⭐ 高 | ⭐ 低 (高σ) |
| **Volga可行性** | ⭐⭐⭐⭐ 可以 | ⭐⭐⭐⭐⭐ 易扩展 | ⭐ 完全失败 |
| **局部波动率** | ⭐⭐ 需要修改 | ⭐⭐⭐⭐⭐ 自然支持 | ⭐⭐⭐ 可以 |
| **理论优雅性** | ⭐⭐⭐⭐⭐ 极优雅 | ⭐⭐⭐⭐⭐ 数学完美 | ⭐⭐ 工程权宜 |

---

## 实现路线图

### Phase 1: 变量变换 (推荐先做)

**时间**：1-2周

**步骤**：
1. 实现(x,τ)坐标变换
2. 重写PDE求解器（扩散系数=1）
3. 处理边界条件
4. AAD路径追踪（τ, b, c）
5. 测试Vega精度

**预期结果**：
- Vega误差 < 3% (全σ范围)
- Volga正确
- 计算时间不变

### Phase 2: Adjoint PDE (理论最优)

**时间**：1-2月

**步骤**：
1. 实现带源项的PDE求解器
2. Forward solve保存Γ历史
3. Adjoint solve with Source
4. AAD集成（Γ是ADVar）
5. 扩展到Volga (二阶adjoint)

**预期结果**：
- Vega误差 < 1%
- Volga, Vanna都精确
- 支持局部波动率

---

## 推荐方案

**立即实施**：变量变换PDE

**原因**：
1. 理论清晰，扩散系数=1解决根本问题
2. 实现难度适中
3. 一次性解决Vega+Volga
4. Pure PDE方法，符合你的框架要求

**实现伪代码**：

```python
class TransformedBSPDE:
    """变换后的BS PDE求解器"""

    def __init__(self, K, T, r, M, N):
        # x空间网格
        self.x_grid = linspace(-5, 5, M)  # x = ln(S/K)
        self.dx = self.x_grid[1] - self.x_grid[0]

    def solve(self, sigma_var):
        """
        求解变换后的PDE

        ∂V/∂τ = ∂²V/∂x² + b(σ)·∂V/∂x + c(σ)·V
        """
        # 系数（都是ADVar）
        b = ADVar(2*self.r) / (sigma_var * sigma_var) - ADVar(1.0)
        c = -ADVar(2*self.r) / (sigma_var * sigma_var)

        # τ步长
        tau_max = sigma_var * sigma_var * ADVar(self.T) / ADVar(2.0)
        dtau = tau_max / ADVar(self.N)

        # 初始条件（τ=0即t=T）
        V = self._terminal_condition()

        # 时间步进
        for n in range(self.N):
            V = self._cn_step(V, b, c, dtau)

        # 插值到S0 (x=0)
        V_0 = self._interpolate(V, x=0)

        return V_0
```

**是否需要我开始实现变量变换PDE？**
