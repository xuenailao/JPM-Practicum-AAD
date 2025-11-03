# PDE Vega问题的最终解决方案提案

## 📋 问题回顾

经过全面测试，以下方案均**失败**：

1. ✅ **Adaptive time step** (N=400) - 失败，Vega趋势仍错
2. ✅ **Ultra-fine grid** (M=151, N=600) - 失败，Vega趋势仍错
3. ⏱️ **Rannacher time-stepping** - 超时，计算太慢

**根本原因确认**：
- CN格式在高σ时数值阻尼
- Price被系统性低估
- ∂V/∂σ < 0 (错误方向)
- 不是网格问题，是格式问题

---

## 💡 核心洞察

**问题不在AAD，而在PDE求解器输出的V(σ)本身**

```
V_PDE(σ) 在高σ时偏差巨大 →  ∂V_PDE/∂σ 错误
AAD正确计算了 ∂V_PDE/∂σ，但V_PDE本身就错了
```

**因此**：
- 不能指望通过AAD修复Vega
- 必须修复PDE求解器本身

---

## 🚀 解决方案架构

### 方案A：Adjoint PDE方法（理论最优）

**核心思想**：

不通过AAD计算∂V/∂σ，而是**直接求解伴随PDE**

**Black-Scholes PDE**:
```
∂V/∂t + (σ²S²/2)∂²V/∂S² + rS∂V/∂S - rV = 0
```

**对σ求导**:
```
∂/∂σ[PDE] = 0

→ ∂²(∂V/∂σ)/∂S² + ... = -σS²∂²V/∂S²  (source term!)
```

**Adjoint PDE for Vega**:
```
∂Vega/∂t + (σ²S²/2)∂²Vega/∂S² + rS∂Vega/∂S - rVega = σS²Γ
```

其中 Γ = ∂²V/∂S² 是Gamma（已知from PDE解）

**优势**:
- 绕过高σ时的数值阻尼问题
- Source term σS²Γ 显式包含σ依赖
- 理论上精确

**实现**:
1. 求解forward PDE得到V和Γ
2. 用Γ作为source term求解adjoint PDE得到Vega
3. 避免了对σ的有限差分

**挑战**:
- 需要实现新的PDE求解器（带source term）
- Γ的精度影响Vega精度

---

### 方案B：变量变换法

**核心思想**：

变换变量使σ不出现在diffusion coefficient中

**标准BS PDE**:
```
∂V/∂t + (σ²S²/2)∂²V/∂S² + rS∂V/∂S - rV = 0
```

**变量变换**: x = ln(S/K), τ = σ²(T-t)/2

```
∂V/∂τ = ∂²V/∂x² + (r/σ² - 0.5)∂V/∂x - (2r/σ²)V
```

**现在**: σ进入drift和discount项，但**不在diffusion项**！

**优势**:
- Diffusion coefficient = 1 (常数！)
- 无数值阻尼问题
- σ依赖性更explicit

**挑战**:
- 需要重写整个PDE求解器
- 边界条件变换复杂
- AAD路径变化

---

### 方案C：两步法（实用折中）

**核心思想**：

分离Price和Vega的计算

**Step 1**: 使用当前PDE计算Price和Delta/Gamma
- 保持快速
- 精度可接受 (σ≤0.25)

**Step 2**: 使用数值稳定的方法计算Vega
- 选项2a: Adjoint PDE
- 选项2b: Pathwise Monte Carlo
- 选项2c: Likelihood Ratio Method

**Monte Carlo + AAD for Vega**:

```python
# 生成路径
S_T = S0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)

# Payoff
V = exp(-r*T) * max(S_T - K, 0)

# AAD自动求导
Vega = ∂V/∂sigma  # MC没有数值阻尼问题！
```

**优势**:
- PDE用于快速Greeks (Δ/Γ)
- MC用于精确Vega/Vanna/Volga
- 充分利用两者优势

---

### 方案D：Malliavin Calculus (理论最强)

**核心思想**：

使用Malliavin权重直接计算Vega

**SDE**:
```
dS_t = r*S_t*dt + sigma*S_t*dW_t
```

**Malliavin导数**:
```
D_sigma S_T = S_T * integral_0^T (dW_s / sigma)
```

**Vega**:
```
Vega = E[Payoff' * D_sigma S_T]
```

**优势**:
- 理论最优
- 不需要路径perturbation
- 与AAD完美结合

**挑战**:
- 需要Malliavin calculus实现
- 复杂度高

---

## 🎯 推荐方案

### 短期（1-2周）：方案C - 混合PDE+MC

**实现步骤**:

1. **保留现有PDE for Delta/Gamma**
   ```python
   # 使用Method A
   delta, gamma = pde_greeks(S0, K, T, r, sigma)
   ```

2. **实现MC+AAD for Vega**
   ```python
   class MCVega:
       def compute_vega(self, S0, K, T, r, sigma, N_paths=10000):
           # sigma as ADVar
           sigma_var = ADVar(sigma, requires_grad=True)

           # Generate paths
           Z = np.random.randn(N_paths)
           S_T = S0 * exp((r - 0.5*sigma_var**2)*T
                          + sigma_var*sqrt(T)*Z)

           # Payoff
           payoff = exp(-r*T) * maximum(S_T - K, 0)
           price = mean(payoff)

           # AAD
           price.backward()
           return sigma_var.grad  # Vega
   ```

3. **组合结果**
   ```python
   def hybrid_greeks(S0, K, T, r, sigma):
       # PDE: 快速且准确
       price, delta, gamma = pde_solver(...)

       # MC: 稳健
       vega, vanna, volga = mc_vega_greeks(...)

       return {
           'price': price,    # PDE
           'delta': delta,    # PDE
           'gamma': gamma,    # PDE
           'vega': vega,      # MC
           'vanna': vanna,    # MC
           'volga': volga     # MC
       }
   ```

**优势**:
- 快速实现
- Delta/Gamma保持高效
- Vega稳健可靠
- MC+AAD高效（比bumping快100×）

---

### 中期（1-2月）：方案B - 变量变换

**实现步骤**:

1. **实现transformed PDE solver**
   - 变量: x = ln(S/K), τ = σ²(T-t)/2
   - Diffusion coefficient = 1

2. **重新设计AAD路径**
   - σ进入τ的定义
   - 追踪∂V/∂τ和∂τ/∂σ

3. **验证精度**
   - 测试Vega在全σ范围

**预期**:
- Vega误差 < 5% for all σ
- Volga正确
- 计算时间相当

---

### 长期（3-6月）：方案A - Adjoint PDE

**实现步骤**:

1. **实现forward PDE**
   - 计算V和Γ

2. **实现adjoint PDE**
   - Source term = σS²Γ
   - 求解Vega

3. **二阶adjoint for Volga**
   - ∂Vega/∂σ

**预期**:
- 所有Greeks精确
- 计算高效
- 理论完美

---

## 📊 方案对比

| 方案 | 实现难度 | 计算时间 | Vega精度 | Volga精度 | 推荐度 |
|------|---------|---------|---------|-----------|--------|
| C: 混合PDE+MC | ⭐ 简单 | ⭐⭐ 中等 | ⭐⭐⭐ 高 | ⭐⭐⭐ 高 | ⭐⭐⭐⭐⭐ |
| B: 变量变换 | ⭐⭐⭐ 中等 | ⭐⭐⭐ 快 | ⭐⭐⭐ 高 | ⭐⭐⭐ 高 | ⭐⭐⭐⭐ |
| A: Adjoint PDE | ⭐⭐⭐⭐ 难 | ⭐⭐⭐ 快 | ⭐⭐⭐⭐ 很高 | ⭐⭐⭐⭐ 很高 | ⭐⭐⭐ |
| D: Malliavin | ⭐⭐⭐⭐⭐ 很难 | ⭐⭐ 慢 | ⭐⭐⭐⭐ 很高 | ⭐⭐⭐⭐ 很高 | ⭐⭐ |

---

## 🔧 立即可行：方案C实现

我现在可以立即实现方案C的MC+AAD部分：

**关键代码**:

```python
def mc_vega_aad(S0, K, T, r, sigma, N_paths=50000):
    """
    Monte Carlo + AAD for Vega

    Advantage: No numerical damping issue
    """
    from aad_edge_pushing.aad.core.var import ADVar
    from aad_edge_pushing.aad.core.tape import global_tape

    global_tape.reset()

    # sigma as ADVar
    sigma_var = ADVar(sigma, requires_grad=True)
    r_var = ADVar(r, requires_grad=False)
    T_var = ADVar(T, requires_grad=False)

    # Generate standard normal samples
    np.random.seed(42)
    Z = np.random.randn(N_paths)

    # Simulate paths
    drift = (r_var - sigma_var * sigma_var * ADVar(0.5)) * T_var
    diffusion = sigma_var * ADVar(np.sqrt(T)) * ADVar(Z[0])

    payoffs = []
    for z in Z:
        S_T_log = ADVar(np.log(S0)) + drift + sigma_var * ADVar(np.sqrt(T)) * ADVar(z)
        S_T = exp(S_T_log)
        payoff = maximum(S_T - ADVar(K), ADVar(0.0))
        payoffs.append(payoff)

    # Average
    price_var = sum(payoffs) / ADVar(N_paths) * exp(-r_var * T_var)

    # AAD backward
    price_var.adj = 1.0
    for node in reversed(global_tape.nodes):
        for parent, deriv in node.parents:
            if parent.requires_grad:
                parent.adj += node.out.adj * deriv

    return price_var.val, sigma_var.adj
```

**测试结果预期**:
- Vega误差 < 1% (with N_paths=50000)
- Vega趋势正确 ✅
- Volga可计算
- 时间: ~100ms per Vega

---

## 📝 总结

**当前状态**:
- ❌ CN-PDE无法正确计算Vega (根本性问题)
- ✅ AAD工作正常
- ✅ Delta/Gamma可用

**建议行动**:

1. **立即**: 实现方案C (混合PDE+MC)
   - 1-2天可完成
   - 生产可用

2. **短期**: 验证MC+AAD精度
   - 测试全σ范围
   - 与解析解对比

3. **中期**: 考虑方案B (变量变换)
   - 如果需要pure PDE方案
   - 理论上应该work

4. **长期**: 研究Adjoint PDE
   - 理论最优
   - 适合复杂模型

**是否继续实现方案C？**
