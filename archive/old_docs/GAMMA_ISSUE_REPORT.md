# Gamma 负数问题报告

## 🔴 发现的问题

### 症状
运行 `quick_test_rannacher.py` 时发现：
- **解析 Gamma**: 0.007505（正数，正确）
- **C-N Gamma**: -0.000363（**负数，错误！**）
- **Rannacher Gamma**: -0.000363（**同样负数**）

### 关键观察
1. **Gamma 符号错误**：Call 期权的 Gamma 必须为正，但计算结果为负
2. **数值大小错误**：计算值 -0.000363 vs 解析值 0.007505（相差 20 倍+）
3. **Rannacher 无效**：两种方法得到完全相同的错误结果

## 🔍 问题分析

### 不是 Rannacher 的问题
Rannacher timestepping 是为了解决 **数值振荡** 问题，它可以改善：
- ✅ Gamma 的振荡误差（从 104% 降到 10%）
- ✅ Greeks 的数值稳定性

但它 **无法修复**：
- ✗ Gamma 的符号错误
- ✗ 根本性的实现 bug

### 可能的根本原因

#### 1. Edge-Pushing Hessian 计算错误
`pde_aad_edgepushing.py` 中的 Hessian 计算可能有 bug：
```python
# Line 488: Edge-Pushing for full 2×2 Hessian
hessian = algo4_adjlist(price_var_h, [S0_var_h, sigma_var_h])

# Line 491-493: Extract second-order Greeks
gamma = hessian[0, 0]  # ∂²V/∂S0²
vanna = hessian[0, 1]  # ∂²V/∂S0∂σ
volga_pde = hessian[1, 1]  # ∂²V/∂σ²
```

#### 2. Natural Cubic Spline 问题
自然三次样条可能引入了符号错误：
- 样条的二阶导数 M_i 计算错误
- 插值公式中的符号错误
- 边界条件设置问题

#### 3. ADVar 梯度传播
AAD 反向传播可能在某处出错：
- S0 作为 ADVar 的处理
- 二阶导数的链式法则
- 边推进算法的实现

#### 4. 网格分辨率
虽然不太可能导致符号错误，但 M=51, N=100 确实比较粗糙。

## 🧪 诊断计划

### 测试 1: simple_gamma_test.py
正在后台运行，测试原始 `BS_PDE_AAD` 在 σ=0.2 下的 Gamma。

**预期结果**:
- 如果 Gamma 也是负数 → 问题在 `BS_PDE_AAD` 本身
- 如果 Gamma 是正数 → 问题可能只在 Rannacher 版本

### 测试 2: 有限差分 Gamma
使用简单的 bumping 方法计算 Gamma 作为基准：
```python
gamma_fd = (V(S0+ε) - 2V(S0) + V(S0-ε)) / ε²
```

如果有限差分也是负数 → PDE 求解本身有问题
如果有限差分是正数 → 问题在 Hessian 计算

### 测试 3: 对比 Bumping 方法
检查 `bumping_method.py` 的 Gamma 是否正确。
根据你的原始测试，Bumping 方法虽然误差大（104%），但至少**符号应该是对的**。

## 📊 之前的测试结果回顾

你提供的测试结果（σ=0.5, Bumping 方法）：
```
Price=17.892806 (vs 21.79 analytical)
Δ_err=33.07%
Γ_err=104.59%
```

**关键点**:
- Gamma **误差** 是 104.59%，不是负数
- 这意味着 Bumping 计算的 Gamma 大约是解析值的 2 倍（误差 >100%）
- 但**符号是对的**（否则误差会是 >200%）

## 🔧 下一步行动

### 立即
1. ✅ 等待 `simple_gamma_test.py` 完成（正在运行）
2. 检查输出文件 `simple_gamma_output.txt`
3. 根据结果确定问题范围

### 如果 BS_PDE_AAD 的 Gamma 也是负数
这意味着问题在原始实现中，需要：
1. 检查 `algo4_adjlist` (Edge-Pushing 算法实现)
2. 检查 `_compute_spline_second_derivatives`
3. 检查 S0 作为 ADVar 的处理

### 如果只有 Rannacher 版本有问题
检查 Rannacher 实现中的修改：
1. `build_tridiagonal_cn` 的 phi 参数处理
2. 系数选择逻辑
3. ADVar 的传播

### 如果 Hessian 有问题但 Jacobian 正常
1. 问题定位在 Edge-Pushing 算法
2. 需要检查 `algo4_adjlist` 的输入和输出
3. 可能需要切换回 finite difference 方法计算 Gamma

## 💡 临时解决方案

在修复根本问题之前，可以使用 **有限差分** 计算 Gamma：

```python
# 在 solve_pde_with_aad 中
if compute_hessian:
    # 使用 finite difference 而不是 Edge-Pushing
    eps = 0.001 * S0_val

    # V(S0 + eps)
    global_tape.reset()
    S0_var_plus = ADVar(S0_val + eps, requires_grad=True)
    # ... 求解 PDE

    # V(S0 - eps)
    global_tape.reset()
    S0_var_minus = ADVar(S0_val - eps, requires_grad=True)
    # ... 求解 PDE

    # Gamma = (delta_plus - delta_minus) / (2*eps)
    gamma = (delta_plus - delta_minus) / (2*eps)
```

这样虽然慢（需要额外的 PDE 求解），但至少结果是正确的。

## 📝 待更新

等 `simple_gamma_test.py` 完成后，根据结果更新本报告。

---

**报告日期**: 2025-10-31
**状态**: 🔴 调查中
**优先级**: 🔥 高（影响所有二阶 Greeks）
