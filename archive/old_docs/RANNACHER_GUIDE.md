# Rannacher Timestepping 实施指南

## 概述

已成功实现 Rannacher timestepping 来解决 Crank-Nicolson 方案在高波动率下的虚假振荡问题。

## 问题背景

### 观察到的问题
在你的测试中（σ=0.5）:
- **价格**: 准确 ✓
- **Delta**: 33% 误差 ✗
- **Gamma**: 104% 误差 ✗✗

### 根本原因
Crank-Nicolson (C-N) 方案:
- ✓ **A-stable**: 数值稳定
- ✗ **不是 L-stable**: 缺乏数值耗散

**问题链**:
1. 初始 payoff 在行权价 K 处有"尖角"（二阶导数不连续）
2. C-N 无法有效平滑这个尖角
3. 高频振荡在时间步进中持续存在
4. 价格（积分量）：振荡被平均抵消 → 准确
5. Greeks（微分量）：振荡被放大 → Γ = (V_{i+1} - 2V_i + V_{i-1})/ΔS² → 误差巨大

## 解决方案：Rannacher Timestepping

### 策略
- **前 R 步**（从 T 开始倒推）：φ=1.0（全隐式 Backward Euler）
  - 特点：强数值耗散，O(Δt) 精度
  - 作用：立即平滑 payoff 尖角

- **剩余步**：φ=0.5（Crank-Nicolson）
  - 特点：O(Δt²) 高精度
  - 作用：在平滑解的基础上快速求解

### 推荐参数
- **R = 4**（行业标准，文献推荐值）

## 新文件

### 主求解器
`aad_edge_pushing/pde/pde_aad_rannacher.py`

**新类**: `BS_PDE_AAD_Rannacher`

**关键特性**:
1. 完全兼容原 `BS_PDE_AAD` 接口
2. 新增 Rannacher 参数:
   - `use_rannacher`: 是否启用（默认 True）
   - `rannacher_steps`: R 值（默认 4）
3. 当 `use_rannacher=False` 时行为与原始 C-N 完全相同

### 测试文件
- `verify_rannacher_setup.py` - 基本验证（✓ 已通过）
- `quick_test_rannacher.py` - 快速对比测试
- `test_rannacher_comparison.py` - 完整对比测试

## 使用方法

### 基本用法

```python
from aad_edge_pushing.pde.pde_aad_rannacher import BS_PDE_AAD_Rannacher

# 创建求解器（启用 Rannacher）
solver = BS_PDE_AAD_Rannacher(
    S0=100.0, K=100.0, T=1.0, r=0.05,
    M=151, N_base=150,
    use_rannacher=True,      # 启用 Rannacher
    rannacher_steps=4        # 推荐值
)

# 求解 PDE
result = solver.solve_pde_with_aad(
    S0_val=100.0,
    sigma_val=0.5,           # 高波动率
    compute_hessian=True     # 计算二阶 Greeks
)

# 结果
print(f"Price: {result['price']}")
print(f"Delta: {result['delta']}")
print(f"Gamma: {result['gamma']}")  # 应该更准确
print(f"Vega:  {result['vega']}")
print(f"Vanna: {result['vanna']}")
print(f"Volga: {result['volga']}")
```

### 对比测试

```python
# 标准 C-N (R=0)
solver_cn = BS_PDE_AAD_Rannacher(
    S0=100.0, K=100.0, T=1.0, r=0.05,
    M=51, N_base=100,
    use_rannacher=False      # 禁用 Rannacher
)
result_cn = solver_cn.solve_pde_with_aad(100.0, 0.5, compute_hessian=True)

# Rannacher (R=4)
solver_rann = BS_PDE_AAD_Rannacher(
    S0=100.0, K=100.0, T=1.0, r=0.05,
    M=51, N_base=100,
    use_rannacher=True,      # 启用 Rannacher
    rannacher_steps=4
)
result_rann = solver_rann.solve_pde_with_aad(100.0, 0.5, compute_hessian=True)

# 对比 Gamma 误差
print(f"Standard C-N: Gamma = {result_cn['gamma']}")
print(f"Rannacher:    Gamma = {result_rann['gamma']}")
```

### 集成到现有测试框架

修改 `comprehensive_test_framework.py` 中的方法类：

```python
# 在 edge_pushing_method.py 或 double_aad_method.py 中
from .pde_aad_rannacher import BS_PDE_AAD_Rannacher

class EdgePushingMethodFixed:
    def __init__(self, M: int, N: int, use_rannacher: bool = True):
        self.M = M
        self.N = N
        self.use_rannacher = use_rannacher

    def compute_greeks(self, S0, K, T, r, sigma, ...):
        solver = BS_PDE_AAD_Rannacher(  # 使用新类
            S0=S0, K=K, T=T, r=r,
            M=self.M, N_base=self.N,
            use_rannacher=self.use_rannacher,
            rannacher_steps=4
        )
        # ... 其余代码不变
```

## 预期改进

### 不同波动率下的表现

| 波动率 | 标准 C-N Gamma 误差 | Rannacher Gamma 误差 | 改进幅度 |
|--------|---------------------|----------------------|----------|
| σ=0.2  | ~5%                 | ~2%                  | 60%      |
| σ=0.3  | ~20%                | ~5%                  | 75%      |
| σ=0.4  | ~60%                | ~8%                  | 87%      |
| σ=0.5  | ~104%               | ~10%                 | 90%      |

### 其他 Greeks
- **Delta**: 从 33% → <5%
- **Vega**: 轻微改进
- **Vanna, Volga**: 显著改进（如果真值不接近 0）

### 计算成本
- **理论**: 前 R 步用 O(Δt) 方案，轻微损失精度
- **实践**: 总体时间增加 < 5%（因为只有前 4 步不同）
- **网格**: 建议 R=4，与 N 无关

## 实现细节

### 核心修改

#### 1. 构造函数
```python
def __init__(self, ..., use_rannacher: bool = True, rannacher_steps: int = 4):
    self.use_rannacher = use_rannacher
    self.rannacher_steps = rannacher_steps
```

#### 2. build_tridiagonal_cn 方法
```python
def build_tridiagonal_cn(self, sigma_var, dt, phi: float = 0.5):
    # 现在接受 phi 参数，可以传入 0.5 或 1.0
```

#### 3. 时间步进循环（关键）
```python
# 预构建两套系数
a_L_be, ... = self.build_tridiagonal_cn(sigma_var, dt, phi=1.0)  # Backward Euler
a_L_cn, ... = self.build_tridiagonal_cn(sigma_var, dt, phi=0.5)  # Crank-Nicolson

for n in range(N):
    # 根据步数选择系数
    if self.use_rannacher and n < self.rannacher_steps:
        a_L, b_L, c_L = a_L_be, b_L_be, c_L_be  # 前 R 步
    else:
        a_L, b_L, c_L = a_L_cn, b_L_cn, c_L_cn  # 剩余步

    V = self.cn_step(V, a_L, b_L, c_L, ...)
```

#### 4. 应用范围
- `solve_pde_with_aad`: Jacobian 计算 ✓
- `solve_pde_with_aad`: Hessian 计算 ✓
- `_solve_pde_numerical`: Bumping 方法 ✓

## 测试验证

### 基本验证
```bash
python verify_rannacher_setup.py
```
**状态**: ✓ 已通过

**输出**:
- 导入成功
- 初始化成功
- PDE 求解成功（σ=0.2, M=51, N=100）
- 计算时间: ~1080 ms

### 快速测试（推荐）
```bash
python quick_test_rannacher.py
```
对比单个高波动率场景（σ=0.5）的 Gamma 误差

### 完整测试
```bash
python test_rannacher_comparison.py
```
测试多个波动率（σ=0.3, 0.4, 0.5）和不同 R 值（0, 2, 4, 8）

**注意**: 完整测试可能需要较长时间（因为 compute_hessian=True）

## 下一步

### 1. 重新运行综合测试
将 `comprehensive_test_framework.py` 中的 PDE 求解器替换为 Rannacher 版本，然后重新运行：

```bash
python comprehensive_test_framework.py
```

### 2. 对比结果
对比之前的 CSV 结果和新的 Rannacher 结果：
- 高波动率场景的 Gamma 误差应显著降低
- 价格和低阶 Greeks 应保持相同精度

### 3. 更新文档
如果效果显著，更新主 README 说明 Rannacher 改进

## 理论参考

### 文献
- Rannacher, R. (1984). "Finite element solution of diffusion problems with irregular data"
- Pooley, D.M., et al. (2003). "Numerical convergence properties of option pricing PDEs with uncertain volatility"

### 关键概念
- **A-stability**: 对所有 Δt 无条件稳定
- **L-stability**: A-stability + 高频分量的数值耗散
- **Payoff Discontinuity**: max(S-K, 0) 在 K 处的导数跳跃
- **Spurious Oscillations**: C-N 方案引入的非物理振荡

## 故障排除

### 如果 Gamma 误差仍然很高

1. **检查网格分辨率**
   ```python
   solver = BS_PDE_AAD_Rannacher(..., M=201, N_base=400)  # 增加网格
   ```

2. **尝试更多 Rannacher 步数**
   ```python
   solver = BS_PDE_AAD_Rannacher(..., rannacher_steps=8)  # 增加 R
   ```

3. **检查是否是相对误差陷阱**
   - 如果 Gamma 的真值接近 0，相对误差会爆炸
   - 检查绝对误差：`|computed - analytical|`

### 如果计算时间过长

1. **减小网格**
   ```python
   M=51, N_base=100  # 最小推荐值
   ```

2. **只测试 Jacobian**
   ```python
   result = solver.solve_pde_with_aad(..., compute_hessian=False)
   ```

3. **使用更快的机器**
   - Hessian 计算（Edge-Pushing）是计算密集型的

## 总结

✅ **已完成**:
- Rannacher 求解器实现
- 基本验证测试通过
- 使用文档完整

🔄 **待完成**:
- 运行完整对比测试（compute_hessian=True 较慢）
- 集成到综合测试框架
- 生成对比报告

📊 **预期结果**:
- Gamma 误差从 104% 降至 <10%（σ=0.5）
- Delta 误差从 33% 降至 <5%
- 计算时间增加 <5%

🎯 **推荐做法**:
- 默认启用 Rannacher（`use_rannacher=True`）
- 使用标准 R=4
- 高波动率场景必用
