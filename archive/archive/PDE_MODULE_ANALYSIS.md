# PDE 模块分析报告

## 目录结构总览

aad_edge_pushing/pde/
├── core/                 # 核心PDE求解器
├── models/              # 波动率模型
├── graph/               # 稀疏结构/邻接图
├── handcraft_aad/       # 手工编写的Hessian计算
├── greeks/              # Greeks计算器
└── aad_integration/     # ADVar自动微分集成

## 依赖关系层次

### 第1层：基础模块（无内部依赖）
1. models/svi_model.py - SVI波动率模型
   - 仅依赖：numpy, scipy

### 第2层：核心求解器
2. core/local_vol_solver.py - 局部波动率PDE求解器
   - 依赖：models/svi_model.py
   - 实现：Crank-Nicolson格式求解BS PDE
   - 包含类：
     * LocalVolSolver: 正向求解
     * LocalVolAdjoint: 带伴随方法的求解器

3. graph/adjacency_graph.py - 参数邻接图
   - 无内部依赖
   - 实现：构建σ[i,n]参数间的依赖关系图

### 第3层：Hessian计算方法（手工实现）
4. handcraft_aad/hessian_computation.py - 基础Hessian计算
   - 依赖：core/, graph/, models/
   - 方法：有限差分法计算Jacobian
   - 复杂度：O(P²)

5. handcraft_aad/hessian_edge_pushing.py - Edge-Pushing优化
   - 依赖：core/, graph/
   - 方法：利用稀疏性的智能Hessian计算
   - 复杂度：O(P × avg_neighbors) ≈ O(P)
   - 加速比：10-100×

6. handcraft_aad/second_order_adjoint.py - 二阶伴随方法
   - 依赖：core/
   - 方法：真正的二阶伴随AD

7. handcraft_aad/true_second_order_ad_optimized.py - 优化的二阶AD
   - 依赖：core/
   - 优化：缓存切向/伴随向量

### 第4层：应用层
8. greeks/second_order_greeks.py - 二阶Greeks计算
   - 依赖：handcraft_aad/hessian_edge_pushing, core/, graph/
   - 功能：计算Vanna, Volga, Cross-Gamma

### 第5层：自动微分集成（独立实现路径）
9. aad_integration/pde_aad_solver.py - ADVar隐式格式
   - 依赖：../../aad/core/var, ../../edge_pushing/algo4, models/
   - 方法：使用ADVar构建计算图的隐式CN格式

10. aad_integration/pde_aad_edge_pushing.py - ADVar显式格式
    - 依赖：../../aad/core/var, ../../edge_pushing/algo4
    - 方法：使用ADVar的简化显式格式

11. aad_integration/capriotti_cn_aad.py - Capriotti修正版
    - 依赖：../../aad/core/var, ../../edge_pushing/algo4
    - 特点：修正边界条件，通过BS解析解验证

## 功能实现对比

### 三种Hessian计算路径

#### 路径A: 手工邻接图 + Edge-Pushing
文件：handcraft_aad/hessian_edge_pushing.py
流程：PDE结构 → 手工构建邻接图 → 稀疏Hessian
优点：
  - 明确控制稀疏结构
  - 针对PDE优化，效率高
  - 适合生产环境
缺点：
  - 需要手工分析PDE依赖关系
  - 不够自动化

#### 路径B: ADVar隐式格式 + Algorithm 4
文件：aad_integration/pde_aad_solver.py
流程：ADVar PDE操作 → 自动计算图 → Algorithm 4提取Hessian
优点：
  - 完全自动化
  - 通用性强
  - 图结构自动捕获所有依赖
缺点：
  - 隐式格式需要求解线性系统（复杂）
  - 计算图可能很大

#### 路径C: ADVar显式格式 + Algorithm 4  
文件：aad_integration/pde_aad_edge_pushing.py, capriotti_cn_aad.py
流程：ADVar显式/半隐式 → 计算图 → Algorithm 4
优点：
  - 相对简单的ADVar集成
  - 自动化程度高
  - Capriotti版本有BS验证
缺点：
  - 显式格式可能有稳定性问题
  - 网格限制（M=20-50）

### 求解器对比

| 求解器 | 格式 | 网格规模 | AAD方式 | 适用场景 |
|--------|------|---------|---------|----------|
| LocalVolSolver | Crank-Nicolson | 200×200 | 无 | 正向定价 |
| LocalVolAdjoint | Crank-Nicolson | 200×200 | 手工伴随 | 一阶Greeks |
| PDEAADSolver | Crank-Nicolson | 100×100 | ADVar隐式 | 研究/概念验证 |
| PDEAADEdgePushing | 显式 | 20×20 | ADVar | 演示概念 |
| CapriottiCNAAD | θ-scheme | 20-100 | ADVar半隐式 | 学术研究 |

## 主要区别

### 1. 手工 vs 自动微分
- **handcraft_aad/**: 手工推导伴随方程，显式编程稀疏结构
- **aad_integration/**: 使用ADVar自动构建计算图

### 2. 稀疏性利用方式
- **手工方法**: 通过LocalVolAdjacency预先构建邻接图
- **自动方法**: Algorithm 4从计算图自动提取稀疏结构

### 3. 数值格式
- **Crank-Nicolson (隐式)**: 需要求解三对角系统，稳定性好，网格可大
- **显式格式**: 直接更新，简单但稳定性限制，网格必须小

### 4. 应用目标
- **core/ + handcraft_aad/**: 生产级别，高性能，大规模网格
- **aad_integration/**: 研究验证，概念证明，算法比较

## 依赖流向图

```
models/svi_model
        ↓
core/local_vol_solver ←──┐
        ↓                 │
graph/adjacency_graph     │
        ↓                 │
handcraft_aad/            │
├─ hessian_computation    │
├─ hessian_edge_pushing   │
├─ second_order_adjoint   │
└─ true_second_order_ad_optimized
        ↓
greeks/second_order_greeks


独立路径（使用外部AAD框架）:
../../aad/core/var + ../../edge_pushing/algo4
        ↓
aad_integration/
├─ pde_aad_solver
├─ pde_aad_edge_pushing  
└─ capriotti_cn_aad
```

## 性能特点

1. **LocalVolSolver**: O(N×M) 正向求解
2. **LocalVolAdjoint**: O(N×M) 一阶导数（伴随法）
3. **HessianComputer**: O(P²) 朴素Hessian
4. **HessianEdgePushing**: O(P×d) 稀疏Hessian，d是平均邻居数
5. **PDEAADSolver**: O(graph_size) 取决于计算图大小
6. **Algorithm 4**: O(|E|) E是边数，对稀疏图很高效

加速比实测：
- Edge-Pushing vs 朴素方法: 10-100×
- 聚焦ATM区域: 额外2-5×加速

## 使用建议

### 生产环境
使用：core/ + handcraft_aad/hessian_edge_pushing
原因：高性能，大网格，经过优化

### 研究/验证
使用：aad_integration/capriotti_cn_aad
原因：有解析解验证，易于修改

### 学习理解
使用：aad_integration/pde_aad_edge_pushing
原因：代码简单清晰，概念明确

### 二阶Greeks
使用：greeks/second_order_greeks
原因：专门优化，提供Vanna/Volga计算
## 详细文件列表与代码规模

### core/
  - local_vol_solver.py                              445 行

### models/
  - svi_model.py                                     338 行

### graph/
  - adjacency_graph.py                               303 行

### handcraft_aad/
  - hessian_computation.py                           349 行
  - hessian_edge_pushing.py                          285 行
  - second_order_adjoint.py                          339 行
  - true_second_order_ad_optimized.py                459 行

### greeks/
  - second_order_greeks.py                           294 行

### aad_integration/
  - capriotti_cn_aad.py                              397 行
  - pde_aad_edge_pushing.py                          341 行
  - pde_aad_solver.py                                395 行


## 关键代码片段解释

### 1. 邻接图构建逻辑 (graph/adjacency_graph.py)

参数σ[i,n]和σ[j,m]相邻当且仅当：
- 空间相邻：|i-j| ≤ 1 (三对角耦合)
- 时间传播：n ≤ m (因果性：早期参数影响后期)
- 或者在同一时间步：n = m且|i-j| ≤ 1

这种结构使得P个参数的邻接图非常稀疏！

### 2. Edge-Pushing关键思想 (handcraft_aad/hessian_edge_pushing.py)

朴素方法：对所有(i,n)和(j,m)计算H[i,n,j,m] → O(P²)
Edge-Pushing：只对相邻参数计算 → O(P × avg_degree)

对于PDE：avg_degree ≈ 5-10（远小于P=40000）
因此加速比 ≈ P / avg_degree ≈ 4000-8000×（理论）
实际：10-100×（因为有基础开销）

### 3. ADVar自动图构建 (aad_integration/)

关键创新：
```python
# 每个PDE操作都用ADVar
V[i] = alpha[i] * V[i-1] + beta[i] * V[i] + gamma[i] * V[i+1]
# ↓ 自动记录到tape
global_tape.record_operation(...)
# ↓ Algorithm 4自动提取稀疏Hessian
H = algo4_adjlist(output_var)
```

无需手工分析依赖关系！

### 4. Crank-Nicolson vs 显式格式

**Crank-Nicolson (隐式)**:
```
(I - 0.5*dt*L) V^{n+1} = (I + 0.5*dt*L) V^n
```
- 需要求解线性系统（三对角）
- 无条件稳定：可用大时间步
- 二阶精度：O(dt²) + O(dS²)

**显式格式**:
```
V^{n+1} = V^n + dt * L * V^n
```
- 直接计算，无需求解
- 稳定性条件：dt ≤ C*dS²（严格限制）
- 一阶精度：O(dt) + O(dS²)

### 5. 为什么有两套实现？

**handcraft_aad/**:
- 目标：最高性能生产代码
- 方法：手工推导伴随方程
- 优点：完全控制，针对性优化
- 网格：200×200（大规模）

**aad_integration/**:
- 目标：验证自动微分方法的可行性
- 方法：ADVar算子重载自动构建图
- 优点：通用性，易于扩展到其他问题
- 网格：20-100（受计算图大小限制）

## 核心算法流程

### 手工Edge-Pushing流程
```
1. 初始化LocalVolSolver(M=200, N=200)
2. 构建LocalVolAdjacency邻接图
   └─ 分析PDE三对角结构 → 稀疏邻接表
3. 正向求解：solver.solve() → V, dV/dS, ...
4. 计算Jacobian：solver.compute_jacobian() → ∂V/∂σ[i,n]
5. Edge-Pushing Hessian:
   for each σ[i,n]:
     for each neighbor σ[j,m] in adjacency[i,n]:
       H[i,n,j,m] = finite_diff(∂V/∂σ[i,n], perturb σ[j,m])
   └─ 只计算O(P×d)项，不是O(P²)
```

### ADVar自动化流程
```
1. 初始化CapriottiCNAAD(M=50, N=50)
2. 创建ADVar参数：sigma = [ADVar(σ_i) for i in range(P)]
3. PDE求解（ADVar操作）:
   for n in range(N):
     构建系数(ADVar) → c, u, l
     三对角求解(ADVar) → V_{n+1}
     └─ 每步自动记录到global_tape
4. 终点值：price = interpolate(V, S0) ← ADVar
5. Algorithm 4:
   H = algo4_adjlist(price)
   └─ 自动从tape提取稀疏Hessian
```

## 典型应用案例

### 案例1: 计算ATM期权的Volga
```python
from aad_edge_pushing.pde import SecondOrderGreeks

greeks = SecondOrderGreeks(M=200, N=200)
vanna, volga = greeks.compute_vanna_volga(
    S0=100, K=100, T=1.0, r=0.05, sigma0=0.2
)
print(f"Volga = {volga:.6f}")  # ∂²V/∂σ²
```

内部使用：
- LocalVolAdjoint (Crank-Nicolson 200×200)
- HessianEdgePushing (稀疏Hessian)
- 聚焦ATM区域参数

### 案例2: 验证ADVar方法精度
```python
from aad_edge_pushing.pde.aad_integration import CapriottiCNAAD

solver = CapriottiCNAAD(M=50, N=50)
result = solver.test_with_black_scholes(sigma=0.2)

print(f"PDE price: {result['pde_price']:.4f}")
print(f"BS price:  {result['bs_price']:.4f}")
print(f"Error:     {result['error']:.2e}")
print(f"Gamma (AAD): {result['gamma_aad']:.6f}")
print(f"Gamma (BS):  {result['gamma_bs']:.6f}")
```

对比：
- AAD自动微分 vs 解析公式
- 验证实现正确性

## 总结

PDE模块实现了**两条平行路径**：

### 路径1: 生产级手工优化
core → graph → handcraft_aad → greeks
- 适用场景：实际交易系统，风险管理
- 性能：最优（200×200网格，10-100×加速）
- 灵活性：中等（修改需重新分析依赖）

### 路径2: 研究级自动微分
aad/core/var → aad_integration
- 适用场景：算法研究，概念验证
- 性能：一般（50×50网格，图开销大）
- 灵活性：高（自动构建，易扩展）

两者结合展示了从**研究到生产**的完整链条！
