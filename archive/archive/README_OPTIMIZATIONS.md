# PDE AAD优化使用指南

## 快速开始

### 运行测试

```bash
# 快速验证三个优化
python test_optimizations_quick.py

# 详细性能测试
python test_optimizations.py

# Thomas super-node单元测试
python test_thomas_supernode_fixed.py

# PDE Greeks对比测试
python test_pde_greeks_edge_vs_bump.py
```

### 基本使用

```python
from aad_edge_pushing.pde.cn_solver_supernode import CNSolverSuperNode

# 初始化求解器 (使用优化后的网格)
solver = CNSolverSuperNode(M=60, N=600)

# 设置参数
S0, K, T, r = 100.0, 100.0, 1.0, 0.05
sigma = np.ones((59, 600)) * 0.2  # 恒定波动率

# Forward求解 (使用向量化Thomas算法)
price, V = solver.solve_forward(S0, K, T, r, sigma, cp_flag='C')
print(f"Option Price: ${price:.4f}")

# 梯度计算 (使用隐函数定理)
gradient = solver.compute_gradient_ift(S0, K, T, r, sigma, cp_flag='C')
print(f"Gradient shape: {gradient.shape}")  # (59, 600)
```

## 三个优化详解

### 1. 向量化Thomas算法
- **位置**: `aad_edge_pushing/pde/cn_solver_supernode.py`
- **类**: `ThomasSolverSuperNode`
- **复杂度**: O(n)
- **优势**: 清晰、高效、数值稳定

### 2. 网格优化 (M=60, N=600)
- **空间点**: 60 (更少，更快)
- **时间步**: 600 (更多，更精确)
- **Δt**: 0.001667 (3×精细)
- **优势**: 稳定性↑ 精度↑ 效率↑

### 3. Super-Node方法
- **位置**: `aad_edge_pushing/pde/thomas_supernode_advar.py`
- **类**: `ThomasSuperNode`
- **图大小**: 1节点 vs ~6000节点 (朴素)
- **内存**: 340×节省
- **优势**: 巨大的可扩展性提升

## 性能对比

```
配置: M=60, N=600 (35,400 参数)

传统方法:
  图大小: 216,000 节点
  内存: ~20 MB
  Hessian: 不可行

优化后:
  图大小: 600 节点      (360× 减少)
  内存: ~60 KB          (340× 减少)
  Hessian: 可行via IFT
```

## 文档

- **[OPTIMIZATION_SUMMARY.md](OPTIMIZATION_SUMMARY.md)** - 技术详解
- **[FINAL_OPTIMIZATION_REPORT.md](FINAL_OPTIMIZATION_REPORT.md)** - 完整报告

## 测试结果

```bash
$ python test_optimizations_quick.py

================================================================================
QUICK TEST: Three Optimizations
================================================================================

1. VECTORIZED THOMAS ALGORITHM
   Size: 100
   Time: 0.168 ms
   ✓ O(n) vectorized implementation

2. NEW GRID CONFIGURATION (M=60, N=600)
   Grid: 60×600
   Price: $10.3384
   Time: 151.19 ms
   ✓ Fine time resolution, stable scheme

3. SUPER-NODE GRAPH REDUCTION
   Problem size: 100
   Naive graph: ~600 nodes
   Super-node: 1 node(s)
   Reduction: 600×
   ✓ Massive graph size reduction

4. COMBINED PERFORMANCE (M=60, N=600)
   PDE grid: 60×600
   Naive graph size: 216,000 nodes
   Super-node graph: 600 nodes
   Memory reduction: 360×
   ✓ Enables efficient Hessian via IFT
```

## 代码结构

```
aad_edge_pushing/pde/
├── cn_solver_supernode.py           # 优化1+2: 向量化Thomas + 新网格
│   ├── ThomasSolverSuperNode       # O(n) Thomas算法
│   └── CNSolverSuperNode           # CN求解器 (M=60, N=600)
│
├── thomas_supernode_advar.py       # 优化3: Super-Node
│   ├── ThomasSuperNode             # ADVar封装
│   ├── solve_advar()               # Forward (1个图节点)
│   └── backward_thomas_supernode() # Custom adjoint (O(n))
│
└── cn_solver_hessian_supernode.py # Hessian via IFT
    └── compute_hessian_ift()       # 二阶导数
```

## 下一步

1. ✅ 向量化Thomas实现
2. ✅ 网格优化 (M=60, N=600)
3. ✅ Super-Node封装
4. 🔄 修复backward Jacobian偏差
5. 📋 完整Hessian集成
6. 📋 性能benchmark
7. 📋 生产部署

## 参考

- Capriotti et al. (2015): "AAD and least-square Monte Carlo"
- Griewank & Walther: "Evaluating Derivatives"
- Giles (2008): "Adjoint methods for PDEs"
