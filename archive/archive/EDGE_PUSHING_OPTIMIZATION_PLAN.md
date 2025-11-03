# Edge-Pushing框架优化方案
## 坚持AAD + Edge-Pushing，优化PDE Greeks计算

**目标**: 在保持Edge-Pushing框架的前提下，将Volga计算时间从42.5s优化到<10s

---

## 📊 当前状态分析

### 性能瓶颈定位

基于论文公式(33): `TIME = O(d* Σd*_i + ℓ)`

**当前实测数据** (原始PDE, Grid 51×50):
```
节点数 ℓ:           65,044
平均度数 d*_i:      ~100 (中期), ~500 (后期)
最大度数 d*:        ~500
Σd*_i:              ~10,000,000
d* × Σd*_i:         ~5×10⁹ operations

Hessian时间: 42.5s
```

### 复杂度来源

**Pushing Stage** ([algo4_adjlist.py:86-118](algo4_adjlist.py:86-118)):
```python
for p, w_pi in neighbors:  # O(d*_i) iterations
    if p == i:
        for j in preds:     # O(|preds|)
            for k in preds:  # O(|preds|)
                W.add(j, k, ...)  # ← O(|preds|²)
```

**单节点成本**: O(d*_i × |preds|²)

**PDE特征**:
- |preds| ≈ 3-10 (三对角 + 系数依赖)
- d*_i 随深度增长: 10 → 500
- 时间耦合导致W矩阵持续填充

---

## 🎯 优化方案 (坚持Edge-Pushing框架)

### 方案1: 单输入Hessian专用算法 ⭐⭐⭐

**核心思想**: 当只需要H[0,0]时，大量W元素计算是浪费的

#### 原理

完整Hessian:
```
计算所有 H[i,j], i,j ∈ inputs
对于Volga: 只需要 H[0,0] (σ是唯一输入)
```

**优化**: Early pruning不相关的W元素

#### 实现

```python
# 新文件: algo4_single_input.py

def algo4_single_input_hessian(output: ADVar, input_var: ADVar) -> float:
    """
    优化版Edge-Pushing: 只计算H[0,0]

    关键优化:
    1. 只追踪与input_var相关的路径
    2. 剪枝W中与H[0,0]无关的元素
    3. 提前终止不贡献到H[0,0]的分支
    """
    var_to_idx = _create_index_mapping(...)
    input_idx = var_to_idx[id(input_var)]

    # W矩阵: 只存储与input_idx相关的行/列
    W = SparseHessian_SingleInput(n_nodes, input_idx)

    for node in reversed(global_tape.nodes):
        i = var_to_idx[id(node.out)]

        # 检查节点i是否在input_idx的依赖路径上
        if not W.is_relevant(i, input_idx):
            continue  # ← 跳过无关节点

        # 只处理相关的W元素
        _pushing_stage_pruned(W, i, preds, d1, input_idx)
        _creating_stage_pruned(W, preds, d2, vbar[i], input_idx)

    return W.get(input_idx, input_idx)
```

#### 复杂度分析

**原始**:
```
计算所有n×n个W元素
成本: O(d* × Σd*_i)
```

**优化后**:
```
只计算与input相关的O(√n)个元素
成本: O(d* × √n) = O(500 × 250) = 125k
vs 原始: O(5×10⁹)

预期加速: ~40,000× (理论)
实际加速: ~10-20× (考虑开销)
```

#### 预期效果

- 时间: 42.5s → **2-5s**
- 精度: 保持不变 (9.08%误差)
- 内存: 减少到原来的1%

---

### 方案2: 利用PDE三对角结构 ⭐⭐

**核心思想**: PDE的Jacobian是三对角的，限制了依赖路径

#### 原理

三对角结构:
```
V[i] 只依赖 V[i-1], V[i], V[i+1]
→ |preds| ≤ 3 (空间维度)
加上时间依赖和系数: |preds| ≤ 10
```

**但W矩阵仍然密集?**

原因: 时间步累积
```
V^n 依赖 V^{n-1} (直接)
V^n 依赖 V^{n-2} (通过V^{n-1})
...
V^n 依赖 V^0 (通过链式传播)

→ W(n, k) ≠ 0 for k=0..n
→ 密集Hessian
```

#### 优化策略

**时间步分块**:
```python
def algo4_pde_structured(output, input_var, time_blocks):
    """
    利用时间步的弱耦合性

    策略:
    1. 将N个时间步分成K块 (K=5-10)
    2. 块内: 完整Edge-Pushing
    3. 块间: 只传播必要的边界信息
    """
    W_global = {}

    for block in time_blocks:
        # 块内完整计算
        W_block = edge_pushing_block(block)

        # 只提取块边界的W元素
        W_boundary = extract_boundary(W_block)

        # 合并到全局
        merge_boundary(W_global, W_boundary)

    return W_global[input_idx, input_idx]
```

**复杂度**:
```
原始: O(d* × Σd*_i)，其中Σd*_i ∝ N²M
分块: O(K × d*_block × Σd*_i_block)
      其中 d*_block ≈ d*/K, Σd*_i_block ≈ (N/K)²M

加速: ~K² = 25-100×
```

#### 预期效果

- 时间: 42.5s → **5-10s**
- 精度: 可能略降 (10-15%误差)
- 需要精细调优块大小

---

### 方案3: 混合精度 + 稀疏化 ⭐

**核心思想**: W矩阵中大部分小元素对最终H[0,0]贡献很小

#### 实现

```python
class AdaptiveSparseW:
    """
    自适应稀疏W矩阵

    策略:
    1. 只保留 |W(i,j)| > threshold 的元素
    2. 动态调整threshold
    3. 最后一层用高精度
    """
    def __init__(self, n, threshold=1e-6):
        self.W = {}
        self.threshold = threshold

    def add(self, i, j, val):
        current = self.W.get((i,j), 0.0)
        new_val = current + val

        if abs(new_val) > self.threshold:
            self.W[(i,j)] = new_val
        elif (i,j) in self.W:
            del self.W[(i,j)]  # 清理小元素

    def adapt_threshold(self, depth):
        # 后期降低threshold (更精确)
        if depth > 0.9 * total_depth:
            self.threshold = 1e-10
```

#### 复杂度

```
有效非零元素数: nnz ≈ 0.1 × n² (vs n²)
成本: O(d* × nnz) = O(0.1 × d* × Σd*_i)

加速: ~10×
```

#### 预期效果

- 时间: 42.5s → **4-8s**
- 精度: 略降 (10-12%误差，仍<目标15%)
- 内存: 减少10×

---

### 方案4: 算法融合 ⭐⭐⭐ (组合方案)

**核心思想**: 结合方案1+2+3的优势

#### 实现架构

```python
def algo4_pde_optimized(output, sigma_var,
                        M, N,
                        time_blocks=5):
    """
    终极优化版Edge-Pushing for PDE Volga

    组合优化:
    1. 单输入剪枝 (方案1)
    2. 时间步分块 (方案2)
    3. 自适应稀疏 (方案3)
    """
    input_idx = get_index(sigma_var)

    # 第一遍: 标记相关节点
    relevant_nodes = mark_relevant_nodes(
        output, input_idx
    )  # O(n)

    # 时间步分块
    blocks = partition_time_steps(N, K=time_blocks)

    W = AdaptiveSparseW(
        n_nodes,
        threshold=1e-6
    )

    for block_id, block in enumerate(blocks):
        # 调整精度
        if block_id == len(blocks) - 1:
            W.set_threshold(1e-10)

        # 块内Edge-Pushing (只处理相关节点)
        for node in reversed(block.nodes):
            i = node.index

            if i not in relevant_nodes:
                continue  # 跳过

            # 标准pushing (稀疏化)
            _pushing_stage_sparse(W, i, ...)

    return W.get(input_idx, input_idx)
```

#### 复杂度分析

```
组合加速:
  方案1: 40,000× (理论)
  方案2: 25× (分块)
  方案3: 10× (稀疏)

实际: 不能简单相乘 (有重叠)
保守估计: 50-100×

预期时间: 42.5s / 50 ≈ 0.8s
乐观估计: 42.5s / 100 ≈ 0.4s
```

#### 预期效果

- 时间: 42.5s → **0.5-1s** 🎯
- 精度: 9-12%误差 (可接受)
- 复杂度: 仍然理论上O(n³)，但常数因子极小

---

## 🛠️ 实施计划

### Phase 1: 基础优化 (3-5天)

**任务**:
1. 实现 `algo4_single_input.py` (方案1)
2. 基准测试
3. 验证精度

**里程碑**:
- 单元测试通过
- (51,50)网格: 42.5s → <5s
- Volga误差保持<10%

### Phase 2: 结构化优化 (5-7天)

**任务**:
1. 实现时间步分块 (方案2)
2. 实现自适应稀疏 (方案3)
3. 性能调优

**里程碑**:
- (51,50): <2s
- (101,100): <10s (当前超时)

### Phase 3: 融合优化 (3-5天)

**任务**:
1. 组合所有优化 (方案4)
2. 端到端测试
3. 文档和API

**里程碑**:
- (51,50): <1s
- (101,100): <5s
- Volga误差<12%

### Phase 4: 验证与集成 (2-3天)

**任务**:
1. 跨不同σ值测试
2. 与解析解对比
3. 集成到生产API

---

## 📊 预期结果对比

| 方案 | 时间 (51,50) | 时间 (101,100) | Volga误差 | 实施时间 |
|------|--------------|----------------|-----------|----------|
| **当前** | 42.5s | >180s (超时) | 9.08% | ✅ 完成 |
| **方案1** | 2-5s | 10-20s | 9% | 3-5天 |
| **方案2** | 5-10s | 20-40s | 10-15% | 5-7天 |
| **方案3** | 4-8s | 15-30s | 10-12% | 3-5天 |
| **方案4** | **0.5-1s** | **2-5s** | **10-12%** | **2-3周** |

---

## 🎯 技术关键点

### 1. 单输入优化的正确性

**证明**:
```
对于 f: ℝⁿ → ℝ, x ∈ ℝ
Hessian H ∈ ℝ¹ˣ¹ (标量)

Edge-Pushing计算所有 H[i,j]
但我们只需要 H[0,0]

W矩阵中只有通过节点0的路径会影响H[0,0]
→ 可以安全剪枝其他路径
```

### 2. 时间步分块的数学基础

**弱耦合假设**:
```
V^n(S, σ) 主要依赖 V^{n-1}
对 V^{n-k} (k>某个值) 的依赖很弱

Hessian:
∂²V^n/∂σ² ≈ ∂²V^{n-1}/∂σ² + 局部项

→ 可以分块计算，只传播边界
```

### 3. 稀疏化的误差控制

**自适应策略**:
```python
# 初期: threshold=1e-6 (激进剪枝)
# 中期: threshold=1e-8 (平衡)
# 后期: threshold=1e-10 (保守)

误差估计:
  累积截断误差 ≈ K × threshold × path_length
  K ≈ 1000 (操作数)
  path_length ≈ 100

  总误差 ≈ 1000 × 1e-6 × 100 = 0.1 (10%)
  vs Volga ≈ 10 → 相对误差1%

可控制！
```

---

## 🔧 实现细节

### 新文件结构

```
aad_edge_pushing/
├── edge_pushing/
│   ├── algo4_adjlist.py          # 现有
│   ├── algo4_single_input.py     # 新：方案1
│   ├── algo4_pde_structured.py   # 新：方案2
│   ├── algo4_adaptive_sparse.py  # 新：方案3
│   └── algo4_pde_optimized.py    # 新：方案4 (融合)
├── pde/
│   └── optimized_volga.py        # 高层API
└── tests/
    └── test_optimization_suite.py
```

### API设计

```python
from aad_edge_pushing.pde import compute_volga_optimized

# 自动选择最佳优化策略
volga = compute_volga_optimized(
    S0=100, K=100, T=1, r=0.05, sigma=0.20,
    grid=(51, 50),
    optimization_level=4  # 0-4: 无优化到全优化
)

# 精细控制
volga = compute_volga_optimized(
    ...,
    optimization={
        'single_input': True,      # 方案1
        'time_blocks': 5,           # 方案2
        'adaptive_sparse': True,    # 方案3
        'threshold': 1e-6
    }
)
```

---

## ✅ 成功标准

### 性能目标

- ✅ (51,50)网格: <1s (vs 当前42.5s)
- ✅ (101,100)网格: <5s (vs 当前超时)
- ✅ 可扩展到(151,150): <15s

### 精度目标

- ✅ Volga误差: <12% (vs 当前9.08%)
- ✅ Vega误差: 保持<1%
- ✅ 符号正确率: 100%

### 工程目标

- ✅ 代码质量: 完整测试覆盖
- ✅ 文档: 详细说明和示例
- ✅ 可维护性: 清晰的模块划分

---

## 🎓 理论贡献

通过这些优化，我们将证明：

**Edge-Pushing可以用于PDE Greeks计算**

前提是：
1. 利用问题结构（单输入、三对角）
2. 智能剪枝和分块
3. 自适应精度控制

这扩展了Edge-Pushing的应用范围，从：
- 原论文: 小规模稠密/大规模稀疏函数
- 扩展到: **中大规模结构化函数 (如PDE)**

---

## 🚀 下一步行动

建议立即开始：

### 第一周: 方案1实现
- 实现 `algo4_single_input.py`
- 验证正确性和性能
- 目标: 42.5s → <5s

### 第二周: 方案2+3实现
- 实现分块和稀疏化
- 性能调优
- 目标: <2s

### 第三周: 方案4融合
- 组合所有优化
- 端到端测试
- 目标: <1s

---

**坚持Edge-Pushing框架，通过算法优化达到实用性能！**
