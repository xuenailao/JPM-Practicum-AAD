# AAD Edge-Pushing Hessian计算框架

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**高效的二阶导数（Hessian）计算，用于期权定价的Greeks计算**

本项目实现了基于自动微分（AAD）的Hessian计算算法，特别针对金融衍生品的二阶Greeks（Gamma、Vanna、Volga等）计算进行优化。

---

## 🎯 核心特性

- ✅ **Algorithm 3（块形式）**：标准的反向模式Hessian计算
- ✅ **Algorithm 4（边推算法）**：稀疏感知的优化版本，**最高62×加速**
- ✅ **机器精度**：二阶导数精确到浮点运算精度
- ✅ **对称稀疏矩阵优化**：使用邻接表，仅存储非零元素
- ✅ **完整测试套件**：21/21测试通过，包含BSM、多项式、复杂函数

---

## 📊 性能对比

基于Black-Scholes-Merton期权定价模型的Greeks计算（5个参数）：

| 方法 | 时间 (ms) | 相对速度 | 精度 | 适用场景 |
|------|-----------|---------|------|---------|
| **BSM解析解** | 3.86 | 1.0× | 机器精度 | 仅BSM |
| **Bumping（有限差分）** | 2.84 | **0.74×** ⚡ | ~1e-5 | n<10参数 |
| **Algo3（块形式）** | 43.51 | 11.3× | 机器精度 | 通用 |
| **Algo4（边推优化）** | 7.55 | 2.0× | 机器精度 | 通用 ⭐ |
| **PDE-CN** | 462.39 | 120× | 0.1-6% | 数值解 |

**大规模性能**（n=200参数，95%稀疏度）：
- Algo4-优化 vs Algo4-原始：**62.25×加速** 🔥
- Algo4-优化 vs Algo3：**5-10×加速**

*详见 [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)*

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/yourusername/AAD.git
cd AAD
pip install -r requirements.txt
```

### 基础使用

```python
from aad_edge_pushing.algo3.algo4_optimized import compute_hessian
from aad_edge_pushing.aad.engine import global_tape
from aad_edge_pushing.aad.advar import ADVar

# 定义函数：f(x, y) = x² + xy + y²
global_tape.reset()
x = ADVar(2.0)
y = ADVar(3.0)
z = x*x + x*y + y*y

# 计算Hessian矩阵
H = compute_hessian(global_tape, seed=1.0)

# 结果（对称矩阵）
# H = [[2, 1],
#      [1, 2]]
```

### BSM Greeks计算

```python
from aad_edge_pushing.examples.bsm_greeks import algo4_greeks

# 计算所有二阶Greeks
greeks = algo4_greeks(
    S0=100,    # 标的价格
    K=100,     # 行权价
    T=1.0,     # 到期时间
    r=0.05,    # 无风险利率
    sigma=0.2  # 波动率
)

print(f"Gamma: {greeks['gamma']:.8f}")   # ∂²V/∂S²
print(f"Vanna: {greeks['vanna']:.8f}")   # ∂²V/∂S∂σ
print(f"Volga: {greeks['volga']:.8f}")   # ∂²V/∂σ²
```

*更多示例见 [examples/](examples/) 目录*

---

## 📚 算法原理

### Algorithm 3：块形式（Block Form）

**理论基础**：Griewank框架的标准反向Hessian算法

**复杂度**：O(ops × |predecessors|²)

**特点**：
- 遍历计算图的所有节点
- 对每个节点计算完整的二阶导数块
- 适用于密集问题

### Algorithm 4：边推算法（Edge-Pushing）

**理论基础**：Gower & Mello (2016) "A New Framework for the Computation of Hessians"

**复杂度**：O(ops + edges) —— **理论最优**

**核心优化**：
1. **稀疏性感知**：仅遍历Hessian的非零元素
2. **邻接表**：O(1)查找非零邻居（vs O(n)扫描）
3. **对称性利用**：仅存储上三角矩阵

**关键代码**：
```python
# 传统方法：O(n)扫描所有节点
for p in range(n_nodes):  # 15,100次迭代
    if W.get(p, i) != 0:  # 仅~50个非零
        # 处理...

# 边推优化：O(degree)仅访问非零邻居
neighbors = W.get_neighbors(i)  # 直接返回~50个非零邻居
for p, w_pi in neighbors:
    # 处理...
```

**实测效果**：
- Line profiler显示：原始版本67.6%时间消耗在O(n)扫描
- 优化后：n=200时加速62.25×

*详见 [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) 第7节*

---

## 🧪 运行基准测试

### 四方法对比

```bash
cd benchmarks
python main_benchmark.py
```

输出：
```
=== Greeks精度对比 ===
方法         Gamma误差(%)  Vanna误差(%)  Volga误差(%)
PDE-CN       0.135        0.251        5.827
Bumping      0.000        0.020        0.004
Algo3        0.000        0.000        0.000 ✓
Algo4        0.000        0.000        0.000 ✓

=== 性能对比 ===
方法         时间(ms)     相对BSM
PDE-CN       462.39      119.91×
Bumping      2.84        0.74×  ⚡最快
Algo3        43.51       11.28×
Algo4        7.55        1.96×  ⭐推荐
```

### 优化效果测试

```bash
python benchmarks/optimization_test.py
```

测试不同规模和稀疏度下的性能提升。

---

## 📁 项目结构

```
AAD/
├── README.md                      # 本文档
├── LITERATURE_REVIEW.md           # 学术文献综述
├── PERFORMANCE_REPORT.md          # 详细性能报告
├── requirements.txt               # Python依赖
│
├── aad_edge_pushing/              # 核心库
│   ├── aad/                       # 自动微分基础
│   │   ├── advar.py              # AD变量类（运算符重载）
│   │   ├── engine.py             # Tape记录引擎
│   │   └── ops/                  # 运算操作
│   │
│   ├── algo3/                     # Hessian算法
│   │   ├── algo3_block.py        # Algorithm 3
│   │   ├── algo4_edge_pushing.py # Algorithm 4（原始）
│   │   ├── algo4_optimized.py    # Algorithm 4（优化） ⭐
│   │   └── symm_sparse_optimized.py  # 对称稀疏矩阵+邻接表
│   │
│   └── examples/                  # 应用示例
│       └── bsm_greeks.py         # Black-Scholes-Merton Greeks
│
├── benchmarks/                    # 性能基准测试
│   ├── main_benchmark.py         # 四方法完整对比
│   └── optimization_test.py      # 优化效果验证
│
├── tests/                         # 单元测试
│   └── test_algo3_comprehensive.py  # 21个测试用例
│
├── examples/                      # 使用示例
│   ├── quick_start.py            # 5分钟快速上手
│   └── bsm_greeks_demo.py        # BSM完整示例
│
└── notebooks/                     # Jupyter notebooks
    └── PDE & BSM (Greeks of vanilla option).ipynb
```

---

## 🔬 技术细节

### 支持的操作

**基础运算**：`+`, `-`, `*`, `/`, `**`

**超越函数**：`exp`, `log`, `sqrt`, `sin`, `cos`, `tanh`

**特殊函数**：`erf`（误差函数，BSM必需）

### Hessian矩阵格式

返回`SymmSparseOptimized`对象：

```python
H = compute_hessian(tape)

# 访问元素
H.get(i, j)  # 获取H[i,j]

# 转换为NumPy数组
import numpy as np
n = len(tape.ops)
H_dense = np.array([[H.get(i,j) for j in range(n)] for i in range(n)])

# 获取非零邻居
neighbors = H.get_neighbors(i)  # [(j, H[i,j]) for j in adj[i]]
```

### 内存优化

- **对称存储**：仅存储(i,j) where i≤j
- **稀疏存储**：仅存储非零元素
- **邻接表**：O(1)邻居查找

**实例**（n=100, 99%稀疏）：
- 完整矩阵：100×100 = 10,000个元素
- 实际存储：~50个非零元素（**200×内存节省**）

---

## 📖 学术背景

本项目基于以下研究成果：

1. **Griewank & Walther (2008)**: "Evaluating Derivatives" —— AD理论基础
2. **Gower & Mello (2016)**: "A New Framework for the Computation of Hessians" —— Edge-Pushing算法
3. **Giles & Glasserman (2006)**: "Smoking Adjoints" —— AAD应用于金融
4. **Capriotti & Giles (2010)**: "Fast Correlation Greeks" —— 大规模Greeks计算

*完整文献综述见 [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)*

---

## 🎓 应用场景

### ✅ 推荐使用

- **解析期权定价公式**的二阶Greeks（BSM、Heston等）
- **复杂数学函数**的Hessian计算
- **中大规模问题**（n≥10个参数）
- **需要机器精度**的梯度/Hessian

### ⚠️ 不推荐使用

- **极小规模**（n<5）：直接用bumping更快
- **Monte Carlo模拟**：需要路径微分（未实现）
- **美式期权**：需要提前行权处理（未实现）
- **生产级大规模**：建议用C++ AAD库（dco/c++, NAG）

---

## 🛠️ 扩展方向

### 短期（已验证可行）
- [ ] Numba JIT加速核心循环（预期5-10×）
- [ ] 批量Greeks计算优化
- [ ] 更多期权类型（奇异期权、利率衍生品）

### 中期（研究中）
- [ ] Taylor展开方法（基于arXiv:2412.05300v2）
- [ ] Monte Carlo AAD（Giles-Glasserman方法）
- [ ] 图着色算法（Hessian压缩）

### 长期（战略方向）
- [ ] JAX迁移（GPU加速 + JIT）
- [ ] C++/Cython重写性能关键路径
- [ ] XVA计算（CVA, DVA, FVA）

*详见 [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md) 第8节*

---

## 🤝 贡献

欢迎贡献！特别是：

1. **新的期权类型**实现
2. **性能优化**（Numba、Cython等）
3. **更多测试用例**
4. **文档改进**

请提交Pull Request或Issue。

---

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

---

## 📬 联系方式

- **项目主页**: [GitHub仓库链接]
- **问题反馈**: [Issues](https://github.com/yourusername/AAD/issues)
- **文献综述**: [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md)
- **性能报告**: [PERFORMANCE_REPORT.md](PERFORMANCE_REPORT.md)

---

## 🙏 致谢

感谢以下开源项目和研究成果：

- Andreas Griewank的开创性AD研究
- Mike Giles & Paul Glasserman的金融AAD应用
- NumPy和SciPy社区

---

**⭐ 如果这个项目对你有帮助，请给个Star！**

---

*最后更新：2025年10月14日*
*基于：Algorithm 3/4完整实现 + 文献综述 + 性能验证*
