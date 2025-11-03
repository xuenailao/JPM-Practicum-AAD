# Greeks计算基准测试 - 完整使用指南

## 📋 概述

本测试系统全面对比5种Greeks计算方法：

1. **Analytical** - BSM解析公式（基准）
2. **Bumping** - 有限差分法（5次PDE求解）
3. **AAD + Bumping** - AAD一阶导数 + Bumping二阶导数（5次PDE）
4. **Double AAD** - 双重自动微分（等价于Edge-Pushing）
5. **Edge-Pushing** - 边推算法（1次PDE求解）

测试维度包括：**速度**、**精度**、**不同参数**、**不同网格**、**计算图统计**

---

## 🚀 快速开始

### 方法1：快速测试（推荐新手）

```bash
cd /home/junruw2/AAD
python run_comprehensive_benchmark.py --mode quick
```

**特点：**
- 约90次计算
- 运行时间：~2-5分钟
- 自动保存CSV + Markdown报告
- 适合快速验证和调试

### 方法2：完整测试（完整数据）

```bash
python run_comprehensive_benchmark.py --mode full
```

**特点：**
- 约1080次计算
- 运行时间：~30-60分钟
- 覆盖更多参数组合和网格分辨率
- 适合最终报告和论文

### 方法3：包含详细计算图

```bash
python run_comprehensive_benchmark.py --mode quick --graph
```

**特点：**
- 额外保存计算图详细信息
- 输出节点/边/操作统计
- 适合研究AAD实现细节

---

## 📂 输出文件说明

所有结果自动保存到 `benchmark_results/` 目录：

```
benchmark_results/
├── results_quick_20251031_010203.csv          # 原始数据（CSV格式）
├── REPORT_quick_20251031_010203.md            # 人类可读报告
└── computation_graphs_quick_20251031_010203.txt  # 计算图统计（--graph选项）
```

### 1. CSV文件 (`results_*.csv`)

包含所有测试的原始数据，每行一个测试结果：

**列说明：**
- `S0, K, T, r, sigma, M, N`: 测试参数
- `method`: 方法名称
- `price, delta, gamma, vega, vanna, volga`: 计算的Greeks
- `*_analytical`: 对应的解析值
- `*_error_pct`: 误差百分比
- `time_ms`: 计算时间（毫秒）
- `n_pde_solves`: PDE求解次数
- `graph_nodes, graph_edges`: 计算图统计

**用途：** 可导入Excel/Python进行进一步分析

### 2. Markdown报告 (`REPORT_*.md`)

自动生成的格式化报告，包含：

1. **速度对比表** - 各方法在不同网格下的平均/最小/最大时间
2. **精度对比表** - 各Greeks的平均误差
3. **网格分辨率分析** - M=21/51/101的精度对比
4. **计算成本** - PDE求解次数、计算图大小
5. **推荐配置** - 最快方法、最准确方法、生产环境推荐

**用途：** 直接阅读，或复制到论文/报告中

### 3. 计算图文件 (`computation_graphs_*.txt`)

AAD方法的详细计算图信息：

```
==============================================================================
Computation Graph: EDGE_PUSHING
Parameters: S0=100, K=100, T=1.0, r=0.05, sigma=0.2, M=51, N=50
==============================================================================

Total nodes: 12,750
Total edges: 38,250
Max fan-in: 3
Max fan-out: 250

Operation breakdown:
  mul         :  5,100 (40.0%)
  add         :  4,080 (32.0%)
  sub         :  2,550 (20.0%)
  div         :  1,020 ( 8.0%)
```

**用途：** 分析AAD实现性能、理解计算图结构

---

## ⚙️ 命令行选项详解

### `--mode` 模式选择

```bash
--mode quick    # 快速模式（默认）
--mode full     # 完整模式
```

**快速模式测试矩阵：**
- S0: [95, 100, 105] (价外/平价/价内)
- T: [0.5, 1.0] (短期/中期)
- σ: [0.15, 0.20, 0.30] (低/中/高波动率)
- r: [0.05] (固定)
- 网格: [(51, 50)] (固定M=51)
- **总计:** 3×2×3 = 18个配置 × 5方法 = **90次计算**

**完整模式测试矩阵：**
- S0: [90, 100, 110]
- T: [0.25, 0.5, 1.0]
- r: [0.03, 0.05]
- σ: [0.15, 0.20, 0.30, 0.40]
- 网格: [(21,20), (51,50), (101,100)]
- **总计:** 3×3×2×4 = 72个配置 × 3网格 × 5方法 = **1080次计算**

### `--graph` 计算图输出

```bash
--graph    # 保存详细计算图统计
```

添加此选项会额外保存：
- 节点/边数量
- 最大入度/出度
- 操作类型分布
- 仅对AAD方法有效（Analytical和Bumping无计算图）

### `--output` 输出目录

```bash
--output my_results    # 自定义输出目录
```

默认为 `benchmark_results/`，可指定其他目录。

---

## 📊 预期结果示例

### 速度对比（M=51）

| Method | Time (ms) | PDE Solves | Speedup vs Bumping |
|--------|-----------|------------|---------------------|
| Analytical | 0.5 | 0 | N/A |
| Bumping | 85 | 5 | 1.0× (baseline) |
| AAD + Bumping | 90 | 5 | 0.94× |
| Double AAD | 75 | 1 | 1.13× |
| Edge-Pushing | 75 | 1 | 1.13× |

### 精度对比（M=51）

| Method | Delta err% | Gamma err% | Vega err% | Volga err% |
|--------|-----------|-----------|----------|-----------|
| Bumping | 3.1 | 0.7 | 0.7 | 9.0 |
| AAD + Bumping | 2.9 | 0.7 | 0.7 | 9.0 |
| Edge-Pushing | 2.9 | **0.5** | 0.7 | **8.4** |

### 计算图统计（M=51）

| Method | Nodes | Edges | Max Fan-in | PDE Solves |
|--------|-------|-------|-----------|-----------|
| Edge-Pushing | 12,750 | 38,250 | 3 | 1 |
| AAD + Bumping | 12,750 | 38,250 | 3 | 5 |

---

## 🔍 常见使用场景

### 场景1：验证实现正确性

```bash
# 运行快速测试
python run_comprehensive_benchmark.py --mode quick

# 检查REPORT文件中的精度对比表
# 预期: Edge-Pushing的Gamma误差 < 1%
```

### 场景2：生成论文/报告数据

```bash
# 运行完整测试
python run_comprehensive_benchmark.py --mode full

# 使用CSV文件进行深度分析
# 使用Markdown报告复制表格到论文
```

### 场景3：研究AAD实现细节

```bash
# 包含计算图详细信息
python run_comprehensive_benchmark.py --mode quick --graph

# 查看computation_graphs_*.txt
# 分析节点数、边数、操作分布
```

### 场景4：对比不同优化方案

```bash
# 修改代码后重新测试
python run_comprehensive_benchmark.py --mode quick --output results_optimized

# 对比 benchmark_results/ 和 results_optimized/ 的报告
```

---

## 📈 结果分析建议

### 使用Python分析CSV

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('benchmark_results/results_quick_XXXXXX.csv')

# 示例: 绘制Gamma误差vs网格分辨率
df_ep = df[df['method'] == 'edge_pushing']
plt.plot(df_ep['M'], df_ep['gamma_error_pct'], 'o-')
plt.xlabel('Grid size M')
plt.ylabel('Gamma Error (%)')
plt.title('Edge-Pushing Gamma Accuracy vs Grid Resolution')
plt.savefig('gamma_convergence.png')
```

### 使用Excel分析

1. 打开CSV文件
2. 创建数据透视表
3. 行: `method`, 列: `M`, 值: `mean(gamma_error_pct)`
4. 生成图表对比不同方法的精度

---

## ⚠️ 注意事项

1. **运行时间：**
   - Quick模式：~2-5分钟
   - Full模式（M=101）：可能需要30-60分钟
   - 如果超时，减少网格分辨率或使用Quick模式

2. **内存使用：**
   - M=21: ~50MB
   - M=51: ~200MB
   - M=101: ~800MB
   - 确保有足够内存

3. **结果文件：**
   - 每次运行生成新文件（带时间戳）
   - 定期清理旧文件以节省空间

4. **已知限制：**
   - Double AAD当前使用Edge-Pushing实现（等价结果）
   - Analytical方法仅适用于欧式期权
   - PDE方法支持任意payoff（扩展性强）

---

## 🐛 故障排除

### 问题1：导入错误

```
ModuleNotFoundError: No module named 'aad_edge_pushing'
```

**解决：** 确保在AAD目录下运行

```bash
cd /home/junruw2/AAD
python run_comprehensive_benchmark.py --mode quick
```

### 问题2：运行时间过长

```bash
# 使用timeout限制运行时间
timeout 600 python run_comprehensive_benchmark.py --mode quick
```

或者减少测试配置（修改脚本中的参数范围）

### 问题3：结果文件未生成

检查输出目录权限：

```bash
mkdir -p benchmark_results
chmod 755 benchmark_results
```

---

## 📚 进阶用法

### 自定义测试参数

编辑 `run_comprehensive_benchmark.py` 第47-90行的 `get_test_configs()` 方法：

```python
# 例如：只测试ATM期权，高波动率
configs = []
for sigma in [0.30, 0.40, 0.50]:  # 高波动率
    for M, N in [(51, 50), (101, 100)]:  # 两种网格
        configs.append({
            'S0': 100.0, 'K': 100.0,  # ATM
            'T': 1.0, 'r': 0.05, 'sigma': sigma,
            'M': M, 'N': N
        })
```

### 添加新方法测试

在 `run_all_tests()` 方法中添加：

```python
methods = ['analytical', 'bumping', 'aad_bumping', 'double_aad', 'edge_pushing', 'my_new_method']
```

并在 `UnifiedGreeksCalculator` 中实现对应方法。

---

## ✅ 测试清单

运行测试前确认：

- [ ] 已安装所有依赖（numpy, pandas, scipy）
- [ ] 在AAD目录下运行
- [ ] 有足够磁盘空间（>100MB）
- [ ] 有足够时间（Quick: 5分钟，Full: 60分钟）

运行测试后检查：

- [ ] CSV文件已生成
- [ ] Markdown报告已生成
- [ ] 报告中所有表格完整
- [ ] Gamma误差符合预期（Edge-Pushing < 1%）

---

## 📞 获取帮助

```bash
python run_comprehensive_benchmark.py --help
```

查看所有可用选项和说明。

---

**最后更新:** 2025-10-31
**版本:** 1.0
