# 综合测试框架配置说明

## 测试场景配置

### 1. Moneyness 测试（3个场景）
**目的：** 测试不同价值状态下的精度
- ATM (At-The-Money): S0=100, K=100
- ITM (In-The-Money): S0=110, K=100
- OTM (Out-of-The-Money): S0=90, K=100

**网格设置：** M=101, N=200 (标准网格，已从 M=51, N=100 提升以确保稳定性)

### 2. Sigma 测试（5个场景）
**目的：** 测试不同波动率下的精度
- σ = 0.1, 0.2, 0.3, 0.4, 0.5

**网格设置（自适应）：**
- σ < 0.4: M=101, N=200 (标准网格)
- σ ≥ 0.4: M=151, N=300 (高波动率专用网格)

**原因：** 高波动率（σ≥0.4）在粗网格下会导致负 Gamma 问题，需要更密集的网格来捕捉快速扩散

### 3. M 变化测试（4个场景）
**目的：** 测试空间网格分辨率的影响
- M = 51, 101, 151, 201

**固定参数：** N=150, ATM, σ=0.2

### 4. N 变化测试（3个场景）
**目的：** 测试时间步数的影响
- N = 100, 200, 400

**固定参数：** M=151, ATM, σ=0.2

## 总测试场景数
**15 个场景** (3 + 5 + 4 + 3)

## 测试方法（每个场景测试5种方法）

1. **BSM 解析解** - 基准真值
2. **Bumping** - 有限差分法
3. **Double-AAD** - AAD + Bumping 混合方法
4. **Edge-Pushing** - 边推进算法
5. **AAD 图统计** - 计算图分析

## 测试指标

### 精度指标
- Delta, Gamma, Vega, Vanna, Volga, Rho 的相对误差 (%)

### 速度指标
- 计算时间 (ms)
- PDE 求解次数
- 相对 Bumping 的加速比

### AAD 图统计
- 节点数量
- 边数量
- 每节点平均时间
- 节点类型分布

## 运行命令

```bash
# 1. 运行完整测试（预计 15-20 分钟）
python comprehensive_test_framework.py

# 2. 生成可视化报告
python visualize_comprehensive_results.py

# 3. 查看 HTML 报告
firefox visualization_output/comprehensive_report.html
```

## 输出文件

### CSV 数据文件
- `comprehensive_test_results_YYYYMMDD_HHMMSS.csv` - 主要结果
- `aad_graph_statistics_YYYYMMDD_HHMMSS.csv` - AAD 图统计

### 可视化图表
- `accuracy_comparison_all_greeks.png` - 精度对比
- `speed_comparison.png` - 速度对比
- `grid_sensitivity_analysis.png` - 网格敏感性
- `moneyness_sensitivity.png` - Moneyness 分析
- `sigma_sensitivity.png` - 波动率敏感性
- `aad_graph_statistics.png` - AAD 图统计
- `comprehensive_report.html` - 完整 HTML 报告

## 性能优化

### ⚠️ 网格设置已更新（2025-10-31）

**原因：** 发现在高波动率（σ=0.5）下，原始快速网格（M=51, N=100）会导致负 Gamma 问题

### 标准网格 (M=101, N=200) - 新默认值
用于 Moneyness 测试和低-中波动率场景（σ < 0.4）
- 优点：对大多数场景足够稳定
- 每个场景约 2-3 分钟
- 相比原始网格速度降低 4×，但保证正确性

### 高波动率网格 (M=151, N=300)
用于高波动率场景（σ ≥ 0.4）
- 优点：确保高扩散系数下的数值稳定性
- 每个场景约 5-8 分钟
- 必需：防止 Gamma 符号错误

### 粗网格 (M=51, N=100) - 已弃用
- ❌ 不再推荐：在 σ≥0.4 时会产生负 Gamma
- 仅在网格敏感性测试中使用以展示问题

### 网格变化测试
M: 51 → 101 → 151 → 201
N: 100 → 200 → 400
- 目的：分析网格分辨率对精度和速度的影响
- **重要：** M=51, N=100 在高 σ 下会失败，这是预期行为

## 预期结果

### 精度排序（预期）
1. BSM 解析解：0% 误差（基准）
2. Edge-Pushing：< 2% 误差
3. Double-AAD：< 2% 误差
4. Bumping：< 5% 误差

### 速度排序（预期）
1. BSM 解析解：最快（< 1 ms）
2. Edge-Pushing：快（约 500-1000 ms）
3. Double-AAD：中等（约 800-1500 ms）
4. Bumping：慢（约 5000-10000 ms）

### 加速比（预期）
- Edge-Pushing vs Bumping: 5-10x
- Double-AAD vs Bumping: 3-8x

## 注意事项

1. **Vanna/Volga 误差**：当真值接近 0 时，相对误差可能很大，但绝对误差很小
2. **内存使用**：大网格（M=201, N=400）可能需要 2-4 GB 内存
3. **运行时间**：完整测试约 30-45 分钟（由于网格增大），建议在后台运行
4. **⚠️ 负 Gamma 问题**：如果在高波动率下看到负 Gamma，请检查网格设置是否足够大（建议 M≥151, N≥300 for σ≥0.4）

## 故障排除

### 如果测试太慢
- 减少 M, N 值
- 减少 Sigma 测试点数
- 只运行部分场景

### 如果内存不足
- 减小最大 M 值（从 201 降到 151）
- 减小最大 N 值（从 400 降到 200）

### 如果出现数值错误

#### 症状：Gamma 为负值
**原因：** 网格分辨率不足以捕捉高波动率扩散
**解决：**
1. 检查波动率 σ：如果 σ≥0.4，使用 M≥151, N≥300
2. 测试不同网格：运行 `test_high_sigma_gamma.py` 确认问题
3. 参考文档：查看 [GRID_FIX_SUMMARY.md](GRID_FIX_SUMMARY.md)

#### 症状：Vanna/Volga 误差超过 1000%
**原因：** 相对误差陷阱（真值接近 0）
**解决：**
1. 检查绝对误差：如果 < 0.01，可忽略
2. 调整参数：远离 Vanna/Volga≈0 的区域
3. 参考文档：查看 [RELATIVE_ERROR_TRAP_EXPLAINED.txt](RELATIVE_ERROR_TRAP_EXPLAINED.txt)

#### 症状：PDE 求解失败或 NaN
**原因：** PDE 稳定性条件违反或边界条件问题
**解决：**
1. 增加时间步数 N（减小 Δt）
2. 检查边界条件设置（S_max 是否足够大）
3. 验证参数合理性（r, σ, T 是否异常）
