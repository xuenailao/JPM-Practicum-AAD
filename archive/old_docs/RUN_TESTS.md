# 快速运行指南

## 问题修复

✅ **已修复**：Edge-Pushing 方法现在会计算完整的 Hessian（包括 vanna 和 volga）

## 运行完整测试

```bash
# 在后台运行完整测试（预计 15-20 分钟）
nohup python comprehensive_test_framework.py > test_output.log 2>&1 &

# 查看实时输出
tail -f test_output.log

# 或者直接在前台运行
python comprehensive_test_framework.py
```

## 测试配置（已优化）

### 快速测试场景（M=51, N=100）
- 3 个 Moneyness 场景
- 5 个 Sigma 场景

### 网格变化场景
- 4 个 M 值：51, 101, 151, 201
- 3 个 N 值：100, 200, 400

**总计：15 个测试场景**

## 生成可视化报告

```bash
# 测试完成后运行
python visualize_comprehensive_results.py

# 会自动找最新的测试结果文件
# 或者手动指定文件
python visualize_comprehensive_results.py comprehensive_test_results_20251031_XXXXXX.csv
```

## 查看结果

```bash
# 在浏览器中打开 HTML 报告
firefox visualization_output/comprehensive_report.html

# 或
google-chrome visualization_output/comprehensive_report.html

# 或直接在文件管理器中双击打开
xdg-open visualization_output/comprehensive_report.html
```

## 输出文件位置

### CSV 数据
- `comprehensive_test_results_YYYYMMDD_HHMMSS.csv`
- `aad_graph_statistics_YYYYMMDD_HHMMSS.csv`

### 可视化图表（在 `visualization_output/` 文件夹）
1. `accuracy_comparison_all_greeks.png` - 所有 Greeks 精度对比
2. `speed_comparison.png` - 计算速度和加速比
3. `grid_sensitivity_analysis.png` - M 和 N 的影响
4. `moneyness_sensitivity.png` - ATM/ITM/OTM 分析
5. `sigma_sensitivity.png` - 波动率影响
6. `aad_graph_statistics.png` - AAD 计算图统计
7. `comprehensive_report.html` - 完整 HTML 报告 ⭐

## 测试内容

每个场景测试 **5 种方法**：

1. **BSM 解析解** - 基准真值（< 1 ms）
2. **Bumping** - 有限差分（慢但稳定）
3. **Double-AAD** - AAD + Bumping 混合
4. **Edge-Pushing** - 边推进算法（最快的 AAD 方法）
5. **AAD 图统计** - 分析计算图结构

每种方法计算 **6 个 Greeks**：
- Delta (Δ) - 一阶：∂V/∂S
- Gamma (Γ) - 二阶：∂²V/∂S²
- Vega (ν) - 一阶：∂V/∂σ
- Vanna - 二阶：∂²V/∂S∂σ
- Volga - 二阶：∂²V/∂σ²
- Rho (ρ) - 一阶：∂V/∂r

## 预期结果

### 精度（相对误差）
- **Delta, Vega, Rho**: < 1%
- **Gamma**: < 5%
- **Vanna, Volga**: < 10% (注意：真值接近 0 时相对误差会变大)

### 速度（加速比 vs Bumping）
- **Edge-Pushing**: 5-10x 加速
- **Double-AAD**: 3-8x 加速
- **BSM 解析解**: 1000x+ 加速（但只适用于简单的 BS 模型）

## 故障排除

### 如果测试卡住或太慢
```bash
# 查看进程
ps aux | grep python | grep comprehensive

# 如果需要终止
pkill -f comprehensive_test_framework

# 减小测试规模（编辑 comprehensive_test_framework.py）
# 修改第 113-114 行：fast_M = 51, fast_N = 100
# 改为：fast_M = 31, fast_N = 50
```

### 如果内存不足
```bash
# 检查内存使用
free -h

# 减小最大网格大小
# 编辑第 149 行：M_values = [51, 101, 151, 201]
# 改为：M_values = [51, 101, 151]
```

### 如果想快速验证
```bash
# 只运行一个场景测试
python quick_test_comprehensive.py
```

## 监控测试进度

```bash
# 查看正在运行的场景
tail -n 50 test_output.log

# 查看已完成的测试数
grep "✓" test_output.log | wc -l

# 估计剩余时间
# 每个场景约 1-3 分钟
# 总共 15 个场景 = 15-45 分钟
```

## 重要提示

⚠️ **Vanna 和 Volga 的相对误差**：
- 当这些 Greeks 的真值接近 0 时，即使绝对误差很小，相对误差也可能很大
- 这是正常现象，不代表数值方法失败
- 详见 `VANNA_VOLGA_ACCURACY_ANALYSIS.md`

✅ **AAD 方法的优势**：
- 一次前向传播 + 一次反向传播 = 所有一阶导数
- Edge-Pushing：一次计算 = 完整 Hessian
- 相比 Bumping 需要多次 PDE 求解，效率提升显著

📊 **查看 AAD 计算图**：
- HTML 报告中包含节点数、边数统计
- 可以看到 Jacobian 和 Hessian 计算的复杂度差异
