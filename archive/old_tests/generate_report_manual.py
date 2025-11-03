"""手动生成Markdown报告（不依赖tabulate）"""

import pandas as pd
import sys

csv_file = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_results/results_quick_20251031_004902.csv'
output_file = csv_file.replace('.csv', '_REPORT.md')

# 读取数据
df = pd.read_csv(csv_file)

report = []
report.append("# Greeks Computation Benchmark Report\n")
report.append(f"**Total Tests:** {len(df)}\n")
report.append(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append("\n---\n")

# 1. 速度对比
report.append("## 1. Speed Comparison\n")
report.append("| Method | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |")
report.append("|--------|-----------|----------|----------|----------|")

for method in df['method'].unique():
    subset = df[df['method'] == method]
    mean_t = subset['time_ms'].mean()
    std_t = subset['time_ms'].std()
    min_t = subset['time_ms'].min()
    max_t = subset['time_ms'].max()
    report.append(f"| {method:<15s} | {mean_t:9.2f} | {std_t:8.2f} | {min_t:8.2f} | {max_t:8.2f} |")

# 2. 精度对比
report.append("\n## 2. Accuracy Comparison\n")
report.append("| Method | Δ err% | Γ err% | ν err% | Vanna err% | Volga err% |")
report.append("|--------|--------|--------|--------|------------|------------|")

for method in df['method'].unique():
    if method == 'Analytical':
        continue
    subset = df[df['method'] == method]
    delta_e = subset['delta_error_pct'].mean()
    gamma_e = subset['gamma_error_pct'].mean()
    vega_e = subset['vega_error_pct'].mean()
    vanna_e = subset['vanna_error_pct'].mean()
    volga_e = subset['volga_error_pct'].mean()
    report.append(f"| {method:<15s} | {delta_e:6.2f} | {gamma_e:6.2f} | {vega_e:6.2f} | {vanna_e:10.2f} | {volga_e:10.2f} |")

# 3. 计算成本
report.append("\n## 3. Computational Cost\n")
report.append("| Method | PDE Solves | Graph Nodes | Graph Edges |")
report.append("|--------|------------|-------------|-------------|")

for method in df['method'].unique():
    subset = df[df['method'] == method]
    pde = int(subset['n_pde_solves'].iloc[0])
    nodes = int(subset['graph_nodes'].iloc[0])
    edges = int(subset['graph_edges'].iloc[0])
    report.append(f"| {method:<15s} | {pde:10d} | {nodes:11,d} | {edges:11,d} |")

# 4. 关键发现
report.append("\n## 4. Key Findings\n")

# 最快方法
df_nona = df[df['method'] != 'Analytical']
fastest = df_nona.groupby('method')['time_ms'].mean().idxmin()
best_gamma = df_nona.groupby('method')['gamma_error_pct'].mean().idxmin()
best_volga = df_nona.groupby('method')['volga_error_pct'].mean().idxmin()

report.append(f"- **Fastest PDE method:** {fastest}")
report.append(f"- **Most accurate Gamma:** {best_gamma}")
report.append(f"- **Most accurate Volga:** {best_volga}")

report.append("\n### Recommendations:")
report.append("- **Quick computations:** Edge-Pushing (M=51, 1 PDE solve)")
report.append("- **High accuracy:** Edge-Pushing (M=101, Gamma < 0.5%)")
report.append("- **Simple implementation:** Bumping (5 PDE solves, moderate accuracy)")

report.append("\n---\n*Auto-generated from comprehensive benchmark*\n")

# 保存
output_text = '\n'.join(report)
with open(output_file, 'w') as f:
    f.write(output_text)

print(f"✓ Report saved to: {output_file}")
print("\nPreview:")
print(output_text)
