"""
分析基准测试结果并生成图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('benchmark_results/results_quick_20251031_004902.csv')

print("="*80)
print("BENCHMARK RESULTS ANALYSIS")
print("="*80)

# 1. 速度分析
print("\n1. SPEED ANALYSIS")
print("-"*80)
speed_stats = df.groupby('method')['time_ms'].agg(['mean', 'std', 'min', 'max'])
print(speed_stats)

# 2. 精度分析
print("\n2. ACCURACY ANALYSIS")
print("-"*80)
df_nona = df[df['method'] != 'Analytical']
accuracy_stats = df_nona.groupby('method')[
    ['delta_error_pct', 'gamma_error_pct', 'vega_error_pct', 'volga_error_pct']
].mean()
print(accuracy_stats)

# 3. 计算图统计
print("\n3. COMPUTATION GRAPH STATISTICS")
print("-"*80)
graph_stats = df.groupby('method')[['n_pde_solves', 'graph_nodes', 'graph_edges']].first()
print(graph_stats)

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: 速度对比（对数尺度）
ax1 = axes[0, 0]
methods = ['Analytical', 'Bumping', 'AAD+Bumping', 'Double-AAD', 'Edge-Pushing']
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
times = []
for method in methods:
    time = df[df['method'] == method]['time_ms'].mean()
    times.append(time)

bars = ax1.bar(range(len(methods)), times, color=colors, alpha=0.8, edgecolor='black')
ax1.set_yscale('log')
ax1.set_ylabel('Time (ms, log scale)', fontsize=11, fontweight='bold')
ax1.set_title('(a) Computation Time Comparison', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(['Anal.', 'Bump.', 'AAD+B', 'D-AAD', 'E-Push'], rotation=0, fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for i, (bar, time) in enumerate(zip(bars, times)):
    if time < 1:
        label = f'{time:.2f}'
    elif time < 1000:
        label = f'{time:.0f}'
    else:
        label = f'{time/1000:.1f}k'
    ax1.text(bar.get_x() + bar.get_width()/2, time*1.5, label,
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 图2: Gamma精度对比
ax2 = axes[0, 1]
gamma_errors = []
method_names = []
for method in methods[1:]:  # 排除Analytical
    err = df[df['method'] == method]['gamma_error_pct'].mean()
    gamma_errors.append(err)
    method_names.append(method)

bars2 = ax2.bar(range(len(method_names)), gamma_errors, color=colors[1:], alpha=0.8, edgecolor='black')
ax2.set_ylabel('Gamma Error (%)', fontsize=11, fontweight='bold')
ax2.set_title('(b) Gamma Accuracy Comparison', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(method_names)))
ax2.set_xticklabels(['Bump.', 'AAD+B', 'D-AAD', 'E-Push'], rotation=0, fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='1% threshold')
ax2.legend(fontsize=9)

# 添加数值标签
for bar, err in zip(bars2, gamma_errors):
    ax2.text(bar.get_x() + bar.get_width()/2, err + 0.2, f'{err:.2f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 图3: PDE求解次数 vs 精度权衡
ax3 = axes[1, 0]
pde_solves = []
gamma_acc = []
labels_pos = []
for method in methods[1:]:
    pde = df[df['method'] == method]['n_pde_solves'].iloc[0]
    gamma_e = df[df['method'] == method]['gamma_error_pct'].mean()
    pde_solves.append(pde)
    gamma_acc.append(gamma_e)
    labels_pos.append(method.replace('AAD+Bumping', 'AAD+B').replace('Double-AAD', 'D-AAD').replace('Edge-Pushing', 'E-Push'))

scatter = ax3.scatter(pde_solves, gamma_acc, s=300, c=colors[1:], alpha=0.7, edgecolor='black', linewidth=2)
ax3.set_xlabel('Number of PDE Solves', fontsize=11, fontweight='bold')
ax3.set_ylabel('Gamma Error (%)', fontsize=11, fontweight='bold')
ax3.set_title('(c) Trade-off: PDE Solves vs Accuracy', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

# 添加标签
for i, label in enumerate(labels_pos):
    ax3.annotate(label, (pde_solves[i], gamma_acc[i]),
                xytext=(10, 5), textcoords='offset points', fontsize=9, fontweight='bold')

# 添加理想区域标注
ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax3.text(0.5, 0.5, 'Better', fontsize=10, color='green', alpha=0.7, fontweight='bold')

# 图4: 计算图规模
ax4 = axes[1, 1]
aad_methods = ['AAD+Bumping', 'Double-AAD', 'Edge-Pushing']
nodes = []
edges = []
for method in aad_methods:
    n = df[df['method'] == method]['graph_nodes'].iloc[0]
    e = df[df['method'] == method]['graph_edges'].iloc[0]
    nodes.append(n)
    edges.append(e)

x = np.arange(len(aad_methods))
width = 0.35
bars1 = ax4.bar(x - width/2, np.array(nodes)/1000, width, label='Nodes',
                color='#3498db', alpha=0.8, edgecolor='black')
bars2 = ax4.bar(x + width/2, np.array(edges)/1000, width, label='Edges',
                color='#e74c3c', alpha=0.8, edgecolor='black')

ax4.set_ylabel('Count (×1000)', fontsize=11, fontweight='bold')
ax4.set_title('(d) Computation Graph Size (M=51)', fontsize=12, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(['AAD+B', 'D-AAD', 'E-Push'], fontsize=10)
ax4.legend(fontsize=10)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height, f'{height:.1f}k',
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('benchmark_results/benchmark_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('benchmark_results/benchmark_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Figures saved:")
print("  - benchmark_results/benchmark_analysis.pdf")
print("  - benchmark_results/benchmark_analysis.png")

# 生成LaTeX表格数据
print("\n" + "="*80)
print("LATEX TABLE DATA")
print("="*80)

print("\n% Table 1: Speed Comparison")
print("\\begin{tabular}{lrrrr}")
print("\\toprule")
print("Method & Mean (ms) & Std (ms) & Min (ms) & Max (ms) \\\\")
print("\\midrule")
for method in methods:
    stats = df[df['method'] == method]['time_ms'].agg(['mean', 'std', 'min', 'max'])
    print(f"{method:15s} & {stats['mean']:8.2f} & {stats['std']:7.2f} & {stats['min']:7.2f} & {stats['max']:8.2f} \\\\\\\\")
print("\\bottomrule")
print("\\end{tabular}")

print("\n% Table 2: Accuracy Comparison")
print("\\begin{tabular}{lrrrr}")
print("\\toprule")
print("Method & $\\Delta$ err\\% & $\\Gamma$ err\\% & $\\nu$ err\\% & Volga err\\% \\\\")
print("\\midrule")
for method in methods[1:]:
    stats = df[df['method'] == method][
        ['delta_error_pct', 'gamma_error_pct', 'vega_error_pct', 'volga_error_pct']
    ].mean()
    print(f"{method:15s} & {stats['delta_error_pct']:6.2f} & {stats['gamma_error_pct']:6.2f} & "
          f"{stats['vega_error_pct']:6.2f} & {stats['volga_error_pct']:7.2f} \\\\\\\\")
print("\\bottomrule")
print("\\end{tabular}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
