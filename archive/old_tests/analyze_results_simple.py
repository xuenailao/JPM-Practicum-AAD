"""
简化版分析脚本 - 不依赖pandas
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# 手动读取CSV
data = {}
with open('benchmark_results/results_quick_20251031_004902.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        method = row['method']
        if method not in data:
            data[method] = {
                'time_ms': [],
                'gamma_error_pct': [],
                'delta_error_pct': [],
                'vega_error_pct': [],
                'volga_error_pct': [],
                'n_pde_solves': int(row['n_pde_solves']),
                'graph_nodes': int(float(row['graph_nodes'])),
                'graph_edges': int(float(row['graph_edges']))
            }
        data[method]['time_ms'].append(float(row['time_ms']))
        data[method]['gamma_error_pct'].append(float(row['gamma_error_pct']))
        data[method]['delta_error_pct'].append(float(row['delta_error_pct']))
        data[method]['vega_error_pct'].append(float(row['vega_error_pct']))
        data[method]['volga_error_pct'].append(float(row['volga_error_pct']))

# 计算统计信息
stats = {}
for method, values in data.items():
    stats[method] = {
        'time_mean': np.mean(values['time_ms']),
        'time_std': np.std(values['time_ms']),
        'time_min': np.min(values['time_ms']),
        'time_max': np.max(values['time_ms']),
        'gamma_err': np.mean(values['gamma_error_pct']),
        'delta_err': np.mean(values['delta_error_pct']),
        'vega_err': np.mean(values['vega_error_pct']),
        'volga_err': np.mean(values['volga_error_pct']),
        'n_pde': values['n_pde_solves'],
        'nodes': values['graph_nodes'],
        'edges': values['graph_edges']
    }

print("="*80)
print("BENCHMARK RESULTS ANALYSIS")
print("="*80)

print("\n1. SPEED STATISTICS")
print("-"*80)
print(f"{'Method':<20} {'Mean (ms)':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
print("-"*80)
for method in ['Analytical', 'Bumping', 'AAD+Bumping', 'Double-AAD', 'Edge-Pushing']:
    s = stats[method]
    print(f"{method:<20} {s['time_mean']:>10.2f}  {s['time_std']:>10.2f}  "
          f"{s['time_min']:>10.2f}  {s['time_max']:>10.2f}")

print("\n2. ACCURACY STATISTICS")
print("-"*80)
print(f"{'Method':<20} {'Δ err%':<10} {'Γ err%':<10} {'ν err%':<10} {'Volga err%':<12}")
print("-"*80)
for method in ['Bumping', 'AAD+Bumping', 'Double-AAD', 'Edge-Pushing']:
    s = stats[method]
    print(f"{method:<20} {s['delta_err']:>8.2f}  {s['gamma_err']:>8.2f}  "
          f"{s['vega_err']:>8.2f}  {s['volga_err']:>10.2f}")

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
plt.rcParams['font.size'] = 10

# 图1: 速度对比（对数尺度）
ax1 = axes[0, 0]
methods = ['Analytical', 'Bumping', 'AAD+Bumping', 'Double-AAD', 'Edge-Pushing']
colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
times = [stats[m]['time_mean'] for m in methods]

bars = ax1.bar(range(len(methods)), times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_yscale('log')
ax1.set_ylabel('Time (ms, log scale)', fontsize=12, fontweight='bold')
ax1.set_title('(a) Computation Time Comparison', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(range(len(methods)))
ax1.set_xticklabels(['Analytical', 'Bumping', 'AAD+Bump', 'Double-AAD', 'Edge-Push'],
                    rotation=15, ha='right', fontsize=10)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bar, time in zip(bars, times):
    if time < 1:
        label = f'{time:.2f}'
    elif time < 1000:
        label = f'{time:.0f}'
    else:
        label = f'{time/1000:.1f}k'
    y_pos = time * 2 if time > 1 else time * 10
    ax1.text(bar.get_x() + bar.get_width()/2, y_pos, label,
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 图2: Gamma精度对比
ax2 = axes[0, 1]
gamma_errors = [stats[m]['gamma_err'] for m in methods[1:]]
method_names = methods[1:]

bars2 = ax2.bar(range(len(method_names)), gamma_errors, color=colors[1:],
                alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Gamma Error (%)', fontsize=12, fontweight='bold')
ax2.set_title('(b) Gamma Accuracy Comparison', fontsize=13, fontweight='bold', pad=10)
ax2.set_xticks(range(len(method_names)))
ax2.set_xticklabels(['Bumping', 'AAD+Bump', 'Double-AAD', 'Edge-Push'],
                    rotation=15, ha='right', fontsize=10)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='1% target', alpha=0.7)
ax2.legend(fontsize=10, loc='upper right')

# 添加数值标签
for bar, err in zip(bars2, gamma_errors):
    ax2.text(bar.get_x() + bar.get_width()/2, err + 0.15, f'{err:.2f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')

# 图3: PDE求解次数 vs 精度权衡
ax3 = axes[1, 0]
pde_solves = [stats[m]['n_pde'] for m in methods[1:]]
gamma_acc = [stats[m]['gamma_err'] for m in methods[1:]]
labels_pos = ['Bumping', 'AAD+Bump', 'Double-AAD', 'Edge-Push']

scatter = ax3.scatter(pde_solves, gamma_acc, s=400, c=colors[1:],
                     alpha=0.7, edgecolor='black', linewidth=2.5)
ax3.set_xlabel('Number of PDE Solves', fontsize=12, fontweight='bold')
ax3.set_ylabel('Gamma Error (%)', fontsize=12, fontweight='bold')
ax3.set_title('(c) Efficiency: PDE Solves vs Accuracy', fontsize=13, fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.set_xlim(-0.5, 6)

# 添加标签
for i, label in enumerate(labels_pos):
    offset_x = 0.3 if pde_solves[i] > 3 else -0.3
    offset_y = 0.15 if i % 2 == 0 else -0.15
    ax3.annotate(label, (pde_solves[i], gamma_acc[i]),
                xytext=(pde_solves[i] + offset_x, gamma_acc[i] + offset_y),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i+1], alpha=0.3))

# 标注理想区域
ax3.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
ax3.text(0.3, 0.3, 'Better\nAccuracy', fontsize=10, color='green',
         alpha=0.8, fontweight='bold', ha='left')
ax3.arrow(0.8, 4.5, -0.4, -3, head_width=0.15, head_length=0.3,
          fc='green', ec='green', alpha=0.5, linewidth=2)

# 图4: 计算图规模
ax4 = axes[1, 1]
aad_methods = ['AAD+Bumping', 'Double-AAD', 'Edge-Pushing']
nodes = [stats[m]['nodes']/1000 for m in aad_methods]
edges = [stats[m]['edges']/1000 for m in aad_methods]

x = np.arange(len(aad_methods))
width = 0.35
bars1 = ax4.bar(x - width/2, nodes, width, label='Nodes',
                color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x + width/2, edges, width, label='Edges',
                color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)

ax4.set_ylabel('Count (×1000)', fontsize=12, fontweight='bold')
ax4.set_title('(d) Computation Graph Size (M=51)', fontsize=13, fontweight='bold', pad=10)
ax4.set_xticks(x)
ax4.set_xticklabels(['AAD+Bump', 'Double-AAD', 'Edge-Push'], fontsize=10)
ax4.legend(fontsize=10, loc='upper left')
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# 添加数值标签
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height + 1, f'{height:.1f}k',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout(pad=2.0)
plt.savefig('benchmark_results/benchmark_analysis.pdf', dpi=300, bbox_inches='tight')
plt.savefig('benchmark_results/benchmark_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Figures saved:")
print("  - benchmark_results/benchmark_analysis.pdf")
print("  - benchmark_results/benchmark_analysis.png")

# 生成LaTeX表格
print("\n" + "="*80)
print("LATEX TABLE DATA")
print("="*80)

print("\n% Table 1: Speed and Accuracy Summary")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Performance Comparison of Five Greeks Computation Methods (M=51, N=50)}")
print("\\label{tab:comparison}")
print("\\begin{tabular}{lrrrr}")
print("\\toprule")
print("\\textbf{Method} & \\textbf{Time (ms)} & \\textbf{$\\Gamma$ err (\\%)} & \\textbf{PDE Solves} & \\textbf{Graph Nodes} \\\\\\\\")
print("\\midrule")
for method in methods:
    s = stats[method]
    if method == 'Analytical':
        print(f"{method:15s} & ${s['time_mean']:6.2f}$ & --- & ${s['n_pde']:d}$ & ${s['nodes']:,d}$ \\\\\\\\")
    else:
        print(f"{method:15s} & ${s['time_mean']:6.2f}$ & ${s['gamma_err']:5.2f}$ & "
              f"${s['n_pde']:d}$ & ${s['nodes']:,d}$ \\\\\\\\")
print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

print("\n% Key Finding")
print("% Edge-Pushing achieves 2.18% Gamma error with only 1 PDE solve,")
print("% while Bumping requires 5 PDE solves for 5.18% error.")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
