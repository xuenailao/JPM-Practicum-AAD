"""
从CSV生成LaTeX表格（不依赖matplotlib）
"""

import csv
import numpy as np

# 读取CSV
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
                'vanna_error_pct': [],
                'volga_error_pct': [],
                'n_pde_solves': int(row['n_pde_solves']),
                'graph_nodes': int(float(row['graph_nodes'])),
                'graph_edges': int(float(row['graph_edges']))
            }
        data[method]['time_ms'].append(float(row['time_ms']))
        data[method]['gamma_error_pct'].append(float(row['gamma_error_pct']))
        data[method]['delta_error_pct'].append(float(row['delta_error_pct']))
        data[method]['vega_error_pct'].append(float(row['vega_error_pct']))
        data[method]['vanna_error_pct'].append(float(row['vanna_error_pct']))
        data[method]['volga_error_pct'].append(float(row['volga_error_pct']))

# 计算统计
stats = {}
for method, values in data.items():
    stats[method] = {
        'time': np.mean(values['time_ms']),
        'gamma': np.mean(values['gamma_error_pct']),
        'delta': np.mean(values['delta_error_pct']),
        'vega': np.mean(values['vega_error_pct']),
        'vanna': np.mean(values['vanna_error_pct']),
        'volga': np.mean(values['volga_error_pct']),
        'pde': values['n_pde_solves'],
        'nodes': values['graph_nodes'],
        'edges': values['graph_edges']
    }

print("="*80)
print("LATEX TABLES FOR PRESENTATION")
print("="*80)

# 表1: 综合对比表
print("\n% ==== TABLE 1: COMPREHENSIVE COMPARISON ====")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Performance Comparison of Five Greeks Computation Methods}")
print("\\label{tab:methods_comparison}")
print("\\resizebox{\\textwidth}{!}{")
print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("\\textbf{Method} & \\textbf{Time (ms)} & \\textbf{$\\Gamma$ Error (\\%)} & \\textbf{Volga Error (\\%)} & \\textbf{PDE Solves} & \\textbf{Graph Nodes} \\\\\\\\")
print("\\midrule")

methods_order = ['Analytical', 'Bumping', 'AAD+Bumping', 'Double-AAD', 'Edge-Pushing']
for method in methods_order:
    s = stats[method]
    if method == 'Analytical':
        print(f"{method:15s} & ${s['time']:.2f}$ & --- & --- & {s['pde']} & {s['nodes']:,} \\\\\\\\")
    else:
        # 高亮最佳值
        gamma_str = f"\\textbf{{{s['gamma']:.2f}}}" if s['gamma'] < 3 else f"{s['gamma']:.2f}"
        volga_str = f"{s['volga']:.1f}"
        print(f"{method:15s} & ${s['time']:.2f}$ & ${gamma_str}$ & ${volga_str}$ & {s['pde']} & {s['nodes']:,} \\\\\\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("}")
print("\\end{table}")

# 表2: 详细精度对比
print("\n% ==== TABLE 2: DETAILED ACCURACY ====")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Detailed Accuracy Comparison (Average Errors in \\%)")
print("\\label{tab:accuracy_detail}")
print("\\begin{tabular}{lccccc}")
print("\\toprule")
print("\\textbf{Method} & \\boldmath{$\\Delta$} & \\boldmath{$\\Gamma$} & \\boldmath{$\\nu$} & \\textbf{Vanna} & \\textbf{Volga} \\\\\\\\")
print("\\midrule")

for method in methods_order[1:]:  # 排除Analytical
    s = stats[method]
    print(f"{method:15s} & {s['delta']:.2f} & {s['gamma']:.2f} & {s['vega']:.2f} & {s['vanna']:.1f} & {s['volga']:.1f} \\\\\\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# 表3: 计算成本对比
print("\n% ==== TABLE 3: COMPUTATIONAL COST ====")
print("\\begin{table}[h]")
print("\\centering")
print("\\caption{Computational Cost Analysis}")
print("\\label{tab:cost}")
print("\\begin{tabular}{lcccc}")
print("\\toprule")
print("\\textbf{Method} & \\textbf{PDE Solves} & \\textbf{Nodes} & \\textbf{Edges} & \\textbf{Complexity} \\\\\\\\")
print("\\midrule")

complexities = {
    'Analytical': '$O(1)$',
    'Bumping': '$O(5 \\cdot MN)$',
    'AAD+Bumping': '$O(5 \\cdot MN)$',
    'Double-AAD': '$O(3 \\cdot MN)$',
    'Edge-Pushing': '$O(MN)$'
}

for method in methods_order:
    s = stats[method]
    nodes_str = f"{s['nodes']:,}" if s['nodes'] > 0 else "---"
    edges_str = f"{s['edges']:,}" if s['edges'] > 0 else "---"
    print(f"{method:15s} & {s['pde']} & {nodes_str} & {edges_str} & {complexities[method]} \\\\\\\\")

print("\\bottomrule")
print("\\end{tabular}")
print("\\end{table}")

# 生成itemize列表用于关键发现
print("\n% ==== KEY FINDINGS (for slides) ====")
print("\\begin{itemize}")
print(f"  \\item \\textbf{{Edge-Pushing}}: Achieves {stats['Edge-Pushing']['gamma']:.2f}\\% Gamma error with only 1 PDE solve")
print(f"  \\item \\textbf{{Bumping}}: Fastest ({stats['Bumping']['time']:.0f} ms) but higher error ({stats['Bumping']['gamma']:.2f}\\%)")
print(f"  \\item \\textbf{{Analytical}}: Baseline reference, machine precision")
print(f"  \\item \\textbf{{Trade-off}}: Edge-Pushing balances speed, accuracy, and graph size")
print("\\end{itemize}")

# 生成数值列表
print("\n% ==== NUMERICAL SUMMARY ====")
print(f"% Best Gamma accuracy: {min(stats[m]['gamma'] for m in methods_order[1:]):.2f}% (Edge-Pushing/Double-AAD/AAD+Bumping)")
print(f"% Fastest PDE method: {min(stats[m]['time'] for m in methods_order[1:]):.0f} ms (Bumping)")
print(f"% Fewest PDE solves: 1 (Edge-Pushing)")
print(f"% Graph size (M=51): ~{stats['Edge-Pushing']['nodes']:,} nodes, ~{stats['Edge-Pushing']['edges']:,} edges")

print("\n" + "="*80)
print("LATEX TABLES GENERATED SUCCESSFULLY")
print("="*80)
print("\nTo use: Copy the tables above into your LaTeX presentation file")
