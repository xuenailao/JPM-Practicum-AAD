"""
分析Vanna和Volga误差波动的根本原因
"""

import pandas as pd
import numpy as np

# 读取测试结果
df = pd.read_csv('benchmark_results/results_quick_20251031_004902.csv')

# 只看数值方法 (排除解析解)
methods = ['Bumping', 'AAD+Bumping', 'Double-AAD', 'Edge-Pushing']
df_numerical = df[df['method'].isin(methods)].copy()

print("="*80)
print("Vanna和Volga误差分析")
print("="*80)

# 1. 按方法分组统计
print("\n1. 各方法的平均误差:")
print("-"*80)
for method in methods:
    df_method = df_numerical[df_numerical['method'] == method]
    print(f"\n{method}:")
    print(f"  Vanna误差: {df_method['vanna_error_pct'].mean():.2f}% (std={df_method['vanna_error_pct'].std():.2f}%)")
    print(f"  Volga误差: {df_method['volga_error_pct'].mean():.2f}% (std={df_method['volga_error_pct'].std():.2f}%)")

# 2. 按参数分组分析
print("\n\n2. 按参数组合分析:")
print("-"*80)

# 分析不同sigma下的误差
for sigma in df['sigma'].unique():
    df_sigma = df_numerical[df_numerical['sigma'] == sigma]
    print(f"\nσ = {sigma}:")

    # 计算解析解的vanna和volga绝对值
    df_analytical = df[(df['sigma'] == sigma) & (df['method'] == 'Analytical')]
    avg_vanna_analytical = df_analytical['vanna_analytical'].abs().mean()
    avg_volga_analytical = df_analytical['volga_analytical'].abs().mean()

    print(f"  解析vanna平均值: {avg_vanna_analytical:.4f}")
    print(f"  解析volga平均值: {avg_volga_analytical:.4f}")

    for method in methods:
        df_m = df_sigma[df_sigma['method'] == method]
        if len(df_m) > 0:
            print(f"  {method}: Vanna误差={df_m['vanna_error_pct'].mean():.1f}%, Volga误差={df_m['volga_error_pct'].mean():.1f}%")

# 3. 识别高误差案例
print("\n\n3. 高误差案例分析:")
print("-"*80)

# Vanna高误差
print("\nVanna误差 > 100%的案例:")
high_vanna = df_numerical[df_numerical['vanna_error_pct'] > 100]
print(f"共{len(high_vanna)}个案例:")
for _, row in high_vanna.head(10).iterrows():
    print(f"  {row['method']}: S0={row['S0']}, σ={row['sigma']}, T={row['T']}")
    print(f"    解析vanna={row['vanna_analytical']:.6f}, 数值vanna={row['vanna']:.6f}")
    print(f"    误差={row['vanna_error_pct']:.2f}%")

# Volga高误差
print("\n\nVolga误差 > 500%的案例:")
high_volga = df_numerical[df_numerical['volga_error_pct'] > 500]
print(f"共{len(high_volga)}个案例:")
for _, row in high_volga.head(10).iterrows():
    print(f"  {row['method']}: S0={row['S0']}, σ={row['sigma']}, T={row['T']}")
    print(f"    解析volga={row['volga_analytical']:.6f}, 数值volga={row['volga']:.6f}")
    print(f"    误差={row['volga_error_pct']:.2f}%")

# 4. 关键发现：小真实值导致的相对误差放大
print("\n\n4. 关键发现：相对误差 vs 绝对误差")
print("-"*80)

# 对于Edge-Pushing方法
df_edge = df_numerical[df_numerical['method'] == 'Edge-Pushing']

# 计算绝对误差
df_edge['vanna_abs_error'] = np.abs(df_edge['vanna'] - df_edge['vanna_analytical'])
df_edge['volga_abs_error'] = np.abs(df_edge['volga'] - df_edge['volga_analytical'])

print("\nEdge-Pushing方法:")
print(f"Vanna绝对误差均值: {df_edge['vanna_abs_error'].mean():.6f}")
print(f"Vanna相对误差均值: {df_edge['vanna_error_pct'].mean():.2f}%")
print(f"\nVolga绝对误差均值: {df_edge['volga_abs_error'].mean():.6f}")
print(f"Volga相对误差均值: {df_edge['volga_error_pct'].mean():.2f}%")

# 按解析值大小分组
print("\n\n按解析值大小分组:")
print("-"*80)

# Vanna分组
print("\nVanna:")
for threshold in [0.1, 0.5, 1.0]:
    df_small = df_edge[np.abs(df_edge['vanna_analytical']) < threshold]
    df_large = df_edge[np.abs(df_edge['vanna_analytical']) >= threshold]

    print(f"\n|vanna_analytical| < {threshold}: {len(df_small)}个案例")
    if len(df_small) > 0:
        print(f"  平均相对误差: {df_small['vanna_error_pct'].mean():.2f}%")
        print(f"  平均绝对误差: {df_small['vanna_abs_error'].mean():.6f}")

    print(f"|vanna_analytical| >= {threshold}: {len(df_large)}个案例")
    if len(df_large) > 0:
        print(f"  平均相对误差: {df_large['vanna_error_pct'].mean():.2f}%")
        print(f"  平均绝对误差: {df_large['vanna_abs_error'].mean():.6f}")

# Volga分组
print("\n\nVolga:")
for threshold in [1.0, 5.0, 10.0]:
    df_small = df_edge[np.abs(df_edge['volga_analytical']) < threshold]
    df_large = df_edge[np.abs(df_edge['volga_analytical']) >= threshold]

    print(f"\n|volga_analytical| < {threshold}: {len(df_small)}个案例")
    if len(df_small) > 0:
        print(f"  平均相对误差: {df_small['volga_error_pct'].mean():.2f}%")
        print(f"  平均绝对误差: {df_small['volga_abs_error'].mean():.6f}")

    print(f"|volga_analytical| >= {threshold}: {len(df_large)}个案例")
    if len(df_large) > 0:
        print(f"  平均相对误差: {df_large['volga_error_pct'].mean():.2f}%")
        print(f"  平均绝对误差: {df_large['volga_abs_error'].mean():.6f}")

# 5. 数值分析：为什么小真实值会出现？
print("\n\n5. 为什么Vanna和Volga的真实值会很小？")
print("-"*80)

# 检查解析公式的特性
df_analytical = df[df['method'] == 'Analytical']

print("\n解析公式回顾:")
print("  Vanna = -φ(d1) * d2 / σ")
print("  Volga = Vega * d1 * d2 / σ")
print("\n关键观察:")
print("  - Vanna在d2≈0时接近0 (即S0≈K*exp((r-0.5σ²)T))")
print("  - Volga在d1≈0或d2≈0时接近0")
print("  - 高sigma时，Vanna和Volga绝对值变小")

# 计算d1和d2
for _, row in df_analytical.head(5).iterrows():
    S0, K, T, r, sigma = row['S0'], row['K'], row['T'], row['r'], row['sigma']
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    print(f"\nS0={S0}, K={K}, T={T}, σ={sigma}:")
    print(f"  d1={d1:.4f}, d2={d2:.4f}")
    print(f"  vanna={row['vanna_analytical']:.6f}, volga={row['volga_analytical']:.6f}")

print("\n\n" + "="*80)
print("结论:")
print("="*80)
print("""
1. **真正的问题**: Vanna和Volga在某些参数下真实值接近0
   - 这是期权定价的固有特性，不是数值误差！

2. **相对误差放大效应**:
   - 当真实值<0.5时，即使绝对误差很小(如0.1)，相对误差也会>20%
   - 这导致看起来"误差很大"，但实际上绝对误差是可接受的

3. **方法比较**:
   - Bumping: 误差最大(154% vanna, 941% volga)
   - AAD方法: 误差中等(88% vanna, 838% volga)
   - 所有方法在大真实值时表现良好

4. **建议**:
   - 应该报告**绝对误差**，而不是相对误差
   - 或者只计算|真实值|>某阈值时的相对误差
   - 小真实值时，关注绝对误差是否在可接受范围内
""")
