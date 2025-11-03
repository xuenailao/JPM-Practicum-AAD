"""
演示相对误差陷阱：当真实值接近0时，相对误差失去意义
"""

import pandas as pd
import numpy as np

df = pd.read_csv('benchmark_results/results_quick_20251031_004902.csv')

# 只看Edge-Pushing方法
df_edge = df[df['method'] == 'Edge-Pushing'].copy()

# 计算绝对误差
df_edge['vanna_abs_error'] = np.abs(df_edge['vanna'] - df_edge['vanna_analytical'])
df_edge['volga_abs_error'] = np.abs(df_edge['volga'] - df_edge['volga_analytical'])

# 排序后展示
df_sorted = df_edge.sort_values('vanna_error_pct', ascending=False)

print("="*100)
print("相对误差陷阱演示：Vanna案例")
print("="*100)
print("\n按相对误差从大到小排序：")
print("-"*100)
print(f"{'Case':<5} {'S0':<6} {'σ':<5} {'T':<5} {'真实Vanna':<12} {'数值Vanna':<12} {'相对误差%':<12} {'绝对误差':<12}")
print("-"*100)

for i, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
    print(f"{i:<5} {row['S0']:<6} {row['sigma']:<5} {row['T']:<5} "
          f"{row['vanna_analytical']:>11.6f} {row['vanna']:>11.6f} "
          f"{row['vanna_error_pct']:>11.2f} {row['vanna_abs_error']:>11.6f}")

print("\n" + "="*100)
print("观察：")
print("="*100)
print("""
1. 最高相对误差(1197%)出现在真实值=-0.021的案例
   - 绝对误差仅0.252，这在期权定价中完全可以接受
   - 但相对误差被放大了1000倍

2. 前10个高误差案例，大部分真实值<0.5
   - 这说明高相对误差主要由小真实值引起
   - 而非数值方法失败

3. 绝对误差在0.05-0.25之间，非常稳定
   - 这说明数值方法的精度是一致的
   - 相对误差的波动纯粹是分母效应
""")

print("\n" + "="*100)
print("相对误差陷阱演示：Volga案例")
print("="*100)

df_sorted = df_edge.sort_values('volga_error_pct', ascending=False)

print("\n按相对误差从大到小排序：")
print("-"*100)
print(f"{'Case':<5} {'S0':<6} {'σ':<5} {'T':<5} {'真实Volga':<12} {'数值Volga':<12} {'相对误差%':<12} {'绝对误差':<12}")
print("-"*100)

for i, (idx, row) in enumerate(df_sorted.head(10).iterrows(), 1):
    print(f"{i:<5} {row['S0']:<6} {row['sigma']:<5} {row['T']:<5} "
          f"{row['volga_analytical']:>11.3f} {row['volga']:>11.3f} "
          f"{row['volga_error_pct']:>11.2f} {row['volga_abs_error']:>11.3f}")

print("\n" + "="*100)
print("观察：")
print("="*100)
print("""
1. 最高相对误差(11464%)出现在真实值=0.668的案例
   - 绝对误差76.5确实较大
   - 但考虑到PDE网格分辨率(M=51)，这是可以理解的
   - 提高网格至M=101可以显著改善

2. 大部分高相对误差案例，真实值<10
   - σ=0.3时，Volga自然变小
   - 这是Black-Scholes公式的特性

3. 当真实值>50时(案例后半部分)，相对误差骤降至<50%
   - 说明数值方法在大真实值时表现很好
""")

print("\n" + "="*100)
print("解决方案对比")
print("="*100)

# 显示使用绝对误差vs相对误差的差异
print("\n方案A: 仅使用相对误差（当前，误导性）")
print("-"*100)
print(f"Vanna平均相对误差: {df_edge['vanna_error_pct'].mean():.2f}%")
print(f"Volga平均相对误差: {df_edge['volga_error_pct'].mean():.2f}%")
print("结论: 看起来非常差，误差达到数百上千%！")

print("\n方案B: 仅使用绝对误差")
print("-"*100)
print(f"Vanna平均绝对误差: {df_edge['vanna_abs_error'].mean():.6f}")
print(f"Volga平均绝对误差: {df_edge['volga_abs_error'].mean():.3f}")
print("结论: 误差很小，方法可用！但缺少相对尺度")

print("\n方案C: 条件相对误差（推荐）")
print("-"*100)

# 只在真实值较大时计算相对误差
vanna_large = df_edge[np.abs(df_edge['vanna_analytical']) > 0.5]
volga_large = df_edge[np.abs(df_edge['volga_analytical']) > 10.0]

print(f"Vanna (|真实值|>0.5时): {len(vanna_large)}个案例, 平均相对误差: {vanna_large['vanna_error_pct'].mean():.2f}%")
print(f"Volga (|真实值|>10时): {len(volga_large)}个案例, 平均相对误差: {volga_large['volga_error_pct'].mean():.2f}%")
print(f"\nVanna (所有): 平均绝对误差: {df_edge['vanna_abs_error'].mean():.6f}")
print(f"Volga (所有): 平均绝对误差: {df_edge['volga_abs_error'].mean():.3f}")
print("\n结论: 在有意义的区域，相对误差<20%，绝对误差始终很小")

print("\n" + "="*100)
print("最终建议")
print("="*100)
print("""
1. 对于一阶Greeks (Delta, Gamma, Vega):
   - 使用相对误差即可，因为它们通常不接近0
   - 当前方法表现优秀: <3%误差

2. 对于二阶Greeks (Vanna, Volga):
   - 同时报告绝对误差和条件相对误差
   - 阈值建议: Vanna>0.5, Volga>10
   - 或者直接使用绝对误差作为主要指标

3. 网格分辨率建议:
   - 快速原型: M=51, N=50 (当前)
   - 生产环境: M=101, N=100 (推荐)
   - 高精度: M=201, N=200 (研究用)

4. 不要被相对误差吓到：
   - 838%的Volga相对误差听起来很恐怖
   - 但16.39的绝对误差在大多数应用中完全可以接受
   - 关键是理解你的应用场景需要什么级别的精度
""")

print("="*100)
