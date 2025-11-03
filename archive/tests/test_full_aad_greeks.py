"""
测试完全AAD+Edge-Pushing Greeks计算
"""
import numpy as np
import sys
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import CapriottiCNAAD

def black_scholes_greeks(S, K, T, r, sigma):
    """解析Greeks"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = S * norm.pdf(d1) * sqrt_T
    vanna = -norm.pdf(d1) * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }

# 参数
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

print("="*80)
print("完全AAD+Edge-Pushing Greeks测试")
print("="*80)

# 测试小网格
M, N = 12, 50
solver = CapriottiCNAAD(M=M, N=N)
solver.S0, solver.K, solver.T, solver.r = S0, K, T, r

print(f"\n配置: M={M}, N={N}")
print(f"参数向量维度: {M} (包含S0 + {M-1}个sigma)")

# 计算Greeks
print("\n计算中...")
greeks = solver.compute_greeks_aad_full(sigma_value=sigma)

# 解析解
analytical = black_scholes_greeks(S0, K, T, r, sigma)

# 打印结果
print(f"\n{'Greek':<10} | {'AAD+EP':<14} | {'解析解':<14} | {'绝对误差':<12} | {'相对误差':<12}")
print("-"*80)

for key in ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']:
    aad_val = greeks[key]
    ana_val = analytical[key]
    abs_err = abs(aad_val - ana_val)
    rel_err = abs_err / abs(ana_val) if abs(ana_val) > 1e-10 else 0

    print(f"{key:<10} | {aad_val:<14.6f} | {ana_val:<14.6f} | {abs_err:<12.2e} | {rel_err:<12.2%}")

# 性能统计
print(f"\n性能统计:")
print(f"  计算时间: {greeks['computation_time_ms']:.2f} ms")
print(f"  PDE求解次数: {greeks['n_pde_solves']}")
print(f"  计算图节点数: {greeks['n_tape_nodes']}")

# Hessian统计
print(f"\nHessian统计:")
stats = greeks['hessian_stats']
print(f"  形状: {stats['shape']}")
print(f"  非零元素: {stats['nnz']} / {stats['total']}")
print(f"  稀疏度: {stats['sparsity']:.2%}")

print("\n" + "="*80)
print("测试完成!")
print("="*80)