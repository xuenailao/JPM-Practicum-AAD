"""
Minimal Test: Capriotti CN AAD - Single Small Grid

Debug version to see where the slowdown is.
"""

import numpy as np
import time
from scipy.stats import norm
from aad_edge_pushing.pde.aad_integration.capriotti_cn_aad import CapriottiCNAAD


def bsm_price(S, K, T, r, sigma):
    """BSM price."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


print("="*80)
print("MINIMAL TEST: Capriotti CN AAD")
print("="*80)

# Very small grid
M, N = 10, 20
print(f"\nGrid: M={M}, N={N} (Interior: {M-2}×{N})")

# Parameters
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")

# BSM
print("\n[1] BSM Analytical...")
t0 = time.time()
price_bs = bsm_price(S0, K, T, r, sigma)
t_bsm = (time.time() - t0) * 1000
print(f"    Price: ${price_bs:.6f}")
print(f"    Time: {t_bsm:.3f} ms")

# PDE AAD
print("\n[2] PDE AAD...")
print("    Creating solver...")
solver = CapriottiCNAAD(M=M, N=N, phi=0.5)
solver.S0 = S0
solver.K = K
solver.T = T
solver.r = r
solver.Smax = 2.0 * K
solver.S_grid = np.linspace(0, solver.Smax, M)
solver.dS = solver.S_grid[1] - solver.S_grid[0]
solver.dt = T / N

sigma_values = np.full(M - 1, sigma)
print(f"    Sigma grid size: {len(sigma_values)}")

print("    Solving PDE...")
t0 = time.time()

try:
    price_pde, gradient, hessian = solver.compute_hessian_cn_algo4(sigma_values)
    t_pde = (time.time() - t0) * 1000

    print(f"    Price: ${price_pde:.6f}")
    print(f"    Time: {t_pde:.3f} ms")
    print(f"    Gradient shape: {gradient.shape}")
    print(f"    Hessian shape: {hessian.shape}")

    print(f"\nResults:")
    print(f"  BSM Price:  ${price_bs:.6f}")
    print(f"  PDE Price:  ${price_pde:.6f}")
    print(f"  Error:      ${abs(price_pde - price_bs):.6f} ({abs(price_pde - price_bs)/price_bs*100:.2f}%)")
    print(f"  BSM Time:   {t_bsm:.3f} ms")
    print(f"  PDE Time:   {t_pde:.3f} ms")

except Exception as e:
    print(f"    ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
