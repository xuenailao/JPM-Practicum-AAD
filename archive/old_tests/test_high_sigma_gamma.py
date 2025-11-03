#!/usr/bin/env python3
"""
Test Gamma at High Volatility (σ=0.5)
"""

import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

print("="*80)
print("HIGH VOLATILITY GAMMA TEST (σ=0.5)")
print("="*80)

# Parameters - High volatility
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.5
M, N = 51, 100

print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
print(f"Grid: M={M}, N={N}")

# Analytical Gamma
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
phi_d1 = norm.pdf(d1)
gamma_analytical = phi_d1 / (S0 * sigma * sqrt_T)

print(f"\nAnalytical Gamma: {gamma_analytical:.8f}")

# Test with Hessian
print(f"\nComputing Hessian (this may take 1-2 minutes)...")

solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
result = solver.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True,
    verbose=False
)

print(f"\nResults:")
print(f"  Price = {result['price']:.6f}")
print(f"  Delta = {result['delta']:.6f}")
print(f"  Gamma = {result['gamma']:.8f}")
print(f"  Vega  = {result['vega']:.6f}")

gamma_computed = result['gamma']

# Check if negative
if gamma_computed < 0:
    print(f"\n  ✗ CRITICAL: Gamma is NEGATIVE!")
    print(f"    Analytical: {gamma_analytical:.8f} (positive)")
    print(f"    Computed:   {gamma_computed:.8f} (negative)")
    print(f"\n  This is likely due to:")
    print(f"    1. Grid too coarse for high volatility (M={M}, N={N})")
    print(f"    2. Numerical instability in spline interpolation")
    print(f"    3. C-N spurious oscillations corrupting Hessian")
else:
    gamma_error = abs((gamma_computed - gamma_analytical) / gamma_analytical) * 100
    print(f"\n  Gamma Error: {gamma_error:.2f}%")

    if gamma_error > 100:
        print(f"  ⚠ Very large error (>{gamma_error:.0f}%), likely due to C-N oscillations")
    elif gamma_error > 50:
        print(f"  ⚠ Large error (~{gamma_error:.0f}%), grid resolution may be insufficient")
    else:
        print(f"  ✓ Moderate error, acceptable for this grid")

print()
