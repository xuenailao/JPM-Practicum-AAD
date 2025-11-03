#!/usr/bin/env python3
"""
Simple Gamma Test - Check if BS_PDE_AAD Gamma is correct
"""

import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

# Parameters - Use σ=0.2 (lower volatility, easier case)
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
M, N = 51, 100

print("="*80)
print("SIMPLE GAMMA TEST")
print("="*80)
print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
print(f"Grid: M={M}, N={N}")

# Analytical Gamma
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
phi_d1 = norm.pdf(d1)
gamma_analytical = phi_d1 / (S0 * sigma * sqrt_T)

print(f"\nAnalytical Gamma: {gamma_analytical:.8f}")

# Test 1: Jacobian only (no Hessian)
print(f"\n[1/2] Computing Jacobian (no Hessian)...")
solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
result_jac = solver.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=False,
    verbose=False
)

print(f"  Price = {result_jac['price']:.6f}")
print(f"  Delta = {result_jac['delta']:.6f}")
print(f"  Vega  = {result_jac['vega']:.6f}")
print(f"  Time  = {result_jac['time_ms']:.1f} ms")

# Test 2: With Hessian
print(f"\n[2/2] Computing Hessian...")
print(f"  (This will take 30-60 seconds...)")

result_hess = solver.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True,
    verbose=False
)

print(f"  Price = {result_hess['price']:.6f}")
print(f"  Delta = {result_hess['delta']:.6f}")
print(f"  Gamma = {result_hess['gamma']:.8f}")
print(f"  Vega  = {result_hess['vega']:.6f}")

# Check Gamma
gamma_computed = result_hess['gamma']
gamma_error = abs((gamma_computed - gamma_analytical) / gamma_analytical) * 100

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"\n  Analytical Gamma:  {gamma_analytical:.8f}")
print(f"  Computed Gamma:    {gamma_computed:.8f}")
print(f"  Relative Error:    {gamma_error:.2f}%")

# Diagnose
if gamma_computed < 0:
    print(f"\n  ✗ CRITICAL ERROR: Gamma is NEGATIVE (should be positive for Call options)")
    print(f"  This indicates a fundamental problem in the implementation.")
elif gamma_error > 50:
    print(f"\n  ⚠ WARNING: Gamma error is very large (>{gamma_error:.0f}%)")
    print(f"  Possible causes:")
    print(f"    - Grid too coarse (try M=151, N=200)")
    print(f"    - Numerical instability")
    print(f"    - Edge-Pushing implementation bug")
elif gamma_error > 10:
    print(f"\n  ⚠ Gamma error is moderate ({gamma_error:.1f}%)")
    print(f"  This is expected for coarse grids at high volatility")
else:
    print(f"\n  ✓ Gamma looks reasonable (error < 10%)")

print()
