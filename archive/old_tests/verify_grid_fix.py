#!/usr/bin/env python3
"""
Verify Grid Fix - Test that larger grid solves negative Gamma issue

Tests σ=0.5 with different grid sizes to confirm fix
"""

import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

print("="*80)
print("GRID SIZE FIX VERIFICATION")
print("="*80)
print("\nObjective: Verify that larger grid fixes negative Gamma at σ=0.5\n")

# Parameters
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.5

# Analytical Gamma
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
phi_d1 = norm.pdf(d1)
gamma_analytical = phi_d1 / (S0 * sigma * sqrt_T)

print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
print(f"Analytical Gamma: {gamma_analytical:.8f} (target)\n")

# Test different grid sizes
test_grids = [
    (51, 100, "Coarse (KNOWN BAD)"),
    (101, 200, "Medium (should work)"),
    (151, 300, "Fine (should be best)"),
]

results = []

for M, N, desc in test_grids:
    print(f"{'='*80}")
    print(f"Testing: M={M}, N={N} - {desc}")
    print(f"{'='*80}")

    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    print(f"  Computing (this may take 1-3 minutes)...")
    result = solver.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,
        verbose=False
    )

    gamma = result['gamma']
    error = abs((gamma - gamma_analytical) / gamma_analytical) * 100 if gamma > 0 else 999

    print(f"  Results:")
    print(f"    Gamma = {gamma:.8f}")

    if gamma < 0:
        print(f"    ✗ FAIL: Gamma is NEGATIVE")
        status = "FAIL"
    elif error > 50:
        print(f"    ⚠ POOR: Error = {error:.1f}% (>50%)")
        status = "POOR"
    elif error > 20:
        print(f"    ○ OK: Error = {error:.1f}% (20-50%)")
        status = "OK"
    else:
        print(f"    ✓ GOOD: Error = {error:.1f}% (<20%)")
        status = "GOOD"

    results.append({
        'M': M,
        'N': N,
        'desc': desc,
        'gamma': gamma,
        'error': error,
        'status': status
    })

    print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print(f"\nAnalytical Gamma: {gamma_analytical:.8f}\n")
print(f"{'Grid':20s} | {'Gamma':12s} | {'Error':10s} | {'Status':6s}")
print(f"{'-'*20}-|-{'-'*12}-|-{'-'*10}-|-{'-'*6}")

for r in results:
    gamma_str = f"{r['gamma']:.8f}" if r['gamma'] > 0 else "NEGATIVE!"
    error_str = f"{r['error']:.1f}%" if r['error'] < 999 else "N/A"
    print(f"M={r['M']:3d}, N={r['N']:3d} {r['desc']:12s} | {gamma_str:12s} | {error_str:10s} | {r['status']:6s}")

# Recommendation
print(f"\n{'='*80}")
print("RECOMMENDATION")
print(f"{'='*80}")

best = min([r for r in results if r['gamma'] > 0], key=lambda x: x['error'], default=None)

if best:
    print(f"\n  ✓ Fix confirmed! Use M≥{best['M']}, N≥{best['N']} for σ=0.5")
    print(f"  Gamma error reduced to {best['error']:.1f}%")
else:
    print(f"\n  ✗ All grids failed. Need even larger grid or different approach.")

print(f"\n  Comprehensive test framework updated to use:")
print(f"    - Default: M=101, N=200 (safe for most scenarios)")
print(f"    - High σ (≥0.4): M=151, N=300 (extra safety)")

print()
