#!/usr/bin/env python3
"""
Ultra-quick test - just verify sign of Gamma
"""
import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

print("="*60)
print("ULTRA-QUICK VERIFICATION")
print("="*60)

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.5

# Analytical
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
gamma_analytical = norm.pdf(d1) / (S0 * sigma * sqrt_T)

print(f"\nσ=0.5, Analytical Gamma = {gamma_analytical:.8f}")

# Test new default grid
print(f"\nTesting M=101, N=200...")
solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=101, N_base=200)
result = solver.solve_pde_with_aad(S0_val=S0, sigma_val=sigma,
                                    compute_hessian=True, verbose=False)
gamma = result['gamma']

print(f"Computed Gamma = {gamma:.8f}")

if gamma > 0:
    error = abs((gamma - gamma_analytical) / gamma_analytical) * 100
    print(f"Error = {error:.1f}%")
    print(f"\n✓ SUCCESS: Gamma is POSITIVE (fix works!)")
else:
    print(f"\n✗ FAIL: Gamma is still NEGATIVE")

print("="*60)
