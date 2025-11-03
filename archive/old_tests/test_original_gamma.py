#!/usr/bin/env python3
"""
Test Original BS_PDE_AAD Gamma

Check if the original solver also has negative Gamma issue
"""

import numpy as np
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

print("="*80)
print("TESTING ORIGINAL BS_PDE_AAD GAMMA")
print("="*80)

# Parameters
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.5
M, N = 51, 100

# Analytical Gamma
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
phi_d1 = norm.pdf(d1)
gamma_analytical = phi_d1 / (S0 * sigma * sqrt_T)

print(f"\nAnalytical Gamma: {gamma_analytical:.6f}")

# Test original solver
print(f"\nTesting original BS_PDE_AAD...")
solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

print(f"  Computing Hessian (this may take a while)...")
result = solver.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True,
    verbose=False
)

print(f"\nResults:")
print(f"  Price = {result['price']:.6f}")
print(f"  Delta = {result['delta']:.6f}")
print(f"  Gamma = {result['gamma']:.6f}")
print(f"  Vega  = {result['vega']:.6f}")

error = abs((result['gamma'] - gamma_analytical) / gamma_analytical) * 100
print(f"\n  Gamma Error: {error:.2f}%")

if result['gamma'] < 0:
    print(f"  ✗ ERROR: Gamma is NEGATIVE (should be positive)")
    print(f"  This is the ROOT CAUSE of the problem!")
else:
    print(f"  ✓ Gamma sign is correct")

print()
