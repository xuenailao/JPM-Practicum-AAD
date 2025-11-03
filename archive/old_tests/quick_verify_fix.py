#!/usr/bin/env python3
"""
Quick verification that larger grid fixes negative Gamma at σ=0.5
"""

import numpy as np
from scipy.stats import norm
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

print("="*80)
print("QUICK GRID FIX VERIFICATION")
print("="*80)
print("\nTesting σ=0.5 with two grid configurations\n")

# Parameters
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.5

# Analytical Gamma
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
phi_d1 = norm.pdf(d1)
gamma_analytical = phi_d1 / (S0 * sigma * sqrt_T)

print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
print(f"Analytical Gamma: {gamma_analytical:.8f} (target)\n")

# Test 1: Coarse grid (KNOWN BAD)
print("="*80)
print("TEST 1: Coarse Grid (M=51, N=100) - KNOWN BAD")
print("="*80)

solver1 = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=51, N_base=100)
print("Computing (30-60 seconds)...")
result1 = solver1.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True,
    verbose=False
)

gamma1 = result1['gamma']
print(f"Gamma = {gamma1:.8f}")

if gamma1 < 0:
    print(f"✗ NEGATIVE - confirms problem")
else:
    error1 = abs((gamma1 - gamma_analytical) / gamma_analytical) * 100
    print(f"Error = {error1:.1f}%")

print()

# Test 2: Larger grid (SHOULD FIX)
print("="*80)
print("TEST 2: Larger Grid (M=101, N=200) - EXPECTED FIX")
print("="*80)

solver2 = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=101, N_base=200)
print("Computing (2-3 minutes)...")
result2 = solver2.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True,
    verbose=False
)

gamma2 = result2['gamma']
print(f"Gamma = {gamma2:.8f}")

if gamma2 < 0:
    print(f"✗ STILL NEGATIVE - fix didn't work!")
else:
    error2 = abs((gamma2 - gamma_analytical) / gamma_analytical) * 100
    print(f"Error = {error2:.1f}%")
    if error2 < 20:
        print(f"✓ GOOD - error <20%")
    elif error2 < 50:
        print(f"○ OK - error 20-50%")
    else:
        print(f"⚠ POOR - error >50%")

print()

# Summary
print("="*80)
print("CONCLUSION")
print("="*80)

print(f"\nAnalytical Gamma: {gamma_analytical:.8f}")
print(f"\nM=51, N=100:   Gamma = {gamma1:.8f}  {'(NEGATIVE)' if gamma1 < 0 else f'(Error: {abs((gamma1 - gamma_analytical) / gamma_analytical) * 100:.1f}%)'}")
print(f"M=101, N=200:  Gamma = {gamma2:.8f}  {'(NEGATIVE)' if gamma2 < 0 else f'(Error: {abs((gamma2 - gamma_analytical) / gamma_analytical) * 100:.1f}%)'}")

if gamma1 < 0 and gamma2 > 0:
    print(f"\n✓ FIX CONFIRMED: Larger grid (M=101, N=200) eliminates negative Gamma")
elif gamma1 < 0 and gamma2 < 0:
    print(f"\n✗ FIX FAILED: Both grids produce negative Gamma")
else:
    print(f"\n? UNEXPECTED: M=51,N=100 gave positive Gamma")

print()
