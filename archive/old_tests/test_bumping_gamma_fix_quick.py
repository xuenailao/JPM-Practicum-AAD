"""
Quick test: Did cubic spline fix Bumping Gamma?
"""

import numpy as np
import sys
from math import log, sqrt
from scipy.stats import norm

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def analytical_gamma(S0, K, T, r, sigma):
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    n_d1 = norm.pdf(d1)
    return n_d1 / (S0 * sigma * sqrt_T)


S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20
M = 51
N = 50

pricer = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

epsilon_S = 1.0

print("Testing Bumping Gamma with Cubic Spline Interpolation")
print("=" * 60)
print()

V_base, _ = pricer._solve_pde_numerical(S0, sigma, fixed_grid=True)
V_up, _ = pricer._solve_pde_numerical(S0 + epsilon_S, sigma, fixed_grid=True)
V_down, _ = pricer._solve_pde_numerical(S0 - epsilon_S, sigma, fixed_grid=True)

gamma_bumping = (V_up - 2 * V_base + V_down) / (epsilon_S ** 2)
gamma_anal = analytical_gamma(S0, K, T, r, sigma)

print(f"V(99) = {V_down:.10f}")
print(f"V(100) = {V_base:.10f}")
print(f"V(101) = {V_up:.10f}")
print()
print(f"Gamma (Bumping):    {gamma_bumping:.10f}")
print(f"Gamma (Analytical): {gamma_anal:.10f}")
print()

if abs(gamma_bumping) < 1e-10:
    print("✗ FAILED: Gamma still = 0")
    print("  Cubic spline interpolation NOT working correctly")
else:
    error = abs(gamma_bumping - gamma_anal) / gamma_anal * 100
    print(f"Error: {error:.2f}%")

    if error < 5:
        print("✓ SUCCESS: Cubic spline FIXED Bumping Gamma!")
    elif error < 20:
        print("○ PARTIAL: Better, but still needs tuning")
    else:
        print("△ POOR: Large error remains")

print()
