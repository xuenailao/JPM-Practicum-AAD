"""
Compare ν=σ² parametrization vs direct σ parametrization

Test if ν parametrization truly reduces Volga error
"""

import numpy as np
import sys
import time
from math import log, sqrt
from scipy.stats import norm

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD
from aad_edge_pushing.pde.pde_aad_nu_parametrization import BS_PDE_AAD_Nu


def analytical_volga(S0, K, T, r, sigma):
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    n_d1 = norm.pdf(d1)
    vega = S0 * n_d1 * sqrt_T
    volga = vega * d1 * d2 / sigma
    return volga, vega


S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20

volga_anal, vega_anal = analytical_volga(S0, K, T, r, sigma)

print("=" * 80)
print("COMPARISON: ν=σ² Parametrization vs Direct σ Parametrization")
print("=" * 80)
print()
print(f"Analytical Volga: {volga_anal:.8f}")
print(f"Analytical Vega:  {vega_anal:.8f}")
print()

# Test different grid resolutions
configs = [(51, 50)]

print(f"{'Grid':>12} {'σ-param Volga':>18} {'Error%':>10} {'ν-param Volga':>18} {'Error%':>10} {'Improvement':>15}")
print("-" * 100)

for M, N in configs:
    # Direct σ parametrization
    pricer_sigma = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    t_start = time.perf_counter()
    result_sigma = pricer_sigma.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,
        fixed_grid=True
    )
    t_sigma = time.perf_counter() - t_start

    volga_sigma = result_sigma['volga']
    error_sigma = abs(volga_sigma - volga_anal) / volga_anal * 100

    # ν=σ² parametrization
    pricer_nu = BS_PDE_AAD_Nu(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    t_start = time.perf_counter()
    result_nu = pricer_nu.solve_pde_with_aad_nu(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True
    )
    t_nu = time.perf_counter() - t_start

    volga_nu = result_nu['volga']
    error_nu = abs(volga_nu - volga_anal) / volga_anal * 100

    improvement = error_sigma - error_nu

    status = "✓" if improvement > 0 else "✗"

    print(f"{M:>5}×{N:<5} {volga_sigma:>18.8f} {error_sigma:>9.2f}% {volga_nu:>18.8f} {error_nu:>9.2f}% {improvement:>+13.2f}% {status}")

print()
print("If improvement > 0: ν parametrization is better")
print("If improvement < 0: σ parametrization is better")
print()

# Detailed comparison for M=51
print("=" * 80)
print("DETAILED ANALYSIS (M=51)")
print("=" * 80)
print()

M, N = 51, 50

pricer_sigma = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
result_sigma = pricer_sigma.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True,
    fixed_grid=True
)

pricer_nu = BS_PDE_AAD_Nu(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
result_nu = pricer_nu.solve_pde_with_aad_nu(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True
)

print("σ-parametrization:")
print(f"  Vega:  {result_sigma['vega']:.8f}  (error: {abs(result_sigma['vega'] - vega_anal) / vega_anal * 100:.2f}%)")
print(f"  Volga: {result_sigma['volga']:.8f}  (error: {abs(result_sigma['volga'] - volga_anal) / volga_anal * 100:.2f}%)")
print()

print("ν-parametrization:")
print(f"  Vega:  {result_nu['vega']:.8f}  (error: {abs(result_nu['vega'] - vega_anal) / vega_anal * 100:.2f}%)")
print(f"  Volga: {result_nu['volga']:.8f}  (error: {abs(result_nu['volga'] - volga_anal) / volga_anal * 100:.2f}%)")
print(f"  dV/dν: {result_nu['dV_dnu']:.8f}")
print(f"  d²V/dν²: {result_nu['d2V_dnu2']:.8f}")
print()

# Check chain rule manually
vega_from_nu = result_nu['dV_dnu'] * 2 * sigma
volga_from_nu = 4 * sigma**2 * result_nu['d2V_dnu2'] + 2 * result_nu['dV_dnu']

print("Chain rule verification:")
print(f"  Vega from chain rule:  {vega_from_nu:.8f}")
print(f"  Volga from chain rule: {volga_from_nu:.8f}")
print(f"  Match result_nu: {abs(vega_from_nu - result_nu['vega']) < 1e-10 and abs(volga_from_nu - result_nu['volga']) < 1e-10}")
print()
