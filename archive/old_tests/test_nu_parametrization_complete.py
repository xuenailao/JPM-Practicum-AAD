"""
Complete test of ν=σ² parametrization

Test at M=51 to see if ν-parametrization truly reduces Volga error
compared to direct σ-parametrization
"""

import numpy as np
import sys
import time
from math import log, sqrt
from scipy.stats import norm

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD
from aad_edge_pushing.pde.pde_aad_nu_parametrization import BS_PDE_AAD_Nu


def analytical_greeks(S0, K, T, r, sigma):
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)

    price = S0 * N_d1 - K * np.exp(-r * T) * norm.cdf(d2)
    delta = N_d1
    vega = S0 * n_d1 * sqrt_T
    gamma = n_d1 / (S0 * sigma * sqrt_T)
    volga = vega * d1 * d2 / sigma
    vanna = -n_d1 * d2 / sigma

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20
M = 51
N = 50

anal = analytical_greeks(S0, K, T, r, sigma)

print("=" * 80)
print("COMPLETE TEST: ν=σ² Parametrization vs Direct σ Parametrization")
print("=" * 80)
print()
print(f"Grid: M={M}, N={N}")
print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
print()

# Analytical baseline
print("Analytical Greeks (Baseline):")
print("-" * 60)
for greek, value in anal.items():
    print(f"  {greek.capitalize():<10} {value:>15.8f}")
print()

# Test 1: Direct σ parametrization
print("=" * 80)
print("TEST 1: Direct σ Parametrization")
print("=" * 80)
print()

pricer_sigma = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

t_start = time.perf_counter()
result_sigma = pricer_sigma.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True,
    fixed_grid=True,
    verbose=True
)
t_sigma = time.perf_counter() - t_start

print()
print("Results (σ-parametrization):")
print("-" * 60)
greeks_sigma = ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']
for greek in greeks_sigma:
    if greek in result_sigma:
        val = result_sigma[greek]
        anal_val = anal[greek]
        error = abs(val - anal_val) / abs(anal_val) * 100 if abs(anal_val) > 1e-10 else 0
        status = "✓" if error < 10 else "○" if error < 20 else "✗"
        print(f"  {greek.capitalize():<10} {val:>15.8f}  (error: {error:>6.2f}%) {status}")

print(f"\nTime: {t_sigma:.2f}s")
print()

# Test 2: ν=σ² parametrization
print("=" * 80)
print("TEST 2: ν=σ² Parametrization")
print("=" * 80)
print()

pricer_nu = BS_PDE_AAD_Nu(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

t_start = time.perf_counter()
result_nu = pricer_nu.solve_pde_with_aad_nu(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=True,
    verbose=True
)
t_nu = time.perf_counter() - t_start

print()
print("Results (ν-parametrization):")
print("-" * 60)
greeks_nu = ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']
for greek in greeks_nu:
    if greek in result_nu:
        val = result_nu[greek]
        anal_val = anal[greek]
        error = abs(val - anal_val) / abs(anal_val) * 100 if abs(anal_val) > 1e-10 else 0
        status = "✓" if error < 10 else "○" if error < 20 else "✗"
        print(f"  {greek.capitalize():<10} {val:>15.8f}  (error: {error:>6.2f}%) {status}")

print(f"\nTime: {t_nu:.2f}s")
print()

# Raw derivatives w.r.t. ν
print("Raw derivatives w.r.t. ν:")
print(f"  dV/dν:   {result_nu['dV_dnu']:.8f}")
print(f"  d²V/dν²: {result_nu['d2V_dnu2']:.8f}")
print()

# Verify chain rule
vega_from_nu = result_nu['dV_dnu'] * 2 * sigma
volga_from_nu = 4 * sigma**2 * result_nu['d2V_dnu2'] + 2 * result_nu['dV_dnu']

print("Chain rule verification:")
print(f"  Vega from chain rule:  {vega_from_nu:.8f}")
print(f"  Vega from result:      {result_nu['vega']:.8f}")
print(f"  Match: {abs(vega_from_nu - result_nu['vega']) < 1e-8}")
print()
print(f"  Volga from chain rule: {volga_from_nu:.8f}")
print(f"  Volga from result:     {result_nu['volga']:.8f}")
print(f"  Match: {abs(volga_from_nu - result_nu['volga']) < 1e-8}")
print()

# Comparison table
print("=" * 80)
print("COMPARISON: σ vs ν Parametrization")
print("=" * 80)
print()

print(f"{'Greek':<10} {'Analytical':>15} {'σ-param':>15} {'Error%':>10} {'ν-param':>15} {'Error%':>10} {'Better':>10}")
print("-" * 100)

for greek in ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']:
    anal_val = anal[greek]
    sigma_val = result_sigma.get(greek, 0)
    nu_val = result_nu.get(greek, 0)

    error_sigma = abs(sigma_val - anal_val) / abs(anal_val) * 100 if abs(anal_val) > 1e-10 else 0
    error_nu = abs(nu_val - anal_val) / abs(anal_val) * 100 if abs(anal_val) > 1e-10 else 0

    if error_nu < error_sigma:
        better = "ν ✓"
    elif error_sigma < error_nu:
        better = "σ"
    else:
        better = "Same"

    print(f"{greek.capitalize():<10} {anal_val:>15.8f} {sigma_val:>15.8f} {error_sigma:>9.2f}% {nu_val:>15.8f} {error_nu:>9.2f}% {better:>10}")

print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

volga_improvement = abs(result_sigma['volga'] - anal['volga']) / abs(anal['volga']) * 100 - \
                   abs(result_nu['volga'] - anal['volga']) / abs(anal['volga']) * 100

print(f"Volga error reduction: {volga_improvement:+.2f}%")
if volga_improvement > 1:
    print(f"  → ν-parametrization BETTER by {volga_improvement:.2f}%")
elif volga_improvement < -1:
    print(f"  → σ-parametrization BETTER by {-volga_improvement:.2f}%")
else:
    print(f"  → No significant difference")

print()
print(f"Time ratio (ν/σ): {t_nu/t_sigma:.2f}x")
print()

# Conclusion
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if volga_improvement > 5:
    print("✓ SUCCESS: ν-parametrization significantly improves Volga accuracy!")
    print(f"  Error reduction: {volga_improvement:.1f}%")
elif volga_improvement > 1:
    print("○ MODEST: ν-parametrization provides modest improvement")
    print(f"  Error reduction: {volga_improvement:.1f}%")
elif volga_improvement > -1:
    print("= NEUTRAL: Both parametrizations perform similarly")
    print(f"  Difference: {abs(volga_improvement):.1f}%")
else:
    print("✗ WORSE: ν-parametrization performs worse than σ-parametrization")
    print(f"  Error increase: {-volga_improvement:.1f}%")

print()
print("Key observations:")
print("  1. Both methods use the same PDE discretization")
print("  2. Difference is in how σ enters the computation graph")
print("  3. ν-parametrization removes σ² nonlinearity in diffusion coefficient")
print("  4. At current grid resolution, may not show significant advantage")
print()
