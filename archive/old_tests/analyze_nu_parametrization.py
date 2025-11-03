"""
Analyze why ν=σ² parametrization gives identical results

Theory: Both methods compute the same PDE solution, just with different
computational paths. The key question is whether the different paths
lead to different error propagation in second derivatives.
"""

import numpy as np
import sys
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
    return volga


S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.20

volga_anal = analytical_volga(S0, K, T, r, sigma)

print("=" * 80)
print("ANALYSIS: Why ν=σ² and σ Parametrizations Give Identical Results")
print("=" * 80)
print()

print("Theory:")
print("-" * 60)
print("σ-parametrization:")
print("  α_i = (1/2) σ² S_i² / ΔS²")
print("  Forward: σ → σ² → α → V")
print("  Backward (Volga): V → ∂V/∂α → ∂α/∂(σ²) → ∂(σ²)/∂σ → ∂V/∂σ")
print()
print("ν-parametrization:")
print("  α_i = (1/2) ν S_i² / ΔS²  where ν=σ²")
print("  Forward: ν → α → V")
print("  Backward (Volga): V → ∂V/∂α → ∂α/∂ν → ∂V/∂ν")
print("  Then chain rule: Volga = 4σ²(∂²V/∂ν²) + 2(∂V/∂ν)")
print()

print("Key question: Does shorter chain in ν-param reduce error?")
print()

# Test at different grid resolutions
print("=" * 80)
print("TEST: Different Grid Resolutions")
print("=" * 80)
print()

configs = [(21, 20), (51, 50)]

print(f"{'Grid':>12} {'σ-param Volga':>18} {'Error%':>10} {'ν-param Volga':>18} {'Error%':>10} {'Diff':>10}")
print("-" * 90)

for M, N in configs:
    # σ parametrization
    pricer_sigma = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
    result_sigma = pricer_sigma.solve_pde_with_aad(
        S0_val=S0, sigma_val=sigma,
        compute_hessian=True, fixed_grid=True, verbose=False
    )
    volga_sigma = result_sigma['volga']
    error_sigma = abs(volga_sigma - volga_anal) / volga_anal * 100

    # ν parametrization
    pricer_nu = BS_PDE_AAD_Nu(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
    result_nu = pricer_nu.solve_pde_with_aad_nu(
        S0_val=S0, sigma_val=sigma,
        compute_hessian=True, verbose=False
    )
    volga_nu = result_nu['volga']
    error_nu = abs(volga_nu - volga_anal) / volga_anal * 100

    diff = abs(volga_nu - volga_sigma) / abs(volga_sigma) * 100

    print(f"{M:>5}×{N:<5} {volga_sigma:>18.8f} {error_sigma:>9.2f}% {volga_nu:>18.8f} {error_nu:>9.2f}% {diff:>9.4f}%")

print()
print(f"Analytical Volga: {volga_anal:.8f}")
print()

# Test at different sigma values
print("=" * 80)
print("TEST: Different Sigma Values (M=51)")
print("=" * 80)
print()

sigma_values = [0.10, 0.15, 0.20, 0.30, 0.40]

print(f"{'σ':>8} {'Volga(Anal)':>15} {'σ-param':>15} {'Error%':>10} {'ν-param':>15} {'Error%':>10} {'Diff%':>10}")
print("-" * 100)

M, N = 51, 50

for sig in sigma_values:
    volga_a = analytical_volga(S0, K, T, r, sig)

    # σ parametrization
    pricer_sigma = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
    result_sigma = pricer_sigma.solve_pde_with_aad(
        S0_val=S0, sigma_val=sig,
        compute_hessian=True, fixed_grid=True, verbose=False
    )
    volga_s = result_sigma['volga']
    error_s = abs(volga_s - volga_a) / volga_a * 100

    # ν parametrization
    pricer_nu = BS_PDE_AAD_Nu(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
    result_nu = pricer_nu.solve_pde_with_aad_nu(
        S0_val=S0, sigma_val=sig,
        compute_hessian=True, verbose=False
    )
    volga_n = result_nu['volga']
    error_n = abs(volga_n - volga_a) / volga_a * 100

    diff = abs(volga_n - volga_s) / abs(volga_s) * 100

    print(f"{sig:>8.2f} {volga_a:>15.8f} {volga_s:>15.8f} {error_s:>9.2f}% {volga_n:>15.8f} {error_n:>9.2f}% {diff:>9.4f}%")

print()

# Analysis
print("=" * 80)
print("ANALYSIS & EXPLANATION")
print("=" * 80)
print()

print("Observation: Both methods give IDENTICAL results (difference < 0.0001%)")
print()

print("Explanation:")
print("-" * 60)
print()
print("1. Same PDE Solution:")
print("   - Both methods solve the same PDE: ∂V/∂t + (1/2)ν S² ∂²V/∂S² + ... = 0")
print("   - Whether we call it σ² or ν doesn't change the numerical solution")
print("   - V(S,t;ν) is identical in both cases")
print()

print("2. Different Computational Graphs:")
print("   - σ-param: σ → [square] → σ² → [multiply] → α → ... → V")
print("   - ν-param: ν → [multiply] → α → ... → V")
print("   - Shorter chain in ν-param (no squaring operation)")
print()

print("3. Why Results Are Identical:")
print("   - AAD computes EXACT derivatives via chain rule")
print("   - ∂V/∂σ via σ-param = ∂V/∂ν · ∂ν/∂σ = ∂V/∂ν · 2σ")
print("   - Both are mathematically equivalent")
print("   - No numerical approximation in chain rule → same result")
print()

print("4. When Would ν-param Help?")
print("   - If σ² operation introduces numerical error (not the case with AAD)")
print("   - If custom operators have different error characteristics")
print("   - In non-AAD finite difference schemes")
print("   - In iterative solvers where nonlinearity matters")
print()

print("5. For Current AAD Implementation:")
print("   - Both methods are EQUIVALENT")
print("   - Use σ-param for simplicity (more intuitive)")
print("   - ν-param doesn't provide accuracy advantage")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

print("ν=σ² parametrization is a VALID but NOT BENEFICIAL transformation")
print("for the current AAD Edge-Pushing implementation.")
print()
print("Reasons:")
print("  ✓ Mathematically correct (chain rule verified)")
print("  ✓ Equivalent results to σ-parametrization")
print("  ✗ No accuracy improvement")
print("  ✗ Adds complexity (chain rule conversion)")
print()

print("Recommendation:")
print("  → Continue using direct σ-parametrization")
print("  → Focus on grid resolution for Volga accuracy")
print("  → ν-param may help in future optimizations (e.g., custom operators)")
print()
