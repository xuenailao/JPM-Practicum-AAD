"""
Test script to verify fixed grid solution for Volga accuracy

This script compares:
1. Adaptive grid (legacy): N depends on sigma → network grid jumping noise
2. Fixed grid (new): N is constant → eliminates dN/dσ
3. Analytical (baseline): Black-Scholes formula → 0% error

Expected results after fix:
- AAD (adaptive): ~26.96% error (legacy behavior)
- AAD (fixed): ~26.96% error (consistent with adaptive when no grid jump)
- Bumping (adaptive): ~60.53% error (grid jumping noise)
- Bumping (fixed): ~26.96% error (should match AAD!)  ← KEY FIX
- Analytical: 0% error (baseline)
"""

import numpy as np
import sys
import time

# Add parent directory to path
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD
from aad_edge_pushing.pde.bsm_analytical import BSMAnalytical
from scipy.stats import norm
from math import log, sqrt, exp


def bsm_analytical_greeks(S0, K, T, r, sigma):
    """Compute BS analytical Greeks"""
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    price = S0 * N_d1 - K * exp(-r * T) * N_d2
    delta = N_d1
    vega = S0 * n_d1 * sqrt_T
    gamma = n_d1 / (S0 * sigma * sqrt_T)
    vanna = -n_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return {
        'price': price,
        'delta': delta,
        'vega': vega,
        'gamma': gamma,
        'vanna': vanna,
        'volga': volga
    }


def compute_volga_bumping(pricer, S0, sigma, epsilon=0.001, fixed_grid=False):
    """
    Compute Volga via bumping (finite difference)

    Volga = ∂²V/∂σ² ≈ [V(σ+ε) - 2V(σ) + V(σ-ε)] / ε²
    """
    # Solve PDE at three sigma values
    V_plus, _ = pricer._solve_pde_numerical(S0, sigma + epsilon, fixed_grid=fixed_grid)
    V_base, _ = pricer._solve_pde_numerical(S0, sigma, fixed_grid=fixed_grid)
    V_minus, _ = pricer._solve_pde_numerical(S0, sigma - epsilon, fixed_grid=fixed_grid)

    volga = (V_plus - 2.0 * V_base + V_minus) / (epsilon ** 2)
    return volga


def main():
    print("=" * 80)
    print("Volga Accuracy Test: Fixed Grid vs Adaptive Grid")
    print("=" * 80)
    print()

    # Test parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    M = 51  # Spatial grid points
    N = 50  # Time steps (for fixed grid)

    print(f"Parameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Grid: M={M}, N={N}")
    print()

    # Create PDE pricer
    pricer = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    # ========================================================================
    # 1. Analytical Baseline (Black-Scholes)
    # ========================================================================
    print("-" * 80)
    print("1. ANALYTICAL (Black-Scholes) - Baseline")
    print("-" * 80)

    bs_result = bsm_analytical_greeks(S0, K, T, r, sigma)

    price_bs = bs_result['price']
    vega_bs = bs_result['vega']
    volga_bs = bs_result['volga']

    print(f"Price:  {price_bs:.8f}")
    print(f"Vega:   {vega_bs:.8f}")
    print(f"Volga:  {volga_bs:.8f} (analytical)")
    print()

    # ========================================================================
    # 2. AAD Edge-Pushing (Adaptive Grid - Legacy)
    # ========================================================================
    print("-" * 80)
    print("2. AAD EDGE-PUSHING (Adaptive Grid) - Legacy Behavior")
    print("-" * 80)

    t_start = time.perf_counter()
    result_aad_adaptive = pricer.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,
        fixed_grid=False,  # Use adaptive timesteps
        use_analytical_volga=False,
        verbose=False
    )
    t_end = time.perf_counter()
    time_aad_adaptive = (t_end - t_start) * 1000

    price_aad = result_aad_adaptive['price']
    vega_aad = result_aad_adaptive['vega']
    volga_aad_adaptive = result_aad_adaptive['volga']

    error_price_aad = abs(price_aad - price_bs) / price_bs * 100
    error_vega_aad = abs(vega_aad - vega_bs) / vega_bs * 100
    error_volga_aad_adaptive = abs(volga_aad_adaptive - volga_bs) / abs(volga_bs) * 100

    print(f"Price:  {price_aad:.8f} (error: {error_price_aad:.2f}%)")
    print(f"Vega:   {vega_aad:.8f} (error: {error_vega_aad:.2f}%)")
    print(f"Volga:  {volga_aad_adaptive:.8f} (error: {error_volga_aad_adaptive:.2f}%)")
    print(f"Time:   {time_aad_adaptive:.1f} ms")
    print()

    # ========================================================================
    # 3. AAD Edge-Pushing (Fixed Grid - NEW)
    # ========================================================================
    print("-" * 80)
    print("3. AAD EDGE-PUSHING (Fixed Grid) - NEW")
    print("-" * 80)

    t_start = time.perf_counter()
    result_aad_fixed = pricer.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,
        fixed_grid=True,  # Use fixed N
        use_analytical_volga=False,
        verbose=False
    )
    t_end = time.perf_counter()
    time_aad_fixed = (t_end - t_start) * 1000

    volga_aad_fixed = result_aad_fixed['volga']
    error_volga_aad_fixed = abs(volga_aad_fixed - volga_bs) / abs(volga_bs) * 100

    print(f"Volga:  {volga_aad_fixed:.8f} (error: {error_volga_aad_fixed:.2f}%)")
    print(f"Time:   {time_aad_fixed:.1f} ms")
    print()

    # ========================================================================
    # 4. Bumping (Adaptive Grid - Legacy)
    # ========================================================================
    print("-" * 80)
    print("4. BUMPING (Adaptive Grid) - Legacy Behavior")
    print("-" * 80)

    epsilon = 0.001 * sigma  # 0.1% of sigma

    t_start = time.perf_counter()
    volga_bump_adaptive = compute_volga_bumping(
        pricer, S0, sigma, epsilon=epsilon, fixed_grid=False
    )
    t_end = time.perf_counter()
    time_bump_adaptive = (t_end - t_start) * 1000

    error_volga_bump_adaptive = abs(volga_bump_adaptive - volga_bs) / abs(volga_bs) * 100

    print(f"Volga:  {volga_bump_adaptive:.8f} (error: {error_volga_bump_adaptive:.2f}%)")
    print(f"Time:   {time_bump_adaptive:.1f} ms")
    print(f"Epsilon: {epsilon:.6f}")
    print()

    # ========================================================================
    # 5. Bumping (Fixed Grid - NEW)
    # ========================================================================
    print("-" * 80)
    print("5. BUMPING (Fixed Grid) - NEW ← KEY FIX")
    print("-" * 80)

    t_start = time.perf_counter()
    volga_bump_fixed = compute_volga_bumping(
        pricer, S0, sigma, epsilon=epsilon, fixed_grid=True
    )
    t_end = time.perf_counter()
    time_bump_fixed = (t_end - t_start) * 1000

    error_volga_bump_fixed = abs(volga_bump_fixed - volga_bs) / abs(volga_bs) * 100

    print(f"Volga:  {volga_bump_fixed:.8f} (error: {error_volga_bump_fixed:.2f}%)")
    print(f"Time:   {time_bump_fixed:.1f} ms")
    print()

    # ========================================================================
    # 6. AAD with Analytical Volga
    # ========================================================================
    print("-" * 80)
    print("6. AAD + ANALYTICAL VOLGA (Reference)")
    print("-" * 80)

    result_aad_analytical = pricer.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,
        fixed_grid=True,
        use_analytical_volga=True,
        verbose=False
    )

    volga_analytical = result_aad_analytical['volga_analytical']
    volga_pde_stored = result_aad_analytical['volga_pde']

    error_analytical = abs(volga_analytical - volga_bs) / abs(volga_bs) * 100

    print(f"Volga (PDE):        {volga_pde_stored:.8f}")
    print(f"Volga (Analytical): {volga_analytical:.8f} (error: {error_analytical:.4f}%)")
    print()

    # ========================================================================
    # Summary Table
    # ========================================================================
    print("=" * 80)
    print("SUMMARY: Volga Comparison")
    print("=" * 80)
    print()
    print(f"{'Method':<30} {'Volga':>12} {'Error':>10} {'Time (ms)':>12}")
    print("-" * 80)
    print(f"{'Analytical (BS)':<30} {volga_bs:>12.8f} {'0.00%':>10} {'-':>12}")
    print(f"{'AAD (Adaptive Grid)':<30} {volga_aad_adaptive:>12.8f} {error_volga_aad_adaptive:>9.2f}% {time_aad_adaptive:>11.1f}")
    print(f"{'AAD (Fixed Grid)':<30} {volga_aad_fixed:>12.8f} {error_volga_aad_fixed:>9.2f}% {time_aad_fixed:>11.1f}")
    print(f"{'Bumping (Adaptive Grid)':<30} {volga_bump_adaptive:>12.8f} {error_volga_bump_adaptive:>9.2f}% {time_bump_adaptive:>11.1f}")
    print(f"{'Bumping (Fixed Grid)':<30} {volga_bump_fixed:>12.8f} {error_volga_bump_fixed:>9.2f}% {time_bump_fixed:>11.1f}")
    print(f"{'AAD + Analytical':<30} {volga_analytical:>12.8f} {error_analytical:>9.4f}% {'-':>12}")
    print()

    # ========================================================================
    # Validation Checks
    # ========================================================================
    print("=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    print()

    # Check 1: AAD adaptive vs fixed should be similar (if no grid jump)
    diff_aad = abs(volga_aad_fixed - volga_aad_adaptive) / abs(volga_aad_adaptive) * 100
    print(f"Check 1: AAD (Fixed vs Adaptive) difference: {diff_aad:.2f}%")
    if diff_aad < 5.0:
        print("  ✓ PASS: AAD methods are consistent")
    else:
        print("  ✗ WARNING: Significant difference, check implementation")
    print()

    # Check 2: Bumping fixed should match AAD fixed (both eliminate dN/dσ)
    diff_methods = abs(volga_bump_fixed - volga_aad_fixed) / abs(volga_aad_fixed) * 100
    print(f"Check 2: Bumping (Fixed) vs AAD (Fixed) difference: {diff_methods:.2f}%")
    if diff_methods < 10.0:
        print("  ✓ PASS: Bumping and AAD converge on fixed grid")
    else:
        print("  ✗ WARNING: Methods should be closer on fixed grid")
    print()

    # Check 3: Bumping error should drop significantly with fixed grid
    error_reduction = error_volga_bump_adaptive - error_volga_bump_fixed
    print(f"Check 3: Bumping error reduction: {error_reduction:.2f}% ")
    print(f"  (Adaptive: {error_volga_bump_adaptive:.2f}% → Fixed: {error_volga_bump_fixed:.2f}%)")
    if error_reduction > 10.0:
        print(f"  ✓ PASS: Fixed grid eliminates grid-jumping noise ({error_reduction:.1f}% improvement)")
    else:
        print("  ✗ WARNING: Expected larger error reduction")
    print()

    # Check 4: Analytical Volga should be nearly perfect
    if error_analytical < 0.01:
        print(f"Check 4: Analytical Volga error: {error_analytical:.4f}%")
        print("  ✓ PASS: Analytical formula is accurate")
    else:
        print(f"  ✗ FAIL: Analytical error too high: {error_analytical:.4f}%")
    print()

    print("=" * 80)
    print("Test completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
