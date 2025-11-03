"""
Test fixes for AAD Volga and Bumping Gamma

1. AAD Volga: Test M=101, M=151 for convergence
2. Bumping Gamma: Test with new cubic spline interpolation
"""

import numpy as np
import sys
import time
from math import log, sqrt
from scipy.stats import norm

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def analytical_greeks(S0, K, T, r, sigma):
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)

    delta = N_d1
    vega = S0 * n_d1 * sqrt_T
    gamma = n_d1 / (S0 * sigma * sqrt_T)
    volga = vega * d1 * d2 / sigma

    return {'gamma': gamma, 'volga': volga, 'delta': delta, 'vega': vega}


def test_bumping_gamma_fix():
    """Test if cubic spline interpolation fixes Bumping Gamma"""
    print("=" * 80)
    print("TEST 1: Bumping Gamma with Cubic Spline Interpolation")
    print("=" * 80)
    print()

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    M = 51
    N = 50

    pricer = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    epsilon_S = 1.0

    print(f"Computing Gamma via bumping (epsilon={epsilon_S}):")
    print()

    V_base, _ = pricer._solve_pde_numerical(S0, sigma, fixed_grid=True)
    V_up, _ = pricer._solve_pde_numerical(S0 + epsilon_S, sigma, fixed_grid=True)
    V_down, _ = pricer._solve_pde_numerical(S0 - epsilon_S, sigma, fixed_grid=True)

    gamma_bumping = (V_up - 2 * V_base + V_down) / (epsilon_S ** 2)

    anal = analytical_greeks(S0, K, T, r, sigma)
    gamma_anal = anal['gamma']

    print(f"  V(S0-ε) = {V_down:.10f}")
    print(f"  V(S0)   = {V_base:.10f}")
    print(f"  V(S0+ε) = {V_up:.10f}")
    print()
    print(f"  Gamma (Bumping):    {gamma_bumping:.10f}")
    print(f"  Gamma (Analytical): {gamma_anal:.10f}")

    if abs(gamma_bumping) < 1e-10:
        print(f"\n  ✗ FAIL: Gamma still ≈ 0")
        print(f"    Cubic spline interpolation NOT working")
    else:
        error = abs(gamma_bumping - gamma_anal) / gamma_anal * 100
        print(f"\n  Error: {error:.2f}%")

        if error < 5:
            print(f"  ✓ SUCCESS: Cubic spline fixed the problem!")
        elif error < 20:
            print(f"  ○ PARTIAL: Better than 0, but still needs improvement")
        else:
            print(f"  ✗ POOR: Large error remains")

    print()
    return gamma_bumping, gamma_anal


def test_aad_volga_grid():
    """Test AAD Volga with different grid resolutions"""
    print("=" * 80)
    print("TEST 2: AAD Volga Grid Convergence")
    print("=" * 80)
    print()

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    anal = analytical_greeks(S0, K, T, r, sigma)
    volga_anal = anal['volga']

    print(f"Analytical Volga: {volga_anal:.8f}")
    print()

    configs = [(51, 50), (101, 100)]

    print(f"{'Grid':>12} {'Volga':>15} {'Error%':>10} {'Time(s)':>10}")
    print("-" * 60)

    for M, N in configs:
        pricer = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

        t_start = time.perf_counter()
        result = pricer.solve_pde_with_aad(
            S0_val=S0,
            sigma_val=sigma,
            compute_hessian=True,
            fixed_grid=True,
            verbose=False
        )
        t_elapsed = time.perf_counter() - t_start

        volga_aad = result['volga']
        error = abs(volga_aad - volga_anal) / volga_anal * 100

        status = "✓" if error < 10 else "○" if error < 20 else "✗"

        print(f"{M:>5}×{N:<5} {volga_aad:>15.8f} {error:>9.2f}% {t_elapsed:>9.1f}s {status}")

    print()


def test_all_greeks_comparison():
    """Compare all Greeks: AAD vs Bumping (both with fixes)"""
    print("=" * 80)
    print("TEST 3: Full Greeks Comparison (with fixes)")
    print("=" * 80)
    print()

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    M = 101  # Use finer grid
    N = 100

    pricer = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    # Analytical
    anal = analytical_greeks(S0, K, T, r, sigma)

    # AAD
    t_start = time.perf_counter()
    result_aad = pricer.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,
        fixed_grid=True
    )
    t_aad = time.perf_counter() - t_start

    # Bumping (Gamma only, since we fixed interpolation)
    epsilon_S = 1.0
    V_base, _ = pricer._solve_pde_numerical(S0, sigma, fixed_grid=True)
    V_up, _ = pricer._solve_pde_numerical(S0 + epsilon_S, sigma, fixed_grid=True)
    V_down, _ = pricer._solve_pde_numerical(S0 - epsilon_S, sigma, fixed_grid=True)
    gamma_bumping = (V_up - 2 * V_base + V_down) / (epsilon_S ** 2)

    # Bumping (Volga)
    epsilon_sigma = 0.001 * sigma
    V_sigma_up, _ = pricer._solve_pde_numerical(S0, sigma + epsilon_sigma, fixed_grid=True)
    V_sigma_down, _ = pricer._solve_pde_numerical(S0, sigma - epsilon_sigma, fixed_grid=True)
    volga_bumping = (V_sigma_up - 2 * V_base + V_sigma_down) / (epsilon_sigma ** 2)

    print(f"Grid: M={M}, N={N}")
    print()
    print(f"{'Greek':<10} {'Analytical':>15} {'AAD':>15} {'Bumping':>15}")
    print("-" * 60)
    print(f"{'Gamma':<10} {anal['gamma']:>15.8f} {result_aad['gamma']:>15.8f} {gamma_bumping:>15.8f}")
    print(f"{'Volga':<10} {anal['volga']:>15.8f} {result_aad['volga']:>15.8f} {volga_bumping:>15.8f}")
    print()

    print(f"{'Greek':<10} {'AAD Error%':>15} {'Bumping Error%':>15}")
    print("-" * 50)

    gamma_err_aad = abs(result_aad['gamma'] - anal['gamma']) / anal['gamma'] * 100
    gamma_err_bump = abs(gamma_bumping - anal['gamma']) / anal['gamma'] * 100
    volga_err_aad = abs(result_aad['volga'] - anal['volga']) / anal['volga'] * 100
    volga_err_bump = abs(volga_bumping - anal['volga']) / anal['volga'] * 100

    print(f"{'Gamma':<10} {gamma_err_aad:>14.2f}% {gamma_err_bump:>14.2f}%")
    print(f"{'Volga':<10} {volga_err_aad:>14.2f}% {volga_err_bump:>14.2f}%")
    print()

    print(f"AAD time: {t_aad:.1f}s")
    print()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "TEST FIXES" + " " * 43 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Test 1: Bumping Gamma fix
    test_bumping_gamma_fix()

    # Test 2: AAD Volga grid convergence
    test_aad_volga_grid()

    # Test 3: Full comparison
    test_all_greeks_comparison()

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("KEY FINDINGS:")
    print("1. Bumping Gamma: Cubic spline interpolation should fix the 0 problem")
    print("2. AAD Volga: Finer grid (M=101) should reduce error significantly")
    print("3. Trade-off: Computation time increases with grid resolution")
    print()


if __name__ == "__main__":
    main()
