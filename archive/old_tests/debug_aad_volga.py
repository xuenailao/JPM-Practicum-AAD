"""
Debug AAD Volga Computation

Goal: Understand why AAD Volga has 27% error
- Trace the computation graph
- Check how sigma propagates through PDE
- Verify Hessian computation
"""

import numpy as np
import sys
from math import log, sqrt, exp
from scipy.stats import norm

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def analytical_volga(S0, K, T, r, sigma):
    """Compute analytical Volga for comparison"""
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    n_d1 = norm.pdf(d1)
    vega = S0 * n_d1 * sqrt_T
    volga = vega * d1 * d2 / sigma

    return volga, vega, d1, d2


def test_vega_curve():
    """Test if Vega changes correctly with sigma"""
    print("=" * 80)
    print("TEST 1: Vega Curve (∂V/∂σ at different σ)")
    print("=" * 80)
    print()

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    M = 51
    N = 50

    sigma_values = [0.18, 0.19, 0.20, 0.21, 0.22]

    print(f"{'σ':>8} {'Vega(PDE)':>15} {'Vega(BS)':>15} {'Error%':>10}")
    print("-" * 60)

    pde_vegas = []
    bs_vegas = []

    for sigma in sigma_values:
        pricer = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

        result = pricer.solve_pde_with_aad(
            S0_val=S0,
            sigma_val=sigma,
            compute_hessian=False,
            fixed_grid=True,
            verbose=False
        )

        vega_pde = result['vega']
        pde_vegas.append(vega_pde)

        # Analytical
        sqrt_T = sqrt(T)
        d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        n_d1 = norm.pdf(d1)
        vega_bs = S0 * n_d1 * sqrt_T
        bs_vegas.append(vega_bs)

        error = abs(vega_pde - vega_bs) / vega_bs * 100

        print(f"{sigma:>8.2f} {vega_pde:>15.8f} {vega_bs:>15.8f} {error:>9.2f}%")

    print()
    print("Now compute Volga from Vega slope:")
    print("-" * 60)

    # Compute Volga via central difference of Vega
    sigma_mid = sigma_values[2]  # 0.20
    h = sigma_values[1] - sigma_values[0]  # 0.01

    # PDE: Central difference
    volga_pde_fd = (pde_vegas[3] - pde_vegas[1]) / (2 * h)

    # BS: Central difference
    volga_bs_fd = (bs_vegas[3] - bs_vegas[1]) / (2 * h)

    # BS: Analytical
    volga_bs_exact, _, d1, d2 = analytical_volga(S0, K, T, r, sigma_mid)

    print(f"\nVolga from finite difference of Vega curve:")
    print(f"  PDE:  {volga_pde_fd:.8f}")
    print(f"  BS:   {volga_bs_fd:.8f}")
    print(f"  Error: {abs(volga_pde_fd - volga_bs_fd) / volga_bs_fd * 100:.2f}%")
    print()
    print(f"Volga analytical (exact formula):")
    print(f"  BS:   {volga_bs_exact:.8f}")
    print()

    return pde_vegas, bs_vegas


def test_aad_volga_computation():
    """Test AAD Hessian computation for Volga"""
    print("=" * 80)
    print("TEST 2: AAD Hessian Computation")
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

    result = pricer.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,
        fixed_grid=True,
        verbose=False
    )

    volga_aad = result['volga']
    vega_aad = result['vega']

    volga_bs, vega_bs, d1, d2 = analytical_volga(S0, K, T, r, sigma)

    print(f"Vega (AAD):  {vega_aad:.8f}")
    print(f"Vega (BS):   {vega_bs:.8f}")
    print(f"Vega error:  {abs(vega_aad - vega_bs) / vega_bs * 100:.2f}%")
    print()
    print(f"Volga (AAD): {volga_aad:.8f}")
    print(f"Volga (BS):  {volga_bs:.8f}")
    print(f"Volga error: {abs(volga_aad - volga_bs) / volga_bs * 100:.2f}%")
    print()

    # Check if Volga matches finite difference of Vega
    print("Check: Does AAD Volga match finite difference of Vega?")

    eps = 0.001

    result_up = pricer.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma + eps,
        compute_hessian=False,
        fixed_grid=True
    )

    result_down = pricer.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma - eps,
        compute_hessian=False,
        fixed_grid=True
    )

    volga_fd = (result_up['vega'] - result_down['vega']) / (2 * eps)

    print(f"  Volga (AAD Hessian):      {volga_aad:.8f}")
    print(f"  Volga (FD of AAD Vega):   {volga_fd:.8f}")
    print(f"  Difference:               {abs(volga_aad - volga_fd) / abs(volga_fd) * 100:.2f}%")
    print()

    if abs(volga_aad - volga_fd) / abs(volga_fd) * 100 < 1.0:
        print("✓ AAD Hessian is CONSISTENT with finite difference")
        print("  → AAD is computing correctly, but the PDE Vega curve is wrong")
    else:
        print("✗ AAD Hessian INCONSISTENT with finite difference")
        print("  → There might be a bug in Edge-Pushing algorithm")

    print()


def test_grid_resolution():
    """Test if increasing grid resolution improves Volga"""
    print("=" * 80)
    print("TEST 3: Grid Resolution Impact on Volga")
    print("=" * 80)
    print()

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    volga_bs, _, _, _ = analytical_volga(S0, K, T, r, sigma)

    grids = [(21, 20), (51, 50), (101, 100)]

    print(f"{'Grid (M×N)':>15} {'Volga':>15} {'Error%':>10} {'Time(ms)':>12}")
    print("-" * 60)

    import time

    for M, N in grids:
        pricer = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

        t_start = time.perf_counter()
        result = pricer.solve_pde_with_aad(
            S0_val=S0,
            sigma_val=sigma,
            compute_hessian=True,
            fixed_grid=True,
            verbose=False
        )
        t_ms = (time.perf_counter() - t_start) * 1000

        volga_aad = result['volga']
        error = abs(volga_aad - volga_bs) / volga_bs * 100

        print(f"{M:>7}×{N:<7} {volga_aad:>15.8f} {error:>9.2f}% {t_ms:>11.1f}")

    print()
    print(f"Analytical baseline: {volga_bs:.8f}")
    print()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "DEBUG AAD VOLGA COMPUTATION" + " " * 31 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Test 1: Vega curve
    test_vega_curve()

    # Test 2: AAD computation
    test_aad_volga_computation()

    # Test 3: Grid resolution
    test_grid_resolution()

    print("=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print()
    print("If Test 2 shows AAD Hessian = FD of Vega:")
    print("  → AAD algorithm is CORRECT")
    print("  → Problem is in PDE discretization of σ derivatives")
    print()
    print("If Test 2 shows AAD Hessian ≠ FD of Vega:")
    print("  → Bug in Edge-Pushing implementation")
    print()


if __name__ == "__main__":
    main()
