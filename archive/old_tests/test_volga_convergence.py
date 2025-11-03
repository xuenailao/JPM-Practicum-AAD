"""
Test Volga convergence with grid resolution

Test if increasing M and N can reduce Volga error to acceptable levels
"""

import numpy as np
import sys
import time
from math import log, sqrt
from scipy.stats import norm

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def analytical_volga(S0, K, T, r, sigma):
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    n_d1 = norm.pdf(d1)
    vega = S0 * n_d1 * sqrt_T
    volga = vega * d1 * d2 / sigma

    return volga


def test_grid_convergence():
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    volga_bs = analytical_volga(S0, K, T, r, sigma)

    print("=" * 80)
    print("VOLGA CONVERGENCE TEST: Grid Resolution")
    print("=" * 80)
    print()
    print(f"Analytical Volga (baseline): {volga_bs:.8f}")
    print()

    # Test configurations
    configs = [
        (21, 20),
        (51, 50),
        (101, 100),
        (151, 150),
    ]

    print(f"{'Grid (M×N)':>15} {'Volga':>15} {'Error%':>10} {'Time(s)':>10}")
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
        error = abs(volga_aad - volga_bs) / volga_bs * 100

        status = ""
        if error < 5:
            status = "✓"
        elif error < 10:
            status = "○"
        else:
            status = "✗"

        print(f"{M:>7}×{N:<7} {volga_aad:>15.8f} {error:>9.2f}% {t_elapsed:>9.1f}s {status}")

    print()
    print("Legend: ✓ <5%  ○ 5-10%  ✗ >10%")
    print()


def main():
    test_grid_convergence()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("If error <5% at M=151 or M=201:")
    print("  → Solution: Use finer grid for accurate Volga")
    print("  → Trade-off: Longer computation time")
    print()
    print("If error still >10% even at M=201:")
    print("  → PDE discretization fundamental limitation")
    print("  → May need alternative approaches (e.g., transformed coordinates)")
    print()


if __name__ == "__main__":
    main()
