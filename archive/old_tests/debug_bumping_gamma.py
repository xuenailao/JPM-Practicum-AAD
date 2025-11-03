"""
Debug Bumping Gamma Computation

Goal: Understand why Bumping returns Gamma = 0
- Check what _solve_pde_numerical returns
- Test finite difference computation
- Compare with grid-based Gamma
"""

import numpy as np
import sys
from math import log, sqrt, exp
from scipy.stats import norm

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def analytical_gamma(S0, K, T, r, sigma):
    """Compute analytical Gamma"""
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    n_d1 = norm.pdf(d1)
    gamma = n_d1 / (S0 * sigma * sqrt_T)
    return gamma


def test_pde_numerical_output():
    """Test what _solve_pde_numerical actually returns"""
    print("=" * 80)
    print("TEST 1: What does _solve_pde_numerical return?")
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

    # Test at different S0 values
    S0_values = [98.0, 99.0, 100.0, 101.0, 102.0]

    print(f"{'S0':>8} {'V(S0)':>15} {'ΔV':>15} {'Δ²V':>15}")
    print("-" * 60)

    prices = []
    for S0_test in S0_values:
        V, V_grid = pricer._solve_pde_numerical(S0_test, sigma, fixed_grid=True)
        prices.append(V)

        print(f"{S0_test:>8.2f} {V:>15.8f}", end="")

        if len(prices) >= 2:
            dV = prices[-1] - prices[-2]
            print(f" {dV:>15.8f}", end="")

            if len(prices) >= 3:
                d2V = prices[-1] - 2*prices[-2] + prices[-3]
                print(f" {d2V:>15.8f}", end="")

        print()

    print()

    # Compute Gamma via finite difference
    h = 1.0  # S0_values spacing
    gamma_fd = (prices[2] - 2*prices[2] + prices[2]) / (h**2)  # This will be 0!

    print(f"Finite difference (WRONG way - using same S0):")
    print(f"  Gamma = [V(100) - 2*V(100) + V(100)] / 1²")
    print(f"        = {gamma_fd:.10f}")
    print()

    # Correct way
    gamma_fd_correct = (prices[3] - 2*prices[2] + prices[1]) / (h**2)

    print(f"Finite difference (CORRECT way - using S0±h):")
    print(f"  Gamma = [V(101) - 2*V(100) + V(99)] / 1²")
    print(f"        = {gamma_fd_correct:.10f}")
    print()

    gamma_anal = analytical_gamma(S0, K, T, r, sigma)
    print(f"Analytical Gamma: {gamma_anal:.10f}")
    print(f"FD Error: {abs(gamma_fd_correct - gamma_anal) / gamma_anal * 100:.2f}%")
    print()


def test_bumping_implementation():
    """Test the actual bumping implementation from test script"""
    print("=" * 80)
    print("TEST 2: Trace Bumping Implementation")
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

    epsilon_S = 0.01 * S0  # 1.0

    print(f"Computing Gamma via bumping:")
    print(f"  epsilon_S = {epsilon_S}")
    print()

    # Exactly as in test script
    V_base, _ = pricer._solve_pde_numerical(S0, sigma, fixed_grid=True)
    V_S_up, _ = pricer._solve_pde_numerical(S0 + epsilon_S, sigma, fixed_grid=True)
    V_S_down, _ = pricer._solve_pde_numerical(S0 - epsilon_S, sigma, fixed_grid=True)

    print(f"  V(S0={S0-epsilon_S:.2f})   = {V_S_down:.10f}")
    print(f"  V(S0={S0:.2f})        = {V_base:.10f}")
    print(f"  V(S0={S0+epsilon_S:.2f})   = {V_S_up:.10f}")
    print()

    gamma_bumping = (V_S_up - 2 * V_base + V_S_down) / (epsilon_S ** 2)

    print(f"  Gamma = [V(101) - 2*V(100) + V(99)] / 1²")
    print(f"        = [{V_S_up:.8f} - 2*{V_base:.8f} + {V_S_down:.8f}] / 1")
    print(f"        = {gamma_bumping:.10f}")
    print()

    gamma_anal = analytical_gamma(S0, K, T, r, sigma)
    print(f"  Analytical Gamma: {gamma_anal:.10f}")

    if abs(gamma_bumping) < 1e-10:
        print(f"\n  ✗ PROBLEM: Gamma ≈ 0!")
        print(f"    This means V(99), V(100), V(101) are TOO SIMILAR")
    else:
        error = abs(gamma_bumping - gamma_anal) / gamma_anal * 100
        print(f"\n  Error: {error:.2f}%")

    print()


def test_price_sensitivity():
    """Test how sensitive the price is to S0 changes"""
    print("=" * 80)
    print("TEST 3: Price Sensitivity to S0")
    print("=" * 80)
    print()

    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20
    M = 51
    N = 50

    # Test very small S0 changes
    S0 = 100.0
    eps_values = [0.001, 0.01, 0.1, 1.0, 2.0]

    print(f"{'ε':>10} {'V(S0+ε)-V(S0)':>20} {'[V(S0+ε)-V(S0)]/ε':>20} {'Delta(approx)':>20}")
    print("-" * 80)

    for eps in eps_values:
        pricer = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

        V_base, _ = pricer._solve_pde_numerical(S0, sigma, fixed_grid=True)
        V_up, _ = pricer._solve_pde_numerical(S0 + eps, sigma, fixed_grid=True)

        dV = V_up - V_base
        delta_approx = dV / eps

        print(f"{eps:>10.3f} {dV:>20.10f} {delta_approx:>20.10f} {delta_approx:>20.10f}")

    # Analytical Delta
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    delta_anal = norm.cdf(d1)

    print()
    print(f"Analytical Delta: {delta_anal:.10f}")
    print()


def check_grid_setup():
    """Check if the spatial grid setup affects interpolation"""
    print("=" * 80)
    print("TEST 4: Grid Setup and Interpolation")
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

    print(f"Spatial grid setup:")
    print(f"  M = {pricer.M}")
    print(f"  S_max = {pricer.S_max:.2f}")
    print(f"  dS = {pricer.dS:.4f}")
    print()

    print(f"Grid points near S0={S0}:")
    S_grid = pricer.S_grid
    idx = np.where((S_grid >= S0 - 5) & (S_grid <= S0 + 5))[0]

    for i in idx:
        marker = " ← S0" if abs(S_grid[i] - S0) < 0.01 else ""
        print(f"  S[{i:2d}] = {S_grid[i]:>8.4f}{marker}")

    print()

    # Solve PDE and check values on grid
    print("Solving PDE and checking grid values...")
    V_at_S0, V_grid = pricer._solve_pde_numerical(S0, sigma, fixed_grid=True)

    print(f"\nInterpolated value at S0={S0}: {V_at_S0:.10f}")
    print()

    print("V values on grid near S0:")
    for i in idx[:10]:  # Limit output
        if i > 0 and i < len(V_grid):
            print(f"  V[{i:2d}] at S={S_grid[i]:>8.4f}: {V_grid[i]:.10f}")

    print()

    # Check if S0 is exactly on grid
    if S0 in S_grid:
        print(f"✓ S0={S0} is EXACTLY on the grid")
    else:
        closest_idx = np.argmin(np.abs(S_grid - S0))
        closest_S = S_grid[closest_idx]
        print(f"✗ S0={S0} is NOT on grid. Closest: S[{closest_idx}]={closest_S:.4f}")
        print(f"  Distance: {abs(S0 - closest_S):.6f}")

    print()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 19 + "DEBUG BUMPING GAMMA COMPUTATION" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    # Test 1: What does _solve_pde_numerical return?
    test_pde_numerical_output()

    # Test 2: Trace bumping implementation
    test_bumping_implementation()

    # Test 3: Price sensitivity
    test_price_sensitivity()

    # Test 4: Grid setup
    check_grid_setup()

    print("=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)
    print()
    print("Possible causes for Gamma = 0:")
    print("  1. V(S0-ε), V(S0), V(S0+ε) are too similar (numerical precision)")
    print("  2. Interpolation error when S0 changes")
    print("  3. Grid doesn't move with S0, only interpolation point changes")
    print("  4. ε is too small or too large")
    print()


if __name__ == "__main__":
    main()
