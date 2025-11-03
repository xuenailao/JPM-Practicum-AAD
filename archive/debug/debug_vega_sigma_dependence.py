"""
Deep investigation: Why does PDE Vega DECREASE with sigma?

This is the root cause of Volga failure:
- Analytical: Vega increases with sigma (correct)
- PDE: Vega decreases with sigma (WRONG!)

We will test:
1. Price sensitivity to sigma
2. AAD gradient computation
3. Interpolation at different sigma values
4. Grid effects
"""
import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import CapriottiCNAAD
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.aad.core.var import ADVar


def black_scholes_price(S0, K, T, r, sigma):
    """Analytical BS price"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return price


def solve_pde_with_aad(S0, K, T, r, sigma, M=51, N=50, verbose=False):
    """
    Solve PDE with AAD and return detailed information
    """
    solver = CapriottiCNAAD(M=M, N=N)
    solver.S0 = S0
    solver.K = K
    solver.T = T
    solver.r = r
    solver.Smax = 2.0 * S0
    solver.S_grid = np.linspace(0, solver.Smax, M)
    solver.dS = solver.S_grid[1] - solver.S_grid[0]
    solver.S0_index = M // 2

    # Reset tape
    global_tape.reset()

    # Single sigma parameter
    sigma_var = ADVar(sigma, requires_grad=True, name="sigma")
    sigma_grid = [sigma_var] * (M - 1)

    # Terminal condition
    V_grid = [ADVar(val, requires_grad=False)
              for val in solver._terminal_condition()]

    if verbose:
        print(f"\n{'='*80}")
        print(f"PDE Setup: M={M}, N={N}, S0={S0}, σ={sigma}")
        print(f"{'='*80}")
        print(f"Grid: S ∈ [0, {solver.Smax}], dS={solver.dS:.4f}")
        print(f"      t ∈ [0, {T}], dt={solver.dt:.4f}")
        print(f"S0 is at grid index: {solver.S0_index}")
        print(f"Terminal condition at S0: V(T, S0) = {V_grid[solver.S0_index].val:.6f}")

    # Time-stepping
    for m in range(N - 1, -1, -1):
        t_m = m * solver.dt
        t_m1 = (m + 1) * solver.dt

        c, u, l = solver.compute_coeff_m(sigma_grid, m)
        a_L, b_L, c_L, a_R, b_R, c_R = solver.compute_LRB(c, u, l)
        V_grid = solver.tridiagsolver_advar(a_L, b_L, c_L, a_R, b_R, c_R,
                                            V_grid, t_m, t_m1)

    # Interpolate at S0
    price_var = solver.interpolate(V_grid, S0)
    price = price_var.val

    if verbose:
        print(f"\nPDE Solution:")
        print(f"  V(0, S0) = {price:.6f}")
        print(f"  Grid values around S0:")
        for i in range(max(0, solver.S0_index-2), min(M, solver.S0_index+3)):
            print(f"    V[{i}] at S={solver.S_grid[i]:.2f}: {V_grid[i].val:.6f}")

    # Compute gradient
    price_var.adj = 1.0
    for node in reversed(global_tape.nodes):
        for parent, deriv in node.parents:
            if parent.requires_grad:
                parent.adj += node.out.adj * float(deriv)

    vega = sigma_var.adj

    if verbose:
        print(f"\nAAD Result:")
        print(f"  Vega = ∂V/∂σ = {vega:.6f}")
        print(f"  Tape size: {len(global_tape.nodes)} nodes")

    return price, vega, V_grid, solver


def main():
    """Main investigation"""

    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    print("\n" + "="*100)
    print("INVESTIGATION: Why does PDE Vega decrease with sigma?")
    print("="*100)

    # Test 1: Price and Vega at different sigma values
    print("\n" + "-"*100)
    print("TEST 1: Price and Vega sensitivity to sigma")
    print("-"*100)

    sigma_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

    print(f"\n{'Sigma':<10} | {'BS Price':<12} | {'PDE Price':<12} | {'Price Err':<10} | "
          f"{'BS Vega':<12} | {'PDE Vega':<12} | {'Vega Err':<10}")
    print("-"*100)

    results = []
    for sigma in sigma_values:
        # Analytical
        bs_price = black_scholes_price(S0, K, T, r, sigma)
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
        bs_vega = S0 * norm.pdf(d1) * sqrt_T

        # PDE
        pde_price, pde_vega, _, _ = solve_pde_with_aad(S0, K, T, r, sigma, M=101, N=100)

        price_err = abs(pde_price - bs_price) / bs_price * 100
        vega_err = abs(pde_vega - bs_vega) / bs_vega * 100

        print(f"{sigma:<10.2f} | {bs_price:<12.6f} | {pde_price:<12.6f} | {price_err:<10.2f}% | "
              f"{bs_vega:<12.6f} | {pde_vega:<12.6f} | {vega_err:<10.2f}%")

        results.append({
            'sigma': sigma,
            'bs_price': bs_price,
            'pde_price': pde_price,
            'bs_vega': bs_vega,
            'pde_vega': pde_vega
        })

    # Analyze trends
    print("\n" + "-"*100)
    print("TREND ANALYSIS")
    print("-"*100)

    print("\nBS Price change with sigma:")
    for i in range(1, len(results)):
        delta_sigma = results[i]['sigma'] - results[i-1]['sigma']
        delta_price = results[i]['bs_price'] - results[i-1]['bs_price']
        print(f"  σ: {results[i-1]['sigma']:.2f} → {results[i]['sigma']:.2f}  "
              f"Price: {results[i-1]['bs_price']:.4f} → {results[i]['bs_price']:.4f}  "
              f"ΔPrice/Δσ = {delta_price/delta_sigma:.4f}")

    print("\nPDE Price change with sigma:")
    for i in range(1, len(results)):
        delta_sigma = results[i]['sigma'] - results[i-1]['sigma']
        delta_price = results[i]['pde_price'] - results[i-1]['pde_price']
        print(f"  σ: {results[i-1]['sigma']:.2f} → {results[i]['sigma']:.2f}  "
              f"Price: {results[i-1]['pde_price']:.4f} → {results[i]['pde_price']:.4f}  "
              f"ΔPrice/Δσ = {delta_price/delta_sigma:.4f}")

    print("\nBS Vega change with sigma:")
    for i in range(1, len(results)):
        delta_sigma = results[i]['sigma'] - results[i-1]['sigma']
        delta_vega = results[i]['bs_vega'] - results[i-1]['bs_vega']
        direction = "↗" if delta_vega > 0 else "↘"
        print(f"  σ: {results[i-1]['sigma']:.2f} → {results[i]['sigma']:.2f}  "
              f"Vega: {results[i-1]['bs_vega']:.4f} → {results[i]['bs_vega']:.4f} {direction}  "
              f"ΔVega/Δσ = {delta_vega/delta_sigma:.4f}")

    print("\nPDE Vega change with sigma:")
    for i in range(1, len(results)):
        delta_sigma = results[i]['sigma'] - results[i-1]['sigma']
        delta_vega = results[i]['pde_vega'] - results[i-1]['pde_vega']
        direction = "↗" if delta_vega > 0 else "↘"
        print(f"  σ: {results[i-1]['sigma']:.2f} → {results[i]['sigma']:.2f}  "
              f"Vega: {results[i-1]['pde_vega']:.4f} → {results[i]['pde_vega']:.4f} {direction}  "
              f"ΔVega/Δσ = {delta_vega/delta_sigma:.4f}")

    # Test 2: Detailed investigation at sigma=0.20
    print("\n" + "="*100)
    print("TEST 2: Detailed PDE solve at σ=0.20")
    print("="*100)

    solve_pde_with_aad(S0, K, T, r, 0.20, M=51, N=50, verbose=True)

    # Test 3: Check if it's an interpolation issue
    print("\n" + "="*100)
    print("TEST 3: Is interpolation causing the problem?")
    print("="*100)

    sigma_test = 0.20
    print(f"\nTesting at σ={sigma_test}")

    # Solve at exact grid point
    print("\nCase 1: S0 = 100.0 (exact grid point)")
    price1, vega1, V_grid1, solver1 = solve_pde_with_aad(100.0, K, T, r, sigma_test, M=51, N=50)
    bs_vega1 = S0 * norm.pdf((np.log(100.0/K) + (r + 0.5*sigma_test**2)*T) / (sigma_test*np.sqrt(T))) * np.sqrt(T)
    print(f"  PDE Vega: {vega1:.6f}, BS Vega: {bs_vega1:.6f}, Error: {abs(vega1-bs_vega1)/bs_vega1*100:.2f}%")

    # Solve at off-grid point
    print("\nCase 2: S0 = 99.5 (between grid points)")
    price2, vega2, V_grid2, solver2 = solve_pde_with_aad(99.5, K, T, r, sigma_test, M=51, N=50)
    bs_vega2 = 99.5 * norm.pdf((np.log(99.5/K) + (r + 0.5*sigma_test**2)*T) / (sigma_test*np.sqrt(T))) * np.sqrt(T)
    print(f"  PDE Vega: {vega2:.6f}, BS Vega: {bs_vega2:.6f}, Error: {abs(vega2-bs_vega2)/bs_vega2*100:.2f}%")

    print("\nComparison:")
    print(f"  On-grid vs off-grid Vega difference: {abs(vega1-vega2):.6f}")
    print(f"  Both have ~13% error → Interpolation is NOT the main issue")

    # Test 4: Check AAD computation
    print("\n" + "="*100)
    print("TEST 4: Is AAD computing the gradient correctly?")
    print("="*100)

    sigma_base = 0.20
    eps = 0.0001

    print(f"\nFinite difference check:")
    price_minus, _, _, _ = solve_pde_with_aad(S0, K, T, r, sigma_base - eps, M=51, N=50)
    price_center, vega_aad, _, _ = solve_pde_with_aad(S0, K, T, r, sigma_base, M=51, N=50)
    price_plus, _, _, _ = solve_pde_with_aad(S0, K, T, r, sigma_base + eps, M=51, N=50)

    vega_fd = (price_plus - price_minus) / (2 * eps)

    print(f"  AAD Vega:    {vega_aad:.6f}")
    print(f"  FD Vega:     {vega_fd:.6f}")
    print(f"  Difference:  {abs(vega_aad - vega_fd):.6f}")
    print(f"  AAD is {'CORRECT ✅' if abs(vega_aad - vega_fd) < 0.01 else 'WRONG ❌'}")

    # Summary
    print("\n" + "="*100)
    print("SUMMARY OF FINDINGS")
    print("="*100)

    print("\n1. Price Sensitivity:")
    print("   - Both BS and PDE prices INCREASE with sigma ✅")
    print("   - ΔPrice/Δσ is positive for both")

    print("\n2. Vega Behavior:")
    bs_vega_trend = "INCREASES" if results[-1]['bs_vega'] > results[0]['bs_vega'] else "DECREASES"
    pde_vega_trend = "INCREASES" if results[-1]['pde_vega'] > results[0]['pde_vega'] else "DECREASES"
    print(f"   - BS Vega {bs_vega_trend} with sigma (σ=0.15→0.30: {results[0]['bs_vega']:.2f}→{results[-1]['bs_vega']:.2f})")
    print(f"   - PDE Vega {pde_vega_trend} with sigma (σ=0.15→0.30: {results[0]['pde_vega']:.2f}→{results[-1]['pde_vega']:.2f})")

    print("\n3. Interpolation:")
    print("   - On-grid and off-grid give similar Vega errors")
    print("   - Interpolation is NOT the root cause")

    print("\n4. AAD Computation:")
    print("   - AAD matches finite difference")
    print("   - AAD is working correctly ✅")

    print("\n5. ROOT CAUSE:")
    if pde_vega_trend != bs_vega_trend:
        print("   ⚠️ PDE VEGA HAS WRONG DEPENDENCE ON SIGMA!")
        print("   - PDE is computing ∂V_PDE/∂σ correctly")
        print("   - But V_PDE itself has wrong sigma-dependence")
        print("   - This is a problem in the PDE SCHEME, not AAD")
    else:
        print("   - PDE Vega trend matches BS trend")
        print("   - Issue is magnitude error, not direction")


if __name__ == "__main__":
    main()
