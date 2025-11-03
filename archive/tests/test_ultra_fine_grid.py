"""
Test Solution 2: Ultra-fine Grid (both space and time)

Problem: Current grid dS≈2.0 is too coarse for high sigma
Strategy: Use much finer spatial grid AND adaptive time step
"""
import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path
import time as time_module

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import CapriottiCNAAD
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.aad.core.var import ADVar


def compute_optimal_grid(sigma, S0=100.0, target_alpha_dt=0.2):
    """
    Compute optimal M and N based on stability

    Target: alpha * dt < target_alpha_dt

    alpha = (sigma^2 * S0^2 / 2) / dS^2
    dS = 2*S0 / (M-1)

    dt = T / N

    Rearranging:
    (sigma^2 * S0^2 / 2) / ((2*S0/(M-1))^2) * (T/N) < target

    Simplify:
    (sigma^2 * (M-1)^2) / (8) * (T/N) < target

    Choose M first based on price accuracy (dS < S0/50)
    Then compute N to satisfy stability
    """
    T = 1.0

    # Target: dS < S0 / 50 for good price accuracy
    dS_target = S0 / 50.0  # dS < 2.0

    # M from dS target
    M = int(2.0 * S0 / dS_target) + 1

    # Round to nice number
    M = ((M + 49) // 50) * 50 + 1

    # Compute actual dS
    dS = 2.0 * S0 / (M - 1)

    # Compute alpha
    alpha = (sigma**2 * S0**2 / 2) / (dS**2)

    # Compute required N
    N_required = int(np.ceil(T * alpha / target_alpha_dt))
    N = ((N_required + 49) // 50) * 50

    dt = T / N

    return M, N, dS, dt, alpha * dt


def solve_pde_fine_grid(S0, K, T, r, sigma, verbose=False):
    """
    Solve PDE with automatically determined fine grid
    """
    M, N, dS, dt, alpha_dt = compute_optimal_grid(sigma, S0)

    if verbose:
        print(f"\nσ={sigma:.3f}: M={M}, N={N}, dS={dS:.4f}, dt={dt:.6f}, α*dt={alpha_dt:.4f}")

    t_start = time_module.perf_counter()

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

    # Compute gradient
    price_var.adj = 1.0
    for node in reversed(global_tape.nodes):
        for parent, deriv in node.parents:
            if parent.requires_grad:
                parent.adj += node.out.adj * float(deriv)

    vega = sigma_var.adj

    t_elapsed = (time_module.perf_counter() - t_start) * 1000

    return price, vega, M, N, t_elapsed


def black_scholes_greeks(S0, K, T, r, sigma):
    """Analytical solution"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma

    return price, vega, volga


def test_ultra_fine_grid():
    """Test ultra-fine grid method"""

    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    print("\n" + "="*140)
    print("SOLUTION 2: ULTRA-FINE GRID (Space + Time Adaptive)")
    print("="*140)

    print("\nStrategy:")
    print("  1. Space: dS < S0/50 (dS < 2.0) for accurate price")
    print("  2. Time: dt*alpha < 0.2 for stability")

    # Test at critical sigma values
    sigma_values = [0.18, 0.20, 0.22, 0.25]

    print("\n" + "-"*140)
    print("Step 1: Determine optimal grid for each sigma")
    print("-"*140)

    print(f"\n{'Sigma':<10} | {'M':<10} | {'N':<10} | {'dS':<12} | {'dt':<12} | {'alpha*dt':<12} | {'Grid Size':<15}")
    print("-"*140)

    for sigma in sigma_values:
        M, N, dS, dt, alpha_dt = compute_optimal_grid(sigma, S0)
        grid_size = f"{M}×{N}"
        print(f"{sigma:<10.2f} | {M:<10d} | {N:<10d} | {dS:<12.6f} | {dt:<12.6f} | {alpha_dt:<12.6f} | {grid_size:<15}")

    # Test Vega computation
    print("\n" + "-"*140)
    print("Step 2: Test Vega computation with ultra-fine grid")
    print("-"*140)

    print(f"\n{'Sigma':<10} | {'BS Price':<12} | {'PDE Price':<12} | {'Err%':<8} | "
          f"{'BS Vega':<12} | {'PDE Vega':<12} | {'Err%':<8} | {'Grid':<12} | {'Time(s)':<10}")
    print("-"*140)

    results = []
    for sigma in sigma_values:
        # Analytical
        bs_price, bs_vega, bs_volga = black_scholes_greeks(S0, K, T, r, sigma)

        # PDE with ultra-fine grid
        pde_price, pde_vega, M, N, t_ms = solve_pde_fine_grid(S0, K, T, r, sigma)

        price_err = abs(pde_price - bs_price) / bs_price * 100
        vega_err = abs(pde_vega - bs_vega) / bs_vega * 100

        print(f"{sigma:<10.2f} | {bs_price:<12.6f} | {pde_price:<12.6f} | {price_err:<8.2f} | "
              f"{bs_vega:<12.6f} | {pde_vega:<12.6f} | {vega_err:<8.2f} | {M}×{N:<7} | {t_ms/1000:<10.1f}")

        results.append({
            'sigma': sigma,
            'bs_price': bs_price,
            'pde_price': pde_price,
            'bs_vega': bs_vega,
            'pde_vega': pde_vega,
            'M': M,
            'N': N,
            'time_s': t_ms/1000
        })

    # Check trend
    print("\n" + "-"*140)
    print("Step 3: Check Vega trend")
    print("-"*140)

    print("\nPDE Vega trend:")
    all_correct = True
    for i in range(len(results)-1):
        delta_sigma = results[i+1]['sigma'] - results[i]['sigma']
        delta_vega = results[i+1]['pde_vega'] - results[i]['pde_vega']
        trend = "↗" if delta_vega > 0 else "↘"
        correct = delta_vega > 0
        all_correct = all_correct and correct
        status = "✅" if correct else "❌"

        print(f"  σ: {results[i]['sigma']:.2f} → {results[i+1]['sigma']:.2f}  "
              f"Vega: {results[i]['pde_vega']:.2f} → {results[i+1]['pde_vega']:.2f} {trend} {status}  "
              f"ΔVega/Δσ = {delta_vega/delta_sigma:+.2f}")

    # Test one Volga
    print("\n" + "-"*140)
    print("Step 4: Test Volga at σ=0.20")
    print("-"*140)

    sigma_test = 0.20
    eps_sigma = 0.002

    print(f"\nComputing Volga at σ={sigma_test} with fine grid")

    _, vega_minus, _, _, _ = solve_pde_fine_grid(S0, K, T, r, sigma_test - eps_sigma)
    _, vega_center, _, _, _ = solve_pde_fine_grid(S0, K, T, r, sigma_test)
    _, vega_plus, _, _, _ = solve_pde_fine_grid(S0, K, T, r, sigma_test + eps_sigma)

    volga_pde = (vega_plus - vega_minus) / (2 * eps_sigma)

    _, _, volga_bs = black_scholes_greeks(S0, K, T, r, sigma_test)

    print(f"\nVega values:")
    print(f"  σ={sigma_test-eps_sigma:.3f}: {vega_minus:.6f}")
    print(f"  σ={sigma_test:.3f}:        {vega_center:.6f}")
    print(f"  σ={sigma_test+eps_sigma:.3f}: {vega_plus:.6f}")

    vega_direction = "↗ increasing" if vega_plus > vega_minus else "↘ decreasing"
    print(f"  Trend: {vega_direction}")

    print(f"\nVolga:")
    print(f"  PDE:        {volga_pde:+.6f}")
    print(f"  Analytical: {volga_bs:+.6f}")
    print(f"  Error:      {abs(volga_pde - volga_bs)/abs(volga_bs)*100:.2f}%")
    print(f"  Sign:       {'✅ Correct' if volga_pde * volga_bs > 0 else '❌ Wrong'}")

    # Summary
    print("\n" + "="*140)
    print("RESULTS SUMMARY")
    print("="*140)

    avg_vega_err = np.mean([abs(r['pde_vega'] - r['bs_vega'])/r['bs_vega']*100 for r in results])
    avg_price_err = np.mean([abs(r['pde_price'] - r['bs_price'])/r['bs_price']*100 for r in results])
    avg_time = np.mean([r['time_s'] for r in results])

    print(f"\n📊 Accuracy:")
    print(f"   - Average Price error: {avg_price_err:.2f}%")
    print(f"   - Average Vega error:  {avg_vega_err:.2f}%")
    print(f"   - Vega trend:          {'✅ All correct' if all_correct else '❌ Still wrong'}")
    print(f"   - Volga sign:          {'✅ Correct' if volga_pde * volga_bs > 0 else '❌ Wrong'}")

    print(f"\n⏱️ Computational cost:")
    print(f"   - Average time: {avg_time:.1f}s per Vega")
    print(f"   - Grid size: M≈{np.mean([r['M'] for r in results]):.0f}, N≈{np.mean([r['N'] for r in results]):.0f}")

    if not all_correct:
        print("\n⚠️ ANALYSIS:")
        print("   Ultra-fine grid still fails → Problem is NOT grid resolution")
        print("   → Must be fundamental issue with CN scheme itself")


if __name__ == "__main__":
    test_ultra_fine_grid()
