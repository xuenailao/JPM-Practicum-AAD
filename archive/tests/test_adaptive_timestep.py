"""
Test Solution 1: Adaptive Time Step

Stability condition: dt * alpha < 0.5
where alpha = (sigma^2 * S^2 / 2) / dS^2

Strategy: Use smaller dt when computing Vega at different sigma values
"""
import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import CapriottiCNAAD
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.aad.core.var import ADVar


def compute_required_timesteps(sigma, M, S0=100.0):
    """
    Compute required number of time steps for stability

    Condition: dt * alpha < 0.5
    alpha = (sigma^2 * S0^2 / 2) / dS^2
    dt = T / N

    Therefore: N > T * alpha / 0.5
    """
    Smax = 2.0 * S0
    dS = Smax / (M - 1)

    alpha_max = (sigma**2 * S0**2 / 2) / (dS**2)

    # Target: dt * alpha < 0.3 (safer than 0.5)
    T = 1.0
    N_required = int(np.ceil(T * alpha_max / 0.3))

    # Round up to nearest 50
    N_required = ((N_required + 49) // 50) * 50

    return N_required, alpha_max


def solve_pde_adaptive(S0, K, T, r, sigma, M=101, verbose=False):
    """
    Solve PDE with adaptive time steps based on sigma
    """
    # Compute required N for this sigma
    N, alpha_max = compute_required_timesteps(sigma, M, S0)

    if verbose:
        print(f"\nσ={sigma:.3f}: alpha_max={alpha_max:.2f}, using N={N} (dt={T/N:.6f})")

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

    return price, vega, N


def black_scholes_greeks(S0, K, T, r, sigma):
    """Analytical solution"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma

    return price, vega, volga


def test_adaptive_timestep():
    """Test adaptive time step method"""

    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    print("\n" + "="*120)
    print("SOLUTION 1: ADAPTIVE TIME STEP FOR STABILITY")
    print("="*120)

    print("\nStrategy: Use smaller dt (more time steps N) at high sigma")
    print("Stability condition: dt * alpha < 0.3")

    # Test at different sigma values
    sigma_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]
    M = 101

    print("\n" + "-"*120)
    print("Step 1: Compute required time steps")
    print("-"*120)

    print(f"\n{'Sigma':<10} | {'alpha_max':<12} | {'N_required':<12} | {'dt':<12} | {'dt*alpha':<12} | {'Status':<15}")
    print("-"*120)

    for sigma in sigma_values:
        N, alpha_max = compute_required_timesteps(sigma, M, S0)
        dt = T / N
        dt_alpha = dt * alpha_max
        status = "✅ Stable" if dt_alpha < 0.5 else "⚠️ Marginal"

        print(f"{sigma:<10.2f} | {alpha_max:<12.2f} | {N:<12d} | {dt:<12.6f} | {dt_alpha:<12.6f} | {status:<15}")

    # Test Vega computation
    print("\n" + "-"*120)
    print("Step 2: Test Vega computation with adaptive time steps")
    print("-"*120)

    print(f"\n{'Sigma':<10} | {'BS Price':<12} | {'PDE Price':<12} | {'Price Err':<10} | "
          f"{'BS Vega':<12} | {'PDE Vega':<12} | {'Vega Err':<10} | {'N used':<10}")
    print("-"*120)

    results = []
    for sigma in sigma_values:
        t_start = time.perf_counter()

        # Analytical
        bs_price, bs_vega, bs_volga = black_scholes_greeks(S0, K, T, r, sigma)

        # PDE with adaptive timestep
        pde_price, pde_vega, N_used = solve_pde_adaptive(S0, K, T, r, sigma, M=M)

        t_elapsed = (time.perf_counter() - t_start) * 1000

        price_err = abs(pde_price - bs_price) / bs_price * 100
        vega_err = abs(pde_vega - bs_vega) / bs_vega * 100

        print(f"{sigma:<10.2f} | {bs_price:<12.6f} | {pde_price:<12.6f} | {price_err:<10.2f}% | "
              f"{bs_vega:<12.6f} | {pde_vega:<12.6f} | {vega_err:<10.2f}% | {N_used:<10d}")

        results.append({
            'sigma': sigma,
            'bs_vega': bs_vega,
            'pde_vega': pde_vega,
            'N': N_used,
            'time_ms': t_elapsed
        })

    # Test Vega trend
    print("\n" + "-"*120)
    print("Step 3: Check if Vega trend is correct")
    print("-"*120)

    print("\nBS Vega trend (should increase with sigma):")
    for i in range(len(results)-1):
        delta_sigma = results[i+1]['sigma'] - results[i]['sigma']
        delta_vega = results[i+1]['bs_vega'] - results[i]['bs_vega']
        trend = "↗" if delta_vega > 0 else "↘"
        print(f"  σ: {results[i]['sigma']:.2f} → {results[i+1]['sigma']:.2f}  "
              f"Vega: {results[i]['bs_vega']:.2f} → {results[i+1]['bs_vega']:.2f} {trend}  "
              f"ΔVega/Δσ = {delta_vega/delta_sigma:+.2f}")

    print("\nPDE Vega trend (with adaptive dt):")
    for i in range(len(results)-1):
        delta_sigma = results[i+1]['sigma'] - results[i]['sigma']
        delta_vega = results[i+1]['pde_vega'] - results[i]['pde_vega']
        trend = "↗" if delta_vega > 0 else "↘"
        match = "✅" if (delta_vega > 0) else "❌"
        print(f"  σ: {results[i]['sigma']:.2f} → {results[i+1]['sigma']:.2f}  "
              f"Vega: {results[i]['pde_vega']:.2f} → {results[i+1]['pde_vega']:.2f} {trend} {match}  "
              f"ΔVega/Δσ = {delta_vega/delta_sigma:+.2f}")

    # Test Volga
    print("\n" + "-"*120)
    print("Step 4: Test Volga computation")
    print("-"*120)

    sigma_test = 0.20
    eps_sigma = 0.002

    print(f"\nComputing Volga at σ={sigma_test} using eps_sigma={eps_sigma}")

    # Compute Vega at three points
    _, vega_minus, _ = solve_pde_adaptive(S0, K, T, r, sigma_test - eps_sigma, M=M)
    _, vega_center, _ = solve_pde_adaptive(S0, K, T, r, sigma_test, M=M)
    _, vega_plus, _ = solve_pde_adaptive(S0, K, T, r, sigma_test + eps_sigma, M=M)

    # Volga via finite difference
    volga_pde = (vega_plus - vega_minus) / (2 * eps_sigma)

    # Analytical
    _, _, volga_bs = black_scholes_greeks(S0, K, T, r, sigma_test)

    print(f"\nVega at different sigma:")
    print(f"  σ={sigma_test-eps_sigma:.3f}: Vega={vega_minus:.6f}")
    print(f"  σ={sigma_test:.3f}:        Vega={vega_center:.6f}")
    print(f"  σ={sigma_test+eps_sigma:.3f}: Vega={vega_plus:.6f}")

    print(f"\nVolga = ∂Vega/∂σ:")
    print(f"  PDE:        {volga_pde:.6f}")
    print(f"  Analytical: {volga_bs:.6f}")
    print(f"  Error:      {abs(volga_pde - volga_bs)/abs(volga_bs)*100:.2f}%")
    print(f"  Sign:       {'✅ Correct' if volga_pde * volga_bs > 0 else '❌ Wrong'}")

    # Summary
    print("\n" + "="*120)
    print("RESULTS SUMMARY")
    print("="*120)

    vega_errors = [abs(r['pde_vega'] - results[i]['bs_vega'])/results[i]['bs_vega']*100
                   for i, r in enumerate(results)]
    avg_vega_error = np.mean(vega_errors)
    max_vega_error = np.max(vega_errors)

    print(f"\n✅ Adaptive time step method:")
    print(f"   - Average Vega error: {avg_vega_error:.2f}%")
    print(f"   - Max Vega error: {max_vega_error:.2f}%")
    print(f"   - Vega trend: {'✅ Correct' if all(r['pde_vega'] < results[i+1]['pde_vega'] for i, r in enumerate(results[:-1])) else '❌ Wrong'}")
    print(f"   - Volga sign: {'✅ Correct' if volga_pde * volga_bs > 0 else '❌ Wrong'}")

    print(f"\n⏱️ Computational cost:")
    print(f"   - σ=0.20: N={results[2]['N']}, time≈{results[2]['time_ms']:.0f}ms")
    print(f"   - σ=0.30: N={results[5]['N']}, time≈{results[5]['time_ms']:.0f}ms")
    print(f"   - Slowdown: {results[5]['N']/results[2]['N']:.1f}×")


if __name__ == "__main__":
    test_adaptive_timestep()
