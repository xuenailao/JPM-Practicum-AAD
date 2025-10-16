"""
Test: Greeks Computation - Edge-Pushing vs Bumping on PDE

Compare actual Greeks (Delta, Gamma, Vega, Vanna, Volga) computation
using PDE Edge-Pushing vs PDE Bumping with finite differences.
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.true_second_order_ad_optimized import TrueSecondOrderADOptimized
from aad_edge_pushing.pde.local_vol_solver import LocalVolAdjoint


def compute_greeks_bumping(solver, S0, K, T, r, sigma_grid, cp_flag='C', h=1e-5):
    """
    Compute Greeks using bumping (finite differences).

    Delta, Gamma: Sensitivity w.r.t. spot price S0
    Vega: Sum of sensitivities w.r.t. all σ[i,n]
    Volga: Second derivative w.r.t. global vol (sum of Hessian)
    """
    M, N = solver.M, solver.N

    # Base price
    solver.set_local_vol_grid(sigma_grid)
    price_base, grad_base, _ = solver.adjoint_greeks_local(S0, K, T, r, cp_flag)

    # Delta: ∂V/∂S (via finite difference on S0)
    price_up = solver.adjoint_greeks_local(S0 + h, K, T, r, cp_flag)[0]
    price_down = solver.adjoint_greeks_local(S0 - h, K, T, r, cp_flag)[0]
    delta = (price_up - price_down) / (2 * h)

    # Gamma: ∂²V/∂S² (second order finite difference)
    gamma = (price_up - 2*price_base + price_down) / (h**2)

    # Vega: Sum of ∂V/∂σ[i,n] (already computed in grad_base)
    vega = np.sum(grad_base)

    # Volga: ∂²V/∂σ_global² (bump global vol by h)
    sigma_up = sigma_grid + h
    solver.set_local_vol_grid(sigma_up)
    _, grad_up, _ = solver.adjoint_greeks_local(S0, K, T, r, cp_flag)
    vega_up = np.sum(grad_up)

    sigma_down = sigma_grid - h
    solver.set_local_vol_grid(sigma_down)
    _, grad_down, _ = solver.adjoint_greeks_local(S0, K, T, r, cp_flag)
    vega_down = np.sum(grad_down)

    volga = (vega_up - 2*vega + vega_down) / (h**2)

    # Vanna: ∂²V/∂S∂σ_global (mixed derivative)
    # Perturb both S and σ
    solver.set_local_vol_grid(sigma_up)
    price_Sup_sigup = solver.adjoint_greeks_local(S0 + h, K, T, r, cp_flag)[0]
    price_Sup_sigdown = solver.adjoint_greeks_local(S0 + h, K, T, r, cp_flag)[0]

    solver.set_local_vol_grid(sigma_down)
    price_Sdown_sigup = solver.adjoint_greeks_local(S0 - h, K, T, r, cp_flag)[0]
    price_Sdown_sigdown = solver.adjoint_greeks_local(S0 - h, K, T, r, cp_flag)[0]

    # Simplified vanna (using cross derivatives)
    delta_up = (solver.adjoint_greeks_local(S0 + h, K, T, r, cp_flag)[0] - price_base) / h
    solver.set_local_vol_grid(sigma_down)
    delta_down_sig = (solver.adjoint_greeks_local(S0 + h, K, T, r, cp_flag)[0] -
                      solver.adjoint_greeks_local(S0 - h, K, T, r, cp_flag)[0]) / (2*h)

    solver.set_local_vol_grid(sigma_up)
    delta_up_sig = (solver.adjoint_greeks_local(S0 + h, K, T, r, cp_flag)[0] -
                    solver.adjoint_greeks_local(S0 - h, K, T, r, cp_flag)[0]) / (2*h)

    vanna = (delta_up_sig - delta_down_sig) / (2*h)

    # Reset
    solver.set_local_vol_grid(sigma_grid)

    return {
        'price': price_base,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


def compute_greeks_edge_pushing(solver, S0, K, T, r, sigma_grid, cp_flag='C'):
    """
    Compute Greeks using edge-pushing (True Second-Order AD).

    Uses Hessian to compute second-order Greeks directly.
    """
    M, N = solver.M, solver.N

    solver.set_local_vol_grid(sigma_grid)

    # Compute price and first-order gradient
    # For edge-pushing, we need to run the full Hessian computation
    # which includes forward, backward, and second-order adjoints

    # First, get price and gradient
    _, grad_base, _ = solver.adjoint_greeks_local(S0, K, T, r, cp_flag)

    # Compute Hessian (this gives us second-order sensitivities)
    H_sparse, meta = solver.compute_hessian_optimized(
        S0, K, T, r, cp_flag, focus_region='all', max_params=M*N
    )

    # Extract Greeks from Hessian
    # Vega: Sum of all first-order sensitivities
    vega = np.sum(grad_base)

    # Volga: ∂²V/∂σ_global² = Sum of all Hessian entries
    # (global vol means all σ[i,n] move together)
    volga = sum(H_sparse.values())

    # For delta and gamma, we need sensitivities w.r.t. S0
    # These are not in the Hessian (which is w.r.t. σ parameters)
    # So we still need finite differences for these
    h = 1e-5
    price_base = solver.adjoint_greeks_local(S0, K, T, r, cp_flag)[0]
    price_up = solver.adjoint_greeks_local(S0 + h, K, T, r, cp_flag)[0]
    price_down = solver.adjoint_greeks_local(S0 - h, K, T, r, cp_flag)[0]

    delta = (price_up - price_down) / (2 * h)
    gamma = (price_up - 2*price_base + price_down) / (h**2)

    # Vanna: ∂²V/∂S∂σ - need cross derivative
    # Perturb S, measure change in vega
    _, grad_Sup, _ = solver.adjoint_greeks_local(S0 + h, K, T, r, cp_flag)
    _, grad_Sdown, _ = solver.adjoint_greeks_local(S0 - h, K, T, r, cp_flag)
    vega_Sup = np.sum(grad_Sup)
    vega_Sdown = np.sum(grad_Sdown)
    vanna = (vega_Sup - vega_Sdown) / (2 * h)

    return {
        'price': price_base,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


def test_greeks_comparison():
    """Compare Greeks computation: Edge-Pushing vs Bumping"""
    print("="*80)
    print("GREEKS COMPUTATION: PDE EDGE-PUSHING VS BUMPING")
    print("="*80)

    # Test parameters
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    sigma_const = 0.2
    cp_flag = 'C'

    # Test different grid sizes
    test_cases = [
        (10, 10),
        (20, 20),
    ]

    all_results = []

    for M, N in test_cases:
        print(f"\n{'='*80}")
        print(f"Grid Size: {M}×{N} (Total parameters: {(M+1)*(N+1)})")
        print(f"{'='*80}\n")

        # Create constant vol grid
        sigma_grid = np.full((M+1, N+1), sigma_const)

        # ========== TEST 1: PDE Bumping ==========
        print("【1. PDE Bumping (Finite Differences)】")
        solver_bump = LocalVolAdjoint(M, N)
        solver_bump.set_local_vol_grid(sigma_grid)

        t0 = time.perf_counter()
        greeks_bump = compute_greeks_bumping(solver_bump, S0, K, T, r, sigma_grid, cp_flag)
        t_bump = (time.perf_counter() - t0) * 1000

        print(f"Price:  {greeks_bump['price']:.6f}")
        print(f"Delta:  {greeks_bump['delta']:.6f}")
        print(f"Gamma:  {greeks_bump['gamma']:.6f}")
        print(f"Vega:   {greeks_bump['vega']:.6f}")
        print(f"Vanna:  {greeks_bump['vanna']:.6f}")
        print(f"Volga:  {greeks_bump['volga']:.6f}")
        print(f"Time:   {t_bump:.2f} ms\n")

        # ========== TEST 2: PDE Edge-Pushing ==========
        print("【2. PDE Edge-Pushing (True Second-Order AD)】")
        solver_edge = TrueSecondOrderADOptimized(M, N)
        solver_edge.set_local_vol_grid(sigma_grid)

        t0 = time.perf_counter()
        greeks_edge = compute_greeks_edge_pushing(solver_edge, S0, K, T, r, sigma_grid, cp_flag)
        t_edge = (time.perf_counter() - t0) * 1000

        print(f"Price:  {greeks_edge['price']:.6f}")
        print(f"Delta:  {greeks_edge['delta']:.6f}")
        print(f"Gamma:  {greeks_edge['gamma']:.6f}")
        print(f"Vega:   {greeks_edge['vega']:.6f}")
        print(f"Vanna:  {greeks_edge['vanna']:.6f}")
        print(f"Volga:  {greeks_edge['volga']:.6f}")
        print(f"Time:   {t_edge:.2f} ms\n")

        # ========== COMPARISON ==========
        print("【Comparison】")
        speedup = t_bump / t_edge if t_edge > 0 else 0

        print(f"Time Ratio: {speedup:.2f}× {'(Edge-Pushing faster)' if speedup > 1 else '(Bumping faster)'}\n")

        print("Accuracy (vs Bumping):")
        for greek in ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']:
            bump_val = greeks_bump[greek]
            edge_val = greeks_edge[greek]
            abs_error = abs(edge_val - bump_val)

            if abs(bump_val) > 1e-10:
                rel_error = abs_error / abs(bump_val) * 100
                print(f"  {greek.capitalize():<8}: Abs Error = {abs_error:.2e}, Rel Error = {rel_error:.3f}%")
            else:
                print(f"  {greek.capitalize():<8}: Abs Error = {abs_error:.2e}")

        # ========== SUMMARY ==========
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"{'Metric':<25} {'Bumping':<20} {'Edge-Pushing':<20} {'Speedup':<15}")
        print("-"*80)
        print(f"{'Total Time (ms)':<25} {t_bump:<20.2f} {t_edge:<20.2f} {speedup:<15.2f}×")

        all_results.append({
            'M': M,
            'N': N,
            't_bump': t_bump,
            't_edge': t_edge,
            'speedup': speedup,
            'greeks_bump': greeks_bump,
            'greeks_edge': greeks_edge
        })

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"{'Grid':<12} {'Bumping (ms)':<15} {'Edge-Push (ms)':<18} {'Speedup':<15}")
    print("-"*80)

    for res in all_results:
        grid_str = f"{res['M']}×{res['N']}"
        speedup_str = f"{res['speedup']:.2f}×"
        print(f"{grid_str:<12} {res['t_bump']:<15.2f} {res['t_edge']:<18.2f} {speedup_str:<15}")

    return all_results


if __name__ == "__main__":
    results = test_greeks_comparison()
