"""
Real Test: PDE Edge-Pushing vs PDE Bumping

Compare optimized edge-pushing implementation against finite difference bumping
for computing Hessian of PDE solution w.r.t. local volatility parameters.
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.true_second_order_ad_optimized import TrueSecondOrderADOptimized
from aad_edge_pushing.pde.local_vol_solver import LocalVolAdjoint


def pde_bumping_hessian(solver, S0, K, T, r, sigma_grid, param_list, cp_flag='C', h=1e-5):
    """
    Compute Hessian using finite difference bumping.

    For each parameter σ[i,n]:
    - Perturb σ[i,n] by +h
    - Recompute gradient (via adjoint)
    - H[i,n,:,:] = (grad_perturbed - grad_base) / h
    """
    M, N = solver.M, solver.N

    # Compute base gradient
    _, grad_base, _ = solver.adjoint_greeks_local(S0, K, T, r, cp_flag)

    # Storage for Hessian (sparse)
    sparse_hessian = {}
    n_entries = 0
    n_gradient_evals = 0

    for idx, (i_param, n_param) in enumerate(param_list):
        # Perturb parameter
        sigma_perturbed = sigma_grid.copy()
        sigma_perturbed[i_param, n_param] += h

        # Recompute gradient with perturbed parameter
        solver.set_local_vol_grid(sigma_perturbed)
        _, grad_perturbed, _ = solver.adjoint_greeks_local(S0, K, T, r, cp_flag)
        n_gradient_evals += 1

        # Compute Hessian row via finite difference
        # Only store entries for parameters in param_list (to match edge-pushing)
        for (j_param, m_param) in param_list:
            H_val = (grad_perturbed[j_param, m_param] - grad_base[j_param, m_param]) / h

            if abs(H_val) > 1e-10:
                sparse_hessian[(i_param, n_param, j_param, m_param)] = H_val
                n_entries += 1

    # Reset to base grid
    solver.set_local_vol_grid(sigma_grid)

    return sparse_hessian, n_entries, n_gradient_evals


def test_edge_pushing_vs_bumping():
    """Compare PDE Edge-Pushing (Optimized) vs PDE Bumping"""
    print("="*80)
    print("PDE EDGE-PUSHING VS PDE BUMPING")
    print("="*80)

    # Test parameters
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    cp_flag = 'C'

    # Test different grid sizes
    test_cases = [
        (10, 10, 20),   # Small grid, 20 params
        (20, 20, 30),   # Medium grid, 30 params
        (20, 20, 50),   # Medium grid, 50 params
    ]

    all_results = []

    for M, N, max_params in test_cases:
        print(f"\n{'='*80}")
        print(f"Grid Size: {M}×{N}, Computing {max_params} parameters")
        print(f"{'='*80}\n")

        # Create constant vol grid
        sigma_grid = np.full((M+1, N+1), 0.2)

        # Get parameter list (ATM region)
        Smax = 4.0 * K
        dS = Smax / M
        dt = T / N
        i_atm = int(S0 / dS)
        i_range = max(2, M // 10)
        n_start = N // 4
        n_end = min(3 * N // 4, N - 1)

        param_list = [(i, n) for n in range(n_start, n_end + 1)
                     for i in range(max(1, i_atm - i_range), min(M, i_atm + i_range + 1))]
        param_list = param_list[:max_params]

        print(f"Actual parameters computed: {len(param_list)}\n")

        # ========== TEST 1: PDE Bumping ==========
        print("【1. PDE Bumping (Finite Difference)】")
        solver_bump = LocalVolAdjoint(M, N)
        solver_bump.set_local_vol_grid(sigma_grid)

        t0 = time.perf_counter()
        H_bump, n_entries_bump, n_grad_evals = pde_bumping_hessian(
            solver_bump, S0, K, T, r, sigma_grid, param_list, cp_flag
        )
        t_bump = (time.perf_counter() - t0) * 1000

        print(f"Time: {t_bump:.2f} ms")
        print(f"Gradient evaluations: {n_grad_evals + 1}")  # +1 for base gradient
        print(f"Non-zero entries: {n_entries_bump:,}")
        print(f"Sparsity: {100*(1 - n_entries_bump/(len(param_list)**2)):.1f}%\n")

        # ========== TEST 2: PDE Edge-Pushing (Optimized) ==========
        print("【2. PDE Edge-Pushing (Optimized, with Caching)】")
        solver_edge = TrueSecondOrderADOptimized(M, N)
        solver_edge.set_local_vol_grid(sigma_grid)

        t0 = time.perf_counter()
        H_edge, meta_edge = solver_edge.compute_hessian_optimized(
            S0, K, T, r, cp_flag, focus_region='atm', max_params=max_params
        )
        t_edge = (time.perf_counter() - t0) * 1000

        print(f"Time: {t_edge:.2f} ms")
        print(f"Tangent/Adjoint solves: {meta_edge['n_tangent_solves']}")
        print(f"Non-zero entries: {meta_edge['n_entries']:,}")
        print(f"Sparsity: {meta_edge['sparsity_percent']:.1f}%")
        print(f"Cache efficiency: {meta_edge['cache_efficiency']:.2f}×\n")

        # ========== VERIFICATION ==========
        print("【Verification】")

        # Compare overlapping entries
        common_keys = list(set(H_bump.keys()) & set(H_edge.keys()))[:20]
        if len(common_keys) > 0:
            max_error = 0.0
            max_rel_error = 0.0

            for key in common_keys:
                abs_error = abs(H_bump[key] - H_edge[key])
                max_error = max(max_error, abs_error)

                if abs(H_bump[key]) > 1e-10:
                    rel_error = abs_error / abs(H_bump[key])
                    max_rel_error = max(max_rel_error, rel_error)

            print(f"Common entries checked: {len(common_keys)}")
            print(f"Max absolute error: {max_error:.2e}")
            print(f"Max relative error: {max_rel_error:.2%}\n")
        else:
            print("No common entries to compare (different sparsity patterns)\n")

        # ========== COMPARISON SUMMARY ==========
        speedup = t_bump / t_edge

        print("="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Metric':<35} {'Bumping':<20} {'Edge-Pushing':<20} {'Ratio':<15}")
        print("-"*80)
        print(f"{'Total Time (ms)':<35} {t_bump:<20.2f} {t_edge:<20.2f} {speedup:<15.2f}×")
        print(f"{'Forward+Backward (ms)':<35} {'N/A':<20} {meta_edge['time_phase1_ms']:<20.2f}")
        print(f"{'Gradient Evaluations':<35} {n_grad_evals+1:<20} {'1 (adjoint)':<20}")
        print(f"{'Tangent Solves':<35} {'0':<20} {meta_edge['n_tangent_solves']:<20}")
        print(f"{'Non-zero Entries':<35} {n_entries_bump:<20,} {meta_edge['n_entries']:<20,}")
        print(f"{'Unique Neighbors Computed':<35} {'N/A':<20} {meta_edge['n_unique_neighbors']:<20}")
        cache_str = f"{meta_edge['cache_efficiency']:.2f}×"
        print(f"{'Cache Efficiency':<35} {'N/A':<20} {cache_str:<20}")

        all_results.append({
            'M': M,
            'N': N,
            'n_params': len(param_list),
            't_bump': t_bump,
            't_edge': t_edge,
            'speedup': speedup,
            'n_entries_bump': n_entries_bump,
            'n_entries_edge': meta_edge['n_entries'],
            'cache_eff': meta_edge['cache_efficiency']
        })

    # ========== FINAL SUMMARY ==========
    print("\n" + "="*80)
    print("FINAL SUMMARY: Edge-Pushing vs Bumping")
    print("="*80)
    print(f"{'Grid':<12} {'Params':<10} {'Bumping (ms)':<15} {'Edge-Push (ms)':<18} {'Speedup':<15}")
    print("-"*80)

    for res in all_results:
        grid_str = f"{res['M']}×{res['N']}"
        speedup_str = f"{res['speedup']:.2f}×"
        if res['speedup'] < 1:
            speedup_str += " (slower)"

        print(f"{grid_str:<12} {res['n_params']:<10} {res['t_bump']:<15.2f} {res['t_edge']:<18.2f} {speedup_str:<15}")

    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    for i, res in enumerate(all_results, 1):
        if res['speedup'] >= 1:
            print(f"{i}. {res['M']}×{res['N']} grid: Edge-Pushing is {res['speedup']:.2f}× FASTER than Bumping")
        else:
            print(f"{i}. {res['M']}×{res['N']} grid: Bumping is {1/res['speedup']:.2f}× faster than Edge-Pushing")

    print(f"\nCache efficiency: {all_results[0]['cache_eff']:.2f}× (consistent across tests)")

    return all_results


if __name__ == "__main__":
    results = test_edge_pushing_vs_bumping()
