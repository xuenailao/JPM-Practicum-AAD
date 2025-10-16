"""
Test: PDE Edge-Pushing Optimized vs Original
Real comparison with actual timing data
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.true_second_order_ad import TrueSecondOrderAD
from aad_edge_pushing.pde.true_second_order_ad_optimized import TrueSecondOrderADOptimized


def test_pde_original_vs_optimized():
    """Compare original vs optimized PDE implementations"""
    print("="*80)
    print("PDE EDGE-PUSHING: ORIGINAL VS OPTIMIZED")
    print("="*80)

    # Test parameters
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    cp_flag = 'C'

    # Test different grid sizes
    test_cases = [
        (10, 10, 30),
        (20, 20, 30),
        (20, 20, 50),
    ]

    all_results = []

    for M, N, max_params in test_cases:
        print(f"\n{'='*80}")
        print(f"Grid Size: {M}×{N}, Computing {max_params} parameters")
        print(f"{'='*80}\n")

        # Create constant vol grid
        sigma_grid = np.full((M+1, N+1), 0.2)

        # Test 1: Original Implementation
        print("【Original Implementation】")
        solver_orig = TrueSecondOrderAD(M, N)
        solver_orig.set_local_vol_grid(sigma_grid)

        t0 = time.perf_counter()
        H_orig, meta_orig = solver_orig.compute_hessian_analytical(
            S0, K, T, r, cp_flag, focus_region='atm', max_params=max_params
        )
        t_orig = (time.perf_counter() - t0) * 1000

        print(f"Time: {t_orig:.2f} ms")
        print(f"Non-zero entries: {meta_orig['n_entries']:,}")
        print(f"Sparsity: {meta_orig['sparsity_percent']:.1f}%\n")

        # Test 2: Optimized Implementation
        print("【Optimized Implementation (with Caching)】")
        solver_opt = TrueSecondOrderADOptimized(M, N)
        solver_opt.set_local_vol_grid(sigma_grid)

        t0 = time.perf_counter()
        H_opt, meta_opt = solver_opt.compute_hessian_optimized(
            S0, K, T, r, cp_flag, focus_region='atm', max_params=max_params
        )
        t_opt = (time.perf_counter() - t0) * 1000

        print(f"Time: {t_opt:.2f} ms")
        print(f"Non-zero entries: {meta_opt['n_entries']:,}")
        print(f"Sparsity: {meta_opt['sparsity_percent']:.1f}%")
        print(f"Cache efficiency: {meta_opt['cache_efficiency']:.2f}×\n")

        # Verify correctness
        print("【Verification】")
        # Compare a few Hessian entries
        common_keys = list(set(H_orig.keys()) & set(H_opt.keys()))[:10]
        max_error = 0.0
        for key in common_keys:
            error = abs(H_orig[key] - H_opt[key])
            max_error = max(max_error, error)

        print(f"Sample keys checked: {len(common_keys)}")
        print(f"Max difference: {max_error:.2e}\n")

        # Summary
        speedup = t_orig / t_opt
        print("="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        print(f"{'Metric':<30} {'Original':<20} {'Optimized':<20} {'Ratio':<15}")
        print("-"*80)
        print(f"{'Total Time (ms)':<30} {t_orig:<20.2f} {t_opt:<20.2f} {speedup:<15.2f}×")
        print(f"{'Forward+Backward (ms)':<30} {meta_orig['time_forward_ms']+meta_orig['time_backward1_ms']:<20.2f} {meta_opt['time_phase1_ms']:<20.2f}")
        print(f"{'Hessian Computation (ms)':<30} {meta_orig['time_hessian_ms']:<20.2f} {meta_opt['time_phase3_ms']+meta_opt['time_phase4_ms']:<20.2f}")
        print(f"{'Non-zero Entries':<30} {meta_orig['n_entries']:<20,} {meta_opt['n_entries']:<20,}")
        print(f"{'Tangent Solves':<30} {meta_orig['n_tangent_solves']:<20,} {meta_opt['n_tangent_solves']:<20,}")
        print(f"{'Unique Neighbors':<30} {'N/A':<20} {meta_opt['n_unique_neighbors']:<20}")
        cache_eff_str = f"{meta_opt['cache_efficiency']:.2f}×"
        print(f"{'Cache Efficiency':<30} {'N/A':<20} {cache_eff_str:<20}")

        all_results.append({
            'M': M,
            'N': N,
            'max_params': max_params,
            't_orig': t_orig,
            't_opt': t_opt,
            'speedup': speedup,
            'cache_efficiency': meta_opt['cache_efficiency'],
            'n_entries_orig': meta_orig['n_entries'],
            'n_entries_opt': meta_opt['n_entries'],
            'max_error': max_error
        })

    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY: All Test Cases")
    print("="*80)
    print(f"{'Grid':<15} {'Params':<10} {'Original (ms)':<15} {'Optimized (ms)':<18} {'Speedup':<12} {'Cache Eff.':<12}")
    print("-"*80)

    for res in all_results:
        print(f"{res['M']}×{res['N']:<12} {res['max_params']:<10} {res['t_orig']:<15.2f} {res['t_opt']:<18.2f} {res['speedup']:<12.2f}× {res['cache_efficiency']:<12.2f}×")

    return all_results


if __name__ == "__main__":
    results = test_pde_original_vs_optimized()
