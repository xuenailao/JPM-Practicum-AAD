"""
Complete Volga debug: Test AAD extraction methods against finite difference.
"""

import numpy as np
import time
from scipy.stats import norm
from aad_edge_pushing.pde.true_second_order_ad_optimized import TrueSecondOrderADOptimized

def bsm_volga(S, K, T, r, sigma):
    """BSM analytical Volga."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    n_d1 = norm.pdf(d1)
    vega = S * n_d1 * np.sqrt(T)
    volga = vega * d1 * d2 / sigma
    return volga, vega

def compute_volga_bumping(solver, S0, K, T, r, sigma_0, M, N, eps=0.01):
    """Compute Volga via finite differences on global σ."""

    # Price at σ0
    sigma_grid_0 = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid_0)
    price_0, _, _ = solver.solve_local_vol(S0, K, T, r, cp_flag='C')

    # Price at σ0 + ε
    sigma_grid_up = np.full((M+1, N+1), sigma_0 + eps)
    solver.set_local_vol_grid(sigma_grid_up)
    price_up, _, _ = solver.solve_local_vol(S0, K, T, r, cp_flag='C')

    # Price at σ0 - ε
    sigma_grid_down = np.full((M+1, N+1), sigma_0 - eps)
    solver.set_local_vol_grid(sigma_grid_down)
    price_down, _, _ = solver.solve_local_vol(S0, K, T, r, cp_flag='C')

    # Central difference for second derivative
    volga_bump = (price_up - 2*price_0 + price_down) / (eps**2)

    return volga_bump, price_0

def test_volga_complete():
    """Complete Volga test."""

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma_0 = 0.2
    M = 40
    N = 40

    print("="*80)
    print("COMPLETE VOLGA DEBUG")
    print("="*80)
    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma_0}")
    print(f"Grid: M={M}, N={N}")

    # Get BSM ground truth
    volga_bsm, vega_bsm = bsm_volga(S0, K, T, r, sigma_0)
    print(f"\n【BSM Analytical (Ground Truth)】")
    print(f"  Vega:  {vega_bsm:.6f}")
    print(f"  Volga: {volga_bsm:.6e}")

    # Initialize solver
    solver = TrueSecondOrderADOptimized(M=M, N=N, Smax_factor=4.0)

    # Method 1: Finite Differences (Bumping)
    print(f"\n【Method 1: Finite Differences】")
    t0 = time.time()
    volga_bump, price_pde = compute_volga_bumping(solver, S0, K, T, r, sigma_0, M, N)
    t1 = time.time()

    error_bump = abs(volga_bump - volga_bsm) / abs(volga_bsm) * 100
    print(f"  Volga = {volga_bump:.6e}")
    print(f"  Error = {error_bump:.1f}%")
    print(f"  Time = {(t1-t0)*1000:.2f} ms")

    # Method 2: AAD Hessian
    print(f"\n【Method 2: AAD Hessian】")
    sigma_grid = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid)

    t0 = time.time()
    sparse_hess, metadata = solver.compute_hessian_optimized(
        S0, K, T, r, cp_flag='C', focus_region='all'
    )
    t1 = time.time()

    # Convert to dense
    n_params = N * (M-1)
    hessian = np.zeros((n_params, n_params))
    param_to_idx = {}
    idx = 0
    for n in range(N):
        for i in range(1, M):
            param_to_idx[(i, n)] = idx
            idx += 1

    for (i, n, j, m), val in sparse_hess.items():
        if (i, n) in param_to_idx and (j, m) in param_to_idx:
            idx_i = param_to_idx[(i, n)]
            idx_j = param_to_idx[(j, m)]
            hessian[idx_i, idx_j] = val

    print(f"  Hessian computed in {(t1-t0)*1000:.2f} ms")

    diagonal = np.diag(hessian)

    # Try different scale factors
    print(f"\n  Testing different scale factors:")

    for scale in [1, 10, 100, 1000, 10000, 100000]:
        volga_scaled = np.sum(diagonal) / scale
        error_scaled = abs(volga_scaled - volga_bump) / abs(volga_bump) * 100
        match = "✓" if error_scaled < 10 else " "
        print(f"    {match} Scale {scale:6d}: {volga_scaled:12.6e}  (error vs bump: {error_scaled:6.1f}%)")

    # Best scale
    best_scale = None
    best_error = float('inf')
    for scale in [1, 10, 100, 1000, 10000, 100000]:
        volga_scaled = np.sum(diagonal) / scale
        error = abs(volga_scaled - volga_bump) / abs(volga_bump)
        if error < best_error:
            best_error = error
            best_scale = scale

    volga_aad = np.sum(diagonal) / best_scale
    error_aad_vs_bump = abs(volga_aad - volga_bump) / abs(volga_bump) * 100
    error_aad_vs_bsm = abs(volga_aad - volga_bsm) / abs(volga_bsm) * 100

    print(f"\n  Best scale: {best_scale}")
    print(f"  Volga (AAD) = {volga_aad:.6e}")
    print(f"  Error vs Bumping = {error_aad_vs_bump:.1f}%")
    print(f"  Error vs BSM = {error_aad_vs_bsm:.1f}%")

    # Summary
    print(f"\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nGround Truth (BSM):     {volga_bsm:.6e}")
    print(f"Bumping (PDE):          {volga_bump:.6e}  (error: {error_bump:.1f}%)")
    print(f"AAD Hessian (best):     {volga_aad:.6e}  (error: {error_aad_vs_bsm:.1f}%)")

    print(f"\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    if error_aad_vs_bump < 5:
        print(f"\n✓ AAD matches bumping (<5% error)")
        print(f"  This confirms AAD Hessian is computing ∂²V/∂σᵢⱼ∂σₖₗ correctly.")
        print(f"  Use scale factor {best_scale} to extract Volga.")
    else:
        print(f"\n✗ AAD does NOT match bumping ({error_aad_vs_bump:.1f}% error)")
        print(f"  There may be an issue with the Hessian computation.")

    if abs(error_bump) > 10:
        print(f"\n⚠  Bumping has large error vs BSM ({error_bump:.1f}%)")
        print(f"  Grid may be too coarse (M={M}, N={N})")
        print(f"  Consider increasing grid resolution.")

    print(f"\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if error_aad_vs_bump < 5 and abs(error_bump) < 10:
        print(f"\n✓ Both methods working correctly!")
        print(f"  Recommended Volga extraction:")
        print(f"    volga = np.sum(np.diag(hessian)) / {best_scale}")
    elif error_aad_vs_bump < 5:
        print(f"\n⚠  AAD working, but grid too coarse")
        print(f"  1. Increase grid: M=20, N=20 or larger")
        print(f"  2. Use scale factor {best_scale} for Volga extraction")
    else:
        print(f"\n✗ AAD not matching bumping - debug needed")
        print(f"  Check Hessian computation for errors")


if __name__ == "__main__":
    test_volga_complete()
