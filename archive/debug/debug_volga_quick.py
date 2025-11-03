"""
Quick debug of Volga extraction from Hessian.

Test different methods to extract ∂²V/∂σ² from discrete Hessian ∂²V/∂σᵢⱼ∂σₖₗ
"""

import numpy as np
import time
from scipy.stats import norm

def bsm_volga(S, K, T, r, sigma):
    """BSM analytical Volga."""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    n_d1 = norm.pdf(d1)
    vega = S * n_d1 * np.sqrt(T)
    volga = vega * d1 * d2 / sigma
    return volga, vega

def test_volga_extraction():
    """Test different Volga extraction methods."""

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma_0 = 0.2
    M = 10
    N = 10

    print("="*80)
    print("QUICK VOLGA EXTRACTION DEBUG")
    print("="*80)
    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma_0}")
    print(f"Grid: M={M}, N={N}")

    # Get BSM ground truth
    volga_bsm, vega_bsm = bsm_volga(S0, K, T, r, sigma_0)
    print(f"\nBSM Analytical:")
    print(f"  Vega:  {vega_bsm:.6f}")
    print(f"  Volga: {volga_bsm:.6e}")

    # Import solver
    from aad_edge_pushing.pde.true_second_order_ad_optimized import TrueSecondOrderADOptimized

    # Initialize and solve
    print(f"\nInitializing PDE solver...")
    solver = TrueSecondOrderADOptimized(M=M, N=N, Smax_factor=4.0)
    sigma_grid = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid)

    print(f"Computing Hessian...")
    t0 = time.time()
    sparse_hess, metadata = solver.compute_hessian_optimized(
        S0, K, T, r, cp_flag='C', focus_region='all'
    )
    t1 = time.time()

    # Convert to dense
    # Note: focus_region='all' gives parameters: for n in range(N) for i in range(1, M)
    # So n_params = N * (M-1) = 10 * 9 = 90
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
    print(f"  Shape: {hessian.shape}")
    print(f"  Non-zero: {np.count_nonzero(hessian):,} / {hessian.size:,}")

    # Test different extraction methods
    print("\n" + "="*80)
    print("TESTING VOLGA EXTRACTION METHODS")
    print("="*80)

    diagonal = np.diag(hessian)

    # Method 1: Mean of diagonal
    volga_1 = np.mean(diagonal) / 10000
    error_1 = abs(volga_1 - volga_bsm) / abs(volga_bsm) * 100
    print(f"\n【Method 1: Mean(diag(H)) / 10000】")
    print(f"  Volga = {volga_1:.6e}")
    print(f"  Error = {error_1:.1f}%")

    # Method 2: Sum of diagonal
    volga_2 = np.sum(diagonal) / 10000
    error_2 = abs(volga_2 - volga_bsm) / abs(volga_bsm) * 100
    print(f"\n【Method 2: Sum(diag(H)) / 10000】")
    print(f"  Volga = {volga_2:.6e}")
    print(f"  Error = {error_2:.1f}%")

    # Method 3: Sum all Hessian elements
    volga_3 = np.sum(hessian) / 10000
    error_3 = abs(volga_3 - volga_bsm) / abs(volga_bsm) * 100
    print(f"\n【Method 3: Sum(H) / 10000】")
    print(f"  Volga = {volga_3:.6e}")
    print(f"  Error = {error_3:.1f}%")

    # Method 4: Weighted sum by grid area (S × t)
    # Each parameter σᵢⱼ affects region ΔS × Δt
    Smax = solver.Smax_factor * K
    dS = Smax / M
    dt = T / N

    # Weight by grid cell area
    weights = np.zeros(n_params)
    for n in range(N):
        for i in range(1, M):
            idx = param_to_idx[(i, n)]
            # Weight by spot level and time to maturity
            S_i = i * dS
            t_remaining = T - n * dt
            weights[idx] = S_i * t_remaining

    # Normalize weights
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else weights

    volga_4 = np.sum(diagonal * weights * n_params) / 10000
    error_4 = abs(volga_4 - volga_bsm) / abs(volga_bsm) * 100
    print(f"\n【Method 4: Weighted sum (S × t) / 10000】")
    print(f"  Volga = {volga_4:.6e}")
    print(f"  Error = {error_4:.1f}%")

    # Method 5: Focus on ATM region only
    i_atm = int(S0 / dS)
    atm_indices = []
    for n in range(N):
        for i in range(max(1, i_atm-2), min(M, i_atm+3)):
            if (i, n) in param_to_idx:
                atm_indices.append(param_to_idx[(i, n)])

    if len(atm_indices) > 0:
        atm_diagonal = diagonal[atm_indices]
        volga_5 = np.mean(atm_diagonal) / 10000
        error_5 = abs(volga_5 - volga_bsm) / abs(volga_bsm) * 100
        print(f"\n【Method 5: Mean(diag(H)) ATM region only / 10000】")
        print(f"  ATM indices: {len(atm_indices)}")
        print(f"  Volga = {volga_5:.6e}")
        print(f"  Error = {error_5:.1f}%")
    else:
        error_5 = 1000
        volga_5 = 0

    # Method 6: Use theoretical scaling (vega^2 relationship)
    # Volga ≈ Vega * (something from Hessian structure)
    # Try: volga = vega^2 / price * mean(diag(H))
    Smax = solver.Smax_factor * K
    dS = Smax / M
    i_S0 = int(S0 / dS)
    price_pde = solver.V_hist[-1][i_S0]

    # Estimate vega from gradient
    # (This is a placeholder - in reality we'd compute it properly)
    vega_scale = vega_bsm / price_pde if price_pde > 0 else 1.0

    volga_6 = np.mean(diagonal) * (vega_scale ** 2) / 100
    error_6 = abs(volga_6 - volga_bsm) / abs(volga_bsm) * 100
    print(f"\n【Method 6: Mean(diag(H)) * (vega/price)^2 / 100】")
    print(f"  Volga = {volga_6:.6e}")
    print(f"  Error = {error_6:.1f}%")

    # Summary
    methods = [
        ('Mean diagonal', volga_1, error_1),
        ('Sum diagonal', volga_2, error_2),
        ('Sum all', volga_3, error_3),
        ('Weighted (S×t)', volga_4, error_4),
        ('ATM only', volga_5, error_5),
        ('Vega-scaled', volga_6, error_6),
    ]

    best = min(methods, key=lambda x: x[2])

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\nBSM Analytical: {volga_bsm:.6e}")
    print(f"\nBest method: {best[0]}")
    print(f"  Value: {best[1]:.6e}")
    print(f"  Error: {best[2]:.1f}%")

    print(f"\nAll methods:")
    for name, val, err in sorted(methods, key=lambda x: x[2]):
        print(f"  {name:20s}: {val:12.6e}  (error: {err:6.1f}%)")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    if best[2] < 10:
        print(f"\n✓ Method '{best[0]}' achieves <10% error!")
        print(f"  Recommended for production use.")
    else:
        print(f"\n✗ Best error is {best[2]:.1f}% - still too large")
        print(f"\n  Possible reasons:")
        print(f"    1. Scale factor (10000) may be wrong")
        print(f"    2. Need to test with bumping method for comparison")
        print(f"    3. Hessian may represent local parameters, not global σ")

    print(f"\n  Next step: Compare with finite difference volga")


if __name__ == "__main__":
    test_volga_extraction()
