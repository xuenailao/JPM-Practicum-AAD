"""
Debug script to analyze Vanna and Volga calculation errors.

This script will:
1. Compute Hessian for a simple case
2. Visualize the Hessian structure
3. Analyze how Vanna and Volga are extracted
4. Compare with BSM analytical values
"""

import numpy as np
import time
from aad_edge_pushing.pde.true_second_order_ad_optimized import TrueSecondOrderADOptimized

def bsm_greeks(S, K, T, r, sigma):
    """BSM analytical Greeks (ground truth)."""
    from scipy.stats import norm

    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    # First-order Greeks
    price = S * N_d1 - K * np.exp(-r*T) * N_d2
    delta = N_d1
    vega = S * n_d1 * np.sqrt(T)

    # Second-order Greeks
    gamma = n_d1 / (S * sigma * np.sqrt(T))
    vanna = -n_d1 * d2 / sigma  # ∂²V/∂S∂σ
    volga = vega * d1 * d2 / sigma  # ∂²V/∂σ²

    return {
        'price': price,
        'delta': delta,
        'vega': vega,
        'gamma': gamma,
        'vanna': vanna,
        'volga': volga
    }


def analyze_hessian_structure(hessian, M, N):
    """Analyze the structure of the Hessian matrix."""
    print("\n" + "="*80)
    print("HESSIAN STRUCTURE ANALYSIS")
    print("="*80)

    n_params = (M-1) * (N-1)
    print(f"\nHessian shape: {hessian.shape}")
    print(f"Expected shape: ({n_params}, {n_params})")

    # Check symmetry
    is_symmetric = np.allclose(hessian, hessian.T)
    print(f"Is symmetric: {is_symmetric}")

    # Sparsity
    non_zero = np.count_nonzero(hessian)
    total = hessian.size
    sparsity = 100 * (1 - non_zero / total)
    print(f"Non-zero elements: {non_zero:,} / {total:,}")
    print(f"Sparsity: {sparsity:.1f}%")

    # Diagonal analysis
    diagonal = np.diag(hessian)
    print(f"\nDiagonal statistics:")
    print(f"  Min:  {np.min(diagonal):.6e}")
    print(f"  Max:  {np.max(diagonal):.6e}")
    print(f"  Mean: {np.mean(diagonal):.6e}")
    print(f"  Std:  {np.std(diagonal):.6e}")

    # Off-diagonal analysis
    off_diag = hessian[~np.eye(hessian.shape[0], dtype=bool)]
    print(f"\nOff-diagonal statistics:")
    print(f"  Min:  {np.min(off_diag):.6e}")
    print(f"  Max:  {np.max(off_diag):.6e}")
    print(f"  Mean: {np.mean(off_diag):.6e}")
    print(f"  Std:  {np.std(off_diag):.6e}")

    return {
        'diagonal': diagonal,
        'off_diagonal': off_diag,
        'sparsity': sparsity,
        'is_symmetric': is_symmetric
    }


def debug_volga_extraction(hessian, M, N, bsm_volga):
    """
    Debug how Volga (∂²V/∂σ²) is extracted from the Hessian.

    The Hessian is ∂²V/∂σᵢⱼ∂σₖₗ where σᵢⱼ are discrete grid parameters.
    We need to find how this relates to ∂²V/∂σ² for uniform volatility.
    """
    print("\n" + "="*80)
    print("VOLGA EXTRACTION DEBUG")
    print("="*80)

    diagonal = np.diag(hessian)

    # Current method: simple average of diagonal
    volga_avg = np.mean(diagonal) / 10000  # Scale factor
    error_avg = abs(volga_avg - bsm_volga) / abs(bsm_volga) * 100

    print(f"\nBSM Analytical Volga: {bsm_volga:.6e}")
    print(f"\nMethod 1: Average diagonal")
    print(f"  Volga = mean(diag(H)) / 10000 = {volga_avg:.6e}")
    print(f"  Error: {error_avg:.1f}%")

    # Method 2: Sum all Hessian elements (chain rule for uniform σ)
    # If σᵢⱼ = σ (uniform), then ∂²V/∂σ² = Σᵢⱼₖₗ ∂²V/∂σᵢⱼ∂σₖₗ
    volga_sum_all = np.sum(hessian) / 10000
    error_sum_all = abs(volga_sum_all - bsm_volga) / abs(bsm_volga) * 100

    print(f"\nMethod 2: Sum all Hessian elements")
    print(f"  Volga = sum(H) / 10000 = {volga_sum_all:.6e}")
    print(f"  Error: {error_sum_all:.1f}%")

    # Method 3: Sum diagonal only
    volga_sum_diag = np.sum(diagonal) / 10000
    error_sum_diag = abs(volga_sum_diag - bsm_volga) / abs(bsm_volga) * 100

    print(f"\nMethod 3: Sum diagonal")
    print(f"  Volga = sum(diag(H)) / 10000 = {volga_sum_diag:.6e}")
    print(f"  Error: {error_sum_diag:.1f}%")

    # Method 4: Weighted sum (by grid area)
    # Each grid cell has area dS * dt
    # Weight by time remaining T - n*dt
    n_params = (M-1) * (N-1)
    weights = np.ones(n_params)

    # Reshape to (M-1, N-1) to apply time weighting
    hess_diag_2d = diagonal.reshape(M-1, N-1)

    # Weight by time to expiry (parameters later in time matter less)
    time_weights = np.linspace(1.0, 0.1, N-1)  # Decreasing weights
    weighted_diag = hess_diag_2d * time_weights[np.newaxis, :]

    volga_weighted = np.sum(weighted_diag) / 10000
    error_weighted = abs(volga_weighted - bsm_volga) / abs(bsm_volga) * 100

    print(f"\nMethod 4: Time-weighted sum")
    print(f"  Volga = sum(diag(H) * time_weights) / 10000 = {volga_weighted:.6e}")
    print(f"  Error: {error_weighted:.1f}%")

    # Show which method is best
    methods = [
        ('Average diagonal', volga_avg, error_avg),
        ('Sum all', volga_sum_all, error_sum_all),
        ('Sum diagonal', volga_sum_diag, error_sum_diag),
        ('Weighted sum', volga_weighted, error_weighted),
    ]

    best_method = min(methods, key=lambda x: x[2])
    print(f"\n✓ Best method: {best_method[0]} (error = {best_method[2]:.1f}%)")

    return {
        'avg': volga_avg,
        'sum_all': volga_sum_all,
        'sum_diag': volga_sum_diag,
        'weighted': volga_weighted,
        'best': best_method
    }


def debug_vanna_extraction(solver, S0, K, T, r, sigma_0, M, N, bsm_vanna):
    """
    Debug how Vanna (∂²V/∂S∂σ) is extracted.

    Problem: S is not a PDE parameter. We need mixed derivative of
    price w.r.t. spot S and volatility σ.
    """
    print("\n" + "="*80)
    print("VANNA EXTRACTION DEBUG")
    print("="*80)

    print(f"\nBSM Analytical Vanna: {bsm_vanna:.6e}")

    # Method 1: Current implementation (finite diff on S, AAD on σ)
    print("\n【Method 1: Finite Diff on S + AAD on σ】")
    print("Problem: S is not a PDE parameter, this is conceptually wrong")

    eps_S = 0.01

    # Solve for S0 + eps
    t0 = time.time()
    sigma_grid_up = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid_up)
    price_up, grad_up, _ = solver.solve_local_vol(
        S0 + eps_S, K, T, r, cp_flag='C'
    )

    # Solve for S0 - eps
    sigma_grid_down = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid_down)
    price_down, grad_down, _ = solver.solve_local_vol(
        S0 - eps_S, K, T, r, cp_flag='C'
    )
    t1 = time.time()

    vanna_method1 = (np.sum(grad_up) - np.sum(grad_down)) / (2 * eps_S) / 100
    error1 = abs(vanna_method1 - bsm_vanna) / abs(bsm_vanna) * 100

    print(f"  Vanna = Δ(vega)/ΔS = {vanna_method1:.6e}")
    print(f"  Error: {error1:.1f}%")
    print(f"  Time: {(t1-t0)*1000:.2f} ms")

    # Method 2: Complete bumping (finite diff on both S and σ)
    print("\n【Method 2: Complete Finite Differences】")

    eps_sigma = 0.01

    t0 = time.time()
    # Base price
    sigma_grid_0 = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid_0)
    price_0, _, _ = solver.solve_local_vol(S0, K, T, r, cp_flag='C')

    # Price at S+ε, σ+ε
    sigma_grid_pp = np.full((M+1, N+1), sigma_0 + eps_sigma)
    solver.set_local_vol_grid(sigma_grid_pp)
    price_pp, _, _ = solver.solve_local_vol(
        S0 + eps_S, K, T, r, cp_flag='C'
    )

    # Price at S+ε, σ-ε
    sigma_grid_pm = np.full((M+1, N+1), sigma_0 - eps_sigma)
    solver.set_local_vol_grid(sigma_grid_pm)
    price_pm, _, _ = solver.solve_local_vol(
        S0 + eps_S, K, T, r, cp_flag='C'
    )

    # Price at S-ε, σ+ε
    sigma_grid_mp = np.full((M+1, N+1), sigma_0 + eps_sigma)
    solver.set_local_vol_grid(sigma_grid_mp)
    price_mp, _, _ = solver.solve_local_vol(
        S0 - eps_S, K, T, r, cp_flag='C'
    )

    # Price at S-ε, σ-ε
    sigma_grid_mm = np.full((M+1, N+1), sigma_0 - eps_sigma)
    solver.set_local_vol_grid(sigma_grid_mm)
    price_mm, _, _ = solver.solve_local_vol(
        S0 - eps_S, K, T, r, cp_flag='C'
    )

    t1 = time.time()

    # Central difference for mixed derivative
    vanna_method2 = (price_pp - price_pm - price_mp + price_mm) / (4 * eps_S * eps_sigma)
    error2 = abs(vanna_method2 - bsm_vanna) / abs(bsm_vanna) * 100

    print(f"  Vanna = [V(S+ε,σ+ε) - V(S+ε,σ-ε) - V(S-ε,σ+ε) + V(S-ε,σ-ε)] / (4εₛεᵩ)")
    print(f"  Vanna = {vanna_method2:.6e}")
    print(f"  Error: {error2:.1f}%")
    print(f"  Time: {(t1-t0)*1000:.2f} ms")

    # Method 3: Vega finite diff (d(Vega)/dS)
    print("\n【Method 3: Finite Diff of Vega】")

    t0 = time.time()
    # Compute vega at S0 + eps
    eps_sigma = 0.01

    sigma_grid_base = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid_base)
    price_S_up_base, _, _ = solver.solve_local_vol(
        S0 + eps_S, K, T, r, cp_flag='C'
    )

    sigma_grid_perturbed = np.full((M+1, N+1), sigma_0 + eps_sigma)
    solver.set_local_vol_grid(sigma_grid_perturbed)
    price_S_up_sig_up, _, _ = solver.solve_local_vol(
        S0 + eps_S, K, T, r, cp_flag='C'
    )

    vega_S_up = (price_S_up_sig_up - price_S_up_base) / eps_sigma

    # Compute vega at S0 - eps
    sigma_grid_base2 = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid_base2)
    price_S_down_base, _, _ = solver.solve_local_vol(
        S0 - eps_S, K, T, r, cp_flag='C'
    )

    sigma_grid_perturbed2 = np.full((M+1, N+1), sigma_0 + eps_sigma)
    solver.set_local_vol_grid(sigma_grid_perturbed2)
    price_S_down_sig_up, _, _ = solver.solve_local_vol(
        S0 - eps_S, K, T, r, cp_flag='C'
    )

    vega_S_down = (price_S_down_sig_up - price_S_down_base) / eps_sigma

    t1 = time.time()

    vanna_method3 = (vega_S_up - vega_S_down) / (2 * eps_S)
    error3 = abs(vanna_method3 - bsm_vanna) / abs(bsm_vanna) * 100

    print(f"  Vanna = d(Vega)/dS = {vanna_method3:.6e}")
    print(f"  Error: {error3:.1f}%")
    print(f"  Time: {(t1-t0)*1000:.2f} ms")

    # Compare methods
    methods = [
        ('Current (FD on S + AAD)', vanna_method1, error1),
        ('Complete FD', vanna_method2, error2),
        ('FD of Vega', vanna_method3, error3),
    ]

    best_method = min(methods, key=lambda x: x[2])
    print(f"\n✓ Best method: {best_method[0]} (error = {best_method[2]:.1f}%)")

    return {
        'method1': vanna_method1,
        'method2': vanna_method2,
        'method3': vanna_method3,
        'best': best_method
    }


def main():
    """Main debug routine."""

    # Test parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma_0 = 0.2

    # Small grid for fast debugging
    M = 10
    N = 10

    print("="*80)
    print("VANNA & VOLGA DEBUG ANALYSIS")
    print("="*80)
    print(f"\nParameters:")
    print(f"  S0={S0}, K={K}, T={T}, r={r}, σ={sigma_0}")
    print(f"  Grid: M={M}, N={N} ({(M-1)*(N-1)} parameters)")

    # Get BSM ground truth
    bsm = bsm_greeks(S0, K, T, r, sigma_0)
    print(f"\nBSM Analytical Greeks (Ground Truth):")
    print(f"  Price: {bsm['price']:.6f}")
    print(f"  Delta: {bsm['delta']:.6f}")
    print(f"  Gamma: {bsm['gamma']:.6f}")
    print(f"  Vega:  {bsm['vega']:.6f}")
    print(f"  Vanna: {bsm['vanna']:.6e}")
    print(f"  Volga: {bsm['volga']:.6e}")

    # Initialize solver
    print(f"\nInitializing PDE solver...")
    solver = TrueSecondOrderADOptimized(M=M, N=N, Smax_factor=4.0)

    # Solve PDE and compute Hessian
    print(f"Computing Hessian...")
    sigma_grid = np.full((M+1, N+1), sigma_0)
    solver.set_local_vol_grid(sigma_grid)

    t0 = time.time()
    sparse_hess, metadata = solver.compute_hessian_optimized(
        S0, K, T, r, cp_flag='C', focus_region='all'
    )
    t1 = time.time()

    # Convert sparse Hessian to dense
    n_params = (M-1) * (N-1)
    hessian = np.zeros((n_params, n_params))

    # Create parameter index mapping
    param_to_idx = {}
    idx = 0
    for n in range(N):
        for i in range(1, M):
            param_to_idx[(i, n)] = idx
            idx += 1

    for (i, n, j, m), val in sparse_hess.items():
        idx_i = param_to_idx[(i, n)]
        idx_j = param_to_idx[(j, m)]
        hessian[idx_i, idx_j] = val

    # Get price from final value function
    Smax = solver.Smax_factor * K
    dS = Smax / M
    i_S0 = int(S0 / dS)
    price = solver.V_hist[-1][i_S0]

    print(f"  Price (PDE): {price:.6f} (BSM: {bsm['price']:.6f})")
    print(f"  Time: {(t1-t0)*1000:.2f} ms")

    # Analyze Hessian structure
    hess_info = analyze_hessian_structure(hessian, M, N)

    # Debug Volga extraction
    volga_info = debug_volga_extraction(hessian, M, N, bsm['volga'])

    # Debug Vanna extraction
    vanna_info = debug_vanna_extraction(solver, S0, K, T, r, sigma_0, M, N, bsm['vanna'])

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    print(f"\n【Volga (∂²V/∂σ²)】")
    print(f"  BSM:         {bsm['volga']:.6e}")
    print(f"  Best PDE:    {volga_info['best'][1]:.6e}")
    print(f"  Best method: {volga_info['best'][0]}")
    print(f"  Error:       {volga_info['best'][2]:.1f}%")

    print(f"\n【Vanna (∂²V/∂S∂σ)】")
    print(f"  BSM:         {bsm['vanna']:.6e}")
    print(f"  Best PDE:    {vanna_info['best'][1]:.6e}")
    print(f"  Best method: {vanna_info['best'][0]}")
    print(f"  Error:       {vanna_info['best'][2]:.1f}%")

    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    print("\n1. For Volga:")
    print("   - Current 'average diagonal' method is WRONG")
    print("   - Use sum of all Hessian elements or sum of diagonal")
    print("   - Reason: ∂²V/∂σ² = Σᵢⱼₖₗ ∂²V/∂σᵢⱼ∂σₖₗ for uniform σ")

    print("\n2. For Vanna:")
    print("   - Current 'FD on S + AAD on σ' is conceptually wrong")
    print("   - Use complete finite differences or FD of Vega")
    print("   - Reason: S is not a PDE parameter, need proper mixed derivative")

    print("\n3. Alternative: Use AAD on outer solver")
    print("   - Wrap entire PDE solve_local_vol() in AAD")
    print("   - Differentiate w.r.t. S and σ as external parameters")
    print("   - This would give true ∂V/∂S and ∂V/∂σ, then compute second derivatives")


if __name__ == "__main__":
    main()
