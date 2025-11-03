#!/usr/bin/env python3
"""
Diagnose Gamma Sign Issue

The problem: Gamma is negative when it should be positive
This script tests different components to isolate the issue
"""

import numpy as np
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_rannacher import BS_PDE_AAD_Rannacher
from aad_edge_pushing.pde.bsm_analytical import BSMAnalytical

def main():
    print("="*80)
    print("GAMMA SIGN ISSUE DIAGNOSIS")
    print("="*80)

    # Test parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.5
    M, N = 51, 100

    # Step 1: Analytical Greeks (baseline)
    print(f"\n[1/5] Analytical Greeks (baseline)...")
    analytical = BSMAnalytical()
    analytical_greeks = analytical.compute_greeks(S0, K, T, r, sigma, 'C')

    print(f"  Price = {analytical_greeks['price']:.6f}")
    print(f"  Delta = {analytical_greeks['delta']:.6f}")
    print(f"  Gamma = {analytical_greeks['gamma']:.6f} (should be POSITIVE)")
    print(f"  Vega  = {analytical_greeks['vega']:.6f}")

    # Step 2: Test PDE solver without Hessian (Jacobian only)
    print(f"\n[2/5] PDE Solver - Jacobian only (no Hessian)...")
    solver = BS_PDE_AAD_Rannacher(
        S0=S0, K=K, T=T, r=r, M=M, N_base=N,
        use_rannacher=False  # Test standard C-N first
    )

    result_jacobian = solver.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=False,  # Only Jacobian
        verbose=False
    )

    print(f"  Price = {result_jacobian['price']:.6f}")
    print(f"  Delta = {result_jacobian['delta']:.6f}")
    print(f"  Vega  = {result_jacobian['vega']:.6f}")
    print(f"  (No Gamma computed yet)")

    # Step 3: Test with Hessian (Edge-Pushing)
    print(f"\n[3/5] PDE Solver - With Hessian (Edge-Pushing)...")
    result_hessian = solver.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,  # Compute Hessian
        verbose=False
    )

    print(f"  Price = {result_hessian['price']:.6f}")
    print(f"  Delta = {result_hessian['delta']:.6f}")
    print(f"  Gamma = {result_hessian['gamma']:.6f} ← PROBLEM: Should be ~{analytical_greeks['gamma']:.6f}")
    print(f"  Vega  = {result_hessian['vega']:.6f}")
    print(f"  Vanna = {result_hessian['vanna']:.6f}")
    print(f"  Volga = {result_hessian['volga']:.6f}")

    # Step 4: Check Hessian matrix
    print(f"\n[4/5] Hessian Matrix Inspection...")
    if 'hessian' in result_hessian:
        hess = result_hessian['hessian']
        print(f"  Hessian shape: {hess.shape}")
        print(f"  Hessian:\n{hess}")
        print(f"  Γ = H[0,0] = {hess[0,0]:.6f}")
        print(f"  Vanna = H[0,1] = H[1,0] = {hess[0,1]:.6f}")
        print(f"  Volga = H[1,1] = {hess[1,1]:.6f}")

    # Step 5: Test with finite difference (bumping) for comparison
    print(f"\n[5/5] Finite Difference Gamma (bumping)...")
    eps = 0.01

    # V(S0 + eps)
    result_plus = solver.solve_pde_with_aad(
        S0_val=S0 + eps,
        sigma_val=sigma,
        compute_hessian=False,
        verbose=False
    )

    # V(S0 - eps)
    result_minus = solver.solve_pde_with_aad(
        S0_val=S0 - eps,
        sigma_val=sigma,
        compute_hessian=False,
        verbose=False
    )

    # V(S0)
    result_mid = solver.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=False,
        verbose=False
    )

    # Gamma = d²V/dS² ≈ (V+ - 2V0 + V-) / eps²
    gamma_fd = (result_plus['price'] - 2*result_mid['price'] + result_minus['price']) / (eps**2)

    print(f"  V(S0-ε) = {result_minus['price']:.6f}")
    print(f"  V(S0)   = {result_mid['price']:.6f}")
    print(f"  V(S0+ε) = {result_plus['price']:.6f}")
    print(f"  Gamma (FD) = {gamma_fd:.6f}")

    # Summary
    print(f"\n{'='*80}")
    print("DIAGNOSIS SUMMARY")
    print(f"{'='*80}")
    print(f"\n  Analytical Gamma:     {analytical_greeks['gamma']:.6f}")
    print(f"  Edge-Pushing Gamma:   {result_hessian['gamma']:.6f} ← WRONG SIGN & MAGNITUDE")
    print(f"  Finite Diff Gamma:    {gamma_fd:.6f}")

    print(f"\n  KEY OBSERVATIONS:")
    if result_hessian['gamma'] < 0:
        print(f"    ✗ Edge-Pushing Gamma is NEGATIVE (should be positive)")
    if abs(result_hessian['gamma'] - analytical_greeks['gamma']) > abs(gamma_fd - analytical_greeks['gamma']):
        print(f"    ✗ Edge-Pushing is WORSE than simple finite difference")

    print(f"\n  POSSIBLE CAUSES:")
    print(f"    1. Edge-Pushing algorithm implementation error")
    print(f"    2. Natural cubic spline second derivative calculation")
    print(f"    3. ADVar gradient propagation in Hessian computation")
    print(f"    4. Grid resolution too coarse (M={M}, N={N})")

    print(f"\n  RECOMMENDED ACTIONS:")
    print(f"    1. Test with larger grid (M=201, N=400)")
    print(f"    2. Compare with original BS_PDE_AAD (non-Rannacher)")
    print(f"    3. Debug Edge-Pushing Hessian computation")
    print(f"    4. Check if spline interpolation preserves convexity")

    print()

if __name__ == "__main__":
    main()
