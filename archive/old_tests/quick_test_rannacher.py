#!/usr/bin/env python3
"""
Quick Rannacher Test - Single Scenario Only

Tests one high-volatility scenario to quickly verify Rannacher improvement
"""

import numpy as np
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_rannacher import BS_PDE_AAD_Rannacher


def compute_analytical_gamma(S0, K, T, r, sigma):
    """Compute analytical Gamma"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    phi_d1 = norm.pdf(d1)
    gamma = phi_d1 / (S0 * sigma * sqrt_T)
    return gamma


def main():
    print("="*80)
    print("QUICK RANNACHER TEST")
    print("="*80)

    # High volatility scenario
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.5
    M, N = 51, 100

    analytical_gamma = compute_analytical_gamma(S0, K, T, r, sigma)
    print(f"\nAnalytical Gamma: {analytical_gamma:.6f}")

    # Test 1: Standard C-N (R=0)
    print(f"\n[1/2] Testing Standard C-N (R=0)...")
    solver_cn = BS_PDE_AAD_Rannacher(
        S0=S0, K=K, T=T, r=r, M=M, N_base=N,
        use_rannacher=False
    )
    result_cn = solver_cn.solve_pde_with_aad(S0, sigma, compute_hessian=True)
    gamma_cn = result_cn['gamma']
    error_cn = abs((gamma_cn - analytical_gamma) / analytical_gamma) * 100
    print(f"  Gamma = {gamma_cn:.6f}, Error = {error_cn:.2f}%")

    # Test 2: Rannacher (R=4)
    print(f"\n[2/2] Testing Rannacher (R=4)...")
    solver_rann = BS_PDE_AAD_Rannacher(
        S0=S0, K=K, T=T, r=r, M=M, N_base=N,
        use_rannacher=True,
        rannacher_steps=4
    )
    result_rann = solver_rann.solve_pde_with_aad(S0, sigma, compute_hessian=True)
    gamma_rann = result_rann['gamma']
    error_rann = abs((gamma_rann - analytical_gamma) / analytical_gamma) * 100
    print(f"  Gamma = {gamma_rann:.6f}, Error = {error_rann:.2f}%")

    # Comparison
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")
    print(f"\n  Analytical:     Gamma = {analytical_gamma:.6f}")
    print(f"  Standard C-N:   Gamma = {gamma_cn:.6f}, Error = {error_cn:.2f}%")
    print(f"  Rannacher R=4:  Gamma = {gamma_rann:.6f}, Error = {error_rann:.2f}%")

    improvement = ((error_cn - error_rann) / error_cn) * 100
    print(f"\n  Error Reduction: {improvement:.1f}%")

    if error_rann < error_cn:
        print(f"\n  ✓ SUCCESS: Rannacher reduces Gamma error!")
    else:
        print(f"\n  ⚠ WARNING: Rannacher did not improve accuracy")

    print()


if __name__ == "__main__":
    main()
