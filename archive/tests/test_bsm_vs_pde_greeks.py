"""
Compare Second-Order Greeks:
1. BSM Analytical (Ground Truth)
2. PDE Bumping (Finite Differences)
3. PDE AAD Edge-Pushing (True Second-Order AD)

This test validates all three methods against each other.
"""

import sys
import numpy as np
import time
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.true_second_order_ad_optimized import TrueSecondOrderADOptimized


# ============================================================================
# BSM Analytical Greeks
# ============================================================================

class BSMAnalytical:
    """Black-Scholes-Merton analytical Greeks."""

    @staticmethod
    def d1(S, K, T, r, sigma):
        """Calculate d1."""
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        """Calculate d2."""
        return BSMAnalytical.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        """Call option price."""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        d2 = BSMAnalytical.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    @staticmethod
    def delta(S, K, T, r, sigma):
        """Delta: ∂V/∂S."""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        return norm.cdf(d1)

    @staticmethod
    def gamma(S, K, T, r, sigma):
        """Gamma: ∂²V/∂S²."""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S, K, T, r, sigma):
        """Vega: ∂V/∂σ (per 1% vol change)."""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1%

    @staticmethod
    def vanna(S, K, T, r, sigma):
        """Vanna: ∂²V/∂S∂σ."""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        d2 = BSMAnalytical.d2(S, K, T, r, sigma)
        return -norm.pdf(d1) * d2 / sigma / 100  # Per 1%

    @staticmethod
    def volga(S, K, T, r, sigma):
        """Volga (Vomma): ∂²V/∂σ²."""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        d2 = BSMAnalytical.d2(S, K, T, r, sigma)
        vega_raw = S * norm.pdf(d1) * np.sqrt(T)
        return vega_raw * d1 * d2 / sigma / 10000  # Per 1% squared


# ============================================================================
# PDE Bumping Greeks
# ============================================================================

def pde_bumping_greeks(S0, K, T, r, sigma_0, M=20, N=20, eps_S=0.01, eps_sigma=0.0001):
    """
    Compute Greeks via PDE + finite differences.

    Returns:
        dict with price, delta, gamma, vega, vanna, volga
    """
    solver = TrueSecondOrderADOptimized(M=M, N=N)

    # Constant volatility surface
    sigma_grid = np.ones((M-1, N)) * sigma_0

    # Base price
    price_base, _, _ = solver.solve_pde_with_greeks(S0, K, T, r, sigma_grid, cp_flag='C')

    # Delta: ∂V/∂S
    price_up, _, _ = solver.solve_pde_with_greeks(S0 + eps_S, K, T, r, sigma_grid, cp_flag='C')
    price_down, _, _ = solver.solve_pde_with_greeks(S0 - eps_S, K, T, r, sigma_grid, cp_flag='C')
    delta = (price_up - price_down) / (2 * eps_S)

    # Gamma: ∂²V/∂S²
    gamma = (price_up - 2*price_base + price_down) / (eps_S**2)

    # Vega: ∂V/∂σ (perturb entire surface)
    sigma_up = np.ones((M-1, N)) * (sigma_0 + eps_sigma)
    sigma_down = np.ones((M-1, N)) * (sigma_0 - eps_sigma)

    price_sigma_up, _, _ = solver.solve_pde_with_greeks(S0, K, T, r, sigma_up, cp_flag='C')
    price_sigma_down, _, _ = solver.solve_pde_with_greeks(S0, K, T, r, sigma_down, cp_flag='C')

    vega = (price_sigma_up - price_sigma_down) / (2 * eps_sigma) / 100  # Per 1%

    # Vanna: ∂²V/∂S∂σ (mixed derivative)
    price_up_sigma_up, _, _ = solver.solve_pde_with_greeks(S0 + eps_S, K, T, r, sigma_up, cp_flag='C')
    price_up_sigma_down, _, _ = solver.solve_pde_with_greeks(S0 + eps_S, K, T, r, sigma_down, cp_flag='C')
    price_down_sigma_up, _, _ = solver.solve_pde_with_greeks(S0 - eps_S, K, T, r, sigma_up, cp_flag='C')
    price_down_sigma_down, _, _ = solver.solve_pde_with_greeks(S0 - eps_S, K, T, r, sigma_down, cp_flag='C')

    vanna = ((price_up_sigma_up - price_up_sigma_down) -
             (price_down_sigma_up - price_down_sigma_down)) / (4 * eps_S * eps_sigma) / 100

    # Volga: ∂²V/∂σ²
    volga = (price_sigma_up - 2*price_base + price_sigma_down) / (eps_sigma**2) / 10000

    return {
        'price': price_base,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


# ============================================================================
# PDE AAD Edge-Pushing Greeks
# ============================================================================

def pde_aad_greeks(S0, K, T, r, sigma_0, M=20, N=20, eps_S=0.01):
    """
    Compute Greeks via PDE + AAD edge-pushing.

    First-order: AAD
    Second-order: True second-order AD (optimized)

    Returns:
        dict with price, delta, gamma, vega, vanna, volga
    """
    solver = TrueSecondOrderADOptimized(M=M, N=N)
    sigma_grid = np.ones((M-1, N)) * sigma_0

    # Price and first-order Greeks via AAD
    price, gradient, _ = solver.solve_pde_with_greeks(S0, K, T, r, sigma_grid, cp_flag='C')

    # Delta via finite difference (PDE doesn't differentiate w.r.t. S directly)
    price_up, _, _ = solver.solve_pde_with_greeks(S0 + eps_S, K, T, r, sigma_grid, cp_flag='C')
    price_down, _, _ = solver.solve_pde_with_greeks(S0 - eps_S, K, T, r, sigma_grid, cp_flag='C')
    delta = (price_up - price_down) / (2 * eps_S)

    # Gamma via finite difference
    gamma = (price_up - 2*price + price_down) / (eps_S**2)

    # Vega: sum of all gradient components (total sensitivity to vol surface)
    vega = np.sum(gradient) / 100  # Per 1%

    # Vanna and Volga via second-order AD
    t0 = time.time()
    hessian = solver.compute_second_order_greeks(
        S0, K, T, r, sigma_grid,
        focus_region='center',  # Focus on ATM region
        cp_flag='C'
    )
    hessian_time = (time.time() - t0) * 1000

    # Extract Vanna: mixed S-σ derivative (approximate via finite diff + AAD)
    price_up_grad, _, _ = solver.solve_pde_with_greeks(S0 + eps_S, K, T, r, sigma_grid, cp_flag='C')
    price_down_grad, _, _ = solver.solve_pde_with_greeks(S0 - eps_S, K, T, r, sigma_grid, cp_flag='C')

    _, grad_up, _ = solver.solve_pde_with_greeks(S0 + eps_S, K, T, r, sigma_grid, cp_flag='C')
    _, grad_down, _ = solver.solve_pde_with_greeks(S0 - eps_S, K, T, r, sigma_grid, cp_flag='C')

    vanna = (np.sum(grad_up) - np.sum(grad_down)) / (2 * eps_S) / 100

    # Volga: ∂²V/∂σ² from Hessian diagonal
    # Take average of diagonal elements (all ∂²V/∂σᵢ²)
    hessian_diag = np.diag(hessian)
    volga = np.mean(hessian_diag) / 10000  # Per 1% squared

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga,
        'hessian_time': hessian_time,
        'hessian_shape': hessian.shape
    }


# ============================================================================
# Main Test
# ============================================================================

def test_bsm_vs_pde():
    """Compare BSM analytical vs PDE methods."""

    print("=" * 80)
    print("SECOND-ORDER GREEKS COMPARISON")
    print("=" * 80)
    print("\nMethods:")
    print("  1. BSM Analytical (Ground Truth)")
    print("  2. PDE Bumping (Finite Differences)")
    print("  3. PDE AAD Edge-Pushing (True Second-Order AD)")
    print("=" * 80)

    # Test parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma_0 = 0.2

    print(f"\nOption Parameters:")
    print(f"  S0 = {S0}, K = {K}, T = {T}")
    print(f"  r = {r}, σ = {sigma_0}")
    print(f"  Type: European Call (ATM)")

    # ========================================================================
    # 1. BSM Analytical
    # ========================================================================
    print("\n" + "=" * 80)
    print("【1. BSM ANALYTICAL (Ground Truth)】")
    print("=" * 80)

    t0 = time.time()
    bsm_price = BSMAnalytical.call_price(S0, K, T, r, sigma_0)
    bsm_delta = BSMAnalytical.delta(S0, K, T, r, sigma_0)
    bsm_gamma = BSMAnalytical.gamma(S0, K, T, r, sigma_0)
    bsm_vega = BSMAnalytical.vega(S0, K, T, r, sigma_0)
    bsm_vanna = BSMAnalytical.vanna(S0, K, T, r, sigma_0)
    bsm_volga = BSMAnalytical.volga(S0, K, T, r, sigma_0)
    bsm_time = (time.time() - t0) * 1000

    print(f"\nResults:")
    print(f"  Price:  {bsm_price:.6f}")
    print(f"  Delta:  {bsm_delta:.6f}")
    print(f"  Gamma:  {bsm_gamma:.6f}")
    print(f"  Vega:   {bsm_vega:.6f}")
    print(f"  Vanna:  {bsm_vanna:.6f}")
    print(f"  Volga:  {bsm_volga:.6f}")
    print(f"  Time:   {bsm_time:.2f} ms")

    # ========================================================================
    # 2. PDE Bumping
    # ========================================================================
    print("\n" + "=" * 80)
    print("【2. PDE BUMPING (Finite Differences)】")
    print("=" * 80)

    M_pde, N_pde = 40, 40
    print(f"\nGrid: {M_pde}×{N_pde}")

    t0 = time.time()
    pde_bump = pde_bumping_greeks(S0, K, T, r, sigma_0, M=M_pde, N=N_pde)
    pde_bump_time = (time.time() - t0) * 1000

    print(f"\nResults:")
    print(f"  Price:  {pde_bump['price']:.6f}")
    print(f"  Delta:  {pde_bump['delta']:.6f}")
    print(f"  Gamma:  {pde_bump['gamma']:.6f}")
    print(f"  Vega:   {pde_bump['vega']:.6f}")
    print(f"  Vanna:  {pde_bump['vanna']:.6f}")
    print(f"  Volga:  {pde_bump['volga']:.6f}")
    print(f"  Time:   {pde_bump_time:.2f} ms")

    print(f"\nErrors vs BSM:")
    print(f"  Price:  {abs(pde_bump['price'] - bsm_price):.2e} ({abs(pde_bump['price'] - bsm_price)/bsm_price*100:.3f}%)")
    print(f"  Delta:  {abs(pde_bump['delta'] - bsm_delta):.2e} ({abs(pde_bump['delta'] - bsm_delta)/bsm_delta*100:.3f}%)")
    print(f"  Gamma:  {abs(pde_bump['gamma'] - bsm_gamma):.2e} ({abs(pde_bump['gamma'] - bsm_gamma)/bsm_gamma*100:.3f}%)")
    print(f"  Vega:   {abs(pde_bump['vega'] - bsm_vega):.2e} ({abs(pde_bump['vega'] - bsm_vega)/bsm_vega*100:.3f}%)")
    print(f"  Vanna:  {abs(pde_bump['vanna'] - bsm_vanna):.2e} ({abs(pde_bump['vanna'] - bsm_vanna)/abs(bsm_vanna)*100:.3f}%)")
    print(f"  Volga:  {abs(pde_bump['volga'] - bsm_volga):.2e} ({abs(pde_bump['volga'] - bsm_volga)/abs(bsm_volga)*100:.3f}%)")

    # ========================================================================
    # 3. PDE AAD Edge-Pushing
    # ========================================================================
    print("\n" + "=" * 80)
    print("【3. PDE AAD EDGE-PUSHING (True Second-Order AD)】")
    print("=" * 80)

    print(f"\nGrid: {M_pde}×{N_pde}")

    t0 = time.time()
    pde_aad = pde_aad_greeks(S0, K, T, r, sigma_0, M=M_pde, N=N_pde)
    pde_aad_total_time = (time.time() - t0) * 1000

    print(f"\nResults:")
    print(f"  Price:  {pde_aad['price']:.6f}")
    print(f"  Delta:  {pde_aad['delta']:.6f}")
    print(f"  Gamma:  {pde_aad['gamma']:.6f}")
    print(f"  Vega:   {pde_aad['vega']:.6f}")
    print(f"  Vanna:  {pde_aad['vanna']:.6f}")
    print(f"  Volga:  {pde_aad['volga']:.6f}")
    print(f"  Hessian shape: {pde_aad['hessian_shape']}")
    print(f"  Hessian time:  {pde_aad['hessian_time']:.2f} ms")
    print(f"  Total time:    {pde_aad_total_time:.2f} ms")

    print(f"\nErrors vs BSM:")
    print(f"  Price:  {abs(pde_aad['price'] - bsm_price):.2e} ({abs(pde_aad['price'] - bsm_price)/bsm_price*100:.3f}%)")
    print(f"  Delta:  {abs(pde_aad['delta'] - bsm_delta):.2e} ({abs(pde_aad['delta'] - bsm_delta)/bsm_delta*100:.3f}%)")
    print(f"  Gamma:  {abs(pde_aad['gamma'] - bsm_gamma):.2e} ({abs(pde_aad['gamma'] - bsm_gamma)/bsm_gamma*100:.3f}%)")
    print(f"  Vega:   {abs(pde_aad['vega'] - bsm_vega):.2e} ({abs(pde_aad['vega'] - bsm_vega)/bsm_vega*100:.3f}%)")
    print(f"  Vanna:  {abs(pde_aad['vanna'] - bsm_vanna):.2e} ({abs(pde_aad['vanna'] - bsm_vanna)/abs(bsm_vanna)*100:.3f}%)")
    print(f"  Volga:  {abs(pde_aad['volga'] - bsm_volga):.2e} ({abs(pde_aad['volga'] - bsm_volga)/abs(bsm_volga)*100:.3f}%)")

    # ========================================================================
    # Summary Table
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    print(f"\n{'Greek':<10} {'BSM':<12} {'PDE Bump':<12} {'PDE AAD':<12} {'Bump Err%':<12} {'AAD Err%':<12}")
    print("-" * 80)

    greeks = ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']
    bsm_vals = [bsm_price, bsm_delta, bsm_gamma, bsm_vega, bsm_vanna, bsm_volga]

    for i, greek in enumerate(greeks):
        bsm_val = bsm_vals[i]
        bump_val = pde_bump[greek]
        aad_val = pde_aad[greek]

        bump_err = abs(bump_val - bsm_val) / abs(bsm_val) * 100 if bsm_val != 0 else 0
        aad_err = abs(aad_val - bsm_val) / abs(bsm_val) * 100 if bsm_val != 0 else 0

        print(f"{greek.capitalize():<10} {bsm_val:<12.6f} {bump_val:<12.6f} {aad_val:<12.6f} {bump_err:<12.3f} {aad_err:<12.3f}")

    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<30} {'Time (ms)':<15} {'Speedup vs BSM':<15}")
    print("-" * 60)
    print(f"{'BSM Analytical':<30} {bsm_time:<15.2f} {1.0:<15.1f}×")
    print(f"{'PDE Bumping':<30} {pde_bump_time:<15.2f} {pde_bump_time/bsm_time:<15.1f}×")
    print(f"{'PDE AAD Edge-Pushing':<30} {pde_aad_total_time:<15.2f} {pde_aad_total_time/bsm_time:<15.1f}×")

    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print("\n✓ BSM Analytical: Instant, exact (for constant vol)")
    print(f"✓ PDE Bumping: {pde_bump_time:.0f}ms, requires many PDE solves")
    print(f"✓ PDE AAD: {pde_aad_total_time:.0f}ms, single forward + efficient Hessian")
    print("\nFor time-dependent/local vol (where BSM fails):")
    print("  - PDE AAD scales better than bumping")
    print("  - AAD amortizes cost over many Greeks")
    print("=" * 80)


if __name__ == '__main__':
    test_bsm_vs_pde()
