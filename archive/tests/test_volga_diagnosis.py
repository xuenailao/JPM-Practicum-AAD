"""
Diagnose Volga calculation issues
"""
import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.greeks_methods_comparison import (
    GreeksMethodA,
    black_scholes_analytical
)


def black_scholes_volga(S0, K, T, r, sigma):
    """Analytical Volga formula"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma
    return volga


def test_volga_calculation():
    """Test different Volga calculation approaches"""

    # Parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # Analytical Greeks
    price_bs, delta_bs, gamma_bs, vega_bs = black_scholes_analytical(S0, K, T, r, sigma)
    volga_bs = black_scholes_volga(S0, K, T, r, sigma)

    print("\n" + "="*80)
    print("VOLGA DIAGNOSTIC TEST")
    print("="*80)
    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"\nAnalytical (Black-Scholes):")
    print(f"  Price: {price_bs:.6f}")
    print(f"  Vega:  {vega_bs:.6f}")
    print(f"  Volga: {volga_bs:.6f}")

    # Test with Method A
    print("\n" + "-"*80)
    print("Method A: Finite Difference on Vega")
    print("-"*80)

    method = GreeksMethodA(M=51, N=50)

    # Test different eps_sigma values
    eps_values = [
        sigma * 0.001,  # 0.1% of sigma
        sigma * 0.01,   # 1% of sigma
        sigma * 0.05,   # 5% of sigma
        sigma * 0.1,    # 10% of sigma
    ]

    print(f"\n{'eps_sigma':<12} | {'Vega-':<12} | {'Vega0':<12} | {'Vega+':<12} | {'Volga':<12} | {'Error':<10}")
    print("-"*90)

    for eps_sigma in eps_values:
        # Compute Vega at three sigma points
        _, vega_minus = method._solve_at_S0(S0, sigma - eps_sigma)
        _, vega_center = method._solve_at_S0(S0, sigma)
        _, vega_plus = method._solve_at_S0(S0, sigma + eps_sigma)

        # Finite difference
        volga_fd = (vega_plus - 2*vega_center + vega_minus) / (eps_sigma ** 2)

        error = abs(volga_fd - volga_bs) / abs(volga_bs) * 100

        print(f"{eps_sigma:<12.6f} | {vega_minus:<12.6f} | {vega_center:<12.6f} | "
              f"{vega_plus:<12.6f} | {volga_fd:<12.6f} | {error:<10.2f}%")

    # Analytical finite difference test (on BS formula itself)
    print("\n" + "-"*80)
    print("Control Test: Finite Difference on Analytical Vega")
    print("-"*80)
    print(f"\n{'eps_sigma':<12} | {'Vega-':<12} | {'Vega0':<12} | {'Vega+':<12} | {'Volga':<12} | {'Error':<10}")
    print("-"*90)

    for eps_sigma in eps_values:
        _, _, _, vega_minus = black_scholes_analytical(S0, K, T, r, sigma - eps_sigma)
        _, _, _, vega_center = black_scholes_analytical(S0, K, T, r, sigma)
        _, _, _, vega_plus = black_scholes_analytical(S0, K, T, r, sigma + eps_sigma)

        volga_fd = (vega_plus - 2*vega_center + vega_minus) / (eps_sigma ** 2)
        error = abs(volga_fd - volga_bs) / abs(volga_bs) * 100

        print(f"{eps_sigma:<12.6f} | {vega_minus:<12.6f} | {vega_center:<12.6f} | "
              f"{vega_plus:<12.6f} | {volga_fd:<12.6f} | {error:<10.2f}%")

    # Key insight
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    print("\n1. If control test (analytical) gives good Volga → formula is correct")
    print("2. If Method A gives wrong Volga → PDE Vega precision is insufficient")
    print("3. Expected: Vega error ~13% amplifies to Volga error ~26-50%")
    print("\n4. Current observation:")
    print(f"   - PDE Vega: {vega_center:.6f}")
    print(f"   - BS Vega:  {vega_bs:.6f}")
    print(f"   - Vega error: {abs(vega_center - vega_bs)/vega_bs*100:.2f}%")
    print("\n5. For second derivative (Volga), error amplifies as:")
    print("   Volga_error ≈ (Vega_error)² for finite difference")
    print(f"   Expected: ~{(abs(vega_center - vega_bs)/vega_bs*100)**2/100:.1f}× worse")


if __name__ == "__main__":
    test_volga_calculation()
