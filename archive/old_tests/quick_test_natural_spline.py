"""
Quick test to verify natural cubic spline implementation
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def bsm_analytical(S0, K, T, r, sigma):
    """BSM analytical Greeks"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0 * sigma * sqrt_T)
    vega = S0 * norm.pdf(d1) * sqrt_T
    vanna = -norm.pdf(d1) * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


def main():
    print("="*80)
    print("Quick Test: Natural Cubic Spline Implementation")
    print("="*80)

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    # Single test: M=51, N=50
    M = 51
    N = 50

    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Grid: M={M}, N={N}")

    # Analytical
    bsm = bsm_analytical(S0, K, T, r, sigma)
    print(f"\nBSM Analytical:")
    print(f"  Price = {bsm['price']:.10f}")
    print(f"  Delta = {bsm['delta']:.10f}")
    print(f"  Gamma = {bsm['gamma']:.10f}")
    print(f"  Vega  = {bsm['vega']:.10f}")

    # PDE with natural spline
    print(f"\n{'-'*80}")
    print("Computing with Natural Cubic Spline...")
    print('-'*80)

    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=True)

    print(f"\nPDE AAD Results:")
    print(f"  Price = {result['price']:.10f}")
    print(f"  Delta = {result['delta']:.10f}")
    print(f"  Gamma = {result.get('gamma', 0.0):.10f}")
    print(f"  Vega  = {result['vega']:.10f}")
    print(f"  Time  = {result['time_ms']:.2f} ms")

    # Errors
    print(f"\n{'-'*80}")
    print("Errors vs BSM:")
    print('-'*80)

    price_err = abs(result['price'] - bsm['price']) / bsm['price'] * 100
    delta_err = abs(result['delta'] - bsm['delta']) / bsm['delta'] * 100
    gamma_err = abs(result.get('gamma', 0.0) - bsm['gamma']) / bsm['gamma'] * 100
    vega_err = abs(result['vega'] - bsm['vega']) / bsm['vega'] * 100

    print(f"  Price: {price_err:.2f}%")
    print(f"  Delta: {delta_err:.2f}%")
    print(f"  Gamma: {gamma_err:.2f}%")
    print(f"  Vega:  {vega_err:.2f}%")

    # Check gamma is non-zero
    gamma_val = result.get('gamma', 0.0)
    if abs(gamma_val) > 1e-10:
        print(f"\n✅ SUCCESS: Gamma = {gamma_val:.10f} (non-zero!)")
        if gamma_err < 15:
            print(f"   ✅ Gamma accuracy is GOOD (< 15% error)")
        elif gamma_err < 30:
            print(f"   ⚠️  Gamma accuracy is MODERATE (15-30% error)")
        else:
            print(f"   ❌ Gamma accuracy is LOW (> 30% error)")
    else:
        print(f"\n❌ FAILURE: Gamma is still zero")

    print("="*80)


if __name__ == "__main__":
    main()
