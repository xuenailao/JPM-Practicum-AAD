"""
Simple Test: Capriotti CN AAD - Small Grid First

Quick validation and speed test.
"""

import numpy as np
import time
from scipy.stats import norm
from aad_edge_pushing.pde.aad_integration.capriotti_cn_aad import CapriottiCNAAD


def bsm_greeks(S, K, T, r, sigma):
    """BSM analytical Greeks."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    vega = S * norm.pdf(d1) * sqrt_T
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vanna = -norm.pdf(d1) * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return price, delta, gamma, vega, vanna, volga


def test_single_size(M, N):
    """Test single grid configuration."""
    print(f"\n{'='*80}")
    print(f"Testing Grid: M={M}, N={N} (Interior: {M-2}×{N}, N/M={N/M:.1f})")
    print(f"{'='*80}")

    # Parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # BSM Analytical
    print("\n[1] BSM Analytical...")
    t0 = time.time()
    price_bs, delta_bs, gamma_bs, vega_bs, vanna_bs, volga_bs = bsm_greeks(S0, K, T, r, sigma)
    t_bsm = (time.time() - t0) * 1000
    print(f"    Time: {t_bsm:.3f} ms")

    # PDE AAD
    print("\n[2] PDE AAD Edge-Pushing...")
    solver = CapriottiCNAAD(M=M, N=N, phi=0.5)
    solver.S0 = S0
    solver.K = K
    solver.T = T
    solver.r = r
    solver.Smax = 2.0 * K
    solver.S_grid = np.linspace(0, solver.Smax, M)
    solver.dS = solver.S_grid[1] - solver.S_grid[0]
    solver.dt = T / N

    sigma_values = np.full(M - 1, sigma)

    t0 = time.time()
    try:
        price_pde, gradient, hessian = solver.compute_hessian_cn_algo4(sigma_values)
        t_pde = (time.time() - t0) * 1000

        # Extract Greeks
        vega_pde = np.mean(gradient) if len(gradient) > 0 else 0.0
        volga_pde = np.mean(np.diag(hessian)) if hessian.shape[0] > 0 else 0.0

        print(f"    Time: {t_pde:.3f} ms")
        print(f"    Graph nodes: {len(solver.solve_pde_cn_advar(sigma_values)[0]._tape.nodes) if hasattr(solver.solve_pde_cn_advar(sigma_values)[0], '_tape') else 'N/A'}")

        # Results
        print(f"\n{'Metric':<15} | {'BSM':<15} | {'PDE AAD':<15} | {'Abs Error':<15} | {'Rel Error (%)':<15}")
        print("-" * 80)
        print(f"{'Price':<15} | ${price_bs:<14.6f} | ${price_pde:<14.6f} | {abs(price_pde - price_bs):<15.2e} | {abs(price_pde - price_bs)/price_bs*100:<15.2f}")
        print(f"{'Vega':<15} | {vega_bs:<15.6f} | {vega_pde:<15.6f} | {abs(vega_pde - vega_bs):<15.2e} | {abs(vega_pde - vega_bs)/vega_bs*100:<15.2f}")
        print(f"{'Volga':<15} | {volga_bs:<15.6f} | {volga_pde:<15.6f} | {abs(volga_pde - volga_bs):<15.2e} | {abs(volga_pde - volga_bs)/abs(volga_bs)*100:<15.2f}")

        return {
            'M': M, 'N': N,
            't_pde': t_pde,
            'price_err': abs(price_pde - price_bs)/price_bs*100,
            'vega_err': abs(vega_pde - vega_bs)/vega_bs*100,
            'volga_err': abs(volga_pde - volga_bs)/abs(volga_bs)*100
        }

    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print("="*80)
    print("  SIMPLE TEST: Capriotti CN AAD")
    print("  Testing progressively larger grids (N >> M)")
    print("="*80)

    # Test configurations: Start small, increase with N >> M
    configs = [
        (10, 20),    # Very small
        (10, 50),    # N=5M
        (15, 75),    # N=5M
        (20, 100),   # N=5M
        (25, 150),   # N=6M
        (30, 200),   # N=6.7M
    ]

    results = []

    for M, N in configs:
        result = test_single_size(M, N)
        if result:
            results.append(result)

    # Summary
    if results:
        print(f"\n\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\n{'M':<8} | {'N':<8} | {'N/M':<8} | {'Time (ms)':<12} | {'Price Err (%)':<15} | {'Vega Err (%)':<15} | {'Volga Err (%)':<15}")
        print("-" * 100)

        for r in results:
            print(f"{r['M']:<8} | {r['N']:<8} | {r['N']/r['M']:<8.1f} | {r['t_pde']:<12.2f} | {r['price_err']:<15.2f} | {r['vega_err']:<15.2f} | {r['volga_err']:<15.2f}")

        print("\nObservations:")
        print(f"  - Smallest grid: {results[0]['t_pde']:.2f} ms")
        print(f"  - Largest grid: {results[-1]['t_pde']:.2f} ms")
        print(f"  - Scaling: {results[-1]['t_pde']/results[0]['t_pde']:.2f}× for {(results[-1]['M']*results[-1]['N'])/(results[0]['M']*results[0]['N']):.2f}× grid size")
        print(f"  - Average Price Error: {np.mean([r['price_err'] for r in results]):.2f}%")
        print(f"  - Average Vega Error: {np.mean([r['vega_err'] for r in results]):.2f}%")


if __name__ == "__main__":
    main()
