"""
Test Grid Resolution Impact on Gamma Accuracy

Tests different grid resolutions (M, N) to see how Gamma accuracy improves.
"""

import numpy as np
from scipy.stats import norm
import time
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


def test_grid_resolution(M, N):
    """Test a specific grid resolution"""
    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    # Analytical
    bsm = bsm_analytical(S0, K, T, r, sigma)

    # PDE with AAD
    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    t_start = time.perf_counter()
    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=False)
    t_end = time.perf_counter()
    time_ms = (t_end - t_start) * 1000.0

    # Compute errors
    errors = {
        'price': abs(result['price'] - bsm['price']) / bsm['price'] * 100,
        'delta': abs(result['delta'] - bsm['delta']) / bsm['delta'] * 100,
        'gamma': abs(result['gamma'] - bsm['gamma']) / bsm['gamma'] * 100,
        'vega': abs(result['vega'] - bsm['vega']) / bsm['vega'] * 100,
        'vanna': abs(result['vanna'] - bsm['vanna']) / abs(bsm['vanna']) * 100,
        'volga': abs(result['volga'] - bsm['volga']) / bsm['volga'] * 100,
    }

    return {
        'M': M,
        'N': N,
        'dS': 300.0 / M,
        'dt': T / N,
        'gamma_pde': result['gamma'],
        'gamma_bsm': bsm['gamma'],
        'errors': errors,
        'time_ms': time_ms
    }


def main():
    print("="*100)
    print("Grid Resolution Test: Impact on Gamma Accuracy")
    print("="*100)
    print("\nTesting S0 as ADVar with cubic Hermite interpolation")
    print("Measuring how Gamma accuracy improves with finer grids\n")

    # Test configurations: (M, N)
    configs = [
        # Small grids
        (21, 20),
        (21, 50),
        (21, 100),
        (21, 200),

        # Medium grids (varying N)
        (51, 50),
        (51, 100),
        (51, 200),
        (51, 400),

        # Large grids (varying N)
        (101, 100),
        (101, 200),
        (101, 400),

        # Very fine grids
        (201, 200),
        (201, 400),
    ]

    print(f"{'M':>5} {'N':>5} {'dS':>8} {'dt':>10} {'Γ_PDE':>12} {'Γ_BSM':>12} {'Error':>8} {'Time':>10}")
    print("-"*100)

    results = []

    for M, N in configs:
        try:
            result = test_grid_resolution(M, N)
            results.append(result)

            print(f"{result['M']:>5d} {result['N']:>5d} "
                  f"{result['dS']:>8.2f} {result['dt']:>10.6f} "
                  f"{result['gamma_pde']:>12.8f} {result['gamma_bsm']:>12.8f} "
                  f"{result['errors']['gamma']:>7.2f}% "
                  f"{result['time_ms']:>9.1f}ms")

        except Exception as e:
            print(f"{M:>5d} {N:>5d} - FAILED: {str(e)[:40]}")

    # Summary by M groups
    print("\n" + "="*100)
    print("Summary by Spatial Resolution (M)")
    print("="*100)

    for M_target in [21, 51, 101, 201]:
        M_results = [r for r in results if r['M'] == M_target]
        if not M_results:
            continue

        print(f"\nM={M_target} (dS={300.0/M_target:.2f}):")
        print(f"  {'N':>5} {'Gamma Error':>12} {'Time':>10}")
        print(f"  {'-'*30}")

        for r in M_results:
            print(f"  {r['N']:>5d} {r['errors']['gamma']:>11.2f}% {r['time_ms']:>9.1f}ms")

        # Best result
        best = min(M_results, key=lambda x: x['errors']['gamma'])
        print(f"  → Best: N={best['N']}, Error={best['errors']['gamma']:.2f}%")

    # Find best overall
    print("\n" + "="*100)
    print("Best Results")
    print("="*100)

    best_gamma = min(results, key=lambda x: x['errors']['gamma'])
    print(f"\nLowest Gamma Error:")
    print(f"  M={best_gamma['M']}, N={best_gamma['N']}")
    print(f"  Gamma Error: {best_gamma['errors']['gamma']:.2f}%")
    print(f"  Time: {best_gamma['time_ms']:.1f}ms")

    # Fastest reasonable accuracy
    reasonable = [r for r in results if r['errors']['gamma'] < 20]
    if reasonable:
        fastest = min(reasonable, key=lambda x: x['time_ms'])
        print(f"\nFastest with <20% Gamma Error:")
        print(f"  M={fastest['M']}, N={fastest['N']}")
        print(f"  Gamma Error: {fastest['errors']['gamma']:.2f}%")
        print(f"  Time: {fastest['time_ms']:.1f}ms")

    print("\n" + "="*100)
    print("Key Findings:")
    print("="*100)
    print("1. Gamma error decreases with finer grids (smaller dS, dt)")
    print("2. Spatial resolution (M) has stronger impact than temporal (N)")
    print("3. Trade-off between accuracy and computational cost")
    print("="*100)


if __name__ == "__main__":
    main()
