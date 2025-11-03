"""
Debug natural cubic spline implementation
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def main():
    print("="*80)
    print("Debug: Natural Cubic Spline")
    print("="*80)

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    M = 21  # Small grid for easier debugging
    N = 20

    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Grid: M={M}, N={N}")

    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    print(f"\nGrid info:")
    print(f"  S_grid (full): {len(solver.S_grid)} points")
    print(f"  S_grid range: [{solver.S_grid[0]:.2f}, {solver.S_grid[-1]:.2f}]")
    print(f"  dS = {solver.dS:.2f}")
    print(f"  S_grid[1:-1] (interior): {len(solver.S_grid[1:-1])} points")
    print(f"  S0={S0} should be in interior: {solver.S_grid[1:-1][0]:.2f} to {solver.S_grid[1:-1][-1]:.2f}")

    # Find where S0 is
    S_interior = solver.S_grid[1:-1]
    idx = np.searchsorted(S_interior, S0)
    print(f"\n  np.searchsorted(S_interior, {S0}) = {idx}")
    if idx > 0 and idx < len(S_interior):
        print(f"  S0={S0:.2f} is between S_interior[{idx-1}]={S_interior[idx-1]:.2f} and S_interior[{idx}]={S_interior[idx]:.2f}")

    # Test simple spline on known function
    print(f"\n{'-'*80}")
    print("Testing spline on f(x) = x^2 (known curvature = 2)")
    print('-'*80)

    x = np.linspace(0, 10, M)
    y = x**2

    print(f"  x range: [{x[0]:.2f}, {x[-1]:.2f}]")
    print(f"  y range: [{y[0]:.2f}, {y[-1]:.2f}]")

    # Compute spline second derivatives
    from aad_edge_pushing.aad.core.var import ADVar
    V_test = [ADVar(val, requires_grad=False) for val in y[1:-1]]
    M_vals_test = solver._compute_spline_second_derivatives(V_test, x[1:-1])

    print(f"\n  Spline second derivatives M_i:")
    print(f"  M_vals length: {len(M_vals_test)} (should be {len(V_test)+2} with boundaries)")
    for k in range(min(5, len(M_vals_test))):
        print(f"    M[{k}] = {M_vals_test[k].val:.6f}")
    print(f"    ...")
    for k in range(max(5, len(M_vals_test)-2), len(M_vals_test)):
        print(f"    M[{k}] = {M_vals_test[k].val:.6f}")

    print(f"\n  Expected: For f(x)=x^2, exact M_i should be ~2.0 everywhere")
    print(f"  Actual interior average: {np.mean([M_vals_test[k].val for k in range(1, len(M_vals_test)-1)]):.6f}")

    # Now run actual PDE
    print(f"\n{'-'*80}")
    print("Running PDE solver with natural spline...")
    print('-'*80)

    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=True)

    print(f"\nResults:")
    print(f"  Price = {result['price']:.6f}")
    print(f"  Delta = {result['delta']:.6f}")
    print(f"  Vega  = {result['vega']:.6f}")

    # BSM analytical
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    bsm_price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    bsm_delta = norm.cdf(d1)
    bsm_vega = S0 * norm.pdf(d1) * sqrt_T

    print(f"\nBSM Analytical:")
    print(f"  Price = {bsm_price:.6f}")
    print(f"  Delta = {bsm_delta:.6f}")
    print(f"  Vega  = {bsm_vega:.6f}")

    print(f"\nErrors:")
    print(f"  Price: {abs(result['price']-bsm_price)/bsm_price*100:.2f}%")
    print(f"  Delta: {abs(result['delta']-bsm_delta)/bsm_delta*100:.2f}%")
    print(f"  Vega:  {abs(result['vega']-bsm_vega)/bsm_vega*100:.2f}%")

    print("="*80)


if __name__ == "__main__":
    main()
