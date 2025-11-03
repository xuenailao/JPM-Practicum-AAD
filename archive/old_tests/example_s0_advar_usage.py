"""
Quick Example: Using S0 as ADVar to Compute Gamma via Edge-Pushing

This demonstrates the new capability where S0 is an ADVar, allowing
Gamma (∂²V/∂S0²) to be computed directly via automatic differentiation
instead of finite differences.
"""

import numpy as np
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def main():
    print("="*80)
    print("Example: S0 as ADVar - Computing Gamma via Edge-Pushing")
    print("="*80)

    # Option parameters
    S0 = 100.0      # Spot price
    K = 100.0       # Strike (ATM)
    T = 1.0         # 1 year to maturity
    r = 0.05        # 5% risk-free rate
    sigma = 0.2     # 20% volatility

    # Grid parameters
    M = 51          # Spatial points
    N = 50          # Time steps (base)

    print(f"\nOption: ATM Call")
    print(f"  S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"  Grid: M={M}, N={N}")

    # Create PDE solver
    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    # Compute Greeks via Edge-Pushing
    print(f"\n{'─'*80}")
    print("Computing Greeks via AAD + Edge-Pushing...")
    print('─'*80)

    result = solver.solve_pde_with_aad(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,  # ← Key: compute full Hessian
        verbose=True
    )

    print(f"\nResults:")
    print(f"  Price = {result['price']:.10f}")
    print(f"  Delta = {result['delta']:.10f}  ← ∂V/∂S0 via AAD")
    print(f"  Gamma = {result['gamma']:.10f}  ← ∂²V/∂S0² via Edge-Pushing!")
    print(f"  Vega  = {result['vega']:.10f}  ← ∂V/∂σ via AAD")
    print(f"  Vanna = {result['vanna']:.10f}  ← ∂²V/∂S0∂σ via Edge-Pushing")
    print(f"  Volga = {result['volga']:.10f}  ← ∂²V/∂σ² via Edge-Pushing")

    # Hessian matrix
    print(f"\nHessian Matrix (2×2):")
    H = result['hessian']
    print(f"  H = [[{H[0,0]:12.10f}, {H[0,1]:12.10f}],")
    print(f"       [{H[1,0]:12.10f}, {H[1,1]:12.10f}]]")
    print(f"\nwhere:")
    print(f"  H[0,0] = Gamma (∂²V/∂S0²)")
    print(f"  H[0,1] = H[1,0] = Vanna (∂²V/∂S0∂σ)")
    print(f"  H[1,1] = Volga (∂²V/∂σ²)")

    # Compare with BSM analytical
    print(f"\n{'─'*80}")
    print("Comparison with BSM Analytical Formula:")
    print('─'*80)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    price_bsm = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    delta_bsm = norm.cdf(d1)
    gamma_bsm = norm.pdf(d1) / (S0 * sigma * sqrt_T)
    vega_bsm = S0 * norm.pdf(d1) * sqrt_T

    print(f"\nBSM Analytical:")
    print(f"  Price = {price_bsm:.10f}")
    print(f"  Delta = {delta_bsm:.10f}")
    print(f"  Gamma = {gamma_bsm:.10f}")
    print(f"  Vega  = {vega_bsm:.10f}")

    print(f"\nErrors:")
    print(f"  Price: {abs(result['price'] - price_bsm)/price_bsm*100:.2f}%")
    print(f"  Delta: {abs(result['delta'] - delta_bsm)/delta_bsm*100:.2f}%")
    print(f"  Gamma: {abs(result['gamma'] - gamma_bsm)/gamma_bsm*100:.2f}%")
    print(f"  Vega:  {abs(result['vega'] - vega_bsm)/vega_bsm*100:.2f}%")

    print(f"\n{'='*80}")
    print("✅ Successfully computed Gamma via Edge-Pushing on S0!")
    print("='*80")
    print("\nKey Achievement:")
    print("  • S0 is now an ADVar (in computation graph)")
    print("  • Cubic Hermite interpolation (C² continuous)")
    print("  • Gamma = ∂²V/∂S0² computed via AD (not FD)")
    print("  • Full 2×2 Hessian matrix in one call")
    print("="*80)


if __name__ == "__main__":
    main()
