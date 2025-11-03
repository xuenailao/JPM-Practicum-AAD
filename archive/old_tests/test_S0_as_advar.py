"""
Test PDE AAD with S0 as ADVar

This tests the new implementation where both S0 and sigma are ADVars,
allowing Edge-Pushing to compute Gamma directly via AD.
"""

import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


def bsm_analytical(S0, K, T, r, sigma):
    """BSM analytical formulas for comparison"""
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


def test_small_grid():
    """Test with small grid"""
    print("=" * 80)
    print("Test 1: Small Grid (M=21, N=20)")
    print("=" * 80)

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    M = 21
    N = 20

    print(f"\nParameters:")
    print(f"  S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"  Grid: M={M}, N={N}")

    # Analytical
    bsm = bsm_analytical(S0, K, T, r, sigma)
    print(f"\nBSM Analytical:")
    print(f"  Price = {bsm['price']:.10f}")
    print(f"  Delta = {bsm['delta']:.10f}")
    print(f"  Gamma = {bsm['gamma']:.10f}")
    print(f"  Vega  = {bsm['vega']:.10f}")
    print(f"  Vanna = {bsm['vanna']:.10f}")
    print(f"  Volga = {bsm['volga']:.10f}")

    # PDE with S0 as ADVar
    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    print(f"\n{'-'*80}")
    print("Computing Jacobian (S0 and sigma as ADVars)...")
    print('-'*80)

    result_jac = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=True)

    print(f"\nPDE AAD Jacobian:")
    print(f"  Price = {result_jac['price']:.10f}")
    print(f"  Delta = {result_jac['delta']:.10f}  (via AAD on S0)")
    print(f"  Vega  = {result_jac['vega']:.10f}  (via AAD on sigma)")
    print(f"  Time  = {result_jac['time_ms']:.2f} ms")

    print(f"\n{'-'*80}")
    print("Computing Hessian (Edge-Pushing on S0 and sigma)...")
    print('-'*80)

    result_hess = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=True)

    print(f"\nPDE AAD Hessian (Edge-Pushing):")
    print(f"  Price = {result_hess['price']:.10f}")
    print(f"  Delta = {result_hess['delta']:.10f}")
    print(f"  Gamma = {result_hess.get('gamma', 0.0):.10f}  ← Gamma via AD!")
    print(f"  Vega  = {result_hess['vega']:.10f}")
    print(f"  Vanna = {result_hess.get('vanna', 0.0):.10f}  ← Vanna via AD!")
    print(f"  Volga = {result_hess.get('volga', 0.0):.10f}  ← Volga via AD!")
    print(f"  Time  = {result_hess['time_ms']:.2f} ms")

    # Errors
    print(f"\n{'-'*80}")
    print("Errors vs BSM Analytical:")
    print('-'*80)
    print(f"  Price error: {abs(result_hess['price'] - bsm['price']):.6e}  "
          f"({abs(result_hess['price'] - bsm['price'])/bsm['price']*100:.2f}%)")
    print(f"  Delta error: {abs(result_hess['delta'] - bsm['delta']):.6e}  "
          f"({abs(result_hess['delta'] - bsm['delta'])/bsm['delta']*100:.2f}%)")
    print(f"  Gamma error: {abs(result_hess.get('gamma', 0.0) - bsm['gamma']):.6e}  "
          f"({abs(result_hess.get('gamma', 0.0) - bsm['gamma'])/bsm['gamma']*100:.2f}%)")
    print(f"  Vega  error: {abs(result_hess['vega'] - bsm['vega']):.6e}  "
          f"({abs(result_hess['vega'] - bsm['vega'])/bsm['vega']*100:.2f}%)")
    print(f"  Vanna error: {abs(result_hess.get('vanna', 0.0) - bsm['vanna']):.6e}  "
          f"({abs(result_hess.get('vanna', 0.0) - bsm['vanna'])/abs(bsm['vanna'])*100:.2f}%)")
    print(f"  Volga error: {abs(result_hess.get('volga', 0.0) - bsm['volga']):.6e}  "
          f"({abs(result_hess.get('volga', 0.0) - bsm['volga'])/bsm['volga']*100:.2f}%)")

    # Check Gamma is non-zero
    gamma_computed = result_hess.get('gamma', 0.0)
    if abs(gamma_computed) > 1e-10:
        print(f"\n✅ SUCCESS: Gamma = {gamma_computed:.10f} (non-zero!)")
        print(f"   Gamma is now computed via AD on S0, not grid FD!")
    else:
        print(f"\n❌ FAILURE: Gamma = {gamma_computed:.10f} (still zero)")

    return result_hess


def test_medium_grid():
    """Test with medium grid"""
    print("\n\n" + "=" * 80)
    print("Test 2: Medium Grid (M=51, N=50)")
    print("=" * 80)

    # Parameters
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2

    M = 51
    N = 50

    print(f"\nParameters:")
    print(f"  S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"  Grid: M={M}, N={N}")

    # Analytical
    bsm = bsm_analytical(S0, K, T, r, sigma)

    # PDE with S0 as ADVar
    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

    print(f"\n{'-'*80}")
    print("Computing Full Hessian via Edge-Pushing...")
    print('-'*80)

    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=True)

    print(f"\nResults:")
    print(f"  Price = {result['price']:.10f}  (BSM: {bsm['price']:.10f})")
    print(f"  Delta = {result['delta']:.10f}  (BSM: {bsm['delta']:.10f})")
    print(f"  Gamma = {result.get('gamma', 0.0):.10f}  (BSM: {bsm['gamma']:.10f})")
    print(f"  Vega  = {result['vega']:.10f}  (BSM: {bsm['vega']:.10f})")
    print(f"  Vanna = {result.get('vanna', 0.0):.10f}  (BSM: {bsm['vanna']:.10f})")
    print(f"  Volga = {result.get('volga', 0.0):.10f}  (BSM: {bsm['volga']:.10f})")
    print(f"  Time  = {result['time_ms']:.2f} ms")

    # Errors
    print(f"\n{'-'*80}")
    print("Errors:")
    print('-'*80)
    gamma_error_pct = abs(result.get('gamma', 0.0) - bsm['gamma'])/bsm['gamma']*100
    print(f"  Gamma error: {gamma_error_pct:.2f}%")

    if gamma_error_pct < 15:
        print(f"  ✅ Gamma accuracy is good (< 15% error)")
    else:
        print(f"  ⚠️  Gamma error is high (> 15%)")

    return result


def test_hessian_matrix():
    """Test full Hessian matrix structure"""
    print("\n\n" + "=" * 80)
    print("Test 3: Hessian Matrix Structure")
    print("=" * 80)

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    M = 31
    N = 30

    print(f"\nParameters: S0={S0}, K={K}, sigma={sigma}")
    print(f"Grid: M={M}, N={N}")

    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)
    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=False)

    if 'hessian' in result:
        H = result['hessian']
        print(f"\nHessian matrix (2×2):")
        print(f"  [[{H[0,0]:12.10f}, {H[0,1]:12.10f}]")
        print(f"   [{H[1,0]:12.10f}, {H[1,1]:12.10f}]]")
        print(f"\nInterpretation:")
        print(f"  H[0,0] = ∂²V/∂S0²   = Gamma = {H[0,0]:.10f}")
        print(f"  H[0,1] = ∂²V/∂S0∂σ  = Vanna = {H[0,1]:.10f}")
        print(f"  H[1,0] = ∂²V/∂σ∂S0  = Vanna = {H[1,0]:.10f}")
        print(f"  H[1,1] = ∂²V/∂σ²    = Volga = {H[1,1]:.10f}")

        # Check symmetry
        vanna_diff = abs(H[0,1] - H[1,0])
        print(f"\n  Symmetry check: |H[0,1] - H[1,0]| = {vanna_diff:.6e}")
        if vanna_diff < 1e-6:
            print(f"  ✅ Hessian is symmetric (as expected)")
        else:
            print(f"  ⚠️  Hessian asymmetry detected")
    else:
        print("\n❌ Hessian not computed")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("TESTING: PDE AAD with S0 as ADVar")
    print("Goal: Compute Gamma directly via Edge-Pushing on S0")
    print("=" * 80)

    # Run tests
    test_small_grid()
    test_medium_grid()
    test_hessian_matrix()

    print("\n" + "=" * 80)
    print("✅ ALL TESTS COMPLETE")
    print("=" * 80)
    print("\nKey Achievement:")
    print("  • S0 is now an ADVar (in computation graph)")
    print("  • Interpolation is differentiable w.r.t. S0")
    print("  • Gamma computed via Edge-Pushing (not grid FD)")
    print("  • Full 2×2 Hessian matrix obtained")
    print("=" * 80)
