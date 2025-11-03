"""
Test: AAD Greeks Validation Against Black-Scholes

This test validates the AAD + Edge-Pushing Greeks computation against:
1. Black-Scholes analytical formulas
2. Numerical stability across different grid sizes
3. Hessian sparsity properties

Test Coverage:
- Price accuracy
- First-order Greeks (Delta, Vega)
- Second-order Greeks (Gamma, Vanna, Volga)
- Hessian structure and sparsity
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import (
    CapriottiCNAAD,
    black_scholes_analytical
)


def compute_bs_second_order_greeks(S, K, T, r, sigma):
    """
    Compute analytical second-order Greeks for Black-Scholes.

    Returns:
        (vanna, volga) where:
        - Vanna = ∂²V/∂S∂σ = -d₂/σ × N'(d₁)
        - Volga = ∂²V/∂σ² = S√T × N'(d₁) × d₁d₂/σ
    """
    from scipy.stats import norm

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Vega (needed for derivatives)
    vega = S * norm.pdf(d1) * sqrt_T

    # Vanna = ∂Δ/∂σ = ∂²V/∂S∂σ
    vanna = -norm.pdf(d1) * d2 / sigma

    # Volga (Vomma) = ∂Vega/∂σ = ∂²V/∂σ²
    volga = vega * d1 * d2 / sigma

    return vanna, volga


def test_price_accuracy(M=50, N=50, sigma=0.2, tolerance=1e-3):
    """
    Test 1: Price Accuracy

    Validates that PDE price matches Black-Scholes analytical price.

    Args:
        M, N: Grid size
        sigma: Volatility
        tolerance: Maximum relative error allowed

    Returns:
        (passed, error, message)
    """
    print("\n" + "="*70)
    print("TEST 1: PRICE ACCURACY")
    print("="*70)

    solver = CapriottiCNAAD(M=M, N=N)

    # Compute Greeks
    greeks = solver.compute_greeks_aad(sigma_value=sigma)
    price_pde = greeks['price']

    # Analytical price
    price_bs, _, _, _ = black_scholes_analytical(solver.S0, solver.K, solver.T, solver.r, sigma)

    # Error
    abs_error = abs(price_pde - price_bs)
    rel_error = abs_error / price_bs

    print(f"PDE Price:        ${price_pde:.6f}")
    print(f"BS Price:         ${price_bs:.6f}")
    print(f"Absolute Error:   {abs_error:.2e}")
    print(f"Relative Error:   {rel_error:.2e}")
    print(f"Tolerance:        {tolerance:.2e}")

    passed = rel_error < tolerance
    status = "✅ PASSED" if passed else "❌ FAILED"
    print(f"\nStatus: {status}")

    return passed, rel_error, "Price accuracy test"


def test_first_order_greeks(M=50, N=50, sigma=0.2, delta_tol=0.01, vega_tol=0.5):
    """
    Test 2: First-Order Greeks (Delta, Vega)

    Validates Delta and Vega against analytical formulas.

    Args:
        M, N: Grid size
        sigma: Volatility
        delta_tol: Absolute tolerance for Delta
        vega_tol: Absolute tolerance for Vega

    Returns:
        (passed, errors, message)
    """
    print("\n" + "="*70)
    print("TEST 2: FIRST-ORDER GREEKS (Delta, Vega)")
    print("="*70)

    solver = CapriottiCNAAD(M=M, N=N)

    # Compute Greeks
    greeks = solver.compute_greeks_aad(sigma_value=sigma, eps_S=0.01)

    # Analytical Greeks
    _, delta_bs, _, vega_bs = black_scholes_analytical(solver.S0, solver.K, solver.T, solver.r, sigma)

    # Errors
    delta_error = abs(greeks['delta'] - delta_bs)
    vega_error = abs(greeks['vega'] - vega_bs)

    print(f"\nDelta:")
    print(f"  AAD:       {greeks['delta']:.6f}")
    print(f"  BS:        {delta_bs:.6f}")
    print(f"  Error:     {delta_error:.4e}")
    print(f"  Tolerance: {delta_tol:.4e}")

    print(f"\nVega:")
    print(f"  AAD:       {greeks['vega']:.6f}")
    print(f"  BS:        {vega_bs:.6f}")
    print(f"  Error:     {vega_error:.4e}")
    print(f"  Tolerance: {vega_tol:.4e}")

    delta_passed = delta_error < delta_tol
    vega_passed = vega_error < vega_tol
    passed = delta_passed and vega_passed

    print(f"\nDelta: {'✅ PASSED' if delta_passed else '❌ FAILED'}")
    print(f"Vega:  {'✅ PASSED' if vega_passed else '❌ FAILED'}")
    print(f"\nOverall: {'✅ PASSED' if passed else '❌ FAILED'}")

    return passed, (delta_error, vega_error), "First-order Greeks test"


def test_second_order_greeks(M=50, N=50, sigma=0.2, gamma_tol=0.001, vanna_tol=0.05, volga_tol=2.0):
    """
    Test 3: Second-Order Greeks (Gamma, Vanna, Volga)

    Validates second-order Greeks against analytical formulas.

    Args:
        M, N: Grid size
        sigma: Volatility
        gamma_tol, vanna_tol, volga_tol: Absolute tolerances

    Returns:
        (passed, errors, message)
    """
    print("\n" + "="*70)
    print("TEST 3: SECOND-ORDER GREEKS (Gamma, Vanna, Volga)")
    print("="*70)

    solver = CapriottiCNAAD(M=M, N=N)

    # Compute Greeks
    greeks = solver.compute_greeks_aad(sigma_value=sigma, eps_S=0.01)

    # Analytical Greeks
    _, _, gamma_bs, _ = black_scholes_analytical(solver.S0, solver.K, solver.T, solver.r, sigma)
    vanna_bs, volga_bs = compute_bs_second_order_greeks(solver.S0, solver.K, solver.T, solver.r, sigma)

    # Errors
    gamma_error = abs(greeks['gamma'] - gamma_bs)
    vanna_error = abs(greeks['vanna'] - vanna_bs)
    volga_error = abs(greeks['volga'] - volga_bs)

    print(f"\nGamma:")
    print(f"  AAD:       {greeks['gamma']:.6f}")
    print(f"  BS:        {gamma_bs:.6f}")
    print(f"  Error:     {gamma_error:.4e}")
    print(f"  Tolerance: {gamma_tol:.4e}")

    print(f"\nVanna:")
    print(f"  AAD:       {greeks['vanna']:.6f}")
    print(f"  BS:        {vanna_bs:.6f}")
    print(f"  Error:     {vanna_error:.4e}")
    print(f"  Tolerance: {vanna_tol:.4e}")

    print(f"\nVolga:")
    print(f"  AAD:       {greeks['volga']:.6f}")
    print(f"  BS:        {volga_bs:.6f}")
    print(f"  Error:     {volga_error:.4e}")
    print(f"  Tolerance: {volga_tol:.4e}")

    gamma_passed = gamma_error < gamma_tol
    vanna_passed = vanna_error < vanna_tol
    volga_passed = volga_error < volga_tol
    passed = gamma_passed and vanna_passed and volga_passed

    print(f"\nGamma: {'✅ PASSED' if gamma_passed else '❌ FAILED'}")
    print(f"Vanna: {'✅ PASSED' if vanna_passed else '❌ FAILED'}")
    print(f"Volga: {'✅ PASSED' if volga_passed else '❌ FAILED'}")
    print(f"\nOverall: {'✅ PASSED' if passed else '❌ FAILED'}")

    return passed, (gamma_error, vanna_error, volga_error), "Second-order Greeks test"


def test_hessian_structure(M=30, N=30, sigma=0.2):
    """
    Test 4: Hessian Structure and Sparsity

    Validates:
    1. Hessian is symmetric
    2. Hessian is sparse
    3. Sparsity pattern matches expectations

    Args:
        M, N: Grid size
        sigma: Volatility

    Returns:
        (passed, stats, message)
    """
    print("\n" + "="*70)
    print("TEST 4: HESSIAN STRUCTURE AND SPARSITY")
    print("="*70)

    solver = CapriottiCNAAD(M=M, N=N)

    # Compute Greeks
    greeks = solver.compute_greeks_aad(sigma_value=sigma)
    hessian = greeks['hessian']

    # Test 4.1: Symmetry
    symmetry_error = np.max(np.abs(hessian - hessian.T))
    is_symmetric = symmetry_error < 1e-10

    print(f"\n4.1 Symmetry Test:")
    print(f"  Max |H - H^T|: {symmetry_error:.4e}")
    print(f"  Symmetric:     {'✅ YES' if is_symmetric else '❌ NO'}")

    # Test 4.2: Sparsity
    stats = greeks['hessian_stats']
    print(f"\n4.2 Sparsity Analysis:")
    print(f"  Matrix size:        {stats['shape']}")
    print(f"  Total elements:     {stats['total']}")
    print(f"  Non-zero elements:  {stats['nnz']}")
    print(f"  Sparsity:           {stats['sparsity']*100:.2f}%")
    print(f"  Avg NNZ per row:    {stats['avg_row_nnz']:.2f}")

    # Expected sparsity (for PDE, should be very sparse)
    expected_min_sparsity = 0.5  # At least 50% sparse
    is_sparse = stats['sparsity'] > expected_min_sparsity

    print(f"\n  Expected sparsity:  >{expected_min_sparsity*100:.0f}%")
    print(f"  Sparse enough:      {'✅ YES' if is_sparse else '❌ NO'}")

    # Test 4.3: Visualize pattern (if small)
    if M <= 30:
        print(f"\n4.3 Sparsity Pattern (first 15×15):")
        display_size = min(15, hessian.shape[0])
        pattern = np.where(np.abs(hessian[:display_size, :display_size]) > 1e-10, '■', '·')
        for i, row in enumerate(pattern):
            print(f"  {i:2d} │ " + " ".join(row))

    passed = is_symmetric and is_sparse
    print(f"\nOverall: {'✅ PASSED' if passed else '❌ FAILED'}")

    return passed, stats, "Hessian structure test"


def test_grid_convergence(sigma=0.2):
    """
    Test 5: Grid Convergence

    Tests that Greeks converge as grid is refined.

    Args:
        sigma: Volatility

    Returns:
        (passed, convergence_data, message)
    """
    print("\n" + "="*70)
    print("TEST 5: GRID CONVERGENCE")
    print("="*70)

    grid_sizes = [(20, 20), (30, 30), (50, 50)]
    results = []

    # Analytical reference
    price_bs, delta_bs, gamma_bs, vega_bs = black_scholes_analytical(100, 100, 1.0, 0.05, sigma)
    vanna_bs, volga_bs = compute_bs_second_order_greeks(100, 100, 1.0, 0.05, sigma)

    print(f"\nTesting convergence across grid sizes: {grid_sizes}")
    print(f"\n{'Grid':<12} | {'Price Err':<12} | {'Delta Err':<12} | {'Gamma Err':<12}")
    print("-" * 55)

    for M, N in grid_sizes:
        solver = CapriottiCNAAD(M=M, N=N)
        greeks = solver.compute_greeks_aad(sigma_value=sigma, eps_S=0.01)

        price_err = abs(greeks['price'] - price_bs)
        delta_err = abs(greeks['delta'] - delta_bs)
        gamma_err = abs(greeks['gamma'] - gamma_bs)

        results.append({
            'grid': (M, N),
            'price_err': price_err,
            'delta_err': delta_err,
            'gamma_err': gamma_err
        })

        print(f"{M}×{N:<10} | {price_err:<12.4e} | {delta_err:<12.4e} | {gamma_err:<12.4e}")

    # Check if errors decrease with finer grids
    price_converging = results[0]['price_err'] > results[-1]['price_err']
    delta_converging = results[0]['delta_err'] > results[-1]['delta_err']

    print(f"\nConvergence Analysis:")
    print(f"  Price converging: {'✅ YES' if price_converging else '❌ NO'}")
    print(f"  Delta converging: {'✅ YES' if delta_converging else '❌ NO'}")

    passed = price_converging and delta_converging
    print(f"\nOverall: {'✅ PASSED' if passed else '❌ FAILED'}")

    return passed, results, "Grid convergence test"


def run_all_tests():
    """Run comprehensive test suite"""

    print("="*80)
    print("  AAD + EDGE-PUSHING GREEKS VALIDATION TEST SUITE")
    print("="*80)
    print()
    print("Testing Crank-Nicolson PDE with AAD + Algorithm 4 for Greeks computation")
    print()

    results = []

    # Test 1: Price Accuracy
    try:
        passed, error, msg = test_price_accuracy(M=50, N=50)
        results.append(('Test 1', msg, passed))
    except Exception as e:
        print(f"❌ Test 1 FAILED with exception: {e}")
        results.append(('Test 1', 'Price accuracy', False))

    # Test 2: First-Order Greeks
    try:
        passed, errors, msg = test_first_order_greeks(M=50, N=50)
        results.append(('Test 2', msg, passed))
    except Exception as e:
        print(f"❌ Test 2 FAILED with exception: {e}")
        results.append(('Test 2', 'First-order Greeks', False))

    # Test 3: Second-Order Greeks
    try:
        passed, errors, msg = test_second_order_greeks(M=50, N=50)
        results.append(('Test 3', msg, passed))
    except Exception as e:
        print(f"❌ Test 3 FAILED with exception: {e}")
        results.append(('Test 3', 'Second-order Greeks', False))

    # Test 4: Hessian Structure
    try:
        passed, stats, msg = test_hessian_structure(M=30, N=30)
        results.append(('Test 4', msg, passed))
    except Exception as e:
        print(f"❌ Test 4 FAILED with exception: {e}")
        results.append(('Test 4', 'Hessian structure', False))

    # Test 5: Grid Convergence
    try:
        passed, conv_data, msg = test_grid_convergence()
        results.append(('Test 5', msg, passed))
    except Exception as e:
        print(f"❌ Test 5 FAILED with exception: {e}")
        results.append(('Test 5', 'Grid convergence', False))

    # Summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)

    total_tests = len(results)
    passed_tests = sum(1 for _, _, passed in results if passed)

    for test_id, desc, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_id:<10} | {desc:<40} | {status}")

    print(f"\n{'Total':<10} | {passed_tests}/{total_tests} tests passed")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! AAD + Edge-Pushing Greeks computation validated.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed. Review output above.")

    print("="*80)

    return passed_tests == total_tests


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
