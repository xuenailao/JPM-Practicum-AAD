"""
Example: Greeks Computation - Three Methods Comparison

This script demonstrates and compares three approaches to computing option Greeks:

Method 1: PDE + Bumping (Finite Difference)
    - Simple but slow: O(n_params × PDE_solve_time)
    - Numerical errors from epsilon choice
    - Location: handcraft_aad/greeks/second_order_greeks.py

Method 2: PDE + Handcraft Edge-Pushing
    - Fast with manual adjacency: 10-100× speedup
    - Still requires finite difference on Jacobian
    - Location: handcraft_aad/hessian_edge_pushing.py

Method 3: PDE + AAD + Edge-Pushing ⭐ (THIS MODULE)
    - Automatic graph construction
    - One PDE solve + Algorithm 4
    - No manual adjacency analysis needed
    - Location: AADgraph/capriotti_cn_aad_edgepushing.py

"""

import numpy as np
import time
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import (
    CapriottiCNAAD,
    black_scholes_analytical
)


def print_header(title):
    """Print formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_greeks_table(method_name, greeks, bs_greeks, computation_time):
    """Print Greeks comparison table"""
    price_bs, delta_bs, gamma_bs, vega_bs = bs_greeks

    print(f"\n{'Greek':<12} | {method_name:<15} | {'Analytical BS':<15} | {'Error':<15}")
    print("-" * 70)
    print(f"{'Price':<12} | ${greeks['price']:<14.6f} | ${price_bs:<14.6f} | {abs(greeks['price'] - price_bs):<14.2e}")
    print(f"{'Delta':<12} | {greeks['delta']:<15.6f} | {delta_bs:<15.6f} | {abs(greeks['delta'] - delta_bs):<14.2e}")
    print(f"{'Gamma':<12} | {greeks['gamma']:<15.6f} | {gamma_bs:<15.6f} | {abs(greeks['gamma'] - gamma_bs):<14.2e}")
    print(f"{'Vega':<12} | {greeks['vega']:<15.6f} | {vega_bs:<15.6f} | {abs(greeks['vega'] - vega_bs):<14.2e}")

    # Vanna and Volga (no analytical comparison for now)
    print(f"{'Vanna':<12} | {greeks['vanna']:<15.6f} | {'N/A':<15} | {'-':<15}")
    print(f"{'Volga':<12} | {greeks['volga']:<15.6f} | {'N/A':<15} | {'-':<15}")

    print(f"\n{'Time (ms)':<12} | {computation_time:<15.2f}")


def method3_aad_edge_pushing(M, N, sigma):
    """
    Method 3: AAD + Edge-Pushing (AUTOMATIC)

    Workflow:
    --------
    1. Model σ as M-1 ADVar parameters (though constant)
    2. Solve PDE with ADVar → automatic graph construction
    3. Algorithm 4 extracts sparse Hessian
    4. Finite difference on S for Delta/Gamma
    5. Finite difference on Vega for Vanna

    Advantages:
    - ✅ Fully automatic (no manual adjacency)
    - ✅ One PDE solve + one Algorithm 4 call
    - ✅ Sparse Hessian extracted automatically
    - ✅ High numerical accuracy (no FD on Jacobian)

    Limitations:
    - ⚠️  Graph overhead (memory + time)
    - ⚠️  Grid size limited (M=20-100 recommended)
    """
    print_header(f"Method 3: AAD + Edge-Pushing (M={M}, N={N})")

    solver = CapriottiCNAAD(M=M, N=N, phi=0.5)

    print(f"Parameters:")
    print(f"  S0 = {solver.S0}, K = {solver.K}, T = {solver.T}, r = {solver.r}")
    print(f"  σ = {sigma}")
    print(f"  Grid: {M}×{N}")
    print(f"  Interior points: {M-2}")
    print(f"  Sigma parameters: {M-1} (constant, but modeled as ADVars)")

    # Compute Greeks
    print("\n🚀 Computing Greeks with AAD + Algorithm 4...")
    t_start = time.perf_counter()
    greeks = solver.compute_greeks_aad(sigma_value=sigma, eps_S=0.01)
    t_total = (time.perf_counter() - t_start) * 1000

    # Analytical Greeks
    price_bs, delta_bs, gamma_bs, vega_bs = black_scholes_analytical(
        solver.S0, solver.K, solver.T, solver.r, sigma
    )

    # Print results
    print_greeks_table("AAD+EP", greeks, (price_bs, delta_bs, gamma_bs, vega_bs), t_total)

    # Additional statistics
    print(f"\n📊 Computational Graph Statistics:")
    print(f"  Tape nodes: {greeks['n_tape_nodes']}")
    print(f"  PDE solves: {greeks['n_pde_solves']} (1 base + 2 Delta/Gamma + 2 Vanna)")

    print(f"\n📐 Hessian Statistics:")
    stats = greeks['hessian_stats']
    print(f"  Shape: {stats['shape']}")
    print(f"  Non-zero entries: {stats['nnz']} / {stats['total']}")
    print(f"  Sparsity: {stats['sparsity']*100:.1f}%")
    print(f"  Avg non-zero per row: {stats['avg_row_nnz']:.1f}")

    # Visualize Hessian pattern (if small enough)
    if M <= 30:
        print(f"\n🔍 Hessian Sparsity Pattern (first 10×10 block):")
        hess = greeks['hessian']
        display_size = min(10, hess.shape[0])
        pattern = np.where(np.abs(hess[:display_size, :display_size]) > 1e-10, '■', '·')
        for row in pattern:
            print("  " + " ".join(row))

    return greeks, (price_bs, delta_bs, gamma_bs, vega_bs)


def compute_vanna_volga_bs_approximation(S, K, T, r, sigma):
    """
    Approximate Vanna and Volga using Black-Scholes formulas.

    These are derived from the analytical Greeks formulas.
    """
    from scipy.stats import norm
    import numpy as np

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # Vega
    vega = S * norm.pdf(d1) * sqrt_T

    # Vanna = ∂Δ/∂σ = ∂²V/∂S∂σ
    vanna = -norm.pdf(d1) * d2 / sigma

    # Volga (Vomma) = ∂Vega/∂σ = ∂²V/∂σ²
    volga = vega * d1 * d2 / sigma

    return vanna, volga


def main():
    """Run comprehensive Greeks computation comparison"""

    print("=" * 80)
    print("  GREEKS COMPUTATION: AAD + EDGE-PUSHING DEMONSTRATION")
    print("=" * 80)
    print()
    print("This example demonstrates Method 3: Automatic graph construction")
    print("with Algorithm 4 for sparse Hessian extraction.")
    print()

    # Test parameters
    sigma = 0.2

    # Test Case 1: Small grid (fast, educational)
    print("\n" + "▶" * 40)
    print("TEST CASE 1: Small Grid (M=20, N=20) - Fast Demo")
    print("▶" * 40)

    greeks_small, bs_greeks = method3_aad_edge_pushing(M=20, N=20, sigma=sigma)

    # Test Case 2: Medium grid (balanced)
    print("\n\n" + "▶" * 40)
    print("TEST CASE 2: Medium Grid (M=50, N=50) - Production Quality")
    print("▶" * 40)

    greeks_medium, _ = method3_aad_edge_pushing(M=50, N=50, sigma=sigma)

    # Analytical Vanna/Volga approximation
    print_header("Analytical Vanna/Volga (Approximation)")
    vanna_bs, volga_bs = compute_vanna_volga_bs_approximation(100, 100, 1.0, 0.05, sigma)
    print(f"Vanna (BS approx): {vanna_bs:.6f}")
    print(f"Volga (BS approx): {volga_bs:.6f}")

    print(f"\nComparison with AAD results:")
    print(f"  Vanna error (M=20): {abs(greeks_small['vanna'] - vanna_bs):.2e}")
    print(f"  Vanna error (M=50): {abs(greeks_medium['vanna'] - vanna_bs):.2e}")
    print(f"  Volga error (M=20): {abs(greeks_small['volga'] - volga_bs):.2e}")
    print(f"  Volga error (M=50): {abs(greeks_medium['volga'] - volga_bs):.2e}")

    # Summary
    print_header("SUMMARY: Key Advantages of AAD + Edge-Pushing")

    print("✅ AUTOMATIC: No manual adjacency analysis needed")
    print("   → Algorithm 4 extracts sparsity from computational graph")
    print()
    print("✅ ACCURATE: No finite difference on Jacobian")
    print("   → Hessian computed via reverse-mode AD")
    print()
    print("✅ EFFICIENT: Sparse structure exploited automatically")
    print(f"   → Sparsity: {greeks_medium['hessian_stats']['sparsity']*100:.1f}%")
    print(f"   → Only {greeks_medium['hessian_stats']['avg_row_nnz']:.1f} non-zero entries per row")
    print()
    print("✅ ONE-CALL: All Greeks in single method")
    print(f"   → Price, Delta, Gamma, Vega, Vanna, Volga + Hessian")
    print()
    print("📝 TRADE-OFF: Graph overhead limits grid size")
    print(f"   → Recommended: M=20-100")
    print(f"   → Graph nodes (M=50): {greeks_medium['n_tape_nodes']}")
    print()

    print("=" * 80)
    print("  DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. See test_aad_greeks_validation.py for comprehensive tests")
    print("  2. Compare with bumping methods in handcraft_aad/greeks/")
    print("  3. Explore Hessian structure for different option types")


if __name__ == "__main__":
    main()
