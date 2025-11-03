#!/usr/bin/env python3
"""
Quick Test of Comprehensive Framework
Tests one scenario to verify all methods work correctly
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from comprehensive_test_framework import ComprehensiveTestFramework

def main():
    """Run quick test"""
    print("="*80)
    print("QUICK TEST - Comprehensive Framework")
    print("="*80)

    framework = ComprehensiveTestFramework()

    # Test single ATM scenario
    test_params = {
        'scenario': 'Quick_Test_ATM',
        'S0': 100.0,
        'K': 100.0,
        'T': 1.0,
        'r': 0.05,
        'sigma': 0.2,
        'M': 151,
        'N': 150
    }

    print("\nRunning single test scenario...")
    result = framework.test_single_scenario(test_params)

    print("\n" + "="*80)
    print("QUICK TEST RESULTS")
    print("="*80)

    print("\nPrice Comparison:")
    print(f"  Analytical:    {result['analytical_price']:.6f}")
    print(f"  Bumping:       {result['bumping_price']:.6f}")
    print(f"  Double-AAD:    {result['double_aad_price']:.6f}")
    print(f"  Edge-Pushing:  {result['edge_pushing_price']:.6f}")

    print("\nDelta Comparison:")
    print(f"  Analytical:    {result['analytical_delta']:.6f}")
    print(f"  Bumping:       {result['bumping_delta']:.6f} (err: {result['bumping_delta_err']:.2f}%)")
    print(f"  Double-AAD:    {result['double_aad_delta']:.6f} (err: {result['double_aad_delta_err']:.2f}%)")
    print(f"  Edge-Pushing:  {result['edge_pushing_delta']:.6f} (err: {result['edge_pushing_delta_err']:.2f}%)")

    print("\nGamma Comparison:")
    print(f"  Analytical:    {result['analytical_gamma']:.6f}")
    print(f"  Bumping:       {result['bumping_gamma']:.6f} (err: {result['bumping_gamma_err']:.2f}%)")
    print(f"  Double-AAD:    {result['double_aad_gamma']:.6f} (err: {result['double_aad_gamma_err']:.2f}%)")
    print(f"  Edge-Pushing:  {result['edge_pushing_gamma']:.6f} (err: {result['edge_pushing_gamma_err']:.2f}%)")

    print("\nComputation Time:")
    print(f"  Analytical:    {result['analytical_time_ms']:.3f} ms")
    print(f"  Bumping:       {result['bumping_time_ms']:.3f} ms")
    print(f"  Double-AAD:    {result['double_aad_time_ms']:.3f} ms")
    print(f"  Edge-Pushing:  {result['edge_pushing_time_ms']:.3f} ms")

    print("\nAAD Graph Statistics:")
    if framework.graph_stats:
        graph = framework.graph_stats[0]
        print(f"  Jacobian: {graph['jacobian_nodes']} nodes, {graph['jacobian_edges']} edges")
        print(f"  Hessian:  {graph['hessian_nodes']} nodes, {graph['hessian_edges']} edges")

    print("\n✓ Quick test completed successfully!")

if __name__ == "__main__":
    main()
