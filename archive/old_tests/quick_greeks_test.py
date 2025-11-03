"""
快速Greeks测试
测试5种方法并打印计算图
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.unified_greeks_interface import UnifiedGreeksCalculator
from aad_edge_pushing.aad.core.graph_utils import print_graph_summary, print_computation_graph
from aad_edge_pushing.aad.core.tape import global_tape


def main():
    print("="*80)
    print(" "*25 + "QUICK GREEKS TEST")
    print("="*80)

    # Test parameters
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    M, N = 51, 50

    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Grid: M={M}, N={N}")

    # Initialize calculator
    calc = UnifiedGreeksCalculator(M=M, N=N)

    # Test methods
    methods = ['analytical', 'bumping', 'aad_bumping', 'double_aad', 'edge_pushing']

    print("\n" + "="*80)
    print("TEST 1: GREEKS COMPUTATION")
    print("="*80)

    print(f"\n{'Method':<20} {'Price':<12} {'Delta':<12} {'Gamma':<12} {'Time (ms)':<12}")
    print("-"*80)

    results = {}
    for method in methods:
        global_tape.reset()

        result = calc.compute_greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma,
            method=method,
            verbose=False,
            track_graph=True
        )

        results[method] = result

        print(f"{result['method']:<20} {result['price']:>10.4f}  "
              f"{result['greeks']['delta']:>10.4f}  "
              f"{result['greeks']['gamma']:>10.6f}  "
              f"{result['time_ms']:>11.1f}")

    # Error comparison
    print("\n" + "="*80)
    print("TEST 2: ERROR ANALYSIS (vs. Analytical)")
    print("="*80)

    ana_result = results['analytical']

    print(f"\n{'Method':<20} {'Price Err%':<15} {'Delta Err%':<15} {'Gamma Err%':<15} {'Volga Err%':<15}")
    print("-"*80)

    for method in ['bumping', 'aad_bumping', 'double_aad', 'edge_pushing']:
        result = results[method]

        price_err = abs(result['price'] - ana_result['price']) / ana_result['price'] * 100
        delta_err = abs(result['greeks']['delta'] - ana_result['greeks']['delta']) / ana_result['greeks']['delta'] * 100
        gamma_err = abs(result['greeks']['gamma'] - ana_result['greeks']['gamma']) / ana_result['greeks']['gamma'] * 100
        volga_err = abs(result['greeks']['volga'] - ana_result['greeks']['volga']) / abs(ana_result['greeks']['volga']) * 100

        print(f"{result['method']:<20} {price_err:>13.2f}%  {delta_err:>13.2f}%  {gamma_err:>13.2f}%  {volga_err:>13.2f}%")

    # Computation graph
    print("\n" + "="*80)
    print("TEST 3: COMPUTATION GRAPH STATISTICS")
    print("="*80)

    print(f"\n{'Method':<20} {'Nodes':<15} {'Edges':<15} {'Max Fan-in':<15} {'PDE Solves':<15}")
    print("-"*80)

    for method in methods:
        result = results[method]
        graph_info = result.get('graph_info', {})

        nodes = graph_info.get('nodes', 0)
        edges = graph_info.get('edges', 0)
        max_fan_in = graph_info.get('max_fan_in', 0)
        pde_solves = result.get('n_pde_solves', 0)

        print(f"{result['method']:<20} {str(nodes):<15} {str(edges):<15} {str(max_fan_in):<15} {pde_solves:<15}")

    # Detailed graph for Edge-Pushing
    print("\n" + "="*80)
    print("TEST 4: DETAILED COMPUTATION GRAPH (Edge-Pushing)")
    print("="*80)

    global_tape.reset()

    result = calc.compute_greeks(
        S0=S0, K=K, T=T, r=r, sigma=sigma,
        method='edge_pushing',
        verbose=False,
        track_graph=True
    )

    print_graph_summary(global_tape, detailed=False)
    print_computation_graph(global_tape, max_nodes=20)

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
