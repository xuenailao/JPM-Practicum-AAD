#!/usr/bin/env python3
"""
Comprehensive Testing Framework for AAD Greek Computation Methods

Tests 5 methods:
1. BSM Analytical
2. Bumping (Finite Difference)
3. AAD + Bumping (Double-AAD)
4. Edge-Pushing
5. AAD2 (Original Double-AAD)

Test Scenarios:
- ATM, ITM, OTM options
- Varying volatility (sigma)
- Varying grid sizes (M, N)

Metrics:
- Accuracy: Relative error vs analytical
- Speed: Computation time
- AAD Graph: Node count, types, timing breakdown
"""

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project paths
sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.bsm_analytical import BSMAnalytical
from aad_edge_pushing.pde.bumping_method import DoubleBumpingFixed
from aad_edge_pushing.pde.double_aad_method import DoubleAADFixed
from aad_edge_pushing.pde.edge_pushing_method import EdgePushingMethodFixed
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD


class AADGraphInspector:
    """Inspector for AAD computation graph"""

    @staticmethod
    def inspect_graph(solver: BS_PDE_AAD, S0: float, sigma: float,
                     compute_hessian: bool = False) -> Dict:
        """
        Inspect AAD computation graph and collect statistics

        Returns:
            Dict with graph statistics: node_count, node_types, timing
        """
        from aad_edge_pushing.aad.core.tape import global_tape

        t_start = time.perf_counter()

        # Reset and run computation
        global_tape.reset()
        result = solver.solve_pde_with_aad(
            S0_val=S0,
            sigma_val=sigma,
            compute_hessian=compute_hessian,
            verbose=False
        )

        t_end = time.perf_counter()
        total_time = (t_end - t_start) * 1000.0  # ms

        # Collect graph statistics
        nodes = global_tape.nodes
        node_count = len(nodes)

        # Count node types
        node_types = {}
        for node in nodes:
            op_type = type(node).__name__
            node_types[op_type] = node_types.get(op_type, 0) + 1

        # Count edges
        edge_count = sum(len(node.parents) for node in nodes)

        return {
            'node_count': node_count,
            'edge_count': edge_count,
            'node_types': node_types,
            'total_time_ms': total_time,
            'avg_time_per_node_us': (total_time * 1000.0 / node_count) if node_count > 0 else 0.0
        }


class ComprehensiveTestFramework:
    """Main testing framework"""

    def __init__(self):
        self.results = []
        self.graph_stats = []

    def generate_test_scenarios(self) -> List[Dict]:
        """
        Generate comprehensive test scenarios

        Returns:
            List of test case dictionaries
        """
        scenarios = []

        # Base parameters
        base_K = 100.0
        base_T = 1.0
        base_r = 0.05
        base_sigma = 0.2
        # Use moderate grid for stability (M=51, N=100 causes negative Gamma at high sigma)
        # After testing: M=51, N=100 at σ=0.5 gives NEGATIVE Gamma
        # Safe values: M≥101, N≥200
        fast_M = 101  # Increased from 51
        fast_N = 200  # Increased from 100
        # Use moderate grid for Grid variation baseline
        base_M = 151
        base_N = 150

        # Scenario 1: ATM, ITM, OTM (use fast grid)
        moneyness_cases = [
            {'name': 'ATM', 'S0': 100.0, 'K': 100.0},
            {'name': 'ITM', 'S0': 110.0, 'K': 100.0},
            {'name': 'OTM', 'S0': 90.0, 'K': 100.0}
        ]

        for case in moneyness_cases:
            scenarios.append({
                'scenario': f'Moneyness_{case["name"]}',
                'S0': case['S0'],
                'K': case['K'],
                'T': base_T,
                'r': base_r,
                'sigma': base_sigma,
                'M': fast_M,
                'N': fast_N
            })

        # Scenario 2: Varying sigma (adaptive grid based on volatility)
        sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for sigma in sigma_values:
            # Use larger grid for high volatility (σ ≥ 0.4)
            if sigma >= 0.4:
                sigma_M, sigma_N = 151, 300  # High volatility needs finer grid
            else:
                sigma_M, sigma_N = fast_M, fast_N  # Normal grid

            scenarios.append({
                'scenario': f'Sigma_{sigma:.1f}',
                'S0': base_K,
                'K': base_K,
                'T': base_T,
                'r': base_r,
                'sigma': sigma,
                'M': sigma_M,
                'N': sigma_N
            })

        # Scenario 3: Varying M (spatial grid)
        M_values = [51, 101, 151, 201]
        for M in M_values:
            scenarios.append({
                'scenario': f'Grid_M{M}',
                'S0': base_K,
                'K': base_K,
                'T': base_T,
                'r': base_r,
                'sigma': base_sigma,
                'M': M,
                'N': base_N
            })

        # Scenario 4: Varying N (temporal grid, N >> M)
        N_values = [100, 200, 400]
        for N in N_values:
            scenarios.append({
                'scenario': f'Grid_N{N}',
                'S0': base_K,
                'K': base_K,
                'T': base_T,
                'r': base_r,
                'sigma': base_sigma,
                'M': base_M,
                'N': N
            })

        return scenarios

    def compute_relative_error(self, computed: float, analytical: float) -> float:
        """Compute relative error with handling for near-zero values"""
        if abs(analytical) < 1e-8:
            return abs(computed - analytical)  # absolute error for near-zero
        return abs((computed - analytical) / analytical) * 100.0

    def test_single_scenario(self, params: Dict) -> Dict:
        """
        Test all 5 methods on a single scenario

        Args:
            params: Dictionary with S0, K, T, r, sigma, M, N

        Returns:
            Dictionary with results for all methods
        """
        S0 = params['S0']
        K = params['K']
        T = params['T']
        r = params['r']
        sigma = params['sigma']
        M = params['M']
        N = params['N']
        scenario = params['scenario']

        print(f"\n{'='*80}")
        print(f"Testing Scenario: {scenario}")
        print(f"  S0={S0}, K={K}, T={T}, r={r}, σ={sigma}, M={M}, N={N}")
        print(f"{'='*80}")

        results = {
            'scenario': scenario,
            'S0': S0,
            'K': K,
            'T': T,
            'r': r,
            'sigma': sigma,
            'M': M,
            'N': N
        }

        # Method 1: BSM Analytical (Ground Truth)
        print("\n[1/5] BSM Analytical...")
        analytical = BSMAnalytical()
        t_start = time.perf_counter()
        analytical_greeks = analytical.compute_greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma, cp_flag='C'
        )
        t_end = time.perf_counter()
        analytical_time = (t_end - t_start) * 1000.0

        results['analytical_price'] = analytical_greeks['price']
        results['analytical_delta'] = analytical_greeks['delta']
        results['analytical_gamma'] = analytical_greeks['gamma']
        results['analytical_vega'] = analytical_greeks['vega']
        results['analytical_vanna'] = analytical_greeks['vanna']
        results['analytical_volga'] = analytical_greeks['volga']
        results['analytical_rho'] = analytical_greeks['rho']
        results['analytical_time_ms'] = analytical_time

        print(f"  ✓ Price={analytical_greeks['price']:.6f}, Time={analytical_time:.3f}ms")

        # Method 2: Bumping
        print("\n[2/5] Bumping Method...")
        bumping = DoubleBumpingFixed(M=M, N=N)
        t_start = time.perf_counter()
        bumping_greeks = bumping.compute_greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma
        )
        t_end = time.perf_counter()
        bumping_time = (t_end - t_start) * 1000.0

        results['bumping_price'] = bumping_greeks['price']
        results['bumping_delta'] = bumping_greeks['delta']
        results['bumping_gamma'] = bumping_greeks['gamma']
        results['bumping_vega'] = bumping_greeks['vega']
        results['bumping_vanna'] = bumping_greeks['vanna']
        results['bumping_volga'] = bumping_greeks['volga']
        results['bumping_rho'] = bumping_greeks['rho']
        results['bumping_time_ms'] = bumping_time
        results['bumping_pde_solves'] = bumping_greeks['pde_solves']

        # Errors
        results['bumping_delta_err'] = self.compute_relative_error(
            bumping_greeks['delta'], analytical_greeks['delta']
        )
        results['bumping_gamma_err'] = self.compute_relative_error(
            bumping_greeks['gamma'], analytical_greeks['gamma']
        )
        results['bumping_vega_err'] = self.compute_relative_error(
            bumping_greeks['vega'], analytical_greeks['vega']
        )
        results['bumping_vanna_err'] = self.compute_relative_error(
            bumping_greeks['vanna'], analytical_greeks['vanna']
        )
        results['bumping_volga_err'] = self.compute_relative_error(
            bumping_greeks['volga'], analytical_greeks['volga']
        )
        results['bumping_rho_err'] = self.compute_relative_error(
            bumping_greeks['rho'], analytical_greeks['rho']
        )

        print(f"  ✓ Price={bumping_greeks['price']:.6f}, Time={bumping_time:.3f}ms")
        print(f"    Δ_err={results['bumping_delta_err']:.2f}%, Γ_err={results['bumping_gamma_err']:.2f}%")

        # Method 3: Double-AAD (AAD + Bumping)
        print("\n[3/5] Double-AAD Method...")
        double_aad = DoubleAADFixed(M=M, N=N)
        t_start = time.perf_counter()
        double_aad_greeks = double_aad.compute_greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma
        )
        t_end = time.perf_counter()
        double_aad_time = (t_end - t_start) * 1000.0

        results['double_aad_price'] = double_aad_greeks['price']
        results['double_aad_delta'] = double_aad_greeks['delta']
        results['double_aad_gamma'] = double_aad_greeks['gamma']
        results['double_aad_vega'] = double_aad_greeks['vega']
        results['double_aad_vanna'] = double_aad_greeks['vanna']
        results['double_aad_volga'] = double_aad_greeks['volga']
        results['double_aad_rho'] = double_aad_greeks['rho']
        results['double_aad_time_ms'] = double_aad_time
        results['double_aad_pde_solves'] = double_aad_greeks['pde_solves']

        # Errors
        results['double_aad_delta_err'] = self.compute_relative_error(
            double_aad_greeks['delta'], analytical_greeks['delta']
        )
        results['double_aad_gamma_err'] = self.compute_relative_error(
            double_aad_greeks['gamma'], analytical_greeks['gamma']
        )
        results['double_aad_vega_err'] = self.compute_relative_error(
            double_aad_greeks['vega'], analytical_greeks['vega']
        )
        results['double_aad_vanna_err'] = self.compute_relative_error(
            double_aad_greeks['vanna'], analytical_greeks['vanna']
        )
        results['double_aad_volga_err'] = self.compute_relative_error(
            double_aad_greeks['volga'], analytical_greeks['volga']
        )
        results['double_aad_rho_err'] = self.compute_relative_error(
            double_aad_greeks['rho'], analytical_greeks['rho']
        )

        print(f"  ✓ Price={double_aad_greeks['price']:.6f}, Time={double_aad_time:.3f}ms")
        print(f"    Δ_err={results['double_aad_delta_err']:.2f}%, Γ_err={results['double_aad_gamma_err']:.2f}%")

        # Method 4: Edge-Pushing
        print("\n[4/5] Edge-Pushing Method...")
        edge_pushing = EdgePushingMethodFixed(M=M, N=N)
        t_start = time.perf_counter()
        edge_pushing_greeks = edge_pushing.compute_greeks(
            S0=S0, K=K, T=T, r=r, sigma=sigma, compute_hessian=True
        )
        t_end = time.perf_counter()
        edge_pushing_time = (t_end - t_start) * 1000.0

        results['edge_pushing_price'] = edge_pushing_greeks['price']
        results['edge_pushing_delta'] = edge_pushing_greeks['delta']
        results['edge_pushing_gamma'] = edge_pushing_greeks['gamma']
        results['edge_pushing_vega'] = edge_pushing_greeks['vega']
        results['edge_pushing_vanna'] = edge_pushing_greeks['vanna']
        results['edge_pushing_volga'] = edge_pushing_greeks['volga']
        results['edge_pushing_rho'] = edge_pushing_greeks['rho']
        results['edge_pushing_time_ms'] = edge_pushing_time
        results['edge_pushing_pde_solves'] = edge_pushing_greeks['pde_solves']

        # Errors
        results['edge_pushing_delta_err'] = self.compute_relative_error(
            edge_pushing_greeks['delta'], analytical_greeks['delta']
        )
        results['edge_pushing_gamma_err'] = self.compute_relative_error(
            edge_pushing_greeks['gamma'], analytical_greeks['gamma']
        )
        results['edge_pushing_vega_err'] = self.compute_relative_error(
            edge_pushing_greeks['vega'], analytical_greeks['vega']
        )
        results['edge_pushing_vanna_err'] = self.compute_relative_error(
            edge_pushing_greeks['vanna'], analytical_greeks['vanna']
        )
        results['edge_pushing_volga_err'] = self.compute_relative_error(
            edge_pushing_greeks['volga'], analytical_greeks['volga']
        )
        results['edge_pushing_rho_err'] = self.compute_relative_error(
            edge_pushing_greeks['rho'], analytical_greeks['rho']
        )

        print(f"  ✓ Price={edge_pushing_greeks['price']:.6f}, Time={edge_pushing_time:.3f}ms")
        print(f"    Δ_err={results['edge_pushing_delta_err']:.2f}%, Γ_err={results['edge_pushing_gamma_err']:.2f}%")

        # Method 5: Inspect AAD Graph for Edge-Pushing
        print("\n[5/5] AAD Graph Inspection...")
        solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N)

        # Jacobian graph
        print("  Inspecting Jacobian graph...")
        jacobian_graph = AADGraphInspector.inspect_graph(
            solver, S0, sigma, compute_hessian=False
        )

        # Hessian graph
        print("  Inspecting Hessian graph...")
        hessian_graph = AADGraphInspector.inspect_graph(
            solver, S0, sigma, compute_hessian=True
        )

        # Store graph statistics
        graph_stat = {
            'scenario': scenario,
            'M': M,
            'N': N,
            'jacobian_nodes': jacobian_graph['node_count'],
            'jacobian_edges': jacobian_graph['edge_count'],
            'jacobian_time_ms': jacobian_graph['total_time_ms'],
            'jacobian_time_per_node_us': jacobian_graph['avg_time_per_node_us'],
            'hessian_nodes': hessian_graph['node_count'],
            'hessian_edges': hessian_graph['edge_count'],
            'hessian_time_ms': hessian_graph['total_time_ms'],
            'hessian_time_per_node_us': hessian_graph['avg_time_per_node_us']
        }

        # Store node types
        for op_type, count in jacobian_graph['node_types'].items():
            graph_stat[f'jacobian_{op_type}'] = count
        for op_type, count in hessian_graph['node_types'].items():
            graph_stat[f'hessian_{op_type}'] = count

        self.graph_stats.append(graph_stat)

        print(f"  ✓ Jacobian: {jacobian_graph['node_count']} nodes, {jacobian_graph['edge_count']} edges")
        print(f"  ✓ Hessian: {hessian_graph['node_count']} nodes, {hessian_graph['edge_count']} edges")

        return results

    def run_all_tests(self):
        """Run all test scenarios"""
        print("\n" + "="*80)
        print("COMPREHENSIVE AAD GREEK COMPUTATION TEST FRAMEWORK")
        print("="*80)

        scenarios = self.generate_test_scenarios()
        print(f"\nTotal test scenarios: {len(scenarios)}")

        for i, scenario in enumerate(scenarios, 1):
            print(f"\n\n{'#'*80}")
            print(f"Scenario {i}/{len(scenarios)}")
            print(f"{'#'*80}")

            result = self.test_single_scenario(scenario)
            self.results.append(result)

        # Save results
        self.save_results()

        # Print summary
        self.print_summary()

    def save_results(self):
        """Save results to CSV files"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save main results
        df_results = pd.DataFrame(self.results)
        results_file = f"comprehensive_test_results_{timestamp}.csv"
        df_results.to_csv(results_file, index=False)
        print(f"\n✓ Results saved to: {results_file}")

        # Save graph statistics
        if self.graph_stats:
            df_graph = pd.DataFrame(self.graph_stats)
            graph_file = f"aad_graph_statistics_{timestamp}.csv"
            df_graph.to_csv(graph_file, index=False)
            print(f"✓ AAD graph statistics saved to: {graph_file}")

    def print_summary(self):
        """Print summary of test results"""
        df = pd.DataFrame(self.results)

        print("\n\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        # Average errors
        print("\n--- Average Errors (%) ---")
        methods = ['bumping', 'double_aad', 'edge_pushing']
        greeks = ['delta', 'gamma', 'vega', 'vanna', 'volga', 'rho']

        for method in methods:
            print(f"\n{method.upper()}:")
            for greek in greeks:
                col = f'{method}_{greek}_err'
                if col in df.columns:
                    avg_err = df[col].mean()
                    max_err = df[col].max()
                    print(f"  {greek:8s}: avg={avg_err:6.2f}%, max={max_err:6.2f}%")

        # Average computation times
        print("\n--- Average Computation Time (ms) ---")
        for method in ['analytical', 'bumping', 'double_aad', 'edge_pushing']:
            col = f'{method}_time_ms'
            if col in df.columns:
                avg_time = df[col].mean()
                print(f"  {method:15s}: {avg_time:8.2f} ms")

        # Speedup ratios
        print("\n--- Speedup vs Bumping ---")
        bumping_avg = df['bumping_time_ms'].mean()
        for method in ['double_aad', 'edge_pushing']:
            col = f'{method}_time_ms'
            if col in df.columns:
                method_avg = df[col].mean()
                speedup = bumping_avg / method_avg if method_avg > 0 else 0
                print(f"  {method:15s}: {speedup:.2f}x")

        # AAD Graph Summary
        if self.graph_stats:
            df_graph = pd.DataFrame(self.graph_stats)
            print("\n--- AAD Graph Statistics ---")
            print(f"  Avg Jacobian nodes: {df_graph['jacobian_nodes'].mean():.0f}")
            print(f"  Avg Hessian nodes: {df_graph['hessian_nodes'].mean():.0f}")
            print(f"  Avg time per node (Jacobian): {df_graph['jacobian_time_per_node_us'].mean():.2f} μs")
            print(f"  Avg time per node (Hessian): {df_graph['hessian_time_per_node_us'].mean():.2f} μs")


def main():
    """Main entry point"""
    framework = ComprehensiveTestFramework()
    framework.run_all_tests()


if __name__ == "__main__":
    main()
