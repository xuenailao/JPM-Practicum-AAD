"""
Benchmark: Test Hessian computation across different volatility levels.

Purpose: Test how the 5 methods perform under different implied volatility scenarios.

Test different σ values: 10%, 20%, 30%, 40%, 50%
For each σ, compute Hessian and compare against BSM analytical solution.
"""

import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import json

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.methods.bumping2 import Bumping2Method
from aad_edge_pushing.pde.methods.aad_bumping import AADBumpingMethod
from aad_edge_pushing.pde.methods.double_aad import DoubleAADMethod
from aad_edge_pushing.pde.methods.edge_pushing import EdgePushingMethod
from aad_edge_pushing.pde.methods.bsm_analytical import BSMAnalyticalMethod


class VolatilityBenchmark:
    """
    Benchmark Hessian computation across different volatility levels.
    """

    def __init__(self):
        self.methods = {
            'Bumping2': Bumping2Method,
            'AAD+Bumping': AADBumpingMethod,
            'Double-AAD': DoubleAADMethod,
            'Edge-Pushing': EdgePushingMethod,
            'BSM-Analytical': BSMAnalyticalMethod,
        }
        self.results = []
        self.output_file = None
        self.progress_file = None

    def run_benchmark(self, S0: float, K: float, T: float, r: float,
                     volatilities: List[float], grid_sizes: List[Tuple[int, int]],
                     methods_to_test: List[str] = None):
        """
        Run benchmark across different volatility levels.

        Args:
            S0: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            volatilities: List of volatility values to test (e.g., [0.1, 0.2, 0.3, 0.4, 0.5])
            grid_sizes: List of (M, N) grid configurations
            methods_to_test: List of method names (None = all)
        """
        if methods_to_test is None:
            methods_to_test = list(self.methods.keys())

        # Setup output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs('benchmark_results', exist_ok=True)
        self.output_file = f'benchmark_results/vol_benchmark_{timestamp}.txt'
        self.progress_file = f'benchmark_results/vol_progress_{timestamp}.json'

        total_tests = len(volatilities) * len(grid_sizes) * len(methods_to_test)

        # Write header
        with open(self.output_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("VOLATILITY SENSITIVITY BENCHMARK - 5 Methods Comparison\n")
            f.write("=" * 100 + "\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  Option Parameters: S0={S0}, K={K}, T={T:.2f}, r={r:.2%}\n")
            f.write(f"  Volatility levels: {len(volatilities)}\n")
            f.write(f"  Grid sizes: {len(grid_sizes)}\n")
            f.write(f"  Methods: {len(methods_to_test)}\n")
            f.write(f"  Total tests: {total_tests}\n\n")
            f.flush()

        print("=" * 100)
        print("VOLATILITY SENSITIVITY BENCHMARK - 5 Methods Comparison")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  Option Parameters: S0={S0}, K={K}, T={T:.2f}, r={r:.2%}")
        print(f"  Volatility levels: {volatilities}")
        print(f"  Grid sizes: {len(grid_sizes)}")
        print(f"  Methods: {len(methods_to_test)}")
        print(f"  Total tests: {total_tests}")
        print(f"  Output file: {self.output_file}")
        print(f"  Progress file: {self.progress_file}\n")

        test_count = 0

        for sigma in volatilities:
            print(f"\n{'='*100}")
            print(f"Volatility: σ={sigma:.1%}")
            print(f"{'='*100}")

            # Write volatility header
            with open(self.output_file, 'a') as f:
                f.write(f"\n{'='*100}\n")
                f.write(f"Volatility: σ={sigma:.1%}\n")
                f.write(f"{'='*100}\n")
                f.flush()

            for M, N in grid_sizes:
                print(f"\n  Grid: M={M}, N={N}")
                print(f"  {'-'*96}")

                # Write grid header
                with open(self.output_file, 'a') as f:
                    f.write(f"\n  Grid: M={M}, N={N}\n")
                    f.write(f"  {'-'*96}\n")
                    f.flush()

                grid_results = {}
                bsm_result = None

                for method_name in methods_to_test:
                    test_count += 1
                    method_class = self.methods[method_name]

                    try:
                        # Create method instance
                        method = method_class(M, N, S0, K, T, r)

                        # Compute Hessian
                        raw_result = method.compute_hessian(S0, sigma)

                        # Flatten Greeks for easier access
                        result = {
                            'price': raw_result['price'],
                            'delta': raw_result['greeks']['delta'],
                            'gamma': raw_result['greeks']['gamma'],
                            'vega': raw_result['greeks']['vega'],
                            'vanna': raw_result['greeks']['vanna'],
                            'volga': raw_result['greeks']['volga'],
                            'time_ms': raw_result['time_ms'],
                            'n_pde_solves': raw_result['n_pde_solves'],
                            'method': method_name
                        }

                        if method_name == 'BSM-Analytical':
                            bsm_result = result

                        grid_results[method_name] = result

                        # Print result
                        output_line = (
                            f"    [{test_count}/{total_tests}] {method_name:<20} ✓ "
                            f"{result['time_ms']:>8.2f}ms  "
                            f"Price={result['price']:.5f}  "
                            f"Gamma={result['gamma']:.6f}\n"
                        )
                        print(output_line, end='')
                        sys.stdout.flush()

                        # Write to file
                        with open(self.output_file, 'a') as f:
                            f.write(output_line)
                            f.flush()

                        # Save to results
                        self.results.append({
                            'sigma': sigma,
                            'M': M,
                            'N': N,
                            'method': method_name,
                            'S0': S0,
                            'K': K,
                            'T': T,
                            'r': r,
                            **result
                        })

                    except Exception as e:
                        error_msg = f"    [{test_count}/{total_tests}] {method_name:<20} ✗ FAILED: {str(e)}\n"
                        print(error_msg, end='')
                        with open(self.output_file, 'a') as f:
                            f.write(error_msg)
                            f.flush()

                # Write error comparison table
                if bsm_result and len(grid_results) > 1:
                    self._write_error_table(grid_results, bsm_result)

                # Save progress
                with open(self.progress_file, 'w') as f:
                    json.dump(self.results, f, indent=2)

        print(f"\n{'='*100}")
        print(f"Benchmark Complete! Total tests: {test_count}")
        print(f"{'='*100}\n")
        print(f"Results saved to: {self.output_file}")
        print(f"Progress saved to: {self.progress_file}")

    def _write_error_table(self, grid_results: Dict, bsm_result: Dict):
        """Write error comparison table vs BSM analytical."""

        print(f"\n    Errors vs BSM Analytical:")
        print(f"    {'Method':<20} {'Price %':<10} {'Gamma %':<10} {'Vega %':<10} "
              f"{'Vanna %':<10} {'Volga %':<10}")
        print(f"    {'-'*70}")

        error_table = "\n    Errors vs BSM Analytical:\n"
        error_table += f"    {'Method':<20} {'Price %':<10} {'Gamma %':<10} {'Vega %':<10} {'Vanna %':<10} {'Volga %':<10}\n"
        error_table += f"    {'-'*70}\n"

        for method_name, result in grid_results.items():
            if method_name == 'BSM-Analytical':
                continue

            errors = {}
            for greek in ['price', 'gamma', 'vega', 'vanna', 'volga']:
                bsm_val = bsm_result[greek]
                val = result[greek]
                if abs(bsm_val) > 1e-10:
                    errors[greek] = abs(val - bsm_val) / abs(bsm_val) * 100
                else:
                    errors[greek] = 0.0

            error_line = (f"    {method_name:<20} {errors['price']:>8.4f}%  "
                        f"{errors['gamma']:>8.4f}%  {errors['vega']:>8.4f}%  "
                        f"{errors['vanna']:>8.4f}%  {errors['volga']:>8.4f}%\n")

            print(error_line, end='')
            error_table += error_line

        # Write to file
        with open(self.output_file, 'a') as f:
            f.write(error_table)
            f.flush()


def main():
    """Run volatility sensitivity benchmark."""
    benchmark = VolatilityBenchmark()

    # Option parameters (ATM call)
    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05

    # Test different volatility levels
    volatilities = [0.10, 0.20, 0.30, 0.40, 0.50]  # 10%, 20%, 30%, 40%, 50%

    # Grid sizes
    grid_sizes = [
        (51, 100),    # Coarse
        (101, 200),   # Fine
    ]

    # Run benchmark
    benchmark.run_benchmark(
        S0=S0,
        K=K,
        T=T,
        r=r,
        volatilities=volatilities,
        grid_sizes=grid_sizes,
        methods_to_test=None  # Test all 5 methods
    )


if __name__ == "__main__":
    main()
