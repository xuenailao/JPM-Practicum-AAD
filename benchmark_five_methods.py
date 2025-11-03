"""
Comprehensive Benchmark Interface for 5 Hessian Computation Methods

Tests all 5 methods across multiple parameter sets and grid resolutions:
1. Bumping2 - Pure finite difference (9 PDE solves)
2. AAD+Bumping - Hybrid method (5 PDE solves)
3. Double-AAD - Forward-over-Reverse (3 PDE solves)
4. Edge-Pushing - Algorithm 4 optimization (1 PDE solve)
5. BSM Analytical - Baseline for accuracy

Metrics:
- Accuracy: Error vs BSM analytical solution for all Greeks
- Speed: Wall-clock time in milliseconds
- Efficiency: Time per PDE solve

Grid configurations: From coarse to fine
Parameter sets: ATM, ITM, OTM, different maturities and volatilities
"""

import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
import time
import json
import os
from typing import Dict, List
from datetime import datetime

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available. Some features will be limited.")

from aad_edge_pushing.pde.methods import (
    Bumping2Method,
    AADBumpingMethod,
    DoubleAADMethod,
    EdgePushingMethod,
    BSMAnalyticalMethod
)


class HessianBenchmark:
    """Comprehensive benchmark suite for Hessian computation methods"""

    def __init__(self, resume_from_file=None):
        self.methods = {
            'Bumping2': Bumping2Method,
            'AAD+Bumping': AADBumpingMethod,
            'Double-AAD': DoubleAADMethod,
            'Edge-Pushing': EdgePushingMethod,
            'BSM-Analytical': BSMAnalyticalMethod
        }
        self.results = []
        self.output_file = None

        # Resume from previous run if specified
        if resume_from_file and os.path.exists(resume_from_file):
            print(f"Resuming from {resume_from_file}...")
            with open(resume_from_file, 'r') as f:
                self.results = json.load(f)
            print(f"Loaded {len(self.results)} previous results")

    def run_single_test(self, method_name: str, method_class,
                       S0: float, K: float, T: float, r: float, sigma: float,
                       M: int, N: int) -> Dict:
        """Run a single test case"""

        # Create method instance
        method = method_class(M=M, N=N, S0=S0, K=K, T=T, r=r)

        # Compute Hessian
        try:
            result = method.compute_hessian(S0, sigma)

            return {
                'method': method_name,
                'S0': S0,
                'K': K,
                'T': T,
                'r': r,
                'sigma': sigma,
                'M': M,
                'N': N,
                'moneyness': S0/K,
                'price': result['price'],
                'delta': result['greeks']['delta'],
                'gamma': result['greeks']['gamma'],
                'vega': result['greeks']['vega'],
                'vanna': result['greeks']['vanna'],
                'volga': result['greeks']['volga'],
                'time_ms': result['time_ms'],
                'n_pde_solves': result['n_pde_solves'],
                'status': 'SUCCESS'
            }
        except Exception as e:
            return {
                'method': method_name,
                'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
                'M': M, 'N': N,
                'moneyness': S0/K,
                'status': f'FAILED: {str(e)}'
            }

    def _check_if_done(self, method_name, param_name, M, N):
        """Check if this test was already completed"""
        for r in self.results:
            if (r.get('method') == method_name and
                r.get('param_set', r.get('S0')) == param_name and
                r.get('M') == M and r.get('N') == N and
                r.get('status') == 'SUCCESS'):
                return True
        return False

    def run_benchmark(self, parameter_sets: List[Dict], grid_sizes: List[tuple],
                     methods_to_test: List[str] = None):
        """
        Run comprehensive benchmark across all configurations

        Args:
            parameter_sets: List of dicts with keys {S0, K, T, r, sigma, name}
            grid_sizes: List of tuples (M, N)
            methods_to_test: List of method names to test (default: all)
        """

        if methods_to_test is None:
            methods_to_test = list(self.methods.keys())

        # Setup output files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_file = f'benchmark_results/benchmark_{timestamp}.txt'
        self.progress_file = f'benchmark_results/progress_{timestamp}.json'

        total_tests = len(parameter_sets) * len(grid_sizes) * len(methods_to_test)
        test_count = 0

        # Write header to file
        with open(self.output_file, 'w') as f:
            f.write("=" * 100 + "\n")
            f.write("HESSIAN COMPUTATION BENCHMARK - 5 Methods Comparison\n")
            f.write("=" * 100 + "\n")
            f.write(f"\nConfiguration:\n")
            f.write(f"  Parameter sets: {len(parameter_sets)}\n")
            f.write(f"  Grid sizes: {len(grid_sizes)}\n")
            f.write(f"  Methods: {len(methods_to_test)}\n")
            f.write(f"  Total tests: {total_tests}\n\n")
            f.flush()

        print("=" * 100)
        print("HESSIAN COMPUTATION BENCHMARK - 5 Methods Comparison")
        print("=" * 100)
        print(f"\nConfiguration:")
        print(f"  Parameter sets: {len(parameter_sets)}")
        print(f"  Grid sizes: {len(grid_sizes)}")
        print(f"  Methods: {len(methods_to_test)}")
        print(f"  Total tests: {total_tests}")
        print(f"  Output file: {self.output_file}")
        print(f"  Progress file: {self.progress_file}\n")

        for param_set in parameter_sets:
            param_name = param_set.get('name', 'Unnamed')
            S0 = param_set['S0']
            K = param_set['K']
            T = param_set['T']
            r = param_set['r']
            sigma = param_set['sigma']

            print(f"\n{'='*100}")
            print(f"Parameter Set: {param_name}")
            print(f"  S0={S0}, K={K}, T={T:.2f}, r={r:.2%}, σ={sigma:.2%}, Moneyness={S0/K:.3f}")
            print(f"{'='*100}")

            # Write parameter set header to file
            with open(self.output_file, 'a') as f:
                f.write(f"\n{'='*100}\n")
                f.write(f"Parameter Set: {param_name}\n")
                f.write(f"  S0={S0}, K={K}, T={T:.2f}, r={r:.2%}, σ={sigma:.2%}, Moneyness={S0/K:.3f}\n")
                f.write(f"{'='*100}\n")
                f.flush()

            for M, N in grid_sizes:
                print(f"\n  Grid: M={M}, N={N}")
                print(f"  {'-'*96}")

                # Write grid header to file
                with open(self.output_file, 'a') as f:
                    f.write(f"\n  Grid: M={M}, N={N}\n")
                    f.write(f"  {'-'*96}\n")
                    f.flush()

                # Store results for this grid size
                grid_results = {}
                bsm_result = None

                for method_name in methods_to_test:
                    test_count += 1
                    method_class = self.methods[method_name]

                    # Check if already done
                    if self._check_if_done(method_name, param_name, M, N):
                        print(f"    [{test_count}/{total_tests}] {method_name:<20} ⊙ SKIPPED (already done)")
                        sys.stdout.flush()
                        continue

                    print(f"    [{test_count}/{total_tests}] {method_name:<20}", end='', flush=True)
                    sys.stdout.flush()  # Immediate output

                    result = self.run_single_test(
                        method_name, method_class,
                        S0, K, T, r, sigma, M, N
                    )

                    # Add param_set name for tracking
                    result['param_set'] = param_name

                    output_line = ""
                    if result['status'] == 'SUCCESS':
                        grid_results[method_name] = result
                        if method_name == 'BSM-Analytical':
                            bsm_result = result

                        output_line = (f"    [{test_count}/{total_tests}] {method_name:<20} "
                                     f"✓ {result['time_ms']:8.2f}ms  "
                                     f"Price={result['price']:8.5f}  "
                                     f"Gamma={result['gamma']:8.6f}\n")
                        print(f" ✓ {result['time_ms']:8.2f}ms  "
                              f"Price={result['price']:8.5f}  "
                              f"Gamma={result['gamma']:8.6f}")
                        sys.stdout.flush()  # Immediate output
                    else:
                        output_line = f"    [{test_count}/{total_tests}] {method_name:<20} ✗ {result['status']}\n"
                        print(f" ✗ {result['status']}")
                        sys.stdout.flush()  # Immediate output

                    self.results.append(result)

                    # Write to output file immediately
                    with open(self.output_file, 'a') as f:
                        f.write(output_line)
                        f.flush()

                    # Save progress JSON after each test
                    with open(self.progress_file, 'w') as f:
                        json.dump(self.results, f, indent=2)

                    sys.stdout.flush()  # Immediate output

                # Print accuracy comparison if BSM available
                if bsm_result and len(grid_results) > 1:
                    print(f"\n    Errors vs BSM Analytical:")
                    print(f"    {'Method':<20} {'Price %':<10} {'Gamma %':<10} {'Vega %':<10} "
                          f"{'Vanna %':<10} {'Volga %':<10}")
                    print(f"    {'-'*70}")

                    # Build error table for file
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

                        print(f"    {method_name:<20} {errors['price']:>8.4f}%  "
                              f"{errors['gamma']:>8.4f}%  {errors['vega']:>8.4f}%  "
                              f"{errors['vanna']:>8.4f}%  {errors['volga']:>8.4f}%")

                        error_table += error_line

                    # Write error table to file immediately
                    with open(self.output_file, 'a') as f:
                        f.write(error_table)
                        f.flush()

        print(f"\n{'='*100}")
        print(f"Benchmark Complete! Total tests: {test_count}")
        print(f"{'='*100}\n")

    def save_results(self, filename: str = 'benchmark_results.csv'):
        """Save results to CSV file"""
        if not HAS_PANDAS:
            print("Warning: pandas not available, saving as JSON instead")
            import json
            json_file = filename.replace('.csv', '.json')
            with open(json_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            print(f"Results saved to {json_file}")
            return self.results

        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        return df

    def print_summary(self):
        """Print summary statistics"""
        successful_results = [r for r in self.results if r['status'] == 'SUCCESS']

        if not successful_results:
            print("No successful results to summarize")
            return

        print("\n" + "="*100)
        print("SUMMARY STATISTICS")
        print("="*100)

        if HAS_PANDAS:
            df = pd.DataFrame(successful_results)

            # Average time by method
            print("\nAverage Time (ms) by Method:")
            print("-"*50)
            time_by_method = df.groupby('method')['time_ms'].agg(['mean', 'std', 'min', 'max'])
            print(time_by_method.to_string())

            # Average time per PDE solve
            print("\nAverage Time per PDE Solve (ms):")
            print("-"*50)
            df['time_per_pde'] = df['time_ms'] / df['n_pde_solves'].replace(0, 1)
            time_per_pde = df.groupby('method')['time_per_pde'].agg(['mean', 'std'])
            print(time_per_pde.to_string())

            # Grid size impact
            print("\nAverage Time by Grid Size:")
            print("-"*50)
            grid_time = df.groupby(['M', 'N'])['time_ms'].mean().reset_index()
            print(grid_time.to_string(index=False))
        else:
            # Manual summary without pandas
            print("\nAverage Time (ms) by Method:")
            print("-"*50)
            methods = {}
            for r in successful_results:
                m = r['method']
                if m not in methods:
                    methods[m] = []
                methods[m].append(r['time_ms'])

            for method, times in sorted(methods.items()):
                print(f"{method:<20} Mean: {np.mean(times):8.2f}  "
                      f"Std: {np.std(times):8.2f}  "
                      f"Min: {np.min(times):8.2f}  "
                      f"Max: {np.max(times):8.2f}")


def create_parameter_sets():
    """Create diverse parameter sets for testing"""
    return [
        # ATM, standard parameters
        {
            'name': 'ATM-Standard',
            'S0': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.2
        },
        # ITM
        {
            'name': 'ITM-10%',
            'S0': 110.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.2
        },
        # OTM
        {
            'name': 'OTM-10%',
            'S0': 90.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.2
        },
        # High volatility
        {
            'name': 'ATM-HighVol',
            'S0': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.4
        },
        # Low volatility
        {
            'name': 'ATM-LowVol',
            'S0': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.1
        },
        # Short maturity
        {
            'name': 'ATM-ShortMat',
            'S0': 100.0, 'K': 100.0, 'T': 0.25, 'r': 0.05, 'sigma': 0.2
        },
        # Long maturity
        {
            'name': 'ATM-LongMat',
            'S0': 100.0, 'K': 100.0, 'T': 2.0, 'r': 0.05, 'sigma': 0.2
        }
    ]


def create_grid_sizes():
    """Create grid size configurations from coarse to fine"""
    return [
        (51, 100),    # Coarse
        (101, 200),   # Medium
        (151, 300),   # Fine
        (201, 400),   # Very fine
    ]


def main():
    """Run comprehensive benchmark"""

    benchmark = HessianBenchmark()

    # Define test configurations
    parameter_sets = create_parameter_sets()
    grid_sizes = create_grid_sizes()

    # Run benchmark
    benchmark.run_benchmark(
        parameter_sets=parameter_sets,
        grid_sizes=grid_sizes,
        methods_to_test=None  # Test all methods
    )

    # Print summary
    benchmark.print_summary()

    # Save results
    df = benchmark.save_results('benchmark_results/hessian_benchmark_results.csv')

    return benchmark, df


def quick_test():
    """Quick test with minimal configuration"""
    print("\n" + "="*100)
    print("QUICK TEST MODE - 2 parameter sets × 2 grid sizes")
    print("="*100)

    benchmark = HessianBenchmark()

    parameter_sets = [
        {'name': 'ATM', 'S0': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.2},
        {'name': 'ITM', 'S0': 110.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'sigma': 0.3}
    ]

    grid_sizes = [(51, 100), (101, 200)]

    benchmark.run_benchmark(
        parameter_sets=parameter_sets,
        grid_sizes=grid_sizes
    )

    benchmark.print_summary()

    return benchmark


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Benchmark Hessian computation methods')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Test mode: quick (2×2) or full (7×4)')

    args = parser.parse_args()

    if args.mode == 'quick':
        benchmark = quick_test()
    else:
        benchmark, df = main()
