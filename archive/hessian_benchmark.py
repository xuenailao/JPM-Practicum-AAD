"""
Unified Hessian Benchmark Framework

Tests all 5 methods for computing (S0, σ) Hessian matrix:
1. Bumping2: Pure finite difference
2. AAD+Bumping: Hybrid method
3. Double-AAD: Nested AAD (via Edge-Pushing)
4. Edge-Pushing: Single-tape Hessian
5. BSM-Analytical: Analytical baseline

Test Scenarios:
- Multiple S0 values (moneyness: OTM, ATM, ITM)
- Multiple σ values (low to high volatility)
- Multiple grid sizes (M, N)

Output:
- CSV: Complete results table
- Metrics: Accuracy (relative errors) and Speed (time, PDE solves)
- Report: Markdown summary with analysis
"""

import numpy as np
import pandas as pd
import time
from typing import List, Tuple, Dict
from pathlib import Path
import sys

sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.methods.method_1_bumping2 import Bumping2Method
from aad_edge_pushing.pde.methods.method_2_aad_bumping import AADBumpingMethod
from aad_edge_pushing.pde.methods.method_3_double_aad import DoubleAADMethod
from aad_edge_pushing.pde.methods.method_4_edge_pushing import EdgePushingMethod
from aad_edge_pushing.pde.methods.method_5_bsm_analytical import BSMAnalyticalMethod


class HessianBenchmark:
    """
    Comprehensive benchmark framework for Hessian computation methods.

    Compares accuracy and performance across different:
    - Methods (5 methods)
    - Parameters (S0, σ)
    - Grid sizes (M, N)
    """

    def __init__(self):
        """Initialize benchmark with all methods."""
        self.method_classes = {
            'Bumping2': Bumping2Method,
            'AAD+Bumping': AADBumpingMethod,
            'Double-AAD': DoubleAADMethod,
            'Edge-Pushing': EdgePushingMethod,
            'BSM-Analytical': BSMAnalyticalMethod
        }

    def _compute_errors(self, result: Dict, truth: Dict) -> Dict:
        """
        Compute relative errors against analytical truth.

        Args:
            result: Numerical result
            truth: Analytical truth

        Returns:
            Dictionary of relative errors (in %)
        """
        errors = {}

        # Price error
        if abs(truth['price']) > 1e-10:
            errors['price_err'] = abs(result['price'] - truth['price']) / abs(truth['price']) * 100
        else:
            errors['price_err'] = 0.0

        # Jacobian errors
        for i, name in enumerate(['delta', 'vega']):
            truth_val = truth['greeks'][name]
            result_val = result['greeks'][name]
            if abs(truth_val) > 1e-10:
                errors[f'{name}_err'] = abs(result_val - truth_val) / abs(truth_val) * 100
            else:
                errors[f'{name}_err'] = abs(result_val - truth_val) * 100  # Absolute error if truth≈0

        # Hessian errors
        for name in ['gamma', 'vanna', 'volga']:
            truth_val = truth['greeks'][name]
            result_val = result['greeks'][name]
            if abs(truth_val) > 1e-10:
                errors[f'{name}_err'] = abs(result_val - truth_val) / abs(truth_val) * 100
            else:
                errors[f'{name}_err'] = abs(result_val - truth_val) * 100  # Absolute error

        return errors

    def run_single_test(self, method_name: str, S0: float, K: float,
                       T: float, r: float, sigma: float,
                       M: int, N: int, truth: Dict) -> Dict:
        """
        Run single test for one method and parameter set.

        Args:
            method_name: Name of method to test
            S0, K, T, r, sigma: Option parameters
            M, N: Grid sizes
            truth: Analytical solution for comparison

        Returns:
            Test result dictionary
        """
        MethodClass = self.method_classes[method_name]

        # Create method instance
        method = MethodClass(M=M, N=N, S0=S0, K=K, T=T, r=r)

        # Compute Hessian
        try:
            result = method.compute_hessian(S0=S0, sigma=sigma)

            # Compute errors
            errors = self._compute_errors(result, truth)

            # Package results
            output = {
                'method': method_name,
                'S0': S0,
                'sigma': sigma,
                'M': M,
                'N': N,
                'price': result['price'],
                'delta': result['greeks']['delta'],
                'gamma': result['greeks']['gamma'],
                'vega': result['greeks']['vega'],
                'vanna': result['greeks']['vanna'],
                'volga': result['greeks']['volga'],
                'time_ms': result['time_ms'],
                'n_pde_solves': result['n_pde_solves'],
                **errors,
                'success': True
            }

        except Exception as e:
            print(f"  ERROR in {method_name}: {e}")
            output = {
                'method': method_name,
                'S0': S0,
                'sigma': sigma,
                'M': M,
                'N': N,
                'success': False,
                'error_msg': str(e)
            }

        return output

    def run_benchmark(self,
                     S0_values: List[float],
                     sigma_values: List[float],
                     grid_configs: List[Tuple[int, int]],
                     K: float = 100.0,
                     T: float = 1.0,
                     r: float = 0.05,
                     output_dir: str = 'benchmark_results') -> pd.DataFrame:
        """
        Run complete benchmark suite.

        Args:
            S0_values: List of initial stock prices
            sigma_values: List of volatilities
            grid_configs: List of (M, N) grid configurations
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            output_dir: Directory for output files

        Returns:
            DataFrame with all results
        """
        print("="*80)
        print("HESSIAN BENCHMARK - Comprehensive Test")
        print("="*80)
        print(f"\nTest Configuration:")
        print(f"  S0 values: {S0_values}")
        print(f"  σ values: {sigma_values}")
        print(f"  Grid configs: {grid_configs}")
        print(f"  K={K}, T={T}, r={r}")
        print()

        results = []
        total_tests = len(S0_values) * len(sigma_values) * len(grid_configs) * 4  # 4 numerical methods
        test_count = 0

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for S0 in S0_values:
            for sigma in sigma_values:
                # Compute analytical truth once per (S0, σ)
                analytical_method = BSMAnalyticalMethod(M=0, N=0, S0=S0, K=K, T=T, r=r)
                truth = analytical_method.compute_hessian(S0=S0, sigma=sigma)

                print(f"\n[S0={S0:.1f}, σ={sigma:.2f}]")
                print(f"  Analytical: Price={truth['price']:.6f}, "
                      f"Gamma={truth['greeks']['gamma']:.6f}, "
                      f"Vanna={truth['greeks']['vanna']:.6f}, "
                      f"Volga={truth['greeks']['volga']:.6f}")

                # Add analytical result to output
                results.append({
                    'method': 'BSM-Analytical',
                    'S0': S0,
                    'sigma': sigma,
                    'M': 0,
                    'N': 0,
                    **truth,
                    'time_ms': truth['time_ms'],
                    'n_pde_solves': 0,
                    'success': True,
                    # Analytical has zero error by definition
                    **{k: 0.0 for k in ['price_err', 'delta_err', 'gamma_err',
                                        'vega_err', 'vanna_err', 'volga_err']}
                })

                for M, N in grid_configs:
                    # Adaptive grid for high volatility
                    if sigma >= 0.4 and M < 151:
                        M_use, N_use = 151, 300
                        print(f"  ⚠ High σ: Using M={M_use}, N={N_use} instead of M={M}, N={N}")
                    else:
                        M_use, N_use = M, N

                    print(f"\n  Grid: M={M_use}, N={N_use}")

                    # Test each numerical method
                    for method_name in ['Bumping2', 'AAD+Bumping', 'Double-AAD', 'Edge-Pushing']:
                        test_count += 1
                        print(f"    [{test_count}/{total_tests}] Testing {method_name}...", end=' ')

                        result = self.run_single_test(
                            method_name=method_name,
                            S0=S0, K=K, T=T, r=r, sigma=sigma,
                            M=M_use, N=N_use,
                            truth=truth
                        )

                        if result['success']:
                            print(f"✓ Time={result['time_ms']:.1f}ms, "
                                  f"Gamma_err={result['gamma_err']:.1f}%, "
                                  f"PDEs={result['n_pde_solves']}")
                        else:
                            print(f"✗ FAILED")

                        results.append(result)

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Save to CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = Path(output_dir) / f"hessian_benchmark_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Results saved to: {csv_path}")

        # Generate summary report
        self._generate_report(df, output_dir, timestamp)

        return df

    def _generate_report(self, df: pd.DataFrame, output_dir: str, timestamp: str):
        """
        Generate markdown summary report.

        Args:
            df: Results DataFrame
            output_dir: Output directory
            timestamp: Timestamp string
        """
        report_path = Path(output_dir) / f"hessian_report_{timestamp}.md"

        with open(report_path, 'w') as f:
            f.write("# Hessian Benchmark Report\n\n")
            f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Success rate
            success_df = df[df['success'] == True]
            f.write(f"## Summary\n\n")
            f.write(f"- Total tests: {len(df)}\n")
            f.write(f"- Successful: {len(success_df)}\n")
            f.write(f"- Failed: {len(df) - len(success_df)}\n\n")

            # Average errors by method
            f.write("## Accuracy by Method (Average Relative Errors)\n\n")
            numeric_df = success_df[success_df['method'] != 'BSM-Analytical']

            if len(numeric_df) > 0:
                error_cols = ['gamma_err', 'vanna_err', 'volga_err']
                method_errors = numeric_df.groupby('method')[error_cols].mean()

                f.write("| Method | Gamma Error (%) | Vanna Error (%) | Volga Error (%) |\n")
                f.write("|--------|-----------------|-----------------|------------------|\n")
                for method in method_errors.index:
                    gamma_err = method_errors.loc[method, 'gamma_err']
                    vanna_err = method_errors.loc[method, 'vanna_err']
                    volga_err = method_errors.loc[method, 'volga_err']
                    f.write(f"| {method} | {gamma_err:.2f} | {vanna_err:.2f} | {volga_err:.2f} |\n")
                f.write("\n")

            # Speed comparison
            f.write("## Speed Comparison (Average)\n\n")
            if len(numeric_df) > 0:
                speed_stats = numeric_df.groupby('method')[['time_ms', 'n_pde_solves']].mean()

                f.write("| Method | Time (ms) | PDE Solves |\n")
                f.write("|--------|-----------|------------|\n")
                for method in speed_stats.index:
                    time_ms = speed_stats.loc[method, 'time_ms']
                    n_pde = speed_stats.loc[method, 'n_pde_solves']
                    f.write(f"| {method} | {time_ms:.1f} | {n_pde:.0f} |\n")
                f.write("\n")

            # Grid sensitivity
            f.write("## Grid Sensitivity\n\n")
            f.write("Effect of grid size on accuracy (Gamma errors):\n\n")

            if 'M' in numeric_df.columns and len(numeric_df) > 0:
                grid_errors = numeric_df.groupby(['M', 'N', 'method'])['gamma_err'].mean().reset_index()
                f.write(grid_errors.to_markdown(index=False))
                f.write("\n\n")

        print(f"✓ Report saved to: {report_path}")


if __name__ == '__main__':
    # Run benchmark
    benchmark = HessianBenchmark()

    # Test configuration
    S0_values = [90, 100, 110]  # OTM, ATM, ITM
    sigma_values = [0.1, 0.2, 0.3, 0.4, 0.5]  # Low to high volatility
    grid_configs = [
        (51, 100),    # Fast grid (may fail at high σ)
        (101, 200),   # Standard grid (recommended)
        (151, 300),   # High-resolution grid (for high σ)
    ]

    # Run tests
    results_df = benchmark.run_benchmark(
        S0_values=S0_values,
        sigma_values=sigma_values,
        grid_configs=grid_configs,
        output_dir='benchmark_results'
    )

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)
    print(f"\nTotal tests: {len(results_df)}")
    print(f"Output directory: benchmark_results/")
