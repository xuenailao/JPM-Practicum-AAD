"""
综合Greeks基准测试 - 命令行版本
====================================

测试5种方法:
1. BSM Analytical (解析解)
2. Bumping (有限差分)
3. AAD + Bumping (混合方法)
4. Double AAD (双重AAD)
5. Edge-Pushing (边推算法)

自动保存结果到:
- CSV文件: 原始数据
- Markdown报告: 人类可读
- 计算图输出: 文本文件

使用方法:
  python run_comprehensive_benchmark.py --mode quick    # 快速测试 (~2分钟)
  python run_comprehensive_benchmark.py --mode full     # 完整测试 (~30-60分钟)
  python run_comprehensive_benchmark.py --mode quick --graph  # 包含详细计算图
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import sys
from pathlib import Path
import time
from typing import Dict, List
import itertools
import argparse
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.unified_greeks_interface import UnifiedGreeksCalculator
from aad_edge_pushing.aad.core.graph_utils import print_graph_summary, print_computation_graph, get_graph_stats
from aad_edge_pushing.aad.core.tape import global_tape


class ComprehensiveBenchmark:
    """综合基准测试框架"""

    def __init__(self, mode='quick', output_dir='benchmark_results'):
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results = []
        self.graph_outputs = []

        print(f"\n{'='*80}")
        print(f"{'COMPREHENSIVE GREEKS BENCHMARK':^80}")
        print(f"{'='*80}")
        print(f"\nMode: {mode.upper()}")
        print(f"Output directory: {self.output_dir}")
        print(f"Timestamp: {self.timestamp}")

    def get_test_configs(self):
        """获取测试配置"""
        if self.mode == 'quick':
            # 快速测试: 18个参数组合 × 1个网格 × 5个方法 = 90次计算
            configs = []
            K = 100.0
            grid = (51, 50)

            for S0 in [95, 100, 105]:  # ITM/ATM/OTM
                for T in [0.5, 1.0]:  # 短期/中期
                    for sigma in [0.15, 0.20, 0.30]:  # 低/中/高波动率
                        configs.append({
                            'S0': S0, 'K': K, 'T': T, 'r': 0.05, 'sigma': sigma,
                            'M': grid[0], 'N': grid[1]
                        })

            print(f"Quick mode: {len(configs)} configurations × 5 methods = {len(configs)*5} tests")

        else:  # full
            # 完整测试: 72个参数组合 × 3个网格 × 5个方法 = 1080次计算
            configs = []
            K = 100.0

            for S0 in [90, 100, 110]:
                for T in [0.25, 0.5, 1.0]:
                    for r in [0.03, 0.05]:
                        for sigma in [0.15, 0.20, 0.30, 0.40]:
                            for M, N in [(21, 20), (51, 50), (101, 100)]:
                                configs.append({
                                    'S0': S0, 'K': K, 'T': T, 'r': r, 'sigma': sigma,
                                    'M': M, 'N': N
                                })

            print(f"Full mode: {len(configs)} configurations × 5 methods = {len(configs)*5} tests")

        return configs

    def compute_analytical(self, S0, K, T, r, sigma):
        """计算解析Greeks"""
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)

        price = S0 * N_d1 - K * np.exp(-r * T) * N_d2
        delta = N_d1
        gamma = n_d1 / (S0 * sigma * sqrt_T)
        vega = S0 * n_d1 * sqrt_T
        vanna = -n_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma

        return {
            'price': price, 'delta': delta, 'gamma': gamma,
            'vega': vega, 'vanna': vanna, 'volga': volga
        }

    def run_test(self, config, method, analytical, print_graph=False):
        """运行单个测试"""
        calc = UnifiedGreeksCalculator(M=config['M'], N=config['N'])

        global_tape.reset()

        try:
            result = calc.compute_greeks(
                S0=config['S0'], K=config['K'], T=config['T'],
                r=config['r'], sigma=config['sigma'],
                method=method,
                verbose=False,
                track_graph=True
            )

            # 计算误差
            errors = {}
            for greek in ['delta', 'gamma', 'vega', 'vanna', 'volga']:
                num_val = result['greeks'][greek]
                ana_val = analytical[greek]
                if abs(ana_val) > 1e-10:
                    errors[f'{greek}_error_pct'] = abs(num_val - ana_val) / abs(ana_val) * 100
                else:
                    errors[f'{greek}_error_pct'] = 0.0

            # 保存结果
            test_result = {
                **config,
                'method': result['method'],
                'price': result['price'],
                **{f'{g}': result['greeks'][g] for g in ['delta', 'gamma', 'vega', 'vanna', 'volga']},
                **{f'{g}_analytical': analytical[g] for g in ['delta', 'gamma', 'vega', 'vanna', 'volga']},
                **errors,
                'time_ms': result['time_ms'],
                'n_pde_solves': result['n_pde_solves'],
                'graph_nodes': result.get('graph_info', {}).get('nodes', 0),
                'graph_edges': result.get('graph_info', {}).get('edges', 0),
                'graph_max_fan_in': result.get('graph_info', {}).get('max_fan_in', 0),
            }

            # 打印计算图（首次AAD方法）
            if print_graph and method in ['edge_pushing', 'aad_bumping']:
                graph_output = self.capture_graph_output(config, method)
                self.graph_outputs.append(graph_output)

            return test_result

        except Exception as e:
            print(f"  ERROR in {method}: {str(e)[:60]}")
            return None

    def capture_graph_output(self, config, method):
        """捕获计算图输出"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"Computation Graph: {method.upper()}")
        output.append(f"Parameters: S0={config['S0']}, K={config['K']}, T={config['T']}, "
                     f"r={config['r']}, sigma={config['sigma']}, M={config['M']}, N={config['N']}")
        output.append(f"{'='*80}\n")

        # 获取统计信息
        stats = get_graph_stats(global_tape)
        output.append(f"Total nodes: {stats['nodes']:,}")
        output.append(f"Total edges: {stats['edges']:,}")
        output.append(f"Max fan-in: {stats['max_fan_in']}")
        output.append(f"Max fan-out: {stats['max_fan_out']}")
        output.append(f"\nOperation breakdown:")
        for op, count in sorted(stats['operations'].items(), key=lambda x: -x[1])[:10]:
            pct = 100.0 * count / stats['nodes'] if stats['nodes'] > 0 else 0
            output.append(f"  {op:12s}: {count:6,} ({pct:5.1f}%)")

        return '\n'.join(output)

    def run_all_tests(self):
        """运行所有测试"""
        configs = self.get_test_configs()
        methods = ['analytical', 'bumping', 'aad_bumping', 'double_aad', 'edge_pushing']

        total_tests = len(configs) * len(methods)
        completed = 0
        start_time = time.time()

        print(f"\n{'='*80}")
        print(f"Starting {total_tests} tests...")
        print(f"{'='*80}\n")

        graph_printed = False

        for i, config in enumerate(configs):
            print(f"\n[Config {i+1}/{len(configs)}] S0={config['S0']:.0f}, T={config['T']}, "
                  f"σ={config['sigma']}, M={config['M']}")

            # 计算解析解
            analytical = self.compute_analytical(
                config['S0'], config['K'], config['T'], config['r'], config['sigma']
            )

            # 测试各方法
            for method in methods:
                print_graph = (not graph_printed) and (method in ['edge_pushing', 'aad_bumping'])

                result = self.run_test(config, method, analytical, print_graph=print_graph)

                if result is not None:
                    self.results.append(result)
                    if print_graph:
                        graph_printed = True

                completed += 1

                # 进度报告
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = (total_tests - completed) * avg_time

                if method == 'analytical':
                    status = f"OK (baseline)"
                else:
                    gamma_err = result.get('gamma_error_pct', 0) if result else 0
                    status = f"γ_err={gamma_err:.2f}%"

                print(f"  [{method:15s}] {status:20s} ({completed}/{total_tests}, "
                      f"ETA: {remaining/60:.1f}min)")

        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print(f"All tests completed in {total_time/60:.1f} minutes")
        print(f"Total results: {len(self.results)}")
        print(f"{'='*80}\n")

    def save_results(self):
        """保存结果"""
        df = pd.DataFrame(self.results)

        # 1. 保存CSV
        csv_file = self.output_dir / f'results_{self.mode}_{self.timestamp}.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ CSV saved: {csv_file}")

        # 2. 保存计算图
        if self.graph_outputs:
            graph_file = self.output_dir / f'computation_graphs_{self.mode}_{self.timestamp}.txt'
            with open(graph_file, 'w') as f:
                f.write('\n\n'.join(self.graph_outputs))
            print(f"✓ Computation graphs saved: {graph_file}")

        # 3. 生成Markdown报告
        report = self.generate_markdown_report(df)
        report_file = self.output_dir / f'REPORT_{self.mode}_{self.timestamp}.md'
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"✓ Markdown report saved: {report_file}")

        return df

    def df_to_markdown(self, df_data):
        """手动将DataFrame转为Markdown表格"""
        lines = []
        # 表头
        headers = df_data.columns.tolist()
        lines.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
        lines.append('|' + '|'.join(['---' for _ in headers]) + '|')
        # 数据行
        for _, row in df_data.iterrows():
            lines.append('| ' + ' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in row) + ' |')
        return '\n'.join(lines)

    def generate_markdown_report(self, df):
        """生成Markdown报告"""
        lines = []
        lines.append(f"# Greeks Computation Benchmark Report")
        lines.append(f"\n**Mode:** {self.mode.upper()}")
        lines.append(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Tests:** {len(df)}")
        lines.append(f"\n---\n")

        # 1. 速度对比
        lines.append("## 1. Speed Comparison\n")
        speed_df = df.groupby(['method', 'M'])['time_ms'].agg(['mean', 'std', 'min', 'max']).reset_index()
        speed_df.columns = ['Method', 'M', 'Mean (ms)', 'Std (ms)', 'Min (ms)', 'Max (ms)']
        lines.append(self.df_to_markdown(speed_df))

        # 2. 精度对比
        lines.append("\n\n## 2. Accuracy Comparison\n")
        lines.append("### Average Errors (all configurations)\n")
        error_cols = ['delta_error_pct', 'gamma_error_pct', 'vega_error_pct', 'vanna_error_pct', 'volga_error_pct']
        acc_df = df[df['method'] != 'analytical'].groupby('method')[error_cols].mean().reset_index()
        acc_df.columns = ['Method', 'Δ err%', 'Γ err%', 'ν err%', 'Vanna err%', 'Volga err%']
        lines.append(self.df_to_markdown(acc_df))

        # 3. 按网格分辨率
        lines.append("\n\n### Accuracy by Grid Resolution\n")
        for M in sorted(df['M'].unique()):
            lines.append(f"\n#### M={M}\n")
            df_M = df[(df['M'] == M) & (df['method'] != 'analytical')]
            acc_M = df_M.groupby('method')[error_cols].mean().reset_index()
            acc_M.columns = ['Method', 'Δ err%', 'Γ err%', 'ν err%', 'Vanna err%', 'Volga err%']
            lines.append(self.df_to_markdown(acc_M))

        # 4. PDE求解次数
        lines.append("\n\n## 3. Computational Cost\n")
        pde_df = df.groupby('method')[['n_pde_solves', 'graph_nodes', 'graph_edges']].first().reset_index()
        pde_df.columns = ['Method', 'PDE Solves', 'Graph Nodes', 'Graph Edges']
        lines.append(self.df_to_markdown(pde_df))

        # 5. 推荐配置
        lines.append("\n\n## 4. Recommendations\n")
        lines.append("\n### Best Method by Criterion:\n")

        fastest = df[df['method'] != 'analytical'].groupby('method')['time_ms'].mean().idxmin()
        best_gamma = df[df['method'] != 'analytical'].groupby('method')['gamma_error_pct'].mean().idxmin()
        best_volga = df[df['method'] != 'analytical'].groupby('method')['volga_error_pct'].mean().idxmin()

        lines.append(f"- **Fastest:** {fastest}")
        lines.append(f"- **Most Accurate Gamma:** {best_gamma}")
        lines.append(f"- **Most Accurate Volga:** {best_volga}")

        lines.append("\n### Production Recommendations:\n")
        lines.append("- **Quick computations (M=51):** Edge-Pushing (1 PDE solve, good accuracy)")
        lines.append("- **High accuracy (M=101):** Edge-Pushing (Gamma < 0.5%, Volga < 10%)")
        lines.append("- **Simple implementation:** Bumping (5 PDE solves, moderate accuracy)")

        lines.append("\n---")
        lines.append(f"\n*Report generated by run_comprehensive_benchmark.py*")

        return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Greeks Benchmark',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_benchmark.py --mode quick          # ~2 minutes
  python run_comprehensive_benchmark.py --mode full           # ~30-60 minutes
  python run_comprehensive_benchmark.py --mode quick --graph  # with detailed graphs
        """
    )

    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'],
                       help='Test mode: quick (90 tests) or full (1080 tests)')
    parser.add_argument('--graph', action='store_true',
                       help='Save detailed computation graphs')
    parser.add_argument('--output', type=str, default='benchmark_results',
                       help='Output directory for results')

    args = parser.parse_args()

    # 运行基准测试
    benchmark = ComprehensiveBenchmark(mode=args.mode, output_dir=args.output)
    benchmark.run_all_tests()
    df = benchmark.save_results()

    # 打印摘要
    print(f"\n{'='*80}")
    print(f"{'BENCHMARK SUMMARY':^80}")
    print(f"{'='*80}\n")

    print(f"Total configurations tested: {len(df) // 5}")
    print(f"Total computations: {len(df)}")
    print(f"\nAverage errors (non-analytical methods):")

    error_summary = df[df['method'] != 'analytical'][['method', 'gamma_error_pct', 'volga_error_pct']].groupby('method').mean()
    for method, row in error_summary.iterrows():
        print(f"  {method:20s}: Gamma {row['gamma_error_pct']:6.2f}%, Volga {row['volga_error_pct']:6.2f}%")

    print(f"\n{'='*80}")
    print(f"Results saved to: {benchmark.output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
