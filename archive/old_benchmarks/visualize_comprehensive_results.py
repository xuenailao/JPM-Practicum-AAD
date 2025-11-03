#!/usr/bin/env python3
"""
Visualization Script for Comprehensive Test Results

Generates:
1. Accuracy comparison plots (error vs scenario)
2. Speed comparison plots (time vs scenario)
3. Grid sensitivity analysis (error/time vs M/N)
4. AAD graph statistics visualization
5. Summary report in HTML format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class ComprehensiveResultsVisualizer:
    """Visualizer for comprehensive test results"""

    def __init__(self, results_csv: str, graph_csv: str = None):
        """
        Initialize visualizer

        Args:
            results_csv: Path to results CSV file
            graph_csv: Path to AAD graph statistics CSV (optional)
        """
        self.df = pd.read_csv(results_csv)
        self.graph_df = pd.read_csv(graph_csv) if graph_csv and Path(graph_csv).exists() else None
        self.output_dir = Path("visualization_output")
        self.output_dir.mkdir(exist_ok=True)

    def plot_accuracy_comparison(self):
        """Plot accuracy comparison across all methods and scenarios"""
        print("Generating accuracy comparison plots...")

        methods = ['bumping', 'double_aad', 'edge_pushing']
        greeks = ['delta', 'gamma', 'vega', 'vanna', 'volga', 'rho']

        # Create subplots for each Greek
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        for idx, greek in enumerate(greeks):
            ax = axes[idx]

            # Prepare data for each method
            for method in methods:
                col = f'{method}_{greek}_err'
                if col in self.df.columns:
                    # Get error values
                    errors = self.df[col].values
                    scenarios = np.arange(len(errors))

                    # Plot
                    ax.plot(scenarios, errors, marker='o', label=method.replace('_', ' ').title(), alpha=0.7)

            ax.set_xlabel('Scenario Index')
            ax.set_ylabel('Relative Error (%)')
            ax.set_title(f'{greek.upper()} Accuracy')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / "accuracy_comparison_all_greeks.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {self.output_dir / 'accuracy_comparison_all_greeks.png'}")

    def plot_speed_comparison(self):
        """Plot speed comparison across methods"""
        print("Generating speed comparison plots...")

        methods = ['analytical', 'bumping', 'double_aad', 'edge_pushing']
        colors = ['green', 'blue', 'orange', 'red']

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1: Absolute time
        ax = axes[0]
        for method, color in zip(methods, colors):
            col = f'{method}_time_ms'
            if col in self.df.columns:
                times = self.df[col].values
                scenarios = np.arange(len(times))
                ax.plot(scenarios, times, marker='o', label=method.title(), color=color, alpha=0.7)

        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Computation Time Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 2: Speedup relative to bumping
        ax = axes[1]
        bumping_times = self.df['bumping_time_ms'].values

        for method, color in zip(['double_aad', 'edge_pushing'], ['orange', 'red']):
            col = f'{method}_time_ms'
            if col in self.df.columns:
                method_times = self.df[col].values
                speedup = bumping_times / method_times
                scenarios = np.arange(len(speedup))
                ax.plot(scenarios, speedup, marker='o', label=method.replace('_', ' ').title(), color=color, alpha=0.7)

        ax.axhline(y=1.0, color='blue', linestyle='--', label='Bumping (baseline)', alpha=0.5)
        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Speedup Factor (vs Bumping)')
        ax.set_title('Speedup vs Bumping Method')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / "speed_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {self.output_dir / 'speed_comparison.png'}")

    def plot_grid_sensitivity(self):
        """Plot grid sensitivity analysis (M and N variations)"""
        print("Generating grid sensitivity plots...")

        # Filter M-varying scenarios
        df_M = self.df[self.df['scenario'].str.contains('Grid_M', na=False)].copy()
        # Filter N-varying scenarios
        df_N = self.df[self.df['scenario'].str.contains('Grid_N', na=False)].copy()

        if len(df_M) == 0 or len(df_N) == 0:
            print("  ⚠ No grid variation data found")
            return

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        methods = ['bumping', 'double_aad', 'edge_pushing']
        colors = ['blue', 'orange', 'red']

        # Plot 1: Gamma error vs M
        ax = axes[0, 0]
        for method, color in zip(methods, colors):
            col = f'{method}_gamma_err'
            if col in df_M.columns:
                ax.plot(df_M['M'], df_M[col], marker='o', label=method.replace('_', ' ').title(), color=color, alpha=0.7)
        ax.set_xlabel('M (Spatial Grid Points)')
        ax.set_ylabel('Gamma Error (%)')
        ax.set_title('Gamma Accuracy vs Spatial Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 2: Computation time vs M
        ax = axes[0, 1]
        for method, color in zip(methods, colors):
            col = f'{method}_time_ms'
            if col in df_M.columns:
                ax.plot(df_M['M'], df_M[col], marker='o', label=method.replace('_', ' ').title(), color=color, alpha=0.7)
        ax.set_xlabel('M (Spatial Grid Points)')
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Computation Time vs Spatial Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 3: Gamma error vs N
        ax = axes[1, 0]
        for method, color in zip(methods, colors):
            col = f'{method}_gamma_err'
            if col in df_N.columns:
                ax.plot(df_N['N'], df_N[col], marker='o', label=method.replace('_', ' ').title(), color=color, alpha=0.7)
        ax.set_xlabel('N (Temporal Steps)')
        ax.set_ylabel('Gamma Error (%)')
        ax.set_title('Gamma Accuracy vs Temporal Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # Plot 4: Computation time vs N
        ax = axes[1, 1]
        for method, color in zip(methods, colors):
            col = f'{method}_time_ms'
            if col in df_N.columns:
                ax.plot(df_N['N'], df_N[col], marker='o', label=method.replace('_', ' ').title(), color=color, alpha=0.7)
        ax.set_xlabel('N (Temporal Steps)')
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Computation Time vs Temporal Resolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / "grid_sensitivity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {self.output_dir / 'grid_sensitivity_analysis.png'}")

    def plot_moneyness_sensitivity(self):
        """Plot sensitivity to moneyness (ATM/ITM/OTM)"""
        print("Generating moneyness sensitivity plots...")

        # Filter moneyness scenarios
        df_money = self.df[self.df['scenario'].str.contains('Moneyness', na=False)].copy()

        if len(df_money) == 0:
            print("  ⚠ No moneyness data found")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        methods = ['bumping', 'double_aad', 'edge_pushing']
        colors = ['blue', 'orange', 'red']
        greeks = ['delta', 'gamma']

        for idx, greek in enumerate(greeks):
            ax = axes[idx]

            x_labels = df_money['scenario'].str.replace('Moneyness_', '').values
            x_pos = np.arange(len(x_labels))

            width = 0.25
            for i, (method, color) in enumerate(zip(methods, colors)):
                col = f'{method}_{greek}_err'
                if col in df_money.columns:
                    errors = df_money[col].values
                    ax.bar(x_pos + i*width, errors, width, label=method.replace('_', ' ').title(), color=color, alpha=0.7)

            ax.set_xlabel('Moneyness')
            ax.set_ylabel('Relative Error (%)')
            ax.set_title(f'{greek.upper()} Error by Moneyness')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(x_labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(self.output_dir / "moneyness_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {self.output_dir / 'moneyness_sensitivity.png'}")

    def plot_sigma_sensitivity(self):
        """Plot sensitivity to volatility (sigma)"""
        print("Generating volatility sensitivity plots...")

        # Filter sigma scenarios
        df_sigma = self.df[self.df['scenario'].str.contains('Sigma', na=False)].copy()

        if len(df_sigma) == 0:
            print("  ⚠ No sigma data found")
            return

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        methods = ['bumping', 'double_aad', 'edge_pushing']
        colors = ['blue', 'orange', 'red']

        # Plot 1: Vega error vs sigma
        ax = axes[0]
        for method, color in zip(methods, colors):
            col = f'{method}_vega_err'
            if col in df_sigma.columns:
                ax.plot(df_sigma['sigma'], df_sigma[col], marker='o', label=method.replace('_', ' ').title(), color=color, alpha=0.7)
        ax.set_xlabel('Volatility (σ)')
        ax.set_ylabel('Vega Error (%)')
        ax.set_title('Vega Accuracy vs Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Volga error vs sigma
        ax = axes[1]
        for method, color in zip(methods, colors):
            col = f'{method}_volga_err'
            if col in df_sigma.columns:
                ax.plot(df_sigma['sigma'], df_sigma[col], marker='o', label=method.replace('_', ' ').title(), color=color, alpha=0.7)
        ax.set_xlabel('Volatility (σ)')
        ax.set_ylabel('Volga Error (%)')
        ax.set_title('Volga Accuracy vs Volatility')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / "sigma_sensitivity.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {self.output_dir / 'sigma_sensitivity.png'}")

    def plot_aad_graph_statistics(self):
        """Plot AAD computation graph statistics"""
        if self.graph_df is None:
            print("⚠ No AAD graph statistics available")
            return

        print("Generating AAD graph statistics plots...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Node count comparison
        ax = axes[0, 0]
        scenarios = self.graph_df.index
        ax.bar(scenarios, self.graph_df['jacobian_nodes'], alpha=0.7, label='Jacobian', color='blue')
        ax.bar(scenarios, self.graph_df['hessian_nodes'], alpha=0.7, label='Hessian', color='red')
        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Number of Nodes')
        ax.set_title('AAD Graph: Node Count')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 2: Edge count comparison
        ax = axes[0, 1]
        ax.bar(scenarios, self.graph_df['jacobian_edges'], alpha=0.7, label='Jacobian', color='blue')
        ax.bar(scenarios, self.graph_df['hessian_edges'], alpha=0.7, label='Hessian', color='red')
        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Number of Edges')
        ax.set_title('AAD Graph: Edge Count')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        # Plot 3: Time per node
        ax = axes[1, 0]
        ax.plot(scenarios, self.graph_df['jacobian_time_per_node_us'], marker='o', label='Jacobian', color='blue', alpha=0.7)
        ax.plot(scenarios, self.graph_df['hessian_time_per_node_us'], marker='o', label='Hessian', color='red', alpha=0.7)
        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Time per Node (μs)')
        ax.set_title('AAD Graph: Time per Node')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Total time
        ax = axes[1, 1]
        ax.plot(scenarios, self.graph_df['jacobian_time_ms'], marker='o', label='Jacobian', color='blue', alpha=0.7)
        ax.plot(scenarios, self.graph_df['hessian_time_ms'], marker='o', label='Hessian', color='red', alpha=0.7)
        ax.set_xlabel('Scenario Index')
        ax.set_ylabel('Total Time (ms)')
        ax.set_title('AAD Graph: Total Computation Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        plt.tight_layout()
        plt.savefig(self.output_dir / "aad_graph_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved: {self.output_dir / 'aad_graph_statistics.png'}")

    def generate_html_report(self):
        """Generate comprehensive HTML report"""
        print("Generating HTML report...")

        html_content = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Comprehensive AAD Greek Computation Test Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            color: #34495e;
            margin-top: 30px;
        }
        .summary-table {
            background-color: white;
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .summary-table th, .summary-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .summary-table th {
            background-color: #3498db;
            color: white;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .plot-container {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 5px;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
        }
        .metric {
            display: inline-block;
            background-color: white;
            padding: 15px 25px;
            margin: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #3498db;
        }
        .metric-label {
            font-size: 14px;
            color: #7f8c8d;
        }
    </style>
</head>
<body>
    <h1>Comprehensive AAD Greek Computation Test Report</h1>
    <p><strong>Generated:</strong> """ + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>

    <h2>Executive Summary</h2>
"""

        # Add summary metrics
        methods = ['bumping', 'double_aad', 'edge_pushing']

        html_content += "<div>\n"
        for method in methods:
            time_col = f'{method}_time_ms'
            if time_col in self.df.columns:
                avg_time = self.df[time_col].mean()
                html_content += f"""
    <div class="metric">
        <div class="metric-label">{method.replace('_', ' ').title()} Avg Time</div>
        <div class="metric-value">{avg_time:.2f} ms</div>
    </div>
"""
        html_content += "</div>\n"

        # Add accuracy summary table
        html_content += """
    <h2>Average Accuracy Summary</h2>
    <table class="summary-table">
        <tr>
            <th>Method</th>
            <th>Delta Error (%)</th>
            <th>Gamma Error (%)</th>
            <th>Vega Error (%)</th>
            <th>Vanna Error (%)</th>
            <th>Volga Error (%)</th>
            <th>Rho Error (%)</th>
        </tr>
"""

        greeks = ['delta', 'gamma', 'vega', 'vanna', 'volga', 'rho']
        for method in methods:
            html_content += f"        <tr><td>{method.replace('_', ' ').title()}</td>"
            for greek in greeks:
                col = f'{method}_{greek}_err'
                if col in self.df.columns:
                    avg_err = self.df[col].mean()
                    html_content += f"<td>{avg_err:.2f}</td>"
                else:
                    html_content += "<td>N/A</td>"
            html_content += "</tr>\n"

        html_content += """
    </table>

    <h2>Visualizations</h2>
"""

        # Add plot images
        plots = [
            ("accuracy_comparison_all_greeks.png", "Accuracy Comparison Across All Greeks"),
            ("speed_comparison.png", "Computation Speed Comparison"),
            ("grid_sensitivity_analysis.png", "Grid Sensitivity Analysis"),
            ("moneyness_sensitivity.png", "Moneyness Sensitivity"),
            ("sigma_sensitivity.png", "Volatility Sensitivity"),
            ("aad_graph_statistics.png", "AAD Computation Graph Statistics")
        ]

        for plot_file, plot_title in plots:
            plot_path = self.output_dir / plot_file
            if plot_path.exists():
                html_content += f"""
    <div class="plot-container">
        <h3>{plot_title}</h3>
        <img src="{plot_file}" alt="{plot_title}">
    </div>
"""

        html_content += """
</body>
</html>
"""

        # Save HTML report
        html_file = self.output_dir / "comprehensive_report.html"
        with open(html_file, 'w') as f:
            f.write(html_content)

        print(f"  ✓ Saved: {html_file}")

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)

        self.plot_accuracy_comparison()
        self.plot_speed_comparison()
        self.plot_grid_sensitivity()
        self.plot_moneyness_sensitivity()
        self.plot_sigma_sensitivity()
        self.plot_aad_graph_statistics()
        self.generate_html_report()

        print("\n" + "="*80)
        print("VISUALIZATION COMPLETE")
        print("="*80)
        print(f"\nAll outputs saved to: {self.output_dir.absolute()}")
        print(f"Open the HTML report: {(self.output_dir / 'comprehensive_report.html').absolute()}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python visualize_comprehensive_results.py <results_csv> [graph_csv]")
        print("\nSearching for latest results files...")

        # Find latest results file
        results_files = sorted(Path('.').glob('comprehensive_test_results_*.csv'), reverse=True)
        graph_files = sorted(Path('.').glob('aad_graph_statistics_*.csv'), reverse=True)

        if not results_files:
            print("Error: No results files found. Run comprehensive_test_framework.py first.")
            sys.exit(1)

        results_csv = str(results_files[0])
        graph_csv = str(graph_files[0]) if graph_files else None

        print(f"Using results file: {results_csv}")
        if graph_csv:
            print(f"Using graph file: {graph_csv}")
    else:
        results_csv = sys.argv[1]
        graph_csv = sys.argv[2] if len(sys.argv) > 2 else None

    visualizer = ComprehensiveResultsVisualizer(results_csv, graph_csv)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
