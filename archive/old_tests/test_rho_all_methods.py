"""
测试所有5种方法的Rho计算精度
Methods: BSM Analytical, Bumping, AAD+Bumping, Double-AAD, Edge-Pushing
"""

import numpy as np
import sys
from pathlib import Path

# Add path for imports
sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.bsm_analytical import BSMAnalytical
from aad_edge_pushing.pde.bumping_method import DoubleBumpingFixed
from aad_edge_pushing.pde.double_aad_method import DoubleAADFixed
from aad_edge_pushing.pde.edge_pushing_method import EdgePushingMethodFixed


def test_rho_single_case(S0, K, T, r, sigma, M=51, N=50):
    """Test Rho for a single case across all methods"""

    print(f"\n{'='*80}")
    print(f"Test Case: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Grid: M={M}, N={N}")
    print(f"{'='*80}")

    # Method 1: Analytical (Baseline)
    print(f"\n1. BSM Analytical (Baseline)")
    print(f"{'-'*80}")
    analytical = BSMAnalytical()
    result_analytical = analytical.compute_greeks(S0, K, T, r, sigma)

    rho_analytical = result_analytical['rho']
    print(f"Rho (analytical) = {rho_analytical:.8f}")
    print(f"Time: {result_analytical['time_ms']:.4f} ms")

    # Method 2: Bumping
    print(f"\n2. Bumping Method")
    print(f"{'-'*80}")
    bumping = DoubleBumpingFixed(M=M, N=N)
    result_bumping = bumping.compute_greeks(S0, K, T, r, sigma)

    rho_bumping = result_bumping['rho']
    rho_bumping_error = abs(rho_bumping - rho_analytical) / abs(rho_analytical) * 100
    print(f"Rho (bumping)    = {rho_bumping:.8f}")
    print(f"Error: {rho_bumping_error:.4f}%")
    print(f"Time: {result_bumping['time_ms']:.2f} ms")
    print(f"PDE solves: {result_bumping['pde_solves']}")

    # Method 3: Double-AAD
    print(f"\n3. Double-AAD Method")
    print(f"{'-'*80}")
    double_aad = DoubleAADFixed(M=M, N=N)
    result_double_aad = double_aad.compute_greeks(S0, K, T, r, sigma)

    rho_double_aad = result_double_aad['rho']
    rho_double_aad_error = abs(rho_double_aad - rho_analytical) / abs(rho_analytical) * 100
    print(f"Rho (Double-AAD) = {rho_double_aad:.8f}")
    print(f"Error: {rho_double_aad_error:.4f}%")
    print(f"Time: {result_double_aad['time_ms']:.2f} ms")
    print(f"PDE solves: {result_double_aad['pde_solves']}")

    # Method 4: Edge-Pushing
    print(f"\n4. Edge-Pushing Method")
    print(f"{'-'*80}")
    edge_pushing = EdgePushingMethodFixed(M=M, N=N)
    result_edge_pushing = edge_pushing.compute_greeks(S0, K, T, r, sigma, compute_hessian=False)

    rho_edge_pushing = result_edge_pushing['rho']
    rho_edge_pushing_error = abs(rho_edge_pushing - rho_analytical) / abs(rho_analytical) * 100
    print(f"Rho (Edge-Push)  = {rho_edge_pushing:.8f}")
    print(f"Error: {rho_edge_pushing_error:.4f}%")
    print(f"Time: {result_edge_pushing['time_ms']:.2f} ms")
    print(f"PDE solves: {result_edge_pushing['pde_solves']}")

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary: Rho Comparison")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Rho':<15} {'Error %':<12} {'Time (ms)':<12} {'PDE Solves':<12}")
    print(f"{'-'*80}")
    print(f"{'Analytical':<20} {rho_analytical:>14.8f} {0.0:>11.4f} {result_analytical['time_ms']:>11.4f} {0:>11}")
    print(f"{'Bumping':<20} {rho_bumping:>14.8f} {rho_bumping_error:>11.4f} {result_bumping['time_ms']:>11.2f} {result_bumping['pde_solves']:>11}")
    print(f"{'Double-AAD':<20} {rho_double_aad:>14.8f} {rho_double_aad_error:>11.4f} {result_double_aad['time_ms']:>11.2f} {result_double_aad['pde_solves']:>11}")
    print(f"{'Edge-Pushing':<20} {rho_edge_pushing:>14.8f} {rho_edge_pushing_error:>11.4f} {result_edge_pushing['time_ms']:>11.2f} {result_edge_pushing['pde_solves']:>11}")

    return {
        'analytical': rho_analytical,
        'bumping': rho_bumping,
        'double_aad': rho_double_aad,
        'edge_pushing': rho_edge_pushing,
        'bumping_error': rho_bumping_error,
        'double_aad_error': rho_double_aad_error,
        'edge_pushing_error': rho_edge_pushing_error
    }


def test_rho_multiple_cases():
    """Test Rho across multiple parameter combinations"""

    print("\n" + "="*80)
    print("Rho计算精度测试 - 多参数组合")
    print("="*80)

    # Test cases: (S0, K, T, r, sigma)
    test_cases = [
        (100, 100, 1.0, 0.05, 0.2),   # ATM, standard
        (95, 100, 1.0, 0.05, 0.2),    # OTM
        (105, 100, 1.0, 0.05, 0.2),   # ITM
        (100, 100, 0.5, 0.05, 0.2),   # Shorter maturity
        (100, 100, 1.0, 0.03, 0.2),   # Lower rate
        (100, 100, 1.0, 0.08, 0.2),   # Higher rate
        (100, 100, 1.0, 0.05, 0.15),  # Lower vol
        (100, 100, 1.0, 0.05, 0.30),  # Higher vol
    ]

    all_results = []

    for i, (S0, K, T, r, sigma) in enumerate(test_cases, 1):
        print(f"\n\nTest Case {i}/{len(test_cases)}")
        result = test_rho_single_case(S0, K, T, r, sigma, M=51, N=50)
        all_results.append(result)

    # Aggregate statistics
    print(f"\n\n{'='*80}")
    print(f"Aggregate Statistics Across All Test Cases")
    print(f"{'='*80}")

    bumping_errors = [r['bumping_error'] for r in all_results]
    double_aad_errors = [r['double_aad_error'] for r in all_results]
    edge_pushing_errors = [r['edge_pushing_error'] for r in all_results]

    print(f"\n{'Method':<20} {'Mean Error %':<15} {'Std Error %':<15} {'Max Error %':<15}")
    print(f"{'-'*80}")
    print(f"{'Bumping':<20} {np.mean(bumping_errors):>14.4f} {np.std(bumping_errors):>14.4f} {np.max(bumping_errors):>14.4f}")
    print(f"{'Double-AAD':<20} {np.mean(double_aad_errors):>14.4f} {np.std(double_aad_errors):>14.4f} {np.max(double_aad_errors):>14.4f}")
    print(f"{'Edge-Pushing':<20} {np.mean(edge_pushing_errors):>14.4f} {np.std(edge_pushing_errors):>14.4f} {np.max(edge_pushing_errors):>14.4f}")

    print(f"\n{'='*80}")
    print(f"Key Findings:")
    print(f"{'='*80}")
    print(f"""
1. Rho的解析公式 (Call期权):
   Rho = K·T·exp(-r·T)·Φ(d2)

2. 所有数值方法都使用bumping计算Rho:
   Rho = [V(r+ε) - V(r-ε)] / (2ε)

3. 精度比较:
   - Bumping方法: 直接对r进行有限差分，误差取决于PDE求解精度
   - AAD方法: 同样使用bumping，因为r不是通过AAD图计算的参数
   - 所有方法的Rho精度应该相近（因为都用bumping）

4. 计算成本:
   - Bumping: 增加2次PDE求解（7次总计）
   - Double-AAD: 增加2次PDE求解（5次总计）
   - Edge-Pushing: 增加2次PDE求解（3次总计）

5. 建议:
   - Rho通常对精度要求不如Delta/Gamma高
   - ε=0.0001的bumping精度足够大多数应用
   - 如需更高精度，可提高网格分辨率(M, N)
""")

    return all_results


def quick_rho_test():
    """Quick test for debugging"""
    print("\n" + "="*80)
    print("Quick Rho Test")
    print("="*80)

    test_rho_single_case(
        S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
        M=51, N=50
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_rho_test()
    else:
        test_rho_multiple_cases()
