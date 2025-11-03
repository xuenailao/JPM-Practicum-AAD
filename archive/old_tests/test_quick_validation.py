"""
快速验证测试 - 确保所有方法都能正常运行
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.unified_greeks_interface import UnifiedGreeksCalculator

def test_all_methods():
    """测试所有5种方法"""
    print("="*80)
    print(" "*25 + "QUICK VALIDATION TEST")
    print("="*80)

    # 测试参数
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    M, N = 51, 50

    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, sigma={sigma}")
    print(f"Grid: M={M}, N={N}")

    calc = UnifiedGreeksCalculator(M=M, N=N)
    methods = ['analytical', 'bumping', 'aad_bumping', 'double_aad', 'edge_pushing']

    print(f"\n{'Method':<20} {'Status':<15} {'Price':<12} {'Gamma':<12}")
    print("-"*80)

    results = {}
    for method in methods:
        try:
            result = calc.compute_greeks(
                S0=S0, K=K, T=T, r=r, sigma=sigma,
                method=method,
                verbose=False,
                track_graph=True
            )

            results[method] = result
            status = "✓ OK"
            price = result['price']
            gamma = result['greeks']['gamma']

            print(f"{result['method']:<20} {status:<15} {price:>10.4f}  {gamma:>10.6f}")

        except Exception as e:
            print(f"{method:<20} ✗ ERROR: {str(e)[:40]}")
            return False

    # 验证误差
    print("\n" + "="*80)
    print("ERROR VALIDATION (vs Analytical)")
    print("="*80)

    ana = results['analytical']

    print(f"\n{'Method':<20} {'Gamma Error%':<15} {'Status':<15}")
    print("-"*80)

    all_passed = True
    for method in ['bumping', 'aad_bumping', 'double_aad', 'edge_pushing']:
        result = results[method]
        gamma_err = abs(result['greeks']['gamma'] - ana['greeks']['gamma']) / ana['greeks']['gamma'] * 100

        if gamma_err < 5.0:  # 误差应该 < 5%
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
            all_passed = False

        print(f"{result['method']:<20} {gamma_err:>13.2f}%  {status:<15}")

    print("\n" + "="*80)
    if all_passed:
        print(" "*25 + "✓ ALL TESTS PASSED")
    else:
        print(" "*25 + "✗ SOME TESTS FAILED")
    print("="*80)

    return all_passed

if __name__ == "__main__":
    success = test_all_methods()
    sys.exit(0 if success else 1)
