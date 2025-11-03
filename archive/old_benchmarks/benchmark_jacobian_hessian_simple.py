"""
简化Jacobian/Hessian基准测试
=================================

只测试最关键的Greeks:
- Jacobian: [Delta, Vega]  (2个参数: S0, σ)
- Hessian: [[Gamma, Vanna], [Vanna, Volga]]  (2×2矩阵)

这样可以专注于验证AAD+Edge-Pushing框架的正确性
"""

import numpy as np
import time
import sys
from pathlib import Path
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import (
    CapriottiCNAAD
)


def bs_greeks_analytical(S, K, T, r, sigma):
    """BS解析Greeks"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    phi_d1 = norm.pdf(d1)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = phi_d1 / (S * sigma * sqrt_T)
    vega = S * phi_d1 * sqrt_T
    vanna = -phi_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


class Method1_Bumping_Simple:
    """Bumping方法 - 只计算S0和σ的导数"""

    def __init__(self, M, N):
        self.M = M
        self.N = N
        from aad_edge_pushing.pde.handcraft_aad.core.local_vol_solver import LocalVolSolver
        self.solver = LocalVolSolver(M, N)

    def _solve_pde(self, S0, K, T, r, sigma):
        sigma_grid = np.full((self.M+1, self.N+1), sigma)
        self.solver.set_local_vol_grid(sigma_grid)
        price, _, _ = self.solver.solve_local_vol(S0, K, T, r, 'C')
        return price

    def compute_greeks(self, S0, K, T, r, sigma):
        """计算Jacobian [∂V/∂S0, ∂V/∂σ] 和 Hessian 2×2"""
        t_start = time.perf_counter()

        eps_S = 1e-4
        eps_sigma = 1e-4 * sigma

        # 价格
        V0 = self._solve_pde(S0, K, T, r, sigma)

        # Delta: ∂V/∂S0
        V_S_plus = self._solve_pde(S0 + eps_S, K, T, r, sigma)
        V_S_minus = self._solve_pde(S0 - eps_S, K, T, r, sigma)
        delta = (V_S_plus - V_S_minus) / (2 * eps_S)

        # Gamma: ∂²V/∂S0²
        gamma = (V_S_plus - 2 * V0 + V_S_minus) / (eps_S ** 2)

        # Vega: ∂V/∂σ
        V_sigma_plus = self._solve_pde(S0, K, T, r, sigma + eps_sigma)
        V_sigma_minus = self._solve_pde(S0, K, T, r, sigma - eps_sigma)
        vega = (V_sigma_plus - V_sigma_minus) / (2 * eps_sigma)

        # Volga: ∂²V/∂σ²
        volga = (V_sigma_plus - 2 * V0 + V_sigma_minus) / (eps_sigma ** 2)

        # Vanna: ∂²V/∂S0∂σ
        V_S_plus_sigma_plus = self._solve_pde(S0 + eps_S, K, T, r, sigma + eps_sigma)
        V_S_plus_sigma_minus = self._solve_pde(S0 + eps_S, K, T, r, sigma - eps_sigma)
        V_S_minus_sigma_plus = self._solve_pde(S0 - eps_S, K, T, r, sigma + eps_sigma)
        V_S_minus_sigma_minus = self._solve_pde(S0 - eps_S, K, T, r, sigma - eps_sigma)

        vanna = (V_S_plus_sigma_plus - V_S_plus_sigma_minus -
                 V_S_minus_sigma_plus + V_S_minus_sigma_minus) / (4 * eps_S * eps_sigma)

        t_end = time.perf_counter()

        return {
            'price': V0,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'vanna': vanna,
            'volga': volga,
            'time_ms': (t_end - t_start) * 1000,
            'n_pde_solves': 9
        }


class Method3_EdgePushing_Simple:
    """AAD + Edge-Pushing - 使用现有capriotti实现"""

    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.solver = CapriottiCNAAD(M=M+2, N=N)

    def compute_greeks(self, S0, K, T, r, sigma):
        """使用AAD计算Greeks"""
        self.solver.S0 = S0
        self.solver.K = K
        self.solver.T = T
        self.solver.r = r

        t_start = time.perf_counter()
        greeks = self.solver.compute_greeks_aad(sigma_value=sigma)
        t_end = time.perf_counter()

        greeks['time_ms'] = (t_end - t_start) * 1000
        return greeks


def run_single_test(M, N, S0, K, T, r, sigma, method_name, method_class):
    """运行单个方法测试"""
    print(f"\n  {'='*70}")
    print(f"  {method_name}")
    print(f"  {'='*70}")

    try:
        method = method_class(M, N)

        print(f"  🔄 运行中...")
        result = method.compute_greeks(S0, K, T, r, sigma)

        # 解析解
        analytical = bs_greeks_analytical(S0, K, T, r, sigma)

        # 误差
        errors = {
            'price': abs(result['price'] - analytical['price']),
            'delta': abs(result['delta'] - analytical['delta']),
            'gamma': abs(result['gamma'] - analytical['gamma']),
            'vega': abs(result['vega'] - analytical['vega']),
            'vanna': abs(result['vanna'] - analytical['vanna']),
            'volga': abs(result['volga'] - analytical['volga'])
        }

        # 打印结果
        print(f"\n  Greeks对比:")
        print(f"  {'Greek':<10} | {'计算值':<12} | {'解析值':<12} | {'绝对误差':<12} | {'相对误差':<12}")
        print(f"  {'-'*70}")

        for greek in ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']:
            anal_val = analytical[greek]
            comp_val = result[greek]
            abs_err = errors[greek]
            rel_err = abs_err / abs(anal_val) if abs(anal_val) > 1e-10 else 0

            print(f"  {greek:<10} | {comp_val:<12.6f} | {anal_val:<12.6f} | "
                  f"{abs_err:<12.2e} | {rel_err:<12.2%}")

        print(f"\n  ⏱️  计算时间: {result['time_ms']:.2f} ms")
        if 'n_pde_solves' in result:
            print(f"  🔢 PDE求解次数: {result['n_pde_solves']}")

        return {
            'success': True,
            'result': result,
            'errors': errors,
            'analytical': analytical
        }

    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def run_benchmark():
    """运行基准测试"""

    print("="*80)
    print("  简化Jacobian/Hessian基准测试")
    print("  参数: θ = [S0, σ]  (2×2 Hessian)")
    print("="*80)

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # 测试配置 (只测试一个小网格以避免超时)
    test_configs = [
        {'M': 10, 'N': 30, 'name': '小网格 (M=10, N=30)'},
    ]

    results_all = []

    for config in test_configs:
        M, N = config['M'], config['N']

        print(f"\n{'#'*80}")
        print(f"# {config['name']}")
        print(f"# 空间步数 M = {M}, 时间步数 N = {N}")
        print(f"{'#'*80}")

        config_results = {'M': M, 'N': N, 'config_name': config['name']}

        # Method 1: Bumping
        result1 = run_single_test(M, N, S0, K, T, r, sigma,
                                 "Method 1: Bumping (有限差分)",
                                 Method1_Bumping_Simple)
        config_results['bumping'] = result1

        # Method 3: Edge-Pushing
        result3 = run_single_test(M, N, S0, K, T, r, sigma,
                                 "Method 3: AAD + Edge-Pushing",
                                 Method3_EdgePushing_Simple)
        config_results['edge_pushing'] = result3

        results_all.append(config_results)

        # 对比总结
        if result1['success'] and result3['success']:
            print(f"\n  {'='*70}")
            print(f"  对比总结")
            print(f"  {'='*70}")

            time1 = result1['result']['time_ms']
            time3 = result3['result']['time_ms']
            speedup = time1 / time3 if time3 > 0 else 0

            print(f"\n  性能对比:")
            print(f"  {'方法':<20} | {'时间(ms)':<12} | {'PDE次数':<10} | {'速度比':<10}")
            print(f"  {'-'*65}")
            print(f"  {'Bumping':<20} | {time1:<12.2f} | {result1['result']['n_pde_solves']:<10} | {'1.0x':<10}")
            print(f"  {'Edge-Pushing':<20} | {time3:<12.2f} | {'1':<10} | {speedup:<10.2f}x")

            print(f"\n  精度对比 (相对误差):")
            print(f"  {'Greek':<12} | {'Bumping':<15} | {'Edge-Pushing':<15} | {'更准确':<10}")
            print(f"  {'-'*65}")

            for greek in ['delta', 'gamma', 'vega', 'vanna', 'volga']:
                err1 = result1['errors'][greek] / abs(result1['analytical'][greek])
                err3 = result3['errors'][greek] / abs(result3['analytical'][greek])
                winner = 'Bumping' if err1 < err3 else 'Edge-Push'
                print(f"  {greek:<12} | {err1:<15.2%} | {err3:<15.2%} | {winner:<10}")

    return results_all


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  简化Greeks基准测试: AAD + Edge-Pushing vs Bumping")
    print("  测试网格: 从小到大，验证收敛性和性能")
    print("="*80)

    results = run_benchmark()

    print("\n\n" + "="*80)
    print("  测试完成!")
    print("="*80)
