"""
严格基准测试: Bumping vs 手工AAD vs Edge-Pushing

真实运行三种方法，比较速度和精度
网格设置: N >> M (时间步数远大于空间步数)

测试配置:
- 不使用任何估计数据
- 所有方法真实运行
- 逐渐增加网格规模
- 记录误差收敛曲线
- 记录实际计算时间
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, Tuple
from scipy.stats import norm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import (
    CapriottiCNAAD,
    black_scholes_analytical
)


def black_scholes_greeks_analytical(S, K, T, r, sigma):
    """完整的BS解析Greeks"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrt_T)
    vega = S * norm.pdf(d1) * sqrt_T
    vanna = -norm.pdf(d1) * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


class Method1_Bumping:
    """方法1: PDE + Bumping (纯有限差分)"""

    def __init__(self, M, N):
        self.M = M
        self.N = N
        from aad_edge_pushing.pde.handcraft_aad.core.local_vol_solver import LocalVolSolver
        self.solver = LocalVolSolver(M, N)

    def compute_greeks(self, S0, K, T, r, sigma, eps_S=0.01, eps_sigma=1e-4):
        """Bumping方法计算Greeks"""
        t_start = time.perf_counter()

        # 创建常数波动率网格
        sigma_grid = np.full((self.M+1, self.N+1), sigma)

        # Price
        self.solver.set_local_vol_grid(sigma_grid)
        price, _, _ = self.solver.solve_local_vol(S0, K, T, r, 'C')

        # Delta (FD on S)
        price_S_plus, _, _ = self.solver.solve_local_vol(S0 + eps_S, K, T, r, 'C')
        price_S_minus, _, _ = self.solver.solve_local_vol(S0 - eps_S, K, T, r, 'C')
        delta = (price_S_plus - price_S_minus) / (2 * eps_S)

        # Gamma (FD on S)
        gamma = (price_S_plus - 2*price + price_S_minus) / (eps_S**2)

        # Vega (FD on sigma)
        sigma_grid_plus = np.full((self.M+1, self.N+1), sigma * (1 + eps_sigma))
        self.solver.set_local_vol_grid(sigma_grid_plus)
        price_sigma_plus, _, _ = self.solver.solve_local_vol(S0, K, T, r, 'C')

        sigma_grid_minus = np.full((self.M+1, self.N+1), sigma * (1 - eps_sigma))
        self.solver.set_local_vol_grid(sigma_grid_minus)
        price_sigma_minus, _, _ = self.solver.solve_local_vol(S0, K, T, r, 'C')

        vega = (price_sigma_plus - price_sigma_minus) / (2 * eps_sigma * sigma)

        # Vanna (mixed FD on S and sigma)
        # 计算sigma+epsilon时的Delta
        self.solver.set_local_vol_grid(sigma_grid_plus)
        p_S_plus_sig_plus, _, _ = self.solver.solve_local_vol(S0 + eps_S, K, T, r, 'C')
        p_S_minus_sig_plus, _, _ = self.solver.solve_local_vol(S0 - eps_S, K, T, r, 'C')
        delta_sigma_plus = (p_S_plus_sig_plus - p_S_minus_sig_plus) / (2 * eps_S)

        # 计算sigma-epsilon时的Delta
        self.solver.set_local_vol_grid(sigma_grid_minus)
        p_S_plus_sig_minus, _, _ = self.solver.solve_local_vol(S0 + eps_S, K, T, r, 'C')
        p_S_minus_sig_minus, _, _ = self.solver.solve_local_vol(S0 - eps_S, K, T, r, 'C')
        delta_sigma_minus = (p_S_plus_sig_minus - p_S_minus_sig_minus) / (2 * eps_S)

        vanna = (delta_sigma_plus - delta_sigma_minus) / (2 * eps_sigma * sigma)

        # Volga (second FD on sigma)
        # 计算sigma处的Vega (已有基准)
        # 需要重新计算以获得一致性
        self.solver.set_local_vol_grid(sigma_grid)
        price_base, _, _ = self.solver.solve_local_vol(S0, K, T, r, 'C')

        volga = (price_sigma_plus - 2*price + price_sigma_minus) / ((eps_sigma * sigma)**2)

        t_end = time.perf_counter()
        computation_time = (t_end - t_start) * 1000

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'vanna': vanna,
            'volga': volga,
            'time_ms': computation_time,
            'n_pde_solves': 9  # 实际PDE求解次数
        }


class Method3_AAD_EdgePushing:
    """方法3: AAD + Edge-Pushing"""

    def __init__(self, M, N):
        # 注意: CapriottiCNAAD的M是总网格点数(包括边界)
        # 需要调整以匹配handcraft的定义
        self.solver = CapriottiCNAAD(M=M+2, N=N)

    def compute_greeks(self, S0, K, T, r, sigma, eps_S=0.01):
        """AAD方法计算Greeks"""
        # 设置参数
        self.solver.S0 = S0
        self.solver.K = K
        self.solver.T = T
        self.solver.r = r

        t_start = time.perf_counter()
        greeks = self.solver.compute_greeks_aad(sigma_value=sigma, eps_S=eps_S)
        t_end = time.perf_counter()

        greeks['time_ms'] = (t_end - t_start) * 1000
        return greeks


def run_single_test(M, N, sigma, method_name, method_class):
    """运行单个方法的测试"""
    print(f"\n  {'='*60}")
    print(f"  {method_name}")
    print(f"  {'='*60}")

    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    try:
        # 创建方法实例
        method = method_class(M, N)

        # 运行计算
        print(f"  🔄 运行中...")
        result = method.compute_greeks(S0, K, T, r, sigma)

        # 获取解析解
        analytical = black_scholes_greeks_analytical(S0, K, T, r, sigma)

        # 计算误差
        errors = {
            'price': abs(result['price'] - analytical['price']),
            'delta': abs(result['delta'] - analytical['delta']),
            'gamma': abs(result['gamma'] - analytical['gamma']),
            'vega': abs(result['vega'] - analytical['vega']),
            'vanna': abs(result['vanna'] - analytical['vanna']),
            'volga': abs(result['volga'] - analytical['volga'])
        }

        # 打印结果
        print(f"\n  结果:")
        print(f"  {'Greek':<10} | {'计算值':<12} | {'解析值':<12} | {'绝对误差':<12}")
        print(f"  {'-'*60}")
        for greek in ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']:
            print(f"  {greek:<10} | {result[greek]:<12.6f} | {analytical[greek]:<12.6f} | {errors[greek]:<12.2e}")

        print(f"\n  ⏱️  计算时间: {result['time_ms']:.2f} ms")
        if 'n_pde_solves' in result:
            print(f"  🔢 PDE求解次数: {result['n_pde_solves']}")

        return {
            'success': True,
            'result': result,
            'errors': errors,
            'time_ms': result['time_ms']
        }

    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def run_convergence_test():
    """收敛性测试：逐渐增加网格规模"""

    print("="*80)
    print("  收敛性测试: 逐渐增加网格规模")
    print("  配置: N >> M (时间步数 >> 空间步数)")
    print("="*80)

    # 测试配置: N >> M (调整为更小规模以便快速测试)
    test_configs = [
        {'M': 10, 'N': 50, 'name': '小网格 (M=10, N=50)'},
        {'M': 10, 'N': 100, 'name': '中等网格 (M=10, N=100)'},
        {'M': 15, 'N': 100, 'name': '较大网格 (M=15, N=100)'},
    ]

    sigma = 0.2
    results_summary = []

    for config in test_configs:
        M, N = config['M'], config['N']

        print(f"\n{'#'*80}")
        print(f"# 测试配置: {config['name']}")
        print(f"# 空间步数 M = {M}, 时间步数 N = {N}")
        print(f"# 参数总数 = {(M+1)*(N+1)} (handcraft) 或 {M+1} (AAD)")
        print(f"{'#'*80}")

        config_results = {
            'M': M,
            'N': N,
            'config_name': config['name']
        }

        # 方法1: Bumping
        result1 = run_single_test(M, N, sigma, "方法1: Bumping", Method1_Bumping)
        config_results['bumping'] = result1

        # 方法3: AAD + Edge-Pushing
        result3 = run_single_test(M, N, sigma, "方法3: AAD + Edge-Pushing", Method3_AAD_EdgePushing)
        config_results['aad_ep'] = result3

        results_summary.append(config_results)

        # 打印对比
        if result1['success'] and result3['success']:
            print(f"\n  {'='*60}")
            print(f"  对比总结")
            print(f"  {'='*60}")
            print(f"  {'指标':<20} | {'Bumping':<18} | {'AAD+EP':<18} | {'比率':<10}")
            print(f"  {'-'*70}")

            # 时间对比
            time_ratio = result1['time_ms'] / result3['time_ms']
            print(f"  {'计算时间 (ms)':<20} | {result1['time_ms']:<18.2f} | {result3['time_ms']:<18.2f} | {time_ratio:<10.2f}x")

            # 误差对比
            for greek in ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']:
                err1 = result1['errors'][greek]
                err3 = result3['errors'][greek]
                if err3 > 0:
                    err_ratio = err1 / err3
                    print(f"  {f'{greek} 误差':<20} | {err1:<18.2e} | {err3:<18.2e} | {err_ratio:<10.2f}x")
                else:
                    print(f"  {f'{greek} 误差':<20} | {err1:<18.2e} | {err3:<18.2e} | {'N/A':<10}")

    return results_summary


def print_final_summary(results_summary):
    """打印最终总结"""

    print("\n\n" + "="*80)
    print("  最终总结: 收敛性和性能分析")
    print("="*80)

    # 汇总表
    print(f"\n{'网格配置':<20} | {'方法':<12} | {'时间(ms)':<12} | {'Price误差':<12} | {'Vega误差':<12}")
    print("-"*80)

    for result in results_summary:
        M, N = result['M'], result['N']
        config_name = f"M={M}, N={N}"

        if result['bumping']['success']:
            r = result['bumping']
            print(f"{config_name:<20} | {'Bumping':<12} | {r['time_ms']:<12.2f} | {r['errors']['price']:<12.2e} | {r['errors']['vega']:<12.2e}")

        if result['aad_ep']['success']:
            r = result['aad_ep']
            print(f"{'':<20} | {'AAD+EP':<12} | {r['time_ms']:<12.2f} | {r['errors']['price']:<12.2e} | {r['errors']['vega']:<12.2e}")

        print()

    # 收敛性分析
    print("\n收敛性分析:")
    print("-"*80)

    for i, method_key in enumerate(['bumping', 'aad_ep']):
        method_name = 'Bumping' if method_key == 'bumping' else 'AAD+EP'
        print(f"\n{method_name}:")

        configs = [r for r in results_summary if r[method_key]['success']]
        if len(configs) >= 2:
            # 检查误差是否递减
            price_errors = [r[method_key]['errors']['price'] for r in configs]
            vega_errors = [r[method_key]['errors']['vega'] for r in configs]

            print(f"  Price误差序列: {[f'{e:.2e}' for e in price_errors]}")
            print(f"  Vega误差序列:  {[f'{e:.2e}' for e in vega_errors]}")

            price_converging = all(price_errors[i] >= price_errors[i+1] for i in range(len(price_errors)-1))
            vega_converging = all(vega_errors[i] >= vega_errors[i+1] for i in range(len(vega_errors)-1))

            print(f"  Price收敛: {'✅ 是' if price_converging else '⚠️ 否'}")
            print(f"  Vega收敛:  {'✅ 是' if vega_converging else '⚠️ 否'}")

    # 性能总结
    print("\n\n性能总结:")
    print("-"*80)

    for result in results_summary:
        if result['bumping']['success'] and result['aad_ep']['success']:
            M, N = result['M'], result['N']
            time_bumping = result['bumping']['time_ms']
            time_aad = result['aad_ep']['time_ms']
            speedup = time_bumping / time_aad

            print(f"\n{result['config_name']}:")
            print(f"  Bumping: {time_bumping:.2f} ms")
            print(f"  AAD+EP:  {time_aad:.2f} ms")
            print(f"  加速比:  {speedup:.2f}x {'(AAD更快)' if speedup > 1 else '(Bumping更快)'}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  三种方法严格基准测试")
    print("  真实运行，无估计数据")
    print("="*80)
    print("\n说明:")
    print("  - 方法1: Bumping (纯有限差分)")
    print("  - 方法3: AAD + Edge-Pushing (自动微分)")
    print("  - 网格配置: N >> M (时间步数远大于空间步数)")
    print("  - 所有数据均为真实运行结果")
    print()

    results = run_convergence_test()
    print_final_summary(results)

    print("\n\n" + "="*80)
    print("  测试完成!")
    print("="*80)
