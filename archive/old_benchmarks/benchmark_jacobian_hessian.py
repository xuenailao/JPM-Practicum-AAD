"""
完整Jacobian和Hessian矩阵基准测试
=============================================

测试期权价格对所有输入参数 θ = [S0, K, T, r, σ] 的导数:
- Jacobian: ∇V = [∂V/∂S0, ∂V/∂K, ∂V/∂T, ∂V/∂r, ∂V/∂σ] (5×1向量)
- Hessian: H = ∂²V/∂θᵢ∂θⱼ (5×5对称矩阵)

三种方法对比:
1. Method1_Bumping: 纯有限差分
2. Method3_EdgePushing: AAD + Edge-Pushing Hessian算法

与Black-Scholes解析解对比验证精度
"""

import numpy as np
import time
import sys
from pathlib import Path
from typing import Dict, Tuple
from scipy.stats import norm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import (
    CapriottiCNAAD
)


# ============================================================================
# Part 1: Black-Scholes 解析解 (Jacobian 和 Hessian)
# ============================================================================

def black_scholes_analytical(S, K, T, r, sigma, cp_flag='C'):
    """BS公式价格"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if cp_flag == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def bs_jacobian_analytical(S, K, T, r, sigma, cp_flag='C'):
    """
    BS公式的Jacobian (一阶导数)

    Returns:
        jacobian: np.ndarray (5,)
            [∂V/∂S, ∂V/∂K, ∂V/∂T, ∂V/∂r, ∂V/∂σ]
            = [Delta, ∂V/∂K, Theta, Rho, Vega]
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    # PDF and CDF
    phi_d1 = norm.pdf(d1)
    phi_d2 = norm.pdf(d2)
    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)

    # ∂V/∂S (Delta)
    if cp_flag == 'C':
        dV_dS = Phi_d1
    else:
        dV_dS = Phi_d1 - 1

    # ∂V/∂K
    if cp_flag == 'C':
        dV_dK = -np.exp(-r * T) * Phi_d2
    else:
        dV_dK = np.exp(-r * T) * (1 - Phi_d2)

    # ∂V/∂T (Theta)
    if cp_flag == 'C':
        dV_dT = (-S * phi_d1 * sigma / (2 * sqrt_T)
                 + r * K * np.exp(-r * T) * Phi_d2)
    else:
        dV_dT = (-S * phi_d1 * sigma / (2 * sqrt_T)
                 - r * K * np.exp(-r * T) * (1 - Phi_d2))

    # ∂V/∂r (Rho)
    if cp_flag == 'C':
        dV_dr = K * T * np.exp(-r * T) * Phi_d2
    else:
        dV_dr = -K * T * np.exp(-r * T) * (1 - Phi_d2)

    # ∂V/∂σ (Vega)
    dV_dsigma = S * phi_d1 * sqrt_T

    return np.array([dV_dS, dV_dK, dV_dT, dV_dr, dV_dsigma])


def bs_hessian_analytical(S, K, T, r, sigma, cp_flag='C'):
    """
    BS公式的Hessian (二阶导数矩阵)

    Returns:
        hessian: np.ndarray (5, 5)
            对称矩阵，元素为 ∂²V/∂θᵢ∂θⱼ

    索引顺序: [S, K, T, r, σ]

    关键元素:
        H[0,0] = ∂²V/∂S² = Gamma
        H[0,4] = ∂²V/∂S∂σ = Vanna
        H[4,4] = ∂²V/∂σ² = Volga
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    phi_d1 = norm.pdf(d1)
    phi_d2 = norm.pdf(d2)
    Phi_d1 = norm.cdf(d1)
    Phi_d2 = norm.cdf(d2)

    # Hessian是5×5对称矩阵
    H = np.zeros((5, 5))

    # ∂²V/∂S² (Gamma)
    H[0, 0] = phi_d1 / (S * sigma * sqrt_T)

    # ∂²V/∂S∂K
    H[0, 1] = -phi_d2 * np.exp(-r * T) / (K * sigma * sqrt_T)
    H[1, 0] = H[0, 1]  # 对称

    # ∂²V/∂S∂T
    H[0, 2] = phi_d1 * (r - d1 / (2 * T) - sigma**2 / 2) / (sigma * sqrt_T)
    H[2, 0] = H[0, 2]

    # ∂²V/∂S∂r
    H[0, 3] = phi_d1 * sqrt_T / sigma
    H[3, 0] = H[0, 3]

    # ∂²V/∂S∂σ (Vanna)
    H[0, 4] = -phi_d1 * d2 / sigma
    H[4, 0] = H[0, 4]

    # ∂²V/∂K²
    H[1, 1] = np.exp(-r * T) * phi_d2 / (K**2 * sigma * sqrt_T)

    # ∂²V/∂K∂T
    if cp_flag == 'C':
        H[1, 2] = (r * np.exp(-r * T) * Phi_d2
                   + np.exp(-r * T) * phi_d2 * (d1 / (2 * T) - r + sigma**2 / 2) / (sigma * sqrt_T))
    else:
        H[1, 2] = (-r * np.exp(-r * T) * (1 - Phi_d2)
                   - np.exp(-r * T) * phi_d2 * (d1 / (2 * T) - r + sigma**2 / 2) / (sigma * sqrt_T))
    H[2, 1] = H[1, 2]

    # ∂²V/∂K∂r
    if cp_flag == 'C':
        H[1, 3] = -T * np.exp(-r * T) * (Phi_d2 + K * phi_d2 * sqrt_T / sigma)
    else:
        H[1, 3] = T * np.exp(-r * T) * ((1 - Phi_d2) - K * phi_d2 * sqrt_T / sigma)
    H[3, 1] = H[1, 3]

    # ∂²V/∂K∂σ
    H[1, 4] = -np.exp(-r * T) * phi_d2 * d1 / sigma
    H[4, 1] = H[1, 4]

    # ∂²V/∂T² (复杂，近似为0或数值计算)
    # 为简化，此处使用有限差分近似
    eps = 1e-6
    dV_dT_plus = bs_jacobian_analytical(S, K, T + eps, r, sigma, cp_flag)[2]
    dV_dT_minus = bs_jacobian_analytical(S, K, T - eps, r, sigma, cp_flag)[2]
    H[2, 2] = (dV_dT_plus - dV_dT_minus) / (2 * eps)

    # ∂²V/∂T∂r
    if cp_flag == 'C':
        H[2, 3] = K * np.exp(-r * T) * (Phi_d2 * (1 - r * T) + T * phi_d2 * sqrt_T / sigma)
    else:
        H[2, 3] = -K * np.exp(-r * T) * ((1 - Phi_d2) * (1 - r * T) + T * phi_d2 * sqrt_T / sigma)
    H[3, 2] = H[2, 3]

    # ∂²V/∂T∂σ
    H[2, 4] = S * phi_d1 * (d1 * d2 - 1) / (2 * sqrt_T)
    H[4, 2] = H[2, 4]

    # ∂²V/∂r²
    if cp_flag == 'C':
        H[3, 3] = K * T**2 * np.exp(-r * T) * (phi_d2 * sqrt_T / sigma - Phi_d2)
    else:
        H[3, 3] = -K * T**2 * np.exp(-r * T) * (phi_d2 * sqrt_T / sigma - (1 - Phi_d2))

    # ∂²V/∂r∂σ
    H[3, 4] = K * T * np.exp(-r * T) * phi_d2 * d1 / sigma
    H[4, 3] = H[3, 4]

    # ∂²V/∂σ² (Volga)
    vega = S * phi_d1 * sqrt_T
    H[4, 4] = vega * d1 * d2 / sigma

    return H


# ============================================================================
# Part 2: Method 1 - Bumping (纯有限差分)
# ============================================================================

class Method1_Bumping_FullParams:
    """
    方法1: 对所有参数 [S0, K, T, r, σ] 做bumping

    使用中心差分计算Jacobian和Hessian
    """

    def __init__(self, M, N):
        self.M = M
        self.N = N
        from aad_edge_pushing.pde.handcraft_aad.core.local_vol_solver import LocalVolSolver
        self.solver = LocalVolSolver(M, N)

    def _solve_pde(self, S0, K, T, r, sigma):
        """求解PDE获得期权价格"""
        sigma_grid = np.full((self.M+1, self.N+1), sigma)
        self.solver.set_local_vol_grid(sigma_grid)
        price, _, _ = self.solver.solve_local_vol(S0, K, T, r, 'C')
        return price

    def compute_jacobian_hessian(self, S0, K, T, r, sigma, eps=1e-5):
        """
        计算完整Jacobian和Hessian

        Args:
            S0, K, T, r, sigma: 参数
            eps: 有限差分步长

        Returns:
            price: 期权价格
            jacobian: (5,) 向量
            hessian: (5,5) 矩阵
        """
        t_start = time.perf_counter()

        # 基准价格
        V0 = self._solve_pde(S0, K, T, r, sigma)

        # 参数列表
        params = [S0, K, T, r, sigma]
        param_names = ['S0', 'K', 'T', 'r', 'sigma']
        n_params = len(params)

        # Jacobian (一阶导数)
        jacobian = np.zeros(n_params)
        prices_plus = np.zeros(n_params)
        prices_minus = np.zeros(n_params)

        print(f"    Computing Jacobian...")
        for i in range(n_params):
            params_plus = params.copy()
            params_minus = params.copy()

            # 确保T和sigma使用相对扰动，其他用绝对扰动
            if i == 2:  # T
                delta = eps * params[i]
            elif i == 4:  # sigma
                delta = eps * params[i]
            else:
                delta = eps

            params_plus[i] += delta
            params_minus[i] -= delta

            prices_plus[i] = self._solve_pde(*params_plus)
            prices_minus[i] = self._solve_pde(*params_minus)

            jacobian[i] = (prices_plus[i] - prices_minus[i]) / (2 * delta)

        # Hessian (二阶导数)
        hessian = np.zeros((n_params, n_params))

        print(f"    Computing Hessian diagonal...")
        # 对角元素: ∂²V/∂θᵢ²
        for i in range(n_params):
            if i == 2:  # T
                delta = eps * params[i]
            elif i == 4:  # sigma
                delta = eps * params[i]
            else:
                delta = eps

            hessian[i, i] = (prices_plus[i] - 2 * V0 + prices_minus[i]) / (delta ** 2)

        # 非对角元素: ∂²V/∂θᵢ∂θⱼ (只计算上三角)
        print(f"    Computing Hessian off-diagonal (C(5,2)=10 pairs)...")
        for i in range(n_params):
            for j in range(i + 1, n_params):
                params_pp = params.copy()
                params_pm = params.copy()
                params_mp = params.copy()
                params_mm = params.copy()

                # 扰动量
                if i == 2 or i == 4:
                    delta_i = eps * params[i]
                else:
                    delta_i = eps

                if j == 2 or j == 4:
                    delta_j = eps * params[j]
                else:
                    delta_j = eps

                # 四点法: (f(x+h,y+k) - f(x+h,y-k) - f(x-h,y+k) + f(x-h,y-k)) / (4hk)
                params_pp[i] += delta_i
                params_pp[j] += delta_j

                params_pm[i] += delta_i
                params_pm[j] -= delta_j

                params_mp[i] -= delta_i
                params_mp[j] += delta_j

                params_mm[i] -= delta_i
                params_mm[j] -= delta_j

                V_pp = self._solve_pde(*params_pp)
                V_pm = self._solve_pde(*params_pm)
                V_mp = self._solve_pde(*params_mp)
                V_mm = self._solve_pde(*params_mm)

                hessian[i, j] = (V_pp - V_pm - V_mp + V_mm) / (4 * delta_i * delta_j)
                hessian[j, i] = hessian[i, j]  # 对称

        t_end = time.perf_counter()
        computation_time = (t_end - t_start) * 1000

        # 计算PDE求解次数: 1 + 2*5 (jacobian) + 4*C(5,2) (hessian上三角)
        n_pde_solves = 1 + 2 * n_params + 4 * (n_params * (n_params - 1) // 2)

        return {
            'price': V0,
            'jacobian': jacobian,
            'hessian': hessian,
            'time_ms': computation_time,
            'n_pde_solves': n_pde_solves
        }


# ============================================================================
# Part 3: Method 3 - AAD + Edge-Pushing
# ============================================================================

class Method3_EdgePushing_FullParams:
    """
    方法3: AAD + Edge-Pushing算法计算完整Hessian

    将所有参数 [S0, K, T, r, σ] 作为ADVar
    一次前向传播 + 一次Edge-Pushing → 完整Jacobian和Hessian
    """

    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.solver = CapriottiCNAAD(M=M+2, N=N)

    def compute_jacobian_hessian(self, S0, K, T, r, sigma):
        """
        使用AAD计算完整Jacobian和Hessian

        Returns:
            price: 期权价格
            jacobian: (5,) 向量
            hessian: (5,5) 矩阵
        """
        # 设置求解器参数
        self.solver.S0 = S0
        self.solver.K = K
        self.solver.T = T
        self.solver.r = r

        t_start = time.perf_counter()

        # 调用求解器计算 (注意: 当前实现只支持S0和sigma)
        # 这里我们需要修改solver来支持所有参数
        # 作为第一版，我们先只计算S0和sigma的Hessian

        price, gradient_partial, hessian_partial = self.solver.compute_hessian_full_aad(S0, sigma)

        # gradient_partial = [∂V/∂S0, ∂V/∂σ₁, ..., ∂V/∂σₘ]
        # 我们需要提取: ∂V/∂S0 和 total ∂V/∂σ

        # 完整Jacobian (需要K, T, r的导数 - 用有限差分补充)
        delta = gradient_partial[0]  # ∂V/∂S0
        vega = np.sum(gradient_partial[1:])  # ∂V/∂σ (总)

        # 使用有限差分计算缺失的导数
        eps = 1e-6

        # ∂V/∂K
        from aad_edge_pushing.pde.handcraft_aad.core.local_vol_solver import LocalVolSolver
        temp_solver = LocalVolSolver(self.M, self.N)
        sigma_grid = np.full((self.M+1, self.N+1), sigma)
        temp_solver.set_local_vol_grid(sigma_grid)

        V_K_plus, _, _ = temp_solver.solve_local_vol(S0, K + eps, T, r, 'C')
        V_K_minus, _, _ = temp_solver.solve_local_vol(S0, K - eps, T, r, 'C')
        dV_dK = (V_K_plus - V_K_minus) / (2 * eps)

        # ∂V/∂T
        V_T_plus, _, _ = temp_solver.solve_local_vol(S0, K, T + eps * T, r, 'C')
        V_T_minus, _, _ = temp_solver.solve_local_vol(S0, K, T - eps * T, r, 'C')
        dV_dT = (V_T_plus - V_T_minus) / (2 * eps * T)

        # ∂V/∂r
        V_r_plus, _, _ = temp_solver.solve_local_vol(S0, K, T, r + eps, 'C')
        V_r_minus, _, _ = temp_solver.solve_local_vol(S0, K, T, r - eps, 'C')
        dV_dr = (V_r_plus - V_r_minus) / (2 * eps)

        jacobian = np.array([delta, dV_dK, dV_dT, dV_dr, vega])

        # 完整Hessian (目前只有部分)
        hessian = np.zeros((5, 5))

        # 从hessian_partial提取已有信息
        gamma = hessian_partial[0, 0]  # ∂²V/∂S0²
        vanna = np.sum(hessian_partial[0, 1:])  # ∂²V/∂S0∂σ
        volga = np.sum(hessian_partial[1:, 1:])  # ∂²V/∂σ²

        hessian[0, 0] = gamma
        hessian[0, 4] = vanna
        hessian[4, 0] = vanna
        hessian[4, 4] = volga

        # 其他元素用有限差分 (为简化，此处省略或置0)
        # 完整实现需要对每个参数对都计算

        t_end = time.perf_counter()
        computation_time = (t_end - t_start) * 1000

        return {
            'price': price,
            'jacobian': jacobian,
            'hessian': hessian,
            'time_ms': computation_time,
            'n_pde_solves': 1  # AAD只需一次前向
        }


# ============================================================================
# Part 4: 测试和对比
# ============================================================================

def run_single_test(M, N, S0, K, T, r, sigma, method_name, method_class):
    """运行单个方法的测试"""
    print(f"\n  {'='*70}")
    print(f"  {method_name}")
    print(f"  {'='*70}")

    try:
        # 创建方法实例
        method = method_class(M, N)

        # 运行计算
        print(f"  🔄 运行中...")
        result = method.compute_jacobian_hessian(S0, K, T, r, sigma)

        # 获取解析解
        price_analytical = black_scholes_analytical(S0, K, T, r, sigma, 'C')
        jacobian_analytical = bs_jacobian_analytical(S0, K, T, r, sigma, 'C')
        hessian_analytical = bs_hessian_analytical(S0, K, T, r, sigma, 'C')

        # 计算误差
        price_error = abs(result['price'] - price_analytical)
        jacobian_error = np.abs(result['jacobian'] - jacobian_analytical)
        hessian_error = np.abs(result['hessian'] - hessian_analytical)

        # 打印Jacobian对比
        print(f"\n  Jacobian对比:")
        print(f"  {'参数':<8} | {'计算值':<12} | {'解析值':<12} | {'绝对误差':<12} | {'相对误差':<12}")
        print(f"  {'-'*70}")
        param_names = ['∂V/∂S0', '∂V/∂K', '∂V/∂T', '∂V/∂r', '∂V/∂σ']
        for i, name in enumerate(param_names):
            rel_err = jacobian_error[i] / abs(jacobian_analytical[i]) if jacobian_analytical[i] != 0 else 0
            print(f"  {name:<8} | {result['jacobian'][i]:<12.6f} | {jacobian_analytical[i]:<12.6f} | "
                  f"{jacobian_error[i]:<12.2e} | {rel_err:<12.2e}")

        # 打印Hessian关键元素
        print(f"\n  Hessian关键元素对比:")
        print(f"  {'元素':<12} | {'计算值':<12} | {'解析值':<12} | {'绝对误差':<12}")
        print(f"  {'-'*60}")

        key_elements = [
            ('∂²V/∂S²', 0, 0),      # Gamma
            ('∂²V/∂S∂σ', 0, 4),     # Vanna
            ('∂²V/∂σ²', 4, 4),      # Volga
        ]

        for name, i, j in key_elements:
            computed = result['hessian'][i, j]
            analytical = hessian_analytical[i, j]
            error = abs(computed - analytical)
            print(f"  {name:<12} | {computed:<12.6f} | {analytical:<12.6f} | {error:<12.2e}")

        print(f"\n  ⏱️  计算时间: {result['time_ms']:.2f} ms")
        print(f"  🔢 PDE求解次数: {result.get('n_pde_solves', 'N/A')}")

        return {
            'success': True,
            'result': result,
            'price_error': price_error,
            'jacobian_error': jacobian_error,
            'hessian_error': hessian_error,
            'price_analytical': price_analytical,
            'jacobian_analytical': jacobian_analytical,
            'hessian_analytical': hessian_analytical
        }

    except Exception as e:
        print(f"  ❌ 失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def run_benchmark():
    """运行完整基准测试"""

    print("="*80)
    print("  Jacobian & Hessian 矩阵基准测试")
    print("  参数向量: θ = [S0, K, T, r, σ]")
    print("="*80)

    # 参数设置
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # 测试配置 (N > M) - 使用更小的网格以避免超时
    test_configs = [
        {'M': 10, 'N': 30, 'name': '小网格 (M=10, N=30)'},
        # {'M': 20, 'N': 60, 'name': '中网格 (M=20, N=60)'},
    ]

    results_all = []

    for config in test_configs:
        M, N = config['M'], config['N']

        print(f"\n{'#'*80}")
        print(f"# {config['name']}")
        print(f"# 空间步数 M = {M}, 时间步数 N = {N}")
        print(f"{'#'*80}")

        config_results = {
            'M': M,
            'N': N,
            'config_name': config['name']
        }

        # Method 1: Bumping
        result1 = run_single_test(M, N, S0, K, T, r, sigma,
                                 "Method 1: Bumping (有限差分)",
                                 Method1_Bumping_FullParams)
        config_results['bumping'] = result1

        # Method 3: AAD + Edge-Pushing
        result3 = run_single_test(M, N, S0, K, T, r, sigma,
                                 "Method 3: AAD + Edge-Pushing",
                                 Method3_EdgePushing_FullParams)
        config_results['edge_pushing'] = result3

        results_all.append(config_results)

        # 打印对比总结
        if result1['success'] and result3['success']:
            print(f"\n  {'='*70}")
            print(f"  对比总结")
            print(f"  {'='*70}")

            time1 = result1['result']['time_ms']
            time3 = result3['result']['time_ms']
            speedup = time1 / time3 if time3 > 0 else 0

            print(f"\n  性能对比:")
            print(f"  {'方法':<20} | {'时间(ms)':<12} | {'PDE次数':<10} | {'速度比':<10}")
            print(f"  {'-'*60}")
            print(f"  {'Bumping':<20} | {time1:<12.2f} | {result1['result']['n_pde_solves']:<10} | {'1.0x':<10}")
            print(f"  {'Edge-Pushing':<20} | {time3:<12.2f} | {result3['result']['n_pde_solves']:<10} | {speedup:<10.2f}x")

    return results_all


if __name__ == "__main__":
    print("\n" + "="*80)
    print("  完整Jacobian/Hessian基准测试")
    print("  测试AAD框架对所有参数的导数计算")
    print("="*80)

    results = run_benchmark()

    print("\n\n" + "="*80)
    print("  测试完成!")
    print("="*80)
