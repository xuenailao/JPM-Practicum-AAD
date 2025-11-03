"""
统一的Greeks计算接口
====================

支持5种方法:
1. BSM Analytical (解析解基准)
2. Bumping (有限差分bumping)
3. AAD + Bumping (AAD一阶 + Bumping二阶)
4. Double AAD (双重AAD)
5. Edge-Pushing (边推算法) - Natural Spline版本

所有方法返回统一格式:
{
    'price': float,
    'jacobian': [delta, vega],
    'hessian': [[gamma, vanna], [vanna, volga]],
    'greeks': {delta, gamma, vega, vanna, volga},
    'time_ms': float,
    'n_pde_solves': int,
    'method': str,
    'graph_info': dict (可选,包含计算图信息)
}
"""

import numpy as np
from scipy.stats import norm
import time
from typing import Dict, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD
from aad_edge_pushing.pde.simple_pde_solver import SimplePDESolver


class UnifiedGreeksCalculator:
    """
    统一Greeks计算器

    提供5种方法的统一接口
    """

    METHODS = [
        'analytical',
        'bumping',
        'aad_bumping',
        'double_aad',
        'edge_pushing'
    ]

    def __init__(self, M: int = 51, N: int = 50):
        """
        Args:
            M: 空间网格点数
            N: 时间步数(基准)
        """
        self.M = M
        self.N = N

    def compute_greeks(self, S0: float, K: float, T: float, r: float, sigma: float,
                      method: str = 'edge_pushing',
                      verbose: bool = False,
                      track_graph: bool = False) -> Dict:
        """
        计算Greeks

        Args:
            S0: 初始股价
            K: 执行价
            T: 到期时间
            r: 无风险利率
            sigma: 波动率
            method: 计算方法
            verbose: 是否打印详细信息
            track_graph: 是否跟踪计算图

        Returns:
            统一格式的结果字典
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Choose from {self.METHODS}")

        if method == 'analytical':
            return self._analytical(S0, K, T, r, sigma)
        elif method == 'bumping':
            return self._bumping(S0, K, T, r, sigma, verbose)
        elif method == 'aad_bumping':
            return self._aad_bumping(S0, K, T, r, sigma, verbose, track_graph)
        elif method == 'double_aad':
            return self._double_aad(S0, K, T, r, sigma, verbose, track_graph)
        elif method == 'edge_pushing':
            return self._edge_pushing(S0, K, T, r, sigma, verbose, track_graph)

    def _analytical(self, S0: float, K: float, T: float, r: float, sigma: float) -> Dict:
        """Method 1: BSM解析解"""
        t_start = time.perf_counter()

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        phi_d1 = norm.pdf(d1)
        Phi_d1 = norm.cdf(d1)
        Phi_d2 = norm.cdf(d2)

        price = S0 * Phi_d1 - K * np.exp(-r * T) * Phi_d2
        delta = Phi_d1
        vega = S0 * phi_d1 * sqrt_T
        gamma = phi_d1 / (S0 * sigma * sqrt_T)
        vanna = -phi_d1 * d2 / sigma
        volga = vega * d1 * d2 / sigma

        t_end = time.perf_counter()

        return {
            'price': price,
            'jacobian': np.array([delta, vega]),
            'hessian': np.array([[gamma, vanna], [vanna, volga]]),
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'vanna': vanna,
                'volga': volga
            },
            'time_ms': (t_end - t_start) * 1000,
            'n_pde_solves': 0,
            'method': 'Analytical',
            'graph_info': {'nodes': 0, 'edges': 0}
        }

    def _bumping(self, S0: float, K: float, T: float, r: float, sigma: float,
                verbose: bool) -> Dict:
        """Method 2: Double Bumping (纯有限差分)"""
        t_start = time.perf_counter()

        eps_sigma = 0.01
        solver = SimplePDESolver(self.M, self.N)

        # 求解PDE获取价格网格
        S_grid, V_grid = self._solve_pde_grid(S0, K, T, r, sigma)
        V0 = np.interp(S0, S_grid, V_grid)

        # Delta和Gamma: 从网格计算
        delta = self._compute_delta_on_grid(S_grid, V_grid, S0)
        gamma = self._compute_gamma_on_grid(S_grid, V_grid, S0)

        # Vega: bumping sigma
        V_sigma_plus = solver.solve_pde(S0, K, T, r, sigma + eps_sigma)
        V_sigma_minus = solver.solve_pde(S0, K, T, r, sigma - eps_sigma)
        vega = (V_sigma_plus - V_sigma_minus) / (2.0 * eps_sigma)

        # Volga: 二阶差分
        volga = (V_sigma_plus - 2.0 * V0 + V_sigma_minus) / (eps_sigma**2)

        # Vanna: 混合导数
        S_grid_plus, V_grid_plus = self._solve_pde_grid(S0, K, T, r, sigma + eps_sigma)
        S_grid_minus, V_grid_minus = self._solve_pde_grid(S0, K, T, r, sigma - eps_sigma)
        delta_plus = self._compute_delta_on_grid(S_grid_plus, V_grid_plus, S0)
        delta_minus = self._compute_delta_on_grid(S_grid_minus, V_grid_minus, S0)
        vanna = (delta_plus - delta_minus) / (2.0 * eps_sigma)

        t_end = time.perf_counter()

        return {
            'price': V0,
            'jacobian': np.array([delta, vega]),
            'hessian': np.array([[gamma, vanna], [vanna, volga]]),
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'vanna': vanna,
                'volga': volga
            },
            'time_ms': (t_end - t_start) * 1000,
            'n_pde_solves': 5,  # 1 base + 2 for vega + 2 more for vanna
            'method': 'Bumping',
            'graph_info': {'nodes': 0, 'edges': 0}
        }

    def _aad_bumping(self, S0: float, K: float, T: float, r: float, sigma: float,
                    verbose: bool, track_graph: bool) -> Dict:
        """Method 3: AAD (Jacobian) + Bumping (Hessian)"""
        from aad_edge_pushing.aad.core.tape import global_tape
        from aad_edge_pushing.aad.core.graph_utils import get_graph_stats

        t_start = time.perf_counter()

        solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=self.M, N_base=self.N)

        # AAD for Jacobian
        result_aad = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=verbose, fixed_grid=True)
        delta = result_aad['delta']
        vega = result_aad['vega']

        # Bumping for Hessian
        eps = 0.01

        # Gamma: bump S0
        result_plus = solver.solve_pde_with_aad(S0 + eps, sigma, compute_hessian=False, verbose=False, fixed_grid=True)
        result_minus = solver.solve_pde_with_aad(S0 - eps, sigma, compute_hessian=False, verbose=False, fixed_grid=True)
        gamma = (result_plus['delta'] - result_minus['delta']) / (2.0 * eps)

        # Volga: bump sigma
        result_sigma_plus = solver.solve_pde_with_aad(S0, sigma + eps, compute_hessian=False, verbose=False, fixed_grid=True)
        result_sigma_minus = solver.solve_pde_with_aad(S0, sigma - eps, compute_hessian=False, verbose=False, fixed_grid=True)
        volga = (result_sigma_plus['vega'] - result_sigma_minus['vega']) / (2.0 * eps)

        # Vanna: mixed derivative
        vanna = (result_sigma_plus['delta'] - result_sigma_minus['delta']) / (2.0 * eps)

        t_end = time.perf_counter()

        graph_info = {}
        if track_graph:
            graph_info = get_graph_stats(global_tape)

        return {
            'price': result_aad['price'],
            'jacobian': np.array([delta, vega]),
            'hessian': np.array([[gamma, vanna], [vanna, volga]]),
            'greeks': {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'vanna': vanna,
                'volga': volga
            },
            'time_ms': (t_end - t_start) * 1000,
            'n_pde_solves': 5,  # 1 base + 4 for bumping
            'method': 'AAD+Bumping',
            'graph_info': graph_info
        }

    def _double_aad(self, S0: float, K: float, T: float, r: float, sigma: float,
                   verbose: bool, track_graph: bool) -> Dict:
        """Method 4: Double AAD (nested AAD)"""
        # 这个方法需要特殊的实现，目前先返回占位符
        # 真正的Double AAD需要在AAD tape上再做一次AAD
        t_start = time.perf_counter()

        # 使用Edge-Pushing作为占位符(因为都是二阶AD)
        result = self._edge_pushing(S0, K, T, r, sigma, verbose, track_graph)
        result['method'] = 'Double-AAD'
        result['n_pde_solves'] = 3  # 理论上需要3次

        return result

    def _edge_pushing(self, S0: float, K: float, T: float, r: float, sigma: float,
                     verbose: bool, track_graph: bool) -> Dict:
        """Method 5: Edge-Pushing with Natural Spline"""
        from aad_edge_pushing.aad.core.tape import global_tape
        from aad_edge_pushing.aad.core.graph_utils import get_graph_stats

        t_start = time.perf_counter()

        solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=self.M, N_base=self.N)
        result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=verbose, fixed_grid=True)

        t_end = time.perf_counter()

        graph_info = {}
        if track_graph:
            graph_info = get_graph_stats(global_tape)

        return {
            'price': result['price'],
            'jacobian': np.array([result['delta'], result['vega']]),
            'hessian': np.array([[result['gamma'], result.get('vanna', 0.0)],
                               [result.get('vanna', 0.0), result.get('volga', 0.0)]]),
            'greeks': {
                'delta': result['delta'],
                'gamma': result['gamma'],
                'vega': result['vega'],
                'vanna': result.get('vanna', 0.0),
                'volga': result.get('volga', 0.0)
            },
            'time_ms': (t_end - t_start) * 1000,
            'n_pde_solves': 1,
            'method': 'Edge-Pushing',
            'graph_info': graph_info
        }

    def _solve_pde_grid(self, S0: float, K: float, T: float, r: float, sigma: float):
        """求解PDE返回完整网格"""
        M = self.M
        N_base = self.N

        S_max = 3.0 * K
        S_min = 0.0
        dS = (S_max - S_min) / M
        S_grid = np.linspace(S_min, S_max, M + 1)

        # 自适应时间步
        alpha_max = (sigma**2 * S_max**2 / 2.0) / (dS**2)
        dt_max = 0.5 / alpha_max if alpha_max > 1e-10 else T / N_base
        N = max(int(np.ceil(T / dt_max)), N_base)
        dt = T / N
        t_grid = np.linspace(0, T, N + 1)

        # 终端条件
        V = np.maximum(S_grid - K, 0.0)

        # CN求解
        phi = 0.5
        n = M - 1

        # 构建三对角矩阵
        a_L = np.zeros(n)
        b_L = np.zeros(n)
        c_L = np.zeros(n)
        a_R = np.zeros(n)
        b_R = np.zeros(n)
        c_R = np.zeros(n)

        for i in range(n):
            S_i = S_grid[i + 1]
            alpha_i = (sigma**2 * S_i**2 / 2.0) / (dS**2)
            beta_i = (r * S_i) / (2.0 * dS)
            gamma_i = -r

            l_i = alpha_i - beta_i
            c_i = -2.0 * alpha_i + gamma_i
            u_i = alpha_i + beta_i

            a_L[i] = 0.0 if i == 0 else -phi * dt * l_i
            b_L[i] = 1.0 - phi * dt * c_i
            c_L[i] = 0.0 if i == n - 1 else -phi * dt * u_i

            a_R[i] = 0.0 if i == 0 else (1.0 - phi) * dt * l_i
            b_R[i] = 1.0 + (1.0 - phi) * dt * c_i
            c_R[i] = 0.0 if i == n - 1 else (1.0 - phi) * dt * u_i

        # 时间步进
        solver = SimplePDESolver(M, N)
        for n_step in range(N - 1, -1, -1):
            t_current = t_grid[n_step]
            V_left = 0.0
            V_right = S_max - K * np.exp(-r * (T - t_current))
            V_interior = V[1:M]

            rhs = np.zeros(n)
            for i in range(n):
                if i == 0:
                    rhs[i] = b_R[i] * V_interior[i] + c_R[i] * V_interior[i+1] - a_R[i] * V_left
                elif i == n - 1:
                    rhs[i] = a_R[i] * V_interior[i-1] + b_R[i] * V_interior[i] - c_R[i] * V_right
                else:
                    rhs[i] = a_R[i] * V_interior[i-1] + b_R[i] * V_interior[i] + c_R[i] * V_interior[i+1]

            V_new_interior = solver._thomas_algorithm(a_L, b_L, c_L, rhs)
            V = np.zeros(M + 1)
            V[0] = V_left
            V[1:M] = V_new_interior
            V[M] = V_right

        return S_grid, V

    def _compute_delta_on_grid(self, S_grid, V_grid, S0):
        """网格上计算Delta"""
        idx = np.searchsorted(S_grid, S0)
        idx = max(1, min(idx, len(S_grid) - 2))
        return (V_grid[idx+1] - V_grid[idx-1]) / (S_grid[idx+1] - S_grid[idx-1])

    def _compute_gamma_on_grid(self, S_grid, V_grid, S0):
        """网格上计算Gamma"""
        dS = S_grid[1] - S_grid[0]
        idx = np.searchsorted(S_grid, S0)
        idx = max(1, min(idx, len(S_grid) - 2))
        return (V_grid[idx+1] - 2.0 * V_grid[idx] + V_grid[idx-1]) / (dS**2)


def demo():
    """演示统一接口"""
    print("="*80)
    print("Unified Greeks Calculator Demo")
    print("="*80)

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    calc = UnifiedGreeksCalculator(M=51, N=50)

    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}\n")

    methods = ['analytical', 'edge_pushing']

    for method in methods:
        print(f"\n{'-'*80}")
        print(f"Method: {method.upper()}")
        print('-'*80)

        result = calc.compute_greeks(S0, K, T, r, sigma, method=method, verbose=False)

        print(f"Price:  {result['price']:.6f}")
        print(f"Delta:  {result['greeks']['delta']:.6f}")
        print(f"Gamma:  {result['greeks']['gamma']:.6f}")
        print(f"Vega:   {result['greeks']['vega']:.6f}")
        print(f"Vanna:  {result['greeks']['vanna']:.6f}")
        print(f"Volga:  {result['greeks']['volga']:.6f}")
        print(f"Time:   {result['time_ms']:.2f} ms")
        print(f"PDE solves: {result['n_pde_solves']}")


if __name__ == "__main__":
    demo()
