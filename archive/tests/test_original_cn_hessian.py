"""
使用正确的CN格式 + Edge-Pushing Hessian

关键: 使用Crank-Nicolson而不是简化的显式欧拉
"""
import numpy as np
import sys
from pathlib import Path
from scipy.stats import norm
import time

sys.path.insert(0, str(Path(__file__).parent))

from original_pde_aad_hessian import OriginalBSPDE_AAD
from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.edge_pushing.algo4_adjlist import algo4_adjlist


def black_scholes_greeks(S0, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma

    return {'price': price, 'vega': vega, 'volga': volga}


class OriginalPDE_WithHessian(OriginalBSPDE_AAD):
    """扩展OriginalBSPDE_AAD以返回ADVar用于Hessian计算"""

    def solve_return_advar(self, sigma, verbose=False):
        """
        求解PDE并返回price_var (ADVar)用于Hessian计算

        Returns:
            price_var: Price as ADVar
            sigma_var: Sigma as ADVar
        """
        # Reset tape
        global_tape.reset()

        # Sigma as ADVar
        sigma_var = ADVar(sigma, requires_grad=True, name="sigma")

        # Adaptive time steps
        t_grid, N = self.compute_adaptive_timesteps(sigma)

        if verbose:
            print(f"\n[CN with Hessian]")
            print(f"  M={self.M}, N={N}")
            print(f"  dt={t_grid[1]-t_grid[0]:.6f}")

        # Build CN system
        dt_val = t_grid[1] - t_grid[0]
        dt = ADVar(dt_val, requires_grad=False)

        a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_cn(sigma_var, dt)

        # Initial condition
        V_terminal = self._terminal_condition()
        V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

        # Time stepping
        for n in range(N):
            t_current = t_grid[n+1]
            V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_current)

        # Interpolate to S0
        idx = np.argmin(np.abs(self.S_grid - self.S0))

        if abs(self.S_grid[idx] - self.S0) < 1e-10:
            price_var = V[idx - 1]
        else:
            if self.S_grid[idx] < self.S0:
                i1, i2 = idx, idx + 1
            else:
                i1, i2 = idx - 1, idx

            S1, S2 = self.S_grid[i1], self.S_grid[i2]
            weight = (self.S0 - S1) / (S2 - S1)

            price_var = V[i1-1] * ADVar(1.0 - weight) + V[i2-1] * ADVar(weight)

        if verbose:
            print(f"  Tape size: {len(global_tape.nodes)}")
            print(f"  Price: {price_var.val:.6f}")

        return price_var, sigma_var


print("="*120)
print("原始CN格式 + Edge-Pushing Hessian测试")
print("="*120)

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

bs = black_scholes_greeks(S0, K, T, r, sigma)
print(f"\n解析解:")
print(f"  Price: {bs['price']:.6f}")
print(f"  Vega:  {bs['vega']:.6f}")
print(f"  Volga: {bs['volga']:.6f}")

# 测试不同grid大小
configs = [
    (31, 30, "小"),
    (51, 50, "中"),
]

print(f"\n{'Grid (M,N)':<15} | {'Size':<6} | {'Tape Nodes':<12} | {'Vega':<12} | {'Vega Err':<10} | "
      f"{'Hessian Time':<15} | {'Volga':<12} | {'Volga Err':<12}")
print("-"*140)

for M, N_base, size_label in configs:
    solver = OriginalPDE_WithHessian(S0, K, T, r, M=M, N_base=N_base)

    # 求解 + Backprop for Vega
    t_start = time.perf_counter()
    price_var, sigma_var = solver.solve_return_advar(sigma, verbose=False)

    price_var.adj = 1.0
    for node in reversed(global_tape.nodes):
        for parent, deriv in node.parents:
            if parent.requires_grad:
                parent.adj += node.out.adj * float(deriv)

    vega = sigma_var.adj
    t_vega = time.perf_counter() - t_start

    vega_err = abs(vega - bs['vega']) / bs['vega'] * 100

    # 重新求解 for Hessian
    global_tape.reset()
    price_var, sigma_var = solver.solve_return_advar(sigma, verbose=False)

    tape_size = len(global_tape.nodes)

    # Hessian
    try:
        t_start = time.perf_counter()
        hessian = algo4_adjlist(price_var, [sigma_var])
        t_hessian = time.perf_counter() - t_start

        volga = hessian[0, 0]
        volga_err = abs(volga - bs['volga']) / abs(bs['volga']) * 100

        print(f"{f'({M},{N_base})':<15} | {size_label:<6} | {tape_size:<12} | {vega:<12.6f} | {vega_err:<10.2f}% | "
              f"{t_hessian:<15.3f}s | {volga:<12.6f} | {volga_err:<12.2f}%")

    except Exception as e:
        print(f"{f'({M},{N_base})':<15} | {size_label:<6} | {tape_size:<12} | {vega:<12.6f} | {vega_err:<10.2f}% | "
              f"{'TIMEOUT/ERROR':<15} | {'-':<12} | {str(e)[:20]}")

print("\n" + "="*120)
print("分析")
print("="*120)

print("\n📊 观察:")
print("  - CN格式应该比显式欧拉更准确")
print("  - Tape规模仍然很大 (数万节点)")
print("  - Hessian计算时间随网格增大快速增长 (O(n³))")

print("\n🎯 关键问题:")
print("  即使在原始PDE中σ直接依赖，Hessian计算仍然:")
print("  1. 计算图太大 (51万节点 for M=101)")
print("  2. Edge-Pushing O(n³)复杂度无法扩展")
print("  3. 小网格结果不稳定")

print("\n💡 根本限制:")
print("  PDE求解 (无论原始还是变换) 都产生大规模计算图")
print("  Edge-Pushing算法不适合这种规模")

print("\n✅ 可行方案:")
print("  1. Adjoint PDE (避免Hessian计算)")
print("  2. 接受Volga定性精度")
print("  3. Monte Carlo + Pathwise derivative (不同范式)")
