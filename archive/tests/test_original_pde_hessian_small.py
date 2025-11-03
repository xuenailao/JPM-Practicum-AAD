"""
测试原始PDE + Edge-Pushing Hessian (小网格)

先用小网格测试Edge-Pushing是否可行
"""
import numpy as np
import sys
from pathlib import Path
from scipy.stats import norm
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape

# Try to import Edge-Pushing (may have bug)
try:
    from aad_edge_pushing.edge_pushing.algo4_adjlist import algo4_adjlist
    HESSIAN_AVAILABLE = True
except:
    HESSIAN_AVAILABLE = False


def black_scholes_greeks(S0, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma

    return {'price': price, 'vega': vega, 'volga': volga}


class SmallOriginalPDE:
    """非常小的原始PDE求解器用于测试Hessian"""

    def __init__(self, S0, K, T, r, M=21, N=20):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.M = M
        self.N = N

        # Grid
        S_min, S_max = 0.0, 3.0 * K
        self.S_grid = np.linspace(S_min, S_max, M)
        self.dS = self.S_grid[1] - self.S_grid[0]

    def solve_return_advar(self, sigma, verbose=False):
        """求解PDE，返回price_var (ADVar)"""

        global_tape.reset()

        sigma_var = ADVar(sigma, requires_grad=True, name="sigma")

        # 时间步 (简化：不用自适应)
        dt_val = self.T / self.N
        dt = ADVar(dt_val)

        if verbose:
            print(f"\n小网格PDE:")
            print(f"  M={self.M}, N={self.N}")
            print(f"  dt={dt_val:.6f}")

        # 初始条件
        V_terminal = np.maximum(self.S_grid - self.K, 0.0)
        V = [ADVar(v) for v in V_terminal[1:-1]]

        # 简化的显式欧拉法 (仅用于小网格测试)
        for n in range(self.N):
            V_new = []
            for i in range(len(V)):
                S_i = self.S_grid[i+1]
                S_i_var = ADVar(S_i)

                # 简化的扩散项
                if i == 0 or i == len(V)-1:
                    V_new.append(V[i])  # 边界
                else:
                    # ∂²V/∂S² ≈ (V_{i+1} - 2V_i + V_{i-1}) / dS²
                    diff2 = (V[i+1] - ADVar(2.0)*V[i] + V[i-1]) / ADVar(self.dS**2)

                    # σ²S²/2 * ∂²V/∂S²
                    diffusion = sigma_var * sigma_var * S_i_var * S_i_var / ADVar(2.0) * diff2

                    # 简化：只保留扩散项 (测试用)
                    V_i_new = V[i] + dt * diffusion

                    V_new.append(V_i_new)

            V = V_new

        # 插值到S0
        idx = np.argmin(np.abs(self.S_grid - self.S0))
        price_var = V[idx-1]

        if verbose:
            print(f"  Tape节点数: {len(global_tape.nodes)}")
            print(f"  Price: {price_var.val:.6f}")

        return price_var, sigma_var


print("="*100)
print("原始PDE + Edge-Pushing Hessian 可行性测试")
print("="*100)

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

# 解析解
bs = black_scholes_greeks(S0, K, T, r, sigma)
print(f"\n解析解:")
print(f"  Vega:  {bs['vega']:.6f}")
print(f"  Volga: {bs['volga']:.6f}")

# 测试不同网格大小
grid_sizes = [(11, 10), (21, 20), (31, 30)]

print(f"\n{'Grid (M,N)':<15} | {'Tape Nodes':<12} | {'Vega':<12} | {'Vega Err':<10} | "
      f"{'Hessian Time':<15} | {'Volga':<12} | {'Volga Err':<12}")
print("-"*120)

for M, N in grid_sizes:
    solver = SmallOriginalPDE(S0, K, T, r, M=M, N=N)

    # 求解
    price_var, sigma_var = solver.solve_return_advar(sigma, verbose=False)

    # Vega (backprop)
    price_var.adj = 1.0
    for node in reversed(global_tape.nodes):
        for parent, deriv in node.parents:
            if parent.requires_grad:
                parent.adj += node.out.adj * float(deriv)

    vega = sigma_var.adj
    vega_err = abs(vega - bs['vega']) / bs['vega'] * 100

    # Hessian (Edge-Pushing)
    global_tape.reset()
    price_var, sigma_var = solver.solve_return_advar(sigma, verbose=False)

    tape_size = len(global_tape.nodes)

    if HESSIAN_AVAILABLE:
        try:
            t_start = time.perf_counter()
            hessian = algo4_adjlist(price_var, [sigma_var])
            t_hessian = time.perf_counter() - t_start

            volga = hessian[0, 0]
            volga_err = abs(volga - bs['volga']) / abs(bs['volga']) * 100

            print(f"{f'({M},{N})':<15} | {tape_size:<12} | {vega:<12.6f} | {vega_err:<10.2f}% | "
                  f"{t_hessian:<15.3f}s | {volga:<12.6f} | {volga_err:<12.2f}%")

        except Exception as e:
            print(f"{f'({M},{N})':<15} | {tape_size:<12} | {vega:<12.6f} | {vega_err:<10.2f}% | "
                  f"{'ERROR':<15} | {'-':<12} | {str(e)[:30]:<12}")
    else:
        print(f"{f'({M},{N})':<15} | {tape_size:<12} | {vega:<12.6f} | {vega_err:<10.2f}% | "
              f"{'N/A':<15} | {'-':<12} | {'algo4 unavail':<12}")

print("\n" + "="*100)
print("结论")
print("="*100)

if HESSIAN_AVAILABLE:
    print("\n✅ 如果Edge-Pushing成功:")
    print("  - 原始PDE + Hessian是可行方案")
    print("  - Volga应该准确 (σ直接依赖)")
    print("  - 需要优化计算图规模或使用更高效的Hessian算法")

    print("\n❌ 如果Edge-Pushing失败/太慢:")
    print("  - 考虑Adjoint PDE方法")
    print("  - 或接受Volga定性精度")
else:
    print("\n⚠️ Edge-Pushing算法不可用")
    print("  需要修复algo4_adjlist的bug")

print("\n🎯 关键洞察:")
print("  原始PDE避免了变换PDE的隐式σ依赖问题")
print("  σ直接出现在扩散系数中，理论上∂²V/∂σ²应该准确")
