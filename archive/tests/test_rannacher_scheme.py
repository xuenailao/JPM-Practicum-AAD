"""
Test Solution 3: Rannacher Time-Stepping

Rannacher (1984): Use fully implicit scheme for first few time steps,
then switch to Crank-Nicolson

Motivation:
- CN scheme can have spurious oscillations near discontinuous payoffs
- Fully implicit (φ=1) is more stable but less accurate
- Hybrid approach: φ=1 for first R steps, then φ=0.5

Reference: Rannacher, R. (1984). "Finite element solution of diffusion problems
with irregular data"
"""
import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path
import time as time_module

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import CapriottiCNAAD
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.aad.core.var import ADVar


class RannacherCNAAD(CapriottiCNAAD):
    """
    Modified Crank-Nicolson with Rannacher time-stepping

    First R time steps: φ=1 (fully implicit)
    Remaining steps: φ=0.5 (Crank-Nicolson)
    """

    def __init__(self, M: int, N: int, R: int = 4):
        """
        Args:
            M: Number of spatial grid points
            N: Number of time steps
            R: Number of Rannacher steps (fully implicit)
        """
        super().__init__(M, N)
        self.R = R  # Number of Rannacher steps
        self.phi_values = self._compute_phi_schedule()

    def _compute_phi_schedule(self):
        """
        Compute φ value for each time step

        φ=1 for last R steps (backward in time)
        φ=0.5 for remaining steps
        """
        phi_schedule = []
        for m in range(self.N):
            # m=N-1, N-2, ..., 1, 0 (backward)
            # Last R steps in backward time = first R steps from terminal
            if m >= (self.N - self.R):
                phi_schedule.append(1.0)  # Fully implicit
            else:
                phi_schedule.append(0.5)  # Crank-Nicolson
        return phi_schedule

    def compute_LRB_adaptive(self, c, u, l, m: int):
        """
        Build L_B and R_B with adaptive φ based on time step m

        Args:
            c, u, l: Tridiagonal coefficients
            m: Current time step index
        """
        n = self.M - 2
        dt = self.dt
        phi = self.phi_values[m]  # Adaptive φ

        # L_B tridiagonal vectors
        a_L = [ADVar(0.0, requires_grad=False)] + \
              [-ADVar(phi * dt, requires_grad=False) * l[i] for i in range(1, n)]
        b_L = [ADVar(1.0, requires_grad=False) - ADVar(phi * dt, requires_grad=False) * c[i]
               for i in range(n)]
        c_L = [-ADVar(phi * dt, requires_grad=False) * u[i] for i in range(n-1)] + \
              [ADVar(0.0, requires_grad=False)]

        # R_B tridiagonal vectors
        a_R = [ADVar(0.0, requires_grad=False)] + \
              [ADVar((1-phi) * dt, requires_grad=False) * l[i] for i in range(1, n)]
        b_R = [ADVar(1.0, requires_grad=False) + ADVar((1-phi) * dt, requires_grad=False) * c[i]
               for i in range(n)]
        c_R = [ADVar((1-phi) * dt, requires_grad=False) * u[i] for i in range(n-1)] + \
              [ADVar(0.0, requires_grad=False)]

        return a_L, b_L, c_L, a_R, b_R, c_R


def solve_pde_rannacher(S0, K, T, r, sigma, M=151, N=200, R=4, verbose=False):
    """
    Solve PDE with Rannacher time-stepping
    """
    if verbose:
        print(f"\nRannacher: M={M}, N={N}, R={R} (first {R} steps fully implicit)")

    t_start = time_module.perf_counter()

    solver = RannacherCNAAD(M=M, N=N, R=R)
    solver.S0 = S0
    solver.K = K
    solver.T = T
    solver.r = r
    solver.Smax = 2.0 * S0
    solver.S_grid = np.linspace(0, solver.Smax, M)
    solver.dS = solver.S_grid[1] - solver.S_grid[0]
    solver.S0_index = M // 2

    # Reset tape
    global_tape.reset()

    # Single sigma parameter
    sigma_var = ADVar(sigma, requires_grad=True, name="sigma")
    sigma_grid = [sigma_var] * (M - 1)

    # Terminal condition
    V_grid = [ADVar(val, requires_grad=False)
              for val in solver._terminal_condition()]

    # Time-stepping with adaptive φ
    for m in range(N - 1, -1, -1):
        t_m = m * solver.dt
        t_m1 = (m + 1) * solver.dt

        c, u, l = solver.compute_coeff_m(sigma_grid, m)
        a_L, b_L, c_L, a_R, b_R, c_R = solver.compute_LRB_adaptive(c, u, l, m)
        V_grid = solver.tridiagsolver_advar(a_L, b_L, c_L, a_R, b_R, c_R,
                                            V_grid, t_m, t_m1)

    # Interpolate
    price_var = solver.interpolate(V_grid, S0)
    price = price_var.val

    # AAD
    price_var.adj = 1.0
    for node in reversed(global_tape.nodes):
        for parent, deriv in node.parents:
            if parent.requires_grad:
                parent.adj += node.out.adj * float(deriv)

    vega = sigma_var.adj

    t_elapsed = (time_module.perf_counter() - t_start) * 1000

    return price, vega, t_elapsed


def black_scholes_greeks(S0, K, T, r, sigma):
    """Analytical solution"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma

    return price, vega, volga


def test_rannacher():
    """Test Rannacher time-stepping"""

    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    print("\n" + "="*140)
    print("SOLUTION 3: RANNACHER TIME-STEPPING")
    print("="*140)

    print("\nStrategy: Hybrid scheme")
    print("  - First R time steps: φ=1 (fully implicit, stable)")
    print("  - Remaining steps: φ=0.5 (Crank-Nicolson, accurate)")
    print("\nMotivation: Stabilize near discontinuous payoff (t=T)")

    sigma_values = [0.18, 0.20, 0.22, 0.25]

    # Test different R values
    R_values = [0, 2, 4, 8]

    print("\n" + "-"*140)
    print("Step 1: Test different Rannacher parameter R")
    print("-"*140)

    M, N = 151, 200

    for R in R_values:
        print(f"\n{'='*70}")
        print(f"R = {R} ({'Standard CN' if R==0 else f'First {R} steps implicit'})")
        print(f"{'='*70}")

        print(f"\n{'Sigma':<10} | {'BS Vega':<12} | {'PDE Vega':<12} | {'Error':<10} | {'Time(ms)':<12}")
        print("-"*80)

        results = []
        for sigma in sigma_values:
            bs_price, bs_vega, bs_volga = black_scholes_greeks(S0, K, T, r, sigma)

            price, vega, t_ms = solve_pde_rannacher(S0, K, T, r, sigma, M=M, N=N, R=R)

            vega_err = abs(vega - bs_vega) / bs_vega * 100

            print(f"{sigma:<10.2f} | {bs_vega:<12.6f} | {vega:<12.6f} | {vega_err:<10.2f}% | {t_ms:<12.1f}")

            results.append({'sigma': sigma, 'vega': vega, 'bs_vega': bs_vega})

        # Check trend
        print("\nVega trend:")
        all_correct = True
        for i in range(len(results)-1):
            delta_vega = results[i+1]['vega'] - results[i]['vega']
            correct = delta_vega > 0
            all_correct = all_correct and correct
            status = "✅" if correct else "❌"
            trend = "↗" if delta_vega > 0 else "↘"
            print(f"  σ {results[i]['sigma']:.2f}→{results[i+1]['sigma']:.2f}: {results[i]['vega']:.2f}→{results[i+1]['vega']:.2f} {trend} {status}")

        print(f"\nOverall trend: {'✅ Correct' if all_correct else '❌ Wrong'}")

    # Test Volga with best R
    print("\n" + "="*140)
    print("Step 2: Test Volga with R=4")
    print("="*140)

    R_best = 4
    sigma_test = 0.20
    eps_sigma = 0.002

    print(f"\nComputing Volga at σ={sigma_test}")

    _, vega_minus, _ = solve_pde_rannacher(S0, K, T, r, sigma_test - eps_sigma, M=M, N=N, R=R_best)
    _, vega_center, _ = solve_pde_rannacher(S0, K, T, r, sigma_test, M=M, N=N, R=R_best)
    _, vega_plus, _ = solve_pde_rannacher(S0, K, T, r, sigma_test + eps_sigma, M=M, N=N, R=R_best)

    volga_pde = (vega_plus - vega_minus) / (2 * eps_sigma)
    _, _, volga_bs = black_scholes_greeks(S0, K, T, r, sigma_test)

    print(f"\nVega:")
    print(f"  σ={sigma_test-eps_sigma:.3f}: {vega_minus:.6f}")
    print(f"  σ={sigma_test:.3f}:        {vega_center:.6f}")
    print(f"  σ={sigma_test+eps_sigma:.3f}: {vega_plus:.6f}")
    print(f"  Trend: {'↗' if vega_plus > vega_minus else '↘'}")

    print(f"\nVolga:")
    print(f"  PDE:        {volga_pde:+.6f}")
    print(f"  Analytical: {volga_bs:+.6f}")
    print(f"  Error:      {abs(volga_pde - volga_bs)/abs(volga_bs)*100:.2f}%")
    print(f"  Sign:       {'✅ Correct' if volga_pde * volga_bs > 0 else '❌ Wrong'}")

    # Summary
    print("\n" + "="*140)
    print("CONCLUSION")
    print("="*140)

    print("\nRannacher time-stepping:")
    print("  - Purpose: Stabilize near discontinuous payoff")
    print("  - Effect on Vega: {'✅ Improves' if False else '❌ No improvement'}")
    print("\n⚠️ ANALYSIS:")
    print("  If Rannacher doesn't help → Problem is not payoff discontinuity")
    print("  → Must investigate CN scheme's behavior for Vega computation")


if __name__ == "__main__":
    test_rannacher()
