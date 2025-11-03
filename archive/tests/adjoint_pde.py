"""
Method 2: Adjoint PDE for Vega

Instead of finite difference: Vega ≈ [V(σ+ε) - V(σ-ε)]/(2ε)
Solve the adjoint PDE directly:

Forward PDE (standard BS):
    ∂V/∂t + L_BS[V] = 0
    V(S,T) = Payoff(S)

Adjoint PDE (for Vega):
    ∂Vega/∂t + L_BS[Vega] = Source(S,t,σ,Γ)
    Vega(S,T) = 0

where:
    L_BS[·] = (σ²S²/2)∂²/∂S² + rS∂/∂S - r(·)
    Source = -σS²Γ
    Γ = ∂²V/∂S² (Gamma from forward solve)
"""
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import CapriottiCNAAD
from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape


class AdjointPDESolver:
    """
    Adjoint PDE solver for computing Vega directly

    Two-step process:
    1. Forward solve: Get V and Gamma
    2. Adjoint solve: Use Gamma as source term to get Vega
    """

    def __init__(self, M: int, N: int):
        """
        Args:
            M: Number of spatial grid points
            N: Number of time steps
        """
        self.M = M
        self.N = N

    def compute_gamma_grid(self, V_grid: List[ADVar], dS: float) -> List[ADVar]:
        """
        Compute Gamma = ∂²V/∂S² using centered finite difference

        Args:
            V_grid: Value grid (interior points)
            dS: Spatial step size

        Returns:
            Gamma grid
        """
        M = len(V_grid) + 2  # Including boundaries
        Gamma = []

        dS_sq = ADVar(dS ** 2, requires_grad=False)

        for i in range(1, M-1):
            if i == 1:
                # Use boundary V[0] = 0
                V_left = ADVar(0.0, requires_grad=False)
                V_center = V_grid[0]
                V_right = V_grid[1]
            elif i == M-2:
                # Use boundary V[M-1] (extrapolate or use boundary condition)
                V_left = V_grid[-2]
                V_center = V_grid[-1]
                V_right = ADVar(0.0, requires_grad=False)  # Approximate
            else:
                V_left = V_grid[i-2]
                V_center = V_grid[i-1]
                V_right = V_grid[i]

            Gamma_i = (V_right - ADVar(2.0, requires_grad=False) * V_center + V_left) / dS_sq
            Gamma.append(Gamma_i)

        return Gamma

    def solve_forward(self, S0: float, K: float, T: float, r: float,
                     sigma_var: ADVar) -> Tuple[List[ADVar], List[List[ADVar]], CapriottiCNAAD]:
        """
        Forward solve: Standard BS PDE

        Returns:
            V_grid: Final value grid (at t=0)
            Gamma_history: Gamma at each time step
            solver: PDE solver object
        """
        solver = CapriottiCNAAD(M=self.M, N=self.N)
        solver.S0 = S0
        solver.K = K
        solver.T = T
        solver.r = r
        solver.Smax = 2.0 * S0
        solver.S_grid = np.linspace(0, solver.Smax, self.M)
        solver.dS = solver.S_grid[1] - solver.S_grid[0]
        solver.S0_index = self.M // 2

        # Sigma grid (all same for constant volatility)
        sigma_grid = [sigma_var] * (self.M - 1)

        # Terminal condition
        V_grid = [ADVar(val, requires_grad=False)
                  for val in solver._terminal_condition()]

        # Store Gamma at each time step
        Gamma_history = []

        # Time-stepping
        for m in range(self.N - 1, -1, -1):
            t_m = m * solver.dt
            t_m1 = (m + 1) * solver.dt

            c, u, l = solver.compute_coeff_m(sigma_grid, m)
            a_L, b_L, c_L, a_R, b_R, c_R = solver.compute_LRB(c, u, l)
            V_grid = solver.tridiagsolver_advar(a_L, b_L, c_L, a_R, b_R, c_R,
                                                V_grid, t_m, t_m1)

            # Compute and store Gamma
            Gamma = self.compute_gamma_grid(V_grid, solver.dS)
            Gamma_history.append(Gamma)

        return V_grid, Gamma_history, solver

    def thomas_algorithm(self, a: List[ADVar], b: List[ADVar], c: List[ADVar],
                        d: List[ADVar]) -> List[ADVar]:
        """
        Thomas algorithm for tridiagonal system

        Args:
            a: Lower diagonal
            b: Main diagonal
            c: Upper diagonal
            d: RHS

        Returns:
            x: Solution
        """
        n = len(d)

        # Forward elimination
        c_prime = [None] * n
        d_prime = [None] * n

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            if i < n-1:
                c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        # Back substitution
        x = [None] * n
        x[-1] = d_prime[-1]

        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def solve_adjoint_with_source(self, solver: CapriottiCNAAD,
                                  sigma_var: ADVar,
                                  Gamma_history: List[List[ADVar]]) -> List[ADVar]:
        """
        Adjoint solve: PDE with source term

        ∂Vega/∂t + L_BS[Vega] = Source
        Source = -σS²Γ

        Args:
            solver: Forward solver (for grid info)
            sigma_var: Volatility as ADVar
            Gamma_history: Gamma at each time step

        Returns:
            Vega_grid: Vega at t=0
        """
        # Terminal condition: Vega(S,T) = 0
        Vega_grid = [ADVar(0.0, requires_grad=False) for _ in range(self.M - 2)]

        # Sigma grid
        sigma_grid = [sigma_var] * (self.M - 1)

        # Time-stepping (backward in time, same as forward)
        for m in range(self.N - 1, -1, -1):
            t_m = m * solver.dt
            t_m1 = (m + 1) * solver.dt

            # Get Gamma at this time step
            Gamma = Gamma_history[self.N - 1 - m]  # Reverse index

            # Source term: -σS²Γ
            Source = []
            for i in range(len(Gamma)):
                S_i = solver.S_grid[i + 1]  # Interior point
                source_i = -sigma_var * ADVar(S_i ** 2, requires_grad=False) * Gamma[i]
                Source.append(source_i)

            # PDE coefficients
            c, u, l = solver.compute_coeff_m(sigma_grid, m)
            a_L, b_L, c_L, a_R, b_R, c_R = solver.compute_LRB(c, u, l)

            # Modified RHS with source term
            # Standard: L_B * V^n = R_B * V^(n+1)
            # With source: L_B * V^n = R_B * V^(n+1) + dt * Source

            # Compute RHS = R_B * Vega + dt * Source
            dt_var = ADVar(solver.dt, requires_grad=False)

            # Build RHS
            rhs = []
            n = len(Vega_grid)

            for i in range(n):
                # R_B matrix-vector product
                if i == 0:
                    rhs_i = a_R[i] * ADVar(0.0, requires_grad=False) + b_R[i] * Vega_grid[i] + c_R[i] * Vega_grid[i+1]
                elif i == n-1:
                    rhs_i = a_R[i] * Vega_grid[i-1] + b_R[i] * Vega_grid[i] + c_R[i] * ADVar(0.0, requires_grad=False)
                else:
                    rhs_i = a_R[i] * Vega_grid[i-1] + b_R[i] * Vega_grid[i] + c_R[i] * Vega_grid[i+1]

                # Add source term
                rhs_i = rhs_i + dt_var * Source[i]
                rhs.append(rhs_i)

            # Solve L_B * Vega_new = rhs using Thomas algorithm
            Vega_grid = self.thomas_algorithm(a_L, b_L, c_L, rhs)

        return Vega_grid

    def compute_vega(self, S0: float, K: float, T: float, r: float, sigma: float,
                     verbose: bool = False) -> Tuple[float, float]:
        """
        Compute option price and Vega using adjoint PDE

        Returns:
            price: Option price at S0
            vega: ∂V/∂σ at S0
        """
        # Reset tape
        global_tape.reset()

        # Sigma as ADVar
        sigma_var = ADVar(sigma, requires_grad=True, name="sigma")

        if verbose:
            print(f"\nAdjoint PDE solve:")
            print(f"  sigma = {sigma}")
            print(f"  Grid: M={self.M}, N={self.N}")

        # Step 1: Forward solve
        V_grid, Gamma_history, solver = self.solve_forward(S0, K, T, r, sigma_var)

        # Interpolate price at S0
        price_var = solver.interpolate(V_grid, S0)
        price = price_var.val

        if verbose:
            print(f"  Forward solve complete: Price = {price:.6f}")
            print(f"  Gamma history stored: {len(Gamma_history)} time steps")

        # Step 2: Adjoint solve
        Vega_grid = self.solve_adjoint_with_source(solver, sigma_var, Gamma_history)

        # Interpolate Vega at S0
        vega_var = solver.interpolate(Vega_grid, S0)
        vega = vega_var.val

        if verbose:
            print(f"  Adjoint solve complete: Vega = {vega:.6f}")

        return price, vega


def test_adjoint_pde():
    """Test adjoint PDE method"""
    from scipy.stats import norm

    def black_scholes_greeks(S0, K, T, r, sigma):
        """Analytical solution"""
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
        d2 = d1 - sigma*sqrt_T

        price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        vega = S0 * norm.pdf(d1) * sqrt_T
        volga = vega * d1 * d2 / sigma

        return price, vega, volga

    # Parameters
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    print("\n" + "="*100)
    print("METHOD 2: ADJOINT PDE TEST")
    print("="*100)

    print("\nTwo-step process:")
    print("  1. Forward PDE: Solve for V and Γ")
    print("  2. Adjoint PDE: Solve ∂Vega/∂t + L_BS[Vega] = -σS²Γ")
    print("\nKey advantage: Direct solve for Vega, no finite difference!")

    # Test at different sigma values
    sigma_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

    # Test different grid sizes
    grid_configs = [(101, 100), (151, 150)]

    for M, N in grid_configs:
        print("\n" + "="*100)
        print(f"Grid: M={M}, N={N}")
        print("="*100)

        solver = AdjointPDESolver(M=M, N=N)

        print(f"\n{'Sigma':<10} | {'BS Price':<12} | {'PDE Price':<12} | {'Price Err':<10} | "
              f"{'BS Vega':<12} | {'PDE Vega':<12} | {'Vega Err':<10} | {'Time(s)':<10}")
        print("-"*120)

        results = []
        for sigma in sigma_values:
            t_start = time.perf_counter()

            # Analytical
            bs_price, bs_vega, bs_volga = black_scholes_greeks(S0, K, T, r, sigma)

            # Adjoint PDE
            try:
                pde_price, pde_vega = solver.compute_vega(S0, K, T, r, sigma)

                t_elapsed = time.perf_counter() - t_start

                price_err = abs(pde_price - bs_price) / bs_price * 100
                vega_err = abs(pde_vega - bs_vega) / bs_vega * 100

                print(f"{sigma:<10.2f} | {bs_price:<12.6f} | {pde_price:<12.6f} | {price_err:<10.2f}% | "
                      f"{bs_vega:<12.6f} | {pde_vega:<12.6f} | {vega_err:<10.2f}% | {t_elapsed:<10.3f}")

                results.append({
                    'sigma': sigma,
                    'bs_vega': bs_vega,
                    'pde_vega': pde_vega,
                    'price_err': price_err,
                    'vega_err': vega_err
                })
            except Exception as e:
                print(f"{sigma:<10.2f} | ERROR: {str(e)}")
                results.append(None)

        # Filter out errors
        results = [r for r in results if r is not None]

        if len(results) > 1:
            # Check Vega trend
            print("\n" + "-"*100)
            print("Vega Trend Analysis:")
            print("-"*100)

            all_correct = True
            for i in range(len(results)-1):
                delta_vega = results[i+1]['pde_vega'] - results[i]['pde_vega']
                correct = delta_vega > 0
                all_correct = all_correct and correct
                trend = "↗" if delta_vega > 0 else "↘"
                status = "✅" if correct else "❌"

                print(f"  σ: {results[i]['sigma']:.2f} → {results[i+1]['sigma']:.2f}  "
                      f"Vega: {results[i]['pde_vega']:.2f} → {results[i+1]['pde_vega']:.2f} {trend} {status}")

            print(f"\nOverall trend: {'✅ CORRECT - Vega increases with sigma!' if all_correct else '❌ WRONG'}")

            # Summary
            avg_price_err = np.mean([r['price_err'] for r in results])
            avg_vega_err = np.mean([r['vega_err'] for r in results])
            max_vega_err = np.max([r['vega_err'] for r in results])

            print(f"\nSummary:")
            print(f"  Average Price Error: {avg_price_err:.2f}%")
            print(f"  Average Vega Error:  {avg_vega_err:.2f}%")
            print(f"  Max Vega Error:      {max_vega_err:.2f}%")


if __name__ == "__main__":
    test_adjoint_pde()
