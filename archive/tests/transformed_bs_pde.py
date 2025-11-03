"""
Method 1: Variable Transformation PDE

Transform BS PDE to remove sigma from diffusion coefficient:
    x = ln(S/K)
    tau = sigma^2 * (T-t) / 2

Result: Diffusion coefficient = 1 (constant!)

Original PDE:
    dV/dt + (sigma^2 * S^2 / 2) * d²V/dS² + r*S*dV/dS - r*V = 0

Transformed PDE:
    dV/dtau = d²V/dx² + b(sigma)*dV/dx + c(sigma)*V

where:
    b(sigma) = 2r/sigma^2 - 1
    c(sigma) = -2r/sigma^2
"""
import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape


class TransformedBSPDE:
    """
    Black-Scholes PDE in transformed coordinates (x, tau)

    Advantages:
    - Diffusion coefficient = 1 (constant, no numerical damping)
    - Sigma only appears in drift and reaction terms (linear)
    - Numerically stable for all sigma values
    """

    def __init__(self, K: float, T: float, r: float, M: int, N: int,
                 x_min: float = -5.0, x_max: float = 5.0):
        """
        Args:
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            M: Number of spatial grid points
            N: Number of time steps
            x_min, x_max: Range of x = ln(S/K)
        """
        self.K = K
        self.T = T
        self.r = r
        self.M = M
        self.N = N

        # Spatial grid (x = ln(S/K))
        self.x_grid = np.linspace(x_min, x_max, M)
        self.dx = self.x_grid[1] - self.x_grid[0]

        # Crank-Nicolson parameter
        self.phi = 0.5

    def _terminal_condition(self) -> np.ndarray:
        """
        Terminal condition: V(x, tau=0) = V(S, t=T)

        For call option: max(S - K, 0) = max(K*exp(x) - K, 0) = K*max(exp(x) - 1, 0)
        """
        S = self.K * np.exp(self.x_grid)
        return np.maximum(S - self.K, 0.0)

    def _boundary_condition_left(self, tau: float) -> float:
        """
        Boundary at x_min (S -> 0): V -> 0
        """
        return 0.0

    def _boundary_condition_right(self, tau: float) -> float:
        """
        Boundary at x_max (S -> infinity): V -> S - K*exp(-r*(T-t))

        In transformed coordinates:
        S = K*exp(x_max)
        t = T - 2*tau/sigma^2, but we need tau parameter

        For simplicity, use S - K*exp(-r*tau_to_time)
        """
        S = self.K * np.exp(self.x_grid[-1])
        # Approximate: for large S, option is deep ITM
        return S - self.K

    def compute_coefficients(self, sigma_var: ADVar, dtau: ADVar) -> Tuple:
        """
        Compute PDE coefficients

        Transformed PDE:
            dV/dtau = d²V/dx² + b*dV/dx + c*V

        where:
            b = 2r/sigma² - 1
            c = -2r/sigma²

        Returns:
            alpha, beta, gamma for tridiagonal system
        """
        # Diffusion coefficient (constant!)
        alpha = ADVar(1.0, requires_grad=False)

        # Drift coefficient: b = 2r/sigma² - 1
        sigma_sq = sigma_var * sigma_var
        b = ADVar(2.0 * self.r, requires_grad=False) / sigma_sq - ADVar(1.0, requires_grad=False)

        # Reaction coefficient: c = -2r/sigma²
        c = -ADVar(2.0 * self.r, requires_grad=False) / sigma_sq

        # Finite difference coefficients
        dx_sq = ADVar(self.dx ** 2, requires_grad=False)
        dx_2 = ADVar(2.0 * self.dx, requires_grad=False)

        # Tridiagonal matrix entries for interior points
        # Using central differences
        coef_alpha = alpha / dx_sq
        coef_beta = b / dx_2
        coef_gamma = c

        return coef_alpha, coef_beta, coef_gamma, dtau

    def build_tridiagonal_system(self, coef_alpha: ADVar, coef_beta: ADVar,
                                 coef_gamma: ADVar, dtau: ADVar) -> Tuple:
        """
        Build tridiagonal system for CN scheme

        L_B * V^n = R_B * V^(n+1)

        where:
            L_B = I - phi*dtau*D
            R_B = I + (1-phi)*dtau*D

            D*V_j = alpha*(V_{j+1} - 2V_j + V_{j-1})
                  + beta*(V_{j+1} - V_{j-1})
                  + gamma*V_j
        """
        n = self.M - 2  # Interior points

        # Tridiagonal entries for D operator
        l_j = coef_alpha - coef_beta  # Lower diagonal
        c_j = -ADVar(2.0, requires_grad=False) * coef_alpha + coef_gamma  # Diagonal
        u_j = coef_alpha + coef_beta  # Upper diagonal

        phi = self.phi

        # L_B = I - phi*dtau*D
        a_L = [-ADVar(phi, requires_grad=False) * dtau * l_j] * n
        a_L[0] = ADVar(0.0, requires_grad=False)  # Boundary

        b_L = [ADVar(1.0, requires_grad=False) - ADVar(phi, requires_grad=False) * dtau * c_j] * n

        c_L = [-ADVar(phi, requires_grad=False) * dtau * u_j] * n
        c_L[-1] = ADVar(0.0, requires_grad=False)  # Boundary

        # R_B = I + (1-phi)*dtau*D
        a_R = [ADVar(1.0 - phi, requires_grad=False) * dtau * l_j] * n
        a_R[0] = ADVar(0.0, requires_grad=False)

        b_R = [ADVar(1.0, requires_grad=False) + ADVar(1.0 - phi, requires_grad=False) * dtau * c_j] * n

        c_R = [ADVar(1.0 - phi, requires_grad=False) * dtau * u_j] * n
        c_R[-1] = ADVar(0.0, requires_grad=False)

        return a_L, b_L, c_L, a_R, b_R, c_R

    def tridiag_solve(self, a: List[ADVar], b: List[ADVar], c: List[ADVar],
                     d: List[ADVar]) -> List[ADVar]:
        """
        Solve tridiagonal system: A*x = d

        Using Thomas algorithm (forward elimination + back substitution)
        """
        n = len(d)

        # Forward elimination
        c_prime = [None] * n
        d_prime = [None] * n

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denom if i < n-1 else ADVar(0.0)
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        # Back substitution
        x = [None] * n
        x[-1] = d_prime[-1]

        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def cn_step(self, V: List[ADVar], a_L: List[ADVar], b_L: List[ADVar],
                c_L: List[ADVar], a_R: List[ADVar], b_R: List[ADVar],
                c_R: List[ADVar], tau_current: float) -> List[ADVar]:
        """
        One Crank-Nicolson time step

        L_B * V^n = R_B * V^(n+1) + boundary terms
        """
        n = self.M - 2

        # Right-hand side: R_B * V^(n+1)
        rhs = [None] * n

        for i in range(n):
            j = i + 1  # Grid index

            if i == 0:
                # Left boundary
                V_left = ADVar(self._boundary_condition_left(tau_current), requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i+1] - a_R[i] * V_left
            elif i == n-1:
                # Right boundary
                V_right = ADVar(self._boundary_condition_right(tau_current), requires_grad=False)
                rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                # Interior
                rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] + c_R[i] * V[i+1]

        # Solve L_B * V_new = rhs
        V_new = self.tridiag_solve(a_L, b_L, c_L, rhs)

        return V_new

    def solve(self, sigma: float, verbose: bool = False) -> Tuple[float, float]:
        """
        Solve transformed BS PDE with AAD

        Returns:
            price: Option price at S=K (x=0)
            vega: dV/dsigma
        """
        # Reset tape
        global_tape.reset()

        # Sigma as ADVar
        sigma_var = ADVar(sigma, requires_grad=True, name="sigma")

        # Tau grid (tau = sigma^2 * (T-t) / 2)
        # tau goes from 0 (at t=T) to tau_max (at t=0)
        tau_max_val = sigma ** 2 * self.T / 2.0
        tau_max = sigma_var * sigma_var * ADVar(self.T, requires_grad=False) / ADVar(2.0, requires_grad=False)
        dtau = tau_max / ADVar(self.N, requires_grad=False)

        if verbose:
            print(f"\nTransformed PDE solve:")
            print(f"  sigma = {sigma}")
            print(f"  tau_max = {tau_max_val:.6f}")
            print(f"  dtau = {tau_max_val/self.N:.6f}")
            print(f"  dx = {self.dx:.6f}")

        # Compute coefficients
        coef_alpha, coef_beta, coef_gamma, dtau = self.compute_coefficients(sigma_var, dtau)

        # Build tridiagonal system
        a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_system(
            coef_alpha, coef_beta, coef_gamma, dtau
        )

        # Initial condition (at tau=0, i.e., t=T)
        V_terminal = self._terminal_condition()
        V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]  # Interior points only

        # Time stepping (forward in tau, backward in t)
        for n in range(self.N):
            tau_current = (n + 1) * (tau_max_val / self.N)
            V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, tau_current)

        # Interpolate to x=0 (S=K)
        # Find index closest to x=0
        idx_center = np.argmin(np.abs(self.x_grid))

        if self.x_grid[idx_center] == 0.0:
            price_var = V[idx_center - 1]  # Interior point index
        else:
            # Linear interpolation
            if self.x_grid[idx_center] < 0:
                i1, i2 = idx_center, idx_center + 1
            else:
                i1, i2 = idx_center - 1, idx_center

            x1, x2 = self.x_grid[i1], self.x_grid[i2]
            weight = (0.0 - x1) / (x2 - x1)

            price_var = V[i1-1] * ADVar(1.0 - weight, requires_grad=False) + V[i2-1] * ADVar(weight, requires_grad=False)

        price = price_var.val

        # AAD backward pass
        price_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)

        vega = sigma_var.adj

        return price, vega


def test_transformed_pde():
    """Test transformed PDE method"""
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
    print("METHOD 1: VARIABLE TRANSFORMATION PDE TEST")
    print("="*100)

    print("\nTransformed coordinates:")
    print("  x = ln(S/K)")
    print("  tau = sigma^2 * (T-t) / 2")
    print("\nKey advantage: Diffusion coefficient = 1 (constant!)")

    # Test at different sigma values
    sigma_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

    # Test different grid sizes
    grid_configs = [(101, 100), (151, 150)]

    for M, N in grid_configs:
        print("\n" + "="*100)
        print(f"Grid: M={M}, N={N}")
        print("="*100)

        solver = TransformedBSPDE(K=K, T=T, r=r, M=M, N=N)

        print(f"\n{'Sigma':<10} | {'BS Price':<12} | {'PDE Price':<12} | {'Price Err':<10} | "
              f"{'BS Vega':<12} | {'PDE Vega':<12} | {'Vega Err':<10} | {'Time(s)':<10}")
        print("-"*120)

        results = []
        for sigma in sigma_values:
            t_start = time.perf_counter()

            # Analytical
            bs_price, bs_vega, bs_volga = black_scholes_greeks(S0, K, T, r, sigma)

            # PDE (note: solver assumes S0=K, so we're pricing ATM)
            pde_price, pde_vega = solver.solve(sigma)

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
    test_transformed_pde()
