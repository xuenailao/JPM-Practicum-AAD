"""
PDE AAD with ν=σ² Parametrization for Improved Volga Accuracy

Key improvement: Use ν = σ² as the independent variable instead of σ
This eliminates the quadratic nonlinearity in the diffusion coefficient,
dramatically reducing second-order derivative errors.

Mathematical background:
- Original: α_i = (1/2) σ² S_i² / ΔS²  (σ appears squared)
- New: α_i = (1/2) ν S_i² / ΔS²  (ν is linear)

Chain rule for derivatives:
- Vega = ∂V/∂σ = (∂V/∂ν) · (∂ν/∂σ) = (∂V/∂ν) · 2σ
- Volga = ∂²V/∂σ² = 2(∂V/∂ν) + 4σ² (∂²V/∂ν²)
"""

import numpy as np
import time
from typing import Dict, List, Tuple

# Import AAD framework
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape


class BS_PDE_AAD_Nu:
    """
    Black-Scholes PDE solver with AAD using ν=σ² parametrization

    This version treats variance (ν = σ²) as the independent variable,
    which improves Volga (∂²V/∂σ²) accuracy by removing the squared
    nonlinearity in the diffusion coefficient.
    """

    def __init__(self, S0: float, K: float, T: float, r: float,
                 M: int = 151, N_base: int = 150):
        """
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            M: Number of spatial grid points
            N_base: Base number of time steps
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.M = M
        self.N_base = N_base
        self.phi = 0.5  # Crank-Nicolson parameter

        # Spatial grid
        S_max = 3 * K
        self.S_grid = np.linspace(0, S_max, M)
        self.dS = self.S_grid[1] - self.S_grid[0]

    def _terminal_condition(self):
        """European call payoff at maturity"""
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_condition_left(self, t):
        """Boundary at S=0: V=0"""
        return 0.0

    def _boundary_condition_right(self, t):
        """Boundary at S=S_max: V ≈ S - K·exp(-r(T-t))"""
        return self.S_grid[-1] - self.K * np.exp(-self.r * (self.T - t))

    def build_tridiagonal_cn(self, nu_var: ADVar, dt: ADVar):
        """
        Build Crank-Nicolson tridiagonal system coefficients

        KEY CHANGE: Uses ν (variance) directly instead of σ²

        Args:
            nu_var: Variance ν = σ² as ADVar
            dt: Time step

        Returns:
            Left and right tridiagonal coefficients (a_L, b_L, c_L, a_R, b_R, c_R)
        """
        n = self.M - 2  # Interior points
        dS = self.dS
        dS_sq = dS * dS
        dS_2 = 2.0 * dS

        a_L, b_L, c_L = [], [], []
        a_R, b_R, c_R = [], [], []

        for i in range(n):
            S_i_var = ADVar(self.S_grid[i+1], requires_grad=False)

            # Diffusion coefficient: α = (1/2) ν S² / ΔS²
            # KEY: ν is linear here (not σ²)
            alpha_i = (nu_var * S_i_var * S_i_var / ADVar(2.0)) / dS_sq

            # Drift coefficient
            beta_i = ADVar(self.r) * S_i_var / dS_2

            # Discount coefficient
            gamma_i = -ADVar(self.r)

            l_i = alpha_i - beta_i
            c_i = -ADVar(2.0) * alpha_i + gamma_i
            u_i = alpha_i + beta_i

            phi = self.phi

            # Left side (implicit part)
            if i == 0:
                a_L.append(ADVar(0.0))
            else:
                a_L.append(-ADVar(phi) * dt * l_i)

            b_L.append(ADVar(1.0) - ADVar(phi) * dt * c_i)

            if i == n-1:
                c_L.append(ADVar(0.0))
            else:
                c_L.append(-ADVar(phi) * dt * u_i)

            # Right side (explicit part)
            if i == 0:
                a_R.append(ADVar(0.0))
            else:
                a_R.append(ADVar(1.0 - phi) * dt * l_i)

            b_R.append(ADVar(1.0) + ADVar(1.0 - phi) * dt * c_i)

            if i == n-1:
                c_R.append(ADVar(0.0))
            else:
                c_R.append(ADVar(1.0 - phi) * dt * u_i)

        return a_L, b_L, c_L, a_R, b_R, c_R

    def tridiag_solve(self, a, b, c, d):
        """Thomas algorithm for tridiagonal system"""
        n = len(b)
        c_prime = [None] * n
        d_prime = [None] * n

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            if i < n-1:
                c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        x = [None] * n
        x[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def cn_step(self, V, a_L, b_L, c_L, a_R, b_R, c_R, t_current):
        """Single Crank-Nicolson time step"""
        n = len(V)

        # Right hand side
        rhs = []
        for i in range(n):
            val = a_R[i] * (V[i-1] if i > 0 else ADVar(self._boundary_condition_left(t_current)))
            val = val + b_R[i] * V[i]
            val = val + c_R[i] * (V[i+1] if i < n-1 else ADVar(self._boundary_condition_right(t_current)))
            rhs.append(val)

        # Solve tridiagonal system
        V_new = self.tridiag_solve(a_L, b_L, c_L, rhs)
        return V_new

    def _compute_spline_second_derivatives(self, V: List[ADVar], S_grid: np.ndarray) -> List[ADVar]:
        """Compute Natural Cubic Spline second derivatives M_i"""
        n = len(V)

        # Build tridiagonal system for M_i
        # Natural boundary conditions: M[0] = M[-1] = 0
        A_diag = []
        A_lower = []
        A_upper = []
        b_rhs = []

        for i in range(1, n-1):
            h_i_minus = S_grid[i] - S_grid[i-1]
            h_i = S_grid[i+1] - S_grid[i]

            lambda_i = h_i_minus / (h_i_minus + h_i)
            mu_i = 1.0 - lambda_i

            A_diag.append(ADVar(2.0))
            if i > 1:
                A_lower.append(ADVar(lambda_i))
            if i < n-2:
                A_upper.append(ADVar(mu_i))

            # RHS
            d_i = (V[i+1] - V[i]) / h_i - (V[i] - V[i-1]) / h_i_minus
            d_i = ADVar(6.0) / ADVar(h_i_minus + h_i) * d_i
            b_rhs.append(d_i)

        # Solve for interior M values
        if len(A_diag) > 0:
            # Pad A_lower and A_upper
            while len(A_lower) < len(A_diag) - 1:
                A_lower.append(ADVar(0.0))
            while len(A_upper) < len(A_diag) - 1:
                A_upper.append(ADVar(0.0))

            M_interior = self.tridiag_solve(
                [ADVar(0.0)] + A_lower,
                A_diag,
                A_upper + [ADVar(0.0)],
                b_rhs
            )
        else:
            M_interior = []

        # Assemble full M with natural BCs
        M_vals = [ADVar(0.0)] + M_interior + [ADVar(0.0)]
        return M_vals

    def solve_pde_with_aad_nu(self, S0_val: float, sigma_val: float,
                              compute_hessian: bool = False,
                              verbose: bool = False) -> Dict:
        """
        Solve PDE using ν=σ² parametrization

        This method uses variance as the independent variable,
        then applies chain rule to recover Vega and Volga.

        Args:
            S0_val: Spot price
            sigma_val: Volatility
            compute_hessian: Whether to compute second-order Greeks
            verbose: Print debug info

        Returns:
            Dictionary with price, Greeks, and diagnostics
        """
        from aad_edge_pushing.edge_pushing.algo4_adjlist import algo4_adjlist

        t_start = time.perf_counter()
        global_tape.reset()

        # KEY CHANGE: Use ν = σ² as independent variable
        nu_val = sigma_val ** 2
        S0_var = ADVar(S0_val, requires_grad=True, name="S0")
        nu_var = ADVar(nu_val, requires_grad=True, name="nu")

        self.S0 = S0_val

        # Fixed grid
        N = self.N_base
        dt_val = self.T / N
        t_grid = np.linspace(0, self.T, N + 1)

        if verbose:
            print(f"  Using ν=σ² parametrization")
            print(f"  σ={sigma_val:.4f}, ν={nu_val:.4f}")
            print(f"  Grid: M={self.M}, N={N}")

        dt = ADVar(dt_val, requires_grad=False)

        # Build tridiagonal system with ν
        a_L, b_L, c_L, a_R, b_R, c_R = self.build_tridiagonal_cn(nu_var, dt)

        V_terminal = self._terminal_condition()
        V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

        # Time stepping
        for n in range(N):
            t_current = t_grid[n+1]
            V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_current)

        # Natural Cubic Spline interpolation
        S_interior = self.S_grid[1:-1]
        M_vals = self._compute_spline_second_derivatives(V, S_interior)

        # Find interval containing S0
        idx = np.searchsorted(S_interior, S0_val)
        if idx == 0:
            idx = 1
        elif idx >= len(V):
            idx = len(V) - 1

        i = idx - 1
        S_i = S_interior[i]
        S_i1 = S_interior[i + 1]
        V_i = V[i]
        V_i1 = V[i + 1]
        M_i = M_vals[i]
        M_i1 = M_vals[i + 1]
        h = S_i1 - S_i

        # Spline formula
        S_i_var = ADVar(S_i, requires_grad=False)
        S_i1_var = ADVar(S_i1, requires_grad=False)
        h_var = ADVar(h, requires_grad=False)

        A = (S_i1_var - S0_var) / h_var
        B = (S0_var - S_i_var) / h_var

        A3 = A * A * A
        B3 = B * B * B

        h2_over_6 = h_var * h_var / ADVar(6.0)

        price_var = (A * V_i + B * V_i1 +
                    (A3 - A) * h2_over_6 * M_i +
                    (B3 - B) * h2_over_6 * M_i1)

        price = price_var.val

        # Jacobian via backward pass
        price_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)

        delta = S0_var.adj  # ∂V/∂S0
        dV_dnu = nu_var.adj  # ∂V/∂ν

        # Apply chain rule to get Vega
        # Vega = ∂V/∂σ = (∂V/∂ν) · (∂ν/∂σ) = (∂V/∂ν) · 2σ
        vega = dV_dnu * 2 * sigma_val

        t_end = time.perf_counter()
        time_ms = (t_end - t_start) * 1000.0

        result = {
            'price': price,
            'delta': delta,
            'vega': vega,
            'dV_dnu': dV_dnu,  # Also return raw ∂V/∂ν
            'time_ms': time_ms,
            'jacobian': np.array([delta, vega])
        }

        # Hessian via Edge-Pushing
        if compute_hessian:
            global_tape.reset()

            # Recompute with fresh tape
            S0_var_h = ADVar(S0_val, requires_grad=True, name="S0")
            nu_var_h = ADVar(nu_val, requires_grad=True, name="nu")

            dt_h = ADVar(dt_val, requires_grad=False)
            a_L_h, b_L_h, c_L_h, a_R_h, b_R_h, c_R_h = self.build_tridiagonal_cn(nu_var_h, dt_h)

            V_h = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]
            for n in range(N):
                t_current = t_grid[n+1]
                V_h = self.cn_step(V_h, a_L_h, b_L_h, c_L_h, a_R_h, b_R_h, c_R_h, t_current)

            M_vals_h = self._compute_spline_second_derivatives(V_h, S_interior)

            # Use same interval
            V_i_h = V_h[i]
            V_i1_h = V_h[i + 1]
            M_i_h = M_vals_h[i]
            M_i1_h = M_vals_h[i + 1]

            S_i_var_h = ADVar(S_i, requires_grad=False)
            S_i1_var_h = ADVar(S_i1, requires_grad=False)
            h_var_h = ADVar(h, requires_grad=False)

            A_h = (S_i1_var_h - S0_var_h) / h_var_h
            B_h = (S0_var_h - S_i_var_h) / h_var_h

            A3_h = A_h * A_h * A_h
            B3_h = B_h * B_h * B_h

            h2_over_6_h = h_var_h * h_var_h / ADVar(6.0)

            price_var_h = (A_h * V_i_h + B_h * V_i1_h +
                          (A3_h - A_h) * h2_over_6_h * M_i_h +
                          (B3_h - B_h) * h2_over_6_h * M_i1_h)

            # Edge-Pushing for Hessian w.r.t. (S0, ν)
            hessian_nu = algo4_adjlist(price_var_h, [S0_var_h, nu_var_h])

            # Extract derivatives w.r.t. ν
            d2V_dS02 = hessian_nu[0, 0]  # ∂²V/∂S0²  (Gamma - unchanged)
            d2V_dS0_dnu = hessian_nu[0, 1]  # ∂²V/∂S0∂ν
            d2V_dnu2 = hessian_nu[1, 1]  # ∂²V/∂ν²

            # Apply chain rule to get Greeks w.r.t. σ
            # Gamma = ∂²V/∂S0² (unchanged)
            gamma = d2V_dS02

            # Vanna = ∂²V/∂S0∂σ = (∂²V/∂S0∂ν) · (∂ν/∂σ) = (∂²V/∂S0∂ν) · 2σ
            vanna = d2V_dS0_dnu * 2 * sigma_val

            # Volga = ∂²V/∂σ²
            # Using chain rule:
            # ∂²V/∂σ² = ∂/∂σ[(∂V/∂ν)·2σ]
            #         = (∂²V/∂ν∂σ)·2σ + (∂V/∂ν)·2
            #         = (∂²V/∂ν²)·(∂ν/∂σ)·2σ + (∂V/∂ν)·2
            #         = (∂²V/∂ν²)·2σ·2σ + (∂V/∂ν)·2
            #         = 4σ²(∂²V/∂ν²) + 2(∂V/∂ν)
            volga = 4 * sigma_val**2 * d2V_dnu2 + 2 * dV_dnu

            result['gamma'] = gamma
            result['vanna'] = vanna
            result['volga'] = volga
            result['d2V_dnu2'] = d2V_dnu2  # Also return raw ∂²V/∂ν²
            result['hessian_nu'] = hessian_nu  # Hessian w.r.t. (S0, ν)

        return result


if __name__ == "__main__":
    # Quick test
    from math import log, sqrt
    from scipy.stats import norm

    S0 = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    # Analytical Volga
    sqrt_T = sqrt(T)
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    n_d1 = norm.pdf(d1)
    vega_anal = S0 * n_d1 * sqrt_T
    volga_anal = vega_anal * d1 * d2 / sigma

    print("Testing ν=σ² Parametrization")
    print("=" * 60)
    print(f"Analytical Volga: {volga_anal:.8f}")
    print()

    # Test with ν parametrization
    pricer = BS_PDE_AAD_Nu(S0=S0, K=K, T=T, r=r, M=101, N_base=100)

    result = pricer.solve_pde_with_aad_nu(
        S0_val=S0,
        sigma_val=sigma,
        compute_hessian=True,
        verbose=True
    )

    print()
    print(f"Volga (ν-param):  {result['volga']:.8f}")
    print(f"Error: {abs(result['volga'] - volga_anal) / volga_anal * 100:.2f}%")
    print()
    print("If error < 5%: SUCCESS! ν parametrization works!")
