"""
Method 2: Double Bumping (FIXED VERSION)
Fix: Use grid-based finite difference instead of interpolated price

Key fixes:
1. Return full price grid V(S, t=0) instead of interpolated V(S0)
2. Compute Gamma using finite difference on grid
3. This avoids Gamma=0 caused by linear interpolation
"""

import numpy as np
import time
from typing import Dict, Tuple
from .simple_pde_solver import SimplePDESolver


class DoubleBumpingFixed:
    """
    Fixed bumping method that computes Gamma correctly

    Key difference: Works with price grids, not interpolated prices
    """

    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N
        self.solver = SimplePDESolver(M, N)

    def _solve_pde_grid(self, S0: float, K: float, T: float, r: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve PDE and return full grid instead of interpolated price

        Returns:
            S_grid: Stock price grid
            V_grid: Option value grid at t=0
        """
        M = self.M
        N_base = self.N

        # Spatial grid
        S_max = 3.0 * K
        S_min = 0.0
        dS = (S_max - S_min) / M
        S_grid = np.linspace(S_min, S_max, M + 1)

        # Adaptive time stepping
        alpha_max = (sigma**2 * S_max**2 / 2.0) / (dS**2)
        dt_max = 0.5 / alpha_max if alpha_max > 1e-10 else T / N_base
        N = max(int(np.ceil(T / dt_max)), N_base)
        dt = T / N
        t_grid = np.linspace(0, T, N + 1)

        # Terminal condition
        V = np.maximum(S_grid - K, 0.0)

        # CN parameters
        phi = 0.5
        n = M - 1

        # Build tridiagonal coefficients
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
            gamma = -r

            l_i = alpha_i - beta_i
            c_i = -2.0 * alpha_i + gamma
            u_i = alpha_i + beta_i

            if i == 0:
                a_L[i] = 0.0
            else:
                a_L[i] = -phi * dt * l_i
            b_L[i] = 1.0 - phi * dt * c_i
            if i == n - 1:
                c_L[i] = 0.0
            else:
                c_L[i] = -phi * dt * u_i

            if i == 0:
                a_R[i] = 0.0
            else:
                a_R[i] = (1.0 - phi) * dt * l_i
            b_R[i] = 1.0 + (1.0 - phi) * dt * c_i
            if i == n - 1:
                c_R[i] = 0.0
            else:
                c_R[i] = (1.0 - phi) * dt * u_i

        # Time stepping
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

            V_new_interior = self.solver._thomas_algorithm(a_L, b_L, c_L, rhs)
            V = np.zeros(M + 1)
            V[0] = V_left
            V[1:M] = V_new_interior
            V[M] = V_right

        return S_grid, V

    def _get_price_at_S0(self, S_grid: np.ndarray, V_grid: np.ndarray, S0: float) -> float:
        """Get price at S0 using linear interpolation"""
        return np.interp(S0, S_grid, V_grid)

    def _get_delta_at_S0(self, S_grid: np.ndarray, V_grid: np.ndarray, S0: float) -> float:
        """Compute Delta at S0 using finite difference on grid"""
        dS = S_grid[1] - S_grid[0]

        # Find grid index closest to S0
        idx = np.searchsorted(S_grid, S0)
        if idx == 0:
            idx = 1
        elif idx >= len(S_grid):
            idx = len(S_grid) - 2

        # Central difference
        delta = (V_grid[idx+1] - V_grid[idx-1]) / (S_grid[idx+1] - S_grid[idx-1])
        return delta

    def _get_gamma_at_S0(self, S_grid: np.ndarray, V_grid: np.ndarray, S0: float) -> float:
        """Compute Gamma at S0 using finite difference on grid"""
        dS = S_grid[1] - S_grid[0]

        # Find grid index closest to S0
        idx = np.searchsorted(S_grid, S0)
        if idx == 0:
            idx = 1
        elif idx >= len(S_grid):
            idx = len(S_grid) - 2

        # Second-order central difference
        gamma = (V_grid[idx+1] - 2.0 * V_grid[idx] + V_grid[idx-1]) / (dS**2)
        return gamma

    def compute_greeks(self, S0: float, K: float, T: float, r: float, sigma: float,
                      eps_S: float = 1.0, eps_sigma: float = 0.01, eps_r: float = 0.0001) -> Dict:
        """
        Compute Greeks using grid-based finite difference

        Key improvement: Gamma computed from price grid, not interpolated price
        """
        t_start = time.perf_counter()

        # Solve PDE on grids (not just interpolated prices)
        S_grid_0, V_grid_0 = self._solve_pde_grid(S0, K, T, r, sigma)

        # For parameter perturbations, we still need interpolated prices
        V0 = self._get_price_at_S0(S_grid_0, V_grid_0, S0)

        # Jacobian: Perturb input parameters
        V_sigma_plus = self.solver.solve_pde(S0, K, T, r, sigma + eps_sigma)
        V_sigma_minus = self.solver.solve_pde(S0, K, T, r, sigma - eps_sigma)
        vega = (V_sigma_plus - V_sigma_minus) / (2.0 * eps_sigma)

        # Rho: 利率敏感性 ∂V/∂r (bumping on r)
        V_r_plus = self.solver.solve_pde(S0, K, T, r + eps_r, sigma)
        V_r_minus = self.solver.solve_pde(S0, K, T, r - eps_r, sigma)
        rho = (V_r_plus - V_r_minus) / (2.0 * eps_r)

        # Delta and Gamma: Compute from grid (NOT from interpolation)
        delta = self._get_delta_at_S0(S_grid_0, V_grid_0, S0)
        gamma = self._get_gamma_at_S0(S_grid_0, V_grid_0, S0)

        # Volga: Second-order finite difference on parameter
        volga = (V_sigma_plus - 2.0 * V0 + V_sigma_minus) / (eps_sigma**2)

        # Vanna: Mixed derivative ∂²V/∂S0∂σ
        # Need grids at σ±ε
        S_grid_sigma_plus, V_grid_sigma_plus = self._solve_pde_grid(S0, K, T, r, sigma + eps_sigma)
        S_grid_sigma_minus, V_grid_sigma_minus = self._solve_pde_grid(S0, K, T, r, sigma - eps_sigma)

        delta_sigma_plus = self._get_delta_at_S0(S_grid_sigma_plus, V_grid_sigma_plus, S0)
        delta_sigma_minus = self._get_delta_at_S0(S_grid_sigma_minus, V_grid_sigma_minus, S0)

        vanna = (delta_sigma_plus - delta_sigma_minus) / (2.0 * eps_sigma)

        jacobian = np.array([delta, vega])
        hessian = np.array([[gamma, vanna], [vanna, volga]])

        t_end = time.perf_counter()
        time_ms = (t_end - t_start) * 1000.0

        return {
            'price': V0,
            'jacobian': jacobian,
            'hessian': hessian,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'rho': rho,
            'vanna': vanna,
            'volga': volga,
            'time_ms': time_ms,
            'pde_solves': 7  # Base + 2 for vega + 2 for rho + 2 more grids for vanna
        }


def test_fixed_bumping():
    """Test fixed bumping method"""
    from scipy.stats import norm

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    # Analytical
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    phi_d1 = norm.pdf(d1)

    gamma_analytical = phi_d1 / (S0 * sigma * sqrt_T)
    print(f"Analytical Gamma: {gamma_analytical:.10f}")

    # Test
    method = DoubleBumpingFixed(M=50, N=150)
    result = method.compute_greeks(S0, K, T, r, sigma)

    print(f"Fixed Bumping Gamma: {result['gamma']:.10f}")
    print(f"Error: {abs(result['gamma'] - gamma_analytical) / gamma_analytical * 100:.2f}%")


if __name__ == "__main__":
    test_fixed_bumping()
