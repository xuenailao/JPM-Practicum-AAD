"""
Simple PDE Solver for Black-Scholes Equation
Pure numerical solver without AAD - for use in finite difference methods

"""

import numpy as np
from typing import Tuple

class SimplePDESolver:
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 M: int = 151, N_base: int = 150):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N_base = N_base
        self.phi = 0.5

        # Adaptive S_max that scales with volatility
        # Expanded from 3σ to 5σ to capture full option tail at high volatility
        S_max = max(5.0 * K, S0 * np.exp((r + 5*sigma) * T))

        # Log-scale grid: x = log(S)
        # This transforms the PDE to have constant diffusion coefficient 0.5*σ²
        # avoiding the S² term that causes instability at high S
        S_min = 1e-3  # Avoid log(0)
        x_min = np.log(S_min)
        x_max = np.log(S_max)

        # Uniform grid in log-space
        self.x_grid = np.linspace(x_min, x_max, M)
        self.dx = self.x_grid[1] - self.x_grid[0]

        # Convert to S-space for boundary conditions and payoff
        self.S_grid = np.exp(self.x_grid)
        self.S0_idx = None

    def _terminal_condition(self) -> np.ndarray:
        """Terminal payoff: max(S - K, 0) for call option"""
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_condition_left(self, t: float) -> float:
        """Boundary condition at S=0: option worth 0"""
        return 0.0

    def _boundary_condition_right(self, t: float) -> float:
        """Boundary condition at S=S_max: option worth S - K*exp(-r*(T-t))"""
        T_remain = self.T - t
        return self.S_grid[-1] - self.K * np.exp(-self.r * T_remain)

    def _solve_pde_numerical(self, S0: float, sigma: float) -> Tuple[float, np.ndarray]:
        """
        Solve PDE numerically without AAD (for bumping/finite difference)

        Args:
            S0: Initial stock price
            sigma: Volatility

        """
        
        N = self.N_base
        dt = self.T / N
        t_grid = np.linspace(0, self.T, N + 1)

        M = self.M
        n = M - 2
        dx = self.dx

        # Build CN coefficients for log-space PDE
        # In x = log(S) space: ∂V/∂t + (r - 0.5σ²)∂V/∂x + 0.5σ²∂²V/∂x² - rV = 0
        # Discretized: α∂²V/∂x² + β∂V/∂x + γV
        a_L = np.zeros(n)
        b_L = np.zeros(n)
        c_L = np.zeros(n)
        a_R = np.zeros(n)
        b_R = np.zeros(n)
        c_R = np.zeros(n)

        # Constant coefficients in log-space (independent of x!)
        alpha = 0.5 * sigma**2 / (dx**2)  # Diffusion (constant!)
        beta = (self.r - 0.5 * sigma**2) / (2.0 * dx)  # Drift
        gamma = -self.r  # Discount

        l = alpha - beta  # Lower diagonal
        c = -2.0 * alpha + gamma  # Main diagonal
        u = alpha + beta  # Upper diagonal

        for i in range(n):
            # Same coefficients for all grid points (uniform in x-space)
            l_i = l
            c_i = c
            u_i = u

            phi = self.phi

            a_L[i] = -phi * dt * l_i if i > 0 else 0.0
            b_L[i] = 1.0 - phi * dt * c_i
            c_L[i] = -phi * dt * u_i if i < n-1 else 0.0

            a_R[i] = (1.0 - phi) * dt * l_i if i > 0 else 0.0
            b_R[i] = 1.0 + (1.0 - phi) * dt * c_i
            c_R[i] = (1.0 - phi) * dt * u_i if i < n-1 else 0.0

        V_terminal = self._terminal_condition()
        V = V_terminal[1:-1].copy()

        # Time stepping
        for n_step in range(N):
            t_current = t_grid[n_step+1]
            V_left = self._boundary_condition_left(t_current)
            V_right = self._boundary_condition_right(t_current)

            rhs = np.zeros(n)
            for i in range(n):
                if i == 0:
                    rhs[i] = b_R[i] * V[i] + c_R[i] * V[i+1] - a_R[i] * V_left
                elif i == n-1:
                    rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] - c_R[i] * V_right
                else:
                    rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] + c_R[i] * V[i+1]

            # Thomas algorithm
            c_prime = np.zeros(n)
            d_prime = np.zeros(n)

            c_prime[0] = c_L[0] / b_L[0]
            d_prime[0] = rhs[0] / b_L[0]

            for i in range(1, n):
                denom = b_L[i] - a_L[i] * c_prime[i-1]
                c_prime[i] = c_L[i] / denom if i < n-1 else 0.0
                d_prime[i] = (rhs[i] - a_L[i] * d_prime[i-1]) / denom

            V[n-1] = d_prime[n-1]
            for i in range(n-2, -1, -1):
                V[i] = d_prime[i] - c_prime[i] * V[i+1]

        # Use Natural Cubic Spline interpolation instead of linear
        # This provides C² continuity and accurate second derivatives
        S_interior = self.S_grid[1:-1]

        # Compute spline second derivatives M_i (tridiagonal solve)
        n_pts = len(V)
        M = np.zeros(n_pts)

        # Build tridiagonal system for natural spline
        # Natural BC: M[0] = M[-1] = 0
        if n_pts > 2:
            # Interior equations
            A_tri = np.zeros((n_pts-2, n_pts-2))
            b_tri = np.zeros(n_pts-2)

            for i in range(n_pts-2):
                h_i = S_interior[i+1] - S_interior[i]
                h_i1 = S_interior[i+2] - S_interior[i+1] if i+1 < n_pts-1 else h_i

                if i > 0:
                    A_tri[i, i-1] = h_i / 6.0
                A_tri[i, i] = (h_i + h_i1) / 3.0
                if i < n_pts-3:
                    A_tri[i, i+1] = h_i1 / 6.0

                d_i = (V[i+2] - V[i+1]) / h_i1 - (V[i+1] - V[i]) / h_i
                b_tri[i] = d_i

            # Solve for interior M values
            M_interior = np.linalg.solve(A_tri, b_tri)
            M[1:-1] = M_interior

        # Find interval containing S0
        idx = np.searchsorted(S_interior, S0)
        if idx == 0:
            idx = 1
        elif idx >= n_pts:
            idx = n_pts - 1

        # Interpolate using cubic spline formula
        i = idx - 1
        S_i = S_interior[i]
        S_i1 = S_interior[i+1]
        V_i = V[i]
        V_i1 = V[i+1]
        M_i = M[i]
        M_i1 = M[i+1]
        h = S_i1 - S_i

        A = (S_i1 - S0) / h
        B = (S0 - S_i) / h

        price = (A * V_i + B * V_i1 +
                ((A**3 - A) * h**2 / 6.0) * M_i +
                ((B**3 - B) * h**2 / 6.0) * M_i1)

        return price, V
