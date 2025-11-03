"""
BS PDE Solver with AAD and Rannacher Timestepping

This module implements Black-Scholes PDE solver with automatic adjoint differentiation (AAD)
and Rannacher timestepping to eliminate spurious oscillations in Greeks computation.

Key Features:
- Rannacher timestepping: Uses Backward Euler for first R steps, then Crank-Nicolson
- Eliminates spurious oscillations from payoff discontinuity at strike
- Significantly improves Gamma accuracy at high volatility
- Natural cubic spline interpolation for accurate Greeks

Theory:
-------
Standard Crank-Nicolson (φ=0.5) is A-stable but NOT L-stable:
- Lacks numerical damping for high-frequency components
- Initial payoff kink at K creates oscillations
- Oscillations persist through time-stepping
- Price (integral) remains accurate, but Greeks (derivatives) are polluted

Rannacher Solution (Rannacher 1984):
- First R steps: φ=1.0 (Backward Euler) → strong damping smooths the kink
- Remaining steps: φ=0.5 (Crank-Nicolson) → 2nd order accuracy
- Recommended: R=4 (industry standard)

Expected Improvements:
- Price: No change (already accurate)
- Delta: 33% → <5% error at σ=0.5
- Gamma: 104% → <10% error at σ=0.5
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import time

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape


class BS_PDE_AAD_Rannacher:
    """
    Black-Scholes PDE solver with AAD and Rannacher timestepping

    Solves the BS PDE using Crank-Nicolson with Rannacher smoothing:
    - First R timesteps use Backward Euler (φ=1.0) to smooth payoff kink
    - Remaining timesteps use Crank-Nicolson (φ=0.5) for accuracy

    Parameters:
    -----------
    S0 : float
        Initial stock price
    K : float
        Strike price
    T : float
        Time to maturity
    r : float
        Risk-free rate
    M : int, optional (default=151)
        Number of spatial grid points
    N_base : int, optional (default=150)
        Number of time steps
    center_on_S0 : bool, optional (default=False)
        Whether to center spatial grid on S0
    use_rannacher : bool, optional (default=True)
        Enable Rannacher timestepping
    rannacher_steps : int, optional (default=4)
        Number of Backward Euler steps (R=4 is industry standard)
    """

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 M: int = 151, N_base: int = 150, center_on_S0: bool = False,
                 use_rannacher: bool = True, rannacher_steps: int = 4):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N_base = N_base
        self.phi = 0.5  # Default CN, will be overridden by Rannacher

        # Rannacher parameters
        self.use_rannacher = use_rannacher
        self.rannacher_steps = rannacher_steps

        # Spatial grid
        if center_on_S0:
            # Build grid centered on S0 to avoid interpolation
            # Ensure S0 is exactly on a grid point
            # Adaptive S_max that scales with volatility (fixes high-vol pricing error)
            S_max = max(3.0 * K, S0 * np.exp((r + 3*sigma) * T))
            total_span = S_max  # from 0 to S_max

            # Find grid spacing that places S0 on a grid point
            # We want: S_grid[i] = S0 for some integer i
            # Try to make S0 be at approximately M//3 from the left
            # so there's room on both sides

            # Target: have about M//3 points below S0, 2*M//3 above
            n_below = M // 3
            n_above = M - n_below - 1  # -1 for the S0 point itself

            # Spacing
            dS_below = S0 / n_below if n_below > 0 else S0
            dS_above = (S_max - S0) / n_above if n_above > 0 else (S_max - S0)

            # Use uniform spacing (average of both)
            self.dS = (dS_below + dS_above) / 2.0

            # Build grid with S0 at index n_below
            S_min = S0 - n_below * self.dS
            S_max_actual = S0 + n_above * self.dS

            self.S_grid = np.linspace(S_min, S_max_actual, M)

            # Verify S0 is on grid (should be at index n_below)
            self.S0_idx = n_below

            # Adjust S_min to be >= 0
            if S_min < 0:
                S_min = 0.0
                self.S_grid = np.linspace(S_min, S_max, M)
                # Find closest grid point to S0
                self.S0_idx = np.argmin(np.abs(self.S_grid - S0))
                # Rebuild grid to center exactly on S0
                self.dS = S_max / (M - 1)
                # Create grid with S0 at specific index
                idx_target = max(1, M // 3)  # Put S0 at about 1/3 from left
                S_min_new = S0 - idx_target * self.dS
                if S_min_new < 0:
                    S_min_new = 0.0
                    self.dS = (S_max - 0.0) / (M - 1)
                    self.S_grid = np.linspace(0.0, S_max, M)
                    # Find where S0 would be
                    self.S0_idx = int(round(S0 / self.dS))
                    # Adjust grid to make S0 exact
                    self.S_grid[self.S0_idx] = S0
                else:
                    S_max_new = S0 + (M - 1 - idx_target) * self.dS
                    self.S_grid = np.linspace(S_min_new, S_max_new, M)
                    self.S0_idx = idx_target
                    # Ensure S0 is exact
                    self.S_grid[self.S0_idx] = S0
        else:
            # Original fixed grid
            # Adaptive S_max that scales with volatility (fixes high-vol pricing error)
            S_min = 0.0
            S_max = max(3.0 * K, S0 * np.exp((r + 3*sigma) * T))
            self.S_grid = np.linspace(S_min, S_max, M)
            self.dS = self.S_grid[1] - self.S_grid[0]
            self.S0_idx = None

    def _compute_spline_second_derivatives(self, V: List[ADVar], S_grid: np.ndarray) -> List[ADVar]:
        """
        Compute second derivatives M_i for natural cubic spline

        Solves tridiagonal system with natural boundary conditions M[0] = M[-1] = 0

        Args:
            V: List of function values at grid points (ADVars)
            S_grid: Grid points (numpy array)

        Returns:
            M_vals: Second derivatives M_i at each grid point (ADVars)
        """
        n = len(V)

        if n < 3:
            # Too few points, return zeros
            return [ADVar(0.0, requires_grad=False) for _ in range(n)]

        # Grid spacings
        h = np.diff(S_grid)  # h[i] = S[i+1] - S[i]

        # Build tridiagonal system: A * M = d
        # Interior equations for i = 1, ..., n-2
        # λ_i * M_{i-1} + 2*M_i + μ_i * M_{i+1} = d_i

        # Coefficients
        lambda_vals = []  # Lower diagonal
        mu_vals = []      # Upper diagonal
        d_vals = []       # RHS

        for i in range(1, n - 1):
            h_im1 = h[i - 1]  # h_{i-1} = S_i - S_{i-1}
            h_i = h[i]        # h_i = S_{i+1} - S_i

            lambda_i = h_im1 / (h_im1 + h_i)
            mu_i = h_i / (h_im1 + h_i)

            # RHS: d_i = 6 / (h_{i-1} + h_i) * [(V_{i+1} - V_i)/h_i - (V_i - V_{i-1})/h_{i-1}]
            d_i = (ADVar(6.0) / ADVar(h_im1 + h_i)) * (
                (V[i + 1] - V[i]) / ADVar(h_i) - (V[i] - V[i - 1]) / ADVar(h_im1)
            )

            lambda_vals.append(lambda_i)
            mu_vals.append(mu_i)
            d_vals.append(d_i)

        # Solve tridiagonal system with natural boundary conditions
        # M[0] = 0, M[n-1] = 0
        n_interior = n - 2  # Number of unknowns

        if n_interior == 0:
            # Only 2 points, both boundary
            return [ADVar(0.0, requires_grad=False) for _ in range(n)]

        # Tridiagonal matrix:
        # [  2      μ_1      0    ...   0    ]
        # [ λ_2      2      μ_2   ...   0    ]
        # [  0      λ_3      2    ...   0    ]
        # [ ...     ...     ...   ...  ...   ]
        # [  0       0       0    ... λ_{n-2} 2 ]

        # Thomas algorithm for tridiagonal system
        a = [ADVar(0.0)] + [ADVar(lam) for lam in lambda_vals]  # Lower diagonal (shifted)
        b = [ADVar(2.0) for _ in range(n_interior)]  # Main diagonal
        c = [ADVar(mu) for mu in mu_vals] + [ADVar(0.0)]  # Upper diagonal (shifted)
        d = d_vals

        # Forward elimination
        c_prime = [None] * n_interior
        d_prime = [None] * n_interior

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n_interior):
            denom = b[i] - a[i] * c_prime[i - 1]
            c_prime[i] = c[i] / denom if i < n_interior - 1 else ADVar(0.0)
            d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

        # Back substitution
        M_interior = [None] * n_interior
        M_interior[-1] = d_prime[-1]

        for i in range(n_interior - 2, -1, -1):
            M_interior[i] = d_prime[i] - c_prime[i] * M_interior[i + 1]

        # Add boundary conditions
        M_vals = [ADVar(0.0, requires_grad=False)] + M_interior + [ADVar(0.0, requires_grad=False)]

        return M_vals

    def _terminal_condition(self) -> np.ndarray:
        return np.maximum(self.S_grid - self.K, 0.0)

    def _boundary_condition_left(self, t: float) -> float:
        return 0.0

    def _boundary_condition_right(self, t: float) -> float:
        T_remain = self.T - t
        return self.S_grid[-1] - self.K * np.exp(-self.r * T_remain)

    def compute_adaptive_timesteps(self, sigma: float) -> Tuple[np.ndarray, int]:
        S_max = self.S_grid[-1]
        dS = self.dS
        alpha_max = (sigma**2 * S_max**2 / 2.0) / (dS**2)
        dt_stable = 0.5 / alpha_max if alpha_max > 1e-10 else self.T / self.N_base
        N = max(int(np.ceil(self.T / dt_stable)), self.N_base)
        t_grid = np.linspace(0, self.T, N + 1)
        return t_grid, N

    def build_tridiagonal_cn(self, sigma_var: ADVar, dt: ADVar, phi: float = 0.5):
        """
        Build tridiagonal coefficients for implicit/explicit scheme

        Args:
            sigma_var: Volatility as ADVar
            dt: Timestep as ADVar
            phi: Implicit/explicit weight
                 φ=0.5 → Crank-Nicolson (2nd order, no damping)
                 φ=1.0 → Backward Euler (1st order, strong damping)

        Returns:
            Tuple of (a_L, b_L, c_L, a_R, b_R, c_R) coefficient lists
        """
        n = self.M - 2
        dS = self.dS
        dS_sq = ADVar(dS**2, requires_grad=False)
        dS_2 = ADVar(2.0 * dS, requires_grad=False)

        a_L, b_L, c_L = [], [], []
        a_R, b_R, c_R = [], [], []

        for i in range(n):
            S_i = self.S_grid[i+1]
            S_i_var = ADVar(S_i, requires_grad=False)

            alpha_i = (sigma_var * sigma_var * S_i_var * S_i_var / ADVar(2.0)) / dS_sq
            beta_i = ADVar(self.r) * S_i_var / dS_2
            gamma_i = -ADVar(self.r)

            l_i = alpha_i - beta_i
            c_i = -ADVar(2.0) * alpha_i + gamma_i
            u_i = alpha_i + beta_i

            # Use phi parameter instead of self.phi
            phi_var = ADVar(phi)

            if i == 0:
                a_L.append(ADVar(0.0))
            else:
                a_L.append(-phi_var * dt * l_i)

            b_L.append(ADVar(1.0) - phi_var * dt * c_i)

            if i == n-1:
                c_L.append(ADVar(0.0))
            else:
                c_L.append(-phi_var * dt * u_i)

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

    def tridiag_solve(self, a: List[ADVar], b: List[ADVar], c: List[ADVar],
                     d: List[ADVar]) -> List[ADVar]:
        n = len(d)
        c_prime = [None] * n
        d_prime = [None] * n

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / denom if i < n-1 else ADVar(0.0)
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        x = [None] * n
        x[-1] = d_prime[-1]

        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def cn_step(self, V: List[ADVar], a_L: List[ADVar], b_L: List[ADVar],
                c_L: List[ADVar], a_R: List[ADVar], b_R: List[ADVar],
                c_R: List[ADVar], t_current: float) -> List[ADVar]:
        n = self.M - 2
        rhs = [None] * n

        for i in range(n):
            if i == 0:
                V_left = ADVar(self._boundary_condition_left(t_current), requires_grad=False)
                rhs[i] = b_R[i] * V[i] + c_R[i] * V[i+1] - a_R[i] * V_left
            elif i == n-1:
                V_right = ADVar(self._boundary_condition_right(t_current), requires_grad=False)
                rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] - c_R[i] * V_right
            else:
                rhs[i] = a_R[i] * V[i-1] + b_R[i] * V[i] + c_R[i] * V[i+1]

        V_new = self.tridiag_solve(a_L, b_L, c_L, rhs)
        return V_new

    def solve_pde_with_aad(self, S0_val: float, sigma_val: float,
                          compute_hessian: bool = False, verbose: bool = False,
                          fixed_grid: bool = False, use_analytical_volga: bool = False):
        """
        Solve PDE with AAD and Rannacher timestepping

        Args:
            S0_val: Initial stock price
            sigma_val: Volatility
            compute_hessian: Whether to compute Hessian (Gamma, Vanna, Volga)
            verbose: Print diagnostic information
            fixed_grid: If True, use fixed N (eliminates dN/dσ in derivatives)
                       If False, use adaptive timesteps (default legacy behavior)
            use_analytical_volga: If True, compute analytical Volga for BS model
                                 (only valid for European options with constant r, σ)
        """
        try:
            from aad_edge_pushing.edge_pushing.algo4_adjlist import algo4_adjlist
        except ImportError:
            from ..edge_pushing.algo4_adjlist import algo4_adjlist

        t_start = time.perf_counter()
        global_tape.reset()

        # KEY CHANGE: S0 is now also an ADVar!
        S0_var = ADVar(S0_val, requires_grad=True, name="S0")
        sigma_var = ADVar(sigma_val, requires_grad=True, name="sigma")
        self.S0 = S0_val

        # Grid selection: fixed or adaptive
        if fixed_grid:
            # Use fixed N to eliminate dN/dσ in Volga computation
            N = self.N_base
            dt_val = self.T / N
            t_grid = np.linspace(0, self.T, N + 1)
        else:
            # Legacy: adaptive timesteps (N depends on sigma_val)
            t_grid, N = self.compute_adaptive_timesteps(sigma_val)

        if verbose:
            dt_vals = np.diff(t_grid)
            print(f"  Grid: M={self.M}, N={N}")
            print(f"  dt={dt_vals[0]:.6f}, dS={self.dS:.4f}")
            if self.use_rannacher:
                print(f"  Rannacher: R={self.rannacher_steps} steps (φ=1.0), then φ=0.5")

        dt_val = t_grid[1] - t_grid[0]
        dt = ADVar(dt_val, requires_grad=False)

        # RANNACHER: Pre-build two sets of coefficients for performance
        # Backward Euler (φ=1.0) for first R steps
        a_L_be, b_L_be, c_L_be, a_R_be, b_R_be, c_R_be = \
            self.build_tridiagonal_cn(sigma_var, dt, phi=1.0)

        # Crank-Nicolson (φ=0.5) for remaining steps
        a_L_cn, b_L_cn, c_L_cn, a_R_cn, b_R_cn, c_R_cn = \
            self.build_tridiagonal_cn(sigma_var, dt, phi=0.5)

        V_terminal = self._terminal_condition()
        V = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

        # RANNACHER: Time stepping with adaptive φ
        for n in range(N):
            t_current = t_grid[n+1]

            # Select coefficients based on Rannacher strategy
            if self.use_rannacher and n < self.rannacher_steps:
                # First R steps: Backward Euler (strong damping)
                a_L, b_L, c_L = a_L_be, b_L_be, c_L_be
                a_R, b_R, c_R = a_R_be, b_R_be, c_R_be
            else:
                # Remaining steps: Crank-Nicolson (2nd order accuracy)
                a_L, b_L, c_L = a_L_cn, b_L_cn, c_L_cn
                a_R, b_R, c_R = a_R_cn, b_R_cn, c_R_cn

            V = self.cn_step(V, a_L, b_L, c_L, a_R, b_R, c_R, t_current)

        # Get price at S0 via Natural Cubic Spline (C² continuous)
        # KEY: Natural spline has globally consistent curvature M_i
        # This gives more accurate Gamma than local Hermite interpolation

        # Step 1: Compute spline second derivatives M_i (tridiagonal solve)
        # Natural boundary conditions: M[0] = M[-1] = 0
        n_interior = len(V)  # Number of interior points
        S_interior = self.S_grid[1:-1]  # Corresponding S values

        # Build tridiagonal system for M_i
        # System: λ_i * M_{i-1} + 2*M_i + μ_i * M_{i+1} = d_i
        # where λ_i, μ_i are based on grid spacing, d_i from V values

        M_vals = self._compute_spline_second_derivatives(V, S_interior)

        # Step 2: Find interval containing S0
        idx = np.searchsorted(S_interior, S0_val)
        if idx == 0:
            idx = 1
        elif idx >= n_interior:
            idx = n_interior - 1

        # Interval [S_i, S_{i+1}] where i = idx-1
        i = idx - 1
        S_i = S_interior[i]
        S_i1 = S_interior[i + 1]
        V_i = V[i]
        V_i1 = V[i + 1]
        # M_vals has same indexing as V (both interior points)
        M_i = M_vals[i]
        M_i1 = M_vals[i + 1]

        h = S_i1 - S_i

        # Step 3: Natural cubic spline formula with S0_var (ADVar)
        # A = (S_{i+1} - s) / h,  B = (s - S_i) / h
        S_i_var = ADVar(S_i, requires_grad=False)
        S_i1_var = ADVar(S_i1, requires_grad=False)
        h_var = ADVar(h, requires_grad=False)

        A = (S_i1_var - S0_var) / h_var
        B = (S0_var - S_i_var) / h_var

        # Cubic terms
        A3 = A * A * A
        B3 = B * B * B

        # Natural spline interpolation formula
        # p(s) = A*V_i + B*V_{i+1} + [(A³-A)*h²/6]*M_i + [(B³-B)*h²/6]*M_{i+1}
        h2_over_6 = h_var * h_var / ADVar(6.0)

        price_var = (A * V_i + B * V_i1 +
                    (A3 - A) * h2_over_6 * M_i +
                    (B3 - B) * h2_over_6 * M_i1)

        # Store interval info for Hessian computation
        spline_info = {
            'i': i,
            'S_i': S_i,
            'S_i1': S_i1,
            'V_i': V_i,
            'V_i1': V_i1,
            'M_i': M_i,
            'M_i1': M_i1,
            'h': h,
            'M_vals': M_vals
        }

        price = price_var.val

        # Jacobian via backward pass
        price_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)

        delta = S0_var.adj  # ∂V/∂S0 via AAD!
        vega = sigma_var.adj

        t_end = time.perf_counter()
        time_ms = (t_end - t_start) * 1000.0

        result = {
            'price': price,
            'delta': delta,
            'vega': vega,
            'time_ms': time_ms,
            'jacobian': np.array([delta, vega])
        }

        # Hessian via Edge-Pushing with Rannacher
        if compute_hessian:
            global_tape.reset()

            # Recompute with fresh tape
            S0_var_h = ADVar(S0_val, requires_grad=True, name="S0")
            sigma_var_h = ADVar(sigma_val, requires_grad=True, name="sigma")

            dt_h = ADVar(dt_val, requires_grad=False)

            # RANNACHER: Pre-build coefficients for Hessian computation
            a_L_be_h, b_L_be_h, c_L_be_h, a_R_be_h, b_R_be_h, c_R_be_h = \
                self.build_tridiagonal_cn(sigma_var_h, dt_h, phi=1.0)

            a_L_cn_h, b_L_cn_h, c_L_cn_h, a_R_cn_h, b_R_cn_h, c_R_cn_h = \
                self.build_tridiagonal_cn(sigma_var_h, dt_h, phi=0.5)

            V_h = [ADVar(v, requires_grad=False) for v in V_terminal[1:-1]]

            # RANNACHER: Time stepping with adaptive φ
            for n in range(N):
                t_current = t_grid[n+1]

                # Select coefficients based on Rannacher strategy
                if self.use_rannacher and n < self.rannacher_steps:
                    a_L_h, b_L_h, c_L_h = a_L_be_h, b_L_be_h, c_L_be_h
                    a_R_h, b_R_h, c_R_h = a_R_be_h, b_R_be_h, c_R_be_h
                else:
                    a_L_h, b_L_h, c_L_h = a_L_cn_h, b_L_cn_h, c_L_cn_h
                    a_R_h, b_R_h, c_R_h = a_R_cn_h, b_R_cn_h, c_R_cn_h

                V_h = self.cn_step(V_h, a_L_h, b_L_h, c_L_h, a_R_h, b_R_h, c_R_h, t_current)

            # Natural cubic spline interpolation with S0_var_h
            # Recompute spline second derivatives
            M_vals_h = self._compute_spline_second_derivatives(V_h, S_interior)

            # Use same interval from Jacobian
            i = spline_info['i']
            S_i = spline_info['S_i']
            S_i1 = spline_info['S_i1']
            h = spline_info['h']
            V_i_h = V_h[i]
            V_i1_h = V_h[i + 1]
            # M_vals_h has same indexing as V_h (both interior points)
            M_i_h = M_vals_h[i]
            M_i1_h = M_vals_h[i + 1]

            # Spline formula with S0_var_h
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

            # Edge-Pushing for full 2×2 Hessian
            hessian = algo4_adjlist(price_var_h, [S0_var_h, sigma_var_h])

            # Extract all second-order Greeks
            gamma = hessian[0, 0]  # ∂²V/∂S0² via AD!
            vanna = hessian[0, 1]  # ∂²V/∂S0∂σ via AD!
            volga_pde = hessian[1, 1]  # ∂²V/∂σ² from PDE

            result['gamma'] = gamma
            result['vanna'] = vanna
            result['volga'] = volga_pde
            result['volga_pde'] = volga_pde  # Store PDE-based Volga
            result['hessian'] = hessian

            # Optionally compute analytical Volga for BS model
            if use_analytical_volga:
                volga_analytical = self._compute_analytical_volga(
                    S0=S0_val,
                    K=self.K,
                    T=self.T,
                    r=self.r,
                    sigma=sigma_val,
                    vega=vega,
                    option_type='call'  # Volga is same for call/put
                )
                result['volga_analytical'] = volga_analytical
                # Override default volga with analytical for accuracy
                result['volga'] = volga_analytical

        return result

    def _compute_delta_on_grid(self, V_grid: np.ndarray, S0: float) -> float:
        """Compute Delta using FD on grid"""
        idx = np.searchsorted(self.S_grid[1:-1], S0)
        if idx == 0:
            idx = 1
        elif idx >= len(V_grid) - 1:
            idx = len(V_grid) - 2

        dS = self.dS
        delta = (V_grid[idx+1] - V_grid[idx-1]) / (2.0 * dS)
        return delta

    def _compute_gamma_on_grid(self, V_grid: np.ndarray, S0: float) -> float:
        """Compute Gamma using FD on grid (KEY FIX!)"""
        idx = np.searchsorted(self.S_grid[1:-1], S0)
        if idx == 0:
            idx = 1
        elif idx >= len(V_grid) - 1:
            idx = len(V_grid) - 2

        dS = self.dS
        gamma = (V_grid[idx+1] - 2.0 * V_grid[idx] + V_grid[idx-1]) / (dS**2)
        return gamma

    def _compute_analytical_volga(self, S0: float, K: float, T: float, r: float,
                                  sigma: float, vega: float, option_type: str = 'call') -> float:
        """
        Compute analytical Volga for Black-Scholes model

        Volga = ∂²V/∂σ² = Vega * (d1 * d2 / σ)

        Args:
            S0: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            vega: Already computed Vega value
            option_type: 'call' or 'put' (both have same Volga for European options)

        Returns:
            volga: Analytical second-order volatility Greek
        """
        from math import log, sqrt

        sqrt_T = sqrt(T)
        d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        volga = vega * d1 * d2 / sigma
        return volga

    def _solve_pde_numerical(self, S0: float, sigma: float, fixed_grid: bool = False) -> Tuple[float, np.ndarray]:
        """
        Solve PDE numerically without AAD (for bumping/finite difference) with Rannacher

        Args:
            S0: Initial stock price
            sigma: Volatility
            fixed_grid: If True, use fixed N (for consistent Volga via bumping)
        """
        if fixed_grid:
            N = self.N_base
            dt = self.T / N
            t_grid = np.linspace(0, self.T, N + 1)
        else:
            t_grid, N = self.compute_adaptive_timesteps(sigma)
            dt = t_grid[1] - t_grid[0]

        M = self.M
        n = M - 2
        dS = self.dS

        # RANNACHER: Build two sets of CN coefficients (numerical)
        # Backward Euler coefficients (φ=1.0)
        a_L_be = np.zeros(n)
        b_L_be = np.zeros(n)
        c_L_be = np.zeros(n)
        a_R_be = np.zeros(n)
        b_R_be = np.zeros(n)
        c_R_be = np.zeros(n)

        # Crank-Nicolson coefficients (φ=0.5)
        a_L_cn = np.zeros(n)
        b_L_cn = np.zeros(n)
        c_L_cn = np.zeros(n)
        a_R_cn = np.zeros(n)
        b_R_cn = np.zeros(n)
        c_R_cn = np.zeros(n)

        for i in range(n):
            S_i = self.S_grid[i+1]
            alpha_i = (sigma**2 * S_i**2 / 2.0) / (dS**2)
            beta_i = (self.r * S_i) / (2.0 * dS)
            gamma_i = -self.r

            l_i = alpha_i - beta_i
            c_i = -2.0 * alpha_i + gamma_i
            u_i = alpha_i + beta_i

            # Backward Euler (φ=1.0)
            phi_be = 1.0
            a_L_be[i] = -phi_be * dt * l_i if i > 0 else 0.0
            b_L_be[i] = 1.0 - phi_be * dt * c_i
            c_L_be[i] = -phi_be * dt * u_i if i < n-1 else 0.0
            a_R_be[i] = (1.0 - phi_be) * dt * l_i if i > 0 else 0.0
            b_R_be[i] = 1.0 + (1.0 - phi_be) * dt * c_i
            c_R_be[i] = (1.0 - phi_be) * dt * u_i if i < n-1 else 0.0

            # Crank-Nicolson (φ=0.5)
            phi_cn = 0.5
            a_L_cn[i] = -phi_cn * dt * l_i if i > 0 else 0.0
            b_L_cn[i] = 1.0 - phi_cn * dt * c_i
            c_L_cn[i] = -phi_cn * dt * u_i if i < n-1 else 0.0
            a_R_cn[i] = (1.0 - phi_cn) * dt * l_i if i > 0 else 0.0
            b_R_cn[i] = 1.0 + (1.0 - phi_cn) * dt * c_i
            c_R_cn[i] = (1.0 - phi_cn) * dt * u_i if i < n-1 else 0.0

        V_terminal = self._terminal_condition()
        V = V_terminal[1:-1].copy()

        # RANNACHER: Time stepping with adaptive coefficients
        for n_step in range(N):
            t_current = t_grid[n_step+1]
            V_left = self._boundary_condition_left(t_current)
            V_right = self._boundary_condition_right(t_current)

            # Select coefficients based on Rannacher strategy
            if self.use_rannacher and n_step < self.rannacher_steps:
                a_L, b_L, c_L = a_L_be, b_L_be, c_L_be
                a_R, b_R, c_R = a_R_be, b_R_be, c_R_be
            else:
                a_L, b_L, c_L = a_L_cn, b_L_cn, c_L_cn
                a_R, b_R, c_R = a_R_cn, b_R_cn, c_R_cn

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
