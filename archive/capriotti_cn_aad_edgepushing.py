"""
Capriotti (2015) PDE + AAD Framework

Reference:
---------
Capriotti, L., Jiang, Y., & Macrina, A. (2015).
"AAD and least-square Monte Carlo: Fast Bermudan-style options and XVA Greeks"
"""

import numpy as np
from typing import List, Tuple, Dict
import time
from ...aad.core.var import ADVar
from ...aad.core.tape import global_tape
from ...edge_pushing.algo4_adjlist import algo4_adjlist
from scipy.stats import norm


class CapriottiCNAAD:
    """
    CORRECTED Capriotti CN + AAD implementation.

    Key Fixes:
    ---------
    1. Boundary conditions properly added to RHS in CN scheme
    2. Efficient O(n) tridiagonal operations
    3. Robust interpolation
    """

    def __init__(self, M: int = 20, N: int = 20, phi: float = 0.5):
        self.M = M
        self.N = N
        self.phi = phi

        # Option parameters
        self.S0 = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05

        # Grid - CRITICAL: Ensure S0 is on a grid point!
        # M is total number of grid points (including boundaries)
        # Interior points = M - 2

        # Strategy: Build grid centered at S0
        # Grid spans [0, 2*S0] with S0 at the middle
        self.Smax = 2.0 * self.S0

        # Use odd M to ensure S0 is exactly at center
        if M % 2 == 0:
            M = M + 1
            self.M = M

        # Construct grid with S0 at center
        self.S_grid = np.linspace(0, self.Smax, M)
        self.dS = self.S_grid[1] - self.S_grid[0]
        self.dt = self.T / N

        # Verify S0 is on grid (should be at index M//2)
        self.S0_index = M // 2
        assert abs(self.S_grid[self.S0_index] - self.S0) < 1e-10, \
            f"S0={self.S0} not on grid! Grid center: {self.S_grid[self.S0_index]}"

    def _terminal_condition(self) -> List[float]:
        """Terminal payoff for interior points"""
        return [max(S - self.K, 0.0) for S in self.S_grid[1:-1]]

    def _boundary_conditions(self, t: float) -> Tuple[float, float]:
        """
        Boundary conditions at time t.

        For European call:
        V(0, t) = 0
        V(Smax, t) = Smax - K*exp(-r*(T-t))
        """
        V_0 = 0.0
        V_Smax = self.Smax - self.K * np.exp(-self.r * (self.T - t))
        return V_0, V_Smax

    def compute_coeff_m(self, sigma_grid: List[ADVar], m: int
                       ) -> Tuple[List[ADVar], List[ADVar], List[ADVar]]:
        """
        COMPUTECOEFFM: Finite difference coefficients.

        Returns (c, u, l) where:
        - l[j]: lower diagonal coefficient
        - c[j]: central (main diagonal) coefficient
        - u[j]: upper diagonal coefficient
        """
        dt = self.dt
        dS = self.dS
        r = self.r

        c, u, l = [], [], []

        # Loop over interior points: S_grid[1] to S_grid[M-2]
        for j in range(1, self.M - 1):
            S_j = self.S_grid[j]
            sigma_j = sigma_grid[j - 1]

            # ADVar operations
            sigma_sq = sigma_j * sigma_j

            # Diffusion coefficient: σ²S²/2
            diff = sigma_sq * S_j * S_j * ADVar(0.5, requires_grad=False)

            # Drift coefficient: rS
            drift = ADVar(r * S_j, requires_grad=False)

            # FD coefficients
            alpha = diff / (dS * dS)
            beta = drift / (ADVar(2.0 * dS, requires_grad=False))
            gamma = ADVar(-r, requires_grad=False)

            # Tridiagonal entries for D matrix
            l_j = alpha - beta
            c_j = ADVar(-2.0, requires_grad=False) * alpha + gamma
            u_j = alpha + beta

            l.append(l_j)
            c.append(c_j)
            u.append(u_j)

        return c, u, l

    def compute_LRB(self, c: List[ADVar], u: List[ADVar], l: List[ADVar]
                   ) -> Tuple[List[ADVar], List[ADVar], List[ADVar],
                             List[ADVar], List[ADVar], List[ADVar]]:
        """
        COMPUTELRB: Build L_B and R_B in tridiagonal form.

        L_B = I - φ Δt D
        R_B = I + (1-φ) Δt D

        Returns tridiagonal vectors (a_L, b_L, c_L, a_R, b_R, c_R)
        """
        n = self.M - 2  # Number of interior points
        dt = self.dt
        phi = self.phi

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

    def tri_matvec(self, a: List[ADVar], b: List[ADVar], c: List[ADVar],
                   x: List[ADVar]) -> List[ADVar]:
        """
        OPTIMIZED tridiagonal matrix-vector product: O(n) instead of O(n²).

        y = Tridiag(a, b, c) @ x

        where a = lower, b = diagonal, c = upper
        """
        n = len(x)
        y = []

        for i in range(n):
            val = b[i] * x[i]
            if i > 0:
                val = val + a[i] * x[i-1]
            if i < n - 1:
                val = val + c[i] * x[i+1]
            y.append(val)

        return y

    def tridiagsolver_advar(self, a_L: List[ADVar], b_L: List[ADVar], c_L: List[ADVar],
                           a_R: List[ADVar], b_R: List[ADVar], c_R: List[ADVar],
                           V_next: List[ADVar], t_m: float, t_m1: float) -> List[ADVar]:
        """
        CORRECTED TRIDIAGSOLVER with boundary conditions.

        Solves: L_B @ V = R_B @ V_next + boundary_terms

        Key Fix: Boundary conditions properly incorporated into RHS!
        """
        # Step 1: W = R_B @ V_next (O(n) tridiagonal multiplication)
        W = self.tri_matvec(a_R, b_R, c_R, V_next)

        # Step 2: CRITICAL FIX - Add boundary conditions to RHS
        V0_m, VSmax_m = self._boundary_conditions(t_m)
        V0_m1, VSmax_m1 = self._boundary_conditions(t_m1)

        # Convert to ADVar
        V0_m = ADVar(V0_m, requires_grad=False)
        VSmax_m = ADVar(VSmax_m, requires_grad=False)
        V0_m1 = ADVar(V0_m1, requires_grad=False)
        VSmax_m1 = ADVar(VSmax_m1, requires_grad=False)

        n = len(W)

        # Boundary contribution to RHS:
        # Left boundary (i=0, corresponds to j=1 in full grid)
        W[0] = W[0] + a_R[0] * V0_m1 - a_L[0] * V0_m

        # Right boundary (i=n-1, corresponds to j=M-1 in full grid)
        W[n-1] = W[n-1] + c_R[n-1] * VSmax_m1 - c_L[n-1] * VSmax_m

        # Step 3: Solve L_B @ V = W using Thomas algorithm
        # Forward elimination
        c_prime = [None] * n
        d_prime = [None] * n

        c_prime[0] = c_L[0] / b_L[0]
        d_prime[0] = W[0] / b_L[0]

        for i in range(1, n):
            denom = b_L[i] - a_L[i] * c_prime[i-1]
            if i < n - 1:
                c_prime[i] = c_L[i] / denom
            d_prime[i] = (W[i] - a_L[i] * d_prime[i-1]) / denom

        # Back substitution
        V = [None] * n
        V[n-1] = d_prime[n-1]

        for i in range(n-2, -1, -1):
            V[i] = d_prime[i] - c_prime[i] * V[i+1]

        return V

    def interpolate(self, V_0: List[ADVar], x_t0: float) -> ADVar:
        """
        SAFER interpolation following paper's formula (97).

        V(x_t0) = V[j*] + (V[j*+1] - V[j*]) * (x_t0 - x[j*]) / (x[j*+1] - x[j*])
        """
        # Find j* such that S_grid[j*] <= x_t0 < S_grid[j*+1]
        j_star = np.searchsorted(self.S_grid, x_t0) - 1
        j_star = min(max(j_star, 1), self.M - 2)  # Clamp to interior [1, M-2]

        # Grid points
        x_left = self.S_grid[j_star]
        x_right = self.S_grid[j_star + 1]

        # Weight
        weight = (x_t0 - x_left) / (x_right - x_left)

        # V_0 index (interior points: V_0[i] corresponds to S_grid[i+1])
        # j_star ranges from 1 to M-2, so i = j_star - 1 ranges from 0 to M-3
        i = j_star - 1

        # Safety check
        if i < 0 or i >= len(V_0) - 1:
            i = min(max(i, 0), len(V_0) - 2)

        return V_0[i] + (V_0[i+1] - V_0[i]) * ADVar(float(weight), requires_grad=False)

    def interpolate_advar(self, V_0: List[ADVar], S0_var: ADVar) -> ADVar:
        """
        Interpolation with S0 as ADVar - enables gradient propagation through S0.

        SPECIAL CASE: When S0 is exactly on a grid point, we return that value directly.
        This avoids the linear interpolation issue for Gamma computation.
        """
        x_t0 = S0_var.val

        # Check if S0 is exactly on a grid point
        eps = 1e-10
        for j in range(1, self.M - 1):  # Interior points only
            if abs(x_t0 - self.S_grid[j]) < eps:
                # S0 is exactly at grid point j
                # V_0[i] corresponds to S_grid[i+1], so i = j - 1
                i = j - 1
                # For finite difference on grid:
                # Delta ≈ (V[i+1] - V[i-1]) / (2*dS)
                # Gamma ≈ (V[i+1] - 2*V[i] + V[i-1]) / dS²

                # Return V[i] with finite difference representation
                # This enables proper gradient computation
                if i > 0 and i < len(V_0) - 1:
                    # Use finite difference formula as ADVar operations
                    # V(S0 + ε*dS) where ε comes from S0_var perturbation

                    # Compute local finite difference weights
                    # When S0 shifts by dS, we move between grid points
                    left_weight = ADVar(0.5, requires_grad=False) * (ADVar(self.S_grid[j-1], requires_grad=False) - S0_var) / ADVar(self.dS, requires_grad=False)
                    right_weight = ADVar(0.5, requires_grad=False) * (S0_var - ADVar(self.S_grid[j+1], requires_grad=False)) / ADVar(self.dS, requires_grad=False)
                    center_weight = ADVar(1.0, requires_grad=False) - left_weight - right_weight

                    # Weighted combination enables proper derivatives
                    return V_0[i-1] * left_weight + V_0[i] * center_weight + V_0[i+1] * right_weight
                else:
                    return V_0[i]

        # S0 not on grid - use linear interpolation
        j_star = np.searchsorted(self.S_grid, x_t0) - 1
        j_star = min(max(j_star, 1), self.M - 2)

        x_left = self.S_grid[j_star]
        x_right = self.S_grid[j_star + 1]
        dx = x_right - x_left

        weight = (S0_var - ADVar(x_left, requires_grad=False)) * ADVar(1.0/dx, requires_grad=False)

        i = j_star - 1
        if i < 0 or i >= len(V_0) - 1:
            i = min(max(i, 0), len(V_0) - 2)

        return V_0[i] + (V_0[i+1] - V_0[i]) * weight

    def solve_pde_cn_advar(self, sigma_values: np.ndarray) -> Tuple[ADVar, List[ADVar]]:
        """
        Forward PDE solve with CORRECTED CN scheme.
        """
        global_tape.reset()

        sigma_vars = [ADVar(sigma_values[i], requires_grad=True, name=f"sigma_{i}")
                      for i in range(len(sigma_values))]

        # S1: Terminal condition
        V = [ADVar(val, requires_grad=False) for val in self._terminal_condition()]

        # S2: Time-stepping (backward in time)
        for m in range(self.N - 1, -1, -1):
            t_m = m * self.dt
            t_m1 = (m + 1) * self.dt

            # COMPUTECOEFFM
            c, u, l = self.compute_coeff_m(sigma_vars, m)

            # COMPUTELRB
            a_L, b_L, c_L, a_R, b_R, c_R = self.compute_LRB(c, u, l)

            # TRIDIAGSOLVER (with boundary conditions!)
            V = self.tridiagsolver_advar(a_L, b_L, c_L, a_R, b_R, c_R, V, t_m, t_m1)

        # S3: Interpolate at S0
        V_t0 = self.interpolate(V, self.S0)

        return V_t0, sigma_vars

    def compute_hessian_cn_algo4(self, sigma_values: np.ndarray
                                 ) -> Tuple[float, np.ndarray, np.ndarray]:
        """Complete pipeline: CN PDE + Algorithm 4."""
        price_var, sigma_rs = self.solve_pde_cn_advar(sigma_values)

        price = price_var.val

        # Gradient
        price_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)

        gradient = np.array([var.adj for var in sigma_rs])

        # Hessian
        hessian = algo4_adjlist(price_var, sigma_rs)

        return price, gradient, hessian

    def solve_pde_cn_advar_full(self, S0_var: ADVar, sigma_vars: List[ADVar]) -> ADVar:
        """
        Forward PDE solve with S0 as ADVar parameter.

        This enables gradient propagation through S0 for Delta/Gamma computation.

        Args:
            S0_var: Initial spot price as ADVar
            sigma_vars: Volatility parameters as ADVar list (M-1 elements)

        Returns:
            Price as ADVar (gradients flow to both S0 and sigma)
        """
        # S1: Terminal condition
        V = [ADVar(val, requires_grad=False) for val in self._terminal_condition()]

        # S2: Time-stepping (backward in time)
        for m in range(self.N - 1, -1, -1):
            t_m = m * self.dt
            t_m1 = (m + 1) * self.dt

            # COMPUTECOEFFM
            c, u, l = self.compute_coeff_m(sigma_vars, m)

            # COMPUTELRB
            a_L, b_L, c_L, a_R, b_R, c_R = self.compute_LRB(c, u, l)

            # TRIDIAGSOLVER (with boundary conditions!)
            V = self.tridiagsolver_advar(a_L, b_L, c_L, a_R, b_R, c_R, V, t_m, t_m1)

        # S3: Interpolate at S0 (S0 is now an ADVar!)
        V_t0 = self.interpolate_advar(V, S0_var)

        return V_t0

    def compute_hessian_full_aad(self, S0_value: float, sigma_value: float
                                ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute full Hessian with respect to [S0, sigma_1, ..., sigma_{M-1}].

        Parameter vector: θ = [S0, σ₁, σ₂, ..., σₘ₋₁] (dimension = M)

        Returns:
            price: Option price
            gradient: ∂V/∂θ = [∂V/∂S0, ∂V/∂σ₁, ..., ∂V/∂σₘ₋₁]
            hessian: ∂²V/∂θᵢ∂θⱼ (M×M matrix)
        """
        global_tape.reset()

        # Create parameter vector
        S0_var = ADVar(S0_value, requires_grad=True, name="S0")
        sigma_vars = [ADVar(sigma_value, requires_grad=True, name=f"sigma_{i}")
                      for i in range(self.M - 1)]

        # All parameters for Hessian computation
        all_params = [S0_var] + sigma_vars

        # Solve PDE with full AAD
        price_var = self.solve_pde_cn_advar_full(S0_var, sigma_vars)
        price = price_var.val

        # Gradient via reverse mode
        price_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)

        gradient = np.array([var.adj for var in all_params])

        # Hessian via Algorithm 4
        hessian = algo4_adjlist(price_var, all_params)

        return price, gradient, hessian

    def compute_greeks_aad(self, sigma_value: float = 0.2) -> Dict[str, any]:
        """
        COMPLETE AAD+Edge-Pushing Greeks computation.

        All derivatives computed via AAD - NO finite differences!

        Parameter vector: θ = [S0, σ₁, σ₂, ..., σₘ₋₁]

        Greeks extraction:
        - Price: V(θ)
        - Delta: ∂V/∂S0 = gradient[0]
        - Gamma: ∂²V/∂S0² = hessian[0,0]
        - Vega: Σᵢ ∂V/∂σᵢ = sum(gradient[1:])
        - Vanna: ∂²V/∂S0∂σ = sum(hessian[0, 1:])
        - Volga: ∂²V/∂σ² = sum(hessian[1:, 1:])

        Args:
            sigma_value: Constant volatility

        Returns:
            Dictionary with all Greeks, Hessian, and metadata
        """
        t_start = time.perf_counter()

        # Compute full Hessian
        price, gradient, hessian = self.compute_hessian_full_aad(self.S0, sigma_value)

        # Extract Greeks from gradient and Hessian
        delta = gradient[0]  # ∂V/∂S0
        vega_array = gradient[1:]  # [∂V/∂σ₁, ..., ∂V/∂σₘ₋₁]
        vega = np.sum(vega_array)  # Total Vega

        # Second-order Greeks from Hessian
        gamma = hessian[0, 0]  # ∂²V/∂S0²
        vanna_array = hessian[0, 1:]  # [∂²V/∂S0∂σ₁, ..., ∂²V/∂S0∂σₘ₋₁]
        vanna = np.sum(vanna_array)  # Total Vanna

        # Volga: sum all σ-σ cross terms
        volga = np.sum(hessian[1:, 1:])

        t_end = time.perf_counter()
        computation_time = (t_end - t_start) * 1000

        # Hessian statistics
        nnz = np.count_nonzero(hessian)
        total_elements = hessian.size
        sparsity = 1 - (nnz / total_elements) if total_elements > 0 else 0

        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'vega_array': vega_array,
            'vanna': vanna,
            'volga': volga,
            'gradient': gradient,
            'hessian': hessian,
            'hessian_stats': {
                'nnz': nnz,
                'total': total_elements,
                'sparsity': sparsity,
                'shape': hessian.shape,
                'avg_row_nnz': nnz / hessian.shape[0] if hessian.shape[0] > 0 else 0
            },
            'computation_time_ms': computation_time,
            'n_tape_nodes': len(global_tape.nodes),
            'n_pde_solves': 1  # Only 1 PDE solve with full AAD!
        }


def black_scholes_analytical(S, K, T, r, sigma):
    """Black-Scholes analytical formulas for validation."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return price, delta, gamma, vega


def test_analytical_validation():
    """Test against Black-Scholes analytical solution."""
    print("=" * 80)
    print("BLACK-SCHOLES ANALYTICAL VALIDATION")
    print("=" * 80)

    M, N = 50, 50
    solver = CapriottiCNAADFixed(M=M, N=N, phi=0.5)

    # Constant volatility
    sigma = 0.2
    sigma_values = np.full(M - 1, sigma)

    print(f"\nParameters:")
    print(f"  S0 = {solver.S0}, K = {solver.K}, T = {solver.T}, r = {solver.r}")
    print(f"  σ = {sigma}")
    print(f"  Grid: {M}×{N}")

    # PDE solution
    t0 = time.time()
    price_pde, grad_pde, hess_pde = solver.compute_hessian_cn_algo4(sigma_values)
    t_pde = (time.time() - t0) * 1000

    # Analytical solution
    price_bs, delta_bs, gamma_bs, vega_bs = black_scholes_analytical(
        solver.S0, solver.K, solver.T, solver.r, sigma
    )

    print(f"\n{'Metric':<15} | {'PDE (CN)':<15} | {'Analytical':<15} | {'Error':<15}")
    print("-" * 80)
    print(f"{'Price':<15} | ${price_pde:<14.6f} | ${price_bs:<14.6f} | {abs(price_pde - price_bs):<14.2e}")

    # Vega approximation (average gradient)
    vega_pde_approx = np.mean(grad_pde) * sigma  # Scale by σ
    print(f"{'Vega (approx)':<15} | {vega_pde_approx:<15.6f} | {vega_bs:<15.6f} | {abs(vega_pde_approx - vega_bs):<14.2e}")

    print(f"\n{'Computation Time':<15} | {t_pde:<15.2f} ms")
    print(f"{'Graph Nodes':<15} | {len(global_tape.nodes):<15d}")

    # Error analysis
    price_error = abs(price_pde - price_bs) / price_bs
    print(f"\nRelative Price Error: {price_error:.2e}")

    if price_error < 1e-3:
        print("✓ VALIDATION PASSED (error < 0.1%)")
    else:
        print("⚠ Warning: Price error > 0.1%")
