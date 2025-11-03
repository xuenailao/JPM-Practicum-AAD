"""
Greeks Computation Methods Comparison Framework

Implements and compares multiple methods for computing option Greeks:
1. Analytical (Black-Scholes) - Ground Truth
2. Finite Difference Bumping - Traditional numerical method
3. AAD with Perturbation (Method A) - Based on multi-asset cross-gamma approach
4. AAD with Single Sigma - Fixed sigma model
5. Grid Direct - Extract from PDE grid without interpolation

Reference: Multi-asset Greeks calculation (FD vs AAD diagram)
"""

import numpy as np
from typing import Dict, Tuple, List
import time
from scipy.stats import norm
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aad_edge_pushing.pde.AADgraph.capriotti_cn_aad_edgepushing import (
    CapriottiCNAAD,
    black_scholes_analytical
)
from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.edge_pushing.algo4_adjlist import algo4_adjlist


class GreeksMethodA:
    """
    Method A: Perturbation + AAD

    Based on the multi-asset cross-gamma diagram approach.
    Idea: Perturb S0, solve PDE multiple times, use AAD to compute gradients,
          then combine using finite difference formulas.

    For Gamma: ∂²V/∂S0²
    - Solve PDE at S0-ε, S0, S0+ε
    - Use AAD to get ∂V/∂σ at each point
    - Apply centered difference formula for second derivative
    """

    def __init__(self, M: int = 51, N: int = 50):
        self.M = M
        self.N = N
        self.base_solver = CapriottiCNAAD(M=M, N=N)

    def _solve_at_S0(self, S0: float, sigma: float) -> Tuple[float, float]:
        """
        Solve PDE at given S0 and compute price + vega using AAD.

        FIXED: Use SINGLE sigma parameter instead of M-1 parameters.

        Returns:
            (price, vega) where vega = ∂V/∂σ via AAD
        """
        # Create a fresh solver with modified S0
        solver = CapriottiCNAAD(M=self.M, N=self.N)

        # Override S0 (need to rebuild grid)
        solver.S0 = S0
        solver.Smax = 2.0 * S0
        solver.S_grid = np.linspace(0, solver.Smax, self.M)
        solver.dS = solver.S_grid[1] - solver.S_grid[0]
        solver.S0_index = self.M // 2

        # Solve with single sigma parameter for AAD
        global_tape.reset()

        # ✅ FIX: Single sigma ADVar (not M-1 independent variables!)
        sigma_var = ADVar(sigma, requires_grad=True, name="sigma")

        # All grid points use the SAME sigma_var (constant volatility)
        sigma_grid = [sigma_var] * (self.M - 1)

        # PDE solve
        V_grid = [ADVar(val, requires_grad=False)
                  for val in solver._terminal_condition()]

        for m in range(self.N - 1, -1, -1):
            t_m = m * solver.dt
            t_m1 = (m + 1) * solver.dt

            c, u, l = solver.compute_coeff_m(sigma_grid, m)
            a_L, b_L, c_L, a_R, b_R, c_R = solver.compute_LRB(c, u, l)
            V_grid = solver.tridiagsolver_advar(a_L, b_L, c_L, a_R, b_R, c_R,
                                                V_grid, t_m, t_m1)

        # Interpolate at S0
        price_var = solver.interpolate(V_grid, S0)
        price = price_var.val

        # Compute gradient via reverse mode
        price_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)

        # ✅ Now vega is the gradient wrt the SINGLE sigma parameter
        vega = sigma_var.adj

        return price, vega

    def compute_greeks(self, S0: float, K: float, T: float, r: float,
                      sigma: float, eps_S: float = None) -> Dict:
        """
        Compute Greeks using Method A: Perturbation + AAD.

        Args:
            S0, K, T, r, sigma: Option parameters
            eps_S: Perturbation size for S0 (default: dS from grid)

        Returns:
            Dictionary with all Greeks
        """
        t_start = time.perf_counter()

        # Auto-select eps_S based on grid spacing
        if eps_S is None:
            dS = 200.0 / self.M  # Smax = 2*S0 = 200
            eps_S = dS  # Use grid spacing

        print(f"  Using eps_S = {eps_S:.4f} (dS ≈ {200.0/self.M:.4f})")

        # Solve at three S0 points: S0-ε, S0, S0+ε
        price_minus, vega_minus = self._solve_at_S0(S0 - eps_S, sigma)
        price_center, vega_center = self._solve_at_S0(S0, sigma)
        price_plus, vega_plus = self._solve_at_S0(S0 + eps_S, sigma)

        # Greeks via finite difference
        delta = (price_plus - price_minus) / (2 * eps_S)
        gamma = (price_plus - 2*price_center + price_minus) / (eps_S ** 2)

        # Vega from center point
        vega = vega_center

        # Vanna: ∂²V/∂S∂σ ≈ ∂Vega/∂S
        vanna = (vega_plus - vega_minus) / (2 * eps_S)

        # Volga: Would need additional sigma perturbations
        # For now, use finite difference on vega
        eps_sigma = sigma * 0.01
        _, vega_sigma_plus = self._solve_at_S0(S0, sigma + eps_sigma)
        _, vega_sigma_minus = self._solve_at_S0(S0, sigma - eps_sigma)
        volga = (vega_sigma_plus - vega_sigma_minus) / (2 * eps_sigma)

        t_elapsed = (time.perf_counter() - t_start) * 1000

        return {
            'method': 'perturbation_aad',
            'price': price_center,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'vanna': vanna,
            'volga': volga,
            'time_ms': t_elapsed,
            'n_pde_solves': 5,  # 3 for Delta/Gamma + 2 for Volga
            'eps_S': eps_S
        }


class GreeksMethodFiniteDiff:
    """Traditional Finite Difference Bumping"""

    def __init__(self, M: int = 51, N: int = 50):
        self.M = M
        self.N = N

    def _solve_pde_simple(self, S0: float, K: float, T: float, r: float, sigma: float) -> float:
        """Simple PDE solve without AAD"""
        solver = CapriottiCNAAD(M=self.M, N=self.N)
        solver.S0 = S0
        solver.K = K
        solver.T = T
        solver.r = r

        # Rebuild grid for new S0
        solver.Smax = 2.0 * S0
        solver.S_grid = np.linspace(0, solver.Smax, self.M)
        solver.dS = solver.S_grid[1] - solver.S_grid[0]
        solver.S0_index = self.M // 2

        # Simple PDE solve with constant sigma (no AAD)
        V = [max(S - K, 0.0) for S in solver.S_grid[1:-1]]

        for m in range(solver.N - 1, -1, -1):
            t_m = m * solver.dt
            t_m1 = (m + 1) * solver.dt

            V_next = V.copy()
            n = len(V)

            # Build tridiagonal system (simplified, constant sigma)
            a = np.zeros(n)
            b = np.zeros(n)
            c = np.zeros(n)
            d = np.zeros(n)

            for j in range(1, self.M - 1):
                i = j - 1
                S_j = solver.S_grid[j]

                alpha = 0.5 * sigma**2 * S_j**2 / (solver.dS**2)
                beta = r * S_j / (2 * solver.dS)
                gamma_coef = -r

                if i < n:
                    a[i] = -0.5 * solver.dt * (alpha - beta) if i > 0 else 0
                    b[i] = 1 - 0.5 * solver.dt * (2*alpha + gamma_coef)
                    c[i] = -0.5 * solver.dt * (alpha + beta) if i < n-1 else 0

                    # RHS
                    rhs = V_next[i]
                    if i > 0:
                        rhs += 0.5 * solver.dt * (alpha - beta) * V_next[i-1]
                    if i < n-1:
                        rhs += 0.5 * solver.dt * (alpha + beta) * V_next[i+1]
                    rhs += 0.5 * solver.dt * (2*alpha + gamma_coef) * V_next[i]

                    # Boundary conditions
                    V0_m, VSmax_m = solver._boundary_conditions(t_m)
                    V0_m1, VSmax_m1 = solver._boundary_conditions(t_m1)

                    if i == 0:
                        rhs += 0.5 * solver.dt * (alpha - beta) * (V0_m1 - V0_m)
                    if i == n-1:
                        rhs += 0.5 * solver.dt * (alpha + beta) * (VSmax_m1 - VSmax_m)

                    d[i] = rhs

            # Solve tridiagonal system (Thomas algorithm)
            V = self._solve_tridiagonal(a, b, c, d)

        # Interpolate at S0
        i = solver.S0_index - 1
        if i >= 0 and i < len(V):
            return V[i]
        return V[len(V)//2]

    def _solve_tridiagonal(self, a, b, c, d):
        """Thomas algorithm for tridiagonal system"""
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        x = np.zeros(n)

        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]

        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            if i < n-1:
                c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom

        x[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]

        return x

    def compute_greeks(self, S0: float, K: float, T: float, r: float,
                      sigma: float, eps_S: float = 0.5, eps_sigma: float = 0.002) -> Dict:
        """Compute Greeks via finite difference bumping"""
        t_start = time.perf_counter()

        # Price at various points
        V_00 = self._solve_pde_simple(S0, K, T, r, sigma)
        V_p0 = self._solve_pde_simple(S0 + eps_S, K, T, r, sigma)
        V_m0 = self._solve_pde_simple(S0 - eps_S, K, T, r, sigma)
        V_0p = self._solve_pde_simple(S0, K, T, r, sigma + eps_sigma)
        V_0m = self._solve_pde_simple(S0, K, T, r, sigma - eps_sigma)

        # Greeks
        delta = (V_p0 - V_m0) / (2 * eps_S)
        gamma = (V_p0 - 2*V_00 + V_m0) / (eps_S ** 2)
        vega = (V_0p - V_0m) / (2 * eps_sigma)

        # Cross derivatives
        V_pp = self._solve_pde_simple(S0 + eps_S, K, T, r, sigma + eps_sigma)
        V_pm = self._solve_pde_simple(S0 + eps_S, K, T, r, sigma - eps_sigma)
        V_mp = self._solve_pde_simple(S0 - eps_S, K, T, r, sigma + eps_sigma)
        V_mm = self._solve_pde_simple(S0 - eps_S, K, T, r, sigma - eps_sigma)

        vanna = (V_pp - V_pm - V_mp + V_mm) / (4 * eps_S * eps_sigma)

        # Volga
        volga = (V_0p - 2*V_00 + V_0m) / (eps_sigma ** 2)

        t_elapsed = (time.perf_counter() - t_start) * 1000

        return {
            'method': 'finite_difference',
            'price': V_00,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'vanna': vanna,
            'volga': volga,
            'time_ms': t_elapsed,
            'n_pde_solves': 9
        }


class GreeksComparisonFramework:
    """
    Complete comparison framework for all Greeks computation methods
    """

    def __init__(self, M: int = 51, N: int = 50):
        self.M = M
        self.N = N

        # Initialize all methods
        self.method_a = GreeksMethodA(M, N)
        self.method_fd = GreeksMethodFiniteDiff(M, N)

    def compute_analytical(self, S0: float, K: float, T: float, r: float, sigma: float) -> Dict:
        """Compute analytical Black-Scholes Greeks"""
        price, delta, gamma, vega = black_scholes_analytical(S0, K, T, r, sigma)

        # Vanna and Volga
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        vanna = -norm.pdf(d1) * d2 / sigma
        volga = vega * d1 * d2 / sigma

        return {
            'method': 'analytical',
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'vanna': vanna,
            'volga': volga,
            'time_ms': 0.0
        }

    def compare_all(self, S0: float = 100.0, K: float = 100.0, T: float = 1.0,
                    r: float = 0.05, sigma: float = 0.2) -> Dict:
        """
        Run all methods and generate comprehensive comparison
        """
        print("=" * 100)
        print("GREEKS COMPUTATION METHODS COMPARISON")
        print("=" * 100)
        print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
        print(f"Grid: M={self.M}, N={self.N}")

        results = {}

        # 1. Analytical
        print("\n[1/3] Computing Analytical Greeks...")
        results['analytical'] = self.compute_analytical(S0, K, T, r, sigma)

        # 2. Method A (Perturbation + AAD)
        print("[2/3] Computing Method A (Perturbation + AAD)...")
        results['method_a'] = self.method_a.compute_greeks(S0, K, T, r, sigma)

        # 3. Finite Difference
        print("[3/3] Computing Finite Difference...")
        results['method_fd'] = self.method_fd.compute_greeks(S0, K, T, r, sigma)

        # Print comparison table
        self._print_comparison_table(results)

        return results

    def _print_comparison_table(self, results: Dict):
        """Print detailed comparison table"""
        analytical = results['analytical']

        print("\n" + "=" * 120)
        print(f"{'Method':<25} | {'Price':>12} | {'Delta':>12} | {'Gamma':>12} | "
              f"{'Vega':>12} | {'Vanna':>12} | {'Volga':>12} | {'Time(ms)':>10}")
        print("=" * 120)

        # Analytical
        self._print_row("Analytical (BS)", analytical, None)
        print("-" * 120)

        # Other methods with errors
        for key in ['method_a', 'method_fd']:
            if key in results:
                self._print_row(results[key]['method'], results[key], analytical)

        print("=" * 120)

    def _print_row(self, name: str, greeks: Dict, analytical: Dict = None):
        """Print a single row in comparison table"""
        print(f"{name:<25} | "
              f"{greeks['price']:12.6f} | {greeks['delta']:12.6f} | "
              f"{greeks['gamma']:12.6f} | {greeks['vega']:12.6f} | "
              f"{greeks['vanna']:12.6f} | {greeks['volga']:12.6f} | "
              f"{greeks['time_ms']:10.2f}")

        if analytical is not None:
            # Error row
            price_err = abs(greeks['price'] - analytical['price']) / analytical['price']
            delta_err = abs(greeks['delta'] - analytical['delta']) / analytical['delta']
            gamma_err = abs(greeks['gamma'] - analytical['gamma']) / analytical['gamma']
            vega_err = abs(greeks['vega'] - analytical['vega']) / analytical['vega']
            vanna_err = abs(greeks['vanna'] - analytical['vanna']) / abs(analytical['vanna'])
            volga_err = abs(greeks['volga'] - analytical['volga']) / analytical['volga']

            print(f"{'  → Relative Error':<25} | "
                  f"{price_err:11.2%} | {delta_err:11.2%} | "
                  f"{gamma_err:11.2%} | {vega_err:11.2%} | "
                  f"{vanna_err:11.2%} | {volga_err:11.2%}")


def main():
    """Main test execution"""
    # Test with different grid sizes
    grid_sizes = [(51, 50), (101, 100)]

    for M, N in grid_sizes:
        print("\n" + "#" * 100)
        print(f"# Grid Size: M={M}, N={N}")
        print("#" * 100)

        framework = GreeksComparisonFramework(M=M, N=N)
        results = framework.compare_all()

        print("\n")


if __name__ == "__main__":
    main()
