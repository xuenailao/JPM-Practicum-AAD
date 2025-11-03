"""
Method 3: Double AAD (FIXED)
Uses fixed version of original_pde_aad_hessian with grid-based Gamma
"""

import numpy as np
from typing import Dict
from .pde_aad_edgepushing import BS_PDE_AAD


class DoubleAADFixed:
    """Fixed Double AAD using grid-based finite difference"""

    def __init__(self, M: int, N: int):
        self.M = M
        self.N = N

    def compute_greeks(self, S0: float, K: float, T: float, r: float, sigma: float,
                      verbose: bool = False, eps_r: float = 0.0001) -> Dict:
        """Compute Greeks via Double AAD with fixed Gamma"""

        solver = BS_PDE_AAD(
            S0=S0,
            K=K,
            T=T,
            r=r,
            M=self.M,
            N_base=self.N
        )

        result = solver.solve_pde_with_aad(
            S0_val=S0,
            sigma_val=sigma,
            compute_hessian=True,
            verbose=verbose
        )

        solver_r_plus = BS_PDE_AAD(
            S0=S0, K=K, T=T, r=r+eps_r, M=self.M, N_base=self.N
        )
        result_r_plus = solver_r_plus.solve_pde_with_aad(
            S0_val=S0, sigma_val=sigma, compute_hessian=False, verbose=False
        )

        solver_r_minus = BS_PDE_AAD(
            S0=S0, K=K, T=T, r=r-eps_r, M=self.M, N_base=self.N
        )
        result_r_minus = solver_r_minus.solve_pde_with_aad(
            S0_val=S0, sigma_val=sigma, compute_hessian=False, verbose=False
        )

        rho = (result_r_plus['price'] - result_r_minus['price']) / (2.0 * eps_r)

        # Build standardized output
        output = {
            'price': result['price'],
            'jacobian': np.array([result['delta'], result['vega']]),
            'hessian': np.array([
                [result['gamma'], result.get('vanna', 0.0)],
                [result.get('vanna', 0.0), result.get('volga', 0.0)]
            ]),
            'delta': result['delta'],
            'gamma': result['gamma'],
            'vega': result['vega'],
            'rho': rho,
            'vanna': result.get('vanna', 0.0),
            'volga': result.get('volga', 0.0),
            'time_ms': result['time_ms'],
            'pde_solves': 5  # 3 for Hessian + 2 for rho
        }

        return output


def test_fixed_double_aad():
    """Test fixed double AAD"""
    from scipy.stats import norm

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    phi_d1 = norm.pdf(d1)
    gamma_analytical = phi_d1 / (S0 * sigma * sqrt_T)

    print(f"Analytical Gamma: {gamma_analytical:.10f}")

    method = DoubleAADFixed(M=50, N=150)
    result = method.compute_greeks(S0, K, T, r, sigma)

    print(f"Fixed Double AAD Gamma: {result['gamma']:.10f}")
    print(f"Error: {abs(result['gamma'] - gamma_analytical) / gamma_analytical * 100:.2f}%")


if __name__ == "__main__":
    test_fixed_double_aad()
