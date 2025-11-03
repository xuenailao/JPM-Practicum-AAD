"""
Method 1: Double Bumping (Bumping2)

Pure finite difference method using two levels of bumping.
Uses SimplePDESolver for numerical PDE solution (no AAD).

Formulas:
    Delta = [V(S0+ε) - V(S0-ε)] / (2ε)
    Gamma = [V(S0+ε) - 2*V(S0) + V(S0-ε)] / ε²
    Vega = [V(σ+ε) - V(σ-ε)] / (2ε)
    Volga = [V(σ+ε) - 2*V(σ) + V(σ-ε)] / ε²
    Vanna = [Delta(σ+ε) - Delta(σ-ε)] / (2ε)

PDE Solves: 9 total
"""

import numpy as np
import time
from typing import Dict
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from .base_method import HessianMethodBase


class Bumping2Method(HessianMethodBase):
    """
    Pure finite difference method using numerical PDE solver.
    """

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float,
                 eps_S: float = None, eps_sigma: float = 0.01):
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "Bumping2"
        # Adaptive eps_S: larger epsilon for better numerical stability
        # Default 0.5% of S0 (minimum 2.0 for absolute scale)
        self.eps_S = eps_S if eps_S is not None else max(2.0, 0.005 * S0)
        self.eps_sigma = eps_sigma

    def _solve_pde(self, S0: float, sigma: float) -> float:
        """Solve PDE numerically and return price at S0"""
        from aad_edge_pushing.pde.simple_pde_solver import SimplePDESolver

        solver = SimplePDESolver(S0=S0, K=self.K, T=self.T, r=self.r, sigma=sigma,
                                M=self.M, N_base=self.N)
        price, _ = solver._solve_pde_numerical(S0, sigma)
        return price

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        start_time = time.time()

        S = S0 if S0 is not None else self.S0
        eps_S = self.eps_S
        eps_sigma = self.eps_sigma

        # 1. Base value
        V0 = self._solve_pde(S, sigma)

        # 2. Perturb S0
        V_Sp = self._solve_pde(S + eps_S, sigma)
        V_Sm = self._solve_pde(S - eps_S, sigma)

        # 3. Perturb sigma
        V_sp = self._solve_pde(S, sigma + eps_sigma)
        V_sm = self._solve_pde(S, sigma - eps_sigma)

        # 4. Cross perturbations for Vanna
        V_Sp_sp = self._solve_pde(S + eps_S, sigma + eps_sigma)
        V_Sm_sp = self._solve_pde(S - eps_S, sigma + eps_sigma)
        V_Sp_sm = self._solve_pde(S + eps_S, sigma - eps_sigma)
        V_Sm_sm = self._solve_pde(S - eps_S, sigma - eps_sigma)

        # === Compute Jacobian ===
        delta = (V_Sp - V_Sm) / (2 * eps_S)
        vega = (V_sp - V_sm) / (2 * eps_sigma)

        # === Compute Hessian ===
        gamma = (V_Sp - 2*V0 + V_Sm) / (eps_S**2)
        volga = (V_sp - 2*V0 + V_sm) / (eps_sigma**2)

        delta_sp = (V_Sp_sp - V_Sm_sp) / (2 * eps_S)
        delta_sm = (V_Sp_sm - V_Sm_sm) / (2 * eps_S)
        vanna = (delta_sp - delta_sm) / (2 * eps_sigma)

        jacobian = np.array([delta, vega])
        hessian = np.array([[gamma, vanna], [vanna, volga]])

        time_ms = (time.time() - start_time) * 1000

        return self._format_result(
            price=V0,
            jacobian=jacobian,
            hessian=hessian,
            time_ms=time_ms,
            n_pde_solves=9
        )
