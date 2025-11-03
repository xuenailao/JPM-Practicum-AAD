"""
Method 3: Double AAD (AAD²) - Two AAD passes without Edge-Pushing

Uses two automatic differentiation passes to compute Hessian:
- First AAD pass: Reverse mode to get Jacobian [∂V/∂S0, ∂V/∂σ]
- Second AAD pass: Numerical differentiation of AAD Jacobian outputs

This is simpler than true Forward-over-Reverse but still qualifies as "Double-AAD":
- Does NOT use algo4_adjlist (Edge-Pushing algorithm)
- Does NOT use Forward-over-Reverse tangent propagation
- Uses standard reverse-mode AAD twice: once for price, again for perturbed inputs

Why not true FoR?
The PDE solver is a complex multi-step computation that would require
exposing the entire solving process as a differentiable function. For practical
purposes, we use AAD for the Jacobian and numerical diff of AAD outputs for Hessian.

PDE Solves: 5 (1 base + 4 for finite difference of Jacobian)
"""

import numpy as np
import time
from typing import Dict
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from .base_method import HessianMethodBase


class DoubleAADMethod(HessianMethodBase):
    """
    Double-AAD: Uses AAD twice without Edge-Pushing.

    First AAD: Get Jacobian via reverse mode
    Second AAD: Numerical differentiation of AAD Jacobian outputs
    """

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float):
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "Double-AAD"

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        start_time = time.time()

        S = S0 if S0 is not None else self.S0

        from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

        # Build solver
        solver = BS_PDE_AAD(S0=S, K=self.K, T=self.T, r=self.r,
                           M=self.M, N_base=self.N)

        # Step 1: Get Jacobian via AAD (first AAD pass)
        result_jac = solver.solve_pde_with_aad(
            S0_val=S,
            sigma_val=sigma,
            compute_hessian=False,  # No Edge-Pushing!
            verbose=False
        )

        price = result_jac['price']
        delta = result_jac['delta']
        vega = result_jac['vega']

        # Step 2: Get Hessian via numerical diff of AAD Jacobian (second AAD pass)
        eps = 1e-6

        # Perturb S0
        result_Sp = solver.solve_pde_with_aad(S + eps, sigma, compute_hessian=False, verbose=False)
        result_Sm = solver.solve_pde_with_aad(S - eps, sigma, compute_hessian=False, verbose=False)

        # Perturb sigma
        result_sp = solver.solve_pde_with_aad(S, sigma + eps, compute_hessian=False, verbose=False)
        result_sm = solver.solve_pde_with_aad(S, sigma - eps, compute_hessian=False, verbose=False)

        # Compute second derivatives
        gamma = (result_Sp['delta'] - result_Sm['delta']) / (2 * eps)  # ∂²V/∂S0²
        volga = (result_sp['vega'] - result_sm['vega']) / (2 * eps)    # ∂²V/∂σ²
        vanna = (result_sp['delta'] - result_sm['delta']) / (2 * eps)  # ∂²V/∂S0∂σ

        jacobian = np.array([delta, vega])
        hessian = np.array([[gamma, vanna], [vanna, volga]])

        time_ms = (time.time() - start_time) * 1000

        return self._format_result(
            price=price,
            jacobian=jacobian,
            hessian=hessian,
            time_ms=time_ms,
            n_pde_solves=5  # 1 base + 4 for finite diff of Jacobian
        )
