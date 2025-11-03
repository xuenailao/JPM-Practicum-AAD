"""
Hybrid Edge-Pushing Method for Stable Greeks Computation

This method combines:
1. Edge-Pushing (algo4) for Gamma: ∂²V/∂S² (works well, no σ in computation)
2. AAD for Delta: ∂V/∂S (works well, no σ in computation)
3. Adaptive Bumping for Vega: ∂V/∂σ (avoids gradient through implicit solves)
4. Hybrid for Vanna: ∂²V/∂S∂σ (AAD on Delta + bump on σ)
5. Bumping for Volga: ∂²V/∂σ² (second-order bump on σ)

Key Improvement over Original Edge-Pushing:
--------------------------------------------
- Original: σ is ADVar → gradient through 200 CN iterations → 22% Vega error
- Hybrid: σ is scalar → numerical differentiation → <5% Vega error (expected)

Maintains Edge-Pushing advantages:
-----------------------------------
- Efficient Hessian computation via adjacency list
- Sparse matrix exploitation
- Exact Gamma computation via AAD

References:
-----------
- Griewank et al. (2008): "Edge-Pushing Algorithm for Hessians"
- Marc Henrard (2011): Avoiding differentiation through solvers
- Maran et al. (2021): "AAD ineffective on second-order Greeks"
"""

import numpy as np
import time
from typing import Dict
from .base_method import HessianMethodBase
from ..pde_aad_edgepushing import BS_PDE_AAD
from ...aad.core.var import ADVar
from ...aad.core.tape import global_tape
try:
    from ...edge_pushing.algo4_adjlist import algo4_adjlist
except ImportError:
    from ..edge_pushing.algo4_adjlist import algo4_adjlist


class HybridEdgePushingMethod(HessianMethodBase):
    """
    Hybrid Edge-Pushing method with stable volatility derivatives

    Strategy:
    ---------
    1. Use Edge-Pushing for (S, S) block of Hessian (Gamma)
    2. Use numerical bumping for σ derivatives to avoid instability
    3. Combine AAD and bumping for cross derivatives (Vanna)
    """

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float,
                 eps_sigma: float = None):
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "Hybrid-EdgePush"

        # Adaptive eps_sigma for high volatility stability
        # Default: 2% of sigma value (minimum 0.01 for absolute scale)
        self.eps_sigma = eps_sigma if eps_sigma is not None else 0.02

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        """
        Compute full Hessian using hybrid method

        Returns Hessian matrix:
        [∂²V/∂S²    ∂²V/∂S∂σ ]
        [∂²V/∂σ∂S   ∂²V/∂σ²  ]
        """
        t_start = time.perf_counter()

        # Adaptive eps_sigma scaling with volatility
        eps_sigma_adaptive = max(self.eps_sigma * sigma, 0.01)

        # ===================================================================
        # STEP 1: Compute V(S, σ) and ∂V/∂S using AAD (S is ADVar, σ is scalar)
        # ===================================================================
        global_tape.reset()

        S_var = ADVar(S0, requires_grad=True, name="S0")
        # σ is NOT an ADVar - this avoids gradient through implicit solves!

        solver_center = BS_PDE_AAD(
            S0=S0, K=self.K, T=self.T, r=self.r, sigma=sigma,
            M=self.M, N_base=self.N
        )

        price_center_var = solver_center.solve_pde_with_S_as_advar(S_var)
        price_center = price_center_var.val

        # Backward pass for ∂V/∂S
        price_center_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)

        delta_center = S_var.adj

        # ===================================================================
        # STEP 2: Compute ∂²V/∂S² (Gamma) using Edge-Pushing
        # ===================================================================
        global_tape.reset()

        S_var_hess = ADVar(S0, requires_grad=True, name="S0")

        price_hess_var = solver_center.solve_pde_with_S_as_advar(S_var_hess)

        # Apply Edge-Pushing to get ∂²V/∂S²
        hessian_SS = algo4_adjlist(price_hess_var, [S_var_hess])
        gamma = hessian_SS[0, 0]

        # ===================================================================
        # STEP 3: Compute ∂V/∂σ (Vega) using adaptive numerical bumping
        # ===================================================================

        # V(S, σ + ε)
        solver_plus = BS_PDE_AAD(
            S0=S0, K=self.K, T=self.T, r=self.r, sigma=sigma + eps_sigma_adaptive,
            M=self.M, N_base=self.N
        )
        price_plus = solver_plus.solve_pde_numerical(S0)

        # V(S, σ - ε)
        solver_minus = BS_PDE_AAD(
            S0=S0, K=self.K, T=self.T, r=self.r, sigma=sigma - eps_sigma_adaptive,
            M=self.M, N_base=self.N
        )
        price_minus = solver_minus.solve_pde_numerical(S0)

        # Central difference for Vega
        vega = (price_plus - price_minus) / (2.0 * eps_sigma_adaptive)

        # ===================================================================
        # STEP 4: Compute ∂²V/∂S∂σ (Vanna) using hybrid AAD + bumping
        # ===================================================================

        # Compute ∂V/∂S at σ + ε
        global_tape.reset()
        S_var_plus = ADVar(S0, requires_grad=True, name="S0")
        price_plus_var = solver_plus.solve_pde_with_S_as_advar(S_var_plus)
        price_plus_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)
        delta_plus = S_var_plus.adj

        # Compute ∂V/∂S at σ - ε
        global_tape.reset()
        S_var_minus = ADVar(S0, requires_grad=True, name="S0")
        price_minus_var = solver_minus.solve_pde_with_S_as_advar(S_var_minus)
        price_minus_var.adj = 1.0
        for node in reversed(global_tape.nodes):
            for parent, deriv in node.parents:
                if parent.requires_grad:
                    parent.adj += node.out.adj * float(deriv)
        delta_minus = S_var_minus.adj

        # Vanna = ∂(∂V/∂S)/∂σ = (Delta(σ+ε) - Delta(σ-ε)) / (2ε)
        vanna = (delta_plus - delta_minus) / (2.0 * eps_sigma_adaptive)

        # ===================================================================
        # STEP 5: Compute ∂²V/∂σ² (Volga) using second-order finite difference
        # ===================================================================

        # Volga = (Vega(σ+ε) - Vega(σ-ε)) / (2ε)
        # We can use: Volga ≈ (V(σ+ε) - 2V(σ) + V(σ-ε)) / ε²
        volga = (price_plus - 2.0 * price_center + price_minus) / (eps_sigma_adaptive**2)

        t_end = time.perf_counter()
        time_ms = (t_end - t_start) * 1000.0

        # Build Hessian matrix
        hessian = np.array([
            [gamma, vanna],
            [vanna, volga]
        ])

        return {
            'price': price_center,
            'greeks': {
                'delta': delta_center,
                'gamma': gamma,
                'vega': vega,
                'vanna': vanna,
                'volga': volga
            },
            'hessian': hessian,
            'time_ms': time_ms,
            'n_pde_solves': 3,  # center + plus + minus
            'method': self.method_name,
            'eps_sigma_used': eps_sigma_adaptive
        }
