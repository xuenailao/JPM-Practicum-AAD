"""
Method 4: Edge-Pushing (Algorithm 4)

State-of-the-art Hessian computation using Edge-Pushing algorithm.
Uses BS_PDE_AAD with compute_hessian=True to invoke algo4_adjlist.

Key Features:
- Natural Cubic Spline interpolation (C² continuous)
- Single backward pass computes full Hessian
- Adjacency list optimization O(degree)
- S0 as ADVar enables true second-order AD

PDE Solves: 1 (single computational tape)
"""

import numpy as np
import time
from typing import Dict
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from .base_method import HessianMethodBase


class EdgePushingMethod(HessianMethodBase):
    """
    Edge-Pushing method with Natural Cubic Spline.
    Optimal method for Hessian computation.
    """

    def __init__(self, M: int, N: int, S0: float, K: float, T: float, r: float):
        super().__init__(M, N, S0, K, T, r)
        self.method_name = "Edge-Pushing"

    def compute_hessian(self, S0: float, sigma: float) -> Dict:
        start_time = time.time()

        S = S0 if S0 is not None else self.S0
        
        from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

        # Use Edge-Pushing for Hessian
        solver = BS_PDE_AAD(S0=S, K=self.K, T=self.T, r=self.r, sigma=sigma,
                           M=self.M, N_base=self.N)
        
        result = solver.solve_pde_with_aad(
            S0_val=S,
            sigma_val=sigma,
            compute_hessian=True,  # Enable Edge-Pushing
            verbose=False
        )

        price = result['price']
        delta = result['delta']
        vega = result['vega']
        gamma = result['gamma']
        vanna = result['vanna']
        volga = result['volga']

        jacobian = np.array([delta, vega])
        hessian = np.array([[gamma, vanna], [vanna, volga]])

        time_ms = (time.time() - start_time) * 1000

        return self._format_result(
            price=price,
            jacobian=jacobian,
            hessian=hessian,
            time_ms=time_ms,
            n_pde_solves=1
        )
