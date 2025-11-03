"""
Methods package for Hessian computation.

Provides 5 independent methods for computing option price Hessian:
1. Bumping2: Pure finite difference (two bumps)
2. AAD+Bumping: Hybrid method (AAD for Jacobian, bumping for Hessian)
3. Double-AAD: Nested AAD (via Edge-Pushing)
4. Edge-Pushing: Single-tape Hessian computation
5. BSM-Analytical: Black-Scholes-Merton analytical formulas (baseline)
"""

from .base_method import HessianMethodBase
from .bumping2 import Bumping2Method
from .aad_bumping import AADBumpingMethod
from .double_aad import DoubleAADMethod
from .edge_pushing import EdgePushingMethod
from .bsm_analytical import BSMAnalyticalMethod

__all__ = [
    'HessianMethodBase',
    'Bumping2Method',
    'AADBumpingMethod',
    'DoubleAADMethod',
    'EdgePushingMethod',
    'BSMAnalyticalMethod'
]
