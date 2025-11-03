"""
Core PDE solvers.

This module contains the fundamental PDE solving infrastructure:
- LocalVolSolver: Crank-Nicolson solver with local volatility
- LocalVolAdjoint: Discrete adjoint method for first-order Greeks
"""

from .local_vol_solver import LocalVolSolver, LocalVolAdjoint

__all__ = ['LocalVolSolver', 'LocalVolAdjoint']
