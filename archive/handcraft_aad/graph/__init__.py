"""
Adjacency graph structures for sparse Hessian computation.

This module contains graph-based structures that exploit sparsity:
- LocalVolAdjacency: Adjacency graph for local volatility parameters
"""

from .adjacency_graph import LocalVolAdjacency

__all__ = ['LocalVolAdjacency']
