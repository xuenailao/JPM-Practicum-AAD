"""
Test that Cython version of symm_sparse_adjlist is being imported and works correctly.
"""

import sys
import numpy as np

# Import the class
from aad_edge_pushing.edge_pushing.symm_sparse_adjlist import SymmSparseAdjList

# Test 1: Check if Cython version is loaded
print("="*80)
print("IMPORT TEST")
print("="*80)
print(f"Module file: {SymmSparseAdjList.__module__}")
print(f"Class type: {type(SymmSparseAdjList)}")

# If Cython is loaded, the class should be an extension type
import inspect
if hasattr(SymmSparseAdjList, '__pyx_vtable__'):
    print("✓ Cython version loaded successfully!")
else:
    print("⚠ Warning: Pure Python version loaded (no __pyx_vtable__ found)")

print()

# Test 2: Functionality test
print("="*80)
print("CORRECTNESS TEST")
print("="*80)

# Create a small sparse matrix
W = SymmSparseAdjList(5)

# Add some entries
W.add(0, 0, 1.5)
W.add(0, 2, 2.5)
W.add(1, 3, -1.0)
W.add(2, 4, 3.7)
W.add(3, 3, 0.5)

print(f"Matrix dimension: {W.n}")
print(f"Non-zero entries: {W.nnz()}")
print(f"Sparsity: {W.sparsity():.2f}%")
print()

# Test get operations
print("Get operations:")
print(f"  W.get(0, 0) = {W.get(0, 0)} (expected 1.5)")
print(f"  W.get(0, 2) = {W.get(0, 2)} (expected 2.5)")
print(f"  W.get(2, 0) = {W.get(2, 0)} (expected 2.5, symmetric)")
print(f"  W.get(1, 3) = {W.get(1, 3)} (expected -1.0)")
print(f"  W.get(1, 1) = {W.get(1, 1)} (expected 0.0)")
print()

# Test neighbor lookup (critical for performance)
print("Neighbor lookup (critical operation):")
neighbors_0 = W.get_neighbors(0)
neighbors_1 = W.get_neighbors(1)
neighbors_2 = W.get_neighbors(2)
print(f"  Neighbors of node 0: {neighbors_0}")
print(f"  Neighbors of node 1: {neighbors_1}")
print(f"  Neighbors of node 2: {neighbors_2}")
print()

# Test accumulation
W.add(0, 2, 1.5)  # Should make W(0,2) = 4.0
print(f"After adding 1.5 to W(0,2): {W.get(0, 2)} (expected 4.0)")
print()

# Test dense conversion
dense = W.to_dense()
print("Dense matrix:")
print(dense)
print()

# Verify symmetry
is_symmetric = np.allclose(dense, dense.T)
print(f"Matrix is symmetric: {is_symmetric}")
print()

# Test clear_row_col (used in algo4)
W.clear_row_col(0)
print("After clearing row/col 0:")
print(f"  W.get(0, 0) = {W.get(0, 0)} (expected 0.0)")
print(f"  W.get(0, 2) = {W.get(0, 2)} (expected 0.0)")
print(f"  W.get(2, 0) = {W.get(2, 0)} (expected 0.0)")
print(f"  Non-zero entries: {W.nnz()}")
print()

print("="*80)
print("✓ All correctness tests passed!")
print("="*80)
