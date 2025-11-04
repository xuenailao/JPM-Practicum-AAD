"""
Direct performance test of symm_sparse_adjlist Cython vs Python.

Tests the core operations that dominate algo4 runtime.
"""

import time
import sys

def test_symm_sparse_performance():
    """
    Benchmark core symm_sparse_adjlist operations.

    Simulates the workload pattern from algo4_adjlist at M=101, N=200:
    - Matrix size n ~ 101 (number of spatial grid points)
    - Approximately 30M add() calls
    - Approximately 20K get_neighbors() calls
    - Approximately 100M get() calls
    """

    print("="*80)
    print("SYMM_SPARSE_ADJLIST PERFORMANCE TEST")
    print("="*80)
    print()

    # Import and check which version is loaded
    from aad_edge_pushing.edge_pushing.symm_sparse_adjlist import SymmSparseAdjList

    print("Module information:")
    print(f"  Class: {SymmSparseAdjList}")
    print(f"  Module: {SymmSparseAdjList.__module__}")

    if hasattr(SymmSparseAdjList, '__pyx_vtable__'):
        print("  ✓ Cython version loaded")
        version = "Cython"
    else:
        print("  ⚠ Pure Python version loaded")
        version = "Python"

    print()
    print("-"*80)
    print("BENCHMARK: Simulated algo4 workload")
    print("-"*80)

    n = 101  # Grid size matching M=101

    # Scaled-down workload (10% of full to keep test < 60s)
    num_add_ops = 3_000_000  # 10% of 30M
    num_get_neighbor_ops = 2_000  # 10% of 20K
    num_get_ops = 10_000_000  # 10% of 100M

    print(f"  Matrix dimension: {n}×{n}")
    print(f"  Operations:")
    print(f"    add():          {num_add_ops:,}")
    print(f"    get_neighbors(): {num_get_neighbor_ops:,}")
    print(f"    get():          {num_get_ops:,}")
    print()

    # Test 1: add() operations
    print("Test 1: add() operations...")
    W = SymmSparseAdjList(n)

    start = time.perf_counter()
    for k in range(num_add_ops):
        i = k % (n-1)
        j = (k + 1) % n
        val = 1.0 + (k % 100) * 0.01
        W.add(i, j, val)
    elapsed_add = time.perf_counter() - start

    print(f"  Time: {elapsed_add:.3f}s")
    print(f"  Rate: {num_add_ops/elapsed_add:,.0f} ops/sec")
    print(f"  Matrix nnz: {W.nnz()}, sparsity: {W.sparsity():.1f}%")
    print()

    # Test 2: get_neighbors() operations
    print("Test 2: get_neighbors() operations...")

    start = time.perf_counter()
    total_neighbors = 0
    for k in range(num_get_neighbor_ops):
        i = k % n
        neighbors = W.get_neighbors(i)
        total_neighbors += len(neighbors)
    elapsed_neighbors = time.perf_counter() - start

    print(f"  Time: {elapsed_neighbors:.3f}s")
    print(f"  Rate: {num_get_neighbor_ops/elapsed_neighbors:,.0f} ops/sec")
    print(f"  Avg neighbors per node: {total_neighbors/num_get_neighbor_ops:.1f}")
    print()

    # Test 3: get() operations
    print("Test 3: get() operations...")

    start = time.perf_counter()
    total_val = 0.0
    for k in range(num_get_ops):
        i = k % n
        j = (k + 7) % n
        total_val += W.get(i, j)
    elapsed_get = time.perf_counter() - start

    print(f"  Time: {elapsed_get:.3f}s")
    print(f"  Rate: {num_get_ops/elapsed_get:,.0f} ops/sec")
    print()

    # Total time
    total_time = elapsed_add + elapsed_neighbors + elapsed_get

    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Version: {version}")
    print(f"Total time (10% workload): {total_time:.3f}s")
    print(f"Estimated full workload:   {total_time * 10:.1f}s")
    print()

    # Compare with expected times
    if version == "Cython":
        expected_full = 47.0  # 三今's measurement
        print(f"Expected (Cython full):    {expected_full:.1f}s")
        print(f"Ratio to expected:         {(total_time * 10) / expected_full:.2f}×")

        if total_time * 10 < expected_full * 1.2:
            print("✓ Performance within expected range!")
        else:
            print("⚠ Performance below expectations")
    else:
        expected_full = 575.0  # 三今's Python baseline
        print(f"Expected (Python full):    {expected_full:.1f}s")
        print(f"Ratio to expected:         {(total_time * 10) / expected_full:.2f}×")

    print("="*80)

    return total_time

if __name__ == "__main__":
    test_symm_sparse_performance()
