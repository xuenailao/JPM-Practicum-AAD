"""
Test the impact of SymmSparse optimization on Algorithm 4 performance.

Compares:
1. Algo3 (baseline)
2. Algo4 (original with O(n) neighbor scan)
3. Algo4 Optimized (with O(1) neighbor lookup via adjacency list)

Expected results:
- Dense scenarios: All similar
- Sparse scenarios: Algo4-Opt should be 2-5× faster than Algo3/Algo4-Original
"""

import numpy as np
import time
from typing import Callable, List

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.algo3.algo3_block import algo3_block
from aad_edge_pushing.algo3.algo4_edge_pushing import algo4_edge_pushing
from aad_edge_pushing.algo3.algo4_optimized import algo4_optimized


def reset_tape():
    """Reset global tape."""
    global_tape.reset()


def create_ad_inputs(values: np.ndarray) -> List[ADVar]:
    """Create AD input variables."""
    return [ADVar(float(v)) for v in values]


# ============================================================================
# Test Functions
# ============================================================================

def sparse_sum_squares(inputs: List[ADVar]) -> ADVar:
    """
    Sparse sum of squares: f = Σxᵢ²
    Hessian: Diagonal (sparsity = 1 - n/n² ≈ 99% for large n)
    """
    return sum(x * x for x in inputs)


def sparse_block_diagonal(inputs: List[ADVar], block_size: int = 10) -> ADVar:
    """
    Block-diagonal structure: f = Σ_blocks Σ_{i,j in block} xᵢxⱼ
    Hessian: Block-diagonal with blocks of size block_size
    """
    n = len(inputs)
    n_blocks = n // block_size

    result = inputs[0] * 0.0  # Initialize

    for block_idx in range(n_blocks):
        start = block_idx * block_size
        end = min(start + block_size, n)

        # Within-block interactions
        for i in range(start, end):
            for j in range(i, end):
                if i == j:
                    result = result + inputs[i] * inputs[i]
                else:
                    result = result + inputs[i] * inputs[j]

    return result


def dense_full_interactions(inputs: List[ADVar]) -> ADVar:
    """
    Dense: f = Σᵢxᵢ² + Σᵢ<ⱼxᵢxⱼ
    Hessian: Fully dense
    """
    n = len(inputs)
    result = sum(x * x for x in inputs)
    for i in range(n):
        for j in range(i + 1, n):
            result = result + inputs[i] * inputs[j]
    return result


# ============================================================================
# Benchmarking
# ============================================================================

def benchmark_three_algorithms(
    func: Callable,
    input_values: np.ndarray,
    func_name: str,
    num_trials: int = 3
):
    """
    Benchmark all three algorithms on the same function.
    """
    n = len(input_values)

    times_3, times_4, times_4opt = [], [], []

    for trial in range(num_trials):
        # Algo3
        reset_tape()
        inputs_3 = create_ad_inputs(input_values)
        output_3 = func(inputs_3)
        t_start = time.perf_counter()
        H3 = algo3_block(output_3, inputs_3)
        t_algo3 = time.perf_counter() - t_start
        times_3.append(t_algo3 * 1000)

        # Algo4 (original)
        reset_tape()
        inputs_4 = create_ad_inputs(input_values)
        output_4 = func(inputs_4)
        t_start = time.perf_counter()
        H4 = algo4_edge_pushing(output_4, inputs_4)
        t_algo4 = time.perf_counter() - t_start
        times_4.append(t_algo4 * 1000)

        # Algo4 Optimized
        reset_tape()
        inputs_4opt = create_ad_inputs(input_values)
        output_4opt = func(inputs_4opt)
        t_start = time.perf_counter()
        H4opt = algo4_optimized(output_4opt, inputs_4opt)
        t_algo4opt = time.perf_counter() - t_start
        times_4opt.append(t_algo4opt * 1000)

    # Statistics
    avg_3 = np.mean(times_3)
    avg_4 = np.mean(times_4)
    avg_4opt = np.mean(times_4opt)

    # Compute sparsity
    nnz = np.count_nonzero(np.abs(H3) > 1e-10)
    sparsity = 100.0 * (1.0 - nnz / (n * n))

    # Verify correctness
    max_diff_4 = np.max(np.abs(H3 - H4))
    max_diff_4opt = np.max(np.abs(H3 - H4opt))

    # Display results
    print(f"\n{'='*80}")
    print(f"TEST: {func_name} (n={n})")
    print(f"{'='*80}")
    print(f"Sparsity: {sparsity:.1f}% ({nnz}/{n*n} non-zero)")
    print(f"\nCorrectness:")
    print(f"  Algo4 vs Algo3 max diff: {max_diff_4:.2e}")
    print(f"  Algo4-Opt vs Algo3 max diff: {max_diff_4opt:.2e}")
    print(f"\nPerformance:")
    print(f"  Algo3:       {avg_3:8.3f} ms  (baseline)")
    print(f"  Algo4:       {avg_4:8.3f} ms  ({avg_3/avg_4:.2f}× vs Algo3)")
    print(f"  Algo4-Opt:   {avg_4opt:8.3f} ms  ({avg_3/avg_4opt:.2f}× vs Algo3)")
    print(f"\n🚀 Optimization impact: {avg_4/avg_4opt:.2f}× speedup (Algo4-Opt vs Algo4)")

    return {
        'n': n,
        'sparsity': sparsity,
        'algo3_ms': avg_3,
        'algo4_ms': avg_4,
        'algo4opt_ms': avg_4opt,
        'speedup_vs_algo3': avg_3 / avg_4opt,
        'opt_impact': avg_4 / avg_4opt
    }


def main():
    print("="*80)
    print("ALGORITHM 4 OPTIMIZATION IMPACT ANALYSIS")
    print("="*80)
    print("\nComparing:")
    print("  1. Algo3 (Block Form - baseline)")
    print("  2. Algo4 (Edge-Pushing with O(n) neighbor scan)")
    print("  3. Algo4-Opt (Edge-Pushing with O(1) neighbor lookup)")
    print("\nHypothesis: Algo4-Opt should eliminate the O(n) scan bottleneck")

    results = []

    # ========================================================================
    # Test 1: Extremely Sparse (Diagonal Hessian)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 1: EXTREMELY SPARSE (DIAGONAL HESSIAN)")
    print("="*80)

    for n in [50, 100, 200]:
        inputs = np.random.randn(n) * 0.5
        result = benchmark_three_algorithms(
            sparse_sum_squares,
            inputs,
            f"Sparse Sum of Squares (n={n})",
            num_trials=5
        )
        results.append(result)

    # ========================================================================
    # Test 2: Moderately Sparse (Block-Diagonal)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 2: MODERATELY SPARSE (BLOCK-DIAGONAL)")
    print("="*80)

    for n, block_size in [(100, 10), (200, 10)]:
        inputs = np.random.randn(n) * 0.5
        result = benchmark_three_algorithms(
            lambda inp: sparse_block_diagonal(inp, block_size),
            inputs,
            f"Block-Diagonal (n={n}, block={block_size})",
            num_trials=3
        )
        results.append(result)

    # ========================================================================
    # Test 3: Dense (Full Interactions)
    # ========================================================================
    print("\n" + "="*80)
    print("SCENARIO 3: DENSE (FULL INTERACTIONS)")
    print("="*80)

    for n in [20, 30]:
        inputs = np.random.randn(n) * 0.5 + 1.0
        result = benchmark_three_algorithms(
            dense_full_interactions,
            inputs,
            f"Dense Full Interactions (n={n})",
            num_trials=3
        )
        results.append(result)

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY: OPTIMIZATION IMPACT")
    print("="*80)
    print(f"{'Test':<40} {'Sparsity':>10} {'Speedup':>10} {'Opt Impact':>12}")
    print("-"*80)

    for i, res in enumerate(results, 1):
        print(f"Test {i:<37} {res['sparsity']:9.1f}% {res['speedup_vs_algo3']:9.2f}× {res['opt_impact']:11.2f}×")

    # Analysis
    sparse_tests = [r for r in results if r['sparsity'] > 85]
    if sparse_tests:
        avg_opt_impact_sparse = np.mean([r['opt_impact'] for r in sparse_tests])
        avg_speedup_sparse = np.mean([r['speedup_vs_algo3'] for r in sparse_tests])

        print(f"\n{'='*80}")
        print("KEY INSIGHTS")
        print(f"{'='*80}")
        print(f"For sparse scenarios (>85% sparsity):")
        print(f"  • Average optimization impact: {avg_opt_impact_sparse:.2f}×")
        print(f"  • Average speedup vs Algo3: {avg_speedup_sparse:.2f}×")
        print(f"\nConclusion:")
        if avg_opt_impact_sparse > 2.0:
            print(f"  ✅ Adjacency list optimization is HIGHLY EFFECTIVE ({avg_opt_impact_sparse:.1f}× faster)")
            print(f"     Eliminated O(n) scan bottleneck successfully!")
        elif avg_opt_impact_sparse > 1.5:
            print(f"  ✅ Optimization shows clear benefit ({avg_opt_impact_sparse:.1f}× faster)")
        elif avg_opt_impact_sparse > 1.2:
            print(f"  ⚠️  Modest improvement ({avg_opt_impact_sparse:.1f}× faster)")
        else:
            print(f"  ❌ Optimization ineffective (need further investigation)")

        if avg_speedup_sparse > 2.0:
            print(f"  ✅ Algo4-Opt significantly outperforms Algo3 for sparse problems")
        elif avg_speedup_sparse > 1.2:
            print(f"  ✅ Algo4-Opt shows advantage over Algo3")
        else:
            print(f"  ⚠️  Similar performance to Algo3")


if __name__ == "__main__":
    main()
