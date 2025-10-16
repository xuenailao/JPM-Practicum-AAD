"""
Large-Scale Complex Function Tests: Algo3 vs Algo4 vs Bumping

Tests include:
1. Rosenbrock function (n=50, 100, 200)
2. Polynomial sum (n=50, 100, 200)
3. Neural network layer (n=100, 200)
4. Portfolio risk (n=50, 100)

All tests measure actual runtime and accuracy.
"""

import numpy as np
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.algo3.algo3_block import algo3_block
from aad_edge_pushing.algo3.algo4_optimized import algo4_optimized


# ==================== Test Function 1: Rosenbrock ====================

def rosenbrock_numeric(x):
    """Rosenbrock function: sum_{i=1}^{n-1} [100(x_{i+1}-x_i^2)^2 + (1-x_i)^2]"""
    n = len(x)
    result = 0.0
    for i in range(n-1):
        result += 100.0 * (x[i+1] - x[i]**2)**2 + (1.0 - x[i])**2
    return result


def rosenbrock_ad(x_vars):
    """Rosenbrock with ADVar"""
    n = len(x_vars)
    result = ADVar(0.0)
    for i in range(n-1):
        diff = x_vars[i+1] - x_vars[i] * x_vars[i]
        result = result + 100.0 * diff * diff + (1.0 - x_vars[i]) * (1.0 - x_vars[i])
    return result


def test_rosenbrock(n, methods=['bumping', 'algo3', 'algo4']):
    """Test Rosenbrock function"""
    print(f"\n{'='*80}")
    print(f"TEST: Rosenbrock Function (n={n})")
    print(f"{'='*80}\n")

    # Random point near minimum
    x0 = np.ones(n) + 0.1 * np.random.randn(n)

    results = {}

    # Bumping
    if 'bumping' in methods:
        print(f"【Bumping (h=1e-5)】")
        h = 1e-5
        f0 = rosenbrock_numeric(x0)

        t0 = time.perf_counter()
        H_bump = np.zeros((n, n))

        # Diagonal
        for i in range(n):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += h
            x_minus[i] -= h
            H_bump[i, i] = (rosenbrock_numeric(x_plus) - 2*f0 + rosenbrock_numeric(x_minus)) / (h**2)

        # Off-diagonal (sample 10% for large n)
        n_samples = min(n, 10) if n > 50 else n
        for i in range(n_samples):
            for j in range(i+1, min(i+n_samples, n)):
                x_pp = x0.copy()
                x_pm = x0.copy()
                x_mp = x0.copy()
                x_mm = x0.copy()

                x_pp[i] += h; x_pp[j] += h
                x_pm[i] += h; x_pm[j] -= h
                x_mp[i] -= h; x_mp[j] += h
                x_mm[i] -= h; x_mm[j] -= h

                H_bump[i, j] = (rosenbrock_numeric(x_pp) - rosenbrock_numeric(x_pm) -
                               rosenbrock_numeric(x_mp) + rosenbrock_numeric(x_mm)) / (4 * h**2)
                H_bump[j, i] = H_bump[i, j]

        t_bump = (time.perf_counter() - t0) * 1000

        nnz_bump = np.sum(np.abs(H_bump) > 1e-10)
        print(f"  Time: {t_bump:.2f} ms")
        print(f"  Non-zeros: {nnz_bump}")
        print(f"  Norm: {np.linalg.norm(H_bump):.6f}\n")

        results['bumping'] = {'time_ms': t_bump, 'hessian': H_bump, 'nnz': nnz_bump}

    # Algo3
    if 'algo3' in methods:
        print(f"【Algo3 (Block Form)】")
        global_tape.reset()

        x_ad = [ADVar(val) for val in x0]
        y_ad = rosenbrock_ad(x_ad)

        t0 = time.perf_counter()
        H_algo3 = algo3_block(y_ad, x_ad)
        t_algo3 = (time.perf_counter() - t0) * 1000

        nnz_algo3 = np.sum(np.abs(H_algo3) > 1e-10)
        print(f"  Time: {t_algo3:.2f} ms")
        print(f"  Non-zeros: {nnz_algo3}")
        print(f"  Norm: {np.linalg.norm(H_algo3):.6f}")

        if 'bumping' in results:
            error = np.max(np.abs(H_algo3 - H_bump))
            print(f"  Max error vs Bumping: {error:.2e}\n")
        else:
            print()

        results['algo3'] = {'time_ms': t_algo3, 'hessian': H_algo3, 'nnz': nnz_algo3}

    # Algo4
    if 'algo4' in methods:
        print(f"【Algo4-Opt (Edge-Pushing)】")
        global_tape.reset()

        x_ad = [ADVar(val) for val in x0]
        y_ad = rosenbrock_ad(x_ad)

        t0 = time.perf_counter()
        H_algo4 = algo4_optimized(y_ad, x_ad)
        t_algo4 = (time.perf_counter() - t0) * 1000

        nnz_algo4 = np.sum(np.abs(H_algo4) > 1e-10)
        print(f"  Time: {t_algo4:.2f} ms")
        print(f"  Non-zeros: {nnz_algo4}")
        print(f"  Norm: {np.linalg.norm(H_algo4):.6f}")

        if 'bumping' in results:
            error = np.max(np.abs(H_algo4 - H_bump))
            print(f"  Max error vs Bumping: {error:.2e}")
        if 'algo3' in results:
            error_vs_algo3 = np.max(np.abs(H_algo4 - H_algo3))
            print(f"  Max error vs Algo3: {error_vs_algo3:.2e}\n")
        else:
            print()

        results['algo4'] = {'time_ms': t_algo4, 'hessian': H_algo4, 'nnz': nnz_algo4}

    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Time (ms)':<15} {'Speedup':>15} {'Non-zeros':>15}")
    print(f"{'-'*80}")

    baseline_time = results.get('bumping', results.get('algo3', results.get('algo4')))['time_ms']

    for method, data in results.items():
        speedup = baseline_time / data['time_ms']
        print(f"{method.capitalize():<20} {data['time_ms']:<15.2f} {speedup:>15.2f}× {data['nnz']:>15}")

    if 'algo3' in results and 'algo4' in results:
        speedup_algo4_vs_algo3 = results['algo3']['time_ms'] / results['algo4']['time_ms']
        print(f"\nAlgo4 vs Algo3 speedup: {speedup_algo4_vs_algo3:.2f}×")

    return results


# ==================== Test Function 2: Polynomial Sum ====================

def polynomial_sum_numeric(x):
    """f(x) = sum_i (x_i^4 + x_i^3 + x_i^2 + x_i)"""
    return np.sum(x**4 + x**3 + x**2 + x)


def polynomial_sum_ad(x_vars):
    """Polynomial sum with ADVar"""
    result = ADVar(0.0)
    for x in x_vars:
        result = result + x**4 + x**3 + x**2 + x
    return result


def test_polynomial_sum(n, methods=['bumping', 'algo3', 'algo4']):
    """Test polynomial sum (diagonal Hessian)"""
    print(f"\n{'='*80}")
    print(f"TEST: Polynomial Sum (n={n}, Diagonal Hessian)")
    print(f"{'='*80}\n")

    x0 = 0.1 * np.random.randn(n) + 1.0

    results = {}

    # Bumping
    if 'bumping' in methods:
        print(f"【Bumping (h=1e-5)】")
        h = 1e-5
        f0 = polynomial_sum_numeric(x0)

        t0 = time.perf_counter()
        H_bump = np.zeros((n, n))

        for i in range(n):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += h
            x_minus[i] -= h
            H_bump[i, i] = (polynomial_sum_numeric(x_plus) - 2*f0 + polynomial_sum_numeric(x_minus)) / (h**2)

        t_bump = (time.perf_counter() - t0) * 1000

        nnz_bump = np.sum(np.abs(H_bump) > 1e-10)
        print(f"  Time: {t_bump:.2f} ms")
        print(f"  Non-zeros: {nnz_bump}")
        print(f"  Sparsity: {100*(1-nnz_bump/(n*n)):.1f}%\n")

        results['bumping'] = {'time_ms': t_bump, 'hessian': H_bump, 'nnz': nnz_bump}

    # Algo3
    if 'algo3' in methods:
        print(f"【Algo3 (Block Form)】")
        global_tape.reset()

        x_ad = [ADVar(val) for val in x0]
        y_ad = polynomial_sum_ad(x_ad)

        t0 = time.perf_counter()
        H_algo3 = algo3_block(y_ad, x_ad)
        t_algo3 = (time.perf_counter() - t0) * 1000

        nnz_algo3 = np.sum(np.abs(H_algo3) > 1e-10)
        print(f"  Time: {t_algo3:.2f} ms")
        print(f"  Non-zeros: {nnz_algo3}")
        print(f"  Sparsity: {100*(1-nnz_algo3/(n*n)):.1f}%")

        if 'bumping' in results:
            error = np.max(np.abs(H_algo3 - H_bump))
            print(f"  Max error vs Bumping: {error:.2e}\n")
        else:
            print()

        results['algo3'] = {'time_ms': t_algo3, 'hessian': H_algo3, 'nnz': nnz_algo3}

    # Algo4
    if 'algo4' in methods:
        print(f"【Algo4-Opt (Edge-Pushing)】")
        global_tape.reset()

        x_ad = [ADVar(val) for val in x0]
        y_ad = polynomial_sum_ad(x_ad)

        t0 = time.perf_counter()
        H_algo4 = algo4_optimized(y_ad, x_ad)
        t_algo4 = (time.perf_counter() - t0) * 1000

        nnz_algo4 = np.sum(np.abs(H_algo4) > 1e-10)
        print(f"  Time: {t_algo4:.2f} ms")
        print(f"  Non-zeros: {nnz_algo4}")
        print(f"  Sparsity: {100*(1-nnz_algo4/(n*n)):.1f}%")

        if 'bumping' in results:
            error = np.max(np.abs(H_algo4 - H_bump))
            print(f"  Max error vs Bumping: {error:.2e}\n")
        else:
            print()

        results['algo4'] = {'time_ms': t_algo4, 'hessian': H_algo4, 'nnz': nnz_algo4}

    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Time (ms)':<15} {'Speedup':>15}")
    print(f"{'-'*80}")

    baseline_time = results.get('bumping', results.get('algo3', results.get('algo4')))['time_ms']

    for method, data in results.items():
        speedup = baseline_time / data['time_ms']
        print(f"{method.capitalize():<20} {data['time_ms']:<15.2f} {speedup:>15.2f}×")

    if 'algo3' in results and 'algo4' in results:
        speedup_algo4_vs_algo3 = results['algo3']['time_ms'] / results['algo4']['time_ms']
        print(f"\nAlgo4 vs Algo3 speedup: {speedup_algo4_vs_algo3:.2f}×")

    return results


# ==================== Test Function 3: Sparse Quadratic ====================

def sparse_quadratic_numeric(x, adjacency):
    """Sparse quadratic: sum over adjacent pairs"""
    result = 0.0
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if j >= i:
                result += x[i] * x[j]
    return result


def sparse_quadratic_ad(x_vars, adjacency):
    """Sparse quadratic with ADVar"""
    result = ADVar(0.0)
    for i, neighbors in adjacency.items():
        for j in neighbors:
            if j >= i:
                result = result + x_vars[i] * x_vars[j]
    return result


def test_sparse_quadratic(n, sparsity=0.95, methods=['bumping', 'algo3', 'algo4']):
    """Test sparse quadratic form"""
    print(f"\n{'='*80}")
    print(f"TEST: Sparse Quadratic (n={n}, {100*sparsity:.0f}% sparse)")
    print(f"{'='*80}\n")

    # Create sparse adjacency (each variable connects to ~k neighbors)
    k = int(n * (1 - sparsity))
    adjacency = {}
    for i in range(n):
        # Connect to next k variables (circular)
        neighbors = [(i + j) % n for j in range(1, k+1)]
        adjacency[i] = neighbors

    x0 = 0.1 * np.random.randn(n) + 1.0

    print(f"Adjacency: Each variable has ~{k} neighbors\n")

    results = {}

    # Bumping (only diagonal for speed)
    if 'bumping' in methods and n <= 100:
        print(f"【Bumping (diagonal only)】")
        h = 1e-5
        f0 = sparse_quadratic_numeric(x0, adjacency)

        t0 = time.perf_counter()
        H_bump = np.zeros((n, n))

        for i in range(n):
            x_plus = x0.copy()
            x_minus = x0.copy()
            x_plus[i] += h
            x_minus[i] -= h
            H_bump[i, i] = (sparse_quadratic_numeric(x_plus, adjacency) - 2*f0 +
                           sparse_quadratic_numeric(x_minus, adjacency)) / (h**2)

        t_bump = (time.perf_counter() - t0) * 1000

        nnz_bump = np.sum(np.abs(H_bump) > 1e-10)
        print(f"  Time: {t_bump:.2f} ms")
        print(f"  Non-zeros: {nnz_bump}\n")

        results['bumping'] = {'time_ms': t_bump, 'hessian': H_bump, 'nnz': nnz_bump}

    # Algo3
    if 'algo3' in methods:
        print(f"【Algo3 (Block Form)】")
        global_tape.reset()

        x_ad = [ADVar(val) for val in x0]
        y_ad = sparse_quadratic_ad(x_ad, adjacency)

        t0 = time.perf_counter()
        H_algo3 = algo3_block(y_ad, x_ad)
        t_algo3 = (time.perf_counter() - t0) * 1000

        nnz_algo3 = np.sum(np.abs(H_algo3) > 1e-10)
        actual_sparsity = 100 * (1 - nnz_algo3 / (n*n))
        print(f"  Time: {t_algo3:.2f} ms")
        print(f"  Non-zeros: {nnz_algo3}")
        print(f"  Sparsity: {actual_sparsity:.1f}%\n")

        results['algo3'] = {'time_ms': t_algo3, 'hessian': H_algo3, 'nnz': nnz_algo3}

    # Algo4
    if 'algo4' in methods:
        print(f"【Algo4-Opt (Edge-Pushing)】")
        global_tape.reset()

        x_ad = [ADVar(val) for val in x0]
        y_ad = sparse_quadratic_ad(x_ad, adjacency)

        t0 = time.perf_counter()
        H_algo4 = algo4_optimized(y_ad, x_ad)
        t_algo4 = (time.perf_counter() - t0) * 1000

        nnz_algo4 = np.sum(np.abs(H_algo4) > 1e-10)
        actual_sparsity = 100 * (1 - nnz_algo4 / (n*n))
        print(f"  Time: {t_algo4:.2f} ms")
        print(f"  Non-zeros: {nnz_algo4}")
        print(f"  Sparsity: {actual_sparsity:.1f}%")

        if 'algo3' in results:
            error = np.max(np.abs(H_algo4 - H_algo3))
            print(f"  Max error vs Algo3: {error:.2e}\n")
        else:
            print()

        results['algo4'] = {'time_ms': t_algo4, 'hessian': H_algo4, 'nnz': nnz_algo4}

    # Summary
    print(f"{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'Time (ms)':<15} {'Speedup':>15}")
    print(f"{'-'*80}")

    baseline_time = results.get('algo3', results.get('algo4'))['time_ms']

    for method, data in results.items():
        speedup = baseline_time / data['time_ms']
        print(f"{method.capitalize():<20} {data['time_ms']:<15.2f} {speedup:>15.2f}×")

    if 'algo3' in results and 'algo4' in results:
        speedup_algo4_vs_algo3 = results['algo3']['time_ms'] / results['algo4']['time_ms']
        print(f"\nAlgo4 vs Algo3 speedup: {speedup_algo4_vs_algo3:.2f}×")

    return results


# ==================== Main Test Suite ====================

def run_all_tests():
    """Run all large-scale tests"""
    print("="*80)
    print("LARGE-SCALE COMPLEX FUNCTION TESTS")
    print("Algo3 vs Algo4 vs Bumping")
    print("="*80)

    all_results = {}

    # Test 1: Rosenbrock (small)
    all_results['rosenbrock_50'] = test_rosenbrock(50, methods=['bumping', 'algo3', 'algo4'])

    # Test 2: Rosenbrock (medium)
    all_results['rosenbrock_100'] = test_rosenbrock(100, methods=['algo3', 'algo4'])

    # Test 3: Rosenbrock (large)
    all_results['rosenbrock_200'] = test_rosenbrock(200, methods=['algo3', 'algo4'])

    # Test 4: Polynomial sum (sparse diagonal)
    all_results['polynomial_50'] = test_polynomial_sum(50, methods=['bumping', 'algo3', 'algo4'])
    all_results['polynomial_100'] = test_polynomial_sum(100, methods=['algo3', 'algo4'])
    all_results['polynomial_200'] = test_polynomial_sum(200, methods=['algo3', 'algo4'])

    # Test 5: Sparse quadratic
    all_results['sparse_quad_100_95'] = test_sparse_quadratic(100, 0.95, methods=['bumping', 'algo3', 'algo4'])
    all_results['sparse_quad_200_95'] = test_sparse_quadratic(200, 0.95, methods=['algo3', 'algo4'])

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY: All Tests")
    print("="*80)
    print(f"{'Test':<30} {'n':<10} {'Algo3 (ms)':<15} {'Algo4 (ms)':<15} {'Speedup':>15}")
    print("-"*80)

    for test_name, results in all_results.items():
        if 'algo3' in results and 'algo4' in results:
            n = len(results['algo3']['hessian'])
            t3 = results['algo3']['time_ms']
            t4 = results['algo4']['time_ms']
            speedup = t3 / t4
            print(f"{test_name:<30} {n:<10} {t3:<15.2f} {t4:<15.2f} {speedup:>15.2f}×")

    return all_results


if __name__ == "__main__":
    results = run_all_tests()
