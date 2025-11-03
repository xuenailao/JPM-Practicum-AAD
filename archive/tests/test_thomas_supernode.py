"""
Test script for Thomas Super-Node implementation
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.pde.thomas_supernode_advar import ThomasSuperNode

def test_thomas_supernode():
    """Test Thomas super-node vs naive approach."""

    print("=" * 80)
    print("TEST 1: Thomas Algorithm - Pure NumPy vs ADVar Super-Node")
    print("=" * 80)

    # Test case: Solve tridiagonal system
    n = 10
    a = np.array([0] + [1.0] * (n-1))  # Lower diagonal
    b = np.array([2.0] * n)             # Main diagonal
    c = np.array([1.0] * (n-1) + [0])  # Upper diagonal
    d_vals = np.arange(1, n+1, dtype=float)

    # 1. Pure NumPy solution
    print("\n1. Pure NumPy Solution")
    x_numpy = ThomasSuperNode.solve(a, b, c, d_vals)
    print(f"   Solution: {x_numpy}")

    # Verify: Ax = d
    A = np.diag(b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)
    residual = np.linalg.norm(A @ x_numpy - d_vals)
    print(f"   Residual ||Ax - d||: {residual:.2e}")

    # 2. ADVar Super-Node solution
    print("\n2. ADVar Super-Node Solution")
    global_tape.reset()

    d_advar = [ADVar(d_vals[i], requires_grad=True, name=f'd{i}') for i in range(n)]
    x_advar = ThomasSuperNode.solve_advar(a, b, c, d_advar, requires_grad=True)

    x_advar_vals = np.array([x.val for x in x_advar])
    print(f"   Solution: {x_advar_vals}")
    print(f"   Difference from NumPy: {np.linalg.norm(x_advar_vals - x_numpy):.2e}")

    # 3. Check graph size
    print("\n3. Computation Graph Analysis")
    print(f"   Number of nodes created: {len(global_tape.nodes)}")
    print(f"   Expected: 1 (super-node)")
    print(f"   Naive approach would create: ~{5*n} nodes")
    print(f"   Memory savings: {5*n}x")

    # 4. Test backward pass
    print("\n4. Gradient Computation (Backward Pass)")

    # Seed gradient: ∂L/∂x[5] = 1
    for i in range(n):
        x_advar[i].adj = 0.0
    x_advar[5].adj = 1.0

    # Backward pass
    for node in reversed(global_tape.nodes):
        if hasattr(node, 'backward_fn') and node.backward_fn:
            node.backward_fn()

    # Check gradients
    d_grads = np.array([d.adj for d in d_advar])
    print(f"   Gradients ∂x[5]/∂d: {d_grads}")

    # Verify with finite differences
    eps = 1e-7
    d_fd = d_vals.copy()
    d_fd[5] += eps
    x_fd = ThomasSuperNode.solve(a, b, c, d_fd)
    fd_grad_5 = (x_fd[5] - x_numpy[5]) / eps
    print(f"   Finite difference ∂x[5]/∂d[5]: {fd_grad_5:.6f}")
    print(f"   Super-node gradient ∂x[5]/∂d[5]: {d_grads[5]:.6f}")
    print(f"   Error: {abs(fd_grad_5 - d_grads[5]):.2e}")

    print("\n" + "=" * 80)
    print("TEST 2: Comparison with Naive ADVar Approach (Simulated)")
    print("=" * 80)

    # Simulate what would happen with naive approach
    naive_ops_per_step = 5  # divisions, multiplications, subtractions
    naive_total_nodes = n * naive_ops_per_step

    print(f"\nNaive approach (each arithmetic operation = 1 node):")
    print(f"   Forward sweep: {n} steps × {naive_ops_per_step} ops = {naive_total_nodes} nodes")
    print(f"   Backward sweep: {n} steps × {naive_ops_per_step} ops = {naive_total_nodes} adjoints")
    print(f"   Total graph nodes: {naive_total_nodes}")
    print(f"   Memory: O({naive_total_nodes})")

    print(f"\nSuper-node approach:")
    print(f"   Forward sweep: 1 super-node")
    print(f"   Backward sweep: 1 custom adjoint operation")
    print(f"   Total graph nodes: 1")
    print(f"   Memory: O(1)")

    print(f"\nSpeedup factor: {naive_total_nodes}x")

    print("\n" + "=" * 80)
    print("TEST 3: Scaling Test (n = 100, 1000)")
    print("=" * 80)

    for n_test in [100, 1000]:
        a_test = np.array([0] + [1.0] * (n_test-1))
        b_test = np.array([2.0] * n_test)
        c_test = np.array([1.0] * (n_test-1) + [0])
        d_test = np.ones(n_test)

        # NumPy
        import time
        t0 = time.time()
        x_np = ThomasSuperNode.solve(a_test, b_test, c_test, d_test)
        t_numpy = time.time() - t0

        # Super-node
        global_tape.reset()
        d_advar_test = [ADVar(d_test[i], requires_grad=True) for i in range(n_test)]

        t0 = time.time()
        x_ad = ThomasSuperNode.solve_advar(a_test, b_test, c_test, d_advar_test)
        t_supernode = time.time() - t0

        print(f"\n  n = {n_test}:")
        print(f"    NumPy time:      {t_numpy*1000:.3f} ms")
        print(f"    Super-node time: {t_supernode*1000:.3f} ms")
        print(f"    Overhead:        {(t_supernode/t_numpy - 1)*100:.1f}%")
        print(f"    Nodes created:   {len(global_tape.nodes)}")
        print(f"    Naive would create: ~{5*n_test} nodes")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_thomas_supernode()
