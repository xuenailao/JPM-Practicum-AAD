"""
Test simple cubic spline interpolation without PDE
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape


def test_simple_function():
    """Test spline on f(x) = x^2"""
    print("="*80)
    print("Test: Natural Cubic Spline on f(x) = x^2")
    print("="*80)

    # Create grid
    x_grid = np.linspace(0, 10, 11)
    y_grid = x_grid**2

    print(f"\nGrid: {len(x_grid)} points from {x_grid[0]} to {x_grid[-1]}")
    print(f"  x: {x_grid}")
    print(f"  y: {y_grid}")

    # Test interpolation at x=5.0 (should give y=25)
    x_test = 5.0
    y_exact = x_test**2

    print(f"\nTest point: x={x_test}, y_exact={y_exact}")

    # Find interval
    idx = np.searchsorted(x_grid, x_test)
    print(f"  searchsorted index: {idx}")
    print(f"  Interval: [{x_grid[idx-1]}, {x_grid[idx]}]")

    # Linear interpolation (baseline)
    w = (x_test - x_grid[idx-1]) / (x_grid[idx] - x_grid[idx-1])
    y_linear = (1-w) * y_grid[idx-1] + w * y_grid[idx]
    print(f"\nLinear interpolation:")
    print(f"  w = {w}")
    print(f"  y = {y_linear}")
    print(f"  Error: {abs(y_linear - y_exact)/y_exact*100:.2f}%")

    # Cubic spline interpolation
    # For uniform grid, natural spline with f(x)=x^2 should give M_i ≈ 2
    h = x_grid[1] - x_grid[0]
    M_i = 2.0  # Approximate for uniform grid with f(x)=x^2
    M_i1 = 2.0

    x_i = x_grid[idx-1]
    x_i1 = x_grid[idx]
    y_i = y_grid[idx-1]
    y_i1 = y_grid[idx]

    A = (x_i1 - x_test) / h
    B = (x_test - x_i) / h

    y_spline = (A * y_i + B * y_i1 +
               (A**3 - A) * h**2 / 6 * M_i +
               (B**3 - B) * h**2 / 6 * M_i1)

    print(f"\nCubic spline interpolation (with M={M_i}):")
    print(f"  A = {A}, B = {B}")
    print(f"  y = {y_spline}")
    print(f"  Error: {abs(y_spline - y_exact)/y_exact*100:.2f}%")

    print("\n" + "="*80)

    # Now test with actual spline computation using ADVar
    print("\nTest with ADVar spline computation:")
    print("="*80)

    global_tape.reset()

    # Convert to ADVars
    V = [ADVar(y, requires_grad=False) for y in y_grid]
    n = len(V)

    # Compute M values manually (natural spline tridiagonal system)
    h_vals = np.diff(x_grid)

    # For uniform grid with n points, interior system is (n-2) x (n-2)
    n_interior = n - 2

    #Build system
    lambda_vals = []
    mu_vals = []
    d_vals = []

    for i in range(1, n - 1):
        h_im1 = h_vals[i - 1]
        h_i = h_vals[i]

        lambda_i = h_im1 / (h_im1 + h_i)
        mu_i = h_i / (h_im1 + h_i)

        d_i = (ADVar(6.0) / ADVar(h_im1 + h_i)) * (
            (V[i + 1] - V[i]) / ADVar(h_i) - (V[i] - V[i - 1]) / ADVar(h_im1)
        )

        lambda_vals.append(lambda_i)
        mu_vals.append(mu_i)
        d_vals.append(d_i)

    print(f"  Tridiagonal system: {n_interior} x {n_interior}")
    print(f"  λ values: {lambda_vals[:3]}...")
    print(f"  μ values: {mu_vals[:3]}...")

    # Solve tridiagonal (Thomas algorithm)
    a = [ADVar(0.0)] + [ADVar(lam) for lam in lambda_vals]
    b = [ADVar(2.0) for _ in range(n_interior)]
    c = [ADVar(mu) for mu in mu_vals] + [ADVar(0.0)]
    d = d_vals

    # Forward elimination
    c_prime = [None] * n_interior
    d_prime = [None] * n_interior

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n_interior):
        denom = b[i] - a[i] * c_prime[i - 1]
        c_prime[i] = c[i] / denom if i < n_interior - 1 else ADVar(0.0)
        d_prime[i] = (d[i] - a[i] * d_prime[i - 1]) / denom

    # Back substitution
    M_interior = [None] * n_interior
    M_interior[-1] = d_prime[-1]

    for i in range(n_interior - 2, -1, -1):
        M_interior[i] = d_prime[i] - c_prime[i] * M_interior[i + 1]

    # Add boundary conditions
    M_vals = [ADVar(0.0, requires_grad=False)] + M_interior + [ADVar(0.0, requires_grad=False)]

    print(f"  M values (first 5): {[M_vals[i].val for i in range(min(5, len(M_vals)))]}")
    print(f"  M average: {np.mean([M_vals[i].val for i in range(1, len(M_vals)-1)]):.6f}")
    print(f"  Expected: ~2.0 for f(x)=x^2")

    # Interpolate at x_test
    x_test_var = ADVar(x_test, requires_grad=True)

    i = idx - 1
    V_i = V[i]
    V_i1 = V[i + 1]
    M_i_val = M_vals[i]
    M_i1_val = M_vals[i + 1]

    x_i_var = ADVar(x_grid[i], requires_grad=False)
    x_i1_var = ADVar(x_grid[i + 1], requires_grad=False)
    h_var = ADVar(h, requires_grad=False)

    A_var = (x_i1_var - x_test_var) / h_var
    B_var = (x_test_var - x_i_var) / h_var

    A3_var = A_var * A_var * A_var
    B3_var = B_var * B_var * B_var

    h2_over_6 = h_var * h_var / ADVar(6.0)

    y_var = (A_var * V_i + B_var * V_i1 +
            (A3_var - A_var) * h2_over_6 * M_i_val +
            (B3_var - B_var) * h2_over_6 * M_i1_val)

    y_spline_ad = y_var.val

    print(f"\n  Spline result: y = {y_spline_ad}")
    print(f"  Exact: y = {y_exact}")
    print(f"  Error: {abs(y_spline_ad - y_exact)/y_exact*100:.2f}%")

    # Check derivative
    y_var.adj = 1.0
    for node in reversed(global_tape.nodes):
        for parent, deriv in node.parents:
            if parent.requires_grad:
                parent.adj += node.out.adj * float(deriv)

    dy_dx_ad = x_test_var.adj
    dy_dx_exact = 2 * x_test

    print(f"\n  First derivative via AD: dy/dx = {dy_dx_ad}")
    print(f"  Exact: dy/dx = {dy_dx_exact}")
    print(f"  Error: {abs(dy_dx_ad - dy_dx_exact)/dy_dx_exact*100:.2f}%")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_simple_function()
