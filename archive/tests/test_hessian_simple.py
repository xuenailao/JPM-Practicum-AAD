"""
Test Hessian computation on a simple function first
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.edge_pushing.algo4_adjlist import algo4_adjlist


def test_simple_hessian():
    """Test Hessian on f(x) = x^4"""

    print("="*80)
    print("Simple Hessian Test: f(x) = x^4")
    print("="*80)

    global_tape.reset()

    x = ADVar(2.0, requires_grad=True, name="x")

    # f(x) = x^4
    y = x * x  # x^2
    z = y * y  # x^4

    print(f"\nf(x) = x^4")
    print(f"x = {x.val}")
    print(f"f(x) = {z.val}")

    # Analytical derivatives
    x_val = 2.0
    f_analytical = x_val ** 4
    df_analytical = 4 * x_val ** 3
    d2f_analytical = 12 * x_val ** 2

    print(f"\nAnalytical:")
    print(f"  f(x) = {f_analytical}")
    print(f"  f'(x) = {df_analytical}")
    print(f"  f''(x) = {d2f_analytical}")

    # Compute Hessian using algo4
    print(f"\nComputing Hessian using Edge-Pushing...")
    print(f"  Nodes in tape: {len(global_tape.nodes)}")

    hessian = algo4_adjlist(z, [x])

    print(f"\nHessian:")
    print(f"  Shape: {hessian.shape}")
    print(f"  H[0,0] = {hessian[0, 0]}")

    # Compare
    error = abs(hessian[0, 0] - d2f_analytical) / abs(d2f_analytical) * 100

    print(f"\nComparison:")
    print(f"  Analytical f''(x) = {d2f_analytical}")
    print(f"  AAD f''(x) = {hessian[0, 0]}")
    print(f"  Error = {error:.2f}%")

    if error < 0.01:
        print("\n✅ Hessian computation works!")
    else:
        print("\n❌ Hessian computation has error")


def test_two_var_hessian():
    """Test Hessian on f(x,y) = x^2*y + x*y^2"""

    print("\n" + "="*80)
    print("Two-variable Hessian Test: f(x,y) = x^2*y + x*y^2")
    print("="*80)

    global_tape.reset()

    x = ADVar(2.0, requires_grad=True, name="x")
    y = ADVar(3.0, requires_grad=True, name="y")

    # f(x,y) = x^2*y + x*y^2
    x2 = x * x
    y2 = y * y
    term1 = x2 * y
    term2 = x * y2
    f = term1 + term2

    print(f"\nf(x,y) = x^2*y + x*y^2")
    print(f"x = {x.val}, y = {y.val}")
    print(f"f(x,y) = {f.val}")

    # Analytical Hessian
    # f_xx = 2y
    # f_xy = 2x + 2y
    # f_yx = 2x + 2y
    # f_yy = 2x

    x_val, y_val = 2.0, 3.0
    f_xx = 2 * y_val
    f_xy = 2 * x_val + 2 * y_val
    f_yy = 2 * x_val

    print(f"\nAnalytical Hessian:")
    print(f"  f_xx = {f_xx}")
    print(f"  f_xy = {f_xy}")
    print(f"  f_yy = {f_yy}")

    # Compute Hessian
    print(f"\nComputing Hessian...")
    print(f"  Nodes in tape: {len(global_tape.nodes)}")

    hessian = algo4_adjlist(f, [x, y])

    print(f"\nAAD Hessian:")
    print(f"  Shape: {hessian.shape}")
    print(f"  H[0,0] (f_xx) = {hessian[0, 0]}")
    print(f"  H[0,1] (f_xy) = {hessian[0, 1]}")
    print(f"  H[1,0] (f_yx) = {hessian[1, 0]}")
    print(f"  H[1,1] (f_yy) = {hessian[1, 1]}")

    # Compare
    err_xx = abs(hessian[0, 0] - f_xx) / abs(f_xx) * 100
    err_xy = abs(hessian[0, 1] - f_xy) / abs(f_xy) * 100
    err_yy = abs(hessian[1, 1] - f_yy) / abs(f_yy) * 100

    print(f"\nErrors:")
    print(f"  f_xx: {err_xx:.4f}%")
    print(f"  f_xy: {err_xy:.4f}%")
    print(f"  f_yy: {err_yy:.4f}%")

    if max(err_xx, err_xy, err_yy) < 0.01:
        print("\n✅ Multi-variable Hessian works!")
    else:
        print("\n❌ Multi-variable Hessian has error")


def test_tape_size():
    """Test how tape size grows with PDE"""

    print("\n" + "="*80)
    print("Tape Size Analysis")
    print("="*80)

    # Simulate a small tridiagonal solve
    global_tape.reset()

    x = ADVar(2.0, requires_grad=True, name="x")

    # Simulate 10 time steps, each with 10 spatial points
    N = 10  # time steps
    M = 10  # spatial points

    V = [ADVar(float(i), requires_grad=False) for i in range(M)]

    for n in range(N):
        V_new = []
        for i in range(M):
            if i == 0:
                # Boundary
                V_new.append(V[i])
            elif i == M-1:
                # Boundary
                V_new.append(V[i])
            else:
                # Interior: simple average (simulates tridiag solve)
                v_i = (V[i-1] + ADVar(2.0) * V[i] + V[i+1]) / ADVar(4.0)
                # Add dependence on x
                v_i = v_i + x * ADVar(0.001)
                V_new.append(v_i)
        V = V_new

    # Final output: sum of all V
    output = V[0]
    for i in range(1, M):
        output = output + V[i]

    print(f"\nSimulated PDE:")
    print(f"  Time steps: {N}")
    print(f"  Spatial points: {M}")
    print(f"  Total operations: ~{N*M}")

    print(f"\nTape size:")
    print(f"  Nodes: {len(global_tape.nodes)}")

    print(f"\nComputing Hessian...")
    import time
    t_start = time.perf_counter()

    try:
        hessian = algo4_adjlist(output, [x])
        t_elapsed = time.perf_counter() - t_start

        print(f"  Time: {t_elapsed:.3f}s")
        print(f"  Hessian[0,0]: {hessian[0, 0]}")
        print("\n✅ Hessian computed successfully!")

    except Exception as e:
        t_elapsed = time.perf_counter() - t_start
        print(f"  Time before error: {t_elapsed:.3f}s")
        print(f"  Error: {e}")
        print("\n❌ Hessian computation failed")

    # Estimate for full PDE
    N_full = 150
    M_full = 151
    scale = (N_full * M_full) / (N * M)

    print(f"\nEstimate for full PDE (M={M_full}, N={N_full}):")
    print(f"  Estimated nodes: ~{len(global_tape.nodes) * scale:.0f}")
    print(f"  Estimated time: ~{t_elapsed * scale:.1f}s")


if __name__ == "__main__":
    test_simple_hessian()
    test_two_var_hessian()
    test_tape_size()
