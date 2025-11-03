"""Test Natural Cubic Spline Implementation"""

import numpy as np
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
gamma_bsm = norm.pdf(d1) / (S0 * sigma * sqrt_T)

print("="*80)
print("Testing Natural Cubic Spline Interpolation")
print("="*80)
print(f"\nBSM Analytical Gamma: {gamma_bsm:.10f}\n")

configs = [
    (51, 50),
    (51, 100),
    (101, 100),
    (101, 200),
]

print(f"{'M':>4} {'N':>4} {'Gamma':>13} {'Error':>8} {'Time':>10}")
print("-"*80)

for M, N in configs:
    try:
        solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=M, N_base=N, center_on_S0=False)
        result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=False)
        
        gamma = result.get('gamma', 0.0)
        error = abs(gamma - gamma_bsm) / gamma_bsm * 100
        time_ms = result.get('time_ms', 0.0)
        
        print(f"{M:>4d} {N:>4d} {gamma:>13.10f} {error:>7.2f}% {time_ms:>9.1f}ms")
    except Exception as e:
        print(f"{M:>4d} {N:>4d} FAILED: {str(e)[:50]}")

print("\n" + "="*80)
print("Natural Spline Key Features:")
print("="*80)
print("• Globally consistent curvature M_i (solves tridiagonal system)")
print("• C² continuous across all intervals")
print("• Natural boundary conditions: M[0] = M[-1] = 0")
print("• More stable than local Hermite interpolation")
print("="*80)
