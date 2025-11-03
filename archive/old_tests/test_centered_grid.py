"""
Test: Centered Grid (S0 on grid point)

When S0 is exactly on a grid point, we don't need interpolation.
But we still need S0 to be an ADVar to compute Gamma.

Key insight: Use finite differences directly on V_grid with S0_var.
"""

import numpy as np
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

# BSM analytical
sqrt_T = np.sqrt(T)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
gamma_bsm = norm.pdf(d1) / (S0 * sigma * sqrt_T)

print("="*80)
print("Testing Centered Grid (S0 on grid point)")
print("="*80)
print(f"\nBSM Analytical Gamma: {gamma_bsm:.10f}\n")

# Test with centered grid
solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, M=51, N_base=50, center_on_S0=True)

print(f"Grid centered on S0:")
print(f"  S0 = {S0}")
print(f"  S0_idx = {solver.S0_idx}")
print(f"  S_grid[S0_idx] = {solver.S_grid[solver.S0_idx]:.10f}")
print(f"  Error = {abs(solver.S_grid[solver.S0_idx] - S0):.2e}\n")

result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=True)

print(f"\nResults:")
print(f"  Price = {result['price']:.10f}")
print(f"  Delta = {result['delta']:.10f}")
print(f"  Gamma = {result.get('gamma', 0.0):.10f}")
print(f"  Vega  = {result['vega']:.10f}")

gamma_error = abs(result.get('gamma', 0.0) - gamma_bsm) / gamma_bsm * 100
print(f"\n  Gamma Error: {gamma_error:.2f}%")

if abs(result.get('gamma', 0.0)) < 1e-10:
    print(f"\n❌ Gamma is still zero!")
    print(f"   This is expected: when S0 is on grid, price doesn't depend on S0_var")
else:
    print(f"\n✅ Gamma = {result.get('gamma', 0.0):.10f} (non-zero!)")
