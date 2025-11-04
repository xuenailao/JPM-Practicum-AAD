"""
Debug: Why does adaptive N not change price/Vega at all?

Hypothesis: The solve_pde_with_aad might be using a different N internally,
or there's caching happening.
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.50
M, N_base = 101, 200

print("DEBUG: Checking if N actually changes")
print("=" * 60)

# Test 1: Fixed N
solver1 = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N_base, adaptive_N=False)
print(f"\nFixed N solver:")
print(f"  solver.N = {solver1.N}")
print(f"  solver.N_base = {solver1.N_base}")
print(f"  solver.dt = {solver1.dt}")
print(f"  solver.cfl_ratio = {solver1.cfl_ratio:.4f}")

# Test 2: Adaptive N
solver2 = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N_base, adaptive_N=True)
print(f"\nAdaptive N solver:")
print(f"  solver.N = {solver2.N}")
print(f"  solver.N_base = {solver2.N_base}")
print(f"  solver.dt = {solver2.dt}")
print(f"  solver.cfl_ratio = {solver2.cfl_ratio:.4f}")

print(f"\nDifference:")
print(f"  N changed: {solver1.N} → {solver2.N}")
print(f"  dt changed: {solver1.dt:.6f} → {solver2.dt:.6f}")

# Now solve PDE and check if results are different
print(f"\n{'='*60}")
print("Solving PDE...")
print('='*60)

result1 = solver1.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)
result2 = solver2.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)

print(f"\nFixed N (N={solver1.N}):")
print(f"  Price: {result1['price']:.8f}")
print(f"  Vega:  {result1['vega']:.8f}")

print(f"\nAdaptive N (N={solver2.N}):")
print(f"  Price: {result2['price']:.8f}")
print(f"  Vega:  {result2['vega']:.8f}")

print(f"\nDifference:")
print(f"  Price: {abs(result1['price'] - result2['price']):.8e}")
print(f"  Vega:  {abs(result1['vega'] - result2['vega']):.8e}")

if abs(result1['price'] - result2['price']) < 1e-10:
    print(f"\n❌ ERROR: Results are IDENTICAL despite different N!")
    print(f"   This suggests solve_pde_with_aad is NOT using self.N")
    print(f"   Need to check how cn_step() is called")
else:
    print(f"\n✓ Results are different as expected")
