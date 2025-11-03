"""
Test Implicit Function Theorem method for computing Vega at high volatility

Compares:
1. Double-AAD (current method with 22% Vega error)
2. Implicit Function Theorem (expected <5% Vega error)
3. BSM Analytical (baseline)

Focus: Does IFT solve the Vega instability problem at σ=50%?
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
import time
from aad_edge_pushing.pde.pde_implicit_function import BS_PDE_ImplicitFunction
from aad_edge_pushing.pde.methods.double_aad import DoubleAADMethod
from aad_edge_pushing.pde.methods.bsm_analytical import BSMAnalyticalMethod

# High volatility test parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.50  # High volatility where Double-AAD fails

# Test with coarse grid first (faster)
M = 51
N = 100

print("="*80)
print("IMPLICIT FUNCTION THEOREM vs DOUBLE-AAD: Vega Test at σ=50%")
print("="*80)
print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r:.2%}, σ={sigma:.1%}")
print(f"Grid: M={M}, N={N}")
print()

# Baseline: BSM Analytical
print("-"*80)
print("BSM ANALYTICAL (Baseline)")
print("-"*80)
bsm = BSMAnalyticalMethod(M=M, N=N, S0=S0, K=K, T=T, r=r)
result_bsm = bsm.compute_hessian(S0, sigma)

print(f"  Price:  {result_bsm['price']:.6f}")
print(f"  Delta:  {result_bsm['greeks']['delta']:.6f}")
print(f"  Gamma:  {result_bsm['greeks']['gamma']:.6f}")
print(f"  Vega:   {result_bsm['greeks']['vega']:.6f}")
print(f"  Time:   {result_bsm['time_ms']:.2f}ms")

# Test 1: Double-AAD (current method)
print("\n" + "-"*80)
print("TEST 1: Double-AAD (Forward-over-Reverse)")
print("-"*80)
print("  Computing...")

double_aad = DoubleAADMethod(M=M, N=N, S0=S0, K=K, T=T, r=r)
result_double = double_aad.compute_hessian(S0, sigma)

print(f"  Price:  {result_double['price']:.6f}")
print(f"  Delta:  {result_double['greeks']['delta']:.6f}")
print(f"  Gamma:  {result_double['greeks']['gamma']:.6f}")
print(f"  Vega:   {result_double['greeks']['vega']:.6f}")
print(f"  Time:   {result_double['time_ms']:.2f}ms")

# Test 2: Implicit Function Theorem
print("\n" + "-"*80)
print("TEST 2: Implicit Function Theorem")
print("-"*80)
print("  Computing...")

t_start = time.perf_counter()

solver_ift = BS_PDE_ImplicitFunction(
    S0=S0, K=K, T=T, r=r, sigma=sigma,
    M=M, N_base=N
)

# Compute price
price_ift, _ = solver_ift.solve_pde_numerical(sigma)

# Compute Vega using implicit function theorem
vega_ift, vega_grid_ift = solver_ift.apply_implicit_function_theorem(
    sigma, eps_sigma=0.01
)

t_end = time.perf_counter()
time_ift = (t_end - t_start) * 1000.0

print(f"  Price:  {price_ift:.6f}")
print(f"  Vega:   {vega_ift:.6f}")
print(f"  Time:   {time_ift:.2f}ms")

# Error Analysis
print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

def calc_error(val, ref):
    """Calculate percentage error"""
    if abs(ref) < 1e-10:
        return 0.0 if abs(val) < 1e-10 else 100.0
    return abs(val - ref) / abs(ref) * 100.0

price_err_double = calc_error(result_double['price'], result_bsm['price'])
vega_err_double = calc_error(result_double['greeks']['vega'], result_bsm['greeks']['vega'])

price_err_ift = calc_error(price_ift, result_bsm['price'])
vega_err_ift = calc_error(vega_ift, result_bsm['greeks']['vega'])

print(f"\n{'Method':<20} {'Price Error':<15} {'Vega Error':<15} {'Time (ms)':<15}")
print("-"*65)
print(f"{'Double-AAD':<20} {price_err_double:>13.2f}% {vega_err_double:>13.2f}% {result_double['time_ms']:>13.1f}")
print(f"{'Implicit Function':<20} {price_err_ift:>13.2f}% {vega_err_ift:>13.2f}% {time_ift:>13.1f}")

# Improvement calculation
if vega_err_double > 1e-10:
    vega_improvement = (vega_err_double - vega_err_ift) / vega_err_double * 100.0
else:
    vega_improvement = 0.0

print("\n" + "-"*80)
print("VEGA ACCURACY IMPROVEMENT")
print("-"*80)
print(f"  Double-AAD Vega Error:     {vega_err_double:.2f}%")
print(f"  IFT Vega Error:            {vega_err_ift:.2f}%")
print(f"  Improvement:               {vega_improvement:.1f}%")

# Conclusion
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if vega_err_ift < 5.0:
    print(f"\n✅ EXCELLENT: IFT achieves {vega_err_ift:.2f}% Vega error (<5% target)")
    print("   Implicit Function Theorem successfully solves the instability problem!")
elif vega_err_ift < 10.0:
    print(f"\n⚠️  GOOD: IFT achieves {vega_err_ift:.2f}% Vega error (<10%)")
    print("   Significant improvement over Double-AAD")
elif vega_err_ift < vega_err_double * 0.5:
    print(f"\n⚠️  IMPROVED: IFT reduces Vega error by {vega_improvement:.1f}%")
    print("   Better than Double-AAD but may need further tuning")
else:
    print(f"\n❌ NO IMPROVEMENT: IFT Vega error {vega_err_ift:.2f}% still high")
    print("   May need to refine Jacobian computation or use finer finite differences")

print(f"\nComputational cost comparison:")
print(f"  Double-AAD:  {result_double['time_ms']:.2f}ms")
print(f"  IFT:         {time_ift:.2f}ms")
if time_ift < result_double['time_ms']:
    print(f"  IFT is {result_double['time_ms'] / time_ift:.1f}x faster!")
else:
    print(f"  IFT is {time_ift / result_double['time_ms']:.1f}x slower (but more accurate)")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("-"*80)
print("1. Literature prediction (Marc Henrard 2011):")
print("   'Thanks to the implicit function theorem, differentiation of the solver")
print("    embedded in the calibration is not required'")
print()
print("2. Our implementation:")
if vega_err_ift < vega_err_double * 0.8:
    print("   ✅ Validates the literature - IFT avoids gradient accumulation")
    print("   ✅ Solves the numerical instability of AAD through implicit solves")
else:
    print("   ⚠️  Implementation may need refinement")
    print("   Consider: finer eps_sigma, better Jacobian approximation")
print("="*80)
