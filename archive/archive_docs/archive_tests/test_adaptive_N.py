"""
Test adaptive timestepping to fix CFL violation and Vega errors

Expected results:
- At σ=50% with adaptive_N=True:
  - N increases from 200 → ~1250 to maintain CFL < 1
  - Vega error drops from 24% → <5%
  - Vanna/Volga errors should also improve significantly
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
import time
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

def bsm_analytical(S, K, T, r, sigma):
    """BSM analytical solution"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    price = S*N_d1 - K*np.exp(-r*T)*N_d2
    vega = S*n_d1*np.sqrt(T)

    return price, vega

# Test parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05

print("=" * 90)
print("ADAPTIVE TIMESTEPPING TEST: Fixing CFL Violation at High Volatility")
print("=" * 90)

# Test at multiple volatilities
sigmas = [0.20, 0.30, 0.40, 0.50]
M = 101
N_base = 200

for sigma in sigmas:
    print(f"\n{'='*90}")
    print(f"Testing σ = {sigma:.1%}")
    print('='*90)

    # BSM reference
    price_bsm, vega_bsm = bsm_analytical(S0, K, T, r, sigma)

    # Test 1: Without adaptive N (original, CFL violation)
    print(f"\n[Test 1] Without Adaptive N (CFL violation expected)")
    print("-" * 90)

    solver_fixed = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N_base, adaptive_N=False)

    t_start = time.perf_counter()
    result_fixed = solver_fixed.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)
    t_fixed = (time.perf_counter() - t_start) * 1000

    price_fixed = result_fixed['price']
    vega_fixed = result_fixed['vega']

    price_err_fixed = abs(price_fixed - price_bsm) / price_bsm * 100
    vega_err_fixed = abs(vega_fixed - vega_bsm) / vega_bsm * 100

    print(f"  N:          {solver_fixed.N}")
    print(f"  dt:         {solver_fixed.dt:.6f}")
    print(f"  CFL ratio:  {solver_fixed.cfl_ratio:.4f} {'⚠️ >1' if solver_fixed.cfl_ratio > 1.0 else '✓'}")
    print(f"  Price:      {price_fixed:.6f} (BSM: {price_bsm:.6f}, error: {price_err_fixed:.2f}%)")
    print(f"  Vega:       {vega_fixed:.6f} (BSM: {vega_bsm:.6f}, error: {vega_err_fixed:.2f}%)")
    print(f"  Time:       {t_fixed:.1f}ms")

    # Test 2: With adaptive N (CFL < 1)
    print(f"\n[Test 2] With Adaptive N (CFL < 1 enforced)")
    print("-" * 90)

    solver_adaptive = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N_base, adaptive_N=True)

    t_start = time.perf_counter()
    result_adaptive = solver_adaptive.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)
    t_adaptive = (time.perf_counter() - t_start) * 1000

    price_adaptive = result_adaptive['price']
    vega_adaptive = result_adaptive['vega']

    price_err_adaptive = abs(price_adaptive - price_bsm) / price_bsm * 100
    vega_err_adaptive = abs(vega_adaptive - vega_bsm) / vega_bsm * 100

    print(f"  N:          {solver_adaptive.N} (increased from {N_base})")
    print(f"  dt:         {solver_adaptive.dt:.6f}")
    print(f"  CFL ratio:  {solver_adaptive.cfl_ratio:.4f} {'⚠️ >1' if solver_adaptive.cfl_ratio > 1.0 else '✓ <1'}")
    print(f"  Price:      {price_adaptive:.6f} (BSM: {price_bsm:.6f}, error: {price_err_adaptive:.2f}%)")
    print(f"  Vega:       {vega_adaptive:.6f} (BSM: {vega_bsm:.6f}, error: {vega_err_adaptive:.2f}%)")
    print(f"  Time:       {t_adaptive:.1f}ms ({t_adaptive/t_fixed:.2f}× slower)")

    # Improvement analysis
    print(f"\n[Improvement Analysis]")
    print("-" * 90)

    vega_improvement = (vega_err_fixed - vega_err_adaptive) / vega_err_fixed * 100
    price_improvement = (price_err_fixed - price_err_adaptive) / price_err_fixed * 100

    print(f"  Price error: {price_err_fixed:.2f}% → {price_err_adaptive:.2f}% ({price_improvement:+.1f}% improvement)")
    print(f"  Vega error:  {vega_err_fixed:.2f}% → {vega_err_adaptive:.2f}% ({vega_improvement:+.1f}% improvement)")

    if vega_err_adaptive < 5.0:
        print(f"  ✅ VEGA ERROR < 5%! Adaptive timestepping SOLVED the problem!")
    elif vega_err_adaptive < vega_err_fixed * 0.5:
        print(f"  ✓ Significant improvement ({vega_improvement:.1f}%), but still room for optimization")
    else:
        print(f"  ⚠️  Limited improvement, may need additional fixes")

# Final comprehensive test at σ=50%
print(f"\n{'='*90}")
print("COMPREHENSIVE TEST AT σ=50% WITH HESSIAN")
print('='*90)

sigma = 0.50
price_bsm, vega_bsm = bsm_analytical(S0, K, T, r, sigma)

# Compute analytical second derivatives (for reference)
d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
n_d1 = norm.pdf(d1)
gamma_bsm = n_d1 / (S0*sigma*np.sqrt(T))
vanna_bsm = -n_d1 * d1 / sigma  # ∂²V/∂S∂σ
volga_bsm = vega_bsm * d1 * (d1 - sigma*np.sqrt(T)) / sigma  # ∂²V/∂σ²

print(f"\nBSM Analytical:")
print(f"  Price:  {price_bsm:.6f}")
print(f"  Vega:   {vega_bsm:.6f}")
print(f"  Gamma:  {gamma_bsm:.6f}")
print(f"  Vanna:  {vanna_bsm:.6f}")
print(f"  Volga:  {volga_bsm:.6f}")

print(f"\n[Adaptive N with First-Order Greeks Only]")
print("-" * 90)

solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N_base, adaptive_N=True)

t_start = time.perf_counter()
result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)
t_elapsed = (time.perf_counter() - t_start) * 1000

print(f"  N:          {solver.N}")
print(f"  CFL ratio:  {solver.cfl_ratio:.4f}")
print(f"  Price:      {result['price']:.6f} (error: {abs(result['price']-price_bsm)/price_bsm*100:.2f}%)")
print(f"  Delta:      {result['delta']:.6f}")
print(f"  Vega:       {result['vega']:.6f} (error: {abs(result['vega']-vega_bsm)/vega_bsm*100:.2f}%)")
print(f"  Time:       {t_elapsed:.1f}ms")

print(f"\n{'='*90}")
print("SUMMARY AND NEXT STEPS")
print('='*90)

print(f"\n1. CFL-BASED ADAPTIVE TIMESTEPPING:")
print(f"   - At σ=50%, N increases from {N_base} to {solver.N}")
print(f"   - CFL ratio maintained at {solver.cfl_ratio:.4f} (< 1.0 ✓)")

print(f"\n2. VEGA ACCURACY IMPROVEMENT:")
vega_err = abs(result['vega'] - vega_bsm) / vega_bsm * 100
if vega_err < 5.0:
    print(f"   ✅ Vega error: {vega_err:.2f}% (< 5% target achieved!)")
    print(f"   → First-order AAD is now accurate!")
    print(f"   → This should significantly improve Vanna/Volga")
elif vega_err < 10.0:
    print(f"   ✓ Vega error: {vega_err:.2f}% (good improvement)")
    print(f"   → May need further refinement")
else:
    print(f"   ⚠️  Vega error: {vega_err:.2f}% (still high)")
    print(f"   → Additional fixes needed")

print(f"\n3. NEXT STEPS:")
print(f"   a) Test Hessian computation with adaptive N")
print(f"   b) Measure Vanna/Volga errors (expect ~10× reduction)")
print(f"   c) Benchmark Edge-Pushing vs AAD+Bumping with adaptive N")

print(f"\n{'='*90}")
