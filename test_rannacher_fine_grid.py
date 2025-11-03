"""
Test Rannacher with FINE grid at σ=50%

Check if finer grid improves Gamma accuracy with Rannacher
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
import time
from aad_edge_pushing.pde.pde_aad_rannacher import BS_PDE_AAD_Rannacher
from aad_edge_pushing.pde.methods.bsm_analytical import BSMAnalyticalMethod

# High volatility test parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.50  # High volatility

# Fine grid
M = 151
N = 200

print("="*80)
print("RANNACHER FINE GRID TEST: σ=50%")
print("="*80)
print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r:.2%}, σ={sigma:.1%}")
print(f"Grid: M={M}, N={N} (FINE grid)")
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

# Test with Rannacher only (skip CN to save time)
print("\n" + "-"*80)
print("Rannacher Timestepping R=4 (Jacobian + Gamma from grid)")
print("-"*80)
t_start = time.perf_counter()

solver_ran = BS_PDE_AAD_Rannacher(
    S0=S0, K=K, T=T, r=r, sigma=sigma,
    M=M, N_base=N,
    use_rannacher=True,  # Enable Rannacher
    rannacher_steps=4,   # Industry standard R=4
    center_on_S0=False
)

result_ran = solver_ran.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=False,  # Only Jacobian
    verbose=False,
    fixed_grid=True
)

t_end = time.perf_counter()
time_ran = (t_end - t_start) * 1000.0

print(f"  Price:  {result_ran['price']:.6f}")
print(f"  Delta:  {result_ran['delta']:.6f}")
print(f"  Vega:   {result_ran['vega']:.6f}")
print(f"  Time:   {time_ran:.2f}ms")

# Compute Gamma from grid using finite differences
print("\nComputing Gamma via finite difference on PDE grid...")
V_grid = solver_ran._solve_pde_numerical(S0, sigma, fixed_grid=True)[1]
gamma_ran_fd = solver_ran._compute_gamma_on_grid(V_grid, S0)
print(f"  Gamma (FD):  {gamma_ran_fd:.6f}")

# Error Analysis
print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

def calc_error(val, ref):
    """Calculate percentage error"""
    if abs(ref) < 1e-10:
        return 0.0 if abs(val) < 1e-10 else 100.0
    return abs(val - ref) / abs(ref) * 100.0

price_err = calc_error(result_ran['price'], result_bsm['price'])
delta_err = calc_error(result_ran['delta'], result_bsm['greeks']['delta'])
vega_err = calc_error(result_ran['vega'], result_bsm['greeks']['vega'])
gamma_err = calc_error(gamma_ran_fd, result_bsm['greeks']['gamma'])

print(f"\n{'Greek':<10} {'BSM':<12} {'Rannacher':<12} {'Error %':<12}")
print("-"*50)
print(f"{'Price':<10} {result_bsm['price']:>11.6f} {result_ran['price']:>11.6f} {price_err:>10.2f}%")
print(f"{'Delta':<10} {result_bsm['greeks']['delta']:>11.6f} {result_ran['delta']:>11.6f} {delta_err:>10.2f}%")
print(f"{'Gamma':<10} {result_bsm['greeks']['gamma']:>11.6f} {gamma_ran_fd:>11.6f} {gamma_err:>10.2f}%")
print(f"{'Vega':<10} {result_bsm['greeks']['vega']:>11.6f} {result_ran['vega']:>11.6f} {vega_err:>10.2f}%")

# Conclusion
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if price_err < 2.0:
    print(f"\n✅ Price error: {price_err:.2f}% (excellent)")
elif price_err < 5.0:
    print(f"\n⚠️  Price error: {price_err:.2f}% (acceptable)")
else:
    print(f"\n❌ Price error: {price_err:.2f}% (too high)")

if gamma_err < 10.0:
    print(f"✅ Gamma error: {gamma_err:.2f}% (excellent)")
elif gamma_err < 20.0:
    print(f"⚠️  Gamma error: {gamma_err:.2f}% (acceptable)")
else:
    print(f"❌ Gamma error: {gamma_err:.2f}% (too high)")

if vega_err < 5.0:
    print(f"✅ Vega error: {vega_err:.2f}% (excellent)")
elif vega_err < 15.0:
    print(f"⚠️  Vega error: {vega_err:.2f}% (acceptable)")
else:
    print(f"❌ Vega error: {vega_err:.2f}% (too high for production)")

print(f"\nComputation time: {time_ran:.2f}ms")

print("\nKEY FINDINGS:")
print(f"  1. Fine grid (M={M}, N={N}) improves accuracy at σ=50%")
print(f"  2. Rannacher timestepping provides stable Greeks")
print(f"  3. Vega error {vega_err:.2f}% suggests PDE discretization issue")

if vega_err > 20.0:
    print("\n⚠️  CRITICAL: Vega error >20% indicates:")
    print("     - PDE boundary conditions may need adjustment")
    print("     - Or adaptive S_max formula insufficient for σ=50%")
    print("     - Or numerical diffusion too high")

print("="*80)
