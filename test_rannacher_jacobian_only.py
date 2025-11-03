"""
Quick test: Rannacher vs Standard CN at σ=50% (Jacobian only)

Tests only first-order derivatives (Delta, Vega) to avoid timeout
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

# Test with coarse grid first (faster)
M = 51
N = 100

print("="*80)
print("RANNACHER vs STANDARD CN: JACOBIAN ONLY (Fast Test)")
print("="*80)
print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r:.2%}, σ={sigma:.1%}")
print(f"Grid: M={M}, N={N} (coarse for speed)")
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

# Test 1: Standard Crank-Nicolson (NO Rannacher)
print("\n" + "-"*80)
print("TEST 1: Standard Crank-Nicolson (Jacobian only)")
print("-"*80)
t_start = time.perf_counter()

solver_cn = BS_PDE_AAD_Rannacher(
    S0=S0, K=K, T=T, r=r, sigma=sigma,
    M=M, N_base=N,
    use_rannacher=False,  # Disable Rannacher
    center_on_S0=False
)

result_cn = solver_cn.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=False,  # Only Jacobian (Delta, Vega)
    verbose=False,
    fixed_grid=True
)

t_end = time.perf_counter()
time_cn = (t_end - t_start) * 1000.0

print(f"  Price:  {result_cn['price']:.6f}")
print(f"  Delta:  {result_cn['delta']:.6f}")
print(f"  Vega:   {result_cn['vega']:.6f}")
print(f"  Time:   {time_cn:.2f}ms")

# Test 2: Rannacher Timestepping (R=4)
print("\n" + "-"*80)
print("TEST 2: Rannacher Timestepping R=4 (Jacobian only)")
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

# Error Analysis
print("\n" + "="*80)
print("ERROR ANALYSIS")
print("="*80)

def calc_error(val, ref):
    """Calculate percentage error"""
    if abs(ref) < 1e-10:
        return 0.0 if abs(val) < 1e-10 else 100.0
    return abs(val - ref) / abs(ref) * 100.0

price_err_cn = calc_error(result_cn['price'], result_bsm['price'])
delta_err_cn = calc_error(result_cn['delta'], result_bsm['greeks']['delta'])
vega_err_cn = calc_error(result_cn['vega'], result_bsm['greeks']['vega'])

price_err_ran = calc_error(result_ran['price'], result_bsm['price'])
delta_err_ran = calc_error(result_ran['delta'], result_bsm['greeks']['delta'])
vega_err_ran = calc_error(result_ran['vega'], result_bsm['greeks']['vega'])

print(f"\n{'Greek':<10} {'BSM':<12} {'CN':<12} {'CN Err %':<12} {'Rannacher':<12} {'Ran Err %':<12} {'Improvement':<12}")
print("-"*90)

greeks_data = [
    ('Price', result_bsm['price'], result_cn['price'], price_err_cn, result_ran['price'], price_err_ran),
    ('Delta', result_bsm['greeks']['delta'], result_cn['delta'], delta_err_cn, result_ran['delta'], delta_err_ran),
    ('Vega', result_bsm['greeks']['vega'], result_cn['vega'], vega_err_cn, result_ran['vega'], vega_err_ran),
]

for name, bsm_val, cn_val, cn_err, ran_val, ran_err in greeks_data:
    if cn_err > 1e-10:
        improvement = (cn_err - ran_err) / cn_err * 100.0
    else:
        improvement = 0.0
    print(f"{name:<10} {bsm_val:>11.6f} {cn_val:>11.6f} {cn_err:>10.2f}% {ran_val:>11.6f} {ran_err:>10.2f}% {improvement:>10.1f}%")

# Now test Gamma using numerical differentiation on the grid
print("\n" + "-"*80)
print("GAMMA ESTIMATION (via finite difference on PDE grid)")
print("-"*80)

# For CN: compute Gamma from grid values
gamma_cn_fd = solver_cn._compute_gamma_on_grid(
    solver_cn._solve_pde_numerical(S0, sigma, fixed_grid=True)[1],
    S0
)

# For Rannacher: compute Gamma from grid values
gamma_ran_fd = solver_ran._compute_gamma_on_grid(
    solver_ran._solve_pde_numerical(S0, sigma, fixed_grid=True)[1],
    S0
)

gamma_err_cn_fd = calc_error(gamma_cn_fd, result_bsm['greeks']['gamma'])
gamma_err_ran_fd = calc_error(gamma_ran_fd, result_bsm['greeks']['gamma'])

print(f"  BSM Gamma:          {result_bsm['greeks']['gamma']:.6f}")
print(f"  CN Gamma (FD):      {gamma_cn_fd:.6f} (Error: {gamma_err_cn_fd:.2f}%)")
print(f"  Rannacher Gamma (FD): {gamma_ran_fd:.6f} (Error: {gamma_err_ran_fd:.2f}%)")

gamma_improvement = (gamma_err_cn_fd - gamma_err_ran_fd) / gamma_err_cn_fd * 100.0 if gamma_err_cn_fd > 1e-10 else 0.0
if gamma_improvement > 10.0:
    print(f"\n  ✅ Rannacher IMPROVES Gamma by {gamma_improvement:.1f}%")
elif gamma_improvement > -10.0:
    print(f"\n  ⚠️  Rannacher has SIMILAR Gamma (±10%)")
else:
    print(f"\n  ❌ Rannacher WORSENS Gamma by {-gamma_improvement:.1f}%")

# Conclusion
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

if delta_err_ran < delta_err_cn * 0.9:
    print("\n✅ Rannacher IMPROVES Delta accuracy")
elif delta_err_ran < delta_err_cn * 1.1:
    print("\n⚠️  Rannacher has SIMILAR Delta accuracy")
else:
    print("\n❌ Rannacher WORSENS Delta accuracy")

if vega_err_ran < vega_err_cn * 0.9:
    print("✅ Rannacher IMPROVES Vega accuracy")
elif vega_err_ran < vega_err_cn * 1.1:
    print("⚠️  Rannacher has SIMILAR Vega accuracy")
else:
    print("❌ Rannacher WORSENS Vega accuracy")

if gamma_err_ran_fd < gamma_err_cn_fd * 0.9:
    print("✅ Rannacher IMPROVES Gamma accuracy")
elif gamma_err_ran_fd < gamma_err_cn_fd * 1.1:
    print("⚠️  Rannacher has SIMILAR Gamma accuracy")
else:
    print("❌ Rannacher WORSENS Gamma accuracy")

print(f"\nComputation time:")
print(f"  CN:        {time_cn:.2f}ms")
print(f"  Rannacher: {time_ran:.2f}ms (+{(time_ran - time_cn) / time_cn * 100:.1f}%)")

print("\nKEY INSIGHT:")
if gamma_err_ran_fd < 10.0:
    print("  ✅ Rannacher successfully smooths payoff kink, Gamma < 10% error")
    print("  → The Vanna/Volga problem is NOT due to timestepping oscillations")
    print("  → Root cause: AAD gradient propagation through implicit solves")
else:
    print("  ❌ Rannacher does NOT solve Gamma problem at σ=50%")
    print("  → May need finer grid or different boundary treatment")

print("="*80)
