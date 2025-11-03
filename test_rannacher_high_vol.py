"""
Test Rannacher timestepping at σ=50% to see if it reduces errors

Compares:
1. Standard Crank-Nicolson (use_rannacher=False)
2. Rannacher timestepping (use_rannacher=True, R=4 steps)
3. BSM Analytical baseline

Focus: Does Rannacher reduce Gamma/Vanna/Volga errors at high volatility?
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
sigma = 0.50  # High volatility where we see problems

# Test with fine grid
M = 101
N = 200

print("="*80)
print("RANNACHER TIMESTEPPING TEST AT HIGH VOLATILITY")
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
print(f"  Vanna:  {result_bsm['greeks']['vanna']:.6f}")
print(f"  Volga:  {result_bsm['greeks']['volga']:.6f}")
print(f"  Time:   {result_bsm['time_ms']:.2f}ms")

# Test 1: Standard Crank-Nicolson (NO Rannacher)
print("\n" + "-"*80)
print("TEST 1: Standard Crank-Nicolson (use_rannacher=False)")
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
    compute_hessian=True,
    verbose=False,
    fixed_grid=True  # Use fixed N for consistent Volga
)

t_end = time.perf_counter()
time_cn = (t_end - t_start) * 1000.0

print(f"  Price:  {result_cn['price']:.6f}")
print(f"  Delta:  {result_cn['delta']:.6f}")
print(f"  Gamma:  {result_cn['gamma']:.6f}")
print(f"  Vega:   {result_cn['vega']:.6f}")
print(f"  Vanna:  {result_cn['vanna']:.6f}")
print(f"  Volga:  {result_cn['volga']:.6f}")
print(f"  Time:   {time_cn:.2f}ms")

# Test 2: Rannacher Timestepping (R=4)
print("\n" + "-"*80)
print("TEST 2: Rannacher Timestepping (use_rannacher=True, R=4)")
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
    compute_hessian=True,
    verbose=False,
    fixed_grid=True  # Use fixed N for consistent Volga
)

t_end = time.perf_counter()
time_ran = (t_end - t_start) * 1000.0

print(f"  Price:  {result_ran['price']:.6f}")
print(f"  Delta:  {result_ran['delta']:.6f}")
print(f"  Gamma:  {result_ran['gamma']:.6f}")
print(f"  Vega:   {result_ran['vega']:.6f}")
print(f"  Vanna:  {result_ran['vanna']:.6f}")
print(f"  Volga:  {result_ran['volga']:.6f}")
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

greeks = ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']
bsm_vals = {
    'price': result_bsm['price'],
    'delta': result_bsm['greeks']['delta'],
    'gamma': result_bsm['greeks']['gamma'],
    'vega': result_bsm['greeks']['vega'],
    'vanna': result_bsm['greeks']['vanna'],
    'volga': result_bsm['greeks']['volga']
}

cn_vals = {
    'price': result_cn['price'],
    'delta': result_cn['delta'],
    'gamma': result_cn['gamma'],
    'vega': result_cn['vega'],
    'vanna': result_cn['vanna'],
    'volga': result_cn['volga']
}

ran_vals = {
    'price': result_ran['price'],
    'delta': result_ran['delta'],
    'gamma': result_ran['gamma'],
    'vega': result_ran['vega'],
    'vanna': result_ran['vanna'],
    'volga': result_ran['volga']
}

print("\nErrors vs BSM Analytical:")
print(f"\n{'Greek':<10} {'BSM':<12} {'CN':<12} {'CN Error':<12} {'Rannacher':<12} {'Ran Error':<12} {'Improvement':<12}")
print("-"*90)

for greek in greeks:
    bsm_val = bsm_vals[greek]
    cn_val = cn_vals[greek]
    ran_val = ran_vals[greek]

    cn_err = calc_error(cn_val, bsm_val)
    ran_err = calc_error(ran_val, bsm_val)

    if cn_err > 1e-10:
        improvement = (cn_err - ran_err) / cn_err * 100.0
    else:
        improvement = 0.0

    print(f"{greek:<10} {bsm_val:>11.6f} {cn_val:>11.6f} {cn_err:>10.2f}% {ran_val:>11.6f} {ran_err:>10.2f}% {improvement:>10.1f}%")

# Numerical stability check
print("\n" + "-"*80)
print("NUMERICAL STABILITY CHECK")
print("-"*80)

# Check if Gamma is positive (must be for call options)
gamma_cn_sign = "✓ POSITIVE" if result_cn['gamma'] > 0 else "✗ NEGATIVE"
gamma_ran_sign = "✓ POSITIVE" if result_ran['gamma'] > 0 else "✗ NEGATIVE"

print(f"  Crank-Nicolson Gamma: {result_cn['gamma']:.6f} {gamma_cn_sign}")
print(f"  Rannacher Gamma:      {result_ran['gamma']:.6f} {gamma_ran_sign}")

# Check if Vanna and Volga have reasonable magnitudes
print(f"\n  Vanna magnitude check:")
print(f"    BSM:              {abs(result_bsm['greeks']['vanna']):.6f}")
print(f"    Crank-Nicolson:   {abs(result_cn['vanna']):.6f} ({calc_error(result_cn['vanna'], result_bsm['greeks']['vanna']):.1f}% error)")
print(f"    Rannacher:        {abs(result_ran['vanna']):.6f} ({calc_error(result_ran['vanna'], result_bsm['greeks']['vanna']):.1f}% error)")

print(f"\n  Volga magnitude check:")
print(f"    BSM:              {result_bsm['greeks']['volga']:.6f}")
print(f"    Crank-Nicolson:   {result_cn['volga']:.6f} ({calc_error(result_cn['volga'], result_bsm['greeks']['volga']):.1f}% error)")
print(f"    Rannacher:        {result_ran['volga']:.6f} ({calc_error(result_ran['volga'], result_bsm['greeks']['volga']):.1f}% error)")

# Conclusion
print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

cn_gamma_err = calc_error(result_cn['gamma'], result_bsm['greeks']['gamma'])
ran_gamma_err = calc_error(result_ran['gamma'], result_bsm['greeks']['gamma'])

cn_vanna_err = calc_error(result_cn['vanna'], result_bsm['greeks']['vanna'])
ran_vanna_err = calc_error(result_ran['vanna'], result_bsm['greeks']['vanna'])

cn_volga_err = calc_error(result_cn['volga'], result_bsm['greeks']['volga'])
ran_volga_err = calc_error(result_ran['volga'], result_bsm['greeks']['volga'])

if ran_gamma_err < cn_gamma_err * 0.9:
    print("\n✅ Rannacher IMPROVES Gamma accuracy by {:.1f}%".format((cn_gamma_err - ran_gamma_err) / cn_gamma_err * 100))
elif ran_gamma_err < cn_gamma_err * 1.1:
    print("\n⚠️  Rannacher has SIMILAR Gamma accuracy (±10%)")
else:
    print("\n❌ Rannacher has WORSE Gamma accuracy")

if ran_vanna_err < cn_vanna_err * 0.9:
    print("✅ Rannacher IMPROVES Vanna accuracy by {:.1f}%".format((cn_vanna_err - ran_vanna_err) / cn_vanna_err * 100))
elif ran_vanna_err < cn_vanna_err * 1.1:
    print("⚠️  Rannacher has SIMILAR Vanna accuracy (±10%)")
else:
    print("❌ Rannacher has WORSE Vanna accuracy")

if ran_volga_err < cn_volga_err * 0.9:
    print("✅ Rannacher IMPROVES Volga accuracy by {:.1f}%".format((cn_volga_err - ran_volga_err) / cn_volga_err * 100))
elif ran_volga_err < cn_volga_err * 1.1:
    print("⚠️  Rannacher has SIMILAR Volga accuracy (±10%)")
else:
    print("❌ Rannacher has WORSE Volga accuracy")

print("\nComputational cost:")
print(f"  Crank-Nicolson:  {time_cn:.2f}ms")
print(f"  Rannacher:       {time_ran:.2f}ms (overhead: {(time_ran - time_cn) / time_cn * 100:.1f}%)")

print("\nRecommendation:")
if ran_gamma_err < 5.0 and ran_vanna_err < 50.0:
    print("  ✅ USE Rannacher for high volatility (σ≥40%)")
    print("     Gamma and Vanna errors acceptable for production")
elif ran_gamma_err < cn_gamma_err * 0.8:
    print("  ⚠️  Rannacher improves Gamma but Vanna/Volga still high")
    print("     Consider for pricing, but not for Vanna/Volga hedging")
else:
    print("  ❌ Rannacher does NOT solve high volatility problem")
    print("     Root cause is likely in AAD gradient propagation, not timestepping")

print("="*80)
