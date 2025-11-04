"""
Test S_max expansion from 3σ to 5σ to fix Vega errors at high volatility

Expected results:
- At σ=50%, S_max increases from 471 → 1218
- Price error: 2.88% → <1%
- Vega error: 24.24% → <5%
- Vanna/Volga should also improve significantly
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
import time
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

def bsm_analytical_full(S, K, T, r, sigma):
    """BSM analytical solution with all Greeks"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    price = S*N_d1 - K*np.exp(-r*T)*N_d2
    delta = N_d1
    gamma = n_d1 / (S*sigma*np.sqrt(T))
    vega = S*n_d1*np.sqrt(T)
    vanna = -n_d1 * d1 / sigma
    volga = vega * d1 * (d2) / sigma

    return {
        'price': price, 'delta': delta, 'gamma': gamma,
        'vega': vega, 'vanna': vanna, 'volga': volga
    }

# Test parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05

print("=" * 90)
print("S_MAX EXPANSION TEST: From 3σ to 5σ at High Volatility")
print("=" * 90)

# Test at multiple volatilities
sigmas = [0.10, 0.20, 0.30, 0.40, 0.50]
M, N_base = 101, 200

print(f"\nGrid: M={M}, N_base={N_base} (adaptive N enabled)")
print(f"\n{'σ':<8} {'S_max':<10} {'BSM Vega':<12} {'PDE Vega':<12} {'BSM Price':<12} {'PDE Price':<12} {'Vega Err%':<12} {'Price Err%':<12}")
print("-" * 100)

for sigma in sigmas:
    # BSM analytical
    bsm = bsm_analytical_full(S0, K, T, r, sigma)

    # PDE solution with expanded S_max
    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N_base, adaptive_N=True)

    # Calculate actual S_max used
    S_max_actual = max(5.0 * K, S0 * np.exp((r + 5*sigma) * T))

    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)

    price_pde = result['price']
    vega_pde = result['vega']

    # Errors
    vega_err = abs(vega_pde - bsm['vega']) / bsm['vega'] * 100
    price_err = abs(price_pde - bsm['price']) / bsm['price'] * 100

    print(f"{sigma:<8.2f} {S_max_actual:<10.1f} {bsm['vega']:<12.4f} {vega_pde:<12.4f} "
          f"{bsm['price']:<12.4f} {price_pde:<12.4f} {vega_err:<12.2f} {price_err:<12.2f}")

# Detailed analysis at σ=50%
print(f"\n{'='*90}")
print("DETAILED ANALYSIS AT σ=50%")
print('='*90)

sigma = 0.50
bsm = bsm_analytical_full(S0, K, T, r, sigma)

print(f"\nBSM Analytical:")
print(f"  Price:  {bsm['price']:.6f}")
print(f"  Delta:  {bsm['delta']:.6f}")
print(f"  Gamma:  {bsm['gamma']:.6f}")
print(f"  Vega:   {bsm['vega']:.6f}")
print(f"  Vanna:  {bsm['vanna']:.6f}")
print(f"  Volga:  {bsm['volga']:.6f}")

solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N_base, adaptive_N=True)

print(f"\nSolver Configuration:")
S_max_new = max(5.0 * K, S0 * np.exp((r + 5*sigma) * T))
S_max_old = max(3.0 * K, S0 * np.exp((r + 3*sigma) * T))
print(f"  S_max (old 3σ): {S_max_old:.1f}")
print(f"  S_max (new 5σ): {S_max_new:.1f}")
print(f"  Expansion:      {S_max_new/S_max_old:.2f}×")
print(f"  N (adaptive):   {solver.N}")
print(f"  CFL ratio:      {solver.cfl_ratio:.4f}")

print(f"\n[Test 1] First-Order Greeks (PDE with Expanded S_max)")
print("-" * 90)

t_start = time.perf_counter()
result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)
t_elapsed = (time.perf_counter() - t_start) * 1000

price_pde = result['price']
delta_pde = result['delta']
vega_pde = result['vega']

price_err = abs(price_pde - bsm['price']) / bsm['price'] * 100
delta_err = abs(delta_pde - bsm['delta']) / bsm['delta'] * 100
vega_err = abs(vega_pde - bsm['vega']) / bsm['vega'] * 100

print(f"  Price:  {price_pde:.6f} (error: {price_err:.2f}%)")
print(f"  Delta:  {delta_pde:.6f} (error: {delta_err:.2f}%)")
print(f"  Vega:   {vega_pde:.6f} (error: {vega_err:.2f}%)")
print(f"  Time:   {t_elapsed:.1f}ms")

print(f"\n[Test 2] Second-Order Greeks (Hessian via Edge-Pushing)")
print("-" * 90)

t_start = time.perf_counter()
result_hessian = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=False)
t_hessian = (time.perf_counter() - t_start) * 1000

gamma_pde = result_hessian['gamma']
vanna_pde = result_hessian['vanna']
volga_pde = result_hessian['volga']

gamma_err = abs(gamma_pde - bsm['gamma']) / bsm['gamma'] * 100
vanna_err = abs(vanna_pde - bsm['vanna']) / abs(bsm['vanna']) * 100
volga_err = abs(volga_pde - bsm['volga']) / abs(bsm['volga']) * 100

print(f"  Gamma:  {gamma_pde:.6f} (error: {gamma_err:.2f}%)")
print(f"  Vanna:  {vanna_pde:.6f} (error: {vanna_err:.2f}%)")
print(f"  Volga:  {volga_pde:.6f} (error: {volga_err:.2f}%)")
print(f"  Time:   {t_hessian:.1f}ms")

# Comparison with old S_max (3σ)
print(f"\n{'='*90}")
print("IMPROVEMENT ANALYSIS: Old (3σ) vs New (5σ)")
print('='*90)

# These are the old results from previous tests at σ=50%, M=101, N=200
old_price_err = 2.88
old_vega_err = 24.24
old_gamma_err = 1.83  # AAD+Bumping was best at this
old_vanna_err = 18.70  # AAD+Bumping
old_volga_err = 303.12  # AAD+Bumping

print(f"\n{'Greek':<10} {'Old Err% (3σ)':<18} {'New Err% (5σ)':<18} {'Improvement':<15}")
print("-" * 70)

improvements = [
    ('Price', old_price_err, price_err),
    ('Vega', old_vega_err, vega_err),
    ('Gamma', old_gamma_err, gamma_err),
    ('Vanna', old_vanna_err, vanna_err),
    ('Volga', old_volga_err, volga_err),
]

for greek, old_err, new_err in improvements:
    if old_err > 0:
        improvement = (old_err - new_err) / old_err * 100
        improvement_str = f"{improvement:+.1f}%"
    else:
        improvement_str = "N/A"

    print(f"{greek:<10} {old_err:>16.2f}% {new_err:>16.2f}% {improvement_str:>14}")

# Success criteria
print(f"\n{'='*90}")
print("SUCCESS CRITERIA")
print('='*90)

success_count = 0
total_tests = 4

print(f"\n✓ indicates target met:")
print(f"  1. Price error < 1%:  {price_err:.2f}% {'✓' if price_err < 1.0 else '✗'}")
if price_err < 1.0:
    success_count += 1

print(f"  2. Vega error < 5%:   {vega_err:.2f}% {'✓' if vega_err < 5.0 else '✗'}")
if vega_err < 5.0:
    success_count += 1

print(f"  3. Gamma error < 5%:  {gamma_err:.2f}% {'✓' if gamma_err < 5.0 else '✗'}")
if gamma_err < 5.0:
    success_count += 1

print(f"  4. Vanna error < 50%: {vanna_err:.2f}% {'✓' if vanna_err < 50.0 else '✗'}")
if vanna_err < 50.0:
    success_count += 1

print(f"\nOverall: {success_count}/{total_tests} targets met")

if success_count == total_tests:
    print(f"\n🎉 ALL TARGETS MET! S_max expansion successfully fixed high volatility errors!")
elif success_count >= 2:
    print(f"\n✓ Significant improvement, but some targets missed")
else:
    print(f"\n⚠️  Limited improvement, further investigation needed")

print(f"\n{'='*90}")
print("CONCLUSION")
print('='*90)

print(f"\nS_max Expansion Impact:")
print(f"  - Domain increased from 3σ to 5σ")
print(f"  - S_max: {S_max_old:.0f} → {S_max_new:.0f} ({S_max_new/S_max_old:.2f}× larger)")
print(f"  - Better captures option tail probability at high volatility")

if vega_err < 5.0:
    print(f"\n✅ First-order Vega is now accurate ({vega_err:.2f}% < 5%)")
    print(f"   → This validates the hypothesis that S_max truncation was the root cause")

    if vanna_err < 50.0 and volga_err < 200.0:
        print(f"\n✅ Second-order Greeks also improved significantly!")
        print(f"   → Edge-Pushing is now competitive for all Greeks at high volatility")
    else:
        print(f"\n⚠️  Second-order Greeks still have errors (Vanna: {vanna_err:.1f}%, Volga: {volga_err:.1f}%)")
        print(f"   → May need additional refinements for Hessian computation")
else:
    print(f"\n⚠️  Vega error still high ({vega_err:.2f}%)")
    print(f"   → S_max expansion alone is not sufficient")
    print(f"   → Need to investigate other error sources:")
    print(f"     1. Boundary condition accuracy")
    print(f"     2. Grid resolution (try larger M)")
    print(f"     3. Time discretization (already using adaptive N)")

print(f"\n{'='*90}")
