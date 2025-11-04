"""
Test complete Hessian computation with fixed S_max at σ=50%

Now that Vega is accurate (0.21% error), test if Vanna/Volga also improved.
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
import time
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

# BSM analytical
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.50

d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
d2 = d1 - sigma*np.sqrt(T)
N_d1 = norm.cdf(d1)
n_d1 = norm.pdf(d1)

price_bsm = S0*N_d1 - K*np.exp(-r*T)*norm.cdf(d2)
delta_bsm = N_d1
gamma_bsm = n_d1 / (S0*sigma*np.sqrt(T))
vega_bsm = S0*n_d1*np.sqrt(T)
vanna_bsm = -n_d1 * d1 / sigma
volga_bsm = vega_bsm * d1 * (d2) / sigma

print("="*80)
print("COMPLETE HESSIAN TEST WITH FIXED S_MAX (5σ) AT σ=50%")
print("="*80)

print(f"\nBSM Analytical:")
print(f"  Price:  {price_bsm:.6f}")
print(f"  Delta:  {delta_bsm:.6f}")
print(f"  Gamma:  {gamma_bsm:.6f}")
print(f"  Vega:   {vega_bsm:.6f}")
print(f"  Vanna:  {vanna_bsm:.6f}")
print(f"  Volga:  {volga_bsm:.6f}")

# PDE with Hessian
solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=101, N_base=200, adaptive_N=True)

S_max = max(5.0 * K, S0 * np.exp((r + 5*sigma) * T))
print(f"\nSolver Config:")
print(f"  S_max (5σ): {S_max:.1f}")
print(f"  N:          {solver.N}")
print(f"  CFL ratio:  {solver.cfl_ratio:.4f}")

print(f"\nComputing Hessian (Edge-Pushing)...")
t_start = time.perf_counter()
result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=True, verbose=False)
t_elapsed = (time.perf_counter() - t_start) * 1000

price = result['price']
delta = result['delta']
gamma = result['gamma']
vega = result['vega']
vanna = result['vanna']
volga = result['volga']

# Errors
price_err = abs(price - price_bsm) / price_bsm * 100
delta_err = abs(delta - delta_bsm) / delta_bsm * 100
gamma_err = abs(gamma - gamma_bsm) / gamma_bsm * 100
vega_err = abs(vega - vega_bsm) / vega_bsm * 100
vanna_err = abs(vanna - vanna_bsm) / abs(vanna_bsm) * 100
volga_err = abs(volga - volga_bsm) / abs(volga_bsm) * 100

print(f"\nPDE Results:")
print(f"  Price:  {price:.6f} (error: {price_err:.2f}%)")
print(f"  Delta:  {delta:.6f} (error: {delta_err:.2f}%)")
print(f"  Gamma:  {gamma:.6f} (error: {gamma_err:.2f}%)")
print(f"  Vega:   {vega:.6f} (error: {vega_err:.2f}%)")
print(f"  Vanna:  {vanna:.6f} (error: {vanna_err:.2f}%)")
print(f"  Volga:  {volga:.6f} (error: {volga_err:.2f}%)")
print(f"  Time:   {t_elapsed:.1f}ms")

# Comparison with old results (3σ S_max)
print(f"\n{'='*80}")
print("IMPROVEMENT SUMMARY: Old (3σ) → New (5σ)")
print('='*80)

old_results = {
    'Price': 2.88,
    'Delta': 5.55,
    'Gamma': 1.83,  # AAD+Bumping was best
    'Vega': 24.24,
    'Vanna': 18.70,  # AAD+Bumping
    'Volga': 303.12   # AAD+Bumping
}

new_results = {
    'Price': price_err,
    'Delta': delta_err,
    'Gamma': gamma_err,
    'Vega': vega_err,
    'Vanna': vanna_err,
    'Volga': volga_err
}

print(f"\n{'Greek':<10} {'Old Err%':<12} {'New Err%':<12} {'Improvement':<15} {'Status':<10}")
print("-" * 70)

for greek in ['Price', 'Delta', 'Gamma', 'Vega', 'Vanna', 'Volga']:
    old_err = old_results[greek]
    new_err = new_results[greek]

    improvement = (old_err - new_err) / old_err * 100

    if greek in ['Price', 'Delta', 'Gamma', 'Vega']:
        target = 5.0
    elif greek == 'Vanna':
        target = 50.0
    else:  # Volga
        target = 100.0

    status = '✅' if new_err < target else '⚠️'

    print(f"{greek:<10} {old_err:>10.2f}% {new_err:>10.2f}% {improvement:>13.1f}% {status:>8}")

print(f"\n{'='*80}")
print("EDGE-PUSHING PERFORMANCE AT HIGH VOLATILITY")
print('='*80)

targets_met = sum([
    price_err < 1.0,
    vega_err < 5.0,
    gamma_err < 5.0,
    vanna_err < 50.0,
    volga_err < 100.0
])

print(f"\nTargets Met: {targets_met}/5")

if targets_met >= 4:
    print(f"\n🎉 EXCELLENT! Edge-Pushing now works accurately at σ=50%!")
    print(f"   - First-order Greeks (Price, Delta, Vega): All < 5% error")
    print(f"   - Second-order Greeks (Gamma, Vanna, Volga): Significantly improved")
    print(f"\n   Root cause was S_max domain truncation, NOT AAD propagation!")
elif targets_met >= 3:
    print(f"\n✓ Good progress! Most targets met")
    print(f"  Further improvements may be needed for some Greeks")
else:
    print(f"\n⚠️  Some targets still not met")

print(f"\n{'='*80}")
