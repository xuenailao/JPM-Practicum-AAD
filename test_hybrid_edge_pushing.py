"""
Test Hybrid Edge-Pushing Method at High Volatility

Compares:
1. Original Edge-Pushing (σ as ADVar → 435% Vanna error)
2. Hybrid Edge-Pushing (σ as scalar + bumping → expected <10% Vanna error)
3. AAD+Bumping (baseline hybrid method → 1.36% Vanna error)
4. BSM Analytical (ground truth)

Key Question: Does Hybrid Edge-Pushing maintain Edge-Pushing's efficiency
while achieving AAD+Bumping's accuracy?
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
from aad_edge_pushing.pde.methods.edge_pushing_hybrid import HybridEdgePushingMethod
from aad_edge_pushing.pde.methods.edge_pushing import EdgePushingMethod
from aad_edge_pushing.pde.methods.aad_bumping import AADBumpingMethod
from aad_edge_pushing.pde.methods.bsm_analytical import BSMAnalyticalMethod

# High volatility test parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.50  # High volatility where original Edge-Pushing fails

# Test with coarse grid first
M = 51
N = 100

print("="*90)
print("HYBRID EDGE-PUSHING vs ORIGINAL: Stability Test at σ=50%")
print("="*90)
print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r:.2%}, σ={sigma:.1%}")
print(f"Grid: M={M}, N={N}")
print()

# Baseline: BSM Analytical
print("-"*90)
print("BSM ANALYTICAL (Ground Truth)")
print("-"*90)
bsm = BSMAnalyticalMethod(M=M, N=N, S0=S0, K=K, T=T, r=r)
result_bsm = bsm.compute_hessian(S0, sigma)

print(f"  Price:  {result_bsm['price']:.6f}")
print(f"  Delta:  {result_bsm['greeks']['delta']:.6f}")
print(f"  Gamma:  {result_bsm['greeks']['gamma']:.6f}")
print(f"  Vega:   {result_bsm['greeks']['vega']:.6f}")
print(f"  Vanna:  {result_bsm['greeks']['vanna']:.6f}")
print(f"  Volga:  {result_bsm['greeks']['volga']:.6f}")
print(f"  Time:   {result_bsm['time_ms']:.2f}ms")

# Test 1: Original Edge-Pushing (known to fail)
print("\n" + "-"*90)
print("TEST 1: Original Edge-Pushing (σ as ADVar)")
print("-"*90)
print("  Computing...")

try:
    edge_original = EdgePushingMethod(M=M, N=N, S0=S0, K=K, T=T, r=r)
    result_original = edge_original.compute_hessian(S0, sigma)

    print(f"  Price:  {result_original['price']:.6f}")
    print(f"  Delta:  {result_original['greeks']['delta']:.6f}")
    print(f"  Gamma:  {result_original['greeks']['gamma']:.6f}")
    print(f"  Vega:   {result_original['greeks']['vega']:.6f}")
    print(f"  Vanna:  {result_original['greeks']['vanna']:.6f}")
    print(f"  Volga:  {result_original['greeks']['volga']:.6f}")
    print(f"  Time:   {result_original['time_ms']:.2f}ms")
except Exception as e:
    print(f"  ❌ Failed: {e}")
    result_original = None

# Test 2: Hybrid Edge-Pushing (new method)
print("\n" + "-"*90)
print("TEST 2: Hybrid Edge-Pushing (σ as scalar + bumping)")
print("-"*90)
print("  Computing...")

hybrid = HybridEdgePushingMethod(M=M, N=N, S0=S0, K=K, T=T, r=r)
result_hybrid = hybrid.compute_hessian(S0, sigma)

print(f"  Price:  {result_hybrid['price']:.6f}")
print(f"  Delta:  {result_hybrid['greeks']['delta']:.6f}")
print(f"  Gamma:  {result_hybrid['greeks']['gamma']:.6f}")
print(f"  Vega:   {result_hybrid['greeks']['vega']:.6f}")
print(f"  Vanna:  {result_hybrid['greeks']['vanna']:.6f}")
print(f"  Volga:  {result_hybrid['greeks']['volga']:.6f}")
print(f"  Time:   {result_hybrid['time_ms']:.2f}ms")
print(f"  eps_σ:  {result_hybrid['eps_sigma_used']:.4f} (adaptive)")

# Test 3: AAD+Bumping (baseline hybrid)
print("\n" + "-"*90)
print("TEST 3: AAD+Bumping (baseline hybrid method)")
print("-"*90)
print("  Computing...")

aad_bump = AADBumpingMethod(M=M, N=N, S0=S0, K=K, T=T, r=r)
result_aad_bump = aad_bump.compute_hessian(S0, sigma)

print(f"  Price:  {result_aad_bump['price']:.6f}")
print(f"  Delta:  {result_aad_bump['greeks']['delta']:.6f}")
print(f"  Gamma:  {result_aad_bump['greeks']['gamma']:.6f}")
print(f"  Vega:   {result_aad_bump['greeks']['vega']:.6f}")
print(f"  Vanna:  {result_aad_bump['greeks']['vanna']:.6f}")
print(f"  Volga:  {result_aad_bump['greeks']['volga']:.6f}")
print(f"  Time:   {result_aad_bump['time_ms']:.2f}ms")

# Error Analysis
print("\n" + "="*90)
print("ERROR ANALYSIS (vs BSM Analytical)")
print("="*90)

def calc_error(val, ref):
    """Calculate percentage error"""
    if abs(ref) < 1e-10:
        return 0.0 if abs(val) < 1e-10 else 100.0
    return abs(val - ref) / abs(ref) * 100.0

greeks = ['price', 'delta', 'gamma', 'vega', 'vanna', 'volga']

print(f"\n{'Greek':<10} {'BSM':<12} {'Original-EP':<15} {'Hybrid-EP':<15} {'AAD+Bump':<15}")
print("-"*67 + " Errors (%) " + "-"*10)

for greek in greeks:
    bsm_val = result_bsm['price'] if greek == 'price' else result_bsm['greeks'][greek]

    if result_original:
        orig_val = result_original['price'] if greek == 'price' else result_original['greeks'][greek]
        orig_err = calc_error(orig_val, bsm_val)
        orig_str = f"{orig_err:>13.2f}%"
    else:
        orig_str = "N/A"

    hybrid_val = result_hybrid['price'] if greek == 'price' else result_hybrid['greeks'][greek]
    hybrid_err = calc_error(hybrid_val, bsm_val)

    aad_val = result_aad_bump['price'] if greek == 'price' else result_aad_bump['greeks'][greek]
    aad_err = calc_error(aad_val, bsm_val)

    print(f"{greek.capitalize():<10} {bsm_val:>11.6f} {orig_str:>14} {hybrid_err:>13.2f}% {aad_err:>13.2f}%")

# Vanna/Volga Focus
print("\n" + "="*90)
print("VANNA & VOLGA ACCURACY COMPARISON")
print("="*90)

vanna_hybrid_err = calc_error(result_hybrid['greeks']['vanna'], result_bsm['greeks']['vanna'])
volga_hybrid_err = calc_error(result_hybrid['greeks']['volga'], result_bsm['greeks']['volga'])

vanna_aad_err = calc_error(result_aad_bump['greeks']['vanna'], result_bsm['greeks']['vanna'])
volga_aad_err = calc_error(result_aad_bump['greeks']['volga'], result_bsm['greeks']['volga'])

if result_original:
    vanna_orig_err = calc_error(result_original['greeks']['vanna'], result_bsm['greeks']['vanna'])
    volga_orig_err = calc_error(result_original['greeks']['volga'], result_bsm['greeks']['volga'])
    print(f"\nOriginal Edge-Pushing:")
    print(f"  Vanna Error:  {vanna_orig_err:>6.2f}%  (EXPECTED: ~435%)")
    print(f"  Volga Error:  {volga_orig_err:>6.2f}%  (EXPECTED: ~2969%)")

print(f"\nHybrid Edge-Pushing:")
print(f"  Vanna Error:  {vanna_hybrid_err:>6.2f}%  (TARGET: <10%)")
print(f"  Volga Error:  {volga_hybrid_err:>6.2f}%  (TARGET: <100%)")

print(f"\nAAD+Bumping (baseline):")
print(f"  Vanna Error:  {vanna_aad_err:>6.2f}%")
print(f"  Volga Error:  {volga_aad_err:>6.2f}%")

# Performance Comparison
print("\n" + "="*90)
print("COMPUTATIONAL EFFICIENCY")
print("="*90)

if result_original:
    print(f"\nOriginal Edge-Pushing: {result_original['time_ms']:>10.2f}ms  ({result_original['n_pde_solves']} PDE solves)")
print(f"Hybrid Edge-Pushing:   {result_hybrid['time_ms']:>10.2f}ms  ({result_hybrid['n_pde_solves']} PDE solves)")
print(f"AAD+Bumping:           {result_aad_bump['time_ms']:>10.2f}ms  ({result_aad_bump['n_pde_solves']} PDE solves)")

# Conclusion
print("\n" + "="*90)
print("CONCLUSION")
print("="*90)

if vanna_hybrid_err < 10.0 and volga_hybrid_err < 100.0:
    print("\n✅ SUCCESS: Hybrid Edge-Pushing achieves target accuracy!")
    print(f"   Vanna: {vanna_hybrid_err:.2f}% (<10% target)")
    print(f"   Volga: {volga_hybrid_err:.2f}% (<100% target)")
elif vanna_hybrid_err < 50.0:
    print(f"\n⚠️  IMPROVED: Hybrid Edge-Pushing reduces errors significantly")
    print(f"   Vanna: {vanna_hybrid_err:.2f}% (vs ~435% original)")
    print(f"   Volga: {volga_hybrid_err:.2f}% (vs ~2969% original)")
else:
    print(f"\n❌ PARTIAL: Hybrid Edge-Pushing still has high errors")
    print(f"   Vanna: {vanna_hybrid_err:.2f}%")
    print(f"   Volga: {volga_hybrid_err:.2f}%")

print("\nKEY INSIGHT:")
print("  By treating σ as scalar instead of ADVar in the PDE solve:")
print("  - Avoids gradient propagation through 100 implicit solves")
print("  - Maintains Edge-Pushing efficiency for Gamma computation")
print("  - Uses adaptive bumping for stable Vega/Vanna/Volga")

if vanna_hybrid_err < vanna_aad_err * 1.2:
    print("\n✅ Hybrid Edge-Pushing matches AAD+Bumping accuracy!")
    print("   → Validates the hybrid approach for Edge-Pushing")

print("="*90)
