"""
Quick test to verify high volatility fix
Test σ=50% which previously had 17.76% price error
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.pde.methods.bumping2 import Bumping2Method
from aad_edge_pushing.pde.methods.bsm_analytical import BSMAnalyticalMethod

# Test parameters (same as failing case)
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.50  # High volatility that caused the problem
M = 51
N = 100

print("="*70)
print("HIGH VOLATILITY FIX VERIFICATION TEST")
print("="*70)
print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma:.1%}")
print(f"Grid: M={M}, N={N}")

# Test Bumping2 (uses SimplePDESolver)
print("\n" + "-"*70)
print("Testing Bumping2 (numerical PDE)...")
bumping2 = Bumping2Method(M=M, N=N, S0=S0, K=K, T=T, r=r)
result_bumping = bumping2.compute_hessian(S0, sigma)

print(f"  Price: {result_bumping['price']:.5f}")
print(f"  Gamma: {result_bumping['greeks']['gamma']:.6f}")

# Test BSM Analytical
print("\n" + "-"*70)
print("Testing BSM Analytical (baseline)...")
bsm = BSMAnalyticalMethod(M=M, N=N, S0=S0, K=K, T=T, r=r)
result_bsm = bsm.compute_hessian(S0, sigma)

print(f"  Price: {result_bsm['price']:.5f}")
print(f"  Gamma: {result_bsm['greeks']['gamma']:.6f}")

# Calculate errors
price_error = abs(result_bumping['price'] - result_bsm['price']) / result_bsm['price'] * 100
gamma_error = abs(result_bumping['greeks']['gamma'] - result_bsm['greeks']['gamma']) / abs(result_bsm['greeks']['gamma']) * 100

print("\n" + "="*70)
print("RESULTS")
print("="*70)
print(f"Price Error: {price_error:.2f}% (was 17.76% before fix)")
print(f"Gamma Error: {gamma_error:.2f}% (was 104.84% before fix)")
print(f"Gamma Sign: {'POSITIVE ✓' if result_bumping['greeks']['gamma'] > 0 else 'NEGATIVE ✗'} (was negative before fix)")

print("\n" + "-"*70)
if price_error < 1.0 and result_bumping['greeks']['gamma'] > 0:
    print("✅ FIX SUCCESSFUL! Errors < 1% and Gamma is positive!")
elif price_error < 5.0:
    print("⚠️  Improved but not perfect. Errors < 5%")
else:
    print("❌ FIX FAILED. Errors still too high.")
print("="*70)
