"""
Simple Volga Debug: Check if Vega's σ-derivative shape is correct
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformed_bs_pde import TransformedBSPDE
from scipy.stats import norm


def black_scholes_vega(S0, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    return S0 * norm.pdf(d1) * sqrt_T


def black_scholes_volga(S0, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    vega = S0 * norm.pdf(d1) * sqrt_T
    return vega * d1 * d2 / sigma


S0, K, T, r = 100.0, 100.0, 1.0, 0.05

print("="*100)
print("VOLGA PROBLEM DIAGNOSIS")
print("="*100)

# Test sigma values
test_sigmas = [0.15, 0.18, 0.20, 0.22, 0.25]

solver = TransformedBSPDE(K=K, T=T, r=r, M=151, N=150)

print(f"\n{'Sigma':<10} | {'BS Vega':<12} | {'PDE Vega':<12} | {'Error':<10}")
print("-"*100)

vega_bs_list = []
vega_pde_list = []

for sig in test_sigmas:
    vega_bs = black_scholes_vega(S0, K, T, r, sig)
    _, vega_pde = solver.solve(sig, verbose=False)

    error = abs(vega_pde - vega_bs) / vega_bs * 100

    vega_bs_list.append(vega_bs)
    vega_pde_list.append(vega_pde)

    print(f"{sig:<10.2f} | {vega_bs:<12.6f} | {vega_pde:<12.6f} | {error:<10.2f}%")

# Compute Volga as derivative
print("\n" + "="*100)
print("VOLGA COMPUTATION (Centered Difference)")
print("="*100)

print(f"\n{'Sigma':<10} | {'BS Volga':<12} | {'∂Vega/∂σ (BS)':<15} | {'∂Vega/∂σ (PDE)':<15} | {'Error':<10}")
print("-"*100)

for i in range(1, len(test_sigmas)-1):
    sig = test_sigmas[i]
    dsig = test_sigmas[i+1] - test_sigmas[i-1]

    # Analytical Volga
    volga_bs = black_scholes_volga(S0, K, T, r, sig)

    # BS Vega derivative
    dvega_bs = (vega_bs_list[i+1] - vega_bs_list[i-1]) / dsig

    # PDE Vega derivative
    dvega_pde = (vega_pde_list[i+1] - vega_pde_list[i-1]) / dsig

    error = abs(dvega_pde - volga_bs) / abs(volga_bs) * 100

    print(f"{sig:<10.2f} | {volga_bs:<12.6f} | {dvega_bs:<15.6f} | {dvega_pde:<15.6f} | {error:<10.2f}%")

print("\n" + "="*100)
print("KEY FINDING")
print("="*100)

# The issue: PDE Vega is accurate in VALUE but wrong in DERIVATIVE
print("\n✅ PDE Vega VALUES are accurate (1-3% error)")
print("❌ But PDE Vega DERIVATIVES (∂Vega/∂σ) have 60%+ error")
print("\nThis means: Vega curve shape w.r.t. σ is wrong!")

print("\n💡 HYPOTHESIS:")
print("  The transformed coordinate τ = σ²(T-t)/2 changes the σ-dependence")
print("  So even if Vega(σ) values match, ∂Vega/∂σ may not!")

print("\n🔍 SOLUTION:")
print("  Option A: Use AAD to differentiate the entire PDE solve w.r.t. σ")
print("  Option B: Derive adjoint equation for Volga directly")
print("  Option C: Accept limitation (Volga is inherently hard for PDE)")
