"""
Simple BSM Analytical Greeks Test
Compare with known values to validate formulas
"""

import numpy as np
from scipy.stats import norm

class BSMAnalytical:
    """Black-Scholes-Merton analytical Greeks."""

    @staticmethod
    def d1(S, K, T, r, sigma):
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))

    @staticmethod
    def d2(S, K, T, r, sigma):
        return BSMAnalytical.d1(S, K, T, r, sigma) - sigma*np.sqrt(T)

    @staticmethod
    def call_price(S, K, T, r, sigma):
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        d2 = BSMAnalytical.d2(S, K, T, r, sigma)
        return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

    @staticmethod
    def delta(S, K, T, r, sigma):
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        return norm.cdf(d1)

    @staticmethod
    def gamma(S, K, T, r, sigma):
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def vega(S, K, T, r, sigma):
        """Vega per 1% vol change"""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100

    @staticmethod
    def vanna(S, K, T, r, sigma):
        """Vanna: ∂²V/∂S∂σ per 1% vol"""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        d2 = BSMAnalytical.d2(S, K, T, r, sigma)
        return -norm.pdf(d1) * d2 / sigma / 100

    @staticmethod
    def volga(S, K, T, r, sigma):
        """Volga: ∂²V/∂σ² per 1%²"""
        d1 = BSMAnalytical.d1(S, K, T, r, sigma)
        d2 = BSMAnalytical.d2(S, K, T, r, sigma)
        vega_raw = S * norm.pdf(d1) * np.sqrt(T)
        return vega_raw * d1 * d2 / sigma / 10000

# Test
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

print("=" * 80)
print("BSM ANALYTICAL GREEKS (Ground Truth)")
print("=" * 80)
print(f"\nParameters: S={S0}, K={K}, T={T}, r={r}, σ={sigma}")
print(f"Moneyness: ATM (S/K = {S0/K:.2f})")

print("\n" + "-" * 80)
print("FIRST-ORDER GREEKS")
print("-" * 80)
price = BSMAnalytical.call_price(S0, K, T, r, sigma)
delta = BSMAnalytical.delta(S0, K, T, r, sigma)
vega = BSMAnalytical.vega(S0, K, T, r, sigma)

print(f"Price:  ${price:.6f}")
print(f"Delta:   {delta:.6f}  (∂V/∂S)")
print(f"Vega:    {vega:.6f}  (∂V/∂σ per 1%)")

print("\n" + "-" * 80)
print("SECOND-ORDER GREEKS")
print("-" * 80)
gamma = BSMAnalytical.gamma(S0, K, T, r, sigma)
vanna = BSMAnalytical.vanna(S0, K, T, r, sigma)
volga = BSMAnalytical.volga(S0, K, T, r, sigma)

print(f"Gamma:   {gamma:.6f}  (∂²V/∂S²)")
print(f"Vanna:   {vanna:.6f}  (∂²V/∂S∂σ per 1%)")
print(f"Volga:   {volga:.6f}  (∂²V/∂σ² per 1%²)")

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)
print(f"\nΔ = {delta:.4f}:")
print(f"  For $1 increase in stock, option gains ${delta:.4f}")

print(f"\nΓ = {gamma:.6f}:")
print(f"  Delta changes by {gamma:.6f} for $1 stock move")
print(f"  Convexity exposure: {'High' if gamma > 0.015 else 'Low'}")

print(f"\nVega = {vega:.6f}:")
print(f"  For 1% vol increase (20%→21%), option gains ${vega:.6f}")

print(f"\nVanna = {vanna:.6f}:")
print(f"  Delta {'decreases' if vanna < 0 else 'increases'} by {abs(vanna):.6f} for 1% vol increase")
print(f"  Cross-gamma: vol-delta interaction")

print(f"\nVolga = {volga:.6f}:")
print(f"  Vega {'increases' if volga > 0 else 'decreases'} by ${abs(volga):.6f} for 1% vol increase")
print(f"  Convexity in vol exposure: {'Positive' if volga > 0 else 'Negative'}")

print("\n" + "=" * 80)
print("These are EXACT values for constant volatility BSM")
print("PDE methods should match these for constant vol surface")
print("=" * 80)
