"""
Test analytical Volga formula and its finite difference approximation
"""
import numpy as np
from scipy.stats import norm


def black_scholes_greeks(S0, K, T, r, sigma):
    """Complete BS Greeks including Volga"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0 * sigma * sqrt_T)

    # First-order Greeks wrt sigma
    vega = S0 * norm.pdf(d1) * sqrt_T

    # Second-order Greeks wrt sigma
    vanna = -norm.pdf(d1) * d2 / sigma  # ∂²V/∂S∂σ
    volga = vega * d1 * d2 / sigma      # ∂²V/∂σ²

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


def test_vega_derivative():
    """Test if ∂Vega/∂σ = Volga analytically"""

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    greeks = black_scholes_greeks(S0, K, T, r, sigma)

    print("\n" + "="*80)
    print("ANALYTICAL VOLGA VERIFICATION")
    print("="*80)
    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"\nGreeks from closed-form formulas:")
    print(f"  Vega:  {greeks['vega']:.6f}")
    print(f"  Volga: {greeks['volga']:.6f}")

    # Test: Compute ∂Vega/∂σ numerically
    print("\n" + "-"*80)
    print("Numerical verification: ∂Vega/∂σ vs Volga")
    print("-"*80)

    eps_values = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    print(f"\n{'eps':<10} | {'∂Vega/∂σ (FD)':<15} | {'Volga (formula)':<15} | {'Match?':<10}")
    print("-"*70)

    for eps in eps_values:
        # Compute dVega/dsigma using finite difference
        greeks_plus = black_scholes_greeks(S0, K, T, r, sigma + eps)
        greeks_minus = black_scholes_greeks(S0, K, T, r, sigma - eps)

        dvega_dsigma = (greeks_plus['vega'] - greeks_minus['vega']) / (2 * eps)

        match = "✅ YES" if abs(dvega_dsigma - greeks['volga']) / abs(greeks['volga']) < 0.01 else "❌ NO"

        print(f"{eps:<10.4f} | {dvega_dsigma:<15.6f} | {greeks['volga']:<15.6f} | {match:<10}")

    # The issue: Volga formula in BS
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    print(f"\nIntermediate values:")
    print(f"  d1 = {d1:.6f}")
    print(f"  d2 = {d2:.6f}")
    print(f"  d1 * d2 = {d1 * d2:.6f}")

    print(f"\nVolga formula: Volga = Vega × d1 × d2 / σ")
    print(f"  = {greeks['vega']:.6f} × {d1:.6f} × {d2:.6f} / {sigma:.6f}")
    print(f"  = {greeks['volga']:.6f}")

    # Check sign
    print(f"\nSign analysis:")
    print(f"  d1 = {d1:.6f} {'(positive)' if d1 > 0 else '(negative)'}")
    print(f"  d2 = {d2:.6f} {'(positive)' if d2 > 0 else '(negative)'}")
    print(f"  d1 × d2 = {d1 * d2:.6f} {'(positive)' if d1*d2 > 0 else '(negative)'}")
    print(f"  → Volga should be {'positive' if d1*d2 > 0 else 'negative'}")

    # Test with different strikes
    print("\n" + "-"*80)
    print("Volga for different moneyness")
    print("-"*80)

    print(f"\n{'Strike':<10} | {'S0/K':<10} | {'d1':<10} | {'d2':<10} | {'Volga':<12} | {'Sign':<10}")
    print("-"*70)

    for K_test in [80, 90, 100, 110, 120]:
        greeks_test = black_scholes_greeks(S0, K_test, T, r, sigma)
        sqrt_T_test = np.sqrt(T)
        d1_test = (np.log(S0 / K_test) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T_test)
        d2_test = d1_test - sigma * sqrt_T_test

        sign = '+' if greeks_test['volga'] > 0 else '-'

        print(f"{K_test:<10.1f} | {S0/K_test:<10.3f} | {d1_test:<10.4f} | {d2_test:<10.4f} | "
              f"{greeks_test['volga']:<12.6f} | {sign:<10}")


if __name__ == "__main__":
    test_vega_derivative()
