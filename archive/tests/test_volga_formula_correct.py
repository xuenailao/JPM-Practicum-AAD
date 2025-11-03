"""
Test the correct formula for Volga
"""
import numpy as np
from scipy.stats import norm


def black_scholes_greeks(S0, K, T, r, sigma):
    """Complete BS Greeks"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma

    return price, vega, volga


def main():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    price_bs, vega_bs, volga_bs = black_scholes_greeks(S0, K, T, r, sigma)

    print("\n" + "="*100)
    print("VOLGA FORMULA VERIFICATION")
    print("="*100)
    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"\nAnalytical: Vega={vega_bs:.6f}, Volga={volga_bs:.6f}")

    print("\n" + "-"*100)
    print("METHOD 1: Volga = ∂Vega/∂σ (FIRST derivative of Vega)")
    print("-"*100)
    print("Formula: (Vega(σ+ε) - Vega(σ-ε)) / (2ε)")
    print(f"\n{'eps':<10} | {'Vega-':<12} | {'Vega+':<12} | {'Volga (FD)':<12} | {'Analytical':<12} | {'Error':<10}")
    print("-"*90)

    for eps in [0.0001, 0.001, 0.01, 0.02]:
        _, vega_minus, _ = black_scholes_greeks(S0, K, T, r, sigma - eps)
        _, vega_plus, _ = black_scholes_greeks(S0, K, T, r, sigma + eps)

        # FIRST derivative formula
        volga_fd = (vega_plus - vega_minus) / (2 * eps)
        error = abs(volga_fd - volga_bs) / abs(volga_bs) * 100

        print(f"{eps:<10.4f} | {vega_minus:<12.6f} | {vega_plus:<12.6f} | "
              f"{volga_fd:<12.6f} | {volga_bs:<12.6f} | {error:<10.2f}%")

    print("\n" + "-"*100)
    print("METHOD 2: Volga = ∂²V/∂σ² (SECOND derivative of Price) - WRONG FOR VEGA!")
    print("-"*100)
    print("Formula: (Vega(σ+ε) - 2×Vega(σ) + Vega(σ-ε)) / ε²")
    print(f"\n{'eps':<10} | {'Vega-':<12} | {'Vega0':<12} | {'Vega+':<12} | {'Result':<12} | {'Error':<10}")
    print("-"*100)

    for eps in [0.0001, 0.001, 0.01, 0.02]:
        _, vega_minus, _ = black_scholes_greeks(S0, K, T, r, sigma - eps)
        _, vega_center, _ = black_scholes_greeks(S0, K, T, r, sigma)
        _, vega_plus, _ = black_scholes_greeks(S0, K, T, r, sigma + eps)

        # SECOND derivative formula (WRONG!)
        result = (vega_plus - 2*vega_center + vega_minus) / (eps ** 2)
        error = abs(result - volga_bs) / abs(volga_bs) * 100

        print(f"{eps:<10.4f} | {vega_minus:<12.6f} | {vega_center:<12.6f} | {vega_plus:<12.6f} | "
              f"{result:<12.6f} | {error:<10.2f}%")

    print("\n" + "="*100)
    print("CONCLUSION")
    print("="*100)
    print("\n✅ METHOD 1 is CORRECT:")
    print("   Volga = ∂Vega/∂σ = (Vega(σ+ε) - Vega(σ-ε)) / (2ε)")
    print("   This is a FIRST derivative (centered difference)")
    print("\n❌ METHOD 2 is WRONG:")
    print("   Using second derivative formula on Vega gives ∂²Vega/∂σ² (not Volga!)")
    print("\n📝 DEFINITION:")
    print("   Volga = ∂²V/∂σ² = ∂(∂V/∂σ)/∂σ = ∂Vega/∂σ")
    print("   It's the FIRST derivative of Vega, not SECOND derivative!")


if __name__ == "__main__":
    main()
