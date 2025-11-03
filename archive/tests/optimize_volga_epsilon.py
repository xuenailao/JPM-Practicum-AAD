"""
Optimize Epsilon for Volga Calculation

Problem: Volga has 68% error with eps_sigma = 0.01*sigma

Test different epsilon values to find optimal choice
"""
import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformed_bs_pde import TransformedBSPDE


def black_scholes_volga(S0, K, T, r, sigma):
    """Analytical Volga"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma
    return volga


def compute_volga_with_eps(S0, K, T, r, sigma, eps_sigma, M=151, N=150):
    """Compute Volga using specific epsilon"""
    solver = TransformedBSPDE(K=K, T=T, r=r, M=M, N=N)

    _, vega_minus = solver.solve(sigma - eps_sigma, verbose=False)
    _, vega_plus = solver.solve(sigma + eps_sigma, verbose=False)

    volga = (vega_plus - vega_minus) / (2 * eps_sigma)

    return volga


def test_epsilon_values():
    """Test different epsilon values"""

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    volga_bs = black_scholes_volga(S0, K, T, r, sigma)

    print("\n" + "="*120)
    print("VOLGA EPSILON OPTIMIZATION")
    print("="*120)

    print(f"\nTest Parameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"Analytical Volga: {volga_bs:.6f}")

    # Test different epsilon values
    eps_factors = [0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]

    print("\n" + "-"*120)
    print("Epsilon Sensitivity Test")
    print("-"*120)

    print(f"\n{'eps/σ':<12} | {'eps_sigma':<12} | {'PDE Volga':<15} | {'Error':<12} | {'Status':<10}")
    print("-"*120)

    results = []
    for eps_factor in eps_factors:
        eps_sigma = sigma * eps_factor

        try:
            volga_pde = compute_volga_with_eps(S0, K, T, r, sigma, eps_sigma)

            error = abs(volga_pde - volga_bs) / abs(volga_bs) * 100
            status = "✅" if error < 10 else "⚠️" if error < 30 else "❌"

            print(f"{eps_factor:<12.4f} | {eps_sigma:<12.6f} | {volga_pde:<15.6f} | {error:<12.2f}% | {status:<10}")

            results.append({
                'eps_factor': eps_factor,
                'eps_sigma': eps_sigma,
                'volga': volga_pde,
                'error': error
            })
        except Exception as e:
            print(f"{eps_factor:<12.4f} | {eps_sigma:<12.6f} | ERROR: {str(e)[:30]}")

    # Find best
    if results:
        best = min(results, key=lambda x: x['error'])

        print("\n" + "="*120)
        print("OPTIMAL EPSILON")
        print("="*120)
        print(f"\nBest eps_factor: {best['eps_factor']:.4f}")
        print(f"Best eps_sigma:  {best['eps_sigma']:.6f}")
        print(f"Volga:           {best['volga']:.6f}")
        print(f"Error:           {best['error']:.2f}%")

    # Test across different sigma values with best epsilon
    if results:
        print("\n" + "="*120)
        print(f"TEST WITH OPTIMAL EPSILON (eps_factor={best['eps_factor']:.4f}) ACROSS DIFFERENT σ")
        print("="*120)

        sigma_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

        print(f"\n{'Sigma':<10} | {'BS Volga':<12} | {'PDE Volga':<12} | {'Error':<10} | {'Sign':<10}")
        print("-"*120)

        for sig in sigma_values:
            volga_bs_test = black_scholes_volga(S0, K, T, r, sig)
            eps_sig = sig * best['eps_factor']

            try:
                volga_pde_test = compute_volga_with_eps(S0, K, T, r, sig, eps_sig)

                error = abs(volga_pde_test - volga_bs_test) / abs(volga_bs_test) * 100
                sign = "✅" if volga_pde_test * volga_bs_test > 0 else "❌"

                print(f"{sig:<10.2f} | {volga_bs_test:<12.6f} | {volga_pde_test:<12.6f} | {error:<10.2f}% | {sign:<10}")
            except Exception as e:
                print(f"{sig:<10.2f} | ERROR: {str(e)[:40]}")

    # Additional test: Richardson extrapolation
    print("\n" + "="*120)
    print("RICHARDSON EXTRAPOLATION TEST")
    print("="*120)

    sigma_test = 0.20
    volga_bs = black_scholes_volga(S0, K, T, r, sigma_test)

    # Two epsilon values: h and h/2
    eps_h = sigma_test * 0.002
    eps_h2 = sigma_test * 0.001

    solver = TransformedBSPDE(K=K, T=T, r=r, M=151, N=150)

    # With h
    _, vega_minus_h = solver.solve(sigma_test - eps_h)
    _, vega_plus_h = solver.solve(sigma_test + eps_h)
    volga_h = (vega_plus_h - vega_minus_h) / (2 * eps_h)

    # With h/2
    _, vega_minus_h2 = solver.solve(sigma_test - eps_h2)
    _, vega_plus_h2 = solver.solve(sigma_test + eps_h2)
    volga_h2 = (vega_plus_h2 - vega_minus_h2) / (2 * eps_h2)

    # Richardson extrapolation
    volga_richardson = (4 * volga_h2 - volga_h) / 3

    print(f"\nAnalytical Volga:  {volga_bs:.6f}")
    print(f"Volga (h):         {volga_h:.6f}  Error: {abs(volga_h - volga_bs)/abs(volga_bs)*100:.2f}%")
    print(f"Volga (h/2):       {volga_h2:.6f}  Error: {abs(volga_h2 - volga_bs)/abs(volga_bs)*100:.2f}%")
    print(f"Volga (Richardson): {volga_richardson:.6f}  Error: {abs(volga_richardson - volga_bs)/abs(volga_bs)*100:.2f}%")

    print("\n" + "="*120)
    print("DIAGNOSIS")
    print("="*120)

    print("\n🔍 Key Findings:")
    print("  1. Volga is very sensitive to epsilon choice")
    print("  2. Smaller epsilon (0.001-0.002) may give better results")
    print("  3. Richardson extrapolation may improve accuracy")

    print("\n💡 Recommendations:")
    print("  1. Use eps_sigma = 0.001 * sigma (0.1% of sigma)")
    print("  2. Consider Richardson extrapolation for critical applications")
    print("  3. Volga inherently has larger error than first-order Greeks")


if __name__ == "__main__":
    test_epsilon_values()
