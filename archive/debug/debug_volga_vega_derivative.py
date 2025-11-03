"""
Debug Volga by examining Vega derivative

Problem: Volga = ∂Vega/∂σ has 68% error regardless of epsilon
Hypothesis: Maybe Vega itself is not smooth enough w.r.t. σ

Test: Compute Vega at many σ points and examine the derivative
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from transformed_bs_pde import TransformedBSPDE
from scipy.stats import norm


def black_scholes_vega(S0, K, T, r, sigma):
    """Analytical Vega"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    vega = S0 * norm.pdf(d1) * sqrt_T
    return vega


def black_scholes_volga(S0, K, T, r, sigma):
    """Analytical Volga"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T
    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma
    return volga


def test_vega_derivative():
    """Test Vega derivative by dense sampling"""

    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    print("\n" + "="*120)
    print("VOLGA DEBUG: EXAMINE VEGA DERIVATIVE")
    print("="*120)

    print("\nHypothesis: Maybe PDE Vega is not smooth enough to differentiate")
    print("Test: Sample Vega at many sigma points and examine derivative\n")

    # Test at sigma = 0.20
    sigma_center = 0.20

    print("-"*120)
    print(f"TEST 1: Dense Vega Sampling Around σ={sigma_center}")
    print("-"*120)

    # Dense sampling: σ from 0.19 to 0.21 with 21 points
    sigma_values = np.linspace(sigma_center - 0.01, sigma_center + 0.01, 21)

    solver = TransformedBSPDE(K=K, T=T, r=r, M=151, N=150)

    vega_bs_list = []
    vega_pde_list = []

    print(f"\n{'Sigma':<12} | {'BS Vega':<15} | {'PDE Vega':<15} | {'Vega Error':<12}")
    print("-"*120)

    for sig in sigma_values:
        vega_bs = black_scholes_vega(S0, K, T, r, sig)
        _, vega_pde = solver.solve(sig, verbose=False)

        error = abs(vega_pde - vega_bs) / vega_bs * 100

        vega_bs_list.append(vega_bs)
        vega_pde_list.append(vega_pde)

        print(f"{sig:<12.6f} | {vega_bs:<15.6f} | {vega_pde:<15.6f} | {error:<12.2f}%")

    # Compute derivatives
    print("\n" + "-"*120)
    print("Vega Derivatives (Finite Difference)")
    print("-"*120)

    print(f"\n{'Sigma Range':<25} | {'BS ∂Vega/∂σ':<15} | {'PDE ∂Vega/∂σ':<15} | {'BS Volga':<15} | {'Error':<12}")
    print("-"*120)

    # Compute centered differences
    for i in range(1, len(sigma_values)-1):
        sig_minus = sigma_values[i-1]
        sig_center = sigma_values[i]
        sig_plus = sigma_values[i+1]

        dsigma = sig_plus - sig_minus

        # BS derivative
        dvega_bs = (vega_bs_list[i+1] - vega_bs_list[i-1]) / dsigma

        # PDE derivative
        dvega_pde = (vega_pde_list[i+1] - vega_pde_list[i-1]) / dsigma

        # BS Volga at center
        volga_bs = black_scholes_volga(S0, K, T, r, sig_center)

        error = abs(dvega_pde - volga_bs) / abs(volga_bs) * 100

        print(f"{sig_minus:.4f} - {sig_plus:.4f} | {dvega_bs:<15.6f} | {dvega_pde:<15.6f} | "
              f"{volga_bs:<15.6f} | {error:<12.2f}%")

    # Test 2: Multiple sigma values
    print("\n" + "="*120)
    print("TEST 2: Volga at Different σ")
    print("="*120)

    test_sigmas = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

    print(f"\n{'Sigma':<10} | {'BS Vega':<12} | {'PDE Vega':<12} | {'Vega Err':<10} | "
          f"{'BS Volga':<12} | {'PDE Volga':<12} | {'Volga Err':<10}")
    print("-"*120)

    for sig in test_sigmas:
        # Vega
        vega_bs = black_scholes_vega(S0, K, T, r, sig)
        _, vega_pde = solver.solve(sig, verbose=False)
        vega_err = abs(vega_pde - vega_bs) / vega_bs * 100

        # Volga using finite difference
        eps = sig * 0.002  # Use smaller epsilon
        _, vega_minus = solver.solve(sig - eps, verbose=False)
        _, vega_plus = solver.solve(sig + eps, verbose=False)
        volga_pde = (vega_plus - vega_minus) / (2 * eps)

        volga_bs = black_scholes_volga(S0, K, T, r, sig)
        volga_err = abs(volga_pde - volga_bs) / abs(volga_bs) * 100

        print(f"{sig:<10.2f} | {vega_bs:<12.6f} | {vega_pde:<12.6f} | {vega_err:<10.2f}% | "
              f"{volga_bs:<12.6f} | {volga_pde:<12.6f} | {volga_err:<10.2f}%")

    # Analysis
    print("\n" + "="*120)
    print("ANALYSIS")
    print("="*120)

    # Check if Vega error correlates with Volga error
    print("\n🔍 Key Observations:")
    print(f"  1. PDE Vega error: 1-3% (excellent!)")
    print(f"  2. But Volga error: 68%+ (terrible!)")
    print(f"  3. This suggests: Vega is accurate but its σ-derivative is wrong")

    print("\n💡 Possible Explanations:")
    print("  A. Vega is accurate in VALUE but wrong in SHAPE")
    print("     → Maybe Vega curve w.r.t. σ has wrong curvature")
    print("  B. Numerical discretization affects derivatives more than values")
    print("     → Similar to how Gamma is harder than Delta")
    print("  C. The transformed PDE coordinate system affects σ-derivatives")
    print("     → τ = σ²(T-t)/2 means ∂/∂σ ≠ direct differentiation")

    print("\n🎯 Next Steps:")
    print("  1. Check: Is PDE Vega's σ-dependence shape correct?")
    print("  2. Compute Volga using adjoint method (avoid finite difference)")
    print("  3. Test: Does Method B (AAD on Vega) work better?")


if __name__ == "__main__":
    test_vega_derivative()
