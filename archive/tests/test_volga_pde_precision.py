"""
Test if PDE Vega precision is causing Volga errors
"""
import numpy as np
from scipy.stats import norm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from aad_edge_pushing.pde.AADgraph.greeks_methods_comparison import GreeksMethodA


def black_scholes_volga(S0, K, T, r, sigma):
    """Analytical formulas"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    vega = S0 * norm.pdf(d1) * sqrt_T
    volga = vega * d1 * d2 / sigma

    return vega, volga


def main():
    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2

    vega_bs, volga_bs = black_scholes_volga(S0, K, T, r, sigma)

    print("\n" + "="*100)
    print("VOLGA ERROR ANALYSIS: PDE Vega Precision Impact")
    print("="*100)
    print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}")
    print(f"\nAnalytical: Vega={vega_bs:.6f}, Volga={volga_bs:.6f}")

    # Test with different grid sizes
    grid_configs = [
        (51, 50),
        (101, 100),
        (151, 150),
    ]

    for M, N in grid_configs:
        print("\n" + "="*100)
        print(f"Grid Size: M={M}, N={N}")
        print("="*100)

        method = GreeksMethodA(M=M, N=N)

        eps_sigma = sigma * 0.01

        # Compute Vega at three sigma points
        _, vega_minus = method._solve_at_S0(S0, sigma - eps_sigma)
        _, vega_center = method._solve_at_S0(S0, sigma)
        _, vega_plus = method._solve_at_S0(S0, sigma + eps_sigma)

        # Analytical Vega at same points for comparison
        vega_minus_bs, _ = black_scholes_volga(S0, K, T, r, sigma - eps_sigma)
        vega_center_bs, _ = black_scholes_volga(S0, K, T, r, sigma)
        vega_plus_bs, _ = black_scholes_volga(S0, K, T, r, sigma + eps_sigma)

        # Compute Volga
        volga_pde = (vega_plus - vega_minus) / (2 * eps_sigma)
        volga_bs_fd = (vega_plus_bs - vega_minus_bs) / (2 * eps_sigma)

        print(f"\nVega Comparison:")
        print(f"  σ - ε: PDE={vega_minus:12.6f}  BS={vega_minus_bs:12.6f}  Error={abs(vega_minus-vega_minus_bs)/vega_minus_bs*100:6.2f}%")
        print(f"  σ    : PDE={vega_center:12.6f}  BS={vega_center_bs:12.6f}  Error={abs(vega_center-vega_center_bs)/vega_center_bs*100:6.2f}%")
        print(f"  σ + ε: PDE={vega_plus:12.6f}  BS={vega_plus_bs:12.6f}  Error={abs(vega_plus-vega_plus_bs)/vega_plus_bs*100:6.2f}%")

        print(f"\nVolga Comparison:")
        print(f"  PDE Volga:         {volga_pde:12.6f}  (from PDE Vegas)")
        print(f"  BS Volga (FD):     {volga_bs_fd:12.6f}  (from BS Vegas via FD)")
        print(f"  BS Volga (exact):  {volga_bs:12.6f}  (analytical formula)")
        print(f"\n  PDE Volga Error:   {abs(volga_pde - volga_bs)/abs(volga_bs)*100:6.2f}%")
        print(f"  BS FD Error:       {abs(volga_bs_fd - volga_bs)/abs(volga_bs)*100:6.2f}%")

        # Diagnosis
        vega_err_pct = abs(vega_center - vega_center_bs) / vega_center_bs * 100
        print(f"\n📊 Diagnosis:")
        print(f"  Vega error:   ~{vega_err_pct:.2f}%")
        print(f"  Volga error:  ~{abs(volga_pde - volga_bs)/abs(volga_bs)*100:.2f}%")
        print(f"  Amplification: {(abs(volga_pde - volga_bs)/abs(volga_bs)*100) / vega_err_pct:.1f}× Vega error")

    print("\n" + "="*100)
    print("CONCLUSION")
    print("="*100)
    print("\nIf BS FD gives correct Volga but PDE doesn't:")
    print("  → Formula is correct, PDE Vega precision is insufficient")
    print("\nTo fix: Need Vega error < 1% to get Volga error < 10%")
    print("  Current: Vega error ~13% → Volga error amplified significantly")


if __name__ == "__main__":
    main()
