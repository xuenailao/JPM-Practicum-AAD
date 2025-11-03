"""
混合方案: AAD for Vega + 优化的FD for Volga

Strategy:
  1. Vega: 使用AAD计算 (精确，1.5%误差) ✅
  2. Volga: 使用有限差分，但基于AAD-Vega而不是PDE-Vega
      - 这确保Vega值本身准确
      - 问题是Vega的σ依赖性形状

Key Insight from diagnosis:
  - PDE Vega VALUES are accurate (1.5% error)
  - But FD on Vega gives wrong Volga (68% error)

New Approach:
  - Use smaller PDE grid to make computation faster
  - Compute Vega at many sigma points (dense sampling)
  - Fit Vega(σ) curve and differentiate analytically

This avoids the Hessian computation complexity while improving accuracy.
"""
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import time

sys.path.insert(0, str(Path(__file__).parent))

from transformed_bs_pde import TransformedBSPDE
from scipy.stats import norm
from scipy.interpolate import CubicSpline


def black_scholes_all_greeks(S0: float, K: float, T: float, r: float, sigma: float) -> Dict:
    """Analytical Black-Scholes Greeks"""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    price = S0 * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S0 * sigma * sqrt_T)
    vega = S0 * norm.pdf(d1) * sqrt_T
    vanna = -norm.pdf(d1) * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'vanna': vanna,
        'volga': volga
    }


class AADVegaSplineVolga:
    """
    Compute Volga using cubic spline on AAD-computed Vega values

    Approach:
      1. Compute Vega at multiple sigma points using AAD
      2. Fit cubic spline to Vega(sigma)
      3. Differentiate spline to get Volga = dVega/dsigma

    This should be more accurate than simple FD because:
      - Uses more points (better curve fitting)
      - Spline smoothing reduces noise
      - Spline derivative is analytical (no truncation error)
    """

    def __init__(self, M: int = 151, N: int = 150):
        self.M = M
        self.N = N
        self.solver = TransformedBSPDE(K=100.0, T=1.0, r=0.05, M=M, N=N)

    def compute_vega_at_sigmas(self, sigma_center: float, n_points: int = 9,
                                delta_sigma: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Vega at multiple sigma values around sigma_center

        Args:
            sigma_center: Center sigma value
            n_points: Number of points (should be odd)
            delta_sigma: Total range ±delta_sigma

        Returns:
            sigma_values: Array of sigma values
            vega_values: Array of Vega values (from AAD)
        """
        # Create symmetric grid around sigma_center
        half_range = delta_sigma
        sigma_min = max(0.05, sigma_center - half_range)
        sigma_max = sigma_center + half_range

        sigma_values = np.linspace(sigma_min, sigma_max, n_points)
        vega_values = np.zeros(n_points)

        for i, sig in enumerate(sigma_values):
            _, vega = self.solver.solve(sig, verbose=False)
            vega_values[i] = vega

        return sigma_values, vega_values

    def compute_volga_via_spline(self, sigma: float, verbose: bool = False) -> Dict:
        """
        Compute Volga using cubic spline interpolation

        Returns:
            Dictionary with price, vega, volga
        """
        t_start = time.perf_counter()

        # Step 1: Compute Vega at center point
        price, vega_center = self.solver.solve(sigma, verbose=False)

        # Step 2: Compute Vega at multiple points for spline
        sigma_values, vega_values = self.compute_vega_at_sigmas(
            sigma, n_points=9, delta_sigma=0.04
        )

        if verbose:
            print(f"\n[Spline Volga] Dense Vega sampling:")
            print(f"  Sigma range: [{sigma_values[0]:.3f}, {sigma_values[-1]:.3f}]")
            print(f"  Points: {len(sigma_values)}")
            for i, (sig, vega) in enumerate(zip(sigma_values, vega_values)):
                marker = " ← center" if abs(sig - sigma) < 1e-6 else ""
                print(f"    σ={sig:.3f}: Vega={vega:.4f}{marker}")

        # Step 3: Fit cubic spline
        spline = CubicSpline(sigma_values, vega_values)

        # Step 4: Compute Volga as derivative of spline
        volga = spline.derivative()(sigma)

        # Also get Vega from spline for consistency check
        vega_spline = spline(sigma)

        t_elapsed = time.perf_counter() - t_start

        if verbose:
            print(f"\n[Spline Volga] Results:")
            print(f"  Vega (direct AAD): {vega_center:.6f}")
            print(f"  Vega (from spline): {vega_spline:.6f}")
            print(f"  Volga (spline derivative): {volga:.6f}")
            print(f"  Time: {t_elapsed:.3f}s")

        return {
            'price': price,
            'vega': vega_center,  # Use direct AAD vega
            'volga': volga,
            'time': t_elapsed
        }


def test_spline_volga():
    """Test spline-based Volga computation"""

    print("\n" + "="*120)
    print("混合方案: AAD Vega + Cubic Spline Volga")
    print("="*120)

    print("\n策略:")
    print("  1. 在多个σ点计算Vega (使用AAD，精确)")
    print("  2. 对Vega(σ)拟合三次样条")
    print("  3. Volga = dVega/dσ (样条的解析导数)")

    print("\n优势:")
    print("  ✅ Vega值精确 (AAD computed)")
    print("  ✅ 使用多点拟合 (比2点FD更稳定)")
    print("  ✅ 样条导数解析 (无截断误差)")
    print("  ✅ 计算速度快 (比Hessian快得多)")

    # Test parameters
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05

    # Create computer
    computer = AADVegaSplineVolga(M=151, N=150)

    print("\n" + "-"*120)
    print("单点详细测试 (σ=0.20)")
    print("-"*120)

    sigma = 0.20
    bs = black_scholes_all_greeks(S0, K, T, r, sigma)

    # Compute with verbose output
    pde = computer.compute_volga_via_spline(sigma, verbose=True)

    vega_err = abs(pde['vega'] - bs['vega']) / bs['vega'] * 100
    volga_err = abs(pde['volga'] - bs['volga']) / abs(bs['volga']) * 100

    print("\n比较结果:")
    print(f"  Vega:   BS={bs['vega']:.6f}, PDE={pde['vega']:.6f}, Error={vega_err:.2f}%")
    print(f"  Volga:  BS={bs['volga']:.6f}, PDE={pde['volga']:.6f}, Error={volga_err:.2f}%")

    print("\n" + "="*120)
    print("跨波动率测试")
    print("="*120)

    sigma_values = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

    print(f"\n{'Sigma':<10} | {'BS Vega':<12} | {'PDE Vega':<12} | {'Vega Err':<10} | "
          f"{'BS Volga':<12} | {'PDE Volga':<12} | {'Volga Err':<10} | {'Time(s)':<10}")
    print("-"*120)

    results = []
    for sig in sigma_values:
        bs = black_scholes_all_greeks(S0, K, T, r, sig)
        pde = computer.compute_volga_via_spline(sig, verbose=False)

        vega_err = abs(pde['vega'] - bs['vega']) / bs['vega'] * 100
        volga_err = abs(pde['volga'] - bs['volga']) / abs(bs['volga']) * 100

        # Check sign
        sign_ok = (pde['volga'] * bs['volga']) > 0

        print(f"{sig:<10.2f} | {bs['vega']:<12.6f} | {pde['vega']:<12.6f} | {vega_err:<10.2f}% | "
              f"{bs['volga']:<12.6f} | {pde['volga']:<12.6f} | {volga_err:<10.2f}% | {pde['time']:<10.3f}")

        results.append({
            'sigma': sig,
            'vega_err': vega_err,
            'volga_err': volga_err,
            'sign_ok': sign_ok
        })

    # Summary
    print("\n" + "="*120)
    print("总结")
    print("="*120)

    avg_vega_err = np.mean([r['vega_err'] for r in results])
    avg_volga_err = np.mean([r['volga_err'] for r in results])
    max_volga_err = np.max([r['volga_err'] for r in results])
    sign_correct = sum([r['sign_ok'] for r in results])

    print(f"\nVega:")
    print(f"  平均误差: {avg_vega_err:.2f}%")
    print(f"  状态: {'✅ 优秀' if avg_vega_err < 3.0 else '⚠️ 需改进'}")

    print(f"\nVolga:")
    print(f"  平均误差: {avg_volga_err:.2f}%")
    print(f"  最大误差: {max_volga_err:.2f}%")
    print(f"  符号正确: {sign_correct}/{len(results)}")

    # Compare with baseline (simple FD)
    baseline_error = 68.0  # From previous tests
    improvement = baseline_error - avg_volga_err

    print(f"\n对比基线 (简单有限差分):")
    print(f"  基线误差: {baseline_error:.1f}%")
    print(f"  Spline误差: {avg_volga_err:.1f}%")
    print(f"  改进: {improvement:+.1f}%")

    if avg_volga_err < 10.0:
        print("\n🎉 成功! Volga误差 <10%，方案达成目标!")
    elif avg_volga_err < 30.0:
        print(f"\n📈 改进! Volga误差从{baseline_error:.0f}%降至{avg_volga_err:.1f}%")
    elif avg_volga_err < baseline_error:
        print(f"\n↗️ 有改进: 误差从{baseline_error:.0f}%降至{avg_volga_err:.1f}%，但仍需优化")
    else:
        print(f"\n⚠️ Spline方法与基线相当")


def test_spline_parameters():
    """Test different spline parameters"""

    print("\n" + "="*120)
    print("Spline参数优化测试")
    print("="*120)

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    bs = black_scholes_all_greeks(S0, K, T, r, sigma)

    # Test different number of points
    print("\n测试不同的采样点数:")
    print(f"{'Points':<10} | {'Delta_sigma':<12} | {'Volga':<12} | {'Error':<10} | {'Time(s)':<10}")
    print("-"*80)

    for n_points in [5, 7, 9, 11, 13]:
        computer = AADVegaSplineVolga(M=151, N=150)

        t_start = time.perf_counter()
        sigma_values, vega_values = computer.compute_vega_at_sigmas(
            sigma, n_points=n_points, delta_sigma=0.04
        )
        spline = CubicSpline(sigma_values, vega_values)
        volga = spline.derivative()(sigma)
        t_elapsed = time.perf_counter() - t_start

        error = abs(volga - bs['volga']) / abs(bs['volga']) * 100
        delta_sig = sigma_values[-1] - sigma_values[0]

        print(f"{n_points:<10} | {delta_sig:<12.4f} | {volga:<12.6f} | {error:<10.2f}% | {t_elapsed:<10.3f}")

    # Test different delta_sigma
    print("\n测试不同的sigma范围 (n_points=9):")
    print(f"{'Delta_sigma':<12} | {'Sigma_range':<25} | {'Volga':<12} | {'Error':<10}")
    print("-"*80)

    for delta_sigma in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        computer = AADVegaSplineVolga(M=151, N=150)
        sigma_values, vega_values = computer.compute_vega_at_sigmas(
            sigma, n_points=9, delta_sigma=delta_sigma
        )
        spline = CubicSpline(sigma_values, vega_values)
        volga = spline.derivative()(sigma)

        error = abs(volga - bs['volga']) / abs(bs['volga']) * 100
        sigma_range = f"[{sigma_values[0]:.3f}, {sigma_values[-1]:.3f}]"

        print(f"{delta_sigma:<12.3f} | {sigma_range:<25} | {volga:<12.6f} | {error:<10.2f}%")


if __name__ == "__main__":
    test_spline_volga()
    print("\n")
    test_spline_parameters()
