"""
完整的Greeks计算方法对比：Algo3 vs Algo4 vs Bumping + PDE Edge-Pushing vs PDE Bumping

测试内容:
1. Algo3 vs Algo4-Opt vs Bumping (BSM Greeks)
2. PDE True Second-Order AD vs PDE Bumping
"""

import numpy as np
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from math import log, sqrt, exp
from scipy.stats import norm

# ==================== Part 1: Algo3 vs Algo4 vs Bumping ====================

def bsm_greeks_analytical(S0, K, T, r, sigma, cp_flag='C'):
    """BSM解析Greeks（基准）"""
    d1 = (log(S0/K) + (r+0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    # 一阶Greeks
    delta = norm.cdf(d1) if cp_flag == 'C' else -norm.cdf(-d1)
    vega = S0 * norm.pdf(d1) * sqrt(T)

    # 二阶Greeks
    gamma = norm.pdf(d1) / (S0 * sigma * sqrt(T))
    vanna = -norm.pdf(d1) * d2 / sigma
    volga = S0 * norm.pdf(d1) * sqrt(T) * d1 * d2 / sigma

    price = S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2) if cp_flag == 'C' else \
            K*exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)

    return {
        'price': price,
        'delta': delta,
        'vega': vega,
        'gamma': gamma,
        'vanna': vanna,
        'volga': volga
    }


def bumping_greeks(S0, K, T, r, sigma, cp_flag='C'):
    """Bumping方法（有限差分）"""
    def bsm_price(S, K, T, r, sig, cp):
        d1 = (log(S/K) + (r+0.5*sig**2)*T) / (sig*sqrt(T))
        d2 = d1 - sig*sqrt(T)
        if cp == 'C':
            return S*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
        else:
            return K*exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

    h = 1e-5
    price = bsm_price(S0, K, T, r, sigma, cp_flag)

    # 二阶Greeks (中心差分)
    gamma = (bsm_price(S0 + h, K, T, r, sigma, cp_flag) -
             2*price +
             bsm_price(S0 - h, K, T, r, sigma, cp_flag)) / (h**2)

    volga = (bsm_price(S0, K, T, r, sigma + h, cp_flag) -
             2*price +
             bsm_price(S0, K, T, r, sigma - h, cp_flag)) / (h**2)

    vanna = (bsm_price(S0 + h, K, T, r, sigma + h, cp_flag) -
             bsm_price(S0 + h, K, T, r, sigma - h, cp_flag) -
             bsm_price(S0 - h, K, T, r, sigma + h, cp_flag) +
             bsm_price(S0 - h, K, T, r, sigma - h, cp_flag)) / (4 * h**2)

    return {
        'price': price,
        'gamma': gamma,
        'vanna': vanna,
        'volga': volga
    }


def algo3_greeks(S0, K, T, r, sigma, cp_flag='C'):
    """Algo3 (Block Form)"""
    from aad_edge_pushing.aad.core.var import ADVar
    from aad_edge_pushing.aad.core.tape import global_tape
    from aad_edge_pushing.aad.ops.transcendental import exp as ad_exp, log as ad_log, erf
    from aad_edge_pushing.algo3.algo3_block import algo3_block

    global_tape.reset()

    S_ad = ADVar(S0)
    sigma_ad = ADVar(sigma)
    T_ad = ADVar(T)
    K_val = K
    r_val = r

    # BSM公式（简化版：只对S和sigma做AD）
    sqrt_T = T_ad ** 0.5
    sqrt_2 = ADVar(np.sqrt(2.0))

    d1 = (ad_log(S_ad / K_val) + (r_val + sigma_ad * sigma_ad * 0.5) * T_ad) / (sigma_ad * sqrt_T)
    d2 = d1 - sigma_ad * sqrt_T

    N_d1 = 0.5 * (1.0 + erf(d1 / sqrt_2))
    N_d2 = 0.5 * (1.0 + erf(d2 / sqrt_2))

    if cp_flag == 'C':
        price_ad = S_ad * N_d1 - K_val * ad_exp(-r_val * T_ad) * N_d2
    else:
        price_ad = K_val * ad_exp(-r_val * T_ad) * N_d2 - S_ad * N_d1

    inputs = [S_ad, sigma_ad]
    H = algo3_block(price_ad, inputs)

    return {
        'price': price_ad.val,
        'gamma': H[0, 0],      # ∂²V/∂S²
        'vanna': H[0, 1],      # ∂²V/∂S∂σ
        'volga': H[1, 1],      # ∂²V/∂σ²
    }


def algo4_greeks(S0, K, T, r, sigma, cp_flag='C'):
    """Algo4-Optimized"""
    from aad_edge_pushing.aad.core.var import ADVar
    from aad_edge_pushing.aad.core.tape import global_tape
    from aad_edge_pushing.aad.ops.transcendental import exp as ad_exp, log as ad_log, erf
    from aad_edge_pushing.algo3.algo4_optimized import algo4_optimized

    global_tape.reset()

    S_ad = ADVar(S0)
    sigma_ad = ADVar(sigma)
    T_ad = ADVar(T)
    K_val = K
    r_val = r

    sqrt_T = T_ad ** 0.5
    sqrt_2 = ADVar(np.sqrt(2.0))

    d1 = (ad_log(S_ad / K_val) + (r_val + sigma_ad * sigma_ad * 0.5) * T_ad) / (sigma_ad * sqrt_T)
    d2 = d1 - sigma_ad * sqrt_T

    N_d1 = 0.5 * (1.0 + erf(d1 / sqrt_2))
    N_d2 = 0.5 * (1.0 + erf(d2 / sqrt_2))

    if cp_flag == 'C':
        price_ad = S_ad * N_d1 - K_val * ad_exp(-r_val * T_ad) * N_d2
    else:
        price_ad = K_val * ad_exp(-r_val * T_ad) * N_d2 - S_ad * N_d1

    inputs = [S_ad, sigma_ad]
    H = algo4_optimized(price_ad, inputs)

    return {
        'price': price_ad.val,
        'gamma': H[0, 0],
        'vanna': H[0, 1],
        'volga': H[1, 1],
    }


def test_algo3_algo4_bumping():
    """Test 1: Algo3 vs Algo4 vs Bumping"""
    print("="*80)
    print("TEST 1: Algo3 vs Algo4-Opt vs Bumping (BSM Greeks)")
    print("="*80)

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    cp_flag = 'C'

    print(f"\n参数: S0={S0}, K={K}, T={T}, r={r}, σ={sigma}, 类型={cp_flag}\n")

    # 解析解（基准）
    print("【1. BSM解析解（基准）】")
    t0 = time.perf_counter()
    bsm_result = bsm_greeks_analytical(S0, K, T, r, sigma, cp_flag)
    t_bsm = (time.perf_counter() - t0) * 1000

    print(f"Gamma: {bsm_result['gamma']:.8f}")
    print(f"Vanna: {bsm_result['vanna']:.8f}")
    print(f"Volga: {bsm_result['volga']:.8f}")
    print(f"时间:  {t_bsm:.4f} ms\n")

    # Bumping
    print("【2. Bumping (h=1e-5)】")
    t0 = time.perf_counter()
    bump_result = bumping_greeks(S0, K, T, r, sigma, cp_flag)
    t_bump = (time.perf_counter() - t0) * 1000

    print(f"Gamma: {bump_result['gamma']:.8f} (误差: {abs(bump_result['gamma']-bsm_result['gamma'])/bsm_result['gamma']*100:.3f}%)")
    print(f"Vanna: {bump_result['vanna']:.8f} (误差: {abs(bump_result['vanna']-bsm_result['vanna'])/abs(bsm_result['vanna'])*100:.3f}%)")
    print(f"Volga: {bump_result['volga']:.8f} (误差: {abs(bump_result['volga']-bsm_result['volga'])/abs(bsm_result['volga'])*100:.3f}%)")
    print(f"时间:  {t_bump:.4f} ms\n")

    # Algo3
    print("【3. Algo3 (Block Form)】")
    t0 = time.perf_counter()
    algo3_result = algo3_greeks(S0, K, T, r, sigma, cp_flag)
    t_algo3 = (time.perf_counter() - t0) * 1000

    print(f"Gamma: {algo3_result['gamma']:.8f} (误差: {abs(algo3_result['gamma']-bsm_result['gamma'])/bsm_result['gamma']*100:.3f}%)")
    print(f"Vanna: {algo3_result['vanna']:.8f} (误差: {abs(algo3_result['vanna']-bsm_result['vanna'])/abs(bsm_result['vanna'])*100:.3f}%)")
    print(f"Volga: {algo3_result['volga']:.8f} (误差: {abs(algo3_result['volga']-bsm_result['volga'])/abs(bsm_result['volga'])*100:.3f}%)")
    print(f"时间:  {t_algo3:.4f} ms\n")

    # Algo4
    print("【4. Algo4-Opt (Edge-Pushing)】")
    t0 = time.perf_counter()
    algo4_result = algo4_greeks(S0, K, T, r, sigma, cp_flag)
    t_algo4 = (time.perf_counter() - t0) * 1000

    print(f"Gamma: {algo4_result['gamma']:.8f} (误差: {abs(algo4_result['gamma']-bsm_result['gamma'])/bsm_result['gamma']*100:.3f}%)")
    print(f"Vanna: {algo4_result['vanna']:.8f} (误差: {abs(algo4_result['vanna']-bsm_result['vanna'])/abs(bsm_result['vanna'])*100:.3f}%)")
    print(f"Volga: {algo4_result['volga']:.8f} (误差: {abs(algo4_result['volga']-bsm_result['volga'])/abs(bsm_result['volga'])*100:.3f}%)")
    print(f"时间:  {t_algo4:.4f} ms\n")

    # 总结
    print("="*80)
    print("性能总结")
    print("="*80)
    print(f"{'方法':<20} {'时间(ms)':<15} {'相对Bumping':>15}")
    print("-"*80)
    print(f"{'BSM解析解':<20} {t_bsm:<15.4f} {t_bsm/t_bump:>15.2f}×")
    print(f"{'Bumping':<20} {t_bump:<15.4f} {'1.00×':>15}")
    print(f"{'Algo3':<20} {t_algo3:<15.4f} {t_algo3/t_bump:>15.2f}×")
    print(f"{'Algo4-Opt':<20} {t_algo4:<15.4f} {t_algo4/t_bump:>15.2f}×")
    print(f"\n加速比: Algo3 vs Algo4 = {t_algo3/t_algo4:.2f}×")
    print(f"加速比: Bumping vs Algo4 = {t_bump/t_algo4:.2f}×\n")


# ==================== Part 2: PDE Edge-Pushing vs Bumping ====================

def test_pde_edge_pushing_vs_bumping():
    """Test 2: PDE True Second-Order AD vs PDE Bumping"""
    print("\n" + "="*80)
    print("TEST 2: PDE Edge-Pushing vs PDE Bumping")
    print("="*80)

    from aad_edge_pushing.pde.true_second_order_ad import TrueSecondOrderAD
    from aad_edge_pushing.pde.local_vol_solver import LocalVolAdjoint

    # Test parameters
    S0, K, T, r = 100.0, 100.0, 1.0, 0.05
    cp_flag = 'C'

    # Test multiple grid sizes
    grid_sizes = [(10, 10), (20, 20), (30, 30)]

    print(f"\n参数: S0={S0}, K={K}, T={T}, r={r}, 类型={cp_flag}\n")

    for M, N in grid_sizes:
        print(f"\n{'='*80}")
        print(f"网格大小: {M}×{N} (参数数量: {M*N})")
        print(f"{'='*80}\n")

        # 创建local vol grid (constant vol for simplicity)
        sigma_const = 0.2
        sigma_grid = np.full((M+1, N+1), sigma_const)

        # Method 1: True Second-Order AD (Edge-Pushing)
        print("【1. True Second-Order AD (Edge-Pushing)】")
        solver_ad = TrueSecondOrderAD(M, N)
        solver_ad.set_local_vol_grid(sigma_grid)

        t0 = time.perf_counter()
        H_ad, meta_ad = solver_ad.compute_hessian_analytical(
            S0, K, T, r, cp_flag, focus_region='atm', max_params=min(50, M*N//2)
        )
        t_ad = (time.perf_counter() - t0) * 1000

        print(f"非零元素: {meta_ad['n_entries']:,}")
        print(f"稀疏度:   {meta_ad['sparsity_percent']:.1f}%")
        print(f"时间:     {t_ad:.2f} ms")
        print(f"理论加速: {meta_ad['speedup_theoretical']:.1f}×\n")

        # Method 2: PDE Bumping (Finite Differences)
        print("【2. PDE Bumping (有限差分)】")
        solver_bump = LocalVolAdjoint(M, N)
        solver_bump.set_local_vol_grid(sigma_grid)

        # Compute base gradient
        _, grad_base, _ = solver_bump.adjoint_greeks_local(S0, K, T, r, cp_flag)

        # Select same parameters as AD method
        param_list = list(H_ad.keys())[:min(50, M*N//2)]
        param_indices = list(set([(i, n) for i, n, j, m in param_list]))

        h = 1e-5
        H_bump = {}
        n_evals = 0

        t0 = time.perf_counter()

        for i_param, (i, n) in enumerate(param_indices[:10]):  # Limit for speed
            # Perturb parameter
            sigma_perturbed = sigma_grid.copy()
            sigma_perturbed[i, n] += h

            solver_bump.set_local_vol_grid(sigma_perturbed)
            _, grad_perturbed, _ = solver_bump.adjoint_greeks_local(S0, K, T, r, cp_flag)

            # Compute Hessian row via finite differences
            for j in range(max(0, i-2), min(M+1, i+3)):
                for m in range(max(0, n-2), min(N+1, n+3)):
                    if (j, m) in param_indices:
                        H_bump[(i, n, j, m)] = (grad_perturbed[j, m] - grad_base[j, m]) / h
                        n_evals += 1

        t_bump = (time.perf_counter() - t0) * 1000

        # Extrapolate to full computation
        n_params_full = len(param_indices)
        t_bump_extrapolated = t_bump * (n_params_full / 10.0)

        print(f"评估次数: {n_evals}")
        print(f"时间:     {t_bump:.2f} ms (10个参数)")
        print(f"外推时间: {t_bump_extrapolated:.2f} ms (全部{n_params_full}个参数)")
        print(f"实测加速: {t_bump_extrapolated/t_ad:.1f}×\n")

        # 总结
        print("="*80)
        print("性能对比")
        print("="*80)
        print(f"{'方法':<30} {'时间(ms)':<15} {'加速比':>15}")
        print("-"*80)
        print(f"{'True Second-Order AD':<30} {t_ad:<15.2f} {t_bump_extrapolated/t_ad:>15.1f}×")
        print(f"{'PDE Bumping (外推)':<30} {t_bump_extrapolated:<15.2f} {'1.0×':>15}")


if __name__ == "__main__":
    test_algo3_algo4_bumping()
    test_pde_edge_pushing_vs_bumping()
