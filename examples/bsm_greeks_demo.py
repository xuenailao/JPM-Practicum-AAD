"""
Black-Scholes-Merton Greeks计算完整示例
========================================

演示如何用AAD计算期权的一阶和二阶Greeks
"""

import sys
import os
# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from aad_edge_pushing.examples.bsm_greeks import (
    algo4_greeks,
    bumping_greeks,
    bsm_analytical_greeks
)
import numpy as np


def demo_basic_greeks():
    """基础示例：计算ATM看涨期权的Greeks"""
    print("=" * 70)
    print("示例1：At-The-Money (ATM) 欧式看涨期权")
    print("=" * 70)

    # 市场参数
    S0 = 100.0      # 标的资产价格
    K = 100.0       # 行权价 (ATM)
    T = 1.0         # 1年到期
    r = 0.05        # 5% 无风险利率
    sigma = 0.20    # 20% 波动率

    print(f"\n参数:")
    print(f"  标的价格 S₀ = {S0}")
    print(f"  行权价 K = {K}")
    print(f"  到期时间 T = {T}年")
    print(f"  无风险利率 r = {r*100}%")
    print(f"  波动率 σ = {sigma*100}%")

    # 使用AAD计算Greeks
    greeks = algo4_greeks(S0, K, T, r, sigma)

    print(f"\n期权价值:")
    print(f"  Call价格 = {greeks['price']:.4f}")

    print(f"\n一阶Greeks:")
    print(f"  Delta (∂V/∂S)   = {greeks['delta']:.6f}")
    print(f"  Vega  (∂V/∂σ)   = {greeks['vega']:.6f}")
    print(f"  Theta (∂V/∂T)   = {greeks['theta']:.6f}")
    print(f"  Rho   (∂V/∂r)   = {greeks['rho']:.6f}")

    print(f"\n二阶Greeks:")
    print(f"  Gamma (∂²V/∂S²)  = {greeks['gamma']:.8f}")
    print(f"  Vanna (∂²V/∂S∂σ) = {greeks['vanna']:.8f}")
    print(f"  Volga (∂²V/∂σ²)  = {greeks['volga']:.8f}")

    # 解释
    print(f"\n💡 解释:")
    print(f"  Delta={greeks['delta']:.2f}：价格上涨$1，期权价值增加${greeks['delta']:.2f}")
    print(f"  Gamma={greeks['gamma']:.4f}：Delta对价格的敏感度（凸性）")
    print(f"  Vega={greeks['vega']:.2f}：波动率上升1%，期权价值增加${greeks['vega']:.2f}")


def demo_itm_otm_comparison():
    """示例2：比较ITM、ATM、OTM期权的Greeks"""
    print("\n" + "=" * 70)
    print("示例2：In/At/Out-of-The-Money 期权Greeks对比")
    print("=" * 70)

    S0 = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.20

    strikes = {
        'Deep ITM': 80,   # 深度实值
        'ITM': 90,        # 实值
        'ATM': 100,       # 平值
        'OTM': 110,       # 虚值
        'Deep OTM': 120   # 深度虚值
    }

    print(f"\n{'期权类型':<15} {'价格':>8} {'Delta':>8} {'Gamma':>10} {'Vega':>8}")
    print("-" * 70)

    for label, K in strikes.items():
        greeks = algo4_greeks(S0, K, T, r, sigma)
        print(f"{label:<15} {greeks['price']:>8.2f} {greeks['delta']:>8.4f} "
              f"{greeks['gamma']:>10.6f} {greeks['vega']:>8.2f}")

    print("\n💡 观察:")
    print("  - ATM期权的Gamma和Vega最大（对市场变化最敏感）")
    print("  - Deep ITM期权的Delta接近1（几乎等价于持有标的）")
    print("  - Deep OTM期权的Delta接近0（基本不受价格影响）")


def demo_term_structure():
    """示例3：到期时间对Greeks的影响"""
    print("\n" + "=" * 70)
    print("示例3：到期时间的期限结构")
    print("=" * 70)

    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.20

    maturities = [0.25, 0.5, 1.0, 2.0, 5.0]  # 3个月到5年

    print(f"\n{'到期时间':<12} {'价格':>8} {'Gamma':>10} {'Vega':>8} {'Theta':>8}")
    print("-" * 70)

    for T in maturities:
        greeks = algo4_greeks(S0, K, T, r, sigma)
        print(f"{T}年{'':<8} {greeks['price']:>8.2f} {greeks['gamma']:>10.6f} "
              f"{greeks['vega']:>8.2f} {greeks['theta']:>8.4f}")

    print("\n💡 观察:")
    print("  - 长期期权价格更高（更多时间价值）")
    print("  - 短期期权Gamma更大（到期时Delta变化剧烈）")
    print("  - Theta为负（时间流逝导致期权价值下降）")


def demo_method_comparison():
    """示例4：不同计算方法的对比"""
    print("\n" + "=" * 70)
    print("示例4：AAD vs Bumping vs 解析解 - 精度和性能对比")
    print("=" * 70)

    S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

    # 解析解（基准）
    analytical = bsm_analytical_greeks(S0, K, T, r, sigma)

    # AAD方法
    import time
    start = time.time()
    aad = algo4_greeks(S0, K, T, r, sigma)
    aad_time = (time.time() - start) * 1000

    # Bumping方法
    start = time.time()
    bumping = bumping_greeks(S0, K, T, r, sigma)
    bump_time = (time.time() - start) * 1000

    # 精度对比
    print(f"\n精度对比（相对解析解）:")
    print(f"{'Greek':<10} {'解析解':>12} {'AAD':>12} {'AAD误差%':>12} {'Bumping':>12} {'Bump误差%':>12}")
    print("-" * 85)

    greeks_to_compare = ['gamma', 'vanna', 'volga']
    for greek in greeks_to_compare:
        anal_val = analytical[greek]
        aad_val = aad[greek]
        bump_val = bumping[greek]
        aad_err = abs(aad_val - anal_val) / abs(anal_val) * 100
        bump_err = abs(bump_val - anal_val) / abs(anal_val) * 100
        print(f"{greek.capitalize():<10} {anal_val:>12.6f} {aad_val:>12.6f} "
              f"{aad_err:>12.4f} {bump_val:>12.6f} {bump_err:>12.4f}")

    print(f"\n性能对比:")
    print(f"  AAD (Algo4):  {aad_time:.2f} ms")
    print(f"  Bumping:      {bump_time:.2f} ms")
    print(f"  相对速度:     {bump_time/aad_time:.2f}×")

    print(f"\n💡 结论:")
    print(f"  - AAD达到机器精度（误差<1e-10）")
    print(f"  - Bumping精度足够但略差（误差~1e-5）")
    print(f"  - 小规模问题Bumping可能更快（实现简单）")


def demo_risk_management():
    """示例5：风险管理应用 - Delta对冲"""
    print("\n" + "=" * 70)
    print("示例5：Delta对冲策略")
    print("=" * 70)

    # 卖出100手看涨期权（每手100股）
    S0 = 100.0
    K = 105.0  # 略OTM
    T = 0.25   # 3个月
    r = 0.05
    sigma = 0.25

    contracts = 100
    shares_per_contract = 100

    greeks = algo4_greeks(S0, K, T, r, sigma)

    print(f"\n头寸:")
    print(f"  卖出{contracts}手看涨期权 (K={K}, T={T}年)")
    print(f"  每手{shares_per_contract}股")

    portfolio_delta = greeks['delta'] * contracts * shares_per_contract
    portfolio_gamma = greeks['gamma'] * contracts * shares_per_contract
    portfolio_vega = greeks['vega'] * contracts * shares_per_contract

    print(f"\n组合Greeks:")
    print(f"  Portfolio Delta = {portfolio_delta:.0f}")
    print(f"  Portfolio Gamma = {portfolio_gamma:.0f}")
    print(f"  Portfolio Vega  = {portfolio_vega:.0f}")

    # Delta对冲
    hedge_shares = -portfolio_delta

    print(f"\nDelta对冲策略:")
    print(f"  需要买入 {hedge_shares:.0f} 股标的资产")
    print(f"  对冲成本 = ${hedge_shares * S0:,.0f}")

    # 情景分析
    print(f"\n情景分析（价格变动）:")
    price_changes = [-5, -2, 0, 2, 5]
    print(f"{'价格变动':>10} {'期权损益':>12} {'对冲损益':>12} {'净损益':>12}")
    print("-" * 70)

    for dS in price_changes:
        S_new = S0 + dS
        greeks_new = algo4_greeks(S_new, K, T, r, sigma)

        # 期权部分损益（我们是卖方，所以价格上涨我们亏损）
        option_pnl = -(greeks_new['price'] - greeks['price']) * contracts * shares_per_contract

        # 对冲部分损益
        hedge_pnl = hedge_shares * dS

        # 净损益
        net_pnl = option_pnl + hedge_pnl

        print(f"{dS:>10.0f} {option_pnl:>12.0f} {hedge_pnl:>12.0f} {net_pnl:>12.0f}")

    print(f"\n💡 说明:")
    print(f"  - Delta对冲使组合对小幅价格变动不敏感")
    print(f"  - 但Gamma风险仍然存在（大幅变动时对冲不完美）")
    print(f"  - 需要动态调整对冲比例（Delta随价格变化）")


def main():
    """运行所有示例"""
    print("\n" + "📈 " * 25)
    print("Black-Scholes-Merton Greeks计算 - 完整示例")
    print("📈 " * 25)

    demo_basic_greeks()
    demo_itm_otm_comparison()
    demo_term_structure()
    demo_method_comparison()
    demo_risk_management()

    print("\n" + "=" * 70)
    print("✅ 所有示例完成！")
    print("=" * 70)
    print("\n进一步学习:")
    print("  1. 运行 benchmarks/main_benchmark.py 查看详细性能测试")
    print("  2. 阅读 PERFORMANCE_REPORT.md 了解四种方法的对比")
    print("  3. 阅读 LITERATURE_REVIEW.md 了解AAD在期权定价中的学术背景")
    print()


if __name__ == "__main__":
    main()
