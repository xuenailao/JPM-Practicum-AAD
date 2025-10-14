"""
Black-Scholes-Merton Greeks计算函数
导出给示例使用
"""

from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np


def bsm_price(S0, K, T, r, sigma, cp_flag='C'):
    """Black-Scholes-Merton解析解"""
    d1 = (log(S0/K) + (r+0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    if cp_flag == 'C':
        return S0*norm.cdf(d1) - K*exp(-r*T)*norm.cdf(d2)
    else:
        return K*exp(-r*T)*norm.cdf(-d2) - S0*norm.cdf(-d1)


def bsm_analytical_greeks(S0, K, T, r, sigma, cp_flag='C'):
    """BSM解析Greeks（一阶和二阶）"""
    d1 = (log(S0/K) + (r+0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    # 一阶Greeks
    delta = norm.cdf(d1) if cp_flag == 'C' else -norm.cdf(-d1)
    vega = S0 * norm.pdf(d1) * sqrt(T)
    theta = -(S0 * norm.pdf(d1) * sigma) / (2 * sqrt(T)) - r * K * exp(-r*T) * norm.cdf(d2)

    if cp_flag == 'C':
        rho = K * T * exp(-r*T) * norm.cdf(d2)
    else:
        rho = -K * T * exp(-r*T) * norm.cdf(-d2)

    # 二阶Greeks
    gamma = norm.pdf(d1) / (S0 * sigma * sqrt(T))
    vanna = -norm.pdf(d1) * d2 / sigma  # ∂²V/∂S∂σ
    volga = S0 * norm.pdf(d1) * sqrt(T) * d1 * d2 / sigma  # ∂²V/∂σ²

    return {
        'price': bsm_price(S0, K, T, r, sigma, cp_flag),
        'delta': delta,
        'vega': vega,
        'theta': theta,
        'rho': rho,
        'gamma': gamma,
        'vanna': vanna,
        'volga': volga
    }


def bumping_greeks(S0, K, T, r, sigma, cp_flag='C'):
    """Bumping方法计算Greeks"""
    h = 1e-5

    price = bsm_price(S0, K, T, r, sigma, cp_flag)

    # 一阶Greeks
    delta = (bsm_price(S0 + h, K, T, r, sigma, cp_flag) - price) / h
    vega = (bsm_price(S0, K, T, r, sigma + h, cp_flag) - price) / h
    theta = (bsm_price(S0, K, T + h, r, sigma, cp_flag) - price) / h
    rho = (bsm_price(S0, K, T, r + h, sigma, cp_flag) - price) / h

    # 二阶Greeks (中心差分)
    gamma = (bsm_price(S0 + h, K, T, r, sigma, cp_flag) -
             2*price +
             bsm_price(S0 - h, K, T, r, sigma, cp_flag)) / (h**2)

    volga = (bsm_price(S0, K, T, r, sigma + h, cp_flag) -
             2*price +
             bsm_price(S0, K, T, r, sigma - h, cp_flag)) / (h**2)

    # Vanna: ∂²V/∂S∂σ
    vanna = (bsm_price(S0 + h, K, T, r, sigma + h, cp_flag) -
             bsm_price(S0 + h, K, T, r, sigma - h, cp_flag) -
             bsm_price(S0 - h, K, T, r, sigma + h, cp_flag) +
             bsm_price(S0 - h, K, T, r, sigma - h, cp_flag)) / (4 * h**2)

    return {
        'price': price,
        'delta': delta,
        'vega': vega,
        'theta': theta,
        'rho': rho,
        'gamma': gamma,
        'vanna': vanna,
        'volga': volga
    }


def algo4_greeks(S0, K, T, r, sigma, cp_flag='C'):
    """使用Algo4-Optimized计算Greeks"""
    from aad_edge_pushing.aad.core.var import ADVar
    from aad_edge_pushing.aad.core.tape import global_tape
    from aad_edge_pushing.aad.ops.transcendental import exp as ad_exp, log as ad_log, erf
    from aad_edge_pushing.algo3.algo4_optimized import algo4_optimized

    global_tape.reset()

    # 创建AD变量
    S_ad = ADVar(S0)
    K_ad = ADVar(K)
    r_ad = ADVar(r)
    sigma_ad = ADVar(sigma)
    T_ad = ADVar(T)

    # BSM公式（AD版本）
    sqrt_T = T_ad ** 0.5
    sqrt_2 = np.sqrt(2.0)

    d1 = (ad_log(S_ad / K_ad) + (r_ad + sigma_ad * sigma_ad * 0.5) * T_ad) / (sigma_ad * sqrt_T)
    d2 = d1 - sigma_ad * sqrt_T

    N_d1 = 0.5 * (1.0 + erf(d1 / sqrt_2))
    N_d2 = 0.5 * (1.0 + erf(d2 / sqrt_2))

    if cp_flag == 'C':
        price_ad = S_ad * N_d1 - K_ad * ad_exp(-r_ad * T_ad) * N_d2
    else:
        price_ad = K_ad * ad_exp(-r_ad * T_ad) * N_d2 - S_ad * N_d1

    # 计算Hessian
    inputs = [S_ad, K_ad, r_ad, sigma_ad, T_ad]
    H = algo4_optimized(price_ad, inputs)

    # 计算一阶导数（反向传播）
    global_tape.reset_adjoints()
    price_ad.adjoint = 1.0
    for node in reversed(global_tape.nodes):
        node.backward()

    # 提取Greeks
    return {
        'price': price_ad.val,
        'delta': S_ad.adjoint,      # ∂V/∂S
        'vega': sigma_ad.adjoint,   # ∂V/∂σ
        'theta': T_ad.adjoint,      # ∂V/∂T
        'rho': r_ad.adjoint,        # ∂V/∂r
        'gamma': H[0, 0],           # ∂²V/∂S²
        'vanna': H[0, 3],           # ∂²V/∂S∂σ
        'volga': H[3, 3],           # ∂²V/∂σ²
    }
