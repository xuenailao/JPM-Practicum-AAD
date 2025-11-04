"""
Detailed Vega error analysis: Compare numerical vs analytical Vega

Key finding from previous diagnostic:
- Vega (AAD): 28.43
- Vega (FD):  31.51
- Error: 9.77% at FIRST-ORDER level

This suggests the issue is in how AAD propagates ∂V/∂σ through the PDE solver,
NOT just in second derivatives.

Strategy:
1. Compare AAD Vega with BSM analytical Vega
2. Test different grid resolutions to see if it's a discretization error
3. Test at different volatilities to see if error scales with σ
4. Identify if error comes from:
   - PDE discretization (space/time)
   - Cubic spline interpolation
   - AAD gradient propagation through Thomas algorithm
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
import time
from scipy.stats import norm
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

def bsm_analytical_greeks(S, K, T, r, sigma):
    """Black-Scholes-Merton analytical Greeks for European call"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1)

    price = S*N_d1 - K*np.exp(-r*T)*N_d2
    delta = N_d1
    gamma = n_d1 / (S*sigma*np.sqrt(T))
    vega = S*n_d1*np.sqrt(T)

    return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega}

# Test parameters
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05

print("=" * 80)
print("DETAILED VEGA ERROR ANALYSIS")
print("=" * 80)

# ============================================================================
# Test 1: Vega error vs volatility
# ============================================================================
print("\n[Test 1] Vega Error vs Volatility")
print("-" * 80)

sigmas = [0.10, 0.20, 0.30, 0.40, 0.50]
M, N = 101, 200

print(f"Grid: M={M}, N={N}")
print(f"\n{'σ':<8} {'BSM Vega':<12} {'AAD Vega':<12} {'FD Vega':<12} {'AAD Err%':<12} {'FD Err%':<12}")
print("-" * 80)

for sigma in sigmas:
    # BSM analytical
    bsm = bsm_analytical_greeks(S0, K, T, r, sigma)
    vega_bsm = bsm['vega']

    # AAD Vega
    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N)
    result_aad = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)
    vega_aad = result_aad['vega']

    # FD Vega
    eps_sigma = 0.01
    solver_p = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma+eps_sigma, M=M, N_base=N)
    solver_m = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma-eps_sigma, M=M, N_base=N)
    price_p = solver_p.solve_pde_with_aad(S0, sigma+eps_sigma, compute_hessian=False, verbose=False)['price']
    price_m = solver_m.solve_pde_with_aad(S0, sigma-eps_sigma, compute_hessian=False, verbose=False)['price']
    vega_fd = (price_p - price_m) / (2 * eps_sigma)

    # Errors
    err_aad = abs(vega_aad - vega_bsm) / vega_bsm * 100
    err_fd = abs(vega_fd - vega_bsm) / vega_bsm * 100

    print(f"{sigma:<8.2f} {vega_bsm:<12.6f} {vega_aad:<12.6f} {vega_fd:<12.6f} {err_aad:<12.2f} {err_fd:<12.2f}")

# ============================================================================
# Test 2: Vega error vs grid resolution
# ============================================================================
print("\n[Test 2] Vega Error vs Grid Resolution (σ=50%)")
print("-" * 80)

sigma = 0.50
bsm = bsm_analytical_greeks(S0, K, T, r, sigma)
vega_bsm = bsm['vega']

grid_sizes = [(51, 100), (101, 200), (201, 400), (301, 600)]

print(f"σ = {sigma:.2f}, BSM Vega = {vega_bsm:.6f}")
print(f"\n{'M':<8} {'N':<8} {'AAD Vega':<12} {'Error %':<12} {'Time (ms)':<12}")
print("-" * 60)

for M, N in grid_sizes:
    t_start = time.perf_counter()
    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N)
    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)
    t_elapsed = (time.perf_counter() - t_start) * 1000

    vega_aad = result['vega']
    err = abs(vega_aad - vega_bsm) / vega_bsm * 100

    print(f"{M:<8} {N:<8} {vega_aad:<12.6f} {err:<12.2f} {t_elapsed:<12.1f}")

# ============================================================================
# Test 3: Compare PDE price vs BSM at different σ
# ============================================================================
print("\n[Test 3] Price Error vs Volatility (M=101, N=200)")
print("-" * 80)

M, N = 101, 200
print(f"\n{'σ':<8} {'BSM Price':<12} {'PDE Price':<12} {'Price Err%':<12}")
print("-" * 50)

for sigma in sigmas:
    bsm = bsm_analytical_greeks(S0, K, T, r, sigma)
    price_bsm = bsm['price']

    solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N)
    result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)
    price_pde = result['price']

    err = abs(price_pde - price_bsm) / price_bsm * 100

    print(f"{sigma:<8.2f} {price_bsm:<12.6f} {price_pde:<12.6f} {err:<12.2f}")

# ============================================================================
# Test 4: Analyze PDE coefficient behavior
# ============================================================================
print("\n[Test 4] PDE Coefficient Scaling with Volatility")
print("-" * 80)

M = 101
print(f"\n{'σ':<8} {'S_max':<10} {'dx':<12} {'alpha':<12} {'beta':<12} {'CFL':<10}")
print("-" * 70)

for sigma in sigmas:
    S_max = max(3.0 * K, S0 * np.exp((r + 3*sigma) * T))
    S_min = 1e-3
    x_min = np.log(S_min)
    x_max = np.log(S_max)
    dx = (x_max - x_min) / (M - 1)
    dx_sq = dx * dx

    alpha = (sigma**2 / 2.0) / dx_sq
    beta = (r - sigma**2 / 2.0) / (2.0 * dx)

    # CFL-like condition for parabolic PDE: dt < dx²/(2*alpha)
    dt_critical = dx_sq / (2 * alpha)
    dt_actual = T / N  # N=200
    cfl_ratio = dt_actual / dt_critical

    print(f"{sigma:<8.2f} {S_max:<10.2f} {dx:<12.6f} {alpha:<12.4f} {beta:<12.4f} {cfl_ratio:<10.4f}")

print(f"\nNote: CFL ratio > 1 means dt > critical dt (may need implicit scheme)")
print(f"      We use Crank-Nicolson (unconditionally stable), but large ratios → errors")

# ============================================================================
# Summary and Conclusion
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY AND ROOT CAUSE")
print("=" * 80)

# Final test at σ=50%
sigma = 0.50
M, N = 101, 200
bsm = bsm_analytical_greeks(S0, K, T, r, sigma)

solver = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N)
result = solver.solve_pde_with_aad(S0, sigma, compute_hessian=False, verbose=False)

price_err = abs(result['price'] - bsm['price']) / bsm['price'] * 100
delta_err = abs(result['delta'] - bsm['delta']) / bsm['delta'] * 100
gamma_err = abs(result['gamma'] - bsm['gamma']) / bsm['gamma'] * 100
vega_err = abs(result['vega'] - bsm['vega']) / bsm['vega'] * 100

print(f"\nAt σ=50%, M={M}, N={N}:")
print(f"  Price error:  {price_err:.2f}%")
print(f"  Delta error:  {delta_err:.2f}%")
print(f"  Gamma error:  {gamma_err:.2f}%")
print(f"  Vega error:   {vega_err:.2f}%")

print(f"\nROOT CAUSE ANALYSIS:")

if vega_err < 5.0:
    print(f"  ✅ Vega error < 5% → AAD propagation is accurate")
    print(f"     Problem is in SECOND derivatives (Vanna/Volga)")
elif vega_err < 15.0:
    print(f"  ⚠️  Vega error ~{vega_err:.1f}% → Moderate AAD propagation error")
    print(f"     Likely causes:")
    print(f"     1. PDE discretization error (need finer grid)")
    print(f"     2. High alpha coefficient ({(sigma**2/2.0)/((x_max-x_min)/(M-1))**2:.2f}) causes stiffness")
    print(f"     3. AAD gradient accumulation through 200 timesteps")
else:
    print(f"  ❌ Vega error > 15% → Significant AAD propagation issues")

# Check if error scales with sigma
print(f"\nERROR SCALING:")
print(f"  If error increases with σ → PDE stiffness/discretization issue")
print(f"  If error constant with σ → AAD implementation issue")

print("\n" + "=" * 80)
