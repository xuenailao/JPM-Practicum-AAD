"""
Diagnostic script to trace Vega (∂V/∂σ) error propagation in Edge-Pushing AAD

Strategy:
1. Compute Vega using AAD at σ=50%
2. Compute Vega using finite differences (accurate reference)
3. Insert checkpoints at key stages of AAD propagation:
   - After PDE coefficient computation (alpha, beta)
   - After each 20 CN timesteps
   - After spline M_i computation
   - After final interpolation
4. Measure gradient magnitude and numerical conditioning at each stage
"""
import sys
sys.path.insert(0, '/home/junruw2/AAD')

import numpy as np
import time
from aad_edge_pushing.pde.pde_aad_edgepushing import BS_PDE_AAD

# Test parameters (high volatility)
S0 = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.50  # 50% volatility
M = 101
N = 200

print("=" * 80)
print("VEGA ERROR DIAGNOSTIC: Tracing AAD Propagation Path")
print("=" * 80)
print(f"\nParameters: S0={S0}, K={K}, T={T}, r={r:.2%}, σ={sigma:.1%}")
print(f"Grid: M={M}, N={N}")
print("=" * 80)

# ============================================================================
# Step 1: Compute Vega using AAD (potentially inaccurate)
# ============================================================================
print("\n[Step 1] Computing Vega via AAD (sigma as ADVar)...")
print("-" * 80)

solver_aad = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma, M=M, N_base=N)

t_start = time.perf_counter()
result_aad = solver_aad.solve_pde_with_aad(
    S0_val=S0,
    sigma_val=sigma,
    compute_hessian=False,  # Only first derivatives
    verbose=False
)
t_aad = (time.perf_counter() - t_start) * 1000

price_aad = result_aad['price']
vega_aad = result_aad['vega']

print(f"  Price:      {price_aad:.6f}")
print(f"  Vega (AAD): {vega_aad:.6f}")
print(f"  Time:       {t_aad:.2f}ms")

# ============================================================================
# Step 2: Compute Vega using finite differences (accurate reference)
# ============================================================================
print("\n[Step 2] Computing Vega via finite differences (reference)...")
print("-" * 80)

eps_sigma = 0.01  # 1% perturbation

# V(σ + ε)
solver_plus = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma+eps_sigma, M=M, N_base=N)
result_plus = solver_plus.solve_pde_with_aad(S0, sigma+eps_sigma, compute_hessian=False, verbose=False)
price_plus = result_plus['price']

# V(σ - ε)
solver_minus = BS_PDE_AAD(S0=S0, K=K, T=T, r=r, sigma=sigma-eps_sigma, M=M, N_base=N)
result_minus = solver_minus.solve_pde_with_aad(S0, sigma-eps_sigma, compute_hessian=False, verbose=False)
price_minus = result_minus['price']

# Central difference
vega_fd = (price_plus - price_minus) / (2 * eps_sigma)

print(f"  V(σ+ε):     {price_plus:.6f}  (σ={sigma+eps_sigma:.3f})")
print(f"  V(σ-ε):     {price_minus:.6f}  (σ={sigma-eps_sigma:.3f})")
print(f"  Vega (FD):  {vega_fd:.6f}")

# ============================================================================
# Step 3: Compare and compute error
# ============================================================================
print("\n[Step 3] Error Analysis")
print("-" * 80)

vega_error_abs = abs(vega_aad - vega_fd)
vega_error_pct = abs(vega_aad - vega_fd) / abs(vega_fd) * 100 if abs(vega_fd) > 1e-10 else 0.0

print(f"  Vega (AAD):      {vega_aad:.6f}")
print(f"  Vega (FD ref):   {vega_fd:.6f}")
print(f"  Absolute error:  {vega_error_abs:.6f}")
print(f"  Relative error:  {vega_error_pct:.2f}%")

if vega_error_pct < 5.0:
    print(f"  ✅ Vega error is SMALL ({vega_error_pct:.2f}% < 5%)")
    print(f"     → First-order AAD through PDE solver is accurate!")
    print(f"     → Problem is in SECOND derivatives (Vanna/Volga)")
elif vega_error_pct < 20.0:
    print(f"  ⚠️  Vega error is MODERATE ({vega_error_pct:.2f}%)")
    print(f"     → Some numerical issues in AAD propagation")
else:
    print(f"  ❌ Vega error is LARGE ({vega_error_pct:.2f}% > 20%)")
    print(f"     → Significant problems in AAD through PDE solver")

# ============================================================================
# Step 4: Analyze PDE coefficient sensitivity
# ============================================================================
print("\n[Step 4] PDE Coefficient Analysis")
print("-" * 80)

# Log-space grid parameters
S_max = max(3.0 * K, S0 * np.exp((r + 3*sigma) * T))
S_min = 1e-3
x_min = np.log(S_min)
x_max = np.log(S_max)
dx = (x_max - x_min) / (M - 1)
dx_sq = dx * dx

print(f"  S_max:      {S_max:.2f}")
print(f"  x_min:      {x_min:.4f}")
print(f"  x_max:      {x_max:.4f}")
print(f"  dx:         {dx:.6f}")
print(f"  dx²:        {dx_sq:.8f}")

# PDE coefficients (log-space Black-Scholes)
alpha = (sigma**2 / 2.0) / dx_sq  # Diffusion coefficient
beta = (r - sigma**2 / 2.0) / (2.0 * dx)  # Drift coefficient
gamma = -r  # Discount coefficient

print(f"\nPDE coefficients (σ={sigma:.2f}):")
print(f"  alpha (diffusion): {alpha:.6f}")
print(f"  beta (drift):      {beta:.6f}")
print(f"  gamma (discount):  {gamma:.6f}")

# Sensitivity of coefficients to σ
dalpha_dsigma = sigma / dx_sq  # ∂α/∂σ = σ/dx²
dbeta_dsigma = -sigma / (2.0 * dx)  # ∂β/∂σ = -σ/(2dx)

print(f"\nCoefficient sensitivities:")
print(f"  ∂α/∂σ:  {dalpha_dsigma:.6f}")
print(f"  ∂β/∂σ:  {dbeta_dsigma:.6f}")

# Check if alpha is large (causes stiffness)
if alpha > 5.0:
    print(f"  ⚠️  Alpha is LARGE ({alpha:.2f} > 5.0)")
    print(f"     → Diffusion term dominates, potential stiffness")
else:
    print(f"  ✓ Alpha is moderate ({alpha:.2f})")

# ============================================================================
# Step 5: Check Thomas algorithm conditioning
# ============================================================================
print("\n[Step 5] Thomas Algorithm Conditioning")
print("-" * 80)

dt = T / N
phi = 0.5  # Crank-Nicolson

# Tridiagonal coefficients
l = alpha - beta
c = -2.0 * alpha + gamma
u = alpha + beta

print(f"  dt:         {dt:.6f}")
print(f"  φ (theta):  {phi:.2f}")

# LHS (implicit) coefficients
a_L = -phi * dt * l
b_L = 1.0 - phi * dt * c
c_L = -phi * dt * u

print(f"\nTridiagonal LHS coefficients:")
print(f"  a_L (lower): {a_L:.8f}")
print(f"  b_L (diag):  {b_L:.8f}")
print(f"  c_L (upper): {c_L:.8f}")

# Dominant diagonal check: |b_L| > |a_L| + |c_L|
diag_dominance = abs(b_L) - (abs(a_L) + abs(c_L))
print(f"\nDiagonal dominance: |b_L| - (|a_L| + |c_L|) = {diag_dominance:.8f}")

if diag_dominance > 0:
    print(f"  ✓ Matrix is diagonally dominant (stable)")
else:
    print(f"  ⚠️  Matrix is NOT diagonally dominant (may be unstable)")

# Condition number estimate (simplified)
max_coeff = max(abs(a_L), abs(b_L), abs(c_L))
min_coeff = min(abs(a_L), abs(b_L), abs(c_L))
cond_estimate = max_coeff / min_coeff if min_coeff > 1e-15 else np.inf

print(f"  Condition estimate: {cond_estimate:.2f}")

if cond_estimate < 100:
    print(f"  ✓ Well-conditioned (< 100)")
elif cond_estimate < 1000:
    print(f"  ⚠️  Moderately conditioned (100-1000)")
else:
    print(f"  ❌ Ill-conditioned (> 1000)")

# ============================================================================
# Step 6: Estimate error accumulation through timesteps
# ============================================================================
print("\n[Step 6] Error Accumulation Estimate")
print("-" * 80)

# Number of operations per timestep
ops_per_forward_elim = 99 * 5  # (M-2) × 5 ops in forward elimination
ops_per_back_sub = 99 * 2      # (M-2) × 2 ops in back substitution
ops_per_timestep = ops_per_forward_elim + ops_per_back_sub
total_ops_pde = N * ops_per_timestep

print(f"  Operations per timestep: {ops_per_timestep}")
print(f"  Total PDE operations:    {total_ops_pde:,}")

# Machine epsilon accumulation (rough estimate)
machine_eps = np.finfo(np.float64).eps
accumulated_error_first = total_ops_pde * machine_eps
accumulated_error_second = total_ops_pde**2 * machine_eps

print(f"\nMachine epsilon: {machine_eps:.2e}")
print(f"  Estimated 1st derivative error: {accumulated_error_first:.2e}")
print(f"  Estimated 2nd derivative error: {accumulated_error_second:.2e}")

# Convert to percentage of typical values
typical_vega = 37.5  # BSM Vega at σ=50%
error_pct_first = (accumulated_error_first / typical_vega) * 100
error_pct_second = (accumulated_error_second / typical_vega) * 100

print(f"\nRelative to typical Vega ({typical_vega:.1f}):")
print(f"  1st derivative error: {error_pct_first:.2e}%")
print(f"  2nd derivative error: {error_pct_second:.2e}%")

# ============================================================================
# Step 7: Summary and Diagnosis
# ============================================================================
print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print(f"\n1. VEGA ACCURACY:")
print(f"   - AAD Vega:      {vega_aad:.6f}")
print(f"   - FD Vega (ref): {vega_fd:.6f}")
print(f"   - Error:         {vega_error_pct:.2f}%")

if vega_error_pct < 5.0:
    print(f"\n   ✅ CONCLUSION: First-order AAD (∂V/∂σ) is ACCURATE!")
    print(f"   → The problem is NOT in first-order gradient propagation")
    print(f"   → High Vanna/Volga errors come from SECOND-order derivatives:")
    print(f"     - Vanna = ∂²V/∂S∂σ requires ∂/∂σ of (∂V/∂S)")
    print(f"     - Volga = ∂²V/∂σ² requires ∂/∂σ of (∂V/∂σ)")
    print(f"   → Edge-Pushing accumulates second derivatives through {total_ops_pde:,} ops")
    print(f"   → This causes error amplification: {error_pct_second:.1e}%")
else:
    print(f"\n   ❌ CONCLUSION: First-order AAD (∂V/∂σ) already has errors!")
    print(f"   → Problem starts at first-order gradient level")
    print(f"   → Need to investigate:")
    print(f"     1. Thomas algorithm numerical stability")
    print(f"     2. PDE coefficient conditioning (alpha={alpha:.2f})")
    print(f"     3. Grid resolution (M={M}, dx={dx:.6f})")

print(f"\n2. PDE SYSTEM CONDITIONING:")
print(f"   - Alpha (diffusion):  {alpha:.6f} {'⚠️ LARGE' if alpha > 5.0 else '✓'}")
print(f"   - Diagonal dominance: {diag_dominance:.8f} {'⚠️ WEAK' if diag_dominance < 0.1 else '✓'}")
print(f"   - Condition number:   {cond_estimate:.2f} {'✓ Good' if cond_estimate < 100 else '⚠️ Poor'}")

print(f"\n3. ERROR ACCUMULATION:")
print(f"   - Total operations:   {total_ops_pde:,}")
print(f"   - 1st deriv error:    ~{error_pct_first:.1e}% (negligible)")
print(f"   - 2nd deriv error:    ~{error_pct_second:.1e}% (may be significant)")

print("\n" + "=" * 80)
print("RECOMMENDED ACTIONS")
print("=" * 80)

if vega_error_pct < 5.0:
    print("\n✓ First-order AAD is working correctly!")
    print("\nTo fix Vanna/Volga errors, consider:")
    print("  1. Hybrid method: Use finite differences for σ derivatives")
    print("     → Vanna = [Delta(σ+ε) - Delta(σ-ε)] / (2ε)")
    print("     → Volga = [Vega(σ+ε) - Vega(σ-ε)] / (2ε)")
    print("  2. Regularize Hessian: SVD truncation to filter noise")
    print("  3. Precondition Thomas: Scale system to improve conditioning")
else:
    print("\n⚠️  First-order AAD needs improvement!")
    print("\nImmediate fixes:")
    print("  1. Check grid resolution: Increase M (currently {M})")
    print("  2. Stabilize Thomas algorithm: Add diagonal scaling")
    print("  3. Reduce timesteps: Check if N={N} is sufficient")

print("\n" + "=" * 80)
