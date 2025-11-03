"""
Analysis: Why sigma_j = sigma_grid[j-1]?

This design allows for LOCAL VOLATILITY models where sigma varies with S.

But for Black-Scholes (constant volatility), this creates issues!
"""
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("\n" + "="*100)
print("ANALYSIS: sigma_grid Design in PDE Solver")
print("="*100)

print("\n" + "-"*100)
print("1. Current Implementation")
print("-"*100)

print("""
Code structure:
    for j in range(1, M-1):              # j = 1, 2, ..., M-2
        S_j = S_grid[j]                   # Spatial grid point
        sigma_j = sigma_grid[j-1]         # ← WHY j-1?

        diff = sigma_j^2 * S_j^2 / 2      # Diffusion coefficient
""")

print("\nGrid indices:")
print("  S_grid:     [S_0, S_1, S_2, ..., S_{M-2}, S_{M-1}]  (length M)")
print("  sigma_grid: [σ_0, σ_1, σ_2, ..., σ_{M-2}]          (length M-1)")
print("              ↑")
print("              j=1 uses σ_0")
print("              j=2 uses σ_1")
print("              ...")

print("\n" + "-"*100)
print("2. Design Purpose: Local Volatility")
print("-"*100)

print("""
This design supports LOCAL VOLATILITY models:
    σ = σ(S, t)  ← Volatility depends on stock price S

Example:
    σ(S) = σ_0 * (S/S_0)^β  (CEV model)

In this case:
    - Each grid point S_j has different volatility σ_j
    - sigma_grid = [σ(S_1), σ(S_2), ..., σ(S_{M-2})]
    - Makes sense to use sigma_grid[j-1] for interior point j
""")

print("\n" + "-"*100)
print("3. Problem for Black-Scholes (Constant Volatility)")
print("-"*100)

print("""
For BLACK-SCHOLES with CONSTANT σ:
    σ(S) = σ  (same everywhere)

Current implementation (Method A):
    sigma_var = ADVar(sigma, requires_grad=True)
    sigma_grid = [sigma_var] * (M-1)

This means:
    sigma_grid = [σ_var, σ_var, ..., σ_var]  ← All point to SAME ADVar!
                  ↑      ↑             ↑
                  same   same          same
""")

print("\n✅ This is CORRECT for constant volatility:")
print("   - All sigma_j share the same ADVar")
print("   - When computing ∂V/∂σ, gradient accumulates from all grid points")
print("   - Single parameter optimization")

print("\n" + "-"*100)
print("4. Index Mapping Analysis")
print("-"*100)

M = 51  # Example
print(f"\nFor M={M}:")
print(f"  S_grid has {M} points: S_0, S_1, ..., S_{M-1}")
print(f"  Interior points: S_1, S_2, ..., S_{M-2} (total {M-2} points)")
print(f"  sigma_grid has {M-1} elements")

print("\nLoop iteration:")
print(f"  j=1:    S_1  uses sigma_grid[0]")
print(f"  j=2:    S_2  uses sigma_grid[1]")
print(f"  ...")
print(f"  j={M-2}: S_{M-2} uses sigma_grid[{M-3}]")

print(f"\n⚠️ Question: Why {M-1} sigma values for {M-2} interior points?")

print("\n" + "-"*100)
print("5. Correct Design Analysis")
print("-"*100)

print("""
OPTION A: Current design (sigma_grid length = M-1)
    Pros:
    - Works for local volatility σ(S) at each grid point
    - Can represent σ at S_1, S_2, ..., S_{M-1}

    Cons:
    - One extra σ value (not used for interior points)
    - Confusing indexing (j-1)

OPTION B: Alternative (sigma_grid length = M-2)
    sigma_grid = [σ_var] * (M-2)

    for j in range(1, M-1):
        sigma_j = sigma_grid[j-1]  # j=1 → sigma_grid[0]

    Pros:
    - Exact match: M-2 interior points, M-2 sigma values
    - Clear correspondence

    Cons:
    - Doesn't match current implementation

OPTION C: Direct indexing (sigma_grid length = M)
    sigma_grid = [0] + [σ_var]*(M-2) + [0]  # Include boundaries

    for j in range(1, M-1):
        sigma_j = sigma_grid[j]  # Direct indexing!

    Pros:
    - Intuitive: sigma_grid[j] for S_grid[j]
    - No -1 offset
""")

print("\n" + "-"*100)
print("6. Impact on Vega Computation")
print("-"*100)

print("""
For constant volatility Black-Scholes:

Current implementation:
    sigma_var = ADVar(sigma, requires_grad=True)
    sigma_grid = [sigma_var] * (M-1)

    All sigma_j = sigma_var (same object)

When computing Vega = ∂V/∂σ:
    - Each grid point j contributes via diff = sigma_j^2 * S_j^2 / 2
    - Gradient flows back through sigma_sq = sigma_j * sigma_j
    - Since all sigma_j are the SAME ADVar:
      ∂V/∂σ = Σ_j (∂V/∂sigma_j)  ← Accumulated gradient

✅ THIS IS CORRECT for constant volatility!

The design supports both:
- Local volatility: Different sigma_j (independent ADVars)
- Constant volatility: Same sigma_j (shared ADVar)
""")

print("\n" + "-"*100)
print("7. Why Vega is Wrong - It's NOT This Design!")
print("-"*100)

print("""
The sigma_grid design is FINE. The real problem is:

At high σ, the PDE solution V_PDE itself is wrong due to numerical damping.

Evidence:
    σ=0.20: V_PDE = 10.354, V_BS = 10.451 (0.93% error)
    σ=0.30: V_PDE = 12.115, V_BS = 14.231 (14.87% error!)

AAD correctly computes ∂V_PDE/∂σ, but V_PDE is wrong!

The design sigma_j = sigma_grid[j-1] allows:
    ✅ Constant σ (all same ADVar) - Our case
    ✅ Local volatility σ(S) (different ADVars) - Future extension

Both work correctly with AAD.
""")

print("\n" + "-"*100)
print("8. Potential Issue: Grid Construction")
print("-"*100)

print("""
Let me check if there's a mismatch in grid construction...

In Method A (greeks_methods_comparison.py):

    sigma_var = ADVar(sigma, requires_grad=True, name="sigma")
    sigma_grid = [sigma_var] * (M - 1)
                                 ^^^^^^
                                 Length M-1

In PDE solver (capriotti_cn_aad_edgepushing.py):

    for j in range(1, M-1):  # j = 1, 2, ..., M-2 (M-2 iterations)
        sigma_j = sigma_grid[j-1]  # Accesses indices 0, 1, ..., M-3

Maximum index accessed: (M-2) - 1 = M-3
sigma_grid length: M-1
Last element: sigma_grid[M-2]

⚠️ sigma_grid[M-2] is NEVER USED!

For M=51:
    sigma_grid has 50 elements [0...49]
    Loop uses indices 0...48
    Element [49] is unused!
""")

print("\n" + "-"*100)
print("9. Test: Is This Causing Vega Error?")
print("-"*100)

print("""
Hypothesis: Unused sigma element causes gradient loss?

Let's check:
    - sigma_grid = [σ_var, σ_var, ..., σ_var]  (50 copies)
    - Loop uses first 49 copies
    - Last copy unused

Impact on gradient:
    - Each used copy contributes to gradient
    - Unused copy: No contribution
    - ∂V/∂σ = Σ_(first 49) contributions

❓ Should it be Σ_(all 50)?

NO! Because:
    - M-2 = 49 interior points
    - 49 diffusion coefficients
    - 49 is CORRECT number

The 50th element is just extra padding.
""")

print("\n" + "="*100)
print("CONCLUSION")
print("="*100)

print("""
✅ The design sigma_j = sigma_grid[j-1] is CORRECT

Purpose:
    1. Supports local volatility models (future)
    2. Works correctly for constant volatility (current)

For Black-Scholes:
    - All sigma_j point to same ADVar ✅
    - Gradient correctly accumulates ✅
    - One unused element is harmless ✅

❌ Vega error is NOT caused by this design!

Real cause:
    - CN scheme numerical damping at high σ
    - Price V_PDE is wrong
    - AAD correctly computes ∂V_PDE/∂σ of wrong V

Recommendation:
    - Keep current design (supports future extensions)
    - Fix Vega using MC+AAD or variable transformation
""")

print("\n" + "-"*100)
print("10. Could We Optimize for Constant Volatility?")
print("-"*100)

print("""
Alternative for BS only:

Instead of:
    sigma_grid = [sigma_var] * (M-1)

Could use:
    # Don't create grid at all!
    # Pass sigma_var directly to each iteration

    for j in range(1, M-1):
        sigma_j = sigma_var  # Direct reference

Benefits:
    - Clearer code for constant σ
    - Slightly less memory

Drawbacks:
    - Loses local volatility support
    - Requires code change

Verdict: NOT WORTH IT
    - Current design works fine
    - Keep flexibility for future
""")

print("\n📌 FINAL ANSWER:")
print("   The sigma_grid[j-1] design is CORRECT and NOT the cause of Vega errors.")
print("   It's a flexible design that supports both constant and local volatility.")
print("   The Vega issue is in the PDE solver's numerical properties, not the indexing.")
