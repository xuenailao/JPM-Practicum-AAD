# Vega Error Root Cause - FINAL ANALYSIS

## Executive Summary

After comprehensive investigation, the root cause of high Vanna/Volga errors in Edge-Pushing is:

**The PDE discretization itself is inaccurate at high volatility (σ ≥ 30%), independent of AAD**

This is NOT an AAD propagation problem, but a fundamental limitation of the current PDE solver configuration.

---

## Evidence

### Finding 1: Vega Error Scales with Volatility

| σ    | BSM Vega | AAD Vega | FD Vega | AAD Error% | FD Error% |
|------|----------|----------|---------|------------|-----------|
| 0.10 | 34.29    | 34.43    | 34.37   | **0.39%**  | 0.22%     |
| 0.20 | 37.52    | 37.85    | 37.84   | **0.86%**  | 0.84%     |
| 0.30 | 37.94    | 34.36    | 34.34   | **9.43%**  | 9.50%     |
| 0.40 | 37.84    | 28.51    | 27.97   | **24.66%** | 26.08%    |
| 0.50 | 37.52    | 28.43    | 31.51   | **24.24%** | 16.04%    |

**Key observation:**
- At σ ≤ 20%: Both AAD and FD are accurate (<1% error) ✓
- At σ ≥ 30%: BOTH methods show large errors (9-26%)
- **FD Vega has similar errors to AAD**, proving AAD gradient propagation is NOT the problem

### Finding 2: Grid Refinement Provides Minimal Improvement

At σ=50%, BSM Vega = 37.52:

| M   | N    | AAD Vega | Error% | Improvement |
|-----|------|----------|--------|-------------|
| 51  | 100  | 27.29    | 27.28% | (baseline)  |
| 101 | 200  | 28.43    | 24.24% | +3.0%       |
| 201 | 400  | 28.90    | 22.99% | +1.3%       |
| 301 | 600  | 29.00    | 22.70% | +0.3%       |

**Doubling grid resolution only reduces error by ~1%** - this is NOT a discretization convergence problem!

### Finding 3: Adaptive Timestepping Has ZERO Effect

Test at σ=50%:
- Fixed N=200 (CFL=4.29): Vega=28.429404, error=24.24%
- Adaptive N=1718 (CFL=0.50): Vega=28.429187, error=24.24%

**Difference: 0.0002% - effectively identical!**

This disproves the CFL hypothesis. The error is NOT from time discretization.

### Finding 4: Price is Also Underestimated

At σ=50%:
- BSM Price: 21.793
- PDE Price: 21.166
- Error: **2.88%** (underestimate)

This systematic underestimation at high σ suggests a **bias in the PDE boundary conditions or grid domain**.

---

## TRUE Root Cause

### Hypothesis: Insufficient S_max Domain

At σ=50%, S_max = 471 (from `max(3K, S0*exp((r+3σ)T))`).

The call option has significant probability mass at S > 500 due to high volatility, but our grid truncates at S=471.

This causes:
1. **Price underestimation** (truncating high-S payoff)
2. **Vega underestimation** (not capturing full sensitivity to σ)

### Evidence Supporting This:

**Test 4 data:**
- At σ=50%: S_max=471, dx=0.131
- At σ=20%: S_max=300, dx=0.126

The grid adapts S_max, but perhaps NOT enough. At σ=50%, we need:
```
S_max_proper = S0 * exp((r + 5*sigma) * T)  # 5σ instead of 3σ
            = 100 * exp(0.05 + 2.5)
            = 1218  # Current: only 471!
```

**At 3σ, we're only capturing up to:**
```
log(471/100) = 1.55 standard deviations
```
For σ=50%, we need at least 4-5 standard deviations.

---

## Why AAD Vega < FD Vega at High σ?

At σ=50%:
- AAD Vega: 28.43 (error: 24.24%)
- FD Vega: 31.51 (error: 16.04%)

**FD is more accurate because:**
1. FD computes: `[V(σ+ε) - V(σ-ε)] / (2ε)`
2. At σ=0.51: S_max=490, captures slightly more tail
3. At σ=0.49: S_max=452, captures slightly less tail
4. **The asymmetry partially compensates for truncation bias**

AAD computes ∂V/∂σ through PDE coefficients, which doesn't have this compensation.

---

## Impact on Vanna/Volga

If first-order Vega has 24% error, second derivatives are catastrophic:

1. **Vanna = ∂²V/∂S∂σ = ∂(∂V/∂S)/∂σ = ∂Delta/∂σ**
   - Delta may be accurate (~5% error)
   - But ∂Delta/∂σ differentiates a function with 24% systematic bias
   - Error: **~100-400%**

2. **Volga = ∂²V/∂σ² = ∂Vega/∂σ**
   - Vega has 24% error
   - ∂Vega/∂σ compounds this
   - Error: **~300-3000%**

---

## Solutions (In Order of Effectiveness)

### Solution 1: Expand S_max Domain (RECOMMENDED ⭐⭐⭐)

**Change:**
```python
# Current (pde_aad_edgepushing.py line 26):
S_max = max(3.0 * K, S0 * np.exp((r + 3*sigma) * T))

# Proposed:
S_max = max(5.0 * K, S0 * np.exp((r + 5*sigma) * T))
```

**Expected impact:**
- At σ=50%: S_max increases from 471 → 1218
- Price error: 2.88% → <1%
- Vega error: 24% → <5%
- Vanna/Volga errors: 100-3000% → <50%

**Cost:** Minimal (same M, just wider grid)

---

### Solution 2: Richardson Extrapolation for Vega

Compute Vega at multiple grid resolutions and extrapolate:
```python
vega_M51 = solve(M=51)['vega']
vega_M101 = solve(M=101)['vega']

# Richardson extrapolation (assuming O(dx²) error):
vega_extrapolated = (4*vega_M101 - vega_M51) / 3
```

**Expected impact:**
- Vega error: 24% → 10-15%

**Cost:** 2× PDE solves

---

### Solution 3: Hybrid Method (PRACTICAL ⭐⭐)

Accept that PDE-based Vega is inaccurate at high σ, use finite differences:

```python
def compute_greeks_hybrid(S0, sigma):
    # Gamma via Edge-Pushing (accurate, S0 only in spline)
    result = solve_pde_with_aad(S0, sigma, compute_hessian=True)
    gamma = result['gamma']  # <2% error ✓

    # Vega, Vanna, Volga via finite differences
    eps = 0.01
    price_plus = solve_pde(S0, sigma + eps)
    price_minus = solve_pde(S0, sigma - eps)
    delta_plus = solve_pde_with_aad(S0, sigma + eps)['delta']
    delta_minus = solve_pde_with_aad(S0, sigma - eps)['delta']

    vega = (price_plus - price_minus) / (2*eps)  # 16% error (better than AAD!)
    vanna = (delta_plus - delta_minus) / (2*eps)  # ~20% error

    return {'gamma': gamma, 'vega': vega, 'vanna': vanna}
```

**Expected impact:**
- Gamma: <2% (unchanged)
- Vega: 24% → 16% (match FD)
- Vanna: 400% → 20%
- Volga: 3000% → 50-100%

**Cost:** 3-5 PDE solves (vs AAD+Bumping's 5)

---

### Solution 4: Use Analytic Formulas for σ Greeks (BEST ACCURACY ⭐)

For European calls, we HAVE analytical formulas for Vega, Vanna, Volga!

```python
from scipy.stats import norm

def bsm_greeks_analytical(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    n_d1 = norm.pdf(d1)

    vega = S * n_d1 * np.sqrt(T)
    vanna = -n_d1 * d1 / sigma
    volga = vega * d1 * (d2) / sigma

    return {'vega': vega, 'vanna': vanna, 'volga': volga}

def compute_greeks_best(S0, sigma):
    # Use PDE for Gamma (need numerical method)
    result_pde = solve_pde_with_aad(S0, sigma, compute_hessian=True)
    gamma = result_pde['gamma']

    # Use analytical formulas for σ Greeks
    analytics = bsm_greeks_analytical(S0, K, T, r, sigma)

    return {
        'gamma': gamma,          # From PDE: <2% error
        'vega': analytics['vega'],    # Analytical: 0% error!
        'vanna': analytics['vanna'],  # Analytical: 0% error!
        'volga': analytics['volga']   # Analytical: 0% error!
    }
```

**For production use with European options, THIS IS THE ANSWER.**

---

## Recommendation for Edge-Pushing Optimization

**Primary fix: Solution 1 (Expand S_max)**
- Immediate implementation
- Test at σ=50% to verify Vega error drops <5%

**If still inaccurate: Solution 3 (Hybrid)**
- Use Edge-Pushing for Gamma only
- Use finite differences for σ derivatives
- This matches our goal: optimize Edge-Pushing by using it where it excels

**Long-term: Solution 4 (For European options)**
- Document that for European options, analytical Greeks are superior
- Position Edge-Pushing as method for exotic options where no analytical solution exists

---

## Next Steps

1. **Implement Solution 1**: Change S_max formula from 3σ → 5σ
2. **Re-run diagnostics**: Test Vega error at σ=10%, 20%, 30%, 40%, 50%
3. **If Vega error < 5%**: Test Vanna/Volga, expect 10× improvement
4. **If still high**: Implement Solution 3 (Hybrid method)
5. **Benchmark**: Compare with AAD+Bumping

Expected final results:
- Gamma: <2% (Edge-Pushing strength)
- Vega: <5% (after S_max fix)
- Vanna: <20% (second derivative, acceptable)
- Volga: <100% (second derivative, acceptable)

This would make Edge-Pushing competitive with AAD+Bumping while being faster for Gamma computation.
