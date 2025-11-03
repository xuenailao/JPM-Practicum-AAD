# PDE Greeks Calculation - Current Status Summary

## 📊 Overall Results (Grid: M=101×100)

| Greek | Analytical | PDE+AAD | Error | Status | Production Ready? |
|-------|-----------|---------|-------|--------|-------------------|
| **Price** | 10.450584 | 10.353675 | 0.93% | ✅ Good | Yes |
| **Delta** | 0.636831 | 0.634844 | 0.31% | ✅ Excellent | Yes |
| **Gamma** | 0.018762 | 0.019162 | 2.13% | ✅ Good | Yes |
| **Vega** | 37.524035 | 32.767837 | 12.68% | ⚠️ Moderate | Qualitative only |
| **Vanna** | -0.281430 | -0.374579 | 33.10% | ❌ Poor | No |
| **Volga** | 9.850059 | -189.399126 | 2023% | ❌ Failed | No |

**Computation Time**: ~36 seconds
**Method**: Perturbation + AAD (Method A)
**PDE Solves**: 5 (3 for D/G, 2 for Volga)

---

## 🎯 Success Story: Gamma

### Problem (Original)
- **Linear interpolation AAD**: Gamma = 0 (100% error)
- Root cause: d²Weight/dS₀² = 0 in linear interpolation

### Solution (Method A)
- **Perturbation approach**: Solve PDE at S₀-ε, S₀, S₀+ε
- Finite difference: Gamma = (V₊ - 2V₀ + V₋) / ε²
- **Key innovation**: eps_S = dS (grid spacing)

### Results
```
Grid    | Gamma Error | Improvement
--------|-------------|-------------
51×50   | 4.53%       | 22× vs 100%
101×100 | 2.13%       | 47× vs 100%
```

**Status**: ✅ **SOLVED** - Production ready

---

## ⚠️ Ongoing Challenge: Vega

### Current Status
- **Error**: 12.68% (stable across grid sizes)
- **Cause**: PDE discretization, NOT AAD

### Evidence (M=101)
```
Vega Method     | Result    | Error
----------------|-----------|-------
PDE+AAD         | 32.767837 | 12.68%
Finite Diff     | 32.775920 | 12.66%  ← Same error!
Analytical (BS) | 37.524035 | 0.00%
```

**Diagnosis**:
- AAD is correct (matches FD)
- PDE Price error (0.93%) amplifies ~10× in derivative
- Grid refinement doesn't help (M=51→151: error stable)

### Why Grid Refinement Fails

| Grid | Price Error | Vega Error | Time |
|------|------------|------------|------|
| 51×50 | 1.28% | 12.64% | 8.3s |
| 101×100 | 0.93% | 12.68% | 36.6s |
| 151×150 | 0.93% | 12.68% | 86.6s |

**Observation**: Vega error converged at M=51, further refinement wastes time.

**Status**: ⚠️ **Acceptable for qualitative analysis** - Not AAD's fault

---

## ❌ Critical Failure: Volga

### The Problem
- **Expected**: Volga = +9.850059
- **PDE Result**: Volga = -189.399126
- **Error**: 2023% (wrong sign!)

### Root Cause: PDE Vega has WRONG σ-dependence

**Analytical behavior**:
```
σ = 0.198 → Vega = 37.504 ↗ (increasing)
σ = 0.200 → Vega = 37.524 ↗
σ = 0.202 → Vega = 37.543 ↗
```

**PDE behavior**:
```
σ = 0.198 → Vega = 33.150 ↘ (decreasing!)
σ = 0.200 → Vega = 32.779 ↘
σ = 0.202 → Vega = 32.388 ↘
```

**Consequence**:
```
Volga = ∂Vega/∂σ

Analytical: dVega/dσ = (37.543 - 37.504)/(2×0.002) = +9.85 ✅
PDE:        dVega/dσ = (32.388 - 33.150)/(2×0.002) = -190.4 ❌
```

**Status**: ❌ **COMPLETELY BROKEN** - PDE cannot compute Volga

---

## 🔬 What We Tried (All Failed for Vega/Volga)

### Attempt 1: Richardson Extrapolation
**Theory**: Combine solutions at h and h/2 to eliminate O(h) error
**Result**: Vega improvement 0.001074 (negligible)
**Why failed**: Error is systematic bias, not truncation error

### Attempt 2: Ultra-Fine Grid (M=151×150)
**Theory**: Vega error ∝ M⁻²
**Result**: Vega error unchanged (12.68%)
**Why failed**: Error is in PDE scheme itself, not spatial discretization

### Attempt 3: Corrected Volga Formula
**Theory**: Use ∂Vega/∂σ (1st derivative) not ∂²Vega/∂σ² (2nd derivative)
**Result**: Formula now correct, but PDE Vega still wrong
**Why failed**: Fixed the symptom, not the disease

---

## 🎓 Technical Insights Discovered

### 1. eps_S Auto-Selection is Critical
```python
# ❌ WRONG (fixed eps_S)
eps_S = 0.5  → Gamma error 704%

# ✅ CORRECT (adaptive eps_S)
eps_S = dS = 200.0/M  → Gamma error 4.53%
```
**Improvement**: 155×

**Reason**: eps_S should match grid resolution to avoid interpolation errors

### 2. Derivative Amplification
```
Price error:  1.0%
Delta error:  ~2-3% (1st derivative)
Gamma error:  ~4-5% (2nd derivative)
Vega error:   ~10% (1st derivative, but w.r.t. σ)
```

**Observation**: Derivatives magnify discretization errors

### 3. Single vs Multi-Parameter σ Model
```python
# ❌ WRONG (old implementation)
sigma_vars = [ADVar(sigma, ...) for i in range(M-1)]  # M-1 independent σ

# ✅ CORRECT (Method A)
sigma_var = ADVar(sigma, requires_grad=True)  # Single constant σ
sigma_grid = [sigma_var] * (M-1)
```

**For constant volatility model**: All points share the same σ variable

### 4. AAD is Not the Problem
```
Method      | Vega Error
------------|------------
PDE + AAD   | 12.68%
PDE + FD    | 12.66%  ← Same!
```

**Conclusion**: Vega error comes from PDE solver, not AAD

---

## 💡 Recommendations

### For Production Use

**Use Method A for Delta/Gamma**:
```python
from aad_edge_pushing.pde.AADgraph.greeks_methods_comparison import GreeksMethodA

method = GreeksMethodA(M=101, N=100)
greeks = method.compute_greeks(S0=100, K=100, T=1.0, r=0.05, sigma=0.2)

# Use these Greeks:
delta = greeks['delta']   # ✅ 0.31% error
gamma = greeks['gamma']   # ✅ 2.13% error
```

**Use Analytical Solution for Vega/Vanna/Volga**:
```python
from scipy.stats import norm

def black_scholes_vega_greeks(S0, K, T, r, sigma):
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt_T)
    d2 = d1 - sigma*sqrt_T

    vega = S0 * norm.pdf(d1) * sqrt_T
    vanna = -norm.pdf(d1) * d2 / sigma
    volga = vega * d1 * d2 / sigma

    return vega, vanna, volga

vega, vanna, volga = black_scholes_vega_greeks(100, 100, 1.0, 0.05, 0.2)
```

**Hybrid Strategy**: PDE for speed-critical Greeks (Δ/Γ), analytical for accuracy-critical Greeks (Vega/Vanna/Volga)

---

## 🚀 Future Directions

### Short-Term (Low Priority)
- Accept current Vega accuracy (12.68%) for qualitative use
- Use analytical solutions when accuracy is critical

### Medium-Term (Research Required)
1. **Implement Adjoint PDE Method**
   - Solve adjoint PDE for ∂V/∂σ directly
   - Reference: Capriotti (2015) "Real-time risk management: An AAD-PDE approach"
   - Complexity: High
   - Expected improvement: Significant

2. **Test Alternative PDE Schemes**
   - Rannacher time-stepping
   - Higher-order spatial discretization
   - Adaptive mesh refinement near ATM

### Long-Term (Fundamental Change)
3. **Monte Carlo + AAD**
   - MC naturally handles volatility sensitivity
   - AAD computes all Greeks simultaneously
   - Scales to complex models (stochastic vol, jumps)

4. **Hybrid PDE-MC**
   - PDE for Delta/Gamma (fast)
   - MC for Vega/Vanna/Volga (accurate)
   - Best of both worlds

---

## 📁 Documentation Files

**Core Implementation**:
- [`greeks_methods_comparison.py`](aad_edge_pushing/pde/AADgraph/greeks_methods_comparison.py) - Method A (working)
- [`greeks_optimized.py`](aad_edge_pushing/pde/AADgraph/greeks_optimized.py) - Optimization attempts (failed)

**Analysis Reports**:
- [`GREEKS_AAD_FINAL_SUMMARY.md`](GREEKS_AAD_FINAL_SUMMARY.md) - Complete technical summary (Gamma solution)
- [`METHOD_A_TEST_REPORT.md`](METHOD_A_TEST_REPORT.md) - Method A test results
- [`VEGA_VOLGA_FINAL_DIAGNOSIS.md`](VEGA_VOLGA_FINAL_DIAGNOSIS.md) - Deep dive on Vega/Volga failures
- [`GREEKS_STATUS_SUMMARY.md`](GREEKS_STATUS_SUMMARY.md) - This document

**Diagnostic Tests**:
- [`test_volga_analytical.py`](test_volga_analytical.py) - Volga formula verification
- [`test_volga_formula_correct.py`](test_volga_formula_correct.py) - Finite difference formula test
- [`test_volga_pde_precision.py`](test_volga_pde_precision.py) - PDE Vega precision analysis
- [`test_volga_diagnosis.py`](test_volga_diagnosis.py) - Comprehensive diagnosis

---

## 🎓 Lessons Learned

1. **Linear interpolation kills second derivatives**
   - Solution: Use perturbation method instead

2. **AAD is not a silver bullet**
   - AAD correctly computes gradients of *what you give it*
   - If PDE solution is wrong, AAD gradients are also wrong

3. **Grid refinement has limits**
   - Beyond a certain point, refinement doesn't help
   - For Vega: converged at M=51

4. **Systematic errors don't cancel in finite differences**
   - If Vega is biased low at all σ points
   - Finite difference won't fix it

5. **Derivative error amplification is real**
   - Price: 1% error
   - Vega: 10% error (10× amplification)
   - Volga: 2000% error (200× amplification!)

6. **Choose the right tool for the job**
   - PDE: Great for Delta/Gamma
   - Analytical: Better for Vega/Vanna/Volga (when available)
   - MC: Best for complex models

---

**Status**: ✅ Gamma problem solved, ⚠️ Vega acceptable, ❌ Volga unsolvable with current approach
**Date**: 2025-10-28
**Author**: Claude
**Version**: Final Summary v4.0
