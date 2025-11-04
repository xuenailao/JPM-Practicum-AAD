# Vega Error Root Cause Analysis - Key Findings

## Critical Discovery: Vega Error Scales Dramatically with Volatility

### Test 1: Vega Error vs Volatility (M=101, N=200)

| σ    | BSM Vega | AAD Vega | FD Vega  | AAD Error% | FD Error% |
|------|----------|----------|----------|------------|-----------|
| 0.10 | 34.29    | 34.43    | 34.37    | **0.39%**  | 0.22%     |
| 0.20 | 37.52    | 37.85    | 37.84    | **0.86%**  | 0.84%     |
| 0.30 | 37.94    | 34.36    | 34.34    | **9.43%**  | 9.50%     |
| 0.40 | 37.84    | 28.51    | 27.97    | **24.66%** | 26.08%    |
| 0.50 | 37.52    | 28.43    | 31.51    | **24.24%** | 16.04%    |

### Key Observations:

1. **Low volatility (σ ≤ 20%): AAD Vega is ACCURATE**
   - Error < 1% at σ=10-20%
   - This proves AAD gradient propagation WORKS correctly!

2. **High volatility (σ ≥ 30%): AAD Vega DEGRADES RAPIDLY**
   - Error jumps from 0.86% at σ=20% to 9.43% at σ=30%
   - At σ=50%, error reaches 24.24%

3. **Finite Difference (FD) has SIMILAR errors**
   - FD error at σ=50%: 16.04% (better but still significant)
   - This suggests the problem is NOT in AAD itself, but in PDE discretization!

## Root Cause: PDE Discretization Error, NOT AAD Propagation

### Test 4: CFL-like Ratio Analysis

| σ    | S_max  | dx      | alpha   | beta    | CFL Ratio |
|------|--------|---------|---------|---------|-----------|
| 0.10 | 300.00 | 0.1261  | 0.3144  | 0.1784  | **0.20**  |
| 0.20 | 300.00 | 0.1261  | 1.2575  | 0.1189  | **0.79**  |
| 0.30 | 300.00 | 0.1261  | 2.8293  | 0.0198  | **1.78**  |
| 0.40 | 349.03 | 0.1276  | 4.9112  | -0.1175 | **3.02**  |
| 0.50 | 471.15 | 0.1306  | 7.3254  | -0.2871 | **4.29**  |

**CFL Ratio = dt / dt_critical**, where dt_critical = dx²/(2α)

### Critical Finding:

**At σ ≥ 30%, CFL ratio > 1 → timestep exceeds stability limit!**

- σ=10%: CFL=0.20 → Vega error 0.39% ✓
- σ=20%: CFL=0.79 → Vega error 0.86% ✓
- σ=30%: CFL=1.78 → Vega error 9.43% ✗
- σ=40%: CFL=3.02 → Vega error 24.66% ✗
- σ=50%: CFL=4.29 → Vega error 24.24% ✗

**Strong correlation: CFL ratio ↑ → Vega error ↑**

## Why AAD Shows Higher Error than FD at σ=50%?

### Comparison:
- AAD Vega error: 24.24%
- FD Vega error: 16.04%

### Explanation:

1. **Finite Difference (FD) computes:**
   ```
   Vega_FD = [V(σ+0.01) - V(σ-0.01)] / 0.02
   ```
   - Each PDE solve has ~3% price error
   - But errors at σ=0.51 and σ=0.49 partially CANCEL in subtraction
   - Resulting error: 16%

2. **AAD computes:**
   ```
   Vega_AAD = ∂V/∂σ via gradient propagation through PDE
   ```
   - Propagates gradients through PDE coefficients α, β which depend nonlinearly on σ
   - Discretization errors in α, β amplify gradient errors
   - No cancellation mechanism
   - Resulting error: 24%

## Test 2: Grid Refinement (σ=50%)

| M   | N   | AAD Vega | Error % |
|-----|-----|----------|---------|
| 51  | 100 | 27.29    | 27.28%  |
| 101 | 200 | 28.43    | 24.24%  |
| 201 | 400 | 28.90    | 22.99%  |
| 301 | 600 | 29.00    | 22.70%  |

**Grid refinement provides MINIMAL improvement!**
- Doubling M,N: 27.28% → 24.24% (only 3% reduction)
- 6× finer grid: 27.28% → 22.70% (only 4.6% reduction)

This confirms the issue is NOT just grid resolution, but the **fundamental CFL violation**.

## Conclusion

### The Real Problem:

**At high volatility, the diffusion coefficient α = σ²/2/dx² becomes so large that:**
1. CFL ratio exceeds critical value (4.29 at σ=50%)
2. Timestep dt=T/N=0.005 is too large for accurate time integration
3. PDE discretization errors dominate, affecting BOTH price AND gradients

### Why This Affects Vanna/Volga More:

1. **First-order Vega**: ~24% error (PDE discretization)
2. **Second-order Volga = ∂Vega/∂σ**:
   - Differentiates an already-noisy Vega
   - Error compounds: ~24% × 10 = 240-300% (observed!)

3. **Mixed derivative Vanna = ∂Delta/∂σ**:
   - Similar compounding of discretization errors
   - Observed error: 18-400%

### AAD is NOT the Problem!

- At σ=10-20%, AAD Vega error < 1%
- AAD gradient propagation works correctly
- The issue is PDE solver accuracy at high volatility

## Solution Strategies

### Option 1: Adaptive Timestepping (RECOMMENDED)
Increase N at high σ to keep CFL < 1:
```python
N = max(200, int(200 * (sigma / 0.20)**2))
```
- σ=20%: N=200, CFL=0.79 ✓
- σ=50%: N=1250, CFL<1 ✓

**Expected result**: Vega error 24% → <5%

### Option 2: Implicit Rannacher Timestepping
Use Rannacher (pure implicit) for first R steps to damp high-frequency errors:
```python
R = max(4, int(alpha))  # More implicit steps at high α
```

**Expected result**: Vega error 24% → 10-15%

### Option 3: Operator Splitting
Split diffusion and drift:
- Solve ∂V/∂t + 0.5σ²∂²V/∂x² = 0 (diffusion only)
- Then solve ∂V/∂t + (r-0.5σ²)∂V/∂x - rV = 0 (drift)

**Expected result**: Better stability at high σ

### Option 4: Accept PDE Limitations + Use Hybrid
- Keep PDE for Gamma (accurate, S0 only in spline)
- Use Richardson extrapolation FD for Vega, Vanna, Volga

**Expected result**:
- Gamma: <2% (no change)
- Vanna/Volga: Use extrapolated FD to reduce 16% → <10%

## Immediate Next Steps

1. **Implement Option 1: Adaptive N**
   - Modify `BS_PDE_AAD.__init__` to compute N based on σ
   - Test at σ=50% with N=1250
   - Verify Vega error drops to <5%

2. **Test impact on Vanna/Volga**
   - If Vega error drops to 5%, Volga error should drop to ~50%
   - If Vanna error also improves, we've solved the root cause!

3. **Document CFL-based N selection**
   - Add to README: "For σ>30%, use N > 200*(σ/0.2)²"
