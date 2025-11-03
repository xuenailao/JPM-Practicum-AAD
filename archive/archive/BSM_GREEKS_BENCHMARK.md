# BSM Greeks Benchmark (Ground Truth)

## Test Parameters
```
S0 = 100.0  (Stock price)
K  = 100.0  (Strike price)
T  = 1.0    (Time to maturity, years)
r  = 0.05   (Risk-free rate, 5%)
σ  = 0.20   (Volatility, 20%)

Option: European Call, ATM (At-The-Money)
```

## Analytical Greeks (Exact Values)

### First-Order Greeks
```
Price  = $10.450584    (Option value)
Delta  =  0.636831     (∂V/∂S - hedge ratio)
Vega   =  0.375240     (∂V/∂σ per 1% vol change)
```

### Second-Order Greeks
```
Gamma  =  0.018762     (∂²V/∂S² - delta convexity)
Vanna  = -0.002814     (∂²V/∂S∂σ per 1% vol - cross-gamma)
Volga  =  0.000985     (∂²V/∂σ² per 1%² - vega convexity)
```

## Interpretation

### Gamma (0.018762)
- **High convexity exposure** for ATM option
- Delta changes by 0.018762 for each $1 stock move
- Positive gamma → Long option benefits from volatility
- Maximum at ATM, decreases for OTM/ITM

### Vanna (-0.002814)
- **Negative vanna** → Delta decreases when vol increases
- For 1% vol increase (20%→21%): Delta decreases by 0.002814
- **Volatility-Delta cross-effect**
- Important for delta hedging under changing vol

### Volga (0.000985)
- **Positive volga** → Vega increases with vol
- For 1% vol increase: Vega increases by $0.000985
- **Convexity in volatility exposure**
- Long options typically have positive volga

## Use Cases

### 1. PDE Validation
PDE methods with **constant volatility surface** should match these values within numerical precision:
- Price: ±0.01%
- Delta/Vega: ±0.1%
- Gamma/Vanna/Volga: ±1-5% (second-order more sensitive)

### 2. AAD Validation
Automatic Adjoint Differentiation should produce:
- First-order (Delta, Vega): Machine precision (~1e-12)
- Second-order (Gamma, Vanna, Volga): High precision (~1e-6 to 1e-9)

### 3. Bumping Method Baseline
Finite differences accuracy depends on step size:
- eps = 0.01: ~1% error
- eps = 0.001: ~0.1% error
- eps = 0.0001: ~0.01% error (but numerical instability risk)

## Comparison Matrix

| Method | Price Error | Delta Error | Gamma Error | Vanna Error | Volga Error | Speed |
|--------|-------------|-------------|-------------|-------------|-------------|-------|
| **BSM Analytical** | 0% (exact) | 0% (exact) | 0% (exact) | 0% (exact) | 0% (exact) | <1ms ⚡ |
| **PDE (Fine Grid)** | <0.01% | <0.1% | <1% | <5% | <10% | ~100ms |
| **PDE Bumping** | <0.01% | <0.5% | <5% | <20% | <50% | ~500ms |
| **PDE AAD** | <0.01% | <0.1% | <2% | <10% | <20% | ~200ms |

## Mathematical Formulas

### BSM Call Price
```
C = S₀Φ(d₁) - Ke^(-rT)Φ(d₂)

where:
  d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
  d₂ = d₁ - σ√T
  Φ(·) = Standard normal CDF
```

### Greeks Formulas

#### Delta
```
Δ = ∂C/∂S = Φ(d₁)
```

#### Gamma
```
Γ = ∂²C/∂S² = φ(d₁) / (S·σ·√T)

where φ(·) = standard normal PDF
```

#### Vega (per 1% vol)
```
ν = ∂C/∂σ = S·φ(d₁)·√T / 100
```

#### Vanna (per 1% vol)
```
∂²C/∂S∂σ = -φ(d₁)·d₂/σ / 100
```

#### Volga (per 1%²)
```
∂²C/∂σ² = S·φ(d₁)·√T · d₁·d₂/σ / 10000
```

## Numerical Validation Tests

### Test 1: Price Convergence
```python
# PDE should converge to BSM as grid refines
M_values = [20, 40, 80, 160]
Expected: Price error ∝ O(1/M²)
```

### Test 2: Greeks Stability
```python
# Greeks should be stable across grid sizes
Gamma should not change by >10% for M∈[40,160]
```

### Test 3: AAD vs Bumping
```python
# AAD should be more accurate than bumping
Vanna_AAD error < Vanna_Bumping error
Volga_AAD error < Volga_Bumping error
```

## Known Issues

### PDE Challenges
1. **Boundary conditions**: Can affect Greeks near S=0 or S=Smax
2. **Time discretization**: Coarse Δt introduces errors in Theta/Vanna
3. **Space discretization**: Coarse ΔS affects Gamma accuracy
4. **Interpolation**: S0 not on grid → interpolation error

### AAD Challenges
1. **Graph size**: Large for fine grids (mitigated by super-node)
2. **Second-order**: Requires true 2nd-order AD (not nested first-order)
3. **Numerical precision**: Accumulation of floating-point errors

### Bumping Challenges
1. **Step size**: Too large → truncation error, too small → roundoff
2. **Cost**: O(n) solves for gradient, O(n²) for Hessian
3. **Cross-derivatives**: Vanna requires 4 PDE solves

## References

1. **Black-Scholes (1973)**: "The Pricing of Options and Corporate Liabilities"
2. **Hull (2018)**: "Options, Futures, and Other Derivatives", Ch. 19
3. **Wilmott et al. (1995)**: "The Mathematics of Financial Derivatives"
4. **Glasserman (2004)**: "Monte Carlo Methods in Financial Engineering"

---

**Generated**: 2025-10-23
**Purpose**: Benchmark for PDE and AAD Greeks validation
