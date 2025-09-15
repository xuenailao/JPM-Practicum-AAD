# Automatic Differentiation Framework for European Options

A Python implementation of Automatic Differentiation (AAD) for calculating first-order Greeks of European options using the Black-Scholes-Merton model.

## Features

- **Dual Automatic Differentiation Modes**:
  - Forward-mode AD using dual numbers
  - Reverse-mode AD (backpropagation) using computation graphs

- **Black-Scholes-Merton Model**:
  - European call and put option pricing
  - Support for both AD modes

- **Greeks Calculation**:
  - Delta (∂V/∂S)
  - Gamma (∂²V/∂S²)
  - Vega (∂V/∂σ)
  - Theta (∂V/∂T)
  - Rho (∂V/∂r)

- **Calibration Tools**:
  - Discount factor calibration using put-call parity
  - Volatility surface calibration with SSVI model
  - Implied volatility calculation

## Installation

```bash
pip install numpy scipy pandas matplotlib pytest
```

## Usage

### Basic AAD Example

```python
from aad_framework import Dual

# Define function f(x) = x² + 3x + 5
def f(x):
    return x**2 + 3*x + 5

# Calculate derivative at x=2
x = Dual(2.0, 1.0)  # value=2, derivative seed=1
result = f(x)
print(f"f(2) = {result.value}")      # 15.0
print(f"f'(2) = {result.derivative}") # 7.0
```

### Greeks Calculation

```python
from greeks_calculator import GreeksCalculator

# Option parameters
S = 100.0   # Spot price
K = 100.0   # Strike price  
T = 1.0     # Time to maturity
r = 0.05    # Risk-free rate
sigma = 0.2 # Volatility

# Calculate Greeks using forward-mode AD
calc = GreeksCalculator(use_forward_mode=True)
greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'call')

print(f"Delta: {greeks['delta']:.4f}")
print(f"Gamma: {greeks['gamma']:.4f}")
print(f"Vega: {greeks['vega']:.4f}")
print(f"Theta: {greeks['theta']:.4f}")
print(f"Rho: {greeks['rho']:.4f}")
```

### Discount Factor Calibration

```python
from calibration import DiscountCurveCalibrator

calibrator = DiscountCurveCalibrator()

# Calibrate from market data
discount_factors = calibrator.calibrate_from_put_call_parity(
    call_prices, put_prices, spot, strikes, maturities
)

# Bootstrap zero rates
maturities, zero_rates = calibrator.bootstrap_zero_rates(discount_factors)
```

## Project Structure

- `aad_framework.py` - Core automatic differentiation implementation
- `bsm_model.py` - Black-Scholes-Merton model with AAD support
- `greeks_calculator.py` - Greeks calculation and analysis tools
- `calibration.py` - Discount factor and volatility surface calibration
- `examples.py` - Comprehensive examples demonstrating all features
- `test_aad_framework.py` - Unit tests

## Running Examples

```bash
python examples.py
```

## Running Tests

```bash
python -m pytest test_aad_framework.py -v
```

## Theory Background

### Forward-Mode AD
Forward-mode AD propagates derivatives alongside function values using dual numbers:
- Efficient for functions with few inputs and many outputs
- Calculates one directional derivative per pass

### Reverse-Mode AD  
Reverse-mode AD builds a computation graph and propagates gradients backward:
- Efficient for functions with many inputs and few outputs
- Calculates all partial derivatives in one backward pass

### Put-Call Parity
Used for discount factor calibration:
```
C - P = S - K * DF(T)
```

### SSVI Model
Surface SVI parameterization for volatility surface:
```
w(k) = θ/2 * (1 + ρ*φ*k + sqrt((φ*k + ρ)² + (1-ρ²)))
```

## Performance Considerations

- Forward-mode AD is preferred for single-Greek calculations
- Reverse-mode AD is more efficient when computing all Greeks simultaneously
- The framework includes numerical stability checks for extreme parameter values

## Future Enhancements

- Second-order Greeks (Vanna, Volga, etc.)
- American option support
- Multi-asset options
- Parallel computation support
- GPU acceleration