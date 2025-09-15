import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from aad_framework import Dual, Variable
from bsm_model import BlackScholesMerton
from greeks_calculator import GreeksCalculator
from calibration import DiscountCurveCalibrator, VolatilitySurfaceCalibrator


def example_basic_aad():
    """Example 1: Basic AAD demonstration with Dual numbers."""
    print("=" * 60)
    print("Example 1: Basic AAD with Dual Numbers")
    print("=" * 60)
    
    # Define a simple function f(x) = x^2 + 3x + 5
    def f(x):
        return x**2 + 3*x + 5
    
    # Calculate derivative at x=2 using dual numbers
    x = Dual(2.0, 1.0)  # value=2, derivative=1
    result = f(x)
    
    print(f"Function: f(x) = x² + 3x + 5")
    print(f"At x=2:")
    print(f"  Value: {result.value}")
    print(f"  Derivative: {result.derivative}")
    print(f"  Analytical derivative: {2*2 + 3} = 7")
    print()


def example_forward_mode_greeks():
    """Example 2: Calculate Greeks using forward-mode AD."""
    print("=" * 60)
    print("Example 2: Forward-Mode AD for Greeks Calculation")
    print("=" * 60)
    
    # Option parameters
    S = 100.0   # Spot price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity (1 year)
    r = 0.05    # Risk-free rate
    sigma = 0.2 # Volatility
    
    # Create BSM model with forward mode
    bsm = BlackScholesMerton(use_forward_mode=True)
    calc = GreeksCalculator(use_forward_mode=True)
    
    # Calculate Greeks for call option
    greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'call')
    
    print("Call Option Greeks (Forward-Mode AD):")
    print(f"  Price:  ${greeks['price']:.4f}")
    print(f"  Delta:  {greeks['delta']:.4f}")
    print(f"  Gamma:  {greeks['gamma']:.4f}")
    print(f"  Vega:   {greeks['vega']:.4f}")
    print(f"  Theta:  {greeks['theta']:.4f}")
    print(f"  Rho:    {greeks['rho']:.4f}")
    print()


def example_reverse_mode_greeks():
    """Example 3: Calculate Greeks using reverse-mode AD."""
    print("=" * 60)
    print("Example 3: Reverse-Mode AD for Greeks Calculation")
    print("=" * 60)
    
    # Option parameters (same as forward mode for comparison)
    S = 100.0
    K = 100.0
    T = 1.0
    r = 0.05
    sigma = 0.2
    
    # Create calculator with reverse mode
    calc = GreeksCalculator(use_forward_mode=False)
    
    # Calculate Greeks for put option
    greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'put')
    
    print("Put Option Greeks (Reverse-Mode AD):")
    print(f"  Price:  ${greeks['price']:.4f}")
    print(f"  Delta:  {greeks['delta']:.4f}")
    print(f"  Gamma:  {greeks['gamma']:.4f}")
    print(f"  Vega:   {greeks['vega']:.4f}")
    print(f"  Theta:  {greeks['theta']:.4f}")
    print(f"  Rho:    {greeks['rho']:.4f}")
    print()


def example_compare_methods():
    """Example 4: Compare AAD with analytical Greeks."""
    print("=" * 60)
    print("Example 4: AAD vs Analytical Greeks Comparison")
    print("=" * 60)
    
    S = 100.0
    K = 95.0
    T = 0.25
    r = 0.03
    sigma = 0.25
    
    calc = GreeksCalculator(use_forward_mode=True)
    
    # Compare methods
    comparison = calc.compare_methods(S, K, T, r, sigma, 'call')
    
    print("Comparison of AAD vs Analytical Greeks:")
    print(comparison)
    print()


def example_discount_calibration():
    """Example 5: Calibrate discount factors from put-call parity."""
    print("=" * 60)
    print("Example 5: Discount Factor Calibration")
    print("=" * 60)
    
    # Generate synthetic market data
    spot = 100.0
    maturities = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    true_rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04, 0.045])
    
    # ATM options
    strikes = np.array([100.0] * len(maturities))
    
    # Calculate synthetic option prices
    bsm = BlackScholesMerton()
    call_prices = []
    put_prices = []
    
    for i, (K, T, r) in enumerate(zip(strikes, maturities, true_rates)):
        call = float(bsm.call_price(spot, K, T, r, 0.2).value 
                    if hasattr(bsm.call_price(spot, K, T, r, 0.2), 'value')
                    else bsm.call_price(spot, K, T, r, 0.2))
        put = float(bsm.put_price(spot, K, T, r, 0.2).value 
                   if hasattr(bsm.put_price(spot, K, T, r, 0.2), 'value')
                   else bsm.put_price(spot, K, T, r, 0.2))
        call_prices.append(call)
        put_prices.append(put)
    
    # Calibrate discount factors
    calibrator = DiscountCurveCalibrator()
    discount_factors = calibrator.calibrate_from_put_call_parity(
        np.array(call_prices), np.array(put_prices),
        spot, strikes, maturities
    )
    
    # Bootstrap zero rates
    mats, zero_rates = calibrator.bootstrap_zero_rates(discount_factors)
    
    print("Calibrated Discount Factors and Zero Rates:")
    print(f"{'Maturity':<10} {'True Rate':<12} {'Calibrated':<12} {'Error':<10}")
    print("-" * 44)
    for i, T in enumerate(mats):
        true_rate = true_rates[i]
        calib_rate = zero_rates[i]
        error = calib_rate - true_rate
        print(f"{T:<10.2f} {true_rate:<12.4f} {calib_rate:<12.4f} {error:<10.6f}")
    print()


def example_sensitivity_analysis():
    """Example 6: Sensitivity analysis of Greeks."""
    print("=" * 60)
    print("Example 6: Greeks Sensitivity Analysis")
    print("=" * 60)
    
    # Base parameters
    base_S = 100.0
    base_K = 100.0
    base_T = 1.0
    base_r = 0.05
    base_sigma = 0.2
    
    calc = GreeksCalculator(use_forward_mode=True)
    
    # Define custom ranges for spot price analysis
    param_ranges = {
        'S': (80, 120, 11)  # Spot from 80 to 120
    }
    
    # Perform sensitivity analysis
    results = calc.sensitivity_analysis(
        base_S, base_K, base_T, base_r, base_sigma,
        'call', param_ranges
    )
    
    # Display results
    spot_results = results[results['parameter'] == 'S']
    print("Delta sensitivity to spot price:")
    print(f"{'Spot':<10} {'Delta':<10} {'Gamma':<10} {'Price':<10}")
    print("-" * 40)
    for _, row in spot_results.iterrows():
        print(f"{row['value']:<10.2f} {row['delta']:<10.4f} "
              f"{row['gamma']:<10.4f} {row['price']:<10.4f}")
    print()


def example_implied_volatility():
    """Example 7: Calculate implied volatility using AAD."""
    print("=" * 60)
    print("Example 7: Implied Volatility Calculation with AAD")
    print("=" * 60)
    
    # Market parameters
    S = 100.0
    K = 105.0
    T = 0.5
    r = 0.03
    market_price = 5.50  # Observed option price
    
    calc = GreeksCalculator(use_forward_mode=True)
    
    # Calculate implied volatility and Greeks
    iv, greeks = calc.implied_volatility_greeks(
        market_price, S, K, T, r, 'call'
    )
    
    print(f"Market option price: ${market_price:.2f}")
    print(f"Implied volatility: {iv:.4f} ({iv*100:.2f}%)")
    print("\nGreeks at implied volatility:")
    print(f"  Delta: {greeks['delta']:.4f}")
    print(f"  Vega:  {greeks['vega']:.4f}")
    print(f"  Gamma: {greeks['gamma']:.4f}")
    print()


def example_volatility_surface():
    """Example 8: Calibrate volatility surface."""
    print("=" * 60)
    print("Example 8: Volatility Surface Calibration")
    print("=" * 60)
    
    # Generate synthetic market data
    spot = 100.0
    r = 0.03
    
    # Create market data grid
    strikes = np.array([90, 95, 100, 105, 110])
    maturities = np.array([0.25, 0.5, 1.0])
    
    market_data = []
    true_vols = {}
    
    # Generate data with volatility smile
    for T in maturities:
        for K in strikes:
            # Simple volatility smile
            moneyness = np.log(K / spot)
            vol = 0.2 + 0.1 * moneyness**2 + 0.05 * np.sqrt(T)
            true_vols[(K, T)] = vol
            
            # Calculate option price
            bsm = BlackScholesMerton()
            price = float(bsm.call_price(spot, K, T, r, vol).value
                         if hasattr(bsm.call_price(spot, K, T, r, vol), 'value')
                         else bsm.call_price(spot, K, T, r, vol))
            
            market_data.append({
                'spot': spot,
                'strike': K,
                'maturity': T,
                'rate': r,
                'option_type': 'call',
                'price': price
            })
    
    df = pd.DataFrame(market_data)
    
    # Calibrate surface
    calibrator = VolatilitySurfaceCalibrator()
    surface_params = calibrator.calibrate_surface(df)
    
    print("SSVI Surface Parameters by Maturity:")
    for T, params in surface_params.items():
        print(f"\nT = {T:.2f} years:")
        print(f"  θ (theta): {params['theta']:.4f}")
        print(f"  ρ (rho):   {params['rho']:.4f}")
        print(f"  φ (phi):   {params['phi']:.4f}")
        print(f"  Success:   {params['success']}")
    print()


def run_all_examples():
    """Run all examples."""
    examples = [
        example_basic_aad,
        example_forward_mode_greeks,
        example_reverse_mode_greeks,
        example_compare_methods,
        example_discount_calibration,
        example_sensitivity_analysis,
        example_implied_volatility,
        example_volatility_surface
    ]
    
    for example in examples:
        example()
        print()  # Just add spacing between examples


if __name__ == "__main__":
    print("AAD Framework Examples for European Options")
    print("=" * 60)
    print()
    
    # Run all examples
    run_all_examples()