import numpy as np
from aad_framework import Dual, Variable, exp, log, sqrt
from bsm_model import BlackScholesMerton
from greeks_calculator import GreeksCalculator
from calibration import DiscountCurveCalibrator


def test_dual_numbers():
    """Test basic dual number operations."""
    print("Testing Dual Numbers...")
    
    # Basic operations
    x = Dual(3.0, 1.0)
    y = x + 2
    assert abs(y.value - 5.0) < 1e-10 and abs(y.derivative - 1.0) < 1e-10
    print("✓ Addition works correctly")
    
    y = x * 4
    assert abs(y.value - 12.0) < 1e-10 and abs(y.derivative - 4.0) < 1e-10
    print("✓ Multiplication works correctly")
    
    y = x ** 2
    assert abs(y.value - 9.0) < 1e-10 and abs(y.derivative - 6.0) < 1e-10
    print("✓ Power works correctly")
    
    # Chain rule
    x = Dual(2.0, 1.0)
    y = exp(x ** 2)
    expected_value = np.exp(4.0)
    expected_derivative = 2 * 2 * np.exp(4.0)
    assert abs(y.value - expected_value) < 1e-10
    assert abs(y.derivative - expected_derivative) < 1e-10
    print("✓ Chain rule works correctly")
    
    print()


def test_variables():
    """Test reverse-mode AD."""
    print("Testing Reverse-Mode AD (Variables)...")
    
    x = Variable(3.0, name='x')
    y = Variable(2.0, name='y')
    
    # f = x * y + x^2
    f = x * y + x ** 2
    f.backward()
    
    assert abs(x.grad - 8.0) < 1e-10  # df/dx = y + 2x = 2 + 6 = 8
    assert abs(y.grad - 3.0) < 1e-10  # df/dy = x = 3
    print("✓ Basic gradient computation works correctly")
    
    print()


def test_black_scholes():
    """Test Black-Scholes model."""
    print("Testing Black-Scholes-Merton Model...")
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    bsm = BlackScholesMerton()
    
    # Test put-call parity
    call = bsm.call_price(S, K, T, r, sigma)
    put = bsm.put_price(S, K, T, r, sigma)
    
    call_value = call.value if hasattr(call, 'value') else float(call)
    put_value = put.value if hasattr(put, 'value') else float(put)
    
    parity = call_value - put_value
    theoretical = S - K * np.exp(-r * T)
    
    assert abs(parity - theoretical) < 1e-10
    print("✓ Put-call parity holds")
    
    # Test option bounds
    lower_bound = max(S - K * np.exp(-r * T), 0)
    assert call_value >= lower_bound - 1e-10
    assert call_value <= S + 1e-10
    print("✓ Call option bounds are satisfied")
    
    print()


def test_greeks():
    """Test Greeks calculation."""
    print("Testing Greeks Calculation...")
    
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.2
    
    # Test forward mode
    calc = GreeksCalculator(use_forward_mode=True)
    aad_greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'call')
    analytical_greeks = calc.analytical_greeks(S, K, T, r, sigma, 'call')
    
    tolerance = 1e-6
    for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']:
        assert abs(aad_greeks[greek] - analytical_greeks[greek]) < tolerance
    print("✓ Forward-mode Greeks match analytical formulas")
    
    # Test delta bounds
    for S_test in [50, 80, 100, 120, 150]:
        call_greeks = calc.calculate_all_greeks(S_test, K, T, r, sigma, 'call')
        put_greeks = calc.calculate_all_greeks(S_test, K, T, r, sigma, 'put')
        
        assert 0 <= call_greeks['delta'] <= 1
        assert -1 <= put_greeks['delta'] <= 0
    print("✓ Delta bounds are satisfied")
    
    # Test gamma symmetry
    call_greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'call')
    put_greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'put')
    assert abs(call_greeks['gamma'] - put_greeks['gamma']) < 1e-10
    print("✓ Gamma is symmetric for calls and puts")
    
    print()


def test_calibration():
    """Test calibration functionality."""
    print("Testing Calibration...")
    
    calibrator = DiscountCurveCalibrator()
    bsm = BlackScholesMerton()
    
    # Generate synthetic data
    spot = 100.0
    strikes = np.array([100.0, 100.0, 100.0])
    maturities = np.array([0.25, 0.5, 1.0])
    true_rates = np.array([0.02, 0.025, 0.03])
    sigma = 0.2
    
    call_prices = []
    put_prices = []
    
    for K, T, r in zip(strikes, maturities, true_rates):
        call = bsm.call_price(spot, K, T, r, sigma)
        put = bsm.put_price(spot, K, T, r, sigma)
        
        call_value = call.value if hasattr(call, 'value') else float(call)
        put_value = put.value if hasattr(put, 'value') else float(put)
        
        call_prices.append(call_value)
        put_prices.append(put_value)
    
    # Calibrate
    discount_factors = calibrator.calibrate_from_put_call_parity(
        np.array(call_prices), np.array(put_prices),
        spot, strikes, maturities
    )
    
    # Check accuracy
    for T, r in zip(maturities, true_rates):
        true_df = np.exp(-r * T)
        calibrated_df = discount_factors[T]
        assert abs(calibrated_df - true_df) < 1e-10
    print("✓ Discount factor calibration is accurate")
    
    print()


def test_numerical_stability():
    """Test numerical stability."""
    print("Testing Numerical Stability...")
    
    calc = GreeksCalculator()
    
    # Very short maturity
    greeks = calc.calculate_all_greeks(100, 100, 0.001, 0.05, 0.2, 'call')
    assert not np.isnan(greeks['delta'])
    assert not np.isinf(greeks['gamma'])
    print("✓ Stable for very short maturity")
    
    # Deep out-of-the-money
    greeks = calc.calculate_all_greeks(50, 100, 1.0, 0.05, 0.2, 'call')
    assert greeks['delta'] >= 0
    assert greeks['price'] >= 0
    print("✓ Stable for deep OTM options")
    
    # Deep in-the-money
    greeks = calc.calculate_all_greeks(150, 100, 1.0, 0.05, 0.2, 'put')
    assert greeks['delta'] <= 0
    assert greeks['price'] >= 0
    print("✓ Stable for deep ITM options")
    
    print()


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("Running AAD Framework Tests")
    print("=" * 60)
    print()
    
    test_dual_numbers()
    test_variables()
    test_black_scholes()
    test_greeks()
    test_calibration()
    test_numerical_stability()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()