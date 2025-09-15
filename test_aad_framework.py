import pytest
import numpy as np
from aad_framework import Dual, Variable, exp, log, sqrt
from bsm_model import BlackScholesMerton
from greeks_calculator import GreeksCalculator
from calibration import DiscountCurveCalibrator, VolatilitySurfaceCalibrator


class TestDualNumbers:
    """Test forward-mode automatic differentiation."""
    
    def test_basic_operations(self):
        """Test basic arithmetic operations."""
        x = Dual(3.0, 1.0)
        
        # Addition
        y = x + 2
        assert abs(y.value - 5.0) < 1e-10
        assert abs(y.derivative - 1.0) < 1e-10
        
        # Multiplication
        y = x * 4
        assert abs(y.value - 12.0) < 1e-10
        assert abs(y.derivative - 4.0) < 1e-10
        
        # Division
        y = x / 2
        assert abs(y.value - 1.5) < 1e-10
        assert abs(y.derivative - 0.5) < 1e-10
        
        # Power
        y = x ** 2
        assert abs(y.value - 9.0) < 1e-10
        assert abs(y.derivative - 6.0) < 1e-10
    
    def test_chain_rule(self):
        """Test chain rule with composite functions."""
        x = Dual(2.0, 1.0)
        
        # f(x) = exp(x^2)
        y = exp(x ** 2)
        expected_value = np.exp(4.0)
        expected_derivative = 2 * 2 * np.exp(4.0)  # 2x * exp(x^2)
        
        assert abs(y.value - expected_value) < 1e-10
        assert abs(y.derivative - expected_derivative) < 1e-10
    
    def test_transcendental_functions(self):
        """Test transcendental functions."""
        x = Dual(1.0, 1.0)
        
        # Test exp
        y = exp(x)
        assert abs(y.value - np.exp(1.0)) < 1e-10
        assert abs(y.derivative - np.exp(1.0)) < 1e-10
        
        # Test log
        y = log(x)
        assert abs(y.value - 0.0) < 1e-10
        assert abs(y.derivative - 1.0) < 1e-10
        
        # Test sqrt
        x = Dual(4.0, 1.0)
        y = sqrt(x)
        assert abs(y.value - 2.0) < 1e-10
        assert abs(y.derivative - 0.25) < 1e-10


class TestVariable:
    """Test reverse-mode automatic differentiation."""
    
    def test_basic_gradients(self):
        """Test basic gradient computation."""
        x = Variable(3.0, name='x')
        y = Variable(2.0, name='y')
        
        # f = x * y + x^2
        f = x * y + x ** 2
        f.backward()
        
        assert abs(x.grad - 8.0) < 1e-10  # df/dx = y + 2x = 2 + 6 = 8
        assert abs(y.grad - 3.0) < 1e-10  # df/dy = x = 3
    
    def test_complex_function(self):
        """Test gradient of complex function."""
        x = Variable(1.0)
        
        # f(x) = exp(x) * log(x + 1)
        f = (x).exp() * (x + 1).log()
        f.backward()
        
        # df/dx = exp(x) * log(x+1) + exp(x) / (x+1)
        expected_grad = np.exp(1) * np.log(2) + np.exp(1) / 2
        assert abs(x.grad - expected_grad) < 1e-10


class TestBlackScholesMerton:
    """Test Black-Scholes-Merton model implementation."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
    
    def test_put_call_parity(self):
        """Test put-call parity: C - P = S - K*exp(-rT)."""
        bsm = BlackScholesMerton()
        
        call = bsm.call_price(self.S, self.K, self.T, self.r, self.sigma)
        put = bsm.put_price(self.S, self.K, self.T, self.r, self.sigma)
        
        call_value = call.value if hasattr(call, 'value') else float(call)
        put_value = put.value if hasattr(put, 'value') else float(put)
        
        parity = call_value - put_value
        theoretical = self.S - self.K * np.exp(-self.r * self.T)
        
        assert abs(parity - theoretical) < 1e-10
    
    def test_option_bounds(self):
        """Test option price bounds."""
        bsm = BlackScholesMerton()
        
        # Call option bounds: max(S - K*exp(-rT), 0) <= C <= S
        call = bsm.call_price(self.S, self.K, self.T, self.r, self.sigma)
        call_value = call.value if hasattr(call, 'value') else float(call)
        
        lower_bound = max(self.S - self.K * np.exp(-self.r * self.T), 0)
        assert call_value >= lower_bound - 1e-10
        assert call_value <= self.S + 1e-10
        
        # Put option bounds: max(K*exp(-rT) - S, 0) <= P <= K*exp(-rT)
        put = bsm.put_price(self.S, self.K, self.T, self.r, self.sigma)
        put_value = put.value if hasattr(put, 'value') else float(put)
        
        lower_bound = max(self.K * np.exp(-self.r * self.T) - self.S, 0)
        upper_bound = self.K * np.exp(-self.r * self.T)
        assert put_value >= lower_bound - 1e-10
        assert put_value <= upper_bound + 1e-10


class TestGreeks:
    """Test Greeks calculation."""
    
    def setup_method(self):
        """Set up test parameters."""
        self.S = 100.0
        self.K = 100.0
        self.T = 1.0
        self.r = 0.05
        self.sigma = 0.2
    
    def test_forward_vs_analytical(self):
        """Test forward-mode Greeks against analytical formulas."""
        calc = GreeksCalculator(use_forward_mode=True)
        
        aad_greeks = calc.calculate_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        analytical_greeks = calc.analytical_greeks(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        
        # Check each Greek (allowing small numerical error)
        tolerance = 1e-6
        assert abs(aad_greeks['delta'] - analytical_greeks['delta']) < tolerance
        assert abs(aad_greeks['gamma'] - analytical_greeks['gamma']) < tolerance
        assert abs(aad_greeks['vega'] - analytical_greeks['vega']) < tolerance
        assert abs(aad_greeks['theta'] - analytical_greeks['theta']) < tolerance
        assert abs(aad_greeks['rho'] - analytical_greeks['rho']) < tolerance
    
    def test_reverse_vs_analytical(self):
        """Test reverse-mode Greeks against analytical formulas."""
        calc = GreeksCalculator(use_forward_mode=False)
        
        aad_greeks = calc.calculate_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, 'put'
        )
        analytical_greeks = calc.analytical_greeks(
            self.S, self.K, self.T, self.r, self.sigma, 'put'
        )
        
        # Check each Greek
        tolerance = 1e-4  # Slightly larger tolerance for reverse mode
        assert abs(aad_greeks['delta'] - analytical_greeks['delta']) < tolerance
        assert abs(aad_greeks['gamma'] - analytical_greeks['gamma']) < tolerance
        assert abs(aad_greeks['vega'] - analytical_greeks['vega']) < tolerance
        assert abs(aad_greeks['theta'] - analytical_greeks['theta']) < tolerance
        assert abs(aad_greeks['rho'] - analytical_greeks['rho']) < tolerance
    
    def test_delta_bounds(self):
        """Test delta bounds: 0 <= call_delta <= 1, -1 <= put_delta <= 0."""
        calc = GreeksCalculator()
        
        # Test various spot prices
        for S in [50, 80, 100, 120, 150]:
            call_greeks = calc.calculate_all_greeks(
                S, self.K, self.T, self.r, self.sigma, 'call'
            )
            put_greeks = calc.calculate_all_greeks(
                S, self.K, self.T, self.r, self.sigma, 'put'
            )
            
            assert 0 <= call_greeks['delta'] <= 1
            assert -1 <= put_greeks['delta'] <= 0
    
    def test_gamma_symmetry(self):
        """Test that gamma is the same for calls and puts."""
        calc = GreeksCalculator()
        
        call_greeks = calc.calculate_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, 'call'
        )
        put_greeks = calc.calculate_all_greeks(
            self.S, self.K, self.T, self.r, self.sigma, 'put'
        )
        
        assert abs(call_greeks['gamma'] - put_greeks['gamma']) < 1e-10


class TestCalibration:
    """Test calibration functionality."""
    
    def test_discount_factor_calibration(self):
        """Test discount factor calibration from put-call parity."""
        calibrator = DiscountCurveCalibrator()
        bsm = BlackScholesMerton()
        
        # Generate synthetic data
        spot = 100.0
        strikes = np.array([100.0, 100.0, 100.0])
        maturities = np.array([0.25, 0.5, 1.0])
        true_rates = np.array([0.02, 0.025, 0.03])
        sigma = 0.2
        
        # Calculate option prices
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
        
        # Check calibration accuracy
        for T, r in zip(maturities, true_rates):
            true_df = np.exp(-r * T)
            calibrated_df = discount_factors[T]
            assert abs(calibrated_df - true_df) < 1e-10
    
    def test_implied_volatility(self):
        """Test implied volatility calculation."""
        calc = GreeksCalculator()
        
        # Known case
        S, K, T, r = 100.0, 100.0, 1.0, 0.05
        true_vol = 0.25
        
        # Calculate option price with known vol
        bsm = BlackScholesMerton()
        call = bsm.call_price(S, K, T, r, true_vol)
        call_value = call.value if hasattr(call, 'value') else float(call)
        
        # Recover implied vol
        iv, _ = calc.implied_volatility_greeks(call_value, S, K, T, r, 'call')
        
        assert abs(iv - true_vol) < 1e-6


def test_numerical_stability():
    """Test numerical stability in extreme cases."""
    calc = GreeksCalculator()
    
    # Test very short maturity
    greeks = calc.calculate_all_greeks(100, 100, 0.001, 0.05, 0.2, 'call')
    assert not np.isnan(greeks['delta'])
    assert not np.isinf(greeks['gamma'])
    
    # Test deep out-of-the-money
    greeks = calc.calculate_all_greeks(50, 100, 1.0, 0.05, 0.2, 'call')
    assert greeks['delta'] >= 0
    assert greeks['price'] >= 0
    
    # Test deep in-the-money
    greeks = calc.calculate_all_greeks(150, 100, 1.0, 0.05, 0.2, 'put')
    assert greeks['delta'] <= 0
    assert greeks['price'] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])