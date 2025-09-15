import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from bsm_model import BlackScholesMerton
from aad_framework import Dual, Variable


class GreeksCalculator:
    """Enhanced Greeks calculator with AAD support and analytical validation."""
    
    def __init__(self, use_forward_mode=True):
        self.bsm = BlackScholesMerton(use_forward_mode=use_forward_mode)
        self.use_forward_mode = use_forward_mode
    
    def calculate_all_greeks(self, S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all first-order Greeks using AAD.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with all Greeks
        """
        if self.use_forward_mode:
            greeks = self.bsm.calculate_greeks_forward(S, K, T, r, sigma, option_type)
        else:
            greeks = self.bsm.calculate_greeks_reverse(S, K, T, r, sigma, option_type)
        
        # Add Gamma (requires special handling)
        greeks['gamma'] = self.bsm.gamma(S, K, T, r, sigma)
        
        # Calculate option price for reference
        if self.use_forward_mode:
            if option_type == 'call':
                price = self.bsm.call_price(S, K, T, r, sigma)
            else:
                price = self.bsm.put_price(S, K, T, r, sigma)
            greeks['price'] = float(price.value if hasattr(price, 'value') else price)
        else:
            # For reverse mode, use analytical price calculation
            from scipy.stats import norm
            d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            if option_type == 'call':
                greeks['price'] = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:
                greeks['price'] = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return greeks
    
    def analytical_greeks(self, S: float, K: float, T: float, r: float, 
                         sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate Greeks using analytical formulas for validation.
        """
        from scipy.stats import norm
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        greeks = {}
        
        if option_type == 'call':
            greeks['delta'] = norm.cdf(d1)
            greeks['price'] = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            greeks['delta'] = norm.cdf(d1) - 1
            greeks['price'] = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        # Greeks common to both calls and puts
        greeks['gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        greeks['vega'] = S * norm.pdf(d1) * np.sqrt(T)
        
        if option_type == 'call':
            greeks['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                              - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            greeks['theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                              + r * K * np.exp(-r * T) * norm.cdf(-d2))
        
        greeks['rho'] = K * T * np.exp(-r * T) * (norm.cdf(d2) if option_type == 'call' else -norm.cdf(-d2))
        
        return greeks
    
    def compare_methods(self, S: float, K: float, T: float, r: float, 
                       sigma: float, option_type: str = 'call') -> pd.DataFrame:
        """
        Compare AAD Greeks with analytical Greeks.
        """
        aad_greeks = self.calculate_all_greeks(S, K, T, r, sigma, option_type)
        analytical = self.analytical_greeks(S, K, T, r, sigma, option_type)
        
        comparison = pd.DataFrame({
            'AAD': aad_greeks,
            'Analytical': analytical
        })
        comparison['Difference'] = comparison['AAD'] - comparison['Analytical']
        comparison['Rel_Error_%'] = 100 * comparison['Difference'] / comparison['Analytical']
        
        return comparison
    
    def sensitivity_analysis(self, base_S: float, base_K: float, base_T: float, 
                           base_r: float, base_sigma: float, option_type: str = 'call',
                           param_ranges: Optional[Dict[str, Tuple[float, float, int]]] = None) -> pd.DataFrame:
        """
        Perform sensitivity analysis on Greeks across parameter ranges.
        
        Args:
            param_ranges: Dict with parameter names as keys and (min, max, steps) tuples
        """
        if param_ranges is None:
            param_ranges = {
                'S': (base_S * 0.8, base_S * 1.2, 21),
                'sigma': (base_sigma * 0.5, base_sigma * 1.5, 21),
                'T': (max(0.01, base_T * 0.1), base_T, 20),
                'r': (0, base_r * 2, 11)
            }
        
        results = []
        
        for param, (min_val, max_val, steps) in param_ranges.items():
            values = np.linspace(min_val, max_val, steps)
            
            for val in values:
                # Set parameters
                params = {
                    'S': base_S,
                    'K': base_K,
                    'T': base_T,
                    'r': base_r,
                    'sigma': base_sigma
                }
                params[param] = val
                
                # Calculate Greeks
                greeks = self.calculate_all_greeks(**params, option_type=option_type)
                
                # Store results
                result = {
                    'parameter': param,
                    'value': val,
                    **greeks
                }
                results.append(result)
        
        return pd.DataFrame(results)
    
    def implied_volatility_greeks(self, option_price: float, S: float, K: float, 
                                 T: float, r: float, option_type: str = 'call',
                                 initial_guess: float = 0.2) -> Tuple[float, Dict[str, float]]:
        """
        Calculate implied volatility and Greeks at that volatility using Newton-Raphson.
        """
        sigma = initial_guess
        max_iterations = 100
        tolerance = 1e-6
        
        for _ in range(max_iterations):
            # Use forward mode for this calculation
            sigma_dual = Dual(sigma, 1.0)
            
            if option_type == 'call':
                price_dual = self.bsm.call_price(S, K, T, r, sigma_dual)
            else:
                price_dual = self.bsm.put_price(S, K, T, r, sigma_dual)
            
            price = price_dual.value
            vega = price_dual.derivative
            
            # Newton-Raphson update
            diff = price - option_price
            if abs(diff) < tolerance:
                break
            
            if abs(vega) < 1e-10:  # Avoid division by zero
                break
                
            sigma = sigma - diff / vega
            sigma = max(0.001, sigma)  # Ensure positive volatility
        
        # Calculate all Greeks at implied volatility
        greeks = self.calculate_all_greeks(S, K, T, r, sigma, option_type)
        
        return sigma, greeks