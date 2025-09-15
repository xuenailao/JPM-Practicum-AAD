import numpy as np
from scipy.stats import norm
from aad_framework import Dual, Variable, exp, log, sqrt, normal_cdf, normal_pdf
from typing import Union, Tuple, Dict


class BlackScholesMerton:
    """Black-Scholes-Merton model for European options with AAD support."""
    
    def __init__(self, use_forward_mode=True):
        """
        Initialize BSM model.
        
        Args:
            use_forward_mode: If True, use forward-mode AD (Dual numbers).
                            If False, use reverse-mode AD (Variables).
        """
        self.use_forward_mode = use_forward_mode
    
    def _d1_d2(self, S, K, T, r, sigma):
        """Calculate d1 and d2 parameters."""
        if self.use_forward_mode:
            d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
            d2 = d1 - sigma * sqrt(T)
        else:
            # For reverse mode, we need to handle operations carefully
            log_moneyness = (S / K).log()
            drift_term = r + 0.5 * sigma ** 2
            d1 = (log_moneyness + drift_term * T) / (sigma * T.sqrt())
            d2 = d1 - sigma * T.sqrt()
        return d1, d2
    
    def call_price(self, S, K, T, r, sigma):
        """
        Calculate European call option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Call option price
        """
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        if self.use_forward_mode:
            N_d1 = normal_cdf(d1)
            N_d2 = normal_cdf(d2)
            call_value = S * N_d1 - K * exp(-r * T) * N_d2
        else:
            # For reverse mode, use scipy.stats.norm directly on values
            N_d1 = norm.cdf(d1.value) if isinstance(d1, Variable) else norm.cdf(d1)
            N_d2 = norm.cdf(d2.value) if isinstance(d2, Variable) else norm.cdf(d2)
            
            # Create Variables for CDF outputs if needed
            if isinstance(d1, Variable):
                N_d1_var = Variable(N_d1)
                N_d2_var = Variable(N_d2)
                
                # Manual gradient computation for normal CDF
                def backward_N_d1():
                    d1.grad += norm.pdf(d1.value) * N_d1_var.grad
                def backward_N_d2():
                    d2.grad += norm.pdf(d2.value) * N_d2_var.grad
                    
                N_d1_var._backward = backward_N_d1
                N_d1_var._prev = {d1}
                N_d2_var._backward = backward_N_d2
                N_d2_var._prev = {d2}
                
                discount = (-r * T).exp()
                call_value = S * N_d1_var - K * discount * N_d2_var
            else:
                call_value = S * N_d1 - K * np.exp(-r * T) * N_d2
        
        return call_value
    
    def put_price(self, S, K, T, r, sigma):
        """
        Calculate European put option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to maturity
            r: Risk-free rate
            sigma: Volatility
            
        Returns:
            Put option price
        """
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        if self.use_forward_mode:
            N_minus_d1 = normal_cdf(-d1)
            N_minus_d2 = normal_cdf(-d2)
            put_value = K * exp(-r * T) * N_minus_d2 - S * N_minus_d1
        else:
            # Similar approach as call option
            N_minus_d1 = norm.cdf(-d1.value if isinstance(d1, Variable) else -d1)
            N_minus_d2 = norm.cdf(-d2.value if isinstance(d2, Variable) else -d2)
            
            if isinstance(d1, Variable):
                N_minus_d1_var = Variable(N_minus_d1)
                N_minus_d2_var = Variable(N_minus_d2)
                
                def backward_N_minus_d1():
                    d1.grad -= norm.pdf(d1.value) * N_minus_d1_var.grad
                def backward_N_minus_d2():
                    d2.grad -= norm.pdf(d2.value) * N_minus_d2_var.grad
                    
                N_minus_d1_var._backward = backward_N_minus_d1
                N_minus_d1_var._prev = {d1}
                N_minus_d2_var._backward = backward_N_minus_d2
                N_minus_d2_var._prev = {d2}
                
                discount = (-r * T).exp()
                put_value = K * discount * N_minus_d2_var - S * N_minus_d1_var
            else:
                put_value = K * np.exp(-r * T) * N_minus_d2 - S * N_minus_d1
        
        return put_value
    
    def calculate_greeks_forward(self, S: float, K: float, T: float, r: float, 
                                sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all first-order Greeks using forward-mode AD.
        
        Returns:
            Dictionary containing Delta, Vega, Theta, Rho
        """
        greeks = {}
        
        # Delta (∂V/∂S)
        S_dual = Dual(S, 1.0)
        if option_type == 'call':
            price_dual = self.call_price(S_dual, K, T, r, sigma)
        else:
            price_dual = self.put_price(S_dual, K, T, r, sigma)
        greeks['delta'] = price_dual.derivative
        
        # Vega (∂V/∂σ)
        sigma_dual = Dual(sigma, 1.0)
        if option_type == 'call':
            price_dual = self.call_price(S, K, T, r, sigma_dual)
        else:
            price_dual = self.put_price(S, K, T, r, sigma_dual)
        greeks['vega'] = price_dual.derivative
        
        # Theta (∂V/∂T) - Note: typically reported as negative of derivative
        T_dual = Dual(T, 1.0)
        if option_type == 'call':
            price_dual = self.call_price(S, K, T_dual, r, sigma)
        else:
            price_dual = self.put_price(S, K, T_dual, r, sigma)
        greeks['theta'] = -price_dual.derivative  # Negative for time decay
        
        # Rho (∂V/∂r)
        r_dual = Dual(r, 1.0)
        if option_type == 'call':
            price_dual = self.call_price(S, K, T, r_dual, sigma)
        else:
            price_dual = self.put_price(S, K, T, r_dual, sigma)
        greeks['rho'] = price_dual.derivative
        
        return greeks
    
    def calculate_greeks_reverse(self, S: float, K: float, T: float, r: float, 
                                sigma: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all first-order Greeks using reverse-mode AD.
        
        Returns:
            Dictionary containing Delta, Vega, Theta, Rho
        """
        # Create variables
        S_var = Variable(S, name='S')
        K_var = Variable(K, name='K')
        T_var = Variable(T, name='T')
        r_var = Variable(r, name='r')
        sigma_var = Variable(sigma, name='sigma')
        
        # Calculate option price
        if option_type == 'call':
            price = self.call_price(S_var, K_var, T_var, r_var, sigma_var)
        else:
            price = self.put_price(S_var, K_var, T_var, r_var, sigma_var)
        
        # Compute all gradients at once
        price.backward()
        
        return {
            'delta': S_var.grad.item() if hasattr(S_var.grad, 'item') else float(S_var.grad),
            'vega': sigma_var.grad.item() if hasattr(sigma_var.grad, 'item') else float(sigma_var.grad),
            'theta': -T_var.grad.item() if hasattr(T_var.grad, 'item') else -float(T_var.grad),
            'rho': r_var.grad.item() if hasattr(r_var.grad, 'item') else float(r_var.grad)
        }
    
    def gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Gamma (∂²V/∂S²) analytically.
        
        Note: Second-order derivatives require special handling in AD.
        """
        # Use analytical formula directly for gamma
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))