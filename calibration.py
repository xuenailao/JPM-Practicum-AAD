import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, least_squares
from scipy.interpolate import interp1d, CubicSpline
from bsm_model import BlackScholesMerton
from greeks_calculator import GreeksCalculator


class DiscountCurveCalibrator:
    """Calibrate discount factors from option prices using put-call parity."""
    
    def __init__(self):
        self.bsm = BlackScholesMerton()
        
    def calibrate_from_put_call_parity(self, 
                                     call_prices: np.ndarray,
                                     put_prices: np.ndarray,
                                     spot: float,
                                     strikes: np.ndarray,
                                     maturities: np.ndarray,
                                     atm_only: bool = True) -> Dict[float, float]:
        """
        Calibrate discount factors using put-call parity:
        C - P = S - K * DF(T)
        
        Args:
            call_prices: Array of call option prices
            put_prices: Array of put option prices
            spot: Current spot price
            strikes: Array of strike prices
            maturities: Array of maturities (in years)
            atm_only: If True, only use ATM options for calibration
            
        Returns:
            Dictionary mapping maturity to discount factor
        """
        discount_factors = {}
        
        if atm_only:
            # Find ATM options (closest to spot)
            atm_indices = []
            unique_maturities = np.unique(maturities)
            
            for T in unique_maturities:
                T_mask = maturities == T
                T_strikes = strikes[T_mask]
                atm_idx = np.argmin(np.abs(T_strikes - spot))
                global_idx = np.where(T_mask)[0][atm_idx]
                atm_indices.append(global_idx)
            
            # Use only ATM options
            indices = np.array(atm_indices)
        else:
            indices = np.arange(len(call_prices))
        
        # Calculate discount factors from put-call parity
        for idx in indices:
            C = call_prices[idx]
            P = put_prices[idx]
            K = strikes[idx]
            T = maturities[idx]
            
            # DF(T) = (S - C + P) / K
            df = (spot - C + P) / K
            
            if T not in discount_factors:
                discount_factors[T] = []
            discount_factors[T].append(df)
        
        # Average discount factors for same maturity
        avg_discount_factors = {}
        for T, dfs in discount_factors.items():
            avg_discount_factors[T] = np.mean(dfs)
        
        return avg_discount_factors
    
    def bootstrap_zero_rates(self, discount_factors: Dict[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bootstrap zero rates from discount factors.
        
        Args:
            discount_factors: Dictionary mapping maturity to discount factor
            
        Returns:
            Tuple of (maturities, zero_rates)
        """
        maturities = np.array(sorted(discount_factors.keys()))
        dfs = np.array([discount_factors[T] for T in maturities])
        
        # Zero rate: r(T) = -log(DF(T)) / T
        zero_rates = -np.log(dfs) / maturities
        
        return maturities, zero_rates
    
    def build_zero_curve(self, maturities: np.ndarray, zero_rates: np.ndarray, 
                        method: str = 'cubic') -> callable:
        """
        Build interpolated zero curve.
        
        Args:
            maturities: Array of maturities
            zero_rates: Array of zero rates
            method: Interpolation method ('linear' or 'cubic')
            
        Returns:
            Interpolation function for zero rates
        """
        if method == 'cubic':
            return CubicSpline(maturities, zero_rates, extrapolate=True)
        else:
            return interp1d(maturities, zero_rates, kind='linear', 
                          fill_value='extrapolate')


class VolatilitySurfaceCalibrator:
    """Calibrate volatility surface from option prices."""
    
    def __init__(self):
        self.bsm = BlackScholesMerton()
        self.greeks_calc = GreeksCalculator()
    
    def calibrate_implied_vols(self, option_prices: np.ndarray, spot: float,
                              strikes: np.ndarray, maturities: np.ndarray,
                              rates: np.ndarray, option_types: List[str],
                              initial_guess: float = 0.2) -> np.ndarray:
        """
        Calibrate implied volatilities from option prices.
        
        Args:
            option_prices: Array of option prices
            spot: Current spot price
            strikes: Array of strike prices
            maturities: Array of maturities
            rates: Array of risk-free rates
            option_types: List of option types ('call' or 'put')
            initial_guess: Initial volatility guess
            
        Returns:
            Array of implied volatilities
        """
        implied_vols = []
        
        for i in range(len(option_prices)):
            iv, _ = self.greeks_calc.implied_volatility_greeks(
                option_prices[i], spot, strikes[i], 
                maturities[i], rates[i], option_types[i],
                initial_guess
            )
            implied_vols.append(iv)
        
        return np.array(implied_vols)
    
    def fit_ssvi(self, log_moneyness: np.ndarray, total_variance: np.ndarray,
                 atm_variance: float) -> Dict[str, float]:
        """
        Fit SSVI (Surface SVI) model to implied volatility data.
        
        SSVI parameterization:
        w(k) = θ/2 * (1 + ρ*φ*k + sqrt((φ*k + ρ)^2 + (1-ρ^2)))
        
        where k = log-moneyness, w = total variance
        
        Args:
            log_moneyness: Array of log-moneyness values
            total_variance: Array of total variance (σ²T)
            atm_variance: ATM total variance for normalization
            
        Returns:
            Dictionary of SSVI parameters
        """
        def ssvi_slice(k, theta, rho, phi):
            """SSVI slice function."""
            return 0.5 * theta * (1 + rho * phi * k + 
                                np.sqrt((phi * k + rho)**2 + (1 - rho**2)))
        
        def objective(params):
            """Objective function for SSVI calibration."""
            theta, rho, phi = params
            
            # Parameter constraints
            if theta <= 0 or phi <= 0:
                return 1e10
            if abs(rho) >= 1:
                return 1e10
            
            # Calculate model total variance
            model_variance = ssvi_slice(log_moneyness, theta, rho, phi)
            
            # Weighted least squares (weight by vega)
            weights = 1.0 / np.sqrt(total_variance + 0.0001)
            error = np.sum(weights * (model_variance - total_variance)**2)
            
            return error
        
        # Initial guess
        initial_params = [atm_variance, 0.0, 0.5]
        
        # Optimization
        result = minimize(objective, initial_params, 
                        method='Nelder-Mead',
                        options={'maxiter': 10000})
        
        theta, rho, phi = result.x
        
        return {
            'theta': theta,
            'rho': rho,
            'phi': phi,
            'success': result.success,
            'error': result.fun
        }
    
    def calibrate_surface(self, market_data: pd.DataFrame) -> Dict:
        """
        Calibrate full volatility surface from market data.
        
        Args:
            market_data: DataFrame with columns:
                - spot, strike, maturity, rate, option_type, price
                
        Returns:
            Dictionary containing calibrated surface parameters
        """
        # First calibrate implied volatilities
        implied_vols = self.calibrate_implied_vols(
            market_data['price'].values,
            market_data['spot'].iloc[0],
            market_data['strike'].values,
            market_data['maturity'].values,
            market_data['rate'].values,
            market_data['option_type'].tolist()
        )
        
        market_data['implied_vol'] = implied_vols
        
        # Group by maturity and fit SSVI for each slice
        surface_params = {}
        
        for T, group in market_data.groupby('maturity'):
            # Calculate log-moneyness and total variance
            spot = group['spot'].iloc[0]
            log_moneyness = np.log(group['strike'] / spot)
            total_variance = group['implied_vol']**2 * T
            
            # Find ATM variance
            atm_idx = np.argmin(np.abs(log_moneyness))
            atm_variance = total_variance.iloc[atm_idx]
            
            # Fit SSVI
            params = self.fit_ssvi(log_moneyness.values, 
                                 total_variance.values,
                                 atm_variance)
            
            surface_params[T] = params
        
        return surface_params
    
    def local_volatility_dupire(self, spot: float, strike: float, maturity: float,
                               surface_params: Dict, rate_curve: callable) -> float:
        """
        Calculate local volatility using Dupire formula.
        
        σ_loc²(K,T) = (∂C/∂T + rK∂C/∂K + qS∂C/∂S) / (0.5K²∂²C/∂K²)
        
        For simplicity, assuming no dividends (q=0).
        """
        # This is a simplified implementation
        # In practice, you'd need numerical differentiation of the call price surface
        
        # Get implied vol from surface
        log_moneyness = np.log(strike / spot)
        
        # Find closest maturity in surface params
        maturities = sorted(surface_params.keys())
        T_idx = np.searchsorted(maturities, maturity)
        
        if T_idx == 0:
            T_params = surface_params[maturities[0]]
        elif T_idx >= len(maturities):
            T_params = surface_params[maturities[-1]]
        else:
            # Interpolate between two maturities
            T1, T2 = maturities[T_idx-1], maturities[T_idx]
            w = (maturity - T1) / (T2 - T1)
            
            # Simple linear interpolation of parameters
            params1 = surface_params[T1]
            params2 = surface_params[T2]
            
            T_params = {
                'theta': (1-w) * params1['theta'] + w * params2['theta'],
                'rho': (1-w) * params1['rho'] + w * params2['rho'],
                'phi': (1-w) * params1['phi'] + w * params2['phi']
            }
        
        # Calculate implied variance
        theta, rho, phi = T_params['theta'], T_params['rho'], T_params['phi']
        implied_var = 0.5 * theta * (1 + rho * phi * log_moneyness + 
                                    np.sqrt((phi * log_moneyness + rho)**2 + (1 - rho**2)))
        
        # Simple approximation for local vol
        # In practice, you'd compute numerical derivatives
        local_vol = np.sqrt(implied_var / maturity)
        
        return local_vol