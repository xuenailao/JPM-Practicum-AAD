import pandas as pd
import numpy as np
from datetime import datetime
from greeks_calculator import GreeksCalculator
from calibration import DiscountCurveCalibrator, VolatilitySurfaceCalibrator
from bsm_model import BlackScholesMerton
import matplotlib.pyplot as plt


def load_and_prepare_spx_data(file_path):
    """Load SPX options data and prepare for analysis."""
    print("Loading SPX options data...")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert strike prices from cents to dollars
    df['strike_price'] = df['strike_price'] / 1000
    
    # Convert dates
    df['date'] = pd.to_datetime(df['date'])
    df['exdate'] = pd.to_datetime(df['exdate'])
    
    # Calculate time to maturity in years
    df['time_to_maturity'] = (df['exdate'] - df['date']).dt.days / 365.25
    
    # Calculate mid prices
    df['mid_price'] = (df['best_bid'] + df['best_offer']) / 2
    
    # Filter out options with missing data
    df = df.dropna(subset=['impl_volatility', 'delta', 'gamma', 'vega', 'theta'])
    
    print(f"Loaded {len(df)} option records")
    
    return df


def estimate_spot_price(df):
    """Estimate spot price from ATM options using put-call parity."""
    # Group by date and expiration
    grouped = df.groupby(['date', 'exdate'])
    
    spot_estimates = []
    
    for (date, exdate), group in grouped:
        # Find pairs of calls and puts at same strike
        calls = group[group['cp_flag'] == 'C'].set_index('strike_price')
        puts = group[group['cp_flag'] == 'P'].set_index('strike_price')
        
        # Find common strikes
        common_strikes = calls.index.intersection(puts.index)
        
        if len(common_strikes) > 0:
            # Use ATM strike (closest to previous estimate or mid-range)
            mid_strike = common_strikes[len(common_strikes)//2]
            
            if mid_strike in calls.index and mid_strike in puts.index:
                # Handle potential duplicates
                C_series = calls.loc[mid_strike, 'mid_price']
                P_series = puts.loc[mid_strike, 'mid_price']
                
                # If series, take first value
                C = C_series.iloc[0] if hasattr(C_series, 'iloc') else C_series
                P = P_series.iloc[0] if hasattr(P_series, 'iloc') else P_series
                
                # Estimate spot using put-call parity (assuming r ≈ 0 for short-term)
                # C - P = S - K
                S_estimate = float(C - P + mid_strike)
                spot_estimates.append(S_estimate)
    
    if spot_estimates:
        return np.median(spot_estimates)
    else:
        # Fallback: use strikes near delta = 0.5 for calls
        atm_calls = df[(df['cp_flag'] == 'C') & (df['delta'] > 0.45) & (df['delta'] < 0.55)]
        if len(atm_calls) > 0:
            return atm_calls['strike_price'].median()
        else:
            return df['strike_price'].median()


def test_greeks_calculation(df, spot_price, risk_free_rate=0.05):
    """Test AAD Greeks calculation against market Greeks."""
    print("\n" + "="*60)
    print("Testing AAD Greeks vs Market Greeks")
    print("="*60)
    
    # Create calculator
    calc = GreeksCalculator(use_forward_mode=True)
    
    # Sample some options for testing
    test_sample = df.sample(min(20, len(df)))
    
    results = []
    
    for idx, row in test_sample.iterrows():
        K = row['strike_price']
        T = row['time_to_maturity']
        market_iv = row['impl_volatility']
        option_type = 'call' if row['cp_flag'] == 'C' else 'put'
        
        # Calculate Greeks using AAD
        aad_greeks = calc.calculate_all_greeks(
            S=spot_price,
            K=K,
            T=T,
            r=risk_free_rate,
            sigma=market_iv,
            option_type=option_type
        )
        
        # Compare with market Greeks
        result = {
            'strike': K,
            'maturity': T,
            'type': option_type,
            'iv': market_iv,
            'market_delta': row['delta'],
            'aad_delta': aad_greeks['delta'],
            'delta_error': abs(aad_greeks['delta'] - row['delta']),
            'market_gamma': row['gamma'],
            'aad_gamma': aad_greeks['gamma'],
            'gamma_error': abs(aad_greeks['gamma'] - row['gamma']),
            'market_vega': row['vega'],
            'aad_vega': aad_greeks['vega'],
            'vega_error': abs(aad_greeks['vega'] - row['vega']),
            'market_theta': row['theta'],
            'aad_theta': aad_greeks['theta'],
            'theta_error': abs(aad_greeks['theta'] - row['theta'])
        }
        results.append(result)
    
    results_df = pd.DataFrame(results)
    
    # Print summary statistics
    print("\nGreeks Comparison Summary:")
    print(f"{'Greek':<10} {'Mean Abs Error':<20} {'Max Abs Error':<20} {'Mean Rel Error %':<20}")
    print("-" * 70)
    
    for greek in ['delta', 'gamma', 'vega', 'theta']:
        mean_error = results_df[f'{greek}_error'].mean()
        max_error = results_df[f'{greek}_error'].max()
        
        # Calculate relative error (avoid division by zero)
        market_vals = results_df[f'market_{greek}'].abs()
        rel_errors = results_df[f'{greek}_error'] / market_vals.where(market_vals > 0.001, 1)
        mean_rel_error = rel_errors.mean() * 100
        
        print(f"{greek:<10} {mean_error:<20.6f} {max_error:<20.6f} {mean_rel_error:<20.2f}")
    
    return results_df


def calibrate_volatility_surface(df, spot_price, risk_free_rate=0.05):
    """Calibrate volatility surface from market data."""
    print("\n" + "="*60)
    print("Calibrating Volatility Surface")
    print("="*60)
    
    # Select options for calibration (liquid strikes)
    calibration_data = df[
        (df['time_to_maturity'] > 0.01) &  # At least a few days to expiry
        (df['time_to_maturity'] < 2.0) &   # Less than 2 years
        (df['strike_price'] > spot_price * 0.7) &  # Not too far OTM
        (df['strike_price'] < spot_price * 1.3)    # Not too far ITM
    ].copy()
    
    # Prepare data for calibration
    calibration_data['spot'] = spot_price
    calibration_data['rate'] = risk_free_rate
    calibration_data['option_type'] = calibration_data['cp_flag'].map({'C': 'call', 'P': 'put'})
    calibration_data = calibration_data.rename(columns={
        'mid_price': 'price',
        'time_to_maturity': 'maturity',
        'strike_price': 'strike'
    })
    
    # Sample by maturity buckets
    maturity_buckets = pd.cut(calibration_data['maturity'], bins=5)
    sampled_data = []
    
    for bucket, group in calibration_data.groupby(maturity_buckets):
        # Sample up to 20 options per bucket
        sampled_data.append(group.sample(min(20, len(group))))
    
    calibration_subset = pd.concat(sampled_data)
    
    print(f"Using {len(calibration_subset)} options for calibration")
    
    # Calibrate surface
    calibrator = VolatilitySurfaceCalibrator()
    surface_params = calibrator.calibrate_surface(
        calibration_subset[['spot', 'strike', 'maturity', 'rate', 'option_type', 'price']]
    )
    
    # Display results
    print("\nCalibrated SSVI Parameters by Maturity:")
    for T in sorted(surface_params.keys()):
        params = surface_params[T]
        print(f"\nT = {T:.3f} years:")
        print(f"  θ (theta): {params['theta']:.4f}")
        print(f"  ρ (rho):   {params['rho']:.4f}")
        print(f"  φ (phi):   {params['phi']:.4f}")
        print(f"  Success:   {params['success']}")
    
    return surface_params


def plot_volatility_smile(df, spot_price):
    """Plot volatility smile for different maturities."""
    print("\n" + "="*60)
    print("Plotting Volatility Smile")
    print("="*60)
    
    # Select a few representative maturities
    unique_maturities = sorted(df['time_to_maturity'].unique())
    selected_maturities = []
    
    # Pick short, medium, and long term maturities
    if len(unique_maturities) >= 3:
        indices = [0, len(unique_maturities)//2, -1]
        selected_maturities = [unique_maturities[i] for i in indices]
    else:
        selected_maturities = unique_maturities[:3]
    
    plt.figure(figsize=(10, 6))
    
    for T in selected_maturities:
        # Get options for this maturity
        maturity_data = df[
            (abs(df['time_to_maturity'] - T) < 0.01) &
            (df['cp_flag'] == 'C')  # Use calls only
        ].copy()
        
        if len(maturity_data) > 0:
            # Calculate moneyness
            maturity_data['moneyness'] = maturity_data['strike_price'] / spot_price
            
            # Sort by moneyness
            maturity_data = maturity_data.sort_values('moneyness')
            
            # Plot
            plt.plot(maturity_data['moneyness'], 
                    maturity_data['impl_volatility'],
                    marker='o', 
                    label=f'T = {T:.3f} years')
    
    plt.xlabel('Moneyness (K/S)')
    plt.ylabel('Implied Volatility')
    plt.title('SPX Implied Volatility Smile')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/home/xuenailao/first_order_AAD/spx_volatility_smile.png')
    print("Volatility smile plot saved to spx_volatility_smile.png")
    plt.close()


def main():
    """Main function to test AAD framework with SPX data."""
    # Load data
    df = load_and_prepare_spx_data('/home/xuenailao/first_order_AAD/SPX_Aug.csv')
    
    # Get first date's data for testing
    first_date = df['date'].min()
    df_first_date = df[df['date'] == first_date]
    
    print(f"\nAnalyzing options for date: {first_date.strftime('%Y-%m-%d')}")
    print(f"Number of options: {len(df_first_date)}")
    
    # Estimate spot price
    spot_price = estimate_spot_price(df_first_date)
    print(f"Estimated spot price: ${spot_price:.2f}")
    
    # Test Greeks calculation
    results_df = test_greeks_calculation(df_first_date, spot_price)
    
    # Save detailed results
    results_df.to_csv('/home/xuenailao/first_order_AAD/greeks_comparison.csv', index=False)
    print("\nDetailed comparison saved to greeks_comparison.csv")
    
    # Calibrate volatility surface
    surface_params = calibrate_volatility_surface(df_first_date, spot_price)
    
    # Plot volatility smile
    plot_volatility_smile(df_first_date, spot_price)
    
    # Test put-call parity with calibrated data
    print("\n" + "="*60)
    print("Testing Put-Call Parity")
    print("="*60)
    
    # Find call-put pairs
    calls = df_first_date[df_first_date['cp_flag'] == 'C'].set_index(['strike_price', 'time_to_maturity'])
    puts = df_first_date[df_first_date['cp_flag'] == 'P'].set_index(['strike_price', 'time_to_maturity'])
    
    common_indices = calls.index.intersection(puts.index)
    
    parity_errors = []
    for idx in common_indices[:10]:  # Test first 10 pairs
        call_price = calls.loc[idx, 'mid_price']
        put_price = puts.loc[idx, 'mid_price']
        strike = idx[0]
        maturity = idx[1]
        
        # Put-call parity: C - P = S - K*exp(-rT)
        # Using r = 0.05 as approximation
        r = 0.05
        theoretical_diff = spot_price - strike * np.exp(-r * maturity)
        actual_diff = call_price - put_price
        error = abs(actual_diff - theoretical_diff)
        
        parity_errors.append(error)
        
        if len(parity_errors) <= 5:  # Print first 5
            # Handle potential series
            actual_diff_val = actual_diff.iloc[0] if hasattr(actual_diff, 'iloc') else actual_diff
            print(f"K={strike:.0f}, T={maturity:.3f}: "
                  f"C-P={actual_diff_val:.2f}, S-K*exp(-rT)={theoretical_diff:.2f}, "
                  f"Error=${error:.2f}")
    
    print(f"\nMean put-call parity error: ${np.mean(parity_errors):.2f}")
    
    print("\n" + "="*60)
    print("SPX Data Testing Complete!")
    print("="*60)


if __name__ == "__main__":
    main()