import numpy as np
from greeks_calculator import GreeksCalculator
import pandas as pd


def test_greek_conventions():
    """Test different Greek conventions to understand market data discrepancies."""
    
    print("Testing Greek Conventions")
    print("=" * 60)
    
    # Example option parameters
    S = 4600.0   # Spot
    K = 4600.0   # ATM strike
    T = 0.25     # 3 months
    r = 0.05     # Risk-free rate
    sigma = 0.15 # Volatility
    
    calc = GreeksCalculator(use_forward_mode=True)
    greeks = calc.calculate_all_greeks(S, K, T, r, sigma, 'call')
    
    print(f"\nOption Parameters:")
    print(f"Spot: ${S:.2f}")
    print(f"Strike: ${K:.2f}")
    print(f"Time to maturity: {T:.2f} years ({T*365:.0f} days)")
    print(f"Volatility: {sigma:.1%}")
    print(f"Risk-free rate: {r:.1%}")
    
    print(f"\nCalculated Greeks (AAD):")
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.6f}")
    print(f"Vega: {greeks['vega']:.2f}")
    print(f"Theta: {greeks['theta']:.2f}")
    print(f"Rho: {greeks['rho']:.2f}")
    
    # Show different conventions
    print(f"\nGreek Convention Variations:")
    
    # Vega conventions
    vega_per_1pct = greeks['vega']
    vega_per_1bp = greeks['vega'] / 100
    print(f"\nVega:")
    print(f"  Per 1% move: {vega_per_1pct:.2f}")
    print(f"  Per 1 basis point: {vega_per_1bp:.4f}")
    
    # Theta conventions
    theta_annual = greeks['theta']
    theta_daily = greeks['theta'] / 365
    theta_positive = -greeks['theta']  # Some systems report as positive decay
    print(f"\nTheta:")
    print(f"  Annual (negative = decay): {theta_annual:.2f}")
    print(f"  Daily decay: {theta_daily:.4f}")
    print(f"  Positive convention: {theta_positive:.2f}")
    
    # Gamma conventions
    gamma_spot = greeks['gamma']
    gamma_pct = greeks['gamma'] * S / 100  # Gamma per 1% spot move
    print(f"\nGamma:")
    print(f"  Per $1 spot move: {gamma_spot:.6f}")
    print(f"  Per 1% spot move: {gamma_pct:.4f}")
    
    # Dollar Greeks
    print(f"\nDollar Greeks (position size = 100 contracts):")
    position_size = 100
    multiplier = 100  # Index multiplier
    dollar_delta = greeks['delta'] * position_size * multiplier * S
    dollar_gamma = greeks['gamma'] * position_size * multiplier * S * S
    dollar_vega = greeks['vega'] * position_size * multiplier
    dollar_theta = greeks['theta'] * position_size * multiplier
    
    print(f"  Dollar Delta: ${dollar_delta:,.0f}")
    print(f"  Dollar Gamma: ${dollar_gamma:,.0f}")
    print(f"  Dollar Vega: ${dollar_vega:,.0f}")
    print(f"  Dollar Theta: ${dollar_theta:,.0f}")
    
    # Create comparison table
    print(f"\n\nPossible Sources of Greek Differences:")
    print("-" * 60)
    print("1. Theta: Market may use positive convention or daily instead of annual")
    print("2. Vega: Market may scale per basis point instead of per 1%")
    print("3. Interest rates: Market uses term structure, we use flat rate")
    print("4. Dividends: SPX has dividend yield not included in our model")
    print("5. Business days: Market may use business day conventions")
    
    return greeks


def analyze_spx_greek_scaling():
    """Analyze the SPX data to understand Greek scaling."""
    
    print("\n\nAnalyzing SPX Data Greek Scaling")
    print("=" * 60)
    
    # Load a sample of the comparison data
    try:
        df = pd.read_csv('/home/xuenailao/first_order_AAD/greeks_comparison.csv')
        
        # Look at theta sign
        print("\nTheta Analysis:")
        print(f"Market theta < 0: {(df['market_theta'] < 0).sum()} options")
        print(f"Market theta > 0: {(df['market_theta'] > 0).sum()} options")
        print(f"AAD theta < 0: {(df['aad_theta'] < 0).sum()} options")
        
        # Check if flipping theta sign improves match
        df['theta_error_flipped'] = abs(df['aad_theta'] + df['market_theta'])
        df['theta_improvement'] = df['theta_error'] - df['theta_error_flipped']
        
        print(f"\nTheta sign flip analysis:")
        print(f"Improved by flipping: {(df['theta_improvement'] > 0).sum()} options")
        print(f"Average improvement: {df['theta_improvement'].mean():.2f}")
        
        # Check vega scaling
        print("\nVega scaling analysis:")
        df['vega_ratio'] = df['market_vega'] / df['aad_vega']
        median_ratio = df['vega_ratio'].median()
        print(f"Median market/AAD vega ratio: {median_ratio:.3f}")
        
        if abs(median_ratio - 0.01) < 0.1:
            print("Market vega appears to be scaled per basis point (1/100)")
        elif abs(median_ratio - 100) < 10:
            print("Market vega appears to be scaled up by 100")
            
    except FileNotFoundError:
        print("Run test_spx_data.py first to generate comparison file")


if __name__ == "__main__":
    test_greek_conventions()
    analyze_spx_greek_scaling()