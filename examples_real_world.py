"""
Real-world example using UltraOptimiser with market data
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from core.optimizer import UltraOptimiser, Goal

def fetch_market_data(tickers, period="1y"):
    """Fetch real market data from Yahoo Finance"""
    print(f"Fetching data for {tickers}...")
    
    # Download data
    data = yf.download(tickers, period=period, progress=False)
    
    # Handle both single and multiple tickers
    if len(tickers) == 1:
        prices = data['Adj Close'].to_frame()
    else:
        # For multiple tickers, extract Adjusted Close prices
        prices = data['Adj Close'] if 'Adj Close' in data.columns.levels[0] else data['Close']
    
    # Ensure we have a DataFrame
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Calculate expected returns (annualized)
    expected_returns = returns.mean() * 252
    
    # Calculate covariance matrix (annualized)
    cov_matrix = returns.cov() * 252
    
    # Get asset names
    if isinstance(prices.columns, pd.MultiIndex):
        asset_names = prices.columns.tolist()
    else:
        asset_names = list(prices.columns)
    
    return expected_returns.values, cov_matrix.values, asset_names

def main():
    # Popular ETF portfolio
    tickers = [
        "SPY",  # S&P 500
        "QQQ",  # NASDAQ
        "IWM",  # Russell 2000
        "EFA",  # International Developed
        "EEM",  # Emerging Markets
        "AGG",  # Bonds
        "GLD",  # Gold
        "VNQ",  # Real Estate
        "DBC",  # Commodities
        "TLT"   # Long-term Treasuries
    ]
    
    try:
        # Fetch real market data
        expected_returns, cov_matrix, asset_names = fetch_market_data(tickers)
        
        # Initialize optimizer
        optimizer = UltraOptimiser(n_assets=len(tickers))
        
        # Define realistic investment goals
        goals = [
            Goal(
                name="Retirement",
                target_amount=2000000,
                time_horizon=25,
                risk_tolerance=0.7,
                priority=0.4
            ),
            Goal(
                name="Children Education",
                target_amount=300000,
                time_horizon=15,
                risk_tolerance=0.5,
                priority=0.3
            ),
            Goal(
                name="House Down Payment",
                target_amount=150000,
                time_horizon=5,
                risk_tolerance=0.3,
                priority=0.3
            )
        ]
        
        # Run optimization
        print("\n" + "="*60)
        print("ULTRAOPTIMISER - REAL MARKET DATA ANALYSIS")
        print("="*60)
        
        result = optimizer.optimize(
            expected_returns=expected_returns,
            covariance_matrix=cov_matrix,
            goals=goals,
            constraints={'max_risk': 0.20, 'max_position': 0.35}
        )
        
        # Display results
        print("\n📊 OPTIMAL PORTFOLIO ALLOCATION:")
        print("-" * 40)
        
        # Sort weights for display
        weights_df = pd.DataFrame({
            'Asset': asset_names,
            'Weight': result.weights * 100
        }).sort_values('Weight', ascending=False)
        
        for _, row in weights_df.iterrows():
            if row['Weight'] > 0.5:  # Only show positions > 0.5%
                print(f"  {row['Asset']:5s}: {row['Weight']:6.2f}%")
        
        print("\n📈 PERFORMANCE METRICS:")
        print("-" * 40)
        print(f"  Expected Return: {result.expected_return:.2%} annually")
        print(f"  Portfolio Risk: {result.risk:.2%}")
        print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Alpha: {result.alpha * 10000:.0f} bps")
        print(f"  Max Drawdown: {result.max_drawdown:.1%}")
        
        print("\n🎯 GOAL ACHIEVEMENT:")
        print("-" * 40)
        print(f"  Overall Probability: {result.goal_achievement_probability:.1%}")
        
        print("\n💰 TAX & COST EFFICIENCY:")
        print("-" * 40)
        print(f"  Tax Alpha: {result.tax_alpha * 10000:.0f} bps")
        print(f"  Transaction Cost: {result.total_cost:.3%}")
        
        print("\n🌍 MARKET REGIME:")
        print("-" * 40)
        print(f"  Current Regime: {result.regime.upper()}")
        
        # Save results
        weights_df.to_csv('portfolio_allocation.csv', index=False)
        print("\n✅ Results saved to portfolio_allocation.csv")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTrying alternative approach with simulated data...")
        return run_with_simulated_data()

def run_with_simulated_data():
    """Run with simulated market data as fallback"""
    print("\n" + "="*60)
    print("ULTRAOPTIMISER - SIMULATED MARKET DATA")
    print("="*60)
    
    # Asset names
    asset_names = ["SPY", "QQQ", "IWM", "EFA", "EEM", "AGG", "GLD", "VNQ", "DBC", "TLT"]
    n_assets = len(asset_names)
    
    # Simulate realistic returns and correlations
    np.random.seed(42)
    
    # Expected returns (realistic annual returns)
    expected_returns = np.array([0.10, 0.12, 0.08, 0.07, 0.09, 0.04, 0.06, 0.08, 0.05, 0.03])
    
    # Create correlation matrix
    correlation = np.array([
        [1.00, 0.95, 0.85, 0.75, 0.70, -0.20, 0.10, 0.65, 0.40, -0.30],  # SPY
        [0.95, 1.00, 0.80, 0.70, 0.65, -0.25, 0.05, 0.60, 0.35, -0.35],  # QQQ
        [0.85, 0.80, 1.00, 0.70, 0.65, -0.15, 0.15, 0.70, 0.45, -0.25],  # IWM
        [0.75, 0.70, 0.70, 1.00, 0.85, -0.10, 0.20, 0.55, 0.50, -0.20],  # EFA
        [0.70, 0.65, 0.65, 0.85, 1.00, -0.05, 0.25, 0.50, 0.55, -0.15],  # EEM
        [-0.20, -0.25, -0.15, -0.10, -0.05, 1.00, 0.10, -0.10, -0.05, 0.85],  # AGG
        [0.10, 0.05, 0.15, 0.20, 0.25, 0.10, 1.00, 0.15, 0.60, 0.15],  # GLD
        [0.65, 0.60, 0.70, 0.55, 0.50, -0.10, 0.15, 1.00, 0.35, -0.15],  # VNQ
        [0.40, 0.35, 0.45, 0.50, 0.55, -0.05, 0.60, 0.35, 1.00, -0.10],  # DBC
        [-0.30, -0.35, -0.25, -0.20, -0.15, 0.85, 0.15, -0.15, -0.10, 1.00]  # TLT
    ])
    
    # Standard deviations (realistic annual volatilities)
    std_devs = np.array([0.16, 0.22, 0.20, 0.18, 0.24, 0.05, 0.15, 0.19, 0.18, 0.12])
    
    # Create covariance matrix
    cov_matrix = np.outer(std_devs, std_devs) * correlation
    
    # Initialize optimizer
    optimizer = UltraOptimiser(n_assets=n_assets)
    
    # Define goals
    goals = [
        Goal(
            name="Retirement",
            target_amount=2000000,
            time_horizon=25,
            risk_tolerance=0.7,
            priority=0.4
        ),
        Goal(
            name="Children Education",
            target_amount=300000,
            time_horizon=15,
            risk_tolerance=0.5,
            priority=0.3
        ),
        Goal(
            name="House Down Payment",
            target_amount=150000,
            time_horizon=5,
            risk_tolerance=0.3,
            priority=0.3
        )
    ]
    
    # Run optimization
    result = optimizer.optimize(
        expected_returns=expected_returns,
        covariance_matrix=cov_matrix,
        goals=goals,
        constraints={'max_risk': 0.20, 'max_position': 0.35}
    )
    
    # Display results
    print("\n📊 OPTIMAL PORTFOLIO ALLOCATION:")
    print("-" * 40)
    
    weights_df = pd.DataFrame({
        'Asset': asset_names,
        'Weight': result.weights * 100
    }).sort_values('Weight', ascending=False)
    
    for _, row in weights_df.iterrows():
        if row['Weight'] > 0.5:
            print(f"  {row['Asset']:5s}: {row['Weight']:6.2f}%")
    
    print("\n📈 PERFORMANCE METRICS:")
    print("-" * 40)
    print(f"  Expected Return: {result.expected_return:.2%} annually")
    print(f"  Portfolio Risk: {result.risk:.2%}")
    print(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"  Alpha: {result.alpha * 10000:.0f} bps")
    print(f"  Max Drawdown: {result.max_drawdown:.1%}")
    
    print("\n🎯 GOAL ACHIEVEMENT:")
    print("-" * 40)
    print(f"  Overall Probability: {result.goal_achievement_probability:.1%}")
    
    print("\n💰 TAX & COST EFFICIENCY:")
    print("-" * 40)
    print(f"  Tax Alpha: {result.tax_alpha * 10000:.0f} bps")
    print(f"  Transaction Cost: {result.total_cost:.3%}")
    
    print("\n🌍 MARKET REGIME:")
    print("-" * 40)
    print(f"  Current Regime: {result.regime.upper()}")
    
    # Save results
    weights_df.to_csv('portfolio_allocation.csv', index=False)
    print("\n✅ Results saved to portfolio_allocation.csv")
    
    return result

if __name__ == "__main__":
    result = main()
    print("\n🎉 Optimization complete!")
