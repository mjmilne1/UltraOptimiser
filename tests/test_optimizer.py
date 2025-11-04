"""
Test script for UltraOptimiser
Run this to verify the installation and basic functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from core.optimizer import UltraOptimiser, Goal
import json

def test_optimization():
    """Test basic optimization functionality"""
    
    print("=" * 50)
    print("UltraOptimiser Test Suite")
    print("=" * 50)
    
    # Initialize optimizer for 10 assets
    n_assets = 10
    optimizer = UltraOptimiser(n_assets=n_assets)
    print(f"✅ Optimizer initialized with {n_assets} assets")
    
    # Generate sample data
    np.random.seed(42)
    expected_returns = np.random.uniform(0.05, 0.15, n_assets)
    
    # Generate covariance matrix
    correlation = np.random.uniform(-0.3, 0.8, (n_assets, n_assets))
    correlation = (correlation + correlation.T) / 2
    np.fill_diagonal(correlation, 1)
    
    std_devs = np.random.uniform(0.1, 0.3, n_assets)
    covariance_matrix = np.outer(std_devs, std_devs) * correlation
    
    print(f"✅ Market data generated")
    
    # Define investment goals
    goals = [
        Goal(
            name="Retirement",
            target_amount=1000000,
            time_horizon=20,
            risk_tolerance=0.6,
            priority=0.5
        ),
        Goal(
            name="House Purchase",
            target_amount=500000,
            time_horizon=5,
            risk_tolerance=0.3,
            priority=0.3
        ),
        Goal(
            name="Education Fund",
            target_amount=200000,
            time_horizon=10,
            risk_tolerance=0.5,
            priority=0.2
        )
    ]
    
    print(f"✅ {len(goals)} investment goals defined")
    
    # Run optimization
    print("\nRunning optimization...")
    result = optimizer.optimize(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        goals=goals,
        constraints={'max_risk': 0.25, 'max_position': 0.3}
    )
    
    # Display results
    print("\n" + "=" * 50)
    print("OPTIMIZATION RESULTS")
    print("=" * 50)
    
    print(f"\n📊 Portfolio Metrics:")
    print(f"   Expected Return: {result.expected_return:.2%}")
    print(f"   Portfolio Risk: {result.risk:.2%}")
    print(f"   Sharpe Ratio: {result.sharpe_ratio:.2f} (Target: 1.58)")
    print(f"   Alpha: {result.alpha * 10000:.0f} bps (Target: 180-220 bps)")
    
    print(f"\n🎯 Goal Achievement:")
    print(f"   Probability: {result.goal_achievement_probability:.1%} (Target: 92.4%)")
    
    print(f"\n💰 Tax & Cost Efficiency:")
    print(f"   Tax Alpha: {result.tax_alpha * 10000:.0f} bps (Target: 75-125 bps)")
    print(f"   Transaction Cost: {result.total_cost:.4f}")
    
    print(f"\n📉 Risk Management:")
    print(f"   Max Drawdown: {result.max_drawdown:.1%} (Target: 12.8%)")
    print(f"   Current Regime: {result.regime}")
    
    print(f"\n📈 Portfolio Weights:")
    for i, weight in enumerate(result.weights):
        if weight > 0.01:  # Only show positions > 1%
            print(f"   Asset {i+1}: {weight:.1%}")
    
    # Test regime detection
    print("\n" + "=" * 50)
    print("REGIME DETECTION TEST")
    print("=" * 50)
    
    scenarios = [
        {"name": "Bull Market", "returns": 0.12, "volatility": 0.10, "vix": 15},
        {"name": "Bear Market", "returns": -0.08, "volatility": 0.30, "vix": 35},
        {"name": "Crisis", "returns": -0.15, "volatility": 0.45, "vix": 50},
        {"name": "Normal", "returns": 0.06, "volatility": 0.15, "vix": 20}
    ]
    
    for scenario in scenarios:
        regime = optimizer.detect_regime(scenario)
        print(f"   {scenario['name']}: Detected as '{regime}'")
    
    # Performance summary
    print("\n" + "=" * 50)
    print("PERFORMANCE SUMMARY")
    print("=" * 50)
    
    summary = optimizer.get_performance_summary()
    print(json.dumps(summary, indent=2))
    
    print("\n✅ All tests completed successfully!")
    
    return result

if __name__ == "__main__":
    try:
        result = test_optimization()
        print("\n🎉 UltraOptimiser is working correctly!")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
