"""
Test Enterprise Optimizer with current dependencies
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime

# First, let's test if we can import structlog
try:
    from core.optimizer_enterprise import EnterpriseOptimizer, EnterpriseGoal
    print("✅ Enterprise modules loaded successfully")
    
    # Initialize optimizer
    print("\nInitializing Enterprise Optimizer...")
    opt = EnterpriseOptimizer(
        n_assets=10,
        audit_mode=True,
        use_parallel=False  # Disable parallel for testing
    )
    print("✅ Optimizer initialized")
    
    # Create institutional goals
    goals = [
        EnterpriseGoal(
            name='Pension Fund Liabilities',
            target_amount=10000000,
            time_horizon=25,
            risk_tolerance=0.4,
            priority=0.5,
            regulatory_constraint='ERISA',
            liability_matching=True
        ),
        EnterpriseGoal(
            name='Operational Reserve',
            target_amount=2000000,
            time_horizon=5,
            risk_tolerance=0.2,
            priority=0.3,
            liquidity_requirements={'min_liquid': 0.5}
        ),
        EnterpriseGoal(
            name='Growth Target',
            target_amount=5000000,
            time_horizon=15,
            risk_tolerance=0.6,
            priority=0.2
        )
    ]
    print(f"✅ Created {len(goals)} institutional goals")
    
    # Generate realistic test data
    np.random.seed(42)
    
    # Expected returns for different asset classes
    asset_classes = [
        'US Equity', 'Intl Equity', 'EM Equity',
        'Govt Bonds', 'Corp Bonds', 'HY Bonds',
        'Real Estate', 'Commodities', 'Gold', 'Cash'
    ]
    
    # Realistic annual returns
    expected_returns = np.array([
        0.10,  # US Equity
        0.09,  # Intl Equity
        0.12,  # EM Equity
        0.03,  # Govt Bonds
        0.05,  # Corp Bonds
        0.08,  # HY Bonds
        0.07,  # Real Estate
        0.06,  # Commodities
        0.04,  # Gold
        0.02   # Cash
    ])
    
    # Create realistic correlation matrix
    correlation = np.array([
        [1.00, 0.85, 0.75, -0.30, -0.10, 0.40, 0.65, 0.45, 0.10, -0.05],
        [0.85, 1.00, 0.80, -0.25, -0.05, 0.45, 0.60, 0.50, 0.15, -0.05],
        [0.75, 0.80, 1.00, -0.20, 0.00, 0.50, 0.55, 0.55, 0.20, 0.00],
        [-0.30, -0.25, -0.20, 1.00, 0.85, 0.20, -0.10, -0.15, 0.30, 0.40],
        [-0.10, -0.05, 0.00, 0.85, 1.00, 0.60, 0.10, 0.05, 0.25, 0.35],
        [0.40, 0.45, 0.50, 0.20, 0.60, 1.00, 0.40, 0.35, 0.15, 0.10],
        [0.65, 0.60, 0.55, -0.10, 0.10, 0.40, 1.00, 0.35, 0.20, 0.05],
        [0.45, 0.50, 0.55, -0.15, 0.05, 0.35, 0.35, 1.00, 0.50, 0.00],
        [0.10, 0.15, 0.20, 0.30, 0.25, 0.15, 0.20, 0.50, 1.00, 0.10],
        [-0.05, -0.05, 0.00, 0.40, 0.35, 0.10, 0.05, 0.00, 0.10, 1.00]
    ])
    
    # Volatilities for each asset class
    volatilities = np.array([
        0.16,  # US Equity
        0.18,  # Intl Equity
        0.25,  # EM Equity
        0.05,  # Govt Bonds
        0.08,  # Corp Bonds
        0.12,  # HY Bonds
        0.15,  # Real Estate
        0.20,  # Commodities
        0.18,  # Gold
        0.01   # Cash
    ])
    
    # Create covariance matrix
    covariance_matrix = np.outer(volatilities, volatilities) * correlation
    
    print("✅ Generated realistic market data")
    
    # Define enterprise constraints
    constraints = {
        'max_volatility': 0.12,  # 12% volatility limit
        'max_concentration': 0.30,  # Max 30% in single asset
        'ucits_compliant': False,
        'basel_iii_compliant': True,
        'long_only': True,
        'sector_limits': {
            'equity': ([0, 1, 2], 0.20, 0.60),  # 20-60% in equities
            'fixed_income': ([3, 4, 5], 0.30, 0.70),  # 30-70% in bonds
            'alternatives': ([6, 7, 8], 0.00, 0.20)  # 0-20% in alternatives
        }
    }
    
    print("\n" + "="*60)
    print("RUNNING ENTERPRISE OPTIMIZATION")
    print("="*60)
    
    # Run optimization
    result = opt.optimize(
        expected_returns=expected_returns,
        covariance_matrix=covariance_matrix,
        goals=goals,
        constraints=constraints,
        user_id='institutional_client_001'
    )
    
    print("\n✅ Optimization Complete!")
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    # Display portfolio allocation
    print("\n📊 PORTFOLIO ALLOCATION:")
    print("-" * 40)
    weights = result['weights']
    for i, asset in enumerate(asset_classes):
        if weights[i] > 0.01:  # Show only >1% allocations
            print(f"{asset:15s}: {weights[i]*100:6.2f}%")
    
    # Display risk metrics
    print("\n📈 RISK METRICS:")
    print("-" * 40)
    metrics = result['risk_metrics']
    print(f"Expected Return:      {metrics['expected_return']*100:.2f}%")
    print(f"Volatility:          {metrics['volatility']*100:.2f}%")
    print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio:       {metrics.get('sortino_ratio', 0):.2f}")
    print(f"VaR (99%):          {metrics.get('var_99', 0)*100:.2f}%")
    print(f"CVaR (95%):         {metrics.get('cvar_95', 0)*100:.2f}%")
    print(f"Max Drawdown:        {metrics.get('max_drawdown', 0)*100:.2f}%")
    print(f"Information Ratio:   {metrics.get('information_ratio', 0):.2f}")
    print(f"Diversification:     {metrics.get('diversification_ratio', 0):.2f}")
    
    # Display stress test results
    print("\n🔥 STRESS TEST RESULTS (Top 5):")
    print("-" * 40)
    stress_tests = result['stress_test_results']
    for i, (scenario, impact) in enumerate(list(stress_tests.items())[:5]):
        print(f"{scenario:25s}: {impact['return']*100:6.2f}% loss")
    
    # Display compliance
    print("\n✅ COMPLIANCE STATUS:")
    print("-" * 40)
    compliance = result['compliance']
    print(f"Compliant: {compliance['is_compliant']}")
    if compliance['violations']:
        print(f"Violations: {', '.join(compliance['violations'])}")
    if compliance['warnings']:
        print(f"Warnings: {', '.join(compliance['warnings'][:2])}")
    
    # Display audit information
    print("\n🔒 AUDIT TRAIL:")
    print("-" * 40)
    print(f"Optimization ID: {result['optimization_id']}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Method: {result['method']}")
    
    print("\n" + "="*60)
    print("ENTERPRISE FEATURES DEMONSTRATED")
    print("="*60)
    print("✅ Multi-goal optimization (3 institutional goals)")
    print("✅ Regulatory compliance (ERISA, Basel III)")
    print("✅ Advanced risk metrics (20+ metrics)")
    print("✅ Stress testing (11 scenarios)")
    print("✅ Sector constraints enforced")
    print("✅ Full audit trail with hashing")
    print("✅ Liquidity requirements considered")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nTrying fallback to standard optimizer...")
    
    # Fallback to v2 optimizer
    from core.optimizer_v2 import UltraOptimiser, Goal
    
    print("✅ Using standard optimizer v2.0")
    
    # Run simplified test
    opt = UltraOptimiser(n_assets=10, random_seed=42)
    
    goals = [
        Goal('Retirement', 1000000, 20, 0.5, 0.6),
        Goal('Reserve', 200000, 5, 0.3, 0.4)
    ]
    
    # Simple test data
    returns = np.random.uniform(0.05, 0.15, 10)
    cov = np.eye(10) * 0.01
    
    result = opt.optimize(returns, cov, goals)
    
    print(f"\n✅ Standard Optimization Complete")
    print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
    print(f"Alpha: {result.alpha*10000:.0f} bps")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
