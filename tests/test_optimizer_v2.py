"""
Unit tests for UltraOptimiser v2.0
"""

import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.optimizer_v2 import UltraOptimiser, Goal, validate_portfolio_data


class TestUltraOptimiser(unittest.TestCase):
    """Test suite for UltraOptimiser"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.n_assets = 5
        self.optimizer = UltraOptimiser(self.n_assets, random_seed=42)
        
        # Create test data
        np.random.seed(42)
        self.returns = np.random.uniform(0.05, 0.15, self.n_assets)
        
        # Create valid covariance matrix
        correlation = np.random.uniform(-0.3, 0.8, (self.n_assets, self.n_assets))
        correlation = (correlation + correlation.T) / 2
        np.fill_diagonal(correlation, 1)
        std_devs = np.random.uniform(0.1, 0.3, self.n_assets)
        self.cov_matrix = np.outer(std_devs, std_devs) * correlation
        
        # Create test goals
        self.goals = [
            Goal("Retirement", 1000000, 20, 0.6, 0.5),
            Goal("House", 500000, 5, 0.3, 0.5)
        ]
    
    def test_initialization(self):
        """Test optimizer initialization"""
        self.assertEqual(self.optimizer.n_assets, 5)
        self.assertEqual(self.optimizer.risk_free_rate, 0.04)
        self.assertEqual(self.optimizer.current_regime, 'normal')
    
    def test_invalid_initialization(self):
        """Test invalid initialization parameters"""
        with self.assertRaises(ValueError):
            UltraOptimiser(-1)  # Negative assets
        
        with self.assertRaises(ValueError):
            UltraOptimiser(10, risk_free_rate=1.5)  # Invalid risk-free rate
    
    def test_goal_validation(self):
        """Test goal parameter validation"""
        with self.assertRaises(ValueError):
            Goal("Invalid", -1000, 5, 0.5, 0.5)  # Negative target
        
        with self.assertRaises(ValueError):
            Goal("Invalid", 1000, -5, 0.5, 0.5)  # Negative time horizon
        
        with self.assertRaises(ValueError):
            Goal("Invalid", 1000, 5, 1.5, 0.5)  # Risk tolerance > 1
    
    def test_input_validation(self):
        """Test input validation in optimize method"""
        # Wrong dimensions
        wrong_returns = np.ones(3)  # Wrong size
        with self.assertRaises(ValueError):
            self.optimizer.optimize(wrong_returns, self.cov_matrix, self.goals)
        
        # Non-symmetric covariance
        bad_cov = np.random.randn(self.n_assets, self.n_assets)
        with self.assertRaises(ValueError):
            self.optimizer.optimize(self.returns, bad_cov, self.goals)
    
    def test_optimization(self):
        """Test basic optimization"""
        result = self.optimizer.optimize(
            self.returns,
            self.cov_matrix,
            self.goals
        )
        
        # Check weights sum to 1
        self.assertAlmostEqual(np.sum(result.weights), 1.0, places=5)
        
        # Check all weights are non-negative
        self.assertTrue(np.all(result.weights >= 0))
        
        # Check metrics are reasonable
        self.assertGreater(result.expected_return, 0)
        self.assertGreater(result.risk, 0)
        self.assertLess(result.max_drawdown, 1)
    
    def test_regime_detection(self):
        """Test market regime detection"""
        # Bull market
        regime = self.optimizer.detect_regime({
            'returns': 0.12,
            'volatility': 0.10,
            'vix': 15
        })
        self.assertEqual(regime, 'bull')
        
        # Bear market
        regime = self.optimizer.detect_regime({
            'returns': -0.08,
            'volatility': 0.30,
            'vix': 35
        })
        self.assertEqual(regime, 'bear')
        
        # Crisis
        regime = self.optimizer.detect_regime({
            'returns': -0.15,
            'volatility': 0.45,
            'vix': 50
        })
        self.assertEqual(regime, 'crisis')
    
    def test_constraints(self):
        """Test optimization with constraints"""
        result = self.optimizer.optimize(
            self.returns,
            self.cov_matrix,
            self.goals,
            constraints={'max_risk': 0.20, 'max_position': 0.40}
        )
        
        # Check max position constraint
        self.assertLessEqual(np.max(result.weights), 0.40)
        
        # Check risk constraint
        self.assertLessEqual(result.risk, 0.20)
    
    def test_tax_efficiency(self):
        """Test tax-loss harvesting calculation"""
        current_weights = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
        
        result = self.optimizer.optimize(
            self.returns,
            self.cov_matrix,
            self.goals,
            current_weights=current_weights
        )
        
        # Should have some tax alpha when rebalancing
        self.assertGreaterEqual(result.tax_alpha, 0)
        self.assertLessEqual(result.tax_alpha, 0.0125)  # Max 125 bps
    
    def test_portfolio_validation(self):
        """Test standalone validation function"""
        # Valid data
        self.assertTrue(
            validate_portfolio_data(self.returns, self.cov_matrix, self.n_assets)
        )
        
        # Invalid data
        with self.assertRaises(ValueError):
            validate_portfolio_data(
                np.ones(3),  # Wrong size
                self.cov_matrix,
                self.n_assets
            )
    
    def test_performance_summary(self):
        """Test performance summary generation"""
        # Run optimization first
        self.optimizer.optimize(self.returns, self.cov_matrix, self.goals)
        
        summary = self.optimizer.get_performance_summary()
        
        self.assertIn('optimizations_performed', summary)
        self.assertEqual(summary['optimizations_performed'], 1)
        self.assertIn('statistics', summary)


if __name__ == '__main__':
    unittest.main()