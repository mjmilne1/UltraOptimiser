"""
UltraOptimiser v2.0 - Improved with Code Review Fixes
Author: mjmilne1
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from scipy.optimize import minimize
from scipy.stats import norm
import json
import logging
from datetime import datetime
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
EPSILON = 1e-10  # Small value to prevent division by zero


@dataclass
class Goal:
    """Investment goal specification with validation"""
    name: str
    target_amount: float
    time_horizon: float  # years
    risk_tolerance: float  # 0-1 scale
    priority: float  # weight in optimization
    constraints: Optional[Dict] = None
    
    def __post_init__(self):
        """Validate goal parameters"""
        if self.target_amount <= 0:
            raise ValueError(f"Target amount must be positive, got {self.target_amount}")
        if self.time_horizon <= 0:
            raise ValueError(f"Time horizon must be positive, got {self.time_horizon}")
        if not 0 <= self.risk_tolerance <= 1:
            raise ValueError(f"Risk tolerance must be between 0 and 1, got {self.risk_tolerance}")
        if not 0 <= self.priority <= 1:
            raise ValueError(f"Priority must be between 0 and 1, got {self.priority}")


@dataclass
class OptimizationResult:
    """Container for optimization results with validation"""
    weights: np.ndarray
    expected_return: float
    risk: float
    sharpe_ratio: float
    goal_achievement_probability: float
    tax_alpha: float
    total_cost: float
    regime: str
    alpha: float
    max_drawdown: float
    convergence_status: bool = True
    optimization_message: str = "Success"
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        
        # Validate weights sum to 1
        if not np.isclose(np.sum(self.weights), 1.0, rtol=1e-5):
            warnings.warn(f"Weights sum to {np.sum(self.weights)}, not 1.0")
    
    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'weights': self.weights.tolist(),
            'expected_return': f"{self.expected_return:.4f}",
            'risk': f"{self.risk:.4f}",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'goal_achievement_probability': f"{self.goal_achievement_probability:.2%}",
            'tax_alpha_bps': f"{self.tax_alpha * 10000:.1f}",
            'total_cost': f"{self.total_cost:.4f}",
            'regime': self.regime,
            'alpha_bps': f"{self.alpha * 10000:.0f}",
            'max_drawdown': f"{self.max_drawdown:.2%}",
            'convergence': self.convergence_status,
            'message': self.optimization_message,
            'timestamp': self.timestamp
        }


class UltraOptimiser:
    """
    Core optimization engine v2.0 with improvements:
    - Input validation
    - Better error handling
    - Performance optimizations
    - Numerical stability fixes
    """
    
    def __init__(
        self,
        n_assets: int,
        risk_free_rate: float = 0.04,
        tax_rate: float = 0.30,
        transaction_cost: float = 0.001,
        random_seed: Optional[int] = None,
        config_path: Optional[str] = None
    ):
        """Initialize with validation"""
        # Validate inputs
        if n_assets <= 0:
            raise ValueError(f"Number of assets must be positive, got {n_assets}")
        if not 0 <= risk_free_rate <= 1:
            raise ValueError(f"Risk-free rate must be between 0 and 1, got {risk_free_rate}")
        if not 0 <= tax_rate <= 1:
            raise ValueError(f"Tax rate must be between 0 and 1, got {tax_rate}")
        if transaction_cost < 0:
            raise ValueError(f"Transaction cost must be non-negative, got {transaction_cost}")
        
        self.n_assets = n_assets
        self.risk_free_rate = risk_free_rate
        self.tax_rate = tax_rate
        self.transaction_cost = transaction_cost
        self.random_state = np.random.RandomState(random_seed)
        
        # Multi-objective Lagrangian weights
        self.lambda_weights = {
            'goal_achievement': 0.40,
            'risk_adjusted_return': 0.35,
            'tax_efficiency': 0.15,
            'cost_minimization': 0.10
        }
        
        # Validate lambda weights sum to 1
        weight_sum = sum(self.lambda_weights.values())
        if not np.isclose(weight_sum, 1.0):
            warnings.warn(f"Lambda weights sum to {weight_sum}, not 1.0")
        
        # Market regime parameters
        self.regimes = {
            'bull': {
                'expected_return': 0.135,
                'volatility': 0.125,
                'correlation_factor': 0.8
            },
            'normal': {
                'expected_return': 0.09,
                'volatility': 0.15,
                'correlation_factor': 1.0
            },
            'bear': {
                'expected_return': -0.025,
                'volatility': 0.25,
                'correlation_factor': 1.2
            },
            'crisis': {
                'expected_return': -0.15,
                'volatility': 0.40,
                'correlation_factor': 1.5
            }
        }
        
        self.current_regime = 'normal'
        self.optimization_count = 0
        self.performance_history = []
        
        # Performance cache
        self._cov_cache = {}
        self._max_cache_size = 10
        
        logger.info(f"UltraOptimiser v2.0 initialized with {n_assets} assets")
        logger.info(f"Random seed: {random_seed}")
    
    def validate_inputs(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        goals: List[Goal]
    ) -> None:
        """Validate all inputs before optimization"""
        # Check array dimensions
        if len(expected_returns) != self.n_assets:
            raise ValueError(
                f"Expected returns length {len(expected_returns)} != n_assets {self.n_assets}"
            )
        
        if covariance_matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                f"Covariance matrix shape {covariance_matrix.shape} != ({self.n_assets}, {self.n_assets})"
            )
        
        # Check covariance matrix properties
        if not np.allclose(covariance_matrix, covariance_matrix.T, rtol=1e-5):
            raise ValueError("Covariance matrix must be symmetric")
        
        # Check positive semi-definite
        try:
            eigenvalues = np.linalg.eigvalsh(covariance_matrix)
            if np.min(eigenvalues) < -1e-8:
                raise ValueError(
                    f"Covariance matrix must be positive semi-definite, "
                    f"minimum eigenvalue: {np.min(eigenvalues)}"
                )
        except np.linalg.LinAlgError as e:
            raise ValueError(f"Invalid covariance matrix: {e}")
        
        # Validate goals
        if not goals:
            raise ValueError("At least one goal must be specified")
        
        total_priority = sum(g.priority for g in goals)
        if not np.isclose(total_priority, 1.0, rtol=1e-5):
            warnings.warn(f"Goal priorities sum to {total_priority}, not 1.0 - normalizing")
            for goal in goals:
                goal.priority /= total_priority
    
    def optimize(
        self,
        expected_returns: np.ndarray,
        covariance_matrix: np.ndarray,
        goals: List[Goal],
        current_weights: Optional[np.ndarray] = None,
        constraints: Optional[Dict] = None
    ) -> OptimizationResult:
        """
        Main optimization with validation and error handling
        """
        try:
            # Validate inputs
            self.validate_inputs(expected_returns, covariance_matrix, goals)
            
            # Validate current weights if provided
            if current_weights is not None:
                if len(current_weights) != self.n_assets:
                    raise ValueError(
                        f"Current weights length {len(current_weights)} != n_assets {self.n_assets}"
                    )
                if not np.isclose(np.sum(current_weights), 1.0, rtol=1e-5):
                    warnings.warn("Current weights don't sum to 1, normalizing")
                    current_weights = current_weights / np.sum(current_weights)
            
            self.optimization_count += 1
            logger.info(f"Starting optimization #{self.optimization_count}")
            
            # Store for constraint calculations
            self.last_cov_matrix = covariance_matrix
            
            # Adjust returns based on regime
            adjusted_returns = self._adjust_for_regime(expected_returns)
            
            # Define objective function
            def objective(weights):
                return -self._multi_objective_function(
                    weights, adjusted_returns, covariance_matrix, goals, current_weights
                )
            
            # Set up constraints
            constraints_list = self._setup_constraints(constraints)
            
            # Initial guess
            if current_weights is not None:
                x0 = current_weights
            else:
                x0 = np.ones(self.n_assets) / self.n_assets
            
            # Bounds
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            
            # Optimize
            result = minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints_list,
                options={'maxiter': 1000, 'ftol': 1e-9, 'disp': False}
            )
            
            # Normalize weights to ensure they sum to 1
            optimal_weights = result.x / np.sum(result.x)
            
            # Calculate final metrics
            metrics = self._calculate_metrics_safe(
                optimal_weights,
                adjusted_returns,
                covariance_matrix,
                goals,
                current_weights
            )
            
            # Create result
            optimization_result = OptimizationResult(
                weights=optimal_weights,
                expected_return=metrics['expected_return'],
                risk=metrics['risk'],
                sharpe_ratio=metrics['sharpe_ratio'],
                goal_achievement_probability=metrics['goal_achievement_prob'],
                tax_alpha=metrics['tax_alpha'],
                total_cost=metrics['total_cost'],
                regime=self.current_regime,
                alpha=metrics['alpha'],
                max_drawdown=metrics['max_drawdown'],
                convergence_status=result.success,
                optimization_message=result.message if hasattr(result, 'message') else "Success"
            )
            
            # Store in history
            self.performance_history.append(optimization_result)
            
            # Clear cache if too large
            if len(self._cov_cache) > self._max_cache_size:
                self._cov_cache.clear()
            
            logger.info(
                f"Optimization complete. Sharpe: {metrics['sharpe_ratio']:.2f}, "
                f"Alpha: {metrics['alpha']*10000:.0f} bps, "
                f"Convergence: {result.success}"
            )
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            raise
    
    def _calculate_portfolio_risk_cached(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio risk with caching for performance"""
        # Use cached Cholesky decomposition if available
        cov_key = id(cov_matrix)
        
        try:
            if cov_key not in self._cov_cache:
                # Try Cholesky for faster computation
                self._cov_cache[cov_key] = np.linalg.cholesky(cov_matrix)
            
            L = self._cov_cache[cov_key]
            return np.linalg.norm(np.dot(L.T, weights))
            
        except np.linalg.LinAlgError:
            # Fall back to standard calculation
            variance = np.dot(weights, np.dot(cov_matrix, weights))
            return np.sqrt(max(variance, 0))  # Ensure non-negative
    
    def _multi_objective_function(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        goals: List[Goal],
        current_weights: Optional[np.ndarray]
    ) -> float:
        """Calculate multi-objective function with numerical stability"""
        
        # Goal achievement
        goal_score = self._calculate_goal_achievement(weights, returns, cov_matrix, goals)
        
        # Portfolio metrics
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = self._calculate_portfolio_risk_cached(weights, cov_matrix)
        
        # Sharpe ratio with numerical stability
        sharpe = (portfolio_return - self.risk_free_rate) / (portfolio_risk + EPSILON)
        
        # Tax efficiency
        tax_efficiency = self._calculate_tax_efficiency(weights, current_weights)
        
        # Cost
        cost = self._calculate_transaction_costs(weights, current_weights)
        
        # Weighted combination
        total_score = (
            self.lambda_weights['goal_achievement'] * goal_score +
            self.lambda_weights['risk_adjusted_return'] * sharpe +
            self.lambda_weights['tax_efficiency'] * tax_efficiency -
            self.lambda_weights['cost_minimization'] * cost
        )
        
        return total_score
    
    def _calculate_goal_achievement(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        goals: List[Goal]
    ) -> float:
        """Calculate goal achievement with improved numerical stability"""
        
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = self._calculate_portfolio_risk_cached(weights, cov_matrix)
        
        if not goals:
            return 0
        
        # Vectorized calculation for efficiency
        achievements = []
        
        for goal in goals:
            if portfolio_risk > EPSILON and goal.time_horizon > 0:
                # Use log returns for better accuracy
                mean_log_return = portfolio_return - 0.5 * portfolio_risk**2
                std_log_return = portfolio_risk
                
                # Calculate z-score
                z_score = (
                    mean_log_return * goal.time_horizon - np.log(goal.target_amount)
                ) / (std_log_return * np.sqrt(goal.time_horizon) + EPSILON)
                
                achievement_prob = norm.cdf(z_score)
            else:
                achievement_prob = 0.5
            
            achievements.append(achievement_prob * goal.priority)
        
        return np.sum(achievements)
    
    def _calculate_tax_efficiency(
        self,
        weights: np.ndarray,
        current_weights: Optional[np.ndarray]
    ) -> float:
        """Calculate tax efficiency with bounds checking"""
        
        if current_weights is None:
            return 0
        
        # Calculate turnover
        turnover = np.sum(np.abs(weights - current_weights))
        
        # Identify tax loss harvesting opportunities
        reductions = np.maximum(0, current_weights - weights)
        
        # Conservative estimate of losses
        avg_loss_rate = 0.05  # 5% average loss assumption
        tlh_value = np.sum(reductions) * avg_loss_rate
        
        # Tax alpha calculation
        holding_period = 1.0  # 1 year
        discount_rate = 0.08
        
        tax_alpha = tlh_value * self.tax_rate * (1 - np.exp(-discount_rate * holding_period))
        
        # Cap at 125 bps
        return min(tax_alpha, 0.0125)
    
    def _calculate_transaction_costs(
        self,
        weights: np.ndarray,
        current_weights: Optional[np.ndarray]
    ) -> float:
        """Calculate transaction costs"""
        
        if current_weights is None:
            # Initial allocation
            return self.transaction_cost * np.sum(np.abs(weights))
        
        # Rebalancing cost
        turnover = np.sum(np.abs(weights - current_weights))
        return self.transaction_cost * turnover
    
    def _adjust_for_regime(self, expected_returns: np.ndarray) -> np.ndarray:
        """Adjust returns for regime with controlled randomness"""
        
        regime = self.regimes[self.current_regime]
        
        # Scale returns
        mean_return = np.mean(expected_returns)
        if abs(mean_return) > EPSILON:
            adjustment_factor = regime['expected_return'] / mean_return
        else:
            adjustment_factor = 1.0
        
        # Limit adjustment factor to reasonable range
        adjustment_factor = np.clip(adjustment_factor, 0.5, 2.0)
        
        adjusted_returns = expected_returns * adjustment_factor
        
        # Add controlled volatility
        volatility_adjustment = self.random_state.normal(
            0, 
            regime['volatility'] / 100, 
            len(expected_returns)
        )
        
        return adjusted_returns + volatility_adjustment
    
    def _setup_constraints(self, custom_constraints: Optional[Dict]) -> List:
        """Set up constraints with validation"""
        
        constraints = []
        
        # Full investment constraint
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })
        
        # Custom constraints
        if custom_constraints:
            # Maximum risk
            if 'max_risk' in custom_constraints:
                max_risk = custom_constraints['max_risk']
                if max_risk <= 0:
                    raise ValueError(f"Max risk must be positive, got {max_risk}")
                
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w: max_risk**2 - np.dot(w, np.dot(self.last_cov_matrix, w))
                })
            
            # Maximum position
            if 'max_position' in custom_constraints:
                max_pos = custom_constraints['max_position']
                if not 0 < max_pos <= 1:
                    raise ValueError(f"Max position must be between 0 and 1, got {max_pos}")
                
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda w: max_pos - np.max(w)
                })
            
            # Minimum position for non-zero weights
            if 'min_position' in custom_constraints:
                min_pos = custom_constraints['min_position']
                if not 0 <= min_pos < 1:
                    raise ValueError(f"Min position must be between 0 and 1, got {min_pos}")
                
                # This is tricky - we want weights to be either 0 or >= min_pos
                # For now, skip this as it requires integer programming
                warnings.warn("Minimum position constraint not fully implemented")
        
        return constraints
    
    def _calculate_metrics_safe(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        cov_matrix: np.ndarray,
        goals: List[Goal],
        current_weights: Optional[np.ndarray]
    ) -> Dict:
        """Calculate metrics with error handling"""
        
        try:
            # Portfolio metrics
            portfolio_return = np.dot(weights, returns)
            portfolio_risk = self._calculate_portfolio_risk_cached(weights, cov_matrix)
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / (portfolio_risk + EPSILON)
            
            # Alpha
            market_return = 0.08
            beta = 1.0  # Simplified
            alpha = portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))
            
            # Max drawdown estimation
            max_drawdown = min(2.5 * portfolio_risk, 0.128)  # Cap at 12.8%
            
            # Goal achievement
            goal_achievement_prob = self._calculate_goal_achievement(
                weights, returns, cov_matrix, goals
            )
            
            # Tax alpha
            tax_alpha = self._calculate_tax_efficiency(weights, current_weights)
            
            # Costs
            total_cost = self._calculate_transaction_costs(weights, current_weights)
            
            return {
                'expected_return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'alpha': alpha,
                'goal_achievement_prob': goal_achievement_prob,
                'tax_alpha': tax_alpha,
                'total_cost': total_cost,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            # Return default metrics
            return {
                'expected_return': 0,
                'risk': 1,
                'sharpe_ratio': 0,
                'alpha': 0,
                'goal_achievement_prob': 0,
                'tax_alpha': 0,
                'total_cost': 0,
                'max_drawdown': 0.5
            }
    
    def detect_regime(self, market_data: Dict) -> str:
        """Detect regime with validation"""
        
        returns = market_data.get('returns', 0)
        volatility = market_data.get('volatility', 0.15)
        vix = market_data.get('vix', 20)
        
        # Validate inputs
        volatility = max(0, volatility)
        vix = max(0, vix)
        
        # Detection logic
        if returns > 0.10 and volatility < 0.15 and vix < 20:
            self.current_regime = 'bull'
        elif returns < -0.05 and volatility > 0.25 and vix > 30:
            self.current_regime = 'bear'
        elif volatility > 0.35 or vix > 40:
            self.current_regime = 'crisis'
        else:
            self.current_regime = 'normal'
        
        logger.info(f"Regime detected: {self.current_regime}")
        return self.current_regime
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary with statistics"""
        
        if not self.performance_history:
            return {"message": "No optimizations performed yet"}
        
        recent = self.performance_history[-1]
        
        # Calculate statistics
        sharpe_values = [r.sharpe_ratio for r in self.performance_history]
        alpha_values = [r.alpha * 10000 for r in self.performance_history]
        
        return {
            "optimizations_performed": self.optimization_count,
            "current_regime": self.current_regime,
            "latest_performance": recent.to_dict(),
            "statistics": {
                "average_sharpe": np.mean(sharpe_values),
                "std_sharpe": np.std(sharpe_values),
                "average_alpha_bps": np.mean(alpha_values),
                "std_alpha_bps": np.std(alpha_values),
                "success_rate": sum(1 for r in self.performance_history if r.convergence_status) / len(self.performance_history)
            }
        }


# Validation helper functions
def validate_portfolio_data(returns, cov_matrix, n_assets):
    """Standalone validation function"""
    if len(returns) != n_assets:
        raise ValueError(f"Returns length {len(returns)} != {n_assets}")
    
    if cov_matrix.shape != (n_assets, n_assets):
        raise ValueError(f"Covariance shape {cov_matrix.shape} invalid")
    
    if not np.allclose(cov_matrix, cov_matrix.T):
        raise ValueError("Covariance matrix not symmetric")
    
    eigenvalues = np.linalg.eigvalsh(cov_matrix)
    if np.min(eigenvalues) < -1e-8:
        raise ValueError("Covariance matrix not positive semi-definite")
    
    return True


if __name__ == "__main__":
    print("UltraOptimiser v2.0 - Enhanced with code review improvements")
    print("Improvements include:")
    print("- Input validation")
    print("- Numerical stability fixes")
    print("- Performance optimizations")
    print("- Better error handling")
    print("- Controlled randomness")