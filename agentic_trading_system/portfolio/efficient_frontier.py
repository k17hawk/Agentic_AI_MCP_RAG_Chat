"""
Efficient Frontier - Markowitz Modern Portfolio Theory
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from agentic_trading_system.utils.logger import logger as logging

class EfficientFrontier:
    """
    Efficient Frontier - Implements Markowitz Modern Portfolio Theory
    
    Finds optimal portfolios that maximize return for given risk level
    or minimize risk for given return target.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.risk_free_rate = config.get("risk_free_rate", 0.02)  # 2%
        self.max_weight = config.get("max_weight", 0.25)  # Max 25% in single asset
        self.min_weight = config.get("min_weight", 0.01)  # Min 1% in single asset
        
        # Optimization parameters
        self.max_iterations = config.get("max_iterations", 1000)
        self.tolerance = config.get("tolerance", 1e-6)
        
        logging.info(f"✅ EfficientFrontier initialized")
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate returns from price data
        """
        returns = prices.pct_change().dropna()
        return returns
    
    def calculate_statistics(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate return statistics
        """
        # Expected returns (annualized)
        expected_returns = returns.mean() * 252
        
        # Covariance matrix (annualized)
        covariance = returns.cov() * 252
        
        # Volatility
        volatility = np.sqrt(np.diag(covariance))
        
        # Correlation matrix
        correlation = returns.corr()
        
        return {
            "expected_returns": expected_returns,
            "covariance": covariance,
            "volatility": volatility,
            "correlation": correlation,
            "assets": list(returns.columns)
        }
    
    def optimize_max_sharpe(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Find portfolio that maximizes Sharpe ratio
        """
        stats = self.calculate_statistics(returns)
        expected_returns = stats["expected_returns"]
        covariance = stats["covariance"]
        n_assets = len(expected_returns)
        
        # Objective function: negative Sharpe ratio
        def negative_sharpe(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds
        bounds = Bounds(self.min_weight, self.max_weight)
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if not result.success:
            logging.warning(f"Optimization failed: {result.message}")
            return self._get_equal_weights(expected_returns, covariance)
        
        weights = result.x
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            "weights": dict(zip(expected_returns.index, weights)),
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe),
            "method": "max_sharpe",
            "risk_free_rate": self.risk_free_rate,
            "assets": list(expected_returns.index)
        }
    
    def optimize_min_volatility(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Find portfolio with minimum volatility
        """
        stats = self.calculate_statistics(returns)
        expected_returns = stats["expected_returns"]
        covariance = stats["covariance"]
        n_assets = len(expected_returns)
        
        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(covariance, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        
        # Bounds
        bounds = Bounds(self.min_weight, self.max_weight)
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if not result.success:
            return self._get_equal_weights(expected_returns, covariance)
        
        weights = result.x
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(portfolio_variance(weights))
        
        return {
            "weights": dict(zip(expected_returns.index, weights)),
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float((portfolio_return - self.risk_free_rate) / portfolio_volatility),
            "method": "min_volatility",
            "assets": list(expected_returns.index)
        }
    
    def optimize_for_return(self, returns: pd.DataFrame, target_return: float) -> Dict[str, Any]:
        """
        Find portfolio that minimizes volatility for a target return
        """
        stats = self.calculate_statistics(returns)
        expected_returns = stats["expected_returns"]
        covariance = stats["covariance"]
        n_assets = len(expected_returns)
        
        # Objective function: portfolio variance
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(covariance, weights))
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.sum(expected_returns * x) - target_return}
        ]
        
        # Bounds
        bounds = Bounds(self.min_weight, self.max_weight)
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if not result.success:
            return self._get_equal_weights(expected_returns, covariance)
        
        weights = result.x
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(portfolio_variance(weights))
        
        return {
            "weights": dict(zip(expected_returns.index, weights)),
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "target_return": target_return,
            "sharpe_ratio": float((portfolio_return - self.risk_free_rate) / portfolio_volatility),
            "method": "target_return",
            "assets": list(expected_returns.index)
        }
    
    def generate_efficient_frontier(self, returns: pd.DataFrame, points: int = 50) -> Dict[str, Any]:
        """
        Generate the efficient frontier
        """
        stats = self.calculate_statistics(returns)
        expected_returns = stats["expected_returns"]
        
        # Get min and max return
        min_vol_portfolio = self.optimize_min_volatility(returns)
        max_return_portfolio = self._get_max_return_portfolio(returns)
        
        min_return = min_vol_portfolio["expected_return"]
        max_return = max_return_portfolio["expected_return"]
        
        # Generate points along frontier
        target_returns = np.linspace(min_return, max_return, points)
        frontier_points = []
        
        for target in target_returns:
            try:
                portfolio = self.optimize_for_return(returns, target)
                frontier_points.append({
                    "return": portfolio["expected_return"],
                    "volatility": portfolio["volatility"],
                    "sharpe": portfolio["sharpe_ratio"],
                    "weights": portfolio["weights"]
                })
            except:
                continue
        
        # Find max Sharpe portfolio
        max_sharpe_portfolio = self.optimize_max_sharpe(returns)
        
        return {
            "frontier": frontier_points,
            "min_volatility": min_vol_portfolio,
            "max_sharpe": max_sharpe_portfolio,
            "max_return": max_return_portfolio,
            "assets": list(expected_returns.index)
        }
    
    def _get_max_return_portfolio(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Get portfolio with maximum return"""
        stats = self.calculate_statistics(returns)
        expected_returns = stats["expected_returns"]
        covariance = stats["covariance"]
        
        # Find asset with highest return
        max_return_asset = expected_returns.idxmax()
        weights = {asset: 0.0 for asset in expected_returns.index}
        weights[max_return_asset] = 1.0
        
        weights_array = np.array([weights[a] for a in expected_returns.index])
        portfolio_return = np.sum(expected_returns * weights_array)
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(covariance, weights_array)))
        
        return {
            "weights": weights,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float((portfolio_return - self.risk_free_rate) / portfolio_volatility),
            "method": "max_return",
            "assets": list(expected_returns.index)
        }
    
    def _get_equal_weights(self, expected_returns: pd.Series, covariance: pd.DataFrame) -> Dict[str, Any]:
        """Get equal-weighted portfolio as fallback"""
        n_assets = len(expected_returns)
        weights = {asset: 1/n_assets for asset in expected_returns.index}
        weights_array = np.array([1/n_assets] * n_assets)
        
        portfolio_return = np.sum(expected_returns * weights_array)
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(covariance, weights_array)))
        
        return {
            "weights": weights,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float((portfolio_return - self.risk_free_rate) / portfolio_volatility),
            "method": "equal_weight",
            "assets": list(expected_returns.index)
        }