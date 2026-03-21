"""
Black-Litterman - Black-Litterman portfolio optimization model
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from agentic_trading_system.utils.logger import logger as logging

class BlackLitterman:
    """
    Black-Litterman Model - Combines market equilibrium with investor views
    
    Features:
    - Market equilibrium returns as prior
    - Investor views as conditional distributions
    - Posterior returns combining both
    - Confidence levels for views
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.risk_aversion = config.get("risk_aversion", 2.5)  # Lambda
        self.tau = config.get("tau", 0.05)  # Uncertainty scaling
        self.risk_free_rate = config.get("risk_free_rate", 0.02)
        
        # Weight constraints
        self.max_weight = config.get("max_weight", 0.25)
        self.min_weight = config.get("min_weight", 0.01)
        
        logging.info(f"✅ BlackLitterman initialized")
    
    def calculate_implied_returns(self, market_caps: np.ndarray, 
                                  covariance: np.ndarray) -> np.ndarray:
        """
        Calculate implied equilibrium returns
        Π = λ * Σ * w_mkt
        """
        # Market weights from market caps
        market_weights = market_caps / np.sum(market_caps)
        
        # Implied returns
        implied_returns = self.risk_aversion * np.dot(covariance, market_weights)
        
        return implied_returns
    
    def build_view_matrix(self, views: List[Dict], assets: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build P (view matrix) and Q (view returns) from investor views
        
        Views format:
        {
            "assets": ["AAPL", "MSFT"],
            "type": "absolute",  # or "relative"
            "return": 0.15,  # Expected return
            "confidence": 0.7  # Confidence level (0-1)
        }
        """
        n_assets = len(assets)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        Omega = np.zeros((n_views, n_views))
        
        for i, view in enumerate(views):
            view_assets = view.get("assets", [])
            view_type = view.get("type", "absolute")
            view_return = view.get("return", 0)
            confidence = view.get("confidence", 0.5)
            
            if view_type == "absolute":
                # Absolute view: single asset return
                if len(view_assets) == 1:
                    asset_idx = assets.index(view_assets[0])
                    P[i, asset_idx] = 1
                    Q[i] = view_return
                    
            elif view_type == "relative":
                # Relative view: portfolio of assets
                if len(view_assets) >= 2:
                    # Assume equal weighting for relative views
                    weight = 1.0 / len(view_assets)
                    for asset in view_assets:
                        asset_idx = assets.index(asset)
                        P[i, asset_idx] = weight
                    Q[i] = view_return
            
            # Uncertainty (inverse of confidence)
            uncertainty = (1 - confidence) * abs(view_return) if view_return != 0 else 0.01
            Omega[i, i] = uncertainty ** 2
        
        return P, Q, Omega
    
    def calculate_posterior_returns(self, prior_returns: np.ndarray,
                                   covariance: np.ndarray,
                                   P: np.ndarray, Q: np.ndarray,
                                   Omega: np.ndarray) -> np.ndarray:
        """
        Calculate posterior returns using Black-Litterman formula
        
        E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1Π + P'Ω^-1Q]
        """
        n_assets = len(prior_returns)
        
        # Prior precision
        prior_precision = np.linalg.inv(self.tau * covariance)
        
        # View precision
        view_precision = np.dot(np.dot(P.T, np.linalg.inv(Omega)), P)
        
        # Posterior precision
        posterior_precision = prior_precision + view_precision
        
        # Posterior returns
        prior_term = np.dot(prior_precision, prior_returns)
        view_term = np.dot(np.dot(P.T, np.linalg.inv(Omega)), Q)
        
        posterior_returns = np.dot(np.linalg.inv(posterior_precision), prior_term + view_term)
        
        return posterior_returns
    
    def optimize(self, prices: pd.DataFrame, market_caps: Dict[str, float],
                views: List[Dict]) -> Dict[str, Any]:
        """
        Perform Black-Litterman optimization
        """
        # Calculate returns and statistics
        returns = prices.pct_change().dropna()
        expected_returns_hist = returns.mean() * 252
        covariance = returns.cov() * 252
        
        assets = list(prices.columns)
        
        # Create market cap array
        market_cap_array = np.array([market_caps.get(asset, 1e9) for asset in assets])
        
        # Calculate implied returns
        implied_returns = self.calculate_implied_returns(market_cap_array, covariance.values)
        
        # Build view matrices
        P, Q, Omega = self.build_view_matrix(views, assets)
        
        # Calculate posterior returns
        posterior_returns = self.calculate_posterior_returns(
            implied_returns, covariance.values, P, Q, Omega
        )
        
        # Optimize portfolio using posterior returns
        optimal_weights = self._optimize_weights(posterior_returns, covariance.values, assets)
        
        # Calculate portfolio metrics
        weights_array = np.array([optimal_weights[a] for a in assets])
        portfolio_return = np.sum(posterior_returns * weights_array)
        portfolio_volatility = np.sqrt(np.dot(weights_array.T, np.dot(covariance.values, weights_array)))
        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
        
        return {
            "weights": optimal_weights,
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe),
            "implied_returns": dict(zip(assets, implied_returns)),
            "posterior_returns": dict(zip(assets, posterior_returns)),
            "views_used": views,
            "assets": assets,
            "method": "black_litterman"
        }
    
    def _optimize_weights(self, expected_returns: np.ndarray, 
                         covariance: np.ndarray,
                         assets: List[str]) -> Dict[str, float]:
        """
        Optimize portfolio weights given expected returns
        """
        n_assets = len(assets)
        
        # Objective: maximize Sharpe ratio
        def negative_sharpe(weights):
            portfolio_return = np.sum(expected_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
            if portfolio_volatility == 0:
                return 0
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
            return -sharpe
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        
        # Bounds
        bounds = Bounds(self.min_weight, self.max_weight)
        
        # Initial guess
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            negative_sharpe,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
        else:
            # Fallback to equal weights
            weights = initial_weights
        
        return dict(zip(assets, weights))