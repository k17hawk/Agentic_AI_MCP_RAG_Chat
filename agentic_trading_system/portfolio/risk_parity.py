"""
Risk Parity - Risk parity portfolio optimization
"""
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from utils.logger import logger as logging

class RiskParity:
    """
    Risk Parity Portfolio - Allocates risk equally across assets
    
    Each asset contributes equally to total portfolio risk
    rather than equal capital allocation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.target_risk_contribution = config.get("target_risk_contribution", None)  # None = equal
        self.max_weight = config.get("max_weight", 0.30)
        self.min_weight = config.get("min_weight", 0.02)
        self.tolerance = config.get("tolerance", 1e-6)
        
        logging.info(f"✅ RiskParity initialized")
    
    def calculate_risk_contributions(self, weights: np.ndarray, 
                                     covariance: np.ndarray) -> np.ndarray:
        """
        Calculate risk contribution of each asset
        RC_i = w_i * (Σw)_i / √(w'Σw)
        """
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        
        if portfolio_volatility == 0:
            return np.zeros_like(weights)
        
        # Marginal risk contribution
        marginal_contrib = np.dot(covariance, weights) / portfolio_volatility
        
        # Component risk contribution
        risk_contrib = weights * marginal_contrib
        
        return risk_contrib
    
    def risk_parity_objective(self, weights: np.ndarray, covariance: np.ndarray,
                             target_contrib: np.ndarray) -> float:
        """
        Objective function for risk parity
        Minimize sum of squared differences between actual and target risk contributions
        """
        risk_contrib = self.calculate_risk_contributions(weights, covariance)
        
        # Normalize risk contributions
        total_risk = np.sum(risk_contrib)
        if total_risk > 0:
            risk_contrib_normalized = risk_contrib / total_risk
        else:
            risk_contrib_normalized = risk_contrib
        
        # Squared differences
        diff = risk_contrib_normalized - target_contrib
        objective = np.sum(diff ** 2)
        
        return objective
    
    def optimize(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """
        Find risk parity portfolio weights
        """
        # Calculate covariance matrix
        covariance = returns.cov() * 252
        n_assets = len(returns.columns)
        
        # Target risk contributions (equal by default)
        if self.target_risk_contribution is None:
            target_contrib = np.ones(n_assets) / n_assets
        else:
            target_contrib = np.array(self.target_risk_contribution)
            target_contrib = target_contrib / np.sum(target_contrib)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # weights sum to 1
        ]
        
        # Bounds
        bounds = [(self.min_weight, self.max_weight) for _ in range(n_assets)]
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        # Optimize
        result = minimize(
            self.risk_parity_objective,
            initial_weights,
            args=(covariance.values, target_contrib),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': self.tolerance}
        )
        
        if not result.success:
            logging.warning(f"Risk parity optimization failed: {result.message}")
            # Fallback to equal weights
            weights = initial_weights
        else:
            weights = result.x
        
        # Calculate final risk contributions
        risk_contrib = self.calculate_risk_contributions(weights, covariance.values)
        total_risk = np.sum(risk_contrib)
        risk_contrib_pct = risk_contrib / total_risk if total_risk > 0 else risk_contrib
        
        # Calculate portfolio metrics
        expected_returns = returns.mean() * 252
        portfolio_return = np.sum(expected_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance.values, weights)))
        
        # Calculate concentration metrics
        herfindahl = np.sum(weights ** 2)
        risk_concentration = np.sum(risk_contrib_pct ** 2)
        
        return {
            "weights": dict(zip(returns.columns, weights)),
            "risk_contributions": dict(zip(returns.columns, risk_contrib)),
            "risk_contributions_pct": dict(zip(returns.columns, risk_contrib_pct)),
            "expected_return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(portfolio_return / portfolio_volatility if portfolio_volatility > 0 else 0),
            "herfindahl_index": float(herfindahl),
            "risk_concentration": float(risk_concentration),
            "method": "risk_parity",
            "assets": list(returns.columns)
        }
    
    def calculate_diversification_ratio(self, weights: np.ndarray, 
                                       covariance: np.ndarray,
                                       volatility: np.ndarray) -> float:
        """
        Calculate diversification ratio
        DR = (w'σ) / √(w'Σw)
        """
        weighted_vol = np.sum(weights * volatility)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(covariance, weights)))
        
        if portfolio_vol == 0:
            return 1.0
        
        return weighted_vol / portfolio_vol