"""
VaR Calculator - Value at Risk calculation
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from scipy import stats
from utils.logger import logger as logging

class VaRCalculator:
    """
    Value at Risk (VaR) calculator
    
    Methods:
    - Historical VaR
    - Parametric VaR (variance-covariance)
    - Monte Carlo VaR
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.confidence_level = config.get("confidence_level", 0.95)
        self.time_horizon = config.get("time_horizon", 1)  # 1 day
        self.historical_periods = config.get("historical_periods", 252)  # 1 year
        
        logging.info(f"✅ VaRCalculator initialized")
    
    def historical_var(self, returns: List[float], portfolio_value: float,
                      confidence: float = None) -> Dict[str, Any]:
        """
        Calculate historical VaR
        """
        if confidence is None:
            confidence = self.confidence_level
        
        returns_array = np.array(returns)
        
        # Sort returns
        sorted_returns = np.sort(returns_array)
        
        # Find VaR at confidence level
        var_index = int((1 - confidence) * len(sorted_returns))
        var_return = sorted_returns[var_index]
        
        # Calculate VaR in currency
        var_amount = portfolio_value * abs(var_return)
        
        # Calculate expected shortfall (CVaR)
        cvar_returns = sorted_returns[:var_index]
        cvar_return = np.mean(cvar_returns) if len(cvar_returns) > 0 else var_return
        cvar_amount = portfolio_value * abs(cvar_return)
        
        return {
            "var_percent": float(abs(var_return) * 100),
            "var_amount": float(var_amount),
            "cvar_percent": float(abs(cvar_return) * 100),
            "cvar_amount": float(cvar_amount),
            "confidence_level": confidence,
            "method": "historical",
            "samples_used": len(returns_array)
        }
    
    def parametric_var(self, returns: List[float], portfolio_value: float,
                      confidence: float = None) -> Dict[str, Any]:
        """
        Calculate parametric VaR (assumes normal distribution)
        """
        if confidence is None:
            confidence = self.confidence_level
        
        returns_array = np.array(returns)
        
        # Calculate parameters
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        # VaR calculation
        var_return = mean_return + z_score * std_return
        var_amount = portfolio_value * abs(var_return)
        
        return {
            "var_percent": float(abs(var_return) * 100),
            "var_amount": float(var_amount),
            "mean_return": float(mean_return * 100),
            "volatility": float(std_return * 100),
            "z_score": float(z_score),
            "confidence_level": confidence,
            "method": "parametric"
        }
    
    def monte_carlo_var(self, returns: List[float], portfolio_value: float,
                       simulations: int = 10000, confidence: float = None) -> Dict[str, Any]:
        """
        Calculate VaR using Monte Carlo simulation
        """
        if confidence is None:
            confidence = self.confidence_level
        
        returns_array = np.array(returns)
        
        # Calculate parameters
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Run simulation
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(mean_return, std_return, simulations)
        
        # Find VaR
        simulated_returns.sort()
        var_index = int((1 - confidence) * simulations)
        var_return = simulated_returns[var_index]
        var_amount = portfolio_value * abs(var_return)
        
        return {
            "var_percent": float(abs(var_return) * 100),
            "var_amount": float(var_amount),
            "confidence_level": confidence,
            "method": "monte_carlo",
            "simulations": simulations
        }
    
    def calculate(self, returns: List[float], portfolio_value: float,
                 method: str = "historical") -> Dict[str, Any]:
        """
        Calculate VaR using specified method
        """
        if method == "historical":
            return self.historical_var(returns, portfolio_value)
        elif method == "parametric":
            return self.parametric_var(returns, portfolio_value)
        elif method == "monte_carlo":
            return self.monte_carlo_var(returns, portfolio_value)
        else:
            raise ValueError(f"Unknown method: {method}")