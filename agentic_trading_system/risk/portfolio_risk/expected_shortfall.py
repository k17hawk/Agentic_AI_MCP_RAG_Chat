"""
Expected Shortfall - Conditional Value at Risk (CVaR)
"""
from typing import Dict, Any, Optional, List
import numpy as np
from scipy import stats
from utils.logger import logger as logging

class ExpectedShortfall:
    """
    Expected Shortfall (CVaR) - Average loss beyond VaR
    
    More conservative than VaR as it considers tail risk
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.confidence_level = config.get("confidence_level", 0.95)
        self.historical_periods = config.get("historical_periods", 252)
        
        logging.info(f"✅ ExpectedShortfall initialized")
    
    def calculate(self, returns: List[float], portfolio_value: float,
                 confidence: float = None) -> Dict[str, Any]:
        """
        Calculate Expected Shortfall (CVaR)
        """
        if confidence is None:
            confidence = self.confidence_level
        
        returns_array = np.array(returns)
        sorted_returns = np.sort(returns_array)
        
        # Find VaR threshold
        var_index = int((1 - confidence) * len(sorted_returns))
        
        # Calculate Expected Shortfall (average of returns beyond VaR)
        tail_returns = sorted_returns[:var_index]
        
        if len(tail_returns) == 0:
            es_return = sorted_returns[0]
        else:
            es_return = np.mean(tail_returns)
        
        es_amount = portfolio_value * abs(es_return)
        
        # Calculate VaR for comparison
        var_return = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[-1]
        var_amount = portfolio_value * abs(var_return)
        
        return {
            "es_percent": float(abs(es_return) * 100),
            "es_amount": float(es_amount),
            "var_percent": float(abs(var_return) * 100),
            "var_amount": float(var_amount),
            "ratio_es_to_var": float(abs(es_return / var_return)) if var_return != 0 else 1,
            "confidence_level": confidence,
            "tail_samples": len(tail_returns),
            "total_samples": len(returns_array)
        }
    
    def parametric_es(self, returns: List[float], portfolio_value: float,
                     confidence: float = None) -> Dict[str, Any]:
        """
        Calculate parametric Expected Shortfall (assumes normal distribution)
        """
        if confidence is None:
            confidence = self.confidence_level
        
        returns_array = np.array(returns)
        
        # Calculate parameters
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - confidence)
        
        # ES formula for normal distribution
        # ES = -μ + σ * φ(Φ⁻¹(α)) / (1-α)
        from scipy import stats
        
        phi_z = stats.norm.pdf(z_score)  # PDF at z-score
        es_return = -mean_return + std_return * phi_z / (1 - confidence)
        
        es_amount = portfolio_value * abs(es_return)
        
        return {
            "es_percent": float(es_return * 100),
            "es_amount": float(es_amount),
            "mean_return": float(mean_return * 100),
            "volatility": float(std_return * 100),
            "confidence_level": confidence,
            "method": "parametric"
        }