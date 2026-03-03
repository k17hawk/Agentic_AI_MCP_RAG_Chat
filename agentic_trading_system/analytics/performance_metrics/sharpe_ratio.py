"""
Sharpe Ratio - Risk-adjusted return metric
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as logging

class SharpeRatio:
    """
    Sharpe Ratio - Measures risk-adjusted return
    
    Formula: (Rp - Rf) / σp
    where:
    - Rp = Portfolio return
    - Rf = Risk-free rate
    - σp = Standard deviation of portfolio returns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.risk_free_rate = config.get("risk_free_rate", 0.02)  # 2% annual
        self.trading_days = config.get("trading_days", 252)
        
        logging.info(f"✅ SharpeRatio initialized")
    
    def calculate(self, returns: List[float], 
                 periods_per_year: int = None) -> Dict[str, Any]:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: List of periodic returns (e.g., daily returns)
            periods_per_year: Number of periods in a year (e.g., 252 for daily)
        """
        if not returns or len(returns) < 2:
            return {
                "sharpe_ratio": 0.0,
                "annualized_sharpe": 0.0,
                "error": "Insufficient return data"
            }
        
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        returns_array = np.array(returns)
        
        # Calculate metrics
        avg_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return {
                "sharpe_ratio": 0.0,
                "annualized_sharpe": 0.0,
                "note": "Zero volatility"
            }
        
        # Periodic risk-free rate
        periodic_rf = self.risk_free_rate / periods_per_year
        
        # Calculate Sharpe
        sharpe = (avg_return - periodic_rf) / std_return
        
        # Annualize
        annualized_sharpe = sharpe * np.sqrt(periods_per_year)
        
        # Interpret result
        interpretation = self._interpret_sharpe(annualized_sharpe)
        
        return {
            "sharpe_ratio": float(sharpe),
            "annualized_sharpe": float(annualized_sharpe),
            "avg_return": float(avg_return),
            "std_return": float(std_return),
            "risk_free_rate": self.risk_free_rate,
            "periods_per_year": periods_per_year,
            "interpretation": interpretation,
            "num_periods": len(returns)
        }
    
    def calculate_from_equity(self, equity_curve: List[float]) -> Dict[str, Any]:
        """
        Calculate Sharpe ratio from equity curve
        """
        if len(equity_curve) < 2:
            return {"sharpe_ratio": 0.0, "error": "Insufficient data"}
        
        # Calculate returns from equity curve
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        return self.calculate(returns.tolist())
    
    def _interpret_sharpe(self, sharpe: float) -> str:
        """
        Provide interpretation of Sharpe ratio
        """
        if sharpe < 0:
            return "Poor (negative risk-adjusted return)"
        elif sharpe < 0.5:
            return "Below average"
        elif sharpe < 1.0:
            return "Average"
        elif sharpe < 2.0:
            return "Good"
        elif sharpe < 3.0:
            return "Very Good"
        else:
            return "Excellent"
    
    def compare(self, returns1: List[float], returns2: List[float],
               label1: str = "Strategy 1", label2: str = "Strategy 2") -> Dict[str, Any]:
        """
        Compare Sharpe ratios of two return streams
        """
        sharpe1 = self.calculate(returns1)
        sharpe2 = self.calculate(returns2)
        
        better = label1 if sharpe1['annualized_sharpe'] > sharpe2['annualized_sharpe'] else label2
        
        return {
            label1: sharpe1,
            label2: sharpe2,
            "better_performer": better,
            "difference": abs(sharpe1['annualized_sharpe'] - sharpe2['annualized_sharpe'])
        }