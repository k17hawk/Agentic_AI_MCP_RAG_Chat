"""
Sortino Ratio - Downside risk-adjusted return metric
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as logging

class SortinoRatio:
    """
    Sortino Ratio - Similar to Sharpe but only considers downside volatility
    
    Formula: (Rp - Rf) / σd
    where:
    - Rp = Portfolio return
    - Rf = Risk-free rate
    - σd = Downside deviation (standard deviation of negative returns)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.risk_free_rate = config.get("risk_free_rate", 0.02)  # 2% annual
        self.target_return = config.get("target_return", 0.0)  # Minimum acceptable return
        self.trading_days = config.get("trading_days", 252)
        
        logging.info(f"✅ SortinoRatio initialized")
    
    def calculate(self, returns: List[float], 
                 periods_per_year: int = None) -> Dict[str, Any]:
        """
        Calculate Sortino ratio
        
        Args:
            returns: List of periodic returns
            periods_per_year: Number of periods in a year
        """
        if not returns or len(returns) < 2:
            return {
                "sortino_ratio": 0.0,
                "annualized_sortino": 0.0,
                "error": "Insufficient return data"
            }
        
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        returns_array = np.array(returns)
        
        # Calculate metrics
        avg_return = np.mean(returns_array)
        
        # Calculate downside deviation (only negative returns below target)
        target = self.target_return / periods_per_year
        downside_returns = returns_array[returns_array < target]
        
        if len(downside_returns) == 0:
            downside_deviation = 0.0
        else:
            squared_deviations = (downside_returns - target) ** 2
            downside_deviation = np.sqrt(np.mean(squared_deviations))
        
        if downside_deviation == 0:
            return {
                "sortino_ratio": float('inf') if avg_return > target else 0.0,
                "annualized_sortino": float('inf') if avg_return > target else 0.0,
                "note": "No downside volatility"
            }
        
        # Periodic risk-free rate
        periodic_rf = self.risk_free_rate / periods_per_year
        
        # Calculate Sortino
        sortino = (avg_return - periodic_rf) / downside_deviation
        
        # Annualize
        annualized_sortino = sortino * np.sqrt(periods_per_year)
        
        # Interpret result
        interpretation = self._interpret_sortino(annualized_sortino)
        
        return {
            "sortino_ratio": float(sortino),
            "annualized_sortino": float(annualized_sortino),
            "avg_return": float(avg_return),
            "downside_deviation": float(downside_deviation),
            "risk_free_rate": self.risk_free_rate,
            "target_return": self.target_return,
            "periods_per_year": periods_per_year,
            "interpretation": interpretation,
            "negative_periods": len(downside_returns),
            "total_periods": len(returns)
        }
    
    def calculate_from_equity(self, equity_curve: List[float]) -> Dict[str, Any]:
        """
        Calculate Sortino ratio from equity curve
        """
        if len(equity_curve) < 2:
            return {"sortino_ratio": 0.0, "error": "Insufficient data"}
        
        # Calculate returns from equity curve
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        return self.calculate(returns.tolist())
    
    def _interpret_sortino(self, sortino: float) -> str:
        """
        Provide interpretation of Sortino ratio
        """
        if sortino < 0:
            return "Poor (negative risk-adjusted return)"
        elif sortino < 1.0:
            return "Below average"
        elif sortino < 2.0:
            return "Good"
        elif sortino < 3.0:
            return "Very Good"
        else:
            return "Excellent"
    
    def compare_with_sharpe(self, returns: List[float]) -> Dict[str, Any]:
        """
        Compare Sortino ratio with Sharpe ratio for the same returns
        """
        from analytics.performance_metrics.sharpe_ratio import SharpeRatio
        
        sharpe_calc = SharpeRatio(self.config)
        
        sharpe = sharpe_calc.calculate(returns)
        sortino = self.calculate(returns)
        
        # Calculate ratio (higher means more asymmetric risk)
        if sharpe['annualized_sharpe'] > 0:
            ratio = sortino['annualized_sortino'] / sharpe['annualized_sharpe']
        else:
            ratio = float('inf') if sortino['annualized_sortino'] > 0 else 0
        
        return {
            "sharpe": sharpe,
            "sortino": sortino,
            "sortino_sharpe_ratio": float(ratio),
            "interpretation": "Returns are symmetric" if 0.9 < ratio < 1.1 else "Returns have skew"
        }