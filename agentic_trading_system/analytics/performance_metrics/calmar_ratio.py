"""
Calmar Ratio - Return to maximum drawdown ratio
"""
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime, timedelta
from utils.logger import logger as logging

class CalmarRatio:
    """
    Calmar Ratio - Measures return relative to maximum drawdown
    
    Formula: Annualized Return / Maximum Drawdown
    
    Higher is better - indicates good returns with controlled drawdowns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.trading_days = config.get("trading_days", 252)
        self.lookback_years = config.get("lookback_years", 3)  # Typical Calmar uses 3 years
        
        logging.info(f"✅ CalmarRatio initialized")
    
    def calculate(self, returns: List[float], 
                 max_drawdown: float = None,
                 periods_per_year: int = None) -> Dict[str, Any]:
        """
        Calculate Calmar ratio
        
        Args:
            returns: List of periodic returns
            max_drawdown: Maximum drawdown percentage (if None, calculate from returns)
            periods_per_year: Number of periods in a year
        """
        if not returns or len(returns) < 2:
            return {
                "calmar_ratio": 0.0,
                "error": "Insufficient return data"
            }
        
        if periods_per_year is None:
            periods_per_year = self.trading_days
        
        returns_array = np.array(returns)
        
        # Calculate annualized return
        total_return = np.prod(1 + returns_array) - 1
        num_years = len(returns) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / num_years) - 1 if num_years > 0 else 0
        
        # Calculate max drawdown if not provided
        if max_drawdown is None:
            from analytics.performance_metrics.max_drawdown import MaxDrawdown
            dd_calc = MaxDrawdown(self.config)
            dd_result = dd_calc.calculate_from_returns(returns)
            max_drawdown = dd_result['max_drawdown_pct'] / 100  # Convert to decimal
        else:
            max_drawdown = max_drawdown / 100 if max_drawdown > 1 else max_drawdown
        
        if max_drawdown == 0:
            return {
                "calmar_ratio": float('inf') if annualized_return > 0 else 0.0,
                "annualized_return": float(annualized_return),
                "max_drawdown": 0.0,
                "note": "Zero drawdown"
            }
        
        # Calculate Calmar ratio
        calmar = annualized_return / max_drawdown
        
        # Interpret result
        interpretation = self._interpret_calmar(calmar)
        
        return {
            "calmar_ratio": float(calmar),
            "annualized_return": float(annualized_return),
            "max_drawdown": float(max_drawdown * 100),  # Back to percentage
            "total_return": float(total_return * 100),
            "period_years": float(num_years),
            "interpretation": interpretation,
            "num_periods": len(returns)
        }
    
    def calculate_from_equity(self, equity_curve: List[float]) -> Dict[str, Any]:
        """
        Calculate Calmar ratio from equity curve
        """
        if len(equity_curve) < 2:
            return {"calmar_ratio": 0.0, "error": "Insufficient data"}
        
        # Calculate returns from equity curve
        equity_array = np.array(equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Calculate max drawdown
        from analytics.performance_metrics.max_drawdown import MaxDrawdown
        dd_calc = MaxDrawdown(self.config)
        dd_result = dd_calc.calculate(equity_curve)
        
        return self.calculate(returns.tolist(), dd_result['max_drawdown_pct'] / 100)
    
    def _interpret_calmar(self, calmar: float) -> str:
        """
        Provide interpretation of Calmar ratio
        """
        if calmar < 0:
            return "Negative (loss-making strategy)"
        elif calmar < 0.5:
            return "Poor risk-adjusted returns"
        elif calmar < 1.0:
            return "Average"
        elif calmar < 2.0:
            return "Good"
        elif calmar < 3.0:
            return "Very Good"
        else:
            return "Excellent"
    
    def compare_strategies(self, strategy_returns: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Compare Calmar ratios of multiple strategies
        """
        results = {}
        
        for name, returns in strategy_returns.items():
            results[name] = self.calculate(returns)
        
        # Sort by Calmar ratio
        sorted_strategies = sorted(
            results.items(),
            key=lambda x: x[1]['calmar_ratio'],
            reverse=True
        )
        
        return {
            "results": results,
            "rankings": [name for name, _ in sorted_strategies],
            "best_strategy": sorted_strategies[0][0] if sorted_strategies else None,
            "worst_strategy": sorted_strategies[-1][0] if sorted_strategies else None
        }