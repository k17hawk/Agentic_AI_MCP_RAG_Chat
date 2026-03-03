"""
Maximum Drawdown - Largest peak-to-trough decline
"""
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime
from utils.logger import logger as  logging

class MaxDrawdown:
    """
    Maximum Drawdown - Largest peak-to-trough decline
    
    Features:
    - Maximum drawdown percentage
    - Drawdown duration
    - Recovery time
    - Drawdown curve
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        logging.info(f"✅ MaxDrawdown initialized")
    
    def calculate(self, equity_curve: List[float]) -> Dict[str, Any]:
        """
        Calculate maximum drawdown from equity curve
        """
        if len(equity_curve) < 2:
            return {
                "max_drawdown_pct": 0.0,
                "max_drawdown_value": 0.0,
                "peak_index": 0,
                "trough_index": 0,
                "recovery_index": None
            }
        
        equity_array = np.array(equity_curve)
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Calculate drawdown
        drawdown = (running_max - equity_array) / running_max
        drawdown_pct = drawdown * 100
        
        # Find maximum drawdown
        max_dd_idx = np.argmax(drawdown)
        max_dd = drawdown[max_dd_idx] * 100
        
        # Find peak before max drawdown
        peak_idx = np.argmax(equity_array[:max_dd_idx + 1])
        peak_value = equity_array[peak_idx]
        trough_value = equity_array[max_dd_idx]
        
        # Find recovery (if any)
        recovery_idx = None
        recovery_date = None
        for i in range(max_dd_idx + 1, len(equity_array)):
            if equity_array[i] >= peak_value:
                recovery_idx = i
                recovery_date = i
                break
        
        # Calculate drawdown duration
        drawdown_duration = max_dd_idx - peak_idx
        recovery_duration = recovery_idx - max_dd_idx if recovery_idx else None
        
        return {
            "max_drawdown_pct": float(max_dd),
            "max_drawdown_value": float(peak_value - trough_value),
            "peak_value": float(peak_value),
            "trough_value": float(trough_value),
            "peak_index": int(peak_idx),
            "trough_index": int(max_dd_idx),
            "recovery_index": int(recovery_idx) if recovery_idx else None,
            "drawdown_duration": int(drawdown_duration),
            "recovery_duration": int(recovery_duration) if recovery_duration else None,
            "fully_recovered": recovery_idx is not None,
            "drawdown_curve": drawdown_pct.tolist()
        }
    
    def calculate_from_returns(self, returns: List[float]) -> Dict[str, Any]:
        """
        Calculate maximum drawdown from returns series
        """
        if not returns:
            return {"max_drawdown_pct": 0.0}
        
        # Build equity curve from returns
        equity = [100]  # Start at 100
        for r in returns:
            equity.append(equity[-1] * (1 + r))
        
        return self.calculate(equity)
    
    def calculate_drawdown_curve(self, equity_curve: List[float]) -> List[float]:
        """
        Calculate full drawdown curve
        """
        equity_array = np.array(equity_curve)
        running_max = np.maximum.accumulate(equity_array)
        drawdown = (running_max - equity_array) / running_max * 100
        
        return drawdown.tolist()
    
    def calculate_average_drawdown(self, equity_curve: List[float]) -> Dict[str, Any]:
        """
        Calculate average drawdown (excluding zero drawdown periods)
        """
        drawdown_curve = self.calculate_drawdown_curve(equity_curve)
        
        # Filter out zero drawdown
        non_zero_dd = [d for d in drawdown_curve if d > 0]
        
        if not non_zero_dd:
            return {
                "average_drawdown": 0.0,
                "median_drawdown": 0.0,
                "std_drawdown": 0.0,
                "max_drawdown": 0.0
            }
        
        return {
            "average_drawdown": float(np.mean(non_zero_dd)),
            "median_drawdown": float(np.median(non_zero_dd)),
            "std_drawdown": float(np.std(non_zero_dd)),
            "max_drawdown": float(np.max(non_zero_dd)),
            "num_drawdown_periods": len(non_zero_dd)
        }
    
    def find_drawdown_periods(self, equity_curve: List[float], 
                             threshold: float = 5.0) -> List[Dict[str, Any]]:
        """
        Find all drawdown periods above threshold
        """
        drawdown_curve = self.calculate_drawdown_curve(equity_curve)
        
        periods = []
        in_drawdown = False
        start_idx = 0
        max_dd_in_period = 0
        
        for i, dd in enumerate(drawdown_curve):
            if dd > threshold and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
                max_dd_in_period = dd
            elif in_drawdown:
                if dd > max_dd_in_period:
                    max_dd_in_period = dd
                
                if dd == 0:
                    # End of drawdown
                    in_drawdown = False
                    periods.append({
                        "start_index": start_idx,
                        "end_index": i,
                        "duration": i - start_idx,
                        "max_drawdown": float(max_dd_in_period)
                    })
                    max_dd_in_period = 0
        
        return periods
    
    def get_worst_drawdowns(self, equity_curve: List[float], 
                           n: int = 5) -> List[Dict[str, Any]]:
        """
        Get the n worst drawdowns
        """
        periods = self.find_drawdown_periods(equity_curve, threshold=0)
        
        # Sort by max drawdown (largest first)
        periods.sort(key=lambda x: x['max_drawdown'], reverse=True)
        
        return periods[:n]
    
    def calculate_underwater_period(self, equity_curve: List[float]) -> float:
        """
        Calculate percentage of time spent underwater (in drawdown)
        """
        drawdown_curve = self.calculate_drawdown_curve(equity_curve)
        
        underwater_periods = sum(1 for dd in drawdown_curve if dd > 0)
        total_periods = len(drawdown_curve)
        
        if total_periods == 0:
            return 0.0
        
        return (underwater_periods / total_periods) * 100