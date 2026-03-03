"""
Recovery Factor - Measures how quickly portfolio recovers from drawdowns
"""
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from utils.logger import logger as  logging

class RecoveryFactor:
    """
    Recovery Factor - Measures recovery from drawdowns
    
    Features:
    - Time to recover from max drawdown
    - Average recovery time
    - Recovery ratio
    - V-shaped vs U-shaped recoveries
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        logging.info(f"✅ RecoveryFactor initialized")
    
    def calculate(self, equity_curve: List[float]) -> Dict[str, Any]:
        """
        Calculate recovery metrics
        """
        if len(equity_curve) < 2:
            return {
                "max_drawdown_recovery_time": None,
                "avg_recovery_time": None,
                "recovery_ratio": None,
                "fully_recovered": False
            }
        
        from analytics.performance_metrics.max_drawdown import MaxDrawdown
        dd_calc = MaxDrawdown(self.config)
        
        # Get max drawdown info
        dd_info = dd_calc.calculate(equity_curve)
        
        # Find all drawdown periods
        periods = dd_calc.find_drawdown_periods(equity_curve, threshold=0)
        
        if not periods:
            return {
                "max_drawdown_recovery_time": 0,
                "avg_recovery_time": 0,
                "recovery_ratio": float('inf'),
                "fully_recovered": True,
                "num_drawdowns": 0
            }
        
        # Calculate recovery times
        recovery_times = []
        recovery_ratios = []
        
        for period in periods:
            if 'end_index' in period:
                recovery_time = period['duration']
                recovery_times.append(recovery_time)
                
                # Recovery ratio = 1 / drawdown % (how much gain needed to recover)
                if period['max_drawdown'] > 0:
                    recovery_ratio = 1 / (period['max_drawdown'] / 100)
                    recovery_ratios.append(recovery_ratio)
        
        # Max drawdown recovery
        max_dd_recovery = None
        if dd_info['recovery_index'] is not None:
            max_dd_recovery = dd_info['recovery_duration']
        
        # Calculate statistics
        avg_recovery = np.mean(recovery_times) if recovery_times else None
        avg_recovery_ratio = np.mean(recovery_ratios) if recovery_ratios else None
        
        # Recovery efficiency (lower is better)
        if avg_recovery and avg_recovery_ratio:
            recovery_efficiency = avg_recovery / avg_recovery_ratio
        else:
            recovery_efficiency = None
        
        return {
            "max_drawdown_recovery_time": int(max_dd_recovery) if max_dd_recovery else None,
            "avg_recovery_time": float(avg_recovery) if avg_recovery else None,
            "avg_recovery_ratio": float(avg_recovery_ratio) if avg_recovery_ratio else None,
            "recovery_efficiency": float(recovery_efficiency) if recovery_efficiency else None,
            "fully_recovered": dd_info['fully_recovered'],
            "num_drawdowns": len(periods),
            "recovered_drawdowns": len([p for p in periods if 'end_index' in p]),
            "recovery_times": [int(t) for t in recovery_times]
        }
    
    def calculate_recovery_factor(self, total_return: float, 
                                 max_drawdown: float) -> float:
        """
        Calculate recovery factor = Total Return / Max Drawdown
        
        Measures how many times the system can withstand the max drawdown
        """
        if max_drawdown == 0:
            return float('inf')
        
        return total_return / max_drawdown
    
    def estimate_recovery_time(self, drawdown_pct: float, 
                              avg_daily_return: float) -> float:
        """
        Estimate time to recover from a drawdown
        """
        if avg_daily_return <= 0:
            return float('inf')
        
        # Recovery time = ln(1/(1-drawdown)) / ln(1+avg_return)
        # Simplified: drawdown / avg_daily_return
        return abs(drawdown_pct) / avg_daily_return
    
    def analyze_recovery_pattern(self, equity_curve: List[float], 
                                drawdown_start: int,
                                drawdown_end: int,
                                recovery_end: int) -> str:
        """
        Analyze the shape of recovery (V-shaped, U-shaped, L-shaped)
        """
        if drawdown_end is None or recovery_end is None:
            return "incomplete"
        
        # Extract recovery period
        recovery_period = equity_curve[drawdown_end:recovery_end + 1]
        
        if len(recovery_period) < 3:
            return "insufficient_data"
        
        # Calculate slope of recovery
        first_half = recovery_period[:len(recovery_period)//2]
        second_half = recovery_period[len(recovery_period)//2:]
        
        first_slope = (first_half[-1] - first_half[0]) / len(first_half)
        second_slope = (second_half[-1] - second_half[0]) / len(second_half)
        
        if first_slope > second_slope * 1.5:
            return "V-shaped"  # Fast initial recovery
        elif second_slope > first_slope * 1.5:
            return "U-shaped"  # Slow initial, fast later
        elif abs(first_slope - second_slope) < 0.1:
            return "linear"
        else:
            return "irregular"
    
    def get_recovery_summary(self, trades: List[Dict[str, Any]],
                            equity_curve: List[float]) -> Dict[str, Any]:
        """
        Get comprehensive recovery analysis
        """
        recovery = self.calculate(equity_curve)
        
        # Calculate total return
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0] * 100
        
        # Get max drawdown
        from analytics.performance_metrics.max_drawdown import MaxDrawdown
        dd_calc = MaxDrawdown(self.config)
        dd_info = dd_calc.calculate(equity_curve)
        
        # Calculate recovery factor
        recovery_factor = self.calculate_recovery_factor(
            total_return, dd_info['max_drawdown_pct']
        )
        
        return {
            "recovery_metrics": recovery,
            "total_return": float(total_return),
            "max_drawdown": dd_info['max_drawdown_pct'],
            "recovery_factor": float(recovery_factor),
            "recovery_factor_interpretation": self._interpret_recovery_factor(recovery_factor),
            "risk_score": self._calculate_risk_score(recovery, recovery_factor)
        }
    
    def _interpret_recovery_factor(self, rf: float) -> str:
        """
        Interpret recovery factor
        """
        if rf == float('inf'):
            return "Perfect (no drawdown)"
        elif rf > 5:
            return "Excellent"
        elif rf > 3:
            return "Very Good"
        elif rf > 2:
            return "Good"
        elif rf > 1:
            return "Adequate"
        else:
            return "Poor"
    
    def _calculate_risk_score(self, recovery: Dict[str, Any], 
                             recovery_factor: float) -> float:
        """
        Calculate risk score based on recovery metrics (0-100, lower is better)
        """
        score = 50  # Start neutral
        
        # Longer recovery time = higher risk
        if recovery.get('avg_recovery_time'):
            if recovery['avg_recovery_time'] > 50:
                score += 20
            elif recovery['avg_recovery_time'] > 20:
                score += 10
            elif recovery['avg_recovery_time'] < 5:
                score -= 10
        
        # Lower recovery factor = higher risk
        if recovery_factor < 1:
            score += 30
        elif recovery_factor < 2:
            score += 15
        elif recovery_factor > 5:
            score -= 15
        
        # Not fully recovered = higher risk
        if not recovery.get('fully_recovered', True):
            score += 20
        
        return max(0, min(100, score))