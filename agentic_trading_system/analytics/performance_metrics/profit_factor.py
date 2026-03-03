"""
Profit Factor - Ratio of gross profits to gross losses
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as logging

class ProfitFactor:
    """
    Profit Factor - Ratio of gross profits to gross losses
    
    Formula: Gross Profit / Gross Loss
    
    Interpretation:
    - > 1.0: Profitable system
    - > 1.5: Good
    - > 2.0: Excellent
    - < 1.0: Losing system
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.min_samples = config.get("min_samples", 5)
        
        logging.info(f"✅ ProfitFactor initialized")
    
    def calculate(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate profit factor from trades
        """
        if not trades:
            return {
                "profit_factor": 0.0,
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "total_trades": 0
            }
        
        # Separate winning and losing trades
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        if gross_loss == 0:
            profit_factor = float('inf') if gross_profit > 0 else 0.0
        else:
            profit_factor = gross_profit / gross_loss
        
        # Interpret result
        interpretation = self._interpret_profit_factor(profit_factor)
        
        return {
            "profit_factor": float(profit_factor) if profit_factor != float('inf') else float('inf'),
            "gross_profit": float(gross_profit),
            "gross_loss": float(gross_loss),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "total_trades": len(trades),
            "interpretation": interpretation,
            "is_profitable": profit_factor > 1.0
        }
    
    def calculate_by_strategy(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate profit factor by strategy
        """
        if not trades:
            return {}
        
        by_strategy = {}
        
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            
            if strategy not in by_strategy:
                by_strategy[strategy] = {
                    'winning_pnl': 0.0,
                    'losing_pnl': 0.0,
                    'wins': 0,
                    'losses': 0
                }
            
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                by_strategy[strategy]['winning_pnl'] += pnl
                by_strategy[strategy]['wins'] += 1
            elif pnl < 0:
                by_strategy[strategy]['losing_pnl'] += abs(pnl)
                by_strategy[strategy]['losses'] += 1
        
        # Calculate profit factors
        results = {}
        for strategy, data in by_strategy.items():
            if data['losing_pnl'] > 0:
                pf = data['winning_pnl'] / data['losing_pnl']
            else:
                pf = float('inf') if data['winning_pnl'] > 0 else 0.0
            
            results[strategy] = {
                "profit_factor": float(pf) if pf != float('inf') else float('inf'),
                "gross_profit": float(data['winning_pnl']),
                "gross_loss": float(data['losing_pnl']),
                "wins": data['wins'],
                "losses": data['losses'],
                "total_trades": data['wins'] + data['losses']
            }
        
        # Sort by profit factor
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['profit_factor'] if x[1]['profit_factor'] != float('inf') else 999,
            reverse=True
        )
        
        return {
            "by_strategy": results,
            "rankings": [s for s, _ in sorted_results],
            "best_strategy": sorted_results[0][0] if sorted_results else None
        }
    
    def calculate_rolling(self, trades: List[Dict[str, Any]], 
                         window: int = 20) -> List[Dict[str, Any]]:
        """
        Calculate rolling profit factor over time
        """
        if len(trades) < window:
            return []
        
        # Sort by exit time
        closed_trades = [t for t in trades if t.get('exit_time') is not None]
        closed_trades.sort(key=lambda x: x['exit_time'])
        
        rolling = []
        
        for i in range(window, len(closed_trades) + 1):
            window_trades = closed_trades[i - window:i]
            
            winning_pnl = sum(t['pnl'] for t in window_trades if t['pnl'] > 0)
            losing_pnl = abs(sum(t['pnl'] for t in window_trades if t['pnl'] < 0))
            
            if losing_pnl > 0:
                pf = winning_pnl / losing_pnl
            else:
                pf = float('inf') if winning_pnl > 0 else 0.0
            
            rolling.append({
                'date': closed_trades[i - 1]['exit_time'],
                'profit_factor': float(pf) if pf != float('inf') else float('inf'),
                'gross_profit': float(winning_pnl),
                'gross_loss': float(losing_pnl),
                'window_start': closed_trades[i - window]['exit_time'],
                'window_end': closed_trades[i - 1]['exit_time']
            })
        
        return rolling
    
    def _interpret_profit_factor(self, pf: float) -> str:
        """
        Provide interpretation of profit factor
        """
        if pf == float('inf'):
            return "Perfect (no losing trades)"
        elif pf > 3.0:
            return "Excellent"
        elif pf > 2.0:
            return "Very Good"
        elif pf > 1.5:
            return "Good"
        elif pf > 1.0:
            return "Profitable"
        elif pf == 1.0:
            return "Breakeven"
        else:
            return "Losing"
    
    def required_win_rate(self, avg_win: float, avg_loss: float) -> float:
        """
        Calculate required win rate for given profit factor target
        """
        if avg_loss <= 0:
            return 0
        
        # PF = (win_rate * avg_win) / ((1 - win_rate) * avg_loss)
        # Solve for win_rate
        # win_rate = PF * avg_loss / (avg_win + PF * avg_loss)
        
        target_pf = 1.5  # Target profit factor
        
        required = (target_pf * avg_loss) / (avg_win + target_pf * avg_loss)
        
        return float(required)