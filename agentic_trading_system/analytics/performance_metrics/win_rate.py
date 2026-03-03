"""
Win Rate - Percentage of winning trades
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as logging

class WinRate:
    """
    Win Rate - Percentage of trades that are profitable
    
    Also provides:
    - Win rate by strategy
    - Win rate by symbol
    - Win rate over time
    - Statistical significance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.min_samples = config.get("min_samples", 10)  # Minimum trades for meaningful stats
        
        logging.info(f"✅ WinRate initialized")
    
    def calculate(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall win rate from trades
        """
        if not trades:
            return {
                "win_rate": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "breakeven_trades": 0
            }
        
        # Count outcomes
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        breakeven_trades = [t for t in trades if t.get('pnl', 0) == 0]
        
        total = len(trades)
        win_rate = len(winning_trades) / total if total > 0 else 0
        
        # Calculate confidence interval
        ci_lower, ci_upper = self._confidence_interval(win_rate, total)
        
        # Statistical significance
        is_significant = self._is_statistically_significant(winning_trades, losing_trades)
        
        return {
            "win_rate": float(win_rate),
            "win_rate_percent": float(win_rate * 100),
            "total_trades": total,
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "breakeven_trades": len(breakeven_trades),
            "confidence_interval": {
                "lower": float(ci_lower),
                "upper": float(ci_upper),
                "level": 0.95
            },
            "statistically_significant": is_significant,
            "needs_more_data": total < self.min_samples
        }
    
    def calculate_by_strategy(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate win rate by strategy
        """
        if not trades:
            return {}
        
        by_strategy = {}
        
        for trade in trades:
            strategy = trade.get('strategy', 'unknown')
            
            if strategy not in by_strategy:
                by_strategy[strategy] = {
                    'trades': [],
                    'wins': 0,
                    'losses': 0,
                    'breakeven': 0
                }
            
            by_strategy[strategy]['trades'].append(trade)
            
            pnl = trade.get('pnl', 0)
            if pnl > 0:
                by_strategy[strategy]['wins'] += 1
            elif pnl < 0:
                by_strategy[strategy]['losses'] += 1
            else:
                by_strategy[strategy]['breakeven'] += 1
        
        # Calculate rates
        results = {}
        for strategy, data in by_strategy.items():
            total = len(data['trades'])
            win_rate = data['wins'] / total if total > 0 else 0
            
            results[strategy] = {
                "win_rate": float(win_rate),
                "win_rate_percent": float(win_rate * 100),
                "total_trades": total,
                "wins": data['wins'],
                "losses": data['losses'],
                "breakeven": data['breakeven']
            }
        
        # Sort by win rate
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['win_rate'],
            reverse=True
        )
        
        return {
            "by_strategy": results,
            "rankings": [s for s, _ in sorted_results],
            "best_strategy": sorted_results[0][0] if sorted_results else None,
            "worst_strategy": sorted_results[-1][0] if sorted_results else None
        }
    
    def calculate_by_symbol(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate win rate by symbol
        """
        if not trades:
            return {}
        
        by_symbol = {}
        
        for trade in trades:
            symbol = trade.get('symbol', 'unknown')
            
            if symbol not in by_symbol:
                by_symbol[symbol] = {
                    'trades': [],
                    'wins': 0,
                    'losses': 0
                }
            
            by_symbol[symbol]['trades'].append(trade)
            
            if trade.get('pnl', 0) > 0:
                by_symbol[symbol]['wins'] += 1
            else:
                by_symbol[symbol]['losses'] += 1
        
        # Calculate rates
        results = {}
        for symbol, data in by_symbol.items():
            total = len(data['trades'])
            win_rate = data['wins'] / total if total > 0 else 0
            
            results[symbol] = {
                "win_rate": float(win_rate),
                "total_trades": total,
                "wins": data['wins'],
                "losses": data['losses']
            }
        
        return results
    
    def calculate_rolling(self, trades: List[Dict[str, Any]], 
                         window: int = 20) -> List[Dict[str, Any]]:
        """
        Calculate rolling win rate over time
        """
        if len(trades) < window:
            return []
        
        # Sort by exit time
        closed_trades = [t for t in trades if t.get('exit_time') is not None]
        closed_trades.sort(key=lambda x: x['exit_time'])
        
        rolling = []
        
        for i in range(window, len(closed_trades) + 1):
            window_trades = closed_trades[i - window:i]
            wins = sum(1 for t in window_trades if t.get('pnl', 0) > 0)
            win_rate = wins / window
            
            rolling.append({
                'date': closed_trades[i - 1]['exit_time'],
                'win_rate': float(win_rate),
                'wins': wins,
                'total': window,
                'window_start': closed_trades[i - window]['exit_time'],
                'window_end': closed_trades[i - 1]['exit_time']
            })
        
        return rolling
    
    def _confidence_interval(self, win_rate: float, n: int, 
                            confidence: float = 0.95) -> tuple:
        """
        Calculate confidence interval for win rate using Wilson score
        """
        if n == 0:
            return (0, 0)
        
        from scipy import stats
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        
        # Wilson score interval
        denominator = 1 + z**2 / n
        center = (win_rate + z**2 / (2 * n)) / denominator
        spread = z * np.sqrt((win_rate * (1 - win_rate) + z**2 / (4 * n)) / n) / denominator
        
        return (center - spread, center + spread)
    
    def _is_statistically_significant(self, wins: List, losses: List) -> bool:
        """
        Test if win rate is statistically significant (above 50%)
        """
        from scipy import stats
        
        n_wins = len(wins)
        n_losses = len(losses)
        n_total = n_wins + n_losses
        
        if n_total < 10:
            return False
        
        # Binomial test
        p_value = stats.binom_test(n_wins, n_total, p=0.5, alternative='greater')
        
        return p_value < 0.05