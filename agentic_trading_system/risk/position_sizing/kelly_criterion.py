"""
Kelly Criterion - Optimal position sizing based on edge and odds
"""
from typing import Dict, Any, Optional,List
import numpy as np
from utils.logger import logger as logging

class KellyCriterion:
    """
    Kelly Criterion for optimal position sizing
    
    Formula: f* = (bp - q) / b
    where:
    - f* = fraction of capital to bet
    - b = odds received on the bet (profit per unit risked)
    - p = probability of winning
    - q = probability of losing (1-p)
    
    For trading: f* = (win_rate * avg_win - loss_rate * avg_loss) / (avg_win * avg_loss)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.default_win_rate = config.get("default_win_rate", 0.55)
        self.default_avg_win = config.get("default_avg_win", 0.10)  # 10%
        self.default_avg_loss = config.get("default_avg_loss", 0.05)  # 5%
        
        # Constraints
        self.max_fraction = config.get("max_fraction", 0.25)  # Max 25% of capital
        self.min_fraction = config.get("min_fraction", 0.01)  # Min 1% of capital
        
        logging.info(f"✅ KellyCriterion initialized")
    
    def calculate(self, win_rate: float = None, avg_win: float = None, 
                 avg_loss: float = None) -> Dict[str, Any]:
        """
        Calculate Kelly fraction
        """
        # Use defaults if not provided
        win_rate = win_rate if win_rate is not None else self.default_win_rate
        avg_win = avg_win if avg_win is not None else self.default_avg_win
        avg_loss = avg_loss if avg_loss is not None else self.default_avg_loss
        
        # Validate inputs
        if not (0 < win_rate < 1):
            raise ValueError(f"Win rate must be between 0 and 1, got {win_rate}")
        if avg_win <= 0:
            raise ValueError(f"Average win must be positive, got {avg_win}")
        if avg_loss <= 0:
            raise ValueError(f"Average loss must be positive, got {avg_loss}")
        
        loss_rate = 1 - win_rate
        
        # Kelly formula
        kelly_fraction = (win_rate * avg_win - loss_rate * avg_loss) / (avg_win * avg_loss)
        
        # Apply constraints
        recommended_fraction = max(self.min_fraction, min(self.max_fraction, kelly_fraction))
        
        # Calculate edge
        edge = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        # Calculate odds
        odds = avg_win / avg_loss if avg_loss > 0 else 0
        
        return {
            "kelly_fraction": float(kelly_fraction),
            "recommended_fraction": float(recommended_fraction),
            "edge": float(edge),
            "odds": float(odds),
            "win_rate": float(win_rate),
            "avg_win": float(avg_win),
            "avg_loss": float(avg_loss),
            "is_valid": kelly_fraction > 0,
            "message": self._get_message(kelly_fraction, recommended_fraction)
        }
    
    def calculate_from_trades(self, trades: List[Dict]) -> Dict[str, Any]:
        """
        Calculate Kelly fraction from historical trades
        """
        if not trades:
            return self.calculate()
        
        # Calculate win rate and averages
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        win_rate = len(wins) / len(trades) if trades else 0
        
        if wins:
            avg_win = np.mean([t['pnl'] / t['position_value'] for t in wins if t.get('position_value')])
        else:
            avg_win = self.default_avg_win
        
        if losses:
            avg_loss = abs(np.mean([t['pnl'] / t['position_value'] for t in losses if t.get('position_value')]))
        else:
            avg_loss = self.default_avg_loss
        
        return self.calculate(win_rate, avg_win, avg_loss)
    
    def _get_message(self, kelly: float, recommended: float) -> str:
        """Get human-readable message"""
        if kelly <= 0:
            return "No positive edge - consider not trading"
        elif kelly > self.max_fraction:
            return f"Kelly suggests {kelly:.1%} but capped at {self.max_fraction:.1%}"
        elif kelly < self.min_fraction:
            return f"Kelly suggests {kelly:.1%} but minimum is {self.min_fraction:.1%}"
        else:
            return f"Optimal Kelly fraction: {kelly:.1%}"