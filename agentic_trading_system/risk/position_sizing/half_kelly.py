"""
Half Kelly - Conservative version of Kelly Criterion
"""
from typing import Dict, Any, Optional
from risk.position_sizing.kelly_criterion import KellyCriterion
from utils.logger import logger as logging

class HalfKelly(KellyCriterion):
    """
    Half Kelly - Uses half of Kelly fraction for more conservative sizing
    
    Formula: f = Kelly / 2
    
    This is more conservative and reduces risk of ruin
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.kelly_multiplier = config.get("kelly_multiplier", 0.5)
        logging.info(f"✅ HalfKelly initialized (multiplier: {self.kelly_multiplier})")
    
    def calculate(self, win_rate: float = None, avg_win: float = None, 
                 avg_loss: float = None) -> Dict[str, Any]:
        """
        Calculate Half Kelly fraction
        """
        # Get full Kelly first
        kelly_result = super().calculate(win_rate, avg_win, avg_loss)
        
        # Apply half Kelly
        full_kelly = kelly_result["kelly_fraction"]
        half_kelly = full_kelly * self.kelly_multiplier
        
        # Apply constraints
        recommended_fraction = max(self.min_fraction, min(self.max_fraction, half_kelly))
        
        return {
            "kelly_fraction": float(full_kelly),
            "half_kelly_fraction": float(half_kelly),
            "recommended_fraction": float(recommended_fraction),
            "multiplier": self.kelly_multiplier,
            "edge": kelly_result["edge"],
            "odds": kelly_result["odds"],
            "win_rate": kelly_result["win_rate"],
            "avg_win": kelly_result["avg_win"],
            "avg_loss": kelly_result["avg_loss"],
            "is_valid": half_kelly > 0,
            "message": f"Half Kelly: {half_kelly:.1%} of capital (Full Kelly: {full_kelly:.1%})"
        }