"""
ATR Stop - Stop loss based on Average True Range
"""
from typing import Dict, Any, Optional,List
import numpy as np
from agentic_trading_system.utils.logger import logger as logging

class ATRStop:
    """
    ATR-based stop loss - Uses Average True Range to set stop distance
    
    Stop = Entry Price - (ATR * multiplier)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default multiplier
        self.default_multiplier = config.get("default_multiplier", 2.0)
        self.min_multiplier = config.get("min_multiplier", 1.0)
        self.max_multiplier = config.get("max_multiplier", 4.0)
        
        # ATR period
        self.atr_period = config.get("atr_period", 14)
        
        logging.info(f"✅ ATRStop initialized")
    
    def calculate(self, entry_price: float, atr: float, 
                 multiplier: float = None, direction: str = "long") -> Dict[str, Any]:
        """
        Calculate stop loss price using ATR
        """
        if multiplier is None:
            multiplier = self.default_multiplier
        
        multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))
        
        if direction.lower() == "long":
            stop_price = entry_price - (atr * multiplier)
            stop_distance = entry_price - stop_price
            stop_pct = (stop_distance / entry_price) * 100
        else:  # short
            stop_price = entry_price + (atr * multiplier)
            stop_distance = stop_price - entry_price
            stop_pct = (stop_distance / entry_price) * 100
        
        return {
            "stop_price": float(stop_price),
            "stop_distance": float(stop_distance),
            "stop_percent": float(stop_pct),
            "entry_price": float(entry_price),
            "atr": float(atr),
            "multiplier": float(multiplier),
            "direction": direction,
            "risk_per_share": float(stop_distance),
            "atr_percent": float(atr / entry_price * 100)
        }
    
    def calculate_multiple(self, entry_price: float, atr: float,
                          multipliers: List[float]) -> List[Dict[str, Any]]:
        """
        Calculate stops for multiple multipliers
        """
        results = []
        for mult in multipliers:
            results.append(self.calculate(entry_price, atr, mult))
        return results
    
    def suggest_multiplier(self, volatility_regime: str) -> float:
        """
        Suggest multiplier based on volatility regime
        """
        suggestions = {
            "very_low": 1.5,
            "low": 1.8,
            "normal": 2.0,
            "high": 2.5,
            "very_high": 3.0,
            "panic": 3.5
        }
        return suggestions.get(volatility_regime, self.default_multiplier)