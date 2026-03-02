"""
Volatility Adjusted - Position sizing based on volatility
"""
from typing import Dict, Any, Optional
import numpy as np
from utils.logger import logger as logging

class VolatilityAdjusted:
    """
    Volatility Adjusted - Scale position size based on volatility
    
    Higher volatility = smaller position
    Lower volatility = larger position
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Base position size (as fraction of capital)
        self.base_fraction = config.get("base_fraction", 0.02)  # 2% base
        
        # Volatility target
        self.target_volatility = config.get("target_volatility", 0.20)  # 20% annualized
        
        # Scaling limits
        self.max_scale = config.get("max_scale", 2.0)  # Can go up to 2x base
        self.min_scale = config.get("min_scale", 0.25)  # Can go down to 0.25x base
        
        logging.info(f"✅ VolatilityAdjusted initialized")
    
    def calculate(self, capital: float, volatility: float, 
                 confidence: float = 1.0) -> Dict[str, Any]:
        """
        Calculate position size based on volatility
        """
        if volatility <= 0:
            scale_factor = 1.0
        else:
            # Scale inversely with volatility
            # If volatility is twice target, position half size
            scale_factor = self.target_volatility / volatility
        
        # Apply limits
        scale_factor = max(self.min_scale, min(self.max_scale, scale_factor))
        
        # Adjust for confidence
        confidence_factor = 0.5 + confidence * 0.5  # 0.5 to 1.0 range
        
        # Final fraction
        fraction = self.base_fraction * scale_factor * confidence_factor
        fraction = max(0.001, min(0.25, fraction))  # Hard limits
        
        position_value = capital * fraction
        
        return {
            "fraction": float(fraction),
            "position_value": float(position_value),
            "capital": float(capital),
            "volatility": float(volatility),
            "target_volatility": self.target_volatility,
            "scale_factor": float(scale_factor),
            "confidence_factor": float(confidence_factor),
            "base_fraction": self.base_fraction
        }
    
    def calculate_with_atr(self, capital: float, atr: float, price: float,
                          atr_periods: int = 14, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Calculate position size using ATR for volatility
        """
        if price <= 0:
            return {"error": "Invalid price"}
        
        # ATR as percentage of price
        atr_pct = atr / price
        
        # Use ATR percentage as volatility measure
        return self.calculate(capital, atr_pct, confidence)
    
    def get_scale_factor(self, volatility: float) -> float:
        """Get just the scale factor based on volatility"""
        if volatility <= 0:
            return 1.0
        return max(self.min_scale, min(self.max_scale, self.target_volatility / volatility))