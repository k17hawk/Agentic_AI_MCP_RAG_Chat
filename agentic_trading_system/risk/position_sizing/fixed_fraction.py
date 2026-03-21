"""
Fixed Fraction - Simple fixed percentage position sizing
"""
from typing import Dict, Any, Optional
from agentic_trading_system.utils.logger import logger as  logging

class FixedFraction:
    """
    Fixed Fraction - Allocate fixed percentage of capital per trade
    
    Simplest approach: risk X% of capital per trade
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default fraction
        self.default_fraction = config.get("default_fraction", 0.02)  # 2% default
        self.max_fraction = config.get("max_fraction", 0.10)  # 10% max
        self.min_fraction = config.get("min_fraction", 0.005)  # 0.5% min
        
        # Scaling options
        self.scale_with_confidence = config.get("scale_with_confidence", True)
        self.confidence_scale_factor = config.get("confidence_scale_factor", 0.5)
        
        logging.info(f"✅ FixedFraction initialized (default: {self.default_fraction:.1%})")
    
    def calculate(self, capital: float, confidence: float = 1.0, 
                 signal_strength: float = 1.0) -> Dict[str, Any]:
        """
        Calculate position size based on fixed fraction
        """
        # Base fraction
        fraction = self.default_fraction
        
        # Scale with confidence if enabled
        if self.scale_with_confidence:
            # Confidence should be 0-1
            confidence = max(0, min(1, confidence))
            fraction = fraction * (1 + (confidence - 0.5) * self.confidence_scale_factor)
        
        # Scale with signal strength
        signal_strength = max(0, min(1, signal_strength))
        fraction = fraction * signal_strength
        
        # Apply limits
        fraction = max(self.min_fraction, min(self.max_fraction, fraction))
        
        # Calculate position value
        position_value = capital * fraction
        
        return {
            "fraction": float(fraction),
            "position_value": float(position_value),
            "capital": float(capital),
            "confidence_multiplier": float(1 + (confidence - 0.5) * self.confidence_scale_factor) if self.scale_with_confidence else 1.0,
            "signal_multiplier": float(signal_strength),
            "min_fraction": self.min_fraction,
            "max_fraction": self.max_fraction
        }
    
    def calculate_with_risk(self, capital: float, risk_per_share: float, 
                           stop_distance: float, confidence: float = 1.0) -> Dict[str, Any]:
        """
        Calculate position size based on fixed risk fraction
        """
        # Risk amount = capital * default_fraction
        risk_amount = capital * self.default_fraction * confidence
        
        # Position size = risk_amount / stop_distance
        if stop_distance <= 0:
            return {
                "error": "Invalid stop distance",
                "shares": 0,
                "position_value": 0
            }
        
        shares = int(risk_amount / stop_distance)
        position_value = shares * risk_per_share
        
        return {
            "shares": shares,
            "position_value": float(position_value),
            "risk_amount": float(risk_amount),
            "risk_per_share": float(risk_per_share),
            "stop_distance": float(stop_distance),
            "fraction": float(self.default_fraction * confidence)
        }