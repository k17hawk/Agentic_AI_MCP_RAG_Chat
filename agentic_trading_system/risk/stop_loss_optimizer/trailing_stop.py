"""
Trailing Stop - Dynamically adjusts stop loss as price moves favorably
"""
from typing import Dict, Any, Optional, List
from utils.logger import logger as logging

class TrailingStop:
    """
    Trailing stop loss - Moves stop as price moves in your favor
    
    Types:
    - Percentage based: Stop trails by fixed percentage
    - ATR based: Stop trails by ATR multiple
    - Parabolic SAR: Uses Parabolic SAR indicator
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.trail_percent = config.get("trail_percent", 5.0)  # 5% trail
        self.atr_multiplier = config.get("atr_multiplier", 2.0)
        self.activation_percent = config.get("activation_percent", 10.0)  # Activate after 10% profit
        
        logging.info(f"✅ TrailingStop initialized")
    
    def calculate_percentage(self, entry_price: float, current_price: float,
                            highest_price: float = None, trail_pct: float = None,
                            direction: str = "long") -> Dict[str, Any]:
        """
        Calculate trailing stop based on percentage
        """
        if trail_pct is None:
            trail_pct = self.trail_percent
        
        if highest_price is None:
            highest_price = max(entry_price, current_price)
        
        if direction.lower() == "long":
            # For long positions, stop trails below highest price
            trail_amount = highest_price * (trail_pct / 100)
            stop_price = highest_price - trail_amount
            
            # Calculate profit
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            
            # Check if activated
            is_activated = profit_pct >= self.activation_percent
            
            # Distance from current price
            distance = current_price - stop_price
            distance_pct = (distance / current_price) * 100
            
        else:  # short
            # For short positions, stop trails above lowest price
            lowest_price = min(entry_price, current_price) if direction == "short" else highest_price
            trail_amount = lowest_price * (trail_pct / 100)
            stop_price = lowest_price + trail_amount
            
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            is_activated = profit_pct >= self.activation_percent
            distance = stop_price - current_price
            distance_pct = (distance / current_price) * 100
        
        return {
            "stop_price": float(stop_price),
            "current_price": float(current_price),
            "entry_price": float(entry_price),
            "highest_price": float(highest_price) if direction == "long" else float(lowest_price),
            "trail_percent": trail_pct,
            "profit_percent": float(profit_pct),
            "is_activated": is_activated,
            "distance_to_stop": float(distance),
            "distance_percent": float(distance_pct),
            "direction": direction,
            "type": "percentage"
        }
    
    def calculate_atr(self, entry_price: float, current_price: float,
                     atr: float, highest_price: float = None,
                     multiplier: float = None, direction: str = "long") -> Dict[str, Any]:
        """
        Calculate trailing stop based on ATR
        """
        if multiplier is None:
            multiplier = self.atr_multiplier
        
        if highest_price is None:
            highest_price = max(entry_price, current_price)
        
        if direction.lower() == "long":
            trail_amount = atr * multiplier
            stop_price = highest_price - trail_amount
            
            profit_pct = ((current_price - entry_price) / entry_price) * 100
            is_activated = profit_pct >= self.activation_percent
            distance = current_price - stop_price
            
        else:  # short
            lowest_price = min(entry_price, current_price)
            trail_amount = atr * multiplier
            stop_price = lowest_price + trail_amount
            
            profit_pct = ((entry_price - current_price) / entry_price) * 100
            is_activated = profit_pct >= self.activation_percent
            distance = stop_price - current_price
        
        return {
            "stop_price": float(stop_price),
            "current_price": float(current_price),
            "entry_price": float(entry_price),
            "highest_price": float(highest_price) if direction == "long" else float(lowest_price),
            "atr": float(atr),
            "atr_multiplier": multiplier,
            "trail_amount": float(trail_amount),
            "profit_percent": float(profit_pct),
            "is_activated": is_activated,
            "distance_to_stop": float(distance),
            "direction": direction,
            "type": "atr"
        }
    
    def update_stop(self, current_stop: float, current_price: float,
                   highest_price: float, trail_pct: float = None) -> float:
        """
        Update trailing stop with new high
        """
        if trail_pct is None:
            trail_pct = self.trail_percent
        
        new_stop = highest_price * (1 - trail_pct / 100)
        
        # Stop can only move up (for longs)
        return max(current_stop, new_stop)