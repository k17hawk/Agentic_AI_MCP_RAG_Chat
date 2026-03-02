"""
Volatility Stop - Stop loss based on price volatility
"""
from typing import Dict, Any, List
import numpy as np
from utils.logger import logger as  logging

class VolatilityStop:
    """
    Volatility-based stop loss - Uses standard deviation of returns
    
    Stop = Entry Price - (price * volatility_multiplier * std_dev)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.default_multiplier = config.get("default_multiplier", 2.0)
        self.lookback_period = config.get("lookback_period", 20)
        
        # Bands
        self.min_multiplier = config.get("min_multiplier", 1.0)
        self.max_multiplier = config.get("max_multiplier", 4.0)
        
        logging.info(f"✅ VolatilityStop initialized")
    
    def calculate(self, entry_price: float, returns: List[float],
                 multiplier: float = None, direction: str = "long") -> Dict[str, Any]:
        """
        Calculate stop loss based on return volatility
        """
        if multiplier is None:
            multiplier = self.default_multiplier
        
        multiplier = max(self.min_multiplier, min(self.max_multiplier, multiplier))
        
        # Calculate volatility
        returns_array = np.array(returns)
        std_dev = np.std(returns_array)
        
        # Convert to price move
        volatility_move = entry_price * std_dev * multiplier
        
        if direction.lower() == "long":
            stop_price = entry_price - volatility_move
            stop_distance = volatility_move
        else:  # short
            stop_price = entry_price + volatility_move
            stop_distance = volatility_move
        
        stop_pct = (stop_distance / entry_price) * 100
        
        return {
            "stop_price": float(stop_price),
            "stop_distance": float(stop_distance),
            "stop_percent": float(stop_pct),
            "entry_price": float(entry_price),
            "volatility": float(std_dev),
            "volatility_annualized": float(std_dev * np.sqrt(252)),
            "multiplier": float(multiplier),
            "direction": direction,
            "risk_per_share": float(stop_distance),
            "lookback_period": self.lookback_period
        }
    
    def calculate_from_prices(self, entry_price: float, prices: List[float],
                             multiplier: float = None, direction: str = "long") -> Dict[str, Any]:
        """
        Calculate stop loss from price series
        """
        # Calculate returns
        price_array = np.array(prices)
        returns = np.diff(price_array) / price_array[:-1]
        
        return self.calculate(entry_price, returns.tolist(), multiplier, direction)