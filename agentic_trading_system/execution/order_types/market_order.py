"""
Market Order - Execute immediately at current market price
"""
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from utils.logger import logger as  logging

class MarketOrder:
    """
    Market Order - Executes immediately at current market price
    
    Characteristics:
    - Immediate execution
    - Price not guaranteed (may slip)
    - High fill probability
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.default_time_in_force = config.get("default_time_in_force", "DAY")
        self.max_slippage = config.get("max_slippage", 0.02)  # 2% max slippage
        
        logging.info(f"✅ MarketOrder initialized")
    
    def create(self, symbol: str, quantity: int, side: str, 
              time_in_force: str = None, client_id: str = None) -> Dict[str, Any]:
        """
        Create a market order
        """
        order = {
            "order_id": str(uuid.uuid4()),
            "client_order_id": client_id or str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "order_type": "MARKET",
            "side": side.upper(),  # BUY or SELL
            "quantity": quantity,
            "time_in_force": time_in_force or self.default_time_in_force,
            "status": "PENDING_NEW",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        logging.info(f"📝 Created market order: {side} {quantity} {symbol}")
        
        return order
    
    def validate(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a market order
        """
        errors = []
        
        # Check required fields
        required = ["symbol", "quantity", "side"]
        for field in required:
            if field not in order:
                errors.append(f"Missing required field: {field}")
        
        # Validate quantity
        if "quantity" in order and order["quantity"] <= 0:
            errors.append("Quantity must be positive")
        
        # Validate side
        if "side" in order and order["side"] not in ["BUY", "SELL"]:
            errors.append("Side must be BUY or SELL")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        return {"valid": True, "order": order}
    
    def estimate_cost(self, order: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Estimate execution cost including slippage
        """
        quantity = order["quantity"]
        
        # Estimate with slippage
        if order["side"] == "BUY":
            estimated_price = current_price * (1 + self.max_slippage)
            estimated_total = quantity * estimated_price
            slippage_cost = estimated_total - (quantity * current_price)
        else:  # SELL
            estimated_price = current_price * (1 - self.max_slippage)
            estimated_total = quantity * estimated_price
            slippage_cost = (quantity * current_price) - estimated_total
        
        return {
            "current_price": current_price,
            "estimated_price": estimated_price,
            "estimated_total": estimated_total,
            "slippage_cost": slippage_cost,
            "slippage_percent": self.max_slippage * 100,
            "quantity": quantity
        }