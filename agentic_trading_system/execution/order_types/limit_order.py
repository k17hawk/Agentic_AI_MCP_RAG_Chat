"""
Limit Order - Execute at specified price or better
"""
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from utils.logger import logger as logging

class LimitOrder:
    """
    Limit Order - Executes at specified price or better
    
    Characteristics:
    - Price guaranteed
    - Execution not guaranteed
    - May not fill if price doesn't reach limit
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.default_time_in_force = config.get("default_time_in_force", "DAY")
        self.min_life_seconds = config.get("min_life_seconds", 60)  # Min order lifetime
        
        logging.info(f"✅ LimitOrder initialized")
    
    def create(self, symbol: str, quantity: int, side: str, limit_price: float,
              time_in_force: str = None, client_id: str = None) -> Dict[str, Any]:
        """
        Create a limit order
        """
        order = {
            "order_id": str(uuid.uuid4()),
            "client_order_id": client_id or str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "order_type": "LIMIT",
            "side": side.upper(),
            "quantity": quantity,
            "limit_price": limit_price,
            "time_in_force": time_in_force or self.default_time_in_force,
            "status": "PENDING_NEW",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        logging.info(f"📝 Created limit order: {side} {quantity} {symbol} @ ${limit_price:.2f}")
        
        return order
    
    def validate(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a limit order
        """
        errors = []
        
        # Check required fields
        required = ["symbol", "quantity", "side", "limit_price"]
        for field in required:
            if field not in order:
                errors.append(f"Missing required field: {field}")
        
        # Validate quantity
        if "quantity" in order and order["quantity"] <= 0:
            errors.append("Quantity must be positive")
        
        # Validate side
        if "side" in order and order["side"] not in ["BUY", "SELL"]:
            errors.append("Side must be BUY or SELL")
        
        # Validate price
        if "limit_price" in order and order["limit_price"] <= 0:
            errors.append("Limit price must be positive")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        return {"valid": True, "order": order}
    
    def check_execution(self, order: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Check if limit order should execute
        """
        limit = order["limit_price"]
        
        if order["side"] == "BUY":
            can_execute = current_price <= limit
            execution_price = min(current_price, limit)
        else:  # SELL
            can_execute = current_price >= limit
            execution_price = max(current_price, limit)
        
        return {
            "can_execute": can_execute,
            "current_price": current_price,
            "limit_price": limit,
            "execution_price": execution_price if can_execute else None,
            "savings": (limit - current_price) if order["side"] == "BUY" and can_execute else
                      (current_price - limit) if order["side"] == "SELL" and can_execute else 0
        }
    
    def adjust_limit(self, order: Dict[str, Any], new_limit: float) -> Dict[str, Any]:
        """
        Adjust limit price of an existing order
        """
        order["limit_price"] = new_limit
        order["updated_at"] = datetime.now().isoformat()
        order["modified"] = True
        
        logging.info(f"📝 Adjusted limit for {order['symbol']} to ${new_limit:.2f}")
        
        return order