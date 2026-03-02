"""
Stop Order - Becomes market order when stop price is triggered
"""
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from utils.logger import logger as logging

class StopOrder:
    """
    Stop Order (Stop Loss) - Becomes market order when stop price is triggered
    
    Types:
    - STOP_LOSS: Sell when price falls below stop
    - STOP_ENTRY: Buy when price rises above stop
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.default_time_in_force = config.get("default_time_in_force", "GTC")
        self.stop_type = config.get("stop_type", "STOP_LOSS")  # STOP_LOSS or STOP_ENTRY
        
        logging.info(f"✅ StopOrder initialized")
    
    def create(self, symbol: str, quantity: int, side: str, stop_price: float,
              stop_type: str = None, time_in_force: str = None, 
              client_id: str = None) -> Dict[str, Any]:
        """
        Create a stop order
        """
        if stop_type is None:
            stop_type = self.stop_type
        
        order = {
            "order_id": str(uuid.uuid4()),
            "client_order_id": client_id or str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "order_type": "STOP",
            "stop_type": stop_type,
            "side": side.upper(),
            "quantity": quantity,
            "stop_price": stop_price,
            "time_in_force": time_in_force or self.default_time_in_force,
            "status": "PENDING_NEW",
            "triggered": False,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        logging.info(f"📝 Created stop order: {stop_type} {side} {quantity} {symbol} @ ${stop_price:.2f}")
        
        return order
    
    def validate(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a stop order
        """
        errors = []
        
        # Check required fields
        required = ["symbol", "quantity", "side", "stop_price", "stop_type"]
        for field in required:
            if field not in order:
                errors.append(f"Missing required field: {field}")
        
        # Validate quantity
        if "quantity" in order and order["quantity"] <= 0:
            errors.append("Quantity must be positive")
        
        # Validate side
        if "side" in order and order["side"] not in ["BUY", "SELL"]:
            errors.append("Side must be BUY or SELL")
        
        # Validate stop price
        if "stop_price" in order and order["stop_price"] <= 0:
            errors.append("Stop price must be positive")
        
        # Validate stop type
        if "stop_type" in order and order["stop_type"] not in ["STOP_LOSS", "STOP_ENTRY"]:
            errors.append("Stop type must be STOP_LOSS or STOP_ENTRY")
        
        if errors:
            return {"valid": False, "errors": errors}
        
        return {"valid": True, "order": order}
    
    def check_trigger(self, order: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Check if stop order should be triggered
        """
        stop = order["stop_price"]
        stop_type = order["stop_type"]
        
        if stop_type == "STOP_LOSS":
            if order["side"] == "SELL":  # Long position stop loss
                triggered = current_price <= stop
            else:  # Short position stop loss
                triggered = current_price >= stop
        else:  # STOP_ENTRY
            if order["side"] == "BUY":  # Buy breakout
                triggered = current_price >= stop
            else:  # Sell breakdown
                triggered = current_price <= stop
        
        return {
            "triggered": triggered,
            "current_price": current_price,
            "stop_price": stop,
            "stop_type": stop_type,
            "side": order["side"]
        }
    
    def trigger(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger the stop order (convert to market order)
        """
        order["triggered"] = True
        order["triggered_at"] = datetime.now().isoformat()
        order["status"] = "TRIGGERED"
        order["updated_at"] = datetime.now().isoformat()
        
        # Create market order from triggered stop
        market_order = {
            "order_id": f"MKT_{order['order_id']}",
            "parent_order_id": order["order_id"],
            "symbol": order["symbol"],
            "order_type": "MARKET",
            "side": order["side"],
            "quantity": order["quantity"],
            "status": "PENDING_NEW",
            "created_at": datetime.now().isoformat()
        }
        
        logging.info(f"🎯 Stop order triggered: {order['symbol']} at ${order['stop_price']:.2f}")
        
        return {
            "stop_order": order,
            "market_order": market_order
        }