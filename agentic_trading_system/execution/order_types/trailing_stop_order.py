# trailing_stop_order.py - Fixed Version
"""
Trailing Stop Order - Stop that follows price movements
"""
from typing import Dict, Any, Optional
from datetime import datetime
import uuid
from agentic_trading_system.utils.logger import logger as logging

class TrailingStopOrder:
    """
    Trailing Stop Order - Stop price that trails the market price
    
    Types:
    - Percentage-based: Stop trails by fixed percentage
    - Fixed distance: Stop trails by fixed dollar amount
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.default_trail_percent = config.get("default_trail_percent", 2.0)  # 2% trail
        self.default_trail_amount = config.get("default_trail_amount", 0.50)  # $0.50 trail
        self.trail_type = config.get("trail_type", "PERCENT")  # PERCENT or FIXED
        self.activation_percent = config.get("activation_percent", 1.0)  # Activate after 1% profit
        
        logging.info(f"✅ TrailingStopOrder initialized")
    
    def create(self, symbol: str, quantity: int, side: str, entry_price: float,
              trail_value: float = None, trail_type: str = None,
              activation_percent: float = None, client_id: str = None) -> Dict[str, Any]:
        """
        Create a trailing stop order
        """
        if trail_type is None:
            trail_type = self.trail_type
        
        if trail_value is None:
            trail_value = (self.default_trail_percent if trail_type == "PERCENT" 
                          else self.default_trail_amount)
        
        if activation_percent is None:
            activation_percent = self.activation_percent
        
        order = {
            "order_id": str(uuid.uuid4()),
            "client_order_id": client_id or str(uuid.uuid4()),
            "symbol": symbol.upper(),
            "order_type": "TRAILING_STOP",
            "side": side.upper(),
            "quantity": quantity,
            "entry_price": entry_price,
            "trail_type": trail_type,
            "trail_value": trail_value,
            "activation_percent": activation_percent,
            "current_stop": None,  # Initialize as None
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "activated": False,
            "status": "ACTIVE",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        logging.info(f"📝 Created trailing stop: {side} {quantity} {symbol} trail: {trail_value}{'%' if trail_type=='PERCENT' else '$'}")
        
        return order
    
    def update(self, order: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Update trailing stop based on current price
        """
        # Initialize highest/lowest if not set
        if order.get("highest_price") is None:
            order["highest_price"] = current_price
        if order.get("lowest_price") is None:
            order["lowest_price"] = current_price
        
        # Update highest/lowest price
        if order["side"] == "SELL":  # Long position
            if current_price > order["highest_price"]:
                order["highest_price"] = current_price
        else:  # Short position
            if current_price < order["lowest_price"]:
                order["lowest_price"] = current_price
        
        # Check activation
        if not order["activated"]:
            profit_pct = self._calculate_profit_percent(order, current_price)
            if profit_pct >= order["activation_percent"]:
                order["activated"] = True
                logging.info(f"🔓 Trailing stop activated for {order['symbol']} at {profit_pct:.1f}% profit")
        
        # Update stop price if activated
        if order["activated"]:
            new_stop = self._calculate_stop_price(order, current_price)
            
            # Initialize current_stop if None
            if order.get("current_stop") is None:
                order["current_stop"] = new_stop
            else:
                # Stop can only move in favorable direction
                if order["side"] == "SELL":  # Long - stop only moves up
                    if new_stop > order["current_stop"]:
                        order["current_stop"] = new_stop
                else:  # Short - stop only moves down
                    if new_stop < order["current_stop"]:
                        order["current_stop"] = new_stop
        
        order["updated_at"] = datetime.now().isoformat()
        
        return order
    
    def check_trigger(self, order: Dict[str, Any], current_price: float) -> Dict[str, Any]:
        """
        Check if trailing stop should be triggered
        """
        if not order["activated"] or order.get("current_stop") is None:
            return {"triggered": False, "reason": "Not activated or stop not set"}
        
        if order["side"] == "SELL":  # Long position
            triggered = current_price <= order["current_stop"]
        else:  # Short position
            triggered = current_price >= order["current_stop"]
        
        return {
            "triggered": triggered,
            "current_price": current_price,
            "stop_price": order["current_stop"],
            "side": order["side"]
        }
    
    def trigger(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger the trailing stop
        """
        order["triggered"] = True
        order["triggered_at"] = datetime.now().isoformat()
        order["status"] = "TRIGGERED"
        
        # Create market order
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
        
        logging.info(f"🎯 Trailing stop triggered for {order['symbol']} at ${order.get('current_stop', 0):.2f}")
        
        return {
            "trailing_stop": order,
            "market_order": market_order
        }
    
    def _calculate_profit_percent(self, order: Dict[str, Any], current_price: float) -> float:
        """Calculate current profit percentage"""
        entry = order["entry_price"]
        
        if entry <= 0:
            return 0.0
        
        if order["side"] == "SELL":  # Long
            return ((current_price - entry) / entry) * 100
        else:  # Short
            return ((entry - current_price) / entry) * 100
    
    def _calculate_stop_price(self, order: Dict[str, Any], current_price: float) -> float:
        """Calculate new stop price based on trail type"""
        if order["trail_type"] == "PERCENT":
            if order["side"] == "SELL":  # Long
                highest = order.get("highest_price", current_price)
                trail_amount = highest * (order["trail_value"] / 100)
                return highest - trail_amount
            else:  # Short
                lowest = order.get("lowest_price", current_price)
                trail_amount = lowest * (order["trail_value"] / 100)
                return lowest + trail_amount
        else:  # FIXED
            if order["side"] == "SELL":  # Long
                highest = order.get("highest_price", current_price)
                return highest - order["trail_value"]
            else:  # Short
                lowest = order.get("lowest_price", current_price)
                return lowest + order["trail_value"]