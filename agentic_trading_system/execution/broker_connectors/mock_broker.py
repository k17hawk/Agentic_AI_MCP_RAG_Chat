"""
Mock Broker - Simple mock for testing
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from utils.logger import logger as  logging
import random

class MockBroker:
    """
    Mock Broker - Simple mock for unit testing
    
    Features:
    - No actual trading
    - Perfect fills
    - Configurable responses
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Mock state
        self.orders = {}
        self.positions = {}
        self.cash = config.get("initial_cash", 100000.0)
        
        # Response configuration
        self.always_succeed = config.get("always_succeed", True)
        self.fail_rate = config.get("fail_rate", 0.0)  # 0-1, probability of failure
        
        logging.info(f"✅ MockBroker initialized")
    
    async def get_account(self) -> Dict[str, Any]:
        """
        Get mock account information
        """
        return {
            "account_id": "mock_001",
            "currency": "USD",
            "cash": self.cash,
            "portfolio_value": self.cash + sum(p.get("market_value", 0) for p in self.positions.values()),
            "status": "ACTIVE",
            "is_mock": True
        }
    
    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a mock order
        """
        # Simulate random failure
        if not self.always_succeed and random.random() < self.fail_rate:
            return {
                "success": False,
                "error": "Mock random failure",
                "order": order
            }
        
        order_id = str(uuid.uuid4())
        
        # Mock price
        price = order.get("limit_price", 100.0)
        
        order_response = {
            "broker_order_id": order_id,
            "client_order_id": order.get("client_order_id"),
            "symbol": order["symbol"],
            "quantity": order["quantity"],
            "filled_quantity": order["quantity"],
            "side": order["side"],
            "order_type": order.get("order_type", "MARKET"),
            "status": "filled",
            "filled_price": price,
            "filled_at": datetime.now().isoformat(),
            "success": True,
            "is_mock": True
        }
        
        self.orders[order_id] = order_response
        
        # Update mock positions
        symbol = order["symbol"]
        if order["side"] == "BUY":
            if symbol in self.positions:
                self.positions[symbol]["quantity"] += order["quantity"]
                self.positions[symbol]["market_value"] += order["quantity"] * price
            else:
                self.positions[symbol] = {
                    "quantity": order["quantity"],
                    "avg_price": price,
                    "market_value": order["quantity"] * price
                }
            self.cash -= order["quantity"] * price
        else:  # SELL
            if symbol in self.positions:
                self.positions[symbol]["quantity"] -= order["quantity"]
                self.positions[symbol]["market_value"] -= order["quantity"] * price
                if self.positions[symbol]["quantity"] <= 0:
                    del self.positions[symbol]
            self.cash += order["quantity"] * price
        
        logging.info(f"✅ Mock order executed: {order['side']} {order['quantity']} {order['symbol']}")
        
        return order_response
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a mock order
        """
        if order_id in self.orders:
            self.orders[order_id]["status"] = "cancelled"
            return True
        return False
    
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get mock order details
        """
        return self.orders.get(order_id)
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get mock positions
        """
        positions = []
        for symbol, pos in self.positions.items():
            positions.append({
                "symbol": symbol,
                "quantity": pos["quantity"],
                "avg_price": pos["avg_price"],
                "market_value": pos["market_value"],
                "current_price": pos["avg_price"]  # Mock - no price change
            })
        return positions