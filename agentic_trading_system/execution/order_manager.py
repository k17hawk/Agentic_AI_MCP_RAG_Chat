"""
Order Manager - Manages order lifecycle
"""
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timedelta
import uuid
import asyncio
from agentic_trading_system.utils.logger import logger as logging

# Import order types
from agentic_trading_system.execution.order_types.market_order import MarketOrder
from agentic_trading_system.execution.order_types.limit_order import LimitOrder
from agentic_trading_system.execution.order_types.stop_order import StopOrder
from agentic_trading_system.execution.order_types.trailing_stop_order import TrailingStopOrder

class OrderManager:
    """
    Order Manager - Manages the complete order lifecycle
    
    Responsibilities:
    - Create orders
    - Track order status
    - Handle order modifications
    - Manage order expiration
    - Route orders to execution
    - Monitor open orders
    - Handle partial fills
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize order types
        self.market_order = MarketOrder(config.get("market_config", {}))
        self.limit_order = LimitOrder(config.get("limit_config", {}))
        self.stop_order = StopOrder(config.get("stop_config", {}))
        self.trailing_stop = TrailingStopOrder(config.get("trailing_config", {}))
        
        # Order storage
        self.orders: Dict[str, Dict[str, Any]] = {}  # order_id -> order
        self.client_orders: Dict[str, str] = {}  # client_order_id -> order_id
        self.broker_orders: Dict[str, str] = {}  # broker_order_id -> order_id
        
        # Active orders (pending, partially filled)
        self.active_orders: Set[str] = set()
        self.pending_orders: Set[str] = set()  # Orders waiting to be submitted
        self.partially_filled: Set[str] = set()  # Orders with partial fills
        
        # Order history
        self.order_history: List[Dict[str, Any]] = []
        self.max_history = config.get("max_history", 1000)
        
        # Execution engine reference (set later)
        self.execution_engine = None
        
        # Expiry checking
        self.expiry_check_interval = config.get("expiry_check_interval", 60)  # seconds
        self._expiry_task = None
        self._start_expiry_checker()
        
        logging.info(f"✅ OrderManager initialized")
    
    def set_execution_engine(self, execution_engine):
        """Set execution engine reference"""
        self.execution_engine = execution_engine
    
    def _start_expiry_checker(self):
        """Start background task to check for expired orders"""
        async def check_expiry():
            while True:
                await asyncio.sleep(self.expiry_check_interval)
                await self._check_expired_orders()
        
        self._expiry_task = asyncio.create_task(check_expiry())
    
    async def _check_expired_orders(self):
        """Check for and handle expired orders"""
        now = datetime.now()
        expired = []
        
        for order_id in self.active_orders.copy():
            order = self.orders.get(order_id)
            if not order:
                continue
            
            # Check time-in-force expiration
            tif = order.get("time_in_force", "DAY")
            
            if tif == "DAY":
                # Day orders expire at market close - simplified check
                # In production, would check actual market hours
                order_time = datetime.fromisoformat(order["created_at"])
                if (now - order_time).total_seconds() > 86400:  # 24 hours
                    expired.append(order_id)
            
            elif tif == "IOC" or tif == "FOK":
                # Immediate or Cancel / Fill or Kill - expire quickly
                order_time = datetime.fromisoformat(order["created_at"])
                if (now - order_time).total_seconds() > 60:  # 60 seconds
                    expired.append(order_id)
            
            elif tif == "GTD":
                # Good Till Date
                expire_time = datetime.fromisoformat(order.get("expire_time", order["created_at"]))
                if now > expire_time:
                    expired.append(order_id)
        
        # Cancel expired orders
        for order_id in expired:
            await self.cancel_order(order_id, reason="expired")
            logging.info(f"⏰ Order expired: {order_id}")
    
    def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new order
        """
        order_type = order_data.get("order_type", "MARKET").upper()
        client_order_id = order_data.get("client_order_id", str(uuid.uuid4()))
        
        # Validate required fields
        required_fields = ["symbol", "quantity", "side"]
        for field in required_fields:
            if field not in order_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Create order based on type
        if order_type == "MARKET":
            order = self.market_order.create(
                symbol=order_data["symbol"],
                quantity=order_data["quantity"],
                side=order_data["side"],
                time_in_force=order_data.get("time_in_force"),
                client_id=client_order_id
            )
        elif order_type == "LIMIT":
            if "limit_price" not in order_data:
                raise ValueError("Limit price required for LIMIT order")
            order = self.limit_order.create(
                symbol=order_data["symbol"],
                quantity=order_data["quantity"],
                side=order_data["side"],
                limit_price=order_data["limit_price"],
                time_in_force=order_data.get("time_in_force"),
                client_id=client_order_id
            )
        elif order_type == "STOP":
            if "stop_price" not in order_data:
                raise ValueError("Stop price required for STOP order")
            order = self.stop_order.create(
                symbol=order_data["symbol"],
                quantity=order_data["quantity"],
                side=order_data["side"],
                stop_price=order_data["stop_price"],
                stop_type=order_data.get("stop_type"),
                time_in_force=order_data.get("time_in_force"),
                client_id=client_order_id
            )
        elif order_type == "TRAILING_STOP":
            order = self.trailing_stop.create(
                symbol=order_data["symbol"],
                quantity=order_data["quantity"],
                side=order_data["side"],
                entry_price=order_data.get("entry_price", 0),
                trail_value=order_data.get("trail_value"),
                trail_type=order_data.get("trail_type"),
                activation_percent=order_data.get("activation_percent"),
                client_id=client_order_id
            )
        else:
            raise ValueError(f"Unknown order type: {order_type}")
        
        # Add additional fields
        order.update({
            "status": "PENDING_NEW",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "filled_quantity": 0,
            "fills": [],
            "notes": order_data.get("notes", ""),
            "tags": order_data.get("tags", [])
        })
        
        # Store order
        order_id = order["order_id"]
        self.orders[order_id] = order
        self.client_orders[client_order_id] = order_id
        self.pending_orders.add(order_id)
        
        logging.info(f"📝 Order created: {order_id} - {order['side']} {order['quantity']} {order['symbol']} ({order_type})")
        
        return order
    
    async def submit_order(self, order_id: str) -> Dict[str, Any]:
        """
        Submit an order for execution
        """
        if order_id not in self.orders:
            return {"success": False, "error": "Order not found"}
        
        order = self.orders[order_id]
        
        if order["status"] not in ["PENDING_NEW", "REJECTED", "SUSPENDED"]:
            return {
                "success": False, 
                "error": f"Order cannot be submitted in status: {order['status']}"
            }
        
        # Update status
        order["status"] = "SUBMITTED"
        order["submitted_at"] = datetime.now().isoformat()
        order["updated_at"] = datetime.now().isoformat()
        
        if order_id in self.pending_orders:
            self.pending_orders.remove(order_id)
        
        # Validate order
        validation = self._validate_order(order)
        if not validation["valid"]:
            order["status"] = "REJECTED"
            order["reject_reason"] = validation["errors"]
            logging.warning(f"❌ Order {order_id} rejected: {validation['errors']}")
            return {
                "success": False,
                "error": validation["errors"],
                "order_id": order_id
            }
        
        # Submit to execution engine
        if self.execution_engine:
            result = await self.execution_engine.execute_order(order)
            
            if result.get("success"):
                order["broker_order_id"] = result.get("broker_order_id")
                if result.get("broker_order_id"):
                    self.broker_orders[result["broker_order_id"]] = order_id
                
                if result.get("status") == "filled":
                    order["status"] = "FILLED"
                    # Add fill data
                    if "filled_price" in result:
                        self._add_fill(order_id, {
                            "quantity": order["quantity"],
                            "price": result["filled_price"],
                            "time": datetime.now().isoformat()
                        })
                else:
                    order["status"] = "ACCEPTED"
                    self.active_orders.add(order_id)
                
                logging.info(f"✅ Order {order_id} submitted successfully")
            else:
                order["status"] = "REJECTED"
                order["reject_reason"] = result.get("error", "Unknown error")
                logging.warning(f"❌ Order {order_id} rejected: {order['reject_reason']}")
            
            return result
        else:
            # Mock execution
            order["status"] = "ACCEPTED"
            order["broker_order_id"] = f"broker_{datetime.now().timestamp()}"
            self.broker_orders[order["broker_order_id"]] = order_id
            self.active_orders.add(order_id)
            
            logging.info(f"✅ Order {order_id} accepted (mock)")
            
            return {
                "success": True,
                "order_id": order_id,
                "broker_order_id": order["broker_order_id"],
                "status": "ACCEPTED"
            }
    
    async def cancel_order(self, order_id: str, reason: str = "cancelled") -> bool:
        """
        Cancel an active order
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order["status"] in ["FILLED", "CANCELLED", "REJECTED", "EXPIRED"]:
            return False
        
        # Cancel in execution engine if submitted
        if self.execution_engine and "broker_order_id" in order:
            cancelled = await self.execution_engine.cancel_order(order["broker_order_id"])
            if not cancelled:
                logging.warning(f"⚠️ Failed to cancel order {order_id} at broker")
                # Still mark as cancelled locally? This is a decision point
        
        # Update status
        old_status = order["status"]
        order["status"] = "CANCELLED"
        order["cancelled_at"] = datetime.now().isoformat()
        order["cancelled_reason"] = reason
        order["updated_at"] = datetime.now().isoformat()
        
        # Remove from active sets
        self.active_orders.discard(order_id)
        self.pending_orders.discard(order_id)
        self.partially_filled.discard(order_id)
        
        # Add to history
        self._add_to_history(order)
        
        logging.info(f"❌ Order cancelled: {order_id} (was {old_status}) - {reason}")
        
        return True
    
    def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Modify an existing order
        """
        if order_id not in self.orders:
            return None
        
        order = self.orders[order_id]
        
        if order["status"] not in ["PENDING_NEW", "SUBMITTED", "ACCEPTED"]:
            logging.warning(f"⚠️ Cannot modify order {order_id} in status {order['status']}")
            return None
        
        # Track changes
        changes = {}
        
        # Modify based on order type
        if order["order_type"] == "LIMIT" and "limit_price" in modifications:
            old_price = order.get("limit_price")
            new_price = modifications["limit_price"]
            if new_price != old_price:
                updated_order = self.limit_order.adjust_limit(order, new_price)
                order.update(updated_order)
                changes["limit_price"] = {"old": old_price, "new": new_price}
        
        elif order["order_type"] == "STOP" and "stop_price" in modifications:
            old_stop = order.get("stop_price")
            new_stop = modifications["stop_price"]
            if new_stop != old_stop:
                order["stop_price"] = new_stop
                changes["stop_price"] = {"old": old_stop, "new": new_stop}
        
        # Common modifications
        if "quantity" in modifications:
            old_qty = order["quantity"]
            new_qty = modifications["quantity"]
            if new_qty != old_qty:
                order["quantity"] = new_qty
                changes["quantity"] = {"old": old_qty, "new": new_qty}
        
        if "time_in_force" in modifications:
            old_tif = order.get("time_in_force")
            new_tif = modifications["time_in_force"]
            if new_tif != old_tif:
                order["time_in_force"] = new_tif
                changes["time_in_force"] = {"old": old_tif, "new": new_tif}
        
        if "trail_value" in modifications and order["order_type"] == "TRAILING_STOP":
            old_trail = order.get("trail_value")
            new_trail = modifications["trail_value"]
            if new_trail != old_trail:
                order["trail_value"] = new_trail
                changes["trail_value"] = {"old": old_trail, "new": new_trail}
        
        if changes:
            order["modified_at"] = datetime.now().isoformat()
            order["modified"] = True
            order["updated_at"] = datetime.now().isoformat()
            logging.info(f"📝 Order modified: {order_id} - {changes}")
        
        return order
    
    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by client order ID"""
        order_id = self.client_orders.get(client_order_id)
        if order_id:
            return self.orders.get(order_id)
        return None
    
    def get_order_by_broker_id(self, broker_order_id: str) -> Optional[Dict[str, Any]]:
        """Get order by broker order ID"""
        order_id = self.broker_orders.get(broker_order_id)
        if order_id:
            return self.orders.get(order_id)
        return None
    
    def get_active_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get all active orders, optionally filtered by symbol"""
        active = []
        for order_id in self.active_orders:
            order = self.orders.get(order_id)
            if order:
                if not symbol or order["symbol"] == symbol:
                    active.append(order)
        return active
    
    def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending (not yet submitted) orders"""
        return [self.orders[oid] for oid in self.pending_orders if oid in self.orders]
    
    def get_partially_filled_orders(self) -> List[Dict[str, Any]]:
        """Get all partially filled orders"""
        return [self.orders[oid] for oid in self.partially_filled if oid in self.orders]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get orders for a specific symbol"""
        return [o for o in self.orders.values() if o["symbol"] == symbol]
    
    def get_orders_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get orders by status"""
        return [o for o in self.orders.values() if o.get("status") == status]
    
    def get_orders_by_date(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """Get orders within a date range"""
        orders = []
        for order in self.orders.values():
            created = datetime.fromisoformat(order["created_at"])
            if start_date <= created <= end_date:
                orders.append(order)
        return orders
    
    def update_order_status(self, order_id: str, status: str, 
                           fill_data: Dict[str, Any] = None) -> bool:
        """
        Update order status (called by execution engine)
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        old_status = order["status"]
        
        # Validate status transition
        if not self._is_valid_transition(old_status, status):
            logging.warning(f"⚠️ Invalid status transition: {old_status} -> {status}")
            return False
        
        order["status"] = status
        order["updated_at"] = datetime.now().isoformat()
        
        if fill_data:
            self._add_fill(order_id, fill_data)
        
        # Update active sets
        if status in ["FILLED", "CANCELLED", "REJECTED", "EXPIRED"]:
            self.active_orders.discard(order_id)
            self.partially_filled.discard(order_id)
            self._add_to_history(order)
        elif status == "PARTIALLY_FILLED":
            self.active_orders.add(order_id)
            self.partially_filled.add(order_id)
        elif status in ["ACCEPTED", "WORKING"]:
            self.active_orders.add(order_id)
            self.partially_filled.discard(order_id)
        
        logging.info(f"📊 Order {order_id} status updated: {old_status} -> {status}")
        
        return True
    
    def _add_fill(self, order_id: str, fill_data: Dict[str, Any]):
        """Add a fill to an order"""
        order = self.orders[order_id]
        
        if "fills" not in order:
            order["fills"] = []
        
        fill = {
            "fill_id": f"fill_{datetime.now().timestamp()}",
            "quantity": fill_data.get("quantity", 0),
            "price": fill_data.get("price", 0),
            "value": fill_data.get("quantity", 0) * fill_data.get("price", 0),
            "time": fill_data.get("time", datetime.now().isoformat()),
            "broker_fill_id": fill_data.get("broker_fill_id")
        }
        
        order["fills"].append(fill)
        
        # Update filled quantity
        order["filled_quantity"] = order.get("filled_quantity", 0) + fill["quantity"]
        
        # Calculate average fill price
        if order["filled_quantity"] > 0:
            total_value = sum(f["value"] for f in order["fills"])
            order["avg_fill_price"] = total_value / order["filled_quantity"]
        
        # Check if fully filled
        if order["filled_quantity"] >= order["quantity"]:
            order["status"] = "FILLED"
            self.active_orders.discard(order_id)
            self.partially_filled.discard(order_id)
            self._add_to_history(order)
        elif order["filled_quantity"] > 0:
            order["status"] = "PARTIALLY_FILLED"
            self.partially_filled.add(order_id)
    
    def check_stop_orders(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Check stop orders against current prices
        """
        triggered = []
        
        for order_id in self.active_orders.copy():
            order = self.orders.get(order_id)
            if not order:
                continue
            
            if order["order_type"] not in ["STOP", "TRAILING_STOP"]:
                continue
            
            symbol = order["symbol"]
            if symbol not in current_prices:
                continue
            
            current_price = current_prices[symbol]
            
            if order["order_type"] == "STOP":
                check = self.stop_order.check_trigger(order, current_price)
                if check["triggered"]:
                    result = self.stop_order.trigger(order)
                    
                    # Create market order from triggered stop
                    market_order_data = {
                        "symbol": symbol,
                        "quantity": order["quantity"],
                        "side": order["side"],
                        "order_type": "MARKET",
                        "parent_order_id": order_id
                    }
                    
                    market_order = self.create_order(market_order_data)
                    
                    triggered.append({
                        "stop_order_id": order_id,
                        "market_order_id": market_order["order_id"],
                        "trigger_price": current_price,
                        "stop_price": order["stop_price"],
                        "result": result
                    })
                    
                    # Cancel the stop order
                    asyncio.create_task(self.cancel_order(order_id, reason="triggered"))
                    
            elif order["order_type"] == "TRAILING_STOP":
                # Update trailing stop
                updated = self.trailing_stop.update(order, current_price)
                self.orders[order_id] = updated
                
                # Check trigger
                check = self.trailing_stop.check_trigger(order, current_price)
                if check["triggered"]:
                    result = self.trailing_stop.trigger(order)
                    
                    # Create market order
                    market_order_data = {
                        "symbol": symbol,
                        "quantity": order["quantity"],
                        "side": order["side"],
                        "order_type": "MARKET",
                        "parent_order_id": order_id
                    }
                    
                    market_order = self.create_order(market_order_data)
                    
                    triggered.append({
                        "stop_order_id": order_id,
                        "market_order_id": market_order["order_id"],
                        "trigger_price": current_price,
                        "stop_price": order["current_stop"],
                        "result": result
                    })
                    
                    asyncio.create_task(self.cancel_order(order_id, reason="triggered"))
        
        return triggered
    
    def _validate_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an order before submission"""
        errors = []
        
        # Validate based on order type
        if order["order_type"] == "MARKET":
            validation = self.market_order.validate(order)
            if not validation.get("valid"):
                errors.extend(validation.get("errors", []))
        
        elif order["order_type"] == "LIMIT":
            validation = self.limit_order.validate(order)
            if not validation.get("valid"):
                errors.extend(validation.get("errors", []))
        
        elif order["order_type"] == "STOP":
            validation = self.stop_order.validate(order)
            if not validation.get("valid"):
                errors.extend(validation.get("errors", []))
        
        # Common validations
        if order["quantity"] <= 0:
            errors.append("Quantity must be positive")
        
        if order["side"] not in ["BUY", "SELL"]:
            errors.append("Side must be BUY or SELL")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    def _is_valid_transition(self, from_status: str, to_status: str) -> bool:
        """Check if status transition is valid"""
        valid_transitions = {
            "PENDING_NEW": ["SUBMITTED", "CANCELLED", "REJECTED"],
            "SUBMITTED": ["ACCEPTED", "REJECTED", "CANCELLED"],
            "ACCEPTED": ["WORKING", "PARTIALLY_FILLED", "CANCELLED", "REJECTED"],
            "WORKING": ["PARTIALLY_FILLED", "FILLED", "CANCELLED"],
            "PARTIALLY_FILLED": ["FILLED", "CANCELLED", "WORKING"],
            "FILLED": [],
            "CANCELLED": [],
            "REJECTED": [],
            "EXPIRED": []
        }
        
        return to_status in valid_transitions.get(from_status, [])
    
    def get_order_summary(self) -> Dict[str, Any]:
        """
        Get order summary statistics
        """
        total_orders = len(self.orders)
        active_count = len(self.active_orders)
        
        status_counts = {}
        type_counts = {}
        side_counts = {"BUY": 0, "SELL": 0}
        total_volume = 0
        total_value = 0
        
        for order in self.orders.values():
            status = order.get("status", "UNKNOWN")
            status_counts[status] = status_counts.get(status, 0) + 1
            
            order_type = order.get("order_type", "UNKNOWN")
            type_counts[order_type] = type_counts.get(order_type, 0) + 1
            
            side = order.get("side")
            if side in side_counts:
                side_counts[side] += 1
            
            if order.get("filled_quantity", 0) > 0:
                total_volume += order["filled_quantity"]
                if "avg_fill_price" in order:
                    total_value += order["filled_quantity"] * order["avg_fill_price"]
        
        return {
            "total_orders": total_orders,
            "active_orders": active_count,
            "pending_orders": len(self.pending_orders),
            "partially_filled": len(self.partially_filled),
            "status_breakdown": status_counts,
            "type_breakdown": type_counts,
            "side_breakdown": side_counts,
            "total_filled_volume": total_volume,
            "total_filled_value": total_value,
            "history_count": len(self.order_history)
        }
    
    def get_order_report(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed report for a specific order"""
        order = self.orders.get(order_id)
        if not order:
            return None
        
        return {
            "order": order,
            "execution_summary": {
                "filled_quantity": order.get("filled_quantity", 0),
                "remaining_quantity": order["quantity"] - order.get("filled_quantity", 0),
                "fill_percentage": (order.get("filled_quantity", 0) / order["quantity"]) * 100 if order["quantity"] > 0 else 0,
                "avg_fill_price": order.get("avg_fill_price"),
                "fills": order.get("fills", [])
            },
            "timeline": {
                "created": order["created_at"],
                "submitted": order.get("submitted_at"),
                "accepted": order.get("accepted_at"),
                "filled": order.get("filled_at"),
                "cancelled": order.get("cancelled_at")
            }
        }
    
    def _add_to_history(self, order: Dict[str, Any]):
        """Add completed order to history"""
        history_entry = order.copy()
        history_entry["archived_at"] = datetime.now().isoformat()
        self.order_history.append(history_entry)
        
        # Trim if needed
        if len(self.order_history) > self.max_history:
            self.order_history = self.order_history[-self.max_history:]
    
    async def stop(self):
        """Graceful shutdown - cancel all active orders"""
        if self._expiry_task:
            self._expiry_task.cancel()
        
        # Cancel all active orders
        for order_id in self.active_orders.copy():
            await self.cancel_order(order_id, reason="shutdown")
        
        logging.info("🛑 OrderManager stopped")