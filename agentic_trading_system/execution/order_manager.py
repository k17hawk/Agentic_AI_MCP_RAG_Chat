import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger


class OrderStatus(Enum):
    """Order status states"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """Order object"""
    order_id: str
    ticker: str
    action: str  # BUY or SELL
    quantity: int
    order_type: str  # MARKET, LIMIT, STOP
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    submitted_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    expiry: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


class OrderManager:
    """Manages order lifecycle"""
    
    def __init__(self, default_expiry_minutes: int = 60):
        self.default_expiry_minutes = default_expiry_minutes
        self.orders: Dict[str, Order] = {}
        
        # Start expiry checker (FIXED)
        self._expiry_task = None
        self._start_expiry_checker()
        
        logger.info(f"✅ OrderManager initialized (default expiry: {default_expiry_minutes} minutes)")
    
    def _start_expiry_checker(self):
        """Start the expiry checker coroutine properly"""
        try:
            loop = asyncio.get_running_loop()
            self._expiry_task = asyncio.create_task(self._check_expiry())
            logger.debug("Started expiry checker in running event loop")
        except RuntimeError:
            logger.debug("No running event loop, expiry checker will start when loop is available")
            self._need_expiry_start = True
    
    async def ensure_expiry_started(self):
        """Ensure expiry checker is started (call this when event loop is running)"""
        if not hasattr(self, '_expiry_task') or self._expiry_task is None:
            self._expiry_task = asyncio.create_task(self._check_expiry())
            logger.info("✅ Order expiry checker started")
    
    async def _check_expiry(self):
        """Background task to check for expired orders"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._process_expired_orders()
            except asyncio.CancelledError:
                logger.info("Order expiry checker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in expiry checker: {e}")
    
    async def _process_expired_orders(self):
        """Process expired orders"""
        now = datetime.now()
        expired = []
        
        for order_id, order in self.orders.items():
            if order.expiry and order.expiry < now:
                if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                    order.status = OrderStatus.EXPIRED
                    expired.append(order_id)
                    logger.info(f"⏰ Order {order_id} for {order.ticker} expired")
        
        if expired:
            logger.debug(f"Processed {len(expired)} expired orders")
    
    async def create_order(
        self,
        ticker: str,
        action: str,
        quantity: int,
        order_type: str = "MARKET",
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        expiry_minutes: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Order:
        """Create a new order"""
        order_id = f"{ticker}_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"
        
        expiry = None
        if expiry_minutes or self.default_expiry_minutes:
            expiry_min = expiry_minutes or self.default_expiry_minutes
            expiry = datetime.now() + timedelta(minutes=expiry_min)
        
        order = Order(
            order_id=order_id,
            ticker=ticker,
            action=action.upper(),
            quantity=quantity,
            order_type=order_type.upper(),
            price=price,
            stop_price=stop_price,
            expiry=expiry,
            metadata=metadata or {}
        )
        
        self.orders[order_id] = order
        logger.info(f"📝 Created order {order_id}: {action} {quantity} {ticker} @ {price or 'MARKET'}")
        
        return order
    
    async def update_status(self, order_id: str, status: OrderStatus, **kwargs):
        """Update order status"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        old_status = order.status
        order.status = status
        
        # Update additional fields
        for key, value in kwargs.items():
            if hasattr(order, key):
                setattr(order, key, value)
        
        # Set filled_at if status is FILLED
        if status == OrderStatus.FILLED and not order.filled_at:
            order.filled_at = datetime.now()
        
        logger.info(f"📊 Order {order_id} status: {old_status.value} → {status.value}")
        return True
    
    async def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    async def get_orders(
        self,
        ticker: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        active_only: bool = True
    ) -> List[Order]:
        """Get orders with filters"""
        orders = list(self.orders.values())
        
        if ticker:
            orders = [o for o in orders if o.ticker == ticker]
        
        if status:
            orders = [o for o in orders if o.status == status]
        
        if active_only:
            active_statuses = [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]
            orders = [o for o in orders if o.status in active_statuses]
        
        return orders
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]:
            logger.warning(f"Cannot cancel order {order_id} - status: {order.status.value}")
            return False
        
        order.status = OrderStatus.CANCELLED
        logger.info(f"❌ Cancelled order {order_id}")
        return True
    
    async def get_stats(self) -> Dict:
        """Get order statistics"""
        orders = list(self.orders.values())
        
        return {
            'total_orders': len(orders),
            'by_status': {
                status.value: len([o for o in orders if o.status == status])
                for status in OrderStatus
            },
            'active_orders': len([o for o in orders if o.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]]),
            'filled_orders': len([o for o in orders if o.status == OrderStatus.FILLED]),
            'default_expiry_minutes': self.default_expiry_minutes
        }
    
    async def stop(self):
        """Stop the expiry checker"""
        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
            logger.info("OrderManager stopped")