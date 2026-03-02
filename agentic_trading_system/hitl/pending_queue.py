"""
Pending Queue - Manages items waiting for human approval
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import uuid
from collections import deque
from utils.logger import logger as logging

class PendingQueue:
    """
    Pending Queue - Manages items waiting for human approval
    
    Features:
    - FIFO queue with priority
    - Per-item expiry
    - Duplicate prevention
    - Status tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Queue configuration
        self.max_size = config.get("max_size", 100)
        self.default_ttl = config.get("default_ttl_seconds", 300)  # 5 minutes
        
        # Priority queues (higher priority = processed first)
        self.high_priority = deque(maxlen=self.max_size)
        self.normal_priority = deque(maxlen=self.max_size)
        self.low_priority = deque(maxlen=self.max_size)
        
        # Indexes for quick lookup
        self.item_index = {}  # item_id -> item
        self.symbol_index = {}  # symbol -> list of item_ids
        
        # Statistics
        self.stats = {
            "total_added": 0,
            "total_processed": 0,
            "total_expired": 0,
            "total_cancelled": 0
        }
        
        logging.info(f"✅ PendingQueue initialized")
    
    async def add(self, item: Dict[str, Any], priority: str = "normal") -> str:
        """
        Add an item to the queue
        Returns item_id
        """
        # Check for duplicate symbol
        symbol = item.get("symbol")
        if symbol and symbol in self.symbol_index:
            existing_ids = self.symbol_index[symbol]
            if existing_ids:
                logging.warning(f"⏭️ {symbol} already in queue, cancelling old")
                for old_id in existing_ids:
                    await self.cancel(old_id)
        
        # Create item
        item_id = str(uuid.uuid4())
        now = datetime.now()
        
        queue_item = {
            "id": item_id,
            "symbol": symbol,
            "data": item,
            "priority": priority,
            "status": "pending",
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=self.default_ttl)).isoformat(),
            "notified": False,
            "reminder_count": 0
        }
        
        # Add to appropriate queue
        if priority == "high":
            self.high_priority.append(queue_item)
        elif priority == "low":
            self.low_priority.append(queue_item)
        else:
            self.normal_priority.append(queue_item)
        
        # Update indexes
        self.item_index[item_id] = queue_item
        if symbol:
            if symbol not in self.symbol_index:
                self.symbol_index[symbol] = []
            self.symbol_index[symbol].append(item_id)
        
        self.stats["total_added"] += 1
        
        logging.info(f"➕ Added to {priority} queue: {symbol or 'unknown'} (ID: {item_id})")
        
        return item_id
    
    async def get_next(self, include_expired: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the next item for processing (highest priority first)
        """
        # Check high priority first
        for queue in [self.high_priority, self.normal_priority, self.low_priority]:
            while queue:
                item = queue[0]  # Peek
                
                # Check if expired
                if not include_expired and self._is_expired(item):
                    queue.popleft()
                    await self._handle_expired(item)
                    continue
                
                # Remove from queue
                queue.popleft()
                
                # Update status
                item["status"] = "processing"
                self.item_index[item["id"]] = item
                
                logging.info(f"➡️ Retrieved from queue: {item.get('symbol')} (ID: {item['id']})")
                return item
        
        return None
    
    async def get_by_id(self, item_id: str) -> Optional[Dict[str, Any]]:
        """Get item by ID"""
        return self.item_index.get(item_id)
    
    async def get_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all items for a symbol"""
        item_ids = self.symbol_index.get(symbol, [])
        items = []
        
        for item_id in item_ids:
            item = self.item_index.get(item_id)
            if item and item["status"] in ["pending", "processing"]:
                items.append(item)
        
        return items
    
    async def mark_processed(self, item_id: str, decision: str, 
                            response: Dict[str, Any] = None) -> bool:
        """Mark an item as processed"""
        item = self.item_index.get(item_id)
        if not item:
            return False
        
        item["status"] = "processed"
        item["processed_at"] = datetime.now().isoformat()
        item["decision"] = decision
        item["response"] = response
        
        self.stats["total_processed"] += 1
        
        # Clean up indexes
        symbol = item.get("symbol")
        if symbol and symbol in self.symbol_index:
            if item_id in self.symbol_index[symbol]:
                self.symbol_index[symbol].remove(item_id)
        
        logging.info(f"✅ Marked as processed: {item.get('symbol')} - {decision}")
        
        return True
    
    async def cancel(self, item_id: str, reason: str = "cancelled") -> bool:
        """Cancel a pending item"""
        item = self.item_index.get(item_id)
        if not item:
            return False
        
        item["status"] = "cancelled"
        item["cancelled_at"] = datetime.now().isoformat()
        item["cancelled_reason"] = reason
        
        self.stats["total_cancelled"] += 1
        
        # Remove from queue if still there
        for queue in [self.high_priority, self.normal_priority, self.low_priority]:
            queue[:] = [i for i in queue if i["id"] != item_id]
        
        logging.info(f"❌ Cancelled: {item.get('symbol')} - {reason}")
        
        return True
    
    async def get_pending_count(self) -> Dict[str, int]:
        """Get count of pending items by priority"""
        return {
            "high": len(self.high_priority),
            "normal": len(self.normal_priority),
            "low": len(self.low_priority),
            "total": len(self.high_priority) + len(self.normal_priority) + len(self.low_priority)
        }
    
    async def get_all_pending(self) -> List[Dict[str, Any]]:
        """Get all pending items"""
        items = []
        for queue in [self.high_priority, self.normal_priority, self.low_priority]:
            items.extend(list(queue))
        return items
    
    def _is_expired(self, item: Dict[str, Any]) -> bool:
        """Check if item has expired"""
        expires_at = datetime.fromisoformat(item["expires_at"])
        return datetime.now() > expires_at
    
    async def _handle_expired(self, item: Dict[str, Any]):
        """Handle expired item"""
        item["status"] = "expired"
        item["expired_at"] = datetime.now().isoformat()
        self.item_index[item["id"]] = item
        
        # Clean up symbol index
        symbol = item.get("symbol")
        if symbol and symbol in self.symbol_index:
            if item["id"] in self.symbol_index[symbol]:
                self.symbol_index[symbol].remove(item["id"])
        
        self.stats["total_expired"] += 1
        logging.info(f"⏰ Expired: {item.get('symbol')} (ID: {item['id']})")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            "high_queue": len(self.high_priority),
            "normal_queue": len(self.normal_priority),
            "low_queue": len(self.low_priority),
            "pending_total": len(self.high_priority) + len(self.normal_priority) + len(self.low_priority),
            "active_items": len(self.item_index)
        }