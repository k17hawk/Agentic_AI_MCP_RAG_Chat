"""
Risk Approved Queue - Queue of trades that passed risk checks
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from collections import deque
import heapq

from utils.logger import logger as  logging

class RiskApprovedQueue:
    """
    Priority queue for trades that passed risk checks
    
    Features:
    - Priority based on risk score (lower risk = higher priority)
    - Expiry of old items
    - Duplicate prevention
    - Queue statistics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Queue configuration
        self.max_size = config.get("max_size", 500)
        self.item_ttl_seconds = config.get("item_ttl_seconds", 300)  # 5 minutes default
        
        # Priority queue (min-heap based on risk score)
        self.queue = []  # List of (risk_score, timestamp, item)
        self.ticker_set = set()  # For quick duplicate checking
        
        # Statistics
        self.stats = {
            "total_added": 0,
            "total_removed": 0,
            "total_expired": 0,
            "total_processed": 0
        }
        
        # Start expiry checker
        self._start_expiry_checker()
        
        logging.info(f"✅ RiskApprovedQueue initialized with max size {self.max_size}")
    
    async def add(self, trade: Dict[str, Any]) -> bool:
        """
        Add a trade to the queue
        Returns True if added, False if duplicate or queue full
        """
        ticker = trade.get("ticker")
        if not ticker:
            logging.warning("Attempted to add trade without ticker")
            return False
        
        # Check for duplicates
        if ticker in self.ticker_set:
            self.stats["total_duplicates"] = self.stats.get("total_duplicates", 0) + 1
            logging.debug(f"⏭️ {ticker} already in queue, skipping")
            return False
        
        # Get risk score (lower is better)
        risk_score = trade.get("risk_score", 0.5)
        
        # Add timestamp
        timestamp = datetime.now()
        expiry = timestamp + timedelta(seconds=self.item_ttl_seconds)
        
        # Add to priority queue (negative risk_score for min-heap)
        heapq.heappush(self.queue, (risk_score, timestamp, trade))
        self.ticker_set.add(ticker)
        
        self.stats["total_added"] += 1
        
        # Trim if needed
        await self._trim_queue()
        
        logging.info(f"➕ Added {ticker} to risk queue (risk: {risk_score:.2f}, size: {len(self.queue)})")
        return True
    
    async def get_next(self) -> Optional[Dict[str, Any]]:
        """
        Get the next trade from the queue (lowest risk first)
        """
        while self.queue:
            # Peek at next item
            risk_score, timestamp, trade = self.queue[0]
            
            # Check if expired
            if datetime.now() > timestamp + timedelta(seconds=self.item_ttl_seconds):
                heapq.heappop(self.queue)
                self.ticker_set.discard(trade.get("ticker"))
                self.stats["total_expired"] += 1
                continue
            
            # Get the item
            heapq.heappop(self.queue)
            self.ticker_set.discard(trade.get("ticker"))
            self.stats["total_removed"] += 1
            
            logging.info(f"➡️ Retrieved {trade.get('ticker')} from risk queue (risk: {risk_score:.2f})")
            return trade
        
        return None
    
    async def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all items in the queue (sorted by risk)
        """
        # Create a copy and sort
        items = [(score, ts, trade) for score, ts, trade in self.queue]
        items.sort(key=lambda x: x[0])  # Sort by risk score
        
        # Filter expired
        now = datetime.now()
        valid_items = []
        
        for score, ts, trade in items:
            if now <= ts + timedelta(seconds=self.item_ttl_seconds):
                valid_items.append(trade)
        
        return valid_items
    
    async def remove(self, ticker: str) -> bool:
        """
        Remove a specific ticker from queue
        """
        # Rebuild queue without the ticker
        new_queue = []
        removed = False
        
        for score, ts, trade in self.queue:
            if trade.get("ticker") != ticker:
                new_queue.append((score, ts, trade))
            else:
                removed = True
                self.ticker_set.discard(ticker)
        
        if removed:
            # Re-heapify
            self.queue = new_queue
            heapq.heapify(self.queue)
            self.stats["total_removed"] += 1
            logging.info(f"➖ Removed {ticker} from risk queue")
        
        return removed
    
    def size(self) -> int:
        """Get current queue size"""
        return len(self.queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.queue) == 0
    
    async def _trim_queue(self):
        """Trim queue to max size"""
        if len(self.queue) <= self.max_size:
            return
        
        # Remove oldest/highest risk items
        excess = len(self.queue) - self.max_size
        
        # Sort by timestamp (oldest first) and risk score (highest first)
        items = [(ts, score, trade) for score, ts, trade in self.queue]
        items.sort(key=lambda x: (x[0], -x[1]))  # Oldest first, then highest risk
        
        # Remove excess
        for i in range(excess):
            ts, score, trade = items[i]
            self.ticker_set.discard(trade.get("ticker"))
            self.stats["total_removed"] += 1
        
        # Rebuild queue
        self.queue = [(score, ts, trade) for ts, score, trade in items[excess:]]
        heapq.heapify(self.queue)
    
    def _start_expiry_checker(self):
        """Start background task to remove expired items"""
        async def check_expiry():
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._remove_expired()
        
        asyncio.create_task(check_expiry())
    
    async def _remove_expired(self):
        """Remove expired items from queue"""
        now = datetime.now()
        expired_count = 0
        
        # Rebuild queue without expired items
        new_queue = []
        
        for score, ts, trade in self.queue:
            if now <= ts + timedelta(seconds=self.item_ttl_seconds):
                new_queue.append((score, ts, trade))
            else:
                self.ticker_set.discard(trade.get("ticker"))
                expired_count += 1
        
        if expired_count > 0:
            self.queue = new_queue
            heapq.heapify(self.queue)
            self.stats["total_expired"] += expired_count
            logging.debug(f"⏰ Removed {expired_count} expired items from risk queue")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            "current_size": len(self.queue),
            "unique_tickers": len(self.ticker_set),
            "max_size": self.max_size,
            "ttl_seconds": self.item_ttl_seconds,
            "utilization": len(self.queue) / self.max_size if self.max_size > 0 else 0
        }
    
    async def clear(self):
        """Clear the entire queue"""
        self.queue = []
        self.ticker_set.clear()
        logging.info("🧹 Cleared risk approved queue")