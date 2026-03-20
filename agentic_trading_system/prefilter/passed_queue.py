"""
Passed Queue - Manages queue of stocks that passed prefilter
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
from collections import deque
from agentic_trading_system.utils.logger import logger as logging
class PassedQueue:
    """
    Manages queue of stocks that passed all quality gates
    
    Features:
    - FIFO queue management
    - Priority handling
    - Expiry of old items
    - Duplicate prevention
    - Queue statistics
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Queue configuration
        self.max_size = config.get("max_size", 1000)
        self.item_ttl_seconds = config.get("item_ttl_seconds", 3600)  # 1 hour default
        self.max_per_ticker = config.get("max_per_ticker", 5)  # Max queued items per ticker
        
        # Initialize queue
        self.queue = deque(maxlen=self.max_size)
        self.ticker_counts = {}
        self.processing = set()  # Items currently being processed
        
        # Statistics
        self.stats = {
            "total_added": 0,
            "total_removed": 0,
            "total_expired": 0,
            "total_duplicates_prevented": 0
        }
        
        # Start expiry checker
        self._start_expiry_checker()
        
        logging.info(f"✅ PassedQueue initialized with max size {self.max_size}")
    
    async def add(self, item: Dict[str, Any]) -> bool:
        """
        Add an item to the queue
        Returns True if added, False if duplicate or queue full
        """
        ticker = item.get("ticker")
        if not ticker:
            logging.warning("Attempted to add item without ticker")
            return False
        
        # Check if already in queue
        if self._is_in_queue(ticker):
            self.stats["total_duplicates_prevented"] += 1
            logging.debug(f"⏭️ {ticker} already in queue, skipping")
            return False
        
        # Check per-ticker limit
        if self.ticker_counts.get(ticker, 0) >= self.max_per_ticker:
            logging.debug(f"⏭️ {ticker} already has {self.max_per_ticker} items in queue")
            return False
        
        # Add expiry timestamp
        item["queue_added_at"] = datetime.now().isoformat()
        item["queue_expires_at"] = (datetime.now() + timedelta(seconds=self.item_ttl_seconds)).isoformat()
        item["queue_id"] = f"{ticker}_{datetime.now().timestamp()}"
        
        # Add to queue
        self.queue.append(item)
        self.ticker_counts[ticker] = self.ticker_counts.get(ticker, 0) + 1
        self.stats["total_added"] += 1
        
        logging.info(f"➕ Added {ticker} to queue (size: {len(self.queue)})")
        return True
    
    async def get_next(self) -> Optional[Dict[str, Any]]:
        """
        Get the next item from the queue (FIFO)
        """
        while self.queue:
            item = self.queue.popleft()
            ticker = item["ticker"]
            
            # Check if expired
            if self._is_expired(item):
                self.ticker_counts[ticker] = max(0, self.ticker_counts.get(ticker, 0) - 1)
                self.stats["total_expired"] += 1
                logging.debug(f"⏰ Removed expired item for {ticker}")
                continue
            
            # Mark as processing
            self.processing.add(item["queue_id"])
            self.stats["total_removed"] += 1
            
            logging.info(f"➡️ Retrieved {ticker} from queue (remaining: {len(self.queue)})")
            return item
        
        return None
    
    async def get_all(self) -> List[Dict[str, Any]]:
        """
        Get all items in the queue (for inspection)
        """
        # Filter out expired items
        valid_items = [item for item in self.queue if not self._is_expired(item)]
        
        # Update queue if we removed expired items
        if len(valid_items) != len(self.queue):
            self.queue = deque(valid_items, maxlen=self.max_size)
        
        return list(self.queue)
    
    async def remove(self, ticker: str) -> bool:
        """
        Remove a specific ticker from queue
        """
        initial_length = len(self.queue)
        self.queue = deque(
            [item for item in self.queue if item["ticker"] != ticker],
            maxlen=self.max_size
        )
        
        removed = len(self.queue) != initial_length
        if removed:
            self.ticker_counts[ticker] = 0
            logging.info(f"➖ Removed {ticker} from queue")
        
        return removed
    
    async def mark_processed(self, queue_id: str):
        """
        Mark an item as processed (remove from processing set)
        """
        if queue_id in self.processing:
            self.processing.remove(queue_id)
    
    def size(self) -> int:
        """Get current queue size"""
        return len(self.queue)
    
    def is_empty(self) -> bool:
        """Check if queue is empty"""
        return len(self.queue) == 0
    
    def _is_in_queue(self, ticker: str) -> bool:
        """Check if ticker is already in queue"""
        return any(item["ticker"] == ticker for item in self.queue)
    
    def _is_expired(self, item: Dict) -> bool:
        """Check if item has expired"""
        expires_at = datetime.fromisoformat(item["queue_expires_at"])
        return datetime.now() > expires_at
    
    def _start_expiry_checker(self):
        """Start background task to remove expired items"""
        async def check_expiry():
            while True:
                await asyncio.sleep(60)  # Check every minute
                await self._remove_expired()
        
        asyncio.create_task(check_expiry())
    
    async def _remove_expired(self):
        """Remove expired items from queue"""
        initial_length = len(self.queue)
        valid_items = []
        
        for item in self.queue:
            if not self._is_expired(item):
                valid_items.append(item)
            else:
                ticker = item["ticker"]
                self.ticker_counts[ticker] = max(0, self.ticker_counts.get(ticker, 0) - 1)
                self.stats["total_expired"] += 1
                logging.debug(f"⏰ Auto-removed expired item for {ticker}")
        
        if len(valid_items) != initial_length:
            self.queue = deque(valid_items, maxlen=self.max_size)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            **self.stats,
            "current_size": len(self.queue),
            "processing_size": len(self.processing),
            "unique_tickers": len(self.ticker_counts),
            "ticker_counts": dict(self.ticker_counts),
            "max_size": self.max_size,
            "ttl_seconds": self.item_ttl_seconds
        }
    
    async def clear(self):
        """Clear the entire queue"""
        self.queue.clear()
        self.ticker_counts.clear()
        self.processing.clear()
        logging.info("🧹 Cleared passed queue")