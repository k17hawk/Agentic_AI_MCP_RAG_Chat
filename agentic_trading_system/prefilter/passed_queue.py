"""
Passed Queue - Manages prefilter-passed candidates with expiry
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class PassedItem:
    """Prefilter-passed trading candidate"""
    ticker: str
    score: float
    timestamp: datetime
    expiry: datetime
    metadata: Dict


class PassedQueue:
    """Queue for prefilter-passed items with expiry handling"""
    
    def __init__(self, expiry_minutes: int = 15):
        self.expiry_minutes = expiry_minutes
        self.passed_items: Dict[str, PassedItem] = {}
        
        # Start expiry checker (FIXED)
        self._expiry_task = None
        self._start_expiry_checker()
        
        logger.info(f"✅ PassedQueue initialized (expiry: {expiry_minutes} minutes)")
    
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
            logger.info("✅ PassedQueue expiry checker started")
    
    async def _check_expiry(self):
        """Background task to remove expired items"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._remove_expired()
            except asyncio.CancelledError:
                logger.info("PassedQueue expiry checker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in expiry checker: {e}")
    
    async def _remove_expired(self):
        """Remove expired items from queue"""
        now = datetime.now()
        expired = []
        
        for key, item in self.passed_items.items():
            if item.expiry < now:
                expired.append(key)
        
        for key in expired:
            del self.passed_items[key]
            logger.debug(f"⏰ Expired: {key}")
        
        if expired:
            logger.debug(f"Removed {len(expired)} expired items, {len(self.passed_items)} remaining")
    
    async def add(self, ticker: str, score: float, metadata: Optional[Dict] = None) -> str:
        """Add a passed item to the queue"""
        key = f"{ticker}_{datetime.now().timestamp()}"
        expiry = datetime.now() + timedelta(minutes=self.expiry_minutes)
        
        item = PassedItem(
            ticker=ticker,
            score=score,
            timestamp=datetime.now(),
            expiry=expiry,
            metadata=metadata or {}
        )
        
        self.passed_items[key] = item
        logger.info(f"✅ Added {ticker} to passed queue (expires at {expiry})")
        return key
    
    async def get(self, ticker: str) -> Optional[PassedItem]:
        """Get the most recent passed item for a ticker"""
        ticker_items = [
            (key, item) for key, item in self.passed_items.items()
            if item.ticker == ticker
        ]
        
        if not ticker_items:
            return None
        
        # Return the most recent
        ticker_items.sort(key=lambda x: x[1].timestamp, reverse=True)
        return ticker_items[0][1]
    
    async def get_all(self) -> List[PassedItem]:
        """Get all passed items"""
        return list(self.passed_items.values())
    
    async def get_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            'total_passed': len(self.passed_items),
            'unique_tickers': len(set(item.ticker for item in self.passed_items.values())),
            'expiry_minutes': self.expiry_minutes
        }
    
    async def stop(self):
        """Stop the expiry checker"""
        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
            logger.info("PassedQueue stopped")