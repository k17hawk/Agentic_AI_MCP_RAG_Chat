"""
Risk Approved Queue - Manages risk-approved candidates with expiry
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class ApprovedItem:
    """Risk-approved trading candidate"""
    ticker: str
    action: str
    shares: int
    price: float
    stop_loss: float
    take_profit: float
    confidence: float
    timestamp: datetime
    expiry: datetime
    metadata: Dict


class RiskApprovedQueue:
    """Queue for risk-approved items with expiry handling"""
    
    def __init__(self, expiry_minutes: int = 30):
        self.expiry_minutes = expiry_minutes
        self.approved_items: Dict[str, ApprovedItem] = {}
        
        # Start expiry checker (FIXED)
        self._expiry_task = None
        self._start_expiry_checker()
        
        logger.info(f"✅ RiskApprovedQueue initialized (expiry: {expiry_minutes} minutes)")
    
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
            logger.info("✅ Expiry checker started")
    
    async def _check_expiry(self):
        """Background task to remove expired items"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._remove_expired()
            except asyncio.CancelledError:
                logger.info("Expiry checker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in expiry checker: {e}")
    
    async def _remove_expired(self):
        """Remove expired items from queue"""
        now = datetime.now()
        expired = []
        
        for key, item in self.approved_items.items():
            if item.expiry < now:
                expired.append(key)
        
        for key in expired:
            del self.approved_items[key]
            logger.info(f"⏰ Expired: {key}")
        
        if expired:
            logger.debug(f"Removed {len(expired)} expired items, {len(self.approved_items)} remaining")
    
    async def add(self, item: ApprovedItem) -> str:
        """Add an approved item to the queue"""
        key = f"{item.ticker}_{item.timestamp.timestamp()}"
        self.approved_items[key] = item
        logger.info(f"✅ Added {item.ticker} to approved queue (expires at {item.expiry})")
        return key
    
    async def get(self, ticker: str) -> Optional[ApprovedItem]:
        """Get the most recent approved item for a ticker"""
        # Find all items for this ticker
        ticker_items = [
            (key, item) for key, item in self.approved_items.items()
            if item.ticker == ticker
        ]
        
        if not ticker_items:
            return None
        
        # Return the most recent (by timestamp)
        ticker_items.sort(key=lambda x: x[1].timestamp, reverse=True)
        return ticker_items[0][1]
    
    async def remove(self, ticker: str):
        """Remove all items for a ticker"""
        keys_to_remove = [key for key, item in self.approved_items.items() if item.ticker == ticker]
        for key in keys_to_remove:
            del self.approved_items[key]
        
        if keys_to_remove:
            logger.info(f"Removed {len(keys_to_remove)} items for {ticker}")
    
    async def get_all(self) -> List[ApprovedItem]:
        """Get all approved items"""
        return list(self.approved_items.values())
    
    async def get_stats(self) -> Dict:
        """Get queue statistics"""
        return {
            'total_approved': len(self.approved_items),
            'unique_tickers': len(set(item.ticker for item in self.approved_items.values())),
            'expiry_minutes': self.expiry_minutes,
            'items': [
                {
                    'ticker': item.ticker,
                    'action': item.action,
                    'expires_at': item.expiry.isoformat(),
                    'time_remaining': (item.expiry - datetime.now()).total_seconds() / 60
                }
                for item in self.approved_items.values()
            ]
        }
    
    async def stop(self):
        """Stop the expiry checker"""
        if self._expiry_task:
            self._expiry_task.cancel()
            try:
                await self._expiry_task
            except asyncio.CancelledError:
                pass
            logger.info("RiskApprovedQueue stopped")