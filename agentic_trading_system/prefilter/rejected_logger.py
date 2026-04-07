import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from loguru import logger


class RejectedLogger:
    """Logs and manages rejected trading candidates"""
    
    def __init__(self, config: Optional[Dict] = None, retention_days: int = 30):
        self.config = config or {}
        self.retention_days = retention_days
        self.rejected_items: List[Dict] = []
        self.rejected_file = Path("data/rejected_items.json")
        self.rejected_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing rejected items
        self._load_rejected()
        
        # Start cleanup task (FIXED)
        self._cleanup_task = None
        self._start_cleanup()
        
        logger.info("✅ RejectedLogger initialized")
    
    def _start_cleanup(self):
        """Start the cleanup coroutine properly"""
        try:
            # Try to get the current event loop
            loop = asyncio.get_running_loop()
            # We're in an async context, create task
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.debug("Started cleanup task in running event loop")
        except RuntimeError:
            # No running event loop, we'll start later
            logger.debug("No running event loop, cleanup will start when loop is available")
            # Store that we need to start later
            self._need_cleanup_start = True
    
    async def ensure_cleanup_started(self):
        """Ensure cleanup loop is started (call this when event loop is running)"""
        if not hasattr(self, '_cleanup_task') or self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("✅ Cleanup loop started")
    
    async def _cleanup_loop(self):
        """Background task to clean up old rejected items"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_items()
            except asyncio.CancelledError:
                logger.info("Cleanup loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_old_items(self):
        """Remove items older than retention_days"""
        cutoff = datetime.now() - timedelta(days=self.retention_days)
        original_count = len(self.rejected_items)
        
        self.rejected_items = [
            item for item in self.rejected_items
            if datetime.fromisoformat(item.get('timestamp', '2000-01-01')) > cutoff
        ]
        
        if len(self.rejected_items) < original_count:
            await self._save_rejected()
            logger.info(f"Cleaned up {original_count - len(self.rejected_items)} old rejected items")
    
    def _load_rejected(self):
        """Load rejected items from file"""
        if self.rejected_file.exists():
            try:
                with open(self.rejected_file, 'r') as f:
                    self.rejected_items = json.load(f)
                logger.info(f"Loaded {len(self.rejected_items)} rejected items")
            except Exception as e:
                logger.error(f"Error loading rejected items: {e}")
                self.rejected_items = []
    
    async def _save_rejected(self):
        """Save rejected items to file"""
        try:
            with open(self.rejected_file, 'w') as f:
                json.dump(self.rejected_items, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving rejected items: {e}")
    
    async def log_rejection(self, ticker: str, reason: str, metadata: Optional[Dict] = None):
        """Log a rejected candidate"""
        rejection = {
            'ticker': ticker,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.rejected_items.append(rejection)
        await self._save_rejected()
        logger.info(f"📝 Rejected {ticker}: {reason}")
    
    async def is_rejected(self, ticker: str, lookback_hours: int = 24) -> bool:
        """Check if a ticker was recently rejected"""
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        
        for item in self.rejected_items:
            if item['ticker'] == ticker:
                item_time = datetime.fromisoformat(item['timestamp'])
                if item_time > cutoff:
                    return True
        return False
    
    async def get_rejection_history(self, ticker: Optional[str] = None) -> List[Dict]:
        """Get rejection history for a ticker or all"""
        if ticker:
            return [item for item in self.rejected_items if item['ticker'] == ticker]
        return self.rejected_items.copy()
    
    async def clear_rejected(self, ticker: Optional[str] = None):
        """Clear rejected items"""
        if ticker:
            self.rejected_items = [item for item in self.rejected_items if item['ticker'] != ticker]
        else:
            self.rejected_items.clear()
        
        await self._save_rejected()
        logger.info(f"Cleared rejected items for {ticker if ticker else 'all'}")
    
    async def get_stats(self) -> Dict:
        """Get rejection statistics"""
        from collections import Counter
        
        reasons = Counter([item['reason'] for item in self.rejected_items])
        tickers = Counter([item['ticker'] for item in self.rejected_items])
        
        return {
            'total_rejections': len(self.rejected_items),
            'unique_tickers': len(tickers),
            'top_reasons': reasons.most_common(5),
            'top_tickers': tickers.most_common(5),
            'retention_days': self.retention_days
        }
    
    async def stop(self):
        """Stop the cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            logger.info("RejectedLogger stopped")