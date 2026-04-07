"""
Timeout Manager - Handles HITL approval timeouts
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, Callable
from dataclasses import dataclass
from loguru import logger


@dataclass
class PendingApproval:
    """Pending approval request"""
    request_id: str
    ticker: str
    action: str
    timestamp: datetime
    expiry: datetime
    callback: Optional[Callable] = None
    metadata: Dict = None


class TimeoutManager:
    """Manages timeouts for human-in-the-loop approvals"""
    
    def __init__(self, default_timeout_minutes: int = 5):
        self.default_timeout_minutes = default_timeout_minutes
        self.pending_approvals: Dict[str, PendingApproval] = {}
        
        # Start timeout checker (FIXED)
        self._checker_task = None
        self._start_checker()
        
        logger.info(f"✅ TimeoutManager initialized (default timeout: {default_timeout_minutes} minutes)")
    
    def _start_checker(self):
        """Start the timeout checker coroutine properly"""
        try:
            loop = asyncio.get_running_loop()
            self._checker_task = asyncio.create_task(self._check_loop())
            logger.debug("Started timeout checker in running event loop")
        except RuntimeError:
            logger.debug("No running event loop, timeout checker will start when loop is available")
            self._need_checker_start = True
    
    async def ensure_checker_started(self):
        """Ensure timeout checker is started (call this when event loop is running)"""
        if not hasattr(self, '_checker_task') or self._checker_task is None:
            self._checker_task = asyncio.create_task(self._check_loop())
            logger.info("✅ Timeout checker started")
    
    async def _check_loop(self):
        """Background task to check for timeouts"""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                await self._process_timeouts()
            except asyncio.CancelledError:
                logger.info("Timeout checker cancelled")
                break
            except Exception as e:
                logger.error(f"Error in timeout checker: {e}")
    
    async def _process_timeouts(self):
        """Process timed-out approvals"""
        now = datetime.now()
        timed_out = []
        
        for request_id, approval in self.pending_approvals.items():
            if approval.expiry < now:
                timed_out.append(request_id)
                logger.info(f"⏰ Request {request_id} for {approval.ticker} timed out")
                
                # Execute timeout callback if provided
                if approval.callback:
                    try:
                        if asyncio.iscoroutinefunction(approval.callback):
                            await approval.callback(approval.request_id, timeout=True)
                        else:
                            approval.callback(approval.request_id, timeout=True)
                    except Exception as e:
                        logger.error(f"Error in timeout callback: {e}")
        
        # Remove timed out approvals
        for request_id in timed_out:
            del self.pending_approvals[request_id]
    
    async def add_request(
        self,
        request_id: str,
        ticker: str,
        action: str,
        timeout_minutes: Optional[int] = None,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict] = None
    ) -> PendingApproval:
        """Add a pending approval request"""
        timeout = timeout_minutes or self.default_timeout_minutes
        now = datetime.now()
        
        approval = PendingApproval(
            request_id=request_id,
            ticker=ticker,
            action=action,
            timestamp=now,
            expiry=now + timedelta(minutes=timeout),
            callback=callback,
            metadata=metadata or {}
        )
        
        self.pending_approvals[request_id] = approval
        logger.info(f"📝 Added request {request_id} for {ticker} (timeout: {timeout} minutes)")
        
        return approval
    
    async def resolve_request(self, request_id: str, approved: bool) -> bool:
        """Resolve a pending request (approve or reject)"""
        if request_id not in self.pending_approvals:
            logger.warning(f"Request {request_id} not found")
            return False
        
        approval = self.pending_approvals[request_id]
        
        if approved:
            logger.info(f"✅ Request {request_id} for {approval.ticker} approved")
        else:
            logger.info(f"❌ Request {request_id} for {approval.ticker} rejected")
        
        # Execute callback if provided
        if approval.callback:
            try:
                if asyncio.iscoroutinefunction(approval.callback):
                    await approval.callback(request_id, approved=approved)
                else:
                    approval.callback(request_id, approved=approved)
            except Exception as e:
                logger.error(f"Error in resolution callback: {e}")
        
        # Remove from pending
        del self.pending_approvals[request_id]
        return True
    
    async def is_pending(self, request_id: str) -> bool:
        """Check if a request is still pending"""
        return request_id in self.pending_approvals
    
    async def get_pending(self, ticker: Optional[str] = None) -> list:
        """Get pending requests"""
        if ticker:
            return [
                approval for approval in self.pending_approvals.values()
                if approval.ticker == ticker
            ]
        return list(self.pending_approvals.values())
    
    async def get_stats(self) -> Dict:
        """Get timeout manager statistics"""
        now = datetime.now()
        
        return {
            'pending_count': len(self.pending_approvals),
            'default_timeout_minutes': self.default_timeout_minutes,
            'pending_requests': [
                {
                    'request_id': req.request_id,
                    'ticker': req.ticker,
                    'expires_in': (req.expiry - now).total_seconds() / 60
                }
                for req in self.pending_approvals.values()
            ]
        }
    
    async def stop(self):
        """Stop the timeout checker"""
        if self._checker_task:
            self._checker_task.cancel()
            try:
                await self._checker_task
            except asyncio.CancelledError:
                pass
            logger.info("TimeoutManager stopped")