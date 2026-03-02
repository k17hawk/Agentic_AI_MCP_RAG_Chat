"""
Timeout Manager - Handles timeouts for pending approvals
"""
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import asyncio
from utils.logger import logger as logging

class TimeoutManager:
    """
    Timeout Manager - Handles timeouts for pending approvals
    
    Features:
    - Per-item timeouts
    - Reminder scheduling
    - Escalation paths
    - Auto-rejection on timeout
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Timeout configuration
        self.default_timeout = config.get("default_timeout_seconds", 300)  # 5 minutes
        self.reminder_interval = config.get("reminder_interval_seconds", 60)  # 1 minute
        self.max_reminders = config.get("max_reminders", 3)
        self.escalation_delay = config.get("escalation_delay_seconds", 600)  # 10 minutes
        
        # Trackers
        self.timeouts = {}  # item_id -> timeout_info
        self.reminders = {}  # item_id -> reminder_count
        
        # Callbacks
        self.on_timeout_callback = None
        self.on_reminder_callback = None
        
        # Start checker
        self.checker_task = None
        self._start_checker()
        
        logging.info(f"✅ TimeoutManager initialized")
    
    def register_timeout(self, item_id: str, timeout_seconds: int = None, 
                        data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Register an item for timeout tracking
        """
        timeout = timeout_seconds or self.default_timeout
        now = datetime.now()
        
        timeout_info = {
            "item_id": item_id,
            "data": data or {},
            "created_at": now.isoformat(),
            "timeout_at": (now + timedelta(seconds=timeout)).isoformat(),
            "reminder_sent": 0,
            "reminder_at": (now + timedelta(seconds=self.reminder_interval)).isoformat(),
            "escalated": False,
            "status": "active"
        }
        
        self.timeouts[item_id] = timeout_info
        self.reminders[item_id] = 0
        
        logging.info(f"⏲️ Timeout registered for {item_id} ({timeout}s)")
        
        return timeout_info
    
    async def cancel_timeout(self, item_id: str) -> bool:
        """Cancel timeout tracking for an item"""
        if item_id in self.timeouts:
            self.timeouts[item_id]["status"] = "cancelled"
            del self.timeouts[item_id]
            if item_id in self.reminders:
                del self.reminders[item_id]
            logging.info(f"⏹️ Timeout cancelled for {item_id}")
            return True
        return False
    
    def set_timeout_callback(self, callback: Callable):
        """Set callback for timeout events"""
        self.on_timeout_callback = callback
    
    def set_reminder_callback(self, callback: Callable):
        """Set callback for reminder events"""
        self.on_reminder_callback = callback
    
    def get_remaining_time(self, item_id: str) -> Optional[int]:
        """Get remaining time in seconds for an item"""
        if item_id not in self.timeouts:
            return None
        
        timeout_info = self.timeouts[item_id]
        timeout_at = datetime.fromisoformat(timeout_info["timeout_at"])
        remaining = (timeout_at - datetime.now()).total_seconds()
        
        return max(0, int(remaining))
    
    def _start_checker(self):
        """Start the timeout checker task"""
        async def check_loop():
            while True:
                await asyncio.sleep(1)  # Check every second
                await self._check_timeouts()
        
        self.checker_task = asyncio.create_task(check_loop())
    
    async def _check_timeouts(self):
        """Check for timeouts and reminders"""
        now = datetime.now()
        
        for item_id, timeout_info in list(self.timeouts.items()):
            if timeout_info["status"] != "active":
                continue
            
            timeout_at = datetime.fromisoformat(timeout_info["timeout_at"])
            
            # Check for timeout
            if now >= timeout_at:
                await self._handle_timeout(item_id)
                continue
            
            # Check for reminder
            reminder_at = datetime.fromisoformat(timeout_info["reminder_at"])
            reminder_count = timeout_info["reminder_sent"]
            
            if now >= reminder_at and reminder_count < self.max_reminders:
                await self._send_reminder(item_id)
    
    async def _handle_timeout(self, item_id: str):
        """Handle a timeout event"""
        timeout_info = self.timeouts.get(item_id)
        if not timeout_info:
            return
        
        timeout_info["status"] = "timeout"
        timeout_info["timeout_occurred_at"] = datetime.now().isoformat()
        
        logging.warning(f"⏰ Timeout occurred for {item_id}")
        
        # Call callback if set
        if self.on_timeout_callback:
            try:
                if asyncio.iscoroutinefunction(self.on_timeout_callback):
                    await self.on_timeout_callback(item_id, timeout_info["data"])
                else:
                    self.on_timeout_callback(item_id, timeout_info["data"])
            except Exception as e:
                logging.error(f"Error in timeout callback: {e}")
        
        # Remove from tracking
        del self.timeouts[item_id]
        if item_id in self.reminders:
            del self.reminders[item_id]
    
    async def _send_reminder(self, item_id: str):
        """Send a reminder"""
        timeout_info = self.timeouts.get(item_id)
        if not timeout_info:
            return
        
        # Update reminder info
        reminder_count = timeout_info["reminder_sent"] + 1
        timeout_info["reminder_sent"] = reminder_count
        
        # Schedule next reminder
        next_reminder = datetime.now() + timedelta(seconds=self.reminder_interval)
        timeout_info["reminder_at"] = next_reminder.isoformat()
        
        logging.info(f"🔔 Sending reminder {reminder_count} for {item_id}")
        
        # Call callback if set
        if self.on_reminder_callback:
            try:
                data = {
                    "item_id": item_id,
                    "reminder_count": reminder_count,
                    "max_reminders": self.max_reminders,
                    "data": timeout_info["data"]
                }
                
                if asyncio.iscoroutinefunction(self.on_reminder_callback):
                    await self.on_reminder_callback(data)
                else:
                    self.on_reminder_callback(data)
            except Exception as e:
                logging.error(f"Error in reminder callback: {e}")