"""
Time Stop - Exit position after specified time period
"""
from typing import Dict, Any
from datetime import datetime, timedelta
from agentic_trading_system.utils.logger import logger as logging

class TimeStop:
    """
    Time-based stop loss - Exit after holding for specified time
    
    Useful for:
    - Avoiding holding through events
    - Time-based mean reversion strategies
    - Earnings plays
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default time periods (in hours)
        self.default_max_hold_hours = config.get("default_max_hold_hours", 24)  # 1 day
        self.short_term_hours = config.get("short_term_hours", 4)  # 4 hours
        self.swing_hours = config.get("swing_hours", 72)  # 3 days
        self.position_hours = config.get("position_hours", 168)  # 1 week
        
        logging.info(f"✅ TimeStop initialized")
    
    def check_expiry(self, entry_time: datetime, current_time: datetime = None,
                    max_hold_hours: float = None) -> Dict[str, Any]:
        """
        Check if position has exceeded maximum hold time
        """
        if current_time is None:
            current_time = datetime.now()
        
        if max_hold_hours is None:
            max_hold_hours = self.default_max_hold_hours
        
        hold_duration = current_time - entry_time
        hold_hours = hold_duration.total_seconds() / 3600
        
        is_expired = hold_hours >= max_hold_hours
        time_remaining = max(0, max_hold_hours - hold_hours)
        
        return {
            "is_expired": is_expired,
            "hold_hours": float(hold_hours),
            "hold_days": float(hold_hours / 24),
            "max_hold_hours": max_hold_hours,
            "time_remaining_hours": float(time_remaining),
            "entry_time": entry_time.isoformat(),
            "current_time": current_time.isoformat(),
            "expiry_percentage": min(100, (hold_hours / max_hold_hours) * 100)
        }
    
    def get_recommended_exit_time(self, entry_time: datetime, 
                                  strategy_type: str = "default") -> datetime:
        """
        Get recommended exit time based on strategy
        """
        hours = {
            "scalp": 4,
            "day_trade": 24,
            "swing": 72,
            "position": 168,
            "default": self.default_max_hold_hours
        }
        
        hold_hours = hours.get(strategy_type, self.default_max_hold_hours)
        return entry_time + timedelta(hours=hold_hours)
    
    def should_exit_before_event(self, entry_time: datetime, 
                                 event_time: datetime,
                                 buffer_hours: float = 1) -> Dict[str, Any]:
        """
        Check if should exit before a specific event
        """
        time_to_event = event_time - entry_time
        hours_to_event = time_to_event.total_seconds() / 3600
        
        return {
            "should_exit": hours_to_event < buffer_hours,
            "hours_to_event": float(hours_to_event),
            "event_time": event_time.isoformat(),
            "buffer_hours": buffer_hours
        }