"""
Scheduled Trigger - Time-based market session triggers
Fires based on market hours, time of day, and calendar events
"""
from typing import List, Optional
from datetime import datetime, time, timedelta
import asyncio
import pytz
from loguru import logger
import re

from triggers.base_trigger import BaseTrigger, TriggerEvent, TriggerConfig

class MarketSession:
    """Market session definitions"""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MID_DAY = "mid_day"
    PRE_CLOSE = "pre_close"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"

class ScheduledTrigger(BaseTrigger):
    """
    Time-based trigger that fires at specific market sessions
    
    Sessions:
    - Pre-market (4:00-9:30): Look for pre-market movers
    - Market Open (9:30-11:30): Opening momentum
    - Mid-day (11:30-14:00): Consolidation/continuation
    - Pre-close (14:00-16:00): Closing momentum
    - After-hours (16:00-20:00): Extended hours moves
    """
    
    def __init__(self, config: dict, memory_agent=None, message_bus=None):
        super().__init__(
            name="ScheduledTrigger",
            config=config,
            memory_agent=memory_agent,
            message_bus=message_bus
        )
        
        # Market timezone (US Eastern)
        self.tz = pytz.timezone('US/Eastern')
        
        # Session queries
        self.session_queries = {
            MarketSession.PRE_MARKET: "Top pre-market gainers today US stocks",
            MarketSession.MARKET_OPEN: "Top stocks gaining at market open today",
            MarketSession.MID_DAY: "Momentum stocks midday trading US market",
            MarketSession.PRE_CLOSE: "Stocks gaining volume pre-market close",
            MarketSession.AFTER_HOURS: "After hours top gainers today",
            MarketSession.CLOSED: "Stocks to watch next session"
        }
        
        # Session times (Eastern Time)
        self.session_times = {
            MarketSession.PRE_MARKET: (time(4, 0), time(9, 29)),
            MarketSession.MARKET_OPEN: (time(9, 30), time(11, 29)),
            MarketSession.MID_DAY: (time(11, 30), time(13, 59)),
            MarketSession.PRE_CLOSE: (time(14, 0), time(15, 59)),
            MarketSession.AFTER_HOURS: (time(16, 0), time(19, 59)),
        }
        
        # Calendar exceptions (holidays, early closes)
        self.holidays = self._load_holidays()
        
        logger.info("📅 ScheduledTrigger initialized")
    
    def _load_holidays(self) -> List[str]:
        """Load market holidays for the year"""
        # In production, fetch from API or config
        current_year = datetime.now().year
        return [
            f"{current_year}-01-01",  # New Year's
            f"{current_year}-01-19",  # MLK Day
            f"{current_year}-02-16",  # Presidents Day
            f"{current_year}-04-10",  # Good Friday
            f"{current_year}-05-25",  # Memorial Day
            f"{current_year}-07-04",  # Independence Day
            f"{current_year}-09-07",  # Labor Day
            f"{current_year}-11-26",  # Thanksgiving
            f"{current_year}-12-25",  # Christmas
        ]
    
    async def scan(self) -> List[TriggerEvent]:
        """
        Scan based on current market session
        Returns trigger events for the current session
        """
        now = datetime.now(self.tz)
        
        # Check if market is open today
        if self._is_market_holiday(now):
            logger.debug("Market holiday - skipping scheduled trigger")
            return []
        
        # Get current session
        session = self._get_current_session(now)
        
        if session == MarketSession.CLOSED:
            logger.debug("Market closed - skipping scheduled trigger")
            return []
        
        # Check if we should run for this session
        if not self._should_run_for_session(session, now):
            return []
        
        logger.info(f"⏰ Scheduled trigger firing for {session}")
        
        # Get query for this session
        query = self.session_queries[session]
        
        # Create trigger event
        event = TriggerEvent(
            symbol="MARKET",  # Special symbol for market-wide events
            source_trigger=self.name,
            event_type=f"SCHEDULED_{session.upper()}",
            confidence=0.7,  # Base confidence for scheduled
            raw_data={
                "session": session,
                "query": query,
                "market_time": now.isoformat(),
                "is_holiday": False,
                "day_of_week": now.weekday(),
                "month": now.month,
                "day": now.day
            },
            processed_data={
                "session_query": query,
                "recommended_focus": self._get_session_focus(session)
            },
            timeframes_detected=["1d"],  # Daily timeframe for scheduled
            primary_timeframe="1d",
            market_regime=await self._get_market_regime(),
            correlation_id=f"sched_{session}_{now.strftime('%Y%m%d')}"
        )
        
        return [event]
    
    async def validate(self, event: TriggerEvent) -> bool:
        """Validate scheduled trigger events"""
        # Always valid for scheduled
        return True
    
    def _get_current_session(self, dt: datetime) -> str:
        """Determine current market session"""
        current_time = dt.time()
        
        for session, (start, end) in self.session_times.items():
            if start <= current_time <= end:
                return session
        
        return MarketSession.CLOSED
    
    def _is_market_holiday(self, dt: datetime) -> bool:
        """Check if market is closed for holiday"""
        date_str = dt.strftime("%Y-%m-%d")
        return date_str in self.holidays or dt.weekday() >= 5  # Weekend
    
    def _should_run_for_session(self, session: str, now: datetime) -> bool:
        """Determine if we should run for this session"""
        # Get last run time from cache
        cache_key = f"last_run_{session}"
        last_run = self.get_cache(cache_key)
        
        if not last_run:
            return True
        
        # Parse last run
        if isinstance(last_run, str):
            last_run = datetime.fromisoformat(last_run)
        
        # Different frequencies per session
        frequencies = {
            MarketSession.PRE_MARKET: timedelta(minutes=15),
            MarketSession.MARKET_OPEN: timedelta(minutes=5),
            MarketSession.MID_DAY: timedelta(minutes=30),
            MarketSession.PRE_CLOSE: timedelta(minutes=5),
            MarketSession.AFTER_HOURS: timedelta(minutes=15),
        }
        
        # Check if enough time has passed
        if now - last_run < frequencies.get(session, timedelta(minutes=30)):
            return False
        
        # Update cache
        self.set_cache(cache_key, now.isoformat(), ttl_seconds=3600)
        
        return True
    
    def _get_session_focus(self, session: str) -> str:
        """Get focus area for this session"""
        focuses = {
            MarketSession.PRE_MARKET: "earnings_reports_premarket_movers",
            MarketSession.MARKET_OPEN: "opening_range_breakouts",
            MarketSession.MID_DAY: "momentum_continuation",
            MarketSession.PRE_CLOSE: "closing_auction_pressure",
            MarketSession.AFTER_HOURS: "extended_hours_earnings",
        }
        return focuses.get(session, "general_market")
    
    async def _get_market_regime(self) -> Optional[str]:
        """Get current market regime from memory"""
        if self.memory:
            return await self.memory.get("current_market_regime")
        return None