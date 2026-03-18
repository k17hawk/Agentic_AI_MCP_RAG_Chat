"""
Date Utils - Date and time utility functions
"""
from datetime import datetime, date, timedelta, time
from typing import List, Optional, Union
import pytz
import calendar
from utils.constants import MARKET_HOURS, MARKET_HOLIDAYS

EASTERN_TZ = pytz.timezone('US/Eastern')
UTC_TZ = pytz.UTC

def parse_date(date_str: str, formats: List[str] = None) -> Optional[datetime]:
    """Parse date string to datetime"""
    if formats is None:
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%d-%m-%Y',
            '%d/%m/%Y',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ'
        ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    
    return None

def format_date(dt: datetime, format: str = '%Y-%m-%d %H:%M:%S') -> str:
    """Format datetime to string"""
    return dt.strftime(format)

def get_market_time() -> datetime:
    """Get current market time (Eastern Time)"""
    return datetime.now(EASTERN_TZ)

def to_eastern(dt: datetime) -> datetime:
    """Convert datetime to Eastern Time"""
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(EASTERN_TZ)

def to_utc(dt: datetime) -> datetime:
    """Convert datetime to UTC"""
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    return dt.astimezone(UTC_TZ)

def is_market_open(dt: Optional[datetime] = None) -> bool:
    """Check if market is open at given time"""
    if dt is None:
        dt = get_market_time()
    
    # Convert to Eastern if needed
    if dt.tzinfo is None:
        dt = EASTERN_TZ.localize(dt)
    else:
        dt = dt.astimezone(EASTERN_TZ)
    
    # Check weekend
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Check holidays
    date_str = dt.strftime('%Y-%m-%d')
    if date_str in MARKET_HOLIDAYS:
        return False
    
    # Check market hours
    current_time = dt.time()
    
    for session, (start, end) in MARKET_HOURS.items():
        if session in ['PRE_MARKET', 'AFTER_HOURS']:
            continue
        if start <= current_time <= end:
            return True
    
    return False

def get_market_session(dt: Optional[datetime] = None) -> str:
    """Get current market session"""
    if dt is None:
        dt = get_market_time()
    
    if dt.tzinfo is None:
        dt = EASTERN_TZ.localize(dt)
    else:
        dt = dt.astimezone(EASTERN_TZ)
    
    current_time = dt.time()
    
    for session, (start, end) in MARKET_HOURS.items():
        if start <= current_time <= end:
            return session.value
    
    return "closed"

def get_trading_days(start_date: Union[str, datetime], 
                    end_date: Union[str, datetime]) -> List[str]:
    """Get list of trading days between dates"""
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)
    
    if start_date.tzinfo is None:
        start_date = EASTERN_TZ.localize(start_date)
    if end_date.tzinfo is None:
        end_date = EASTERN_TZ.localize(end_date)
    
    trading_days = []
    current = start_date
    
    while current <= end_date:
        if is_market_open(current):
            trading_days.append(current.strftime('%Y-%m-%d'))
        current += timedelta(days=1)
    
    return trading_days

def date_range(start_date: Union[str, datetime], 
              end_date: Union[str, datetime],
              step_days: int = 1) -> List[datetime]:
    """Generate date range"""
    if isinstance(start_date, str):
        start_date = parse_date(start_date)
    if isinstance(end_date, str):
        end_date = parse_date(end_date)
    
    dates = []
    current = start_date
    
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=step_days)
    
    return dates

def time_ago(dt: Union[str, datetime]) -> str:
    """Get human-readable time ago string"""
    if isinstance(dt, str):
        dt = parse_date(dt)
    
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    
    now = datetime.now(pytz.UTC)
    diff = now - dt
    
    seconds = diff.total_seconds()
    
    if seconds < 60:
        return f"{int(seconds)} seconds ago"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif seconds < 2592000:
        days = int(seconds / 86400)
        return f"{days} day{'s' if days != 1 else ''} ago"
    elif seconds < 31536000:
        months = int(seconds / 2592000)
        return f"{months} month{'s' if months != 1 else ''} ago"
    else:
        years = int(seconds / 31536000)
        return f"{years} year{'s' if years != 1 else ''} ago"

def next_market_open(dt: Optional[datetime] = None) -> datetime:
    """Get next market open time"""
    if dt is None:
        dt = get_market_time()
    
    current = dt
    
    while True:
        current += timedelta(days=1)
        if is_market_open(current.replace(hour=9, minute=30)):
            return current.replace(hour=9, minute=30, second=0, microsecond=0)

def previous_market_close(dt: Optional[datetime] = None) -> datetime:
    """Get previous market close time"""
    if dt is None:
        dt = get_market_time()
    
    current = dt
    
    while True:
        current -= timedelta(days=1)
        if is_market_open(current.replace(hour=16, minute=0)):
            return current.replace(hour=16, minute=0, second=0, microsecond=0)

def get_quarter(date: Optional[datetime] = None) -> int:
    """Get quarter of the year (1-4)"""
    if date is None:
        date = datetime.now()
    return (date.month - 1) // 3 + 1

def get_fiscal_year(date: Optional[datetime] = None) -> int:
    """Get fiscal year (for companies with fiscal year not matching calendar)"""
    if date is None:
        date = datetime.now()
    return date.year

def days_between(date1: Union[str, datetime], 
                date2: Union[str, datetime]) -> int:
    """Get number of days between two dates"""
    if isinstance(date1, str):
        date1 = parse_date(date1)
    if isinstance(date2, str):
        date2 = parse_date(date2)
    
    diff = date2 - date1
    return abs(diff.days)