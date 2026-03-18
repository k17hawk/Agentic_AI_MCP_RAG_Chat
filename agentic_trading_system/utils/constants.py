"""
Constants - System-wide constants and enums
"""
from enum import Enum, auto
from datetime import time

# Market Sessions
class MarketSession(Enum):
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"
    MID_DAY = "mid_day"
    PRE_CLOSE = "pre_close"
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"

# Market hours (Eastern Time)
MARKET_HOURS = {
    MarketSession.PRE_MARKET: (time(4, 0), time(9, 29)),
    MarketSession.MARKET_OPEN: (time(9, 30), time(11, 29)),
    MarketSession.MID_DAY: (time(11, 30), time(13, 59)),
    MarketSession.PRE_CLOSE: (time(14, 0), time(15, 59)),
    MarketSession.AFTER_HOURS: (time(16, 0), time(19, 59)),
}

# Order Types
class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

# Order Side
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

# Order Status
class OrderStatus(Enum):
    PENDING_NEW = "PENDING_NEW"
    SUBMITTED = "SUBMITTED"
    ACCEPTED = "ACCEPTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"

# Time in Force
class TimeInForce(Enum):
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date

# Trigger Types
class TriggerType(Enum):
    PRICE = "price"
    VOLUME = "volume"
    NEWS = "news"
    PATTERN = "pattern"
    SOCIAL = "social"
    SCHEDULED = "scheduled"

# Signal Types
class SignalType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    NEWS = "news"
    SOCIAL = "social"
    PATTERN = "pattern"

# Analysis Types
class AnalysisType(Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    TIMEFRAME = "timeframe"
    RISK = "risk"

# Risk Levels
class RiskLevel(Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

# Trade Outcomes
class TradeOutcome(Enum):
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    OPEN = "open"

# Priority Levels
class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Timeframes
TIMEFRAMES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1hour",
    "4h": "4hour",
    "1d": "1day",
    "1w": "1week",
    "1M": "1month"
}

# Default timeframes for analysis (YOUR 60-DAY REQUIREMENT)
DEFAULT_TIMEFRAMES = {
    "short": 5,      # 5 days
    "medium": 20,    # 20 days
    "long": 60,      # 60 days - YOUR CORE
    "yearly": 252    # 1 year
}

# Market holidays (simplified - would need full calendar in production)
MARKET_HOLIDAYS = [
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Presidents Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
]

# Currency symbols
CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CNY": "¥",
    "AUD": "A$",
    "CAD": "C$",
    "CHF": "Fr",
    "HKD": "HK$",
    "SGD": "S$"
}

# Exchange codes
EXCHANGE_CODES = {
    "NYSE": "New York Stock Exchange",
    "NASDAQ": "NASDAQ",
    "AMEX": "American Stock Exchange",
    "BATS": "BATS Global Markets",
    "IEX": "Investors Exchange",
    "TSX": "Toronto Stock Exchange",
    "LSE": "London Stock Exchange",
    "HKEX": "Hong Kong Exchange",
    "TSE": "Tokyo Stock Exchange",
    "SSE": "Shanghai Stock Exchange"
}

# Data directories
DATA_DIRS = {
    "raw": "data/raw",
    "processed": "data/processed",
    "models": "data/models",
    "reports": "data/reports",
    "charts": "data/charts",
    "logs": "data/logs",
    "cache": "data/cache",
    "exports": "data/exports",
    "backups": "data/backups"
}

# API endpoints
API_ENDPOINTS = {
    "yahoo": "https://query1.finance.yahoo.com",
    "alphavantage": "https://www.alphavantage.co",
    "fmp": "https://financialmodelingprep.com",
    "newsapi": "https://newsapi.org",
    "twitter": "https://api.twitter.com/2",
    "reddit": "https://oauth.reddit.com",
    "sec": "https://www.sec.gov",
    "fred": "https://api.stlouisfed.org"
}

# HTTP Status codes
HTTP_STATUS = {
    "OK": 200,
    "CREATED": 201,
    "ACCEPTED": 202,
    "NO_CONTENT": 204,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "RATE_LIMIT": 429,
    "SERVER_ERROR": 500
}

# Cache TTLs (seconds)
CACHE_TTL = {
    "price": 60,           # 1 minute
    "quote": 60,           # 1 minute
    "company_info": 3600,  # 1 hour
    "news": 300,           # 5 minutes
    "social": 900,         # 15 minutes
    "technical": 300,      # 5 minutes
    "fundamental": 86400,  # 24 hours
    "analysis": 300,       # 5 minutes
    "decision": 300        # 5 minutes
}

# Default pagination
PAGINATION = {
    "default_limit": 100,
    "max_limit": 1000,
    "default_offset": 0
}

# Logging levels
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}