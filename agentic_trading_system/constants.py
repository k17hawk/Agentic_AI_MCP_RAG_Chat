# =============================================================================
# discovery/config/constants.py (COMPLETE)
# =============================================================================
"""
Constants for the discovery package.
All string literals, keys, and thresholds are defined here.
"""

from typing import Final, Set
from enum import Enum


# =============================================================================
# Enums for configuration
# =============================================================================

class SearchDepth(str, Enum):
    """Search depth for Tavily."""
    BASIC = "basic"
    ADVANCED = "advanced"
    
    def __str__(self) -> str:
        return self.value


class TimeFilter(str, Enum):
    """Time filter for social media searches."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL = "all"
    
    def __str__(self) -> str:
        return self.value


class SortOrder(str, Enum):
    """Sort order for search results."""
    RELEVANCE = "relevance"
    NEWEST = "newest"
    OLDEST = "oldest"
    
    def __str__(self) -> str:
        return self.value


# =============================================================================
# Entity Types
# =============================================================================
class EntityType:
    """Entity types for extraction."""
    TICKER: Final[str] = "ticker"
    COMPANY: Final[str] = "company"
    PERSON: Final[str] = "person"
    ORGANIZATION: Final[str] = "organization"
    LOCATION: Final[str] = "location"
    DATE: Final[str] = "date"
    CURRENCY: Final[str] = "currency"
    INDUSTRY: Final[str] = "industry"
    PERCENTAGE: Final[str] = "percentage"
    CONTACT: Final[str] = "contact"
    MARKET_INDICATOR: Final[str] = "market_indicator"
    STOCK_EXCHANGE: Final[str] = "stock_exchange"
    FINANCIAL_TERM: Final[str] = "financial_term"


# =============================================================================
# Data Sources
# =============================================================================
class Source:
    """Data source identifiers."""
    TAVILY: Final[str] = "tavily"
    NEWS: Final[str] = "news"
    SOCIAL: Final[str] = "social"
    SEC: Final[str] = "sec"
    OPTIONS: Final[str] = "options"
    MACRO: Final[str] = "macro"
    REDDIT: Final[str] = "reddit"
    STOCKTWITS: Final[str] = "stocktwits"
    TWITTER: Final[str] = "twitter"
    ALPHA_VANTAGE: Final[str] = "alphavantage"
    FMP: Final[str] = "fmp"
    FRED: Final[str] = "fred"
    YAHOO_FINANCE: Final[str] = "yfinance"


# =============================================================================
# Content Types
# =============================================================================
class ContentType:
    """Content type classifications."""
    EARNINGS: Final[str] = "earnings"
    MA: Final[str] = "ma"
    ANALYST_RATING: Final[str] = "analyst_rating"
    IPO: Final[str] = "ipo"
    CORPORATE_ACTION: Final[str] = "corporate_action"
    REGULATORY: Final[str] = "regulatory"
    MANAGEMENT: Final[str] = "management"
    PRICE_TARGET: Final[str] = "price_target"
    GENERAL: Final[str] = "general"
    WEB_SEARCH: Final[str] = "web_search"
    NEWS: Final[str] = "news"
    SOCIAL_POST: Final[str] = "social_post"
    FILING: Final[str] = "filing"
    OPTION_FLOW: Final[str] = "option_flow"
    MACRO_DATA: Final[str] = "macro_data"


# =============================================================================
# Search Types
# =============================================================================
class SearchType:
    """Search type identifiers for source filtering."""
    GENERAL: Final[str] = "general"
    NEWS: Final[str] = "news"
    SOCIAL: Final[str] = "social"
    FUNDAMENTAL: Final[str] = "fundamental"
    TECHNICAL: Final[str] = "technical"
    MACRO: Final[str] = "macro"


# =============================================================================
# Entity Extraction
# =============================================================================
class EntityExtraction:
    """Constants for entity extraction."""
    MAX_ENTITIES_PER_TYPE: Final[int] = 10
    MAX_TEXT_LENGTH: Final[int] = 100000
    
    # Single-letter tickers that are valid
    VALID_SINGLE_LETTER_TICKERS: Final[Set[str]] = {
        'A', 'C', 'F', 'G', 'H', 'J', 'M', 'R', 'T', 'V', 'Z'
    }
    
    # High authority news sources
    HIGH_AUTHORITY_SOURCES: Final[Set[str]] = {
        'reuters', 'bloomberg', 'wsj', 'wall street journal',
        'financial times', 'ft.com', 'nytimes', 'economist'
    }
    
    MEDIUM_AUTHORITY_SOURCES: Final[Set[str]] = {
        'cnbc', 'yahoo', 'seeking alpha', 'marketwatch',
        'forbes', 'business insider', 'investopedia'
    }
    
    LOW_AUTHORITY_SOURCES: Final[Set[str]] = {
        'twitter', 'reddit', 'stocktwits', 'facebook',
        'linkedin', 'medium', 'wordpress'
    }
    
    # Source authority scores
    SOURCE_AUTHORITY_SCORES: Final[dict] = {
        'high': 1.0,
        'medium': 0.7,
        'low': 0.3,
        'default': 0.5
    }
    
    # Relevance score range
    MIN_RELEVANCE: Final[float] = 0.0
    MAX_RELEVANCE: Final[float] = 1.0
    DEFAULT_RELEVANCE: Final[float] = 0.5


# =============================================================================
# Filing Types
# =============================================================================
class FilingType:
    """SEC filing type constants."""
    FORM_10K: Final[str] = "10-K"
    FORM_10Q: Final[str] = "10-Q"
    FORM_8K: Final[str] = "8-K"
    FORM_4: Final[str] = "4"
    FORM_13F_HR: Final[str] = "13F-HR"
    
    # Importance mapping
    IMPORTANCE_HIGH: Final[str] = "high"
    IMPORTANCE_MEDIUM: Final[str] = "medium"
    IMPORTANCE_LOW: Final[str] = "low"
    
    IMPORTANCE_MAP: Final[dict] = {
        "10-K": IMPORTANCE_HIGH,
        "8-K": IMPORTANCE_HIGH,
        "4": IMPORTANCE_HIGH,
        "10-Q": IMPORTANCE_MEDIUM,
        "13F-HR": IMPORTANCE_MEDIUM,
        "DEF 14A": IMPORTANCE_MEDIUM,
        "S-1": IMPORTANCE_HIGH,
        "S-4": IMPORTANCE_MEDIUM
    }


# =============================================================================
# Rate Limiting
# =============================================================================
class RateLimit:
    """Rate limiting constants."""
    WINDOW_SECONDS: Final[int] = 60
    TAVILY_REQUESTS_PER_MINUTE: Final[int] = 10
    NEWSAPI_REQUESTS_PER_MINUTE: Final[int] = 30
    ALPHA_VANTAGE_REQUESTS_PER_MINUTE: Final[int] = 5
    FMP_REQUESTS_PER_MINUTE: Final[int] = 30
    SEC_REQUESTS_PER_SECOND: Final[int] = 5
    FRED_REQUESTS_PER_MINUTE: Final[int] = 20
    REDDIT_REQUESTS_PER_MINUTE: Final[int] = 30
    STOCKTWITS_REQUESTS_PER_MINUTE: Final[int] = 60


# =============================================================================
# Cache TTLs
# =============================================================================
class CacheTTL:
    """Cache TTL constants in seconds."""
    SHORT: Final[int] = 60  # 1 minute
    MEDIUM: Final[int] = 900  # 15 minutes
    LONG: Final[int] = 3600  # 1 hour
    DAY: Final[int] = 86400  # 24 hours
    WEEK: Final[int] = 604800  # 7 days
    
    # Specific defaults
    PRICE: Final[int] = SHORT
    NEWS: Final[int] = MEDIUM
    SOCIAL: Final[int] = MEDIUM
    TAVILY: Final[int] = MEDIUM
    SEC: Final[int] = DAY
    OPTIONS: Final[int] = MEDIUM
    MACRO: Final[int] = DAY


# =============================================================================
# Scoring Weights
# =============================================================================
class ScoringWeights:
    """Scoring weights for ranking."""
    # Relevance scoring
    TITLE_MATCH_MULTIPLIER: Final[float] = 10.0
    CONTENT_MATCH_MULTIPLIER: Final[float] = 3.0
    
    # Authority scores
    HIGH_AUTHORITY_BONUS: Final[float] = 10.0
    MEDIUM_AUTHORITY_BONUS: Final[float] = 8.0
    LOW_AUTHORITY_BONUS: Final[float] = 3.0
    
    # Recency bonuses
    RECENT_24H_BONUS: Final[float] = 5.0
    RECENT_72H_BONUS: Final[float] = 2.0
    
    # Source weights for aggregation
    DEFAULT_SOURCE_WEIGHTS: Final[dict] = {
        Source.TAVILY: 0.25,
        Source.NEWS: 0.20,
        Source.SOCIAL: 0.15,
        Source.SEC: 0.15,
        Source.OPTIONS: 0.15,
        Source.MACRO: 0.10
    }


# =============================================================================
# Option Flow Thresholds
# =============================================================================
class OptionFlowThresholds:
    """Option flow detection thresholds."""
    VOLUME_RATIO_UNUSUAL: Final[float] = 2.0
    MIN_PREMIUM_UNUSUAL: Final[float] = 100000  # $100k
    CONTRACT_MULTIPLIER: Final[int] = 100  # 100 shares per contract


# =============================================================================
# Sentiment
# =============================================================================
class Sentiment:
    """Sentiment constants."""
    POSITIVE: Final[str] = "positive"
    NEGATIVE: Final[str] = "negative"
    NEUTRAL: Final[str] = "neutral"
    
    MIN_CONFIDENCE: Final[float] = 0.3
    REQUIRED_SOURCES: Final[int] = 2
    MIN_SENTIMENT_SCORE: Final[float] = 0.6


# =============================================================================
# Timeframes
# =============================================================================
class Timeframe:
    """Timeframe constants."""
    MINUTE_5: Final[str] = "5m"
    MINUTE_15: Final[str] = "15m"
    HOUR_1: Final[str] = "1h"
    DAY_1: Final[str] = "1d"
    WEEK_1: Final[str] = "1wk"
    MONTH_1: Final[str] = "1mo"
    
    REQUIRED_PERIODS: Final[dict] = {
        MINUTE_5: "5d",
        MINUTE_15: "10d",
        HOUR_1: "1mo",
        DAY_1: "6mo",
        WEEK_1: "1y",
        MONTH_1: "5y"
    }


# =============================================================================
# Logging
# =============================================================================
class LogLevel:
    """Log level constants."""
    DEBUG: Final[str] = "DEBUG"
    INFO: Final[str] = "INFO"
    WARNING: Final[str] = "WARNING"
    ERROR: Final[str] = "ERROR"
    CRITICAL: Final[str] = "CRITICAL"



BLACKLIST = {
            # Articles / conjunctions / pronouns
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS', 'THESE', 'THOSE',
            'HAVE', 'WILL', 'YOUR', 'ABOUT', 'WOULD', 'COULD', 'SHOULD',
            'ALL', 'ANY', 'NOT', 'BUT', 'NOR', 'OR', 'YET', 'SO',
            'ITS', 'OUR', 'THEIR', 'HIS', 'HER', 'OWN',
            'BEEN', 'BEING', 'ALSO', 'MORE', 'THAN', 'JUST', 'ONLY',
            'THEY', 'THEM', 'THEN', 'WHEN', 'WHERE', 'WHAT', 'WHICH', 'WHO', 'WHY', 'HOW',

            # Finance / business abbreviations
            'NYSE', 'NASDAQ', 'AMEX', 'SEC', 'CEO', 'CFO', 'COO', 'CTO', 'CIO',
            'EPS', 'YOY', 'QOQ', 'TTM', 'ATH', 'ATL', 'GDP', 'CPI', 'PMI',
            'ESG', 'ROI', 'ROE', 'DCF', 'IPO', 'ETF', 'REIT', 'SPAC',

            # Currencies
            'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD', 'CHF', 'HKD', 'SGD',

            # Legal suffixes
            'INC', 'CORP', 'LTD', 'LLC', 'LP', 'PLC', 'CO', 'GROUP', 'HOLDINGS',

            # Market / trading words
            'STOCK', 'STOCKS', 'SHARE', 'SHARES', 'PRICE', 'MARKET', 'MARKETS',
            'FUND', 'FUNDS', 'BOND', 'BONDS', 'TRADE', 'TRADES', 'CHART', 'CHARTS',
            'NEWS', 'DATA', 'HIGH', 'LOW', 'OPEN', 'CLOSE',
            'CALL', 'CALLS', 'BULL', 'BEAR', 'LONG', 'SHORT',

            # Price-action words
            'GAIN', 'GAINS', 'LOSS', 'RALLY', 'SURGE', 'DROP', 'FALL',
            'RISE', 'JUMP', 'SLIDE', 'SPIKE', 'DIP', 'PEAK', 'FLAT',
            'UP', 'DOWN', 'RANGE', 'LEVEL', 'MARK', 'BASE', 'ZONE',

            # Geopolitical words
            'WAR', 'WARS', 'DEAL', 'RISK', 'FEAR', 'TALK', 'TALKS',
            'PLAN', 'BILL', 'ACT', 'LAW', 'RULE',

            # Time words
            'WEEK', 'MONTH', 'YEAR', 'DAILY', 'LIVE', 'PRE', 'POST', 'LATE', 'EARLY',

            # UI / meta words
            'REAL', 'BEST', 'TOP', 'NEW', 'OLD', 'KEY', 'MAIN', 'PLUS', 'PRO', 'API', 'APP',
            'IDEAS', 'IDEA', 'QUOTE', 'QUOTES', 'VIEW', 'VIEWS',
            'ALERT', 'ALERTS', 'LOGIN', 'SIGN', 'TERMS', 'HELP',
            'ABOUT', 'HOME', 'BACK', 'NEXT', 'PREV', 'READ',
            'SHOW', 'HIDE', 'FULL', 'LIVE', 'FREE', 'MORE', 'LESS',

            # Privacy / legal
            'CCPA', 'GDPR', 'DMCA', 'EULA', 'TOS',

            # Full company names
            'APPLE', 'GOOGLE', 'META', 'TESLA', 'AMAZON', 'MICROSOFT',
            'NVIDIA', 'DISNEY', 'NETFLIX', 'INTEL', 'CISCO', 'ORACLE',
        }

COMMON_WORDS = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THIS', 'THAT', 'HAVE', 'WILL'}

# =============================================================================
# Export all constants for easy access
# =============================================================================
__all__ = [
    'SearchDepth',
    'TimeFilter',
    'SortOrder',
    'EntityType',
    'Source',
    'ContentType',
    'SearchType',
    'EntityExtraction',
    'FilingType',
    'RateLimit',
    'CacheTTL',
    'ScoringWeights',
    'OptionFlowThresholds',
    'Sentiment',
    'Timeframe',
    'LogLevel'
]