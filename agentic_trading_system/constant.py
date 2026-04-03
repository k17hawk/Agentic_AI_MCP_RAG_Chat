class EntityExtraction:
    """Entity extraction constants."""
    MAX_TEXT_LENGTH = 50000
    MAX_ENTITIES_PER_TYPE = 50
    MAX_RELEVANCE = 1.0
    MIN_RELEVANCE = 0.0
    DEFAULT_RELEVANCE = 0.5
    
    VALID_SINGLE_LETTER_TICKERS = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'}
    
    HIGH_AUTHORITY_SOURCES = ['reuters', 'bloomberg', 'wsj', 'ft.com', 'financialtimes', 
                               'economist', 'nytimes', 'washingtonpost', 'sec.gov']
    MEDIUM_AUTHORITY_SOURCES = ['cnbc', 'marketwatch', 'seekingalpha', 'yahoo', 'finance',
                                 'investopedia', 'fool', 'barron\'s', 'forbes', 'businessinsider']
    LOW_AUTHORITY_SOURCES = ['reddit', 'twitter', 'stocktwits', 'facebook', 'linkedin',
                              'medium', 'substack', 'wordpress', 'blogspot', 'tumblr']
    
    SOURCE_AUTHORITY_SCORES = {
        'high': 0.9,
        'medium': 0.6,
        'low': 0.3,
        'default': 0.5
    }


class ScoringWeights:
    """Scoring weights for ranking."""
    TITLE_MATCH_MULTIPLIER = 10.0
    CONTENT_MATCH_MULTIPLIER = 2.0
    HIGH_AUTHORITY_BONUS = 20.0
    MEDIUM_AUTHORITY_BONUS = 10.0
    LOW_AUTHORITY_BONUS = 5.0
    RECENT_24H_BONUS = 15.0
    RECENT_72H_BONUS = 5.0


class Source:
    """Data source constants."""
    TAVILY = "tavily"
    NEWS = "news"
    SOCIAL = "social"
    SEC = "sec"
    OPTIONS = "options"
    MACRO = "macro"
    REDDIT = "reddit"
    STOCKTWITS = "stocktwits"
    TWITTER = "twitter"
    YAHOO_FINANCE = "yahoo_finance"
    FRED = "fred"
    FMP = "fmp"
    ALPHA_VANTAGE = "alpha_vantage"
    NEWSAPI = "newsapi"


class SearchType:
    """Search type constants."""
    GENERAL = "general"
    NEWS = "news"
    SOCIAL = "social"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    MACRO = "macro"


class ContentType:
    """Content type constants."""
    EARNINGS = "earnings"
    MA = "merger_acquisition"
    ANALYST_RATING = "analyst_rating"
    IPO = "ipo"
    CORPORATE_ACTION = "corporate_action"
    REGULATORY = "regulatory"
    MANAGEMENT = "management"
    PRICE_TARGET = "price_target"
    GENERAL = "general"


class RateLimit:
    """Rate limit constants."""
    WINDOW_SECONDS = 60
    TAVILY_REQUESTS_PER_MINUTE = 10
    NEWSAPI_REQUESTS_PER_MINUTE = 30
    REDDIT_REQUESTS_PER_MINUTE = 20
    SEC_REQUESTS_PER_SECOND = 5
    FMP_REQUESTS_PER_MINUTE = 30
    FRED_REQUESTS_PER_MINUTE = 20


class CacheTTL:
    """Cache TTL constants in seconds."""
    TAVILY = 1800  # 30 minutes
    NEWS = 900    # 15 minutes
    SOCIAL = 600  # 10 minutes
    OPTIONS = 900  # 15 minutes
    DAY = 86400   # 24 hours


class OptionFlowThresholds:
    """Options flow thresholds."""
    CONTRACT_MULTIPLIER = 100


class Sentiment:
    """Sentiment constants."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class FilingType:
    """SEC filing type constants."""
    FORM_10K = "10-K"
    FORM_10Q = "10-Q"
    FORM_8K = "8-K"
    FORM_4 = "4"
    FORM_13F_HR = "13F-HR"
    
    IMPORTANCE_MAP = {
        FORM_10K: 0.9,
        FORM_10Q: 0.8,
        FORM_8K: 0.7,
        FORM_4: 0.6,
        FORM_13F_HR: 0.5,
    }
    IMPORTANCE_LOW = 0.3


# Common words that shouldn't be treated as tickers
COMMON_WORDS = {
    'THE', 'AND', 'FOR', 'BUT', 'NOT', 'ITS', 'ARE', 'WAS',
    'HAS', 'ALL', 'NEW', 'TOP', 'BIG', 'OLD', 'OUR', 'GET',
    'GOT', 'LET', 'PUT', 'SET', 'USE', 'MAY', 'CAN', 'DID',
}

BLACKLIST = {
    'A', 'AN', 'THE', 'AND', 'OR', 'BUT', 'FOR', 'WITH', 'FROM',
    'INC', 'CORP', 'LTD', 'LLC', 'PLC', 'ETF', 'IPO', 'CEO', 'CFO',
    'COO', 'CTO', 'CIO', 'EPS', 'PE', 'ROE', 'ROA', 'EBITDA',
}