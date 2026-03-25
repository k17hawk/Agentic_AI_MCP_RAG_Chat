# =============================================================================
# discovery/config/config_entity.py
# =============================================================================
"""
Configuration entities for the discovery package.
Typed dataclasses defining required and optional configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from enum import Enum


# =============================================================================
# Enums
# =============================================================================
class SearchDepth(str, Enum):
    """Search depth for Tavily."""
    BASIC = "basic"
    ADVANCED = "advanced"


class SortOrder(str, Enum):
    """Sort order for search results."""
    RELEVANCE = "relevance"
    NEWEST = "newest"
    OLDEST = "oldest"


class TimeFilter(str, Enum):
    """Time filter for social media."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"
    ALL = "all"


# =============================================================================
# Base Configuration
# =============================================================================
@dataclass
class BaseConfig:
    """Base configuration with common settings."""
    enabled: bool = True
    rate_limit: int = 30
    cache_ttl_minutes: int = 15
    timeout_seconds: int = 10
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "rate_limit": self.rate_limit,
            "cache_ttl_minutes": self.cache_ttl_minutes,
            "timeout_seconds": self.timeout_seconds
        }


# =============================================================================
# Source Configurations
# =============================================================================
@dataclass
class TavilyConfig(BaseConfig):
    """Tavily search configuration."""
    api_key: Optional[str] = None
    base_url: str = "https://api.tavily.com"
    search_depth: SearchDepth = SearchDepth.BASIC
    max_results: int = 10
    include_answer: bool = True
    include_raw_content: bool = False
    include_domains: List[str] = field(default_factory=list)
    exclude_domains: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.rate_limit = 10  # Tavily specific rate limit


@dataclass
class NewsAPIConfig(BaseConfig):
    """News API configuration."""
    news_api_key: Optional[str] = None
    alpha_vantage_key: Optional[str] = None
    fmp_key: Optional[str] = None
    sources: List[str] = field(default_factory=lambda: [
        "reuters", "bloomberg", "cnbc", "wsj", "financial-times"
    ])
    lookback_days: int = 7
    language: str = "en"
    page_size: int = 50
    
    def __post_init__(self):
        self.rate_limit = 30


@dataclass
class SocialMediaConfig(BaseConfig):
    """Social media configuration."""
    twitter_bearer_token: Optional[str] = None
    reddit_user_agent: str = "TradingBot/1.0"
    reddit_limit: int = 100
    reddit_time_filter: TimeFilter = TimeFilter.WEEK
    reddit_subreddits: List[str] = field(default_factory=lambda: [
        "wallstreetbets", "stocks", "investing", "stockmarket", "options"
    ])
    reddit_comment_limit: int = 10
    reddit_sort: str = "relevance"
    stocktwits_limit: int = 30
    platforms: List[str] = field(default_factory=lambda: ["reddit", "stocktwits"])
    lookback_hours: int = 24
    request_delay: float = 2.0
    
    def __post_init__(self):
        self.rate_limit = 20


@dataclass
class SECFilingsConfig(BaseConfig):
    """SEC filings configuration."""
    user_agent: str = "TradingBot/1.0"
    base_url: str = "https://www.sec.gov"
    filing_types: List[str] = field(default_factory=lambda: [
        "10-K", "10-Q", "8-K", "4", "13F-HR"
    ])
    known_ciks: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        self.rate_limit = 5  # SEC requires slower rate


@dataclass
class OptionsFlowConfig(BaseConfig):
    """Options flow configuration."""
    fmp_key: Optional[str] = None
    volume_threshold: float = 2.0  # 2x average
    premium_threshold: float = 100000  # $100k
    use_alternative: bool = True
    alternative_source: str = "marketbeat"
    
    def __post_init__(self):
        self.rate_limit = 30


@dataclass
class MacroDataConfig(BaseConfig):
    """Macroeconomic data configuration."""
    fred_api_key: Optional[str] = None
    indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def __post_init__(self):
        self.rate_limit = 20
        self.cache_ttl_minutes = 1440  # 24 hours for macro data


# =============================================================================
# Entity Extractor Configuration
# =============================================================================
@dataclass
class NLPExtractorConfig:
    """NLP extractor configuration."""
    spacy_model: str = "en_core_web_sm"
    entity_types: List[str] = field(default_factory=lambda: [
        "PERSON", "ORG", "GPE", "MONEY", "DATE", "PERCENT"
    ])


@dataclass
class RegexExtractorConfig:
    """Regex extractor configuration."""
    ticker_pattern: str = r'\b[A-Z]{1,5}\b'
    exclude_tickers: List[str] = field(default_factory=lambda: [
        "A", "I", "CEO", "CFO", "CTO", "IPO", "ETF", "GDP", "CPI",
        "NYSE", "NASDAQ"
    ])


# =============================================================================
# Enricher Configuration
# =============================================================================
@dataclass
class EnricherConfig:
    """Data enricher configuration."""
    cache_ttl_minutes: int = 60
    price_cache_ttl_minutes: int = 5
    price_source: str = "yahoo"
    fundamentals_source: str = "sec"


# =============================================================================
# Main Discovery Configuration
# =============================================================================
@dataclass
class DiscoveryConfig:
    """Main discovery configuration."""
    # Source configurations
    tavily: TavilyConfig = field(default_factory=TavilyConfig)
    news: NewsAPIConfig = field(default_factory=NewsAPIConfig)
    social: SocialMediaConfig = field(default_factory=SocialMediaConfig)
    sec: SECFilingsConfig = field(default_factory=SECFilingsConfig)
    options: OptionsFlowConfig = field(default_factory=OptionsFlowConfig)
    macro: MacroDataConfig = field(default_factory=MacroDataConfig)
    
    # Extractor configurations
    nlp: NLPExtractorConfig = field(default_factory=NLPExtractorConfig)
    regex: RegexExtractorConfig = field(default_factory=RegexExtractorConfig)
    
    # Enricher configuration
    enricher: EnricherConfig = field(default_factory=EnricherConfig)
    
    # General settings
    max_workers: int = 5
    cache_ttl_minutes: int = 15
    source_weights: Dict[str, float] = field(default_factory=lambda: {
        "tavily": 0.25,
        "news": 0.20,
        "social": 0.15,
        "sec": 0.15,
        "options": 0.15,
        "macro": 0.10
    })
    
    # Rate limit monitoring
    rate_limit_monitoring_enabled: bool = True
    rate_limit_alert_threshold: float = 0.8
    rate_limit_cooldown_minutes: int = 60
    
    @classmethod
    def from_yaml(cls, yaml_config: Dict[str, Any]) -> "DiscoveryConfig":
        """Create config from YAML dictionary."""
        discovery_cfg = yaml_config.get("discovery_config", {})
        
        # Build source configs
        tavily_cfg = TavilyConfig(**discovery_cfg.get("tavily_config", {}))
        news_cfg = NewsAPIConfig(**discovery_cfg.get("news_config", {}))
        social_cfg = SocialMediaConfig(**discovery_cfg.get("social_config", {}))
        sec_cfg = SECFilingsConfig(**discovery_cfg.get("sec_config", {}))
        options_cfg = OptionsFlowConfig(**discovery_cfg.get("options_config", {}))
        macro_cfg = MacroDataConfig(**discovery_cfg.get("macro_config", {}))
        
        nlp_cfg = NLPExtractorConfig(**discovery_cfg.get("nlp_config", {}))
        regex_cfg = RegexExtractorConfig(**discovery_cfg.get("regex_config", {}))
        enricher_cfg = EnricherConfig(**discovery_cfg.get("enricher_config", {}))
        
        return cls(
            tavily=tavily_cfg,
            news=news_cfg,
            social=social_cfg,
            sec=sec_cfg,
            options=options_cfg,
            macro=macro_cfg,
            nlp=nlp_cfg,
            regex=regex_cfg,
            enricher=enricher_cfg,
            max_workers=discovery_cfg.get("max_workers", 5),
            cache_ttl_minutes=discovery_cfg.get("cache_ttl_minutes", 15),
            source_weights=discovery_cfg.get("source_weights", cls.source_weights),
            rate_limit_monitoring_enabled=discovery_cfg.get("rate_limit_monitoring", {}).get("enabled", True),
            rate_limit_alert_threshold=discovery_cfg.get("rate_limit_monitoring", {}).get("alert_threshold", 0.8),
            rate_limit_cooldown_minutes=discovery_cfg.get("rate_limit_monitoring", {}).get("cooldown_minutes", 60)
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tavily_config": self.tavily.__dict__,
            "news_config": self.news.__dict__,
            "social_config": self.social.__dict__,
            "sec_config": self.sec.__dict__,
            "options_config": self.options.__dict__,
            "macro_config": self.macro.__dict__,
            "nlp_config": self.nlp.__dict__,
            "regex_config": self.regex.__dict__,
            "enricher_config": self.enricher.__dict__,
            "max_workers": self.max_workers,
            "cache_ttl_minutes": self.cache_ttl_minutes,
            "source_weights": self.source_weights
        }