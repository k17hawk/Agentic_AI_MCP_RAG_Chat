# =============================================================================
# discovery/config/loader.py
# =============================================================================
"""
Configuration loader for discovery package.
Loads configuration from YAML files and environment variables.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Import config entities
from agentic_trading_system.config.config__entity import (
    DiscoveryConfig,
    TavilyConfig,
    NewsAPIConfig,
    SocialMediaConfig,
    SECFilingsConfig,
    OptionsFlowConfig,
    MacroDataConfig,
    NLPExtractorConfig,
    RegexExtractorConfig,
    EnricherConfig
)

# Import constants
from agentic_trading_system.constants import SearchDepth, TimeFilter, SearchType, RateLimit, CacheTTL
# =============================================================================
# discovery/config/loader.py (UPDATED - fix imports)
# =============================================================================
"""
Configuration loader for discovery package.
Loads configuration from YAML files and environment variables.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()



class DiscoveryConfigLoader:
    """
    Loads discovery configuration from YAML files and environment variables.
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the config loader.
        
        Args:
            config_dir: Directory containing YAML config files.
                        Defaults to 'config' in project root.
        """
        if config_dir is None:
            # Default to config directory in project root
            # This file is at: discovery/config/loader.py
            # So project root is 4 levels up
            current_dir = Path(__file__).resolve().parent
            # discovery/config -> discovery -> agentic_trading_system -> project_root
            project_root = current_dir.parent.parent.parent
            config_dir = project_root / "config"
        
        self.config_dir = Path(config_dir)
        self._config_cache: Optional[DiscoveryConfig] = None
        
    def load(self, config_file: str = "discovery_config.yaml") -> DiscoveryConfig:
        """
        Load discovery configuration.
        
        Args:
            config_file: Name of the YAML config file
            
        Returns:
            DiscoveryConfig object
        """
        # Return cached config if available
        if self._config_cache is not None:
            return self._config_cache
        
        config_path = self.config_dir / config_file
        
        # Check if config file exists
        if not config_path.exists():
            print(f"⚠️ Config file not found: {config_path}")
            print("   Using default configuration...")
            self._config_cache = self._create_default_config()
            return self._config_cache
        
        try:
            import yaml
            
            with open(config_path, 'r') as f:
                yaml_content = f.read()
            
            # Replace environment variables
            yaml_content = self._replace_env_vars(yaml_content)
            yaml_config = yaml.safe_load(yaml_content)
            
            # Build config from YAML
            self._config_cache = self._build_config(yaml_config)
            
            print(f"✅ Loaded discovery config from: {config_path}")
            return self._config_cache
            
        except Exception as e:
            print(f"❌ Error loading config from {config_path}: {e}")
            print("   Using default configuration...")
            self._config_cache = self._create_default_config()
            return self._config_cache
    
    def _replace_env_vars(self, content: str) -> str:
        """
        Replace ${VAR:-default} with environment variables.
        
        Supports:
            ${VAR} - replaces with env var value
            ${VAR:-default} - replaces with env var or default if not set
        """
        pattern = r'\${([A-Za-z0-9_]+)(?:::-([^}]*))?}'
        
        def replace(match):
            var_name = match.group(1)
            default = match.group(2) or ''
            
            # Get from environment
            value = os.getenv(var_name)
            
            if value is not None:
                return value
            return default
        
        return re.sub(pattern, replace, content)
    
    def _build_config(self, yaml_config: Dict[str, Any]) -> DiscoveryConfig:
        """
        Build DiscoveryConfig from YAML dictionary.
        
        Args:
            yaml_config: Parsed YAML configuration
            
        Returns:
            DiscoveryConfig object
        """
        # Get discovery_config section
        discovery_cfg = yaml_config.get("discovery_config", {})
        
        # Build Tavily config
        tavily_config = self._build_tavily_config(discovery_cfg.get("tavily_config", {}))
        
        # Build News config
        news_config = self._build_news_config(discovery_cfg.get("news_config", {}))
        
        # Build Social config
        social_config = self._build_social_config(discovery_cfg.get("social_config", {}))
        
        # Build SEC config
        sec_config = self._build_sec_config(discovery_cfg.get("sec_config", {}))
        
        # Build Options config
        options_config = self._build_options_config(discovery_cfg.get("options_config", {}))
        
        # Build Macro config
        macro_config = self._build_macro_config(discovery_cfg.get("macro_config", {}))
        
        # Build NLP config
        nlp_config = self._build_nlp_config(discovery_cfg.get("nlp_config", {}))
        
        # Build Regex config
        regex_config = self._build_regex_config(discovery_cfg.get("regex_config", {}))
        
        # Build Enricher config
        enricher_config = self._build_enricher_config(discovery_cfg.get("enricher_config", {}))
        
        # Build main config
        return DiscoveryConfig(
            tavily=tavily_config,
            news=news_config,
            social=social_config,
            sec=sec_config,
            options=options_config,
            macro=macro_config,
            nlp=nlp_config,
            regex=regex_config,
            enricher=enricher_config,
            max_workers=discovery_cfg.get("max_workers", 5),
            cache_ttl_minutes=discovery_cfg.get("cache_ttl_minutes", 15),
            source_weights=discovery_cfg.get("source_weights", {
                "tavily": 0.25,
                "news": 0.20,
                "social": 0.15,
                "sec": 0.15,
                "options": 0.15,
                "macro": 0.10
            }),
            rate_limit_monitoring_enabled=discovery_cfg.get("rate_limit_monitoring", {}).get("enabled", True),
            rate_limit_alert_threshold=discovery_cfg.get("rate_limit_monitoring", {}).get("alert_threshold", 0.8),
            rate_limit_cooldown_minutes=discovery_cfg.get("rate_limit_monitoring", {}).get("cooldown_minutes", 60)
        )
    
    def _build_tavily_config(self, cfg: Dict[str, Any]) -> TavilyConfig:
        """Build Tavily configuration."""
        search_depth = cfg.get("search_depth", "basic")
        # Handle string or enum
        if isinstance(search_depth, str):
            try:
                search_depth = SearchDepth(search_depth)
            except ValueError:
                search_depth = SearchDepth.BASIC
        
        return TavilyConfig(
            enabled=cfg.get("enabled", True),
            api_key=cfg.get("api_key"),
            base_url=cfg.get("base_url", "https://api.tavily.com"),
            search_depth=search_depth,
            max_results=cfg.get("max_results", 10),
            include_answer=cfg.get("include_answer", True),
            include_raw_content=cfg.get("include_raw_content", False),
            include_domains=cfg.get("include_domains", []),
            exclude_domains=cfg.get("exclude_domains", []),
            rate_limit=cfg.get("rate_limit", RateLimit.TAVILY_REQUESTS_PER_MINUTE),
            cache_ttl_minutes=cfg.get("cache_ttl_minutes", CacheTTL.TAVILY // 60),
            timeout_seconds=cfg.get("timeout_seconds", 10)
        )
    
    def _build_news_config(self, cfg: Dict[str, Any]) -> NewsAPIConfig:
        """Build News API configuration."""
        return NewsAPIConfig(
            enabled=cfg.get("enabled", True),
            news_api_key=cfg.get("news_api_key"),
            alpha_vantage_key=cfg.get("alpha_vantage_key"),
            fmp_key=cfg.get("fmp_key"),
            sources=cfg.get("sources", [
                "reuters", "bloomberg", "cnbc", "wsj", "financial-times"
            ]),
            lookback_days=cfg.get("lookback_days", 7),
            language=cfg.get("language", "en"),
            page_size=cfg.get("page_size", 50),
            rate_limit=cfg.get("rate_limit", RateLimit.NEWSAPI_REQUESTS_PER_MINUTE),
            cache_ttl_minutes=cfg.get("cache_ttl_minutes", CacheTTL.NEWS // 60),
            timeout_seconds=cfg.get("timeout_seconds", 10)
        )
    
    def _build_social_config(self, cfg: Dict[str, Any]) -> SocialMediaConfig:
        """Build Social Media configuration."""
        reddit_cfg = cfg.get("reddit", {})
        stocktwits_cfg = cfg.get("stocktwits", {})
        
        # Handle time filter
        time_filter = reddit_cfg.get("time_filter", "week")
        if isinstance(time_filter, str):
            try:
                time_filter_enum = TimeFilter(time_filter)
            except ValueError:
                time_filter_enum = TimeFilter.WEEK
        else:
            time_filter_enum = TimeFilter.WEEK
        
        return SocialMediaConfig(
            enabled=cfg.get("enabled", True),
            twitter_bearer_token=cfg.get("twitter_bearer_token"),
            reddit_user_agent=reddit_cfg.get("user_agent", "TradingBot/1.0"),
            reddit_limit=reddit_cfg.get("limit", 100),
            reddit_time_filter=time_filter_enum,
            reddit_subreddits=reddit_cfg.get("subreddits", [
                "wallstreetbets", "stocks", "investing", "stockmarket", "options"
            ]),
            reddit_comment_limit=reddit_cfg.get("comment_limit", 10),
            reddit_sort=reddit_cfg.get("sort", "relevance"),
            stocktwits_limit=stocktwits_cfg.get("limit", 30),
            platforms=cfg.get("platforms", ["reddit", "stocktwits"]),
            lookback_hours=cfg.get("lookback_hours", 24),
            request_delay=cfg.get("request_delay", 2.0),
            rate_limit=cfg.get("rate_limit", RateLimit.REDDIT_REQUESTS_PER_MINUTE),
            cache_ttl_minutes=cfg.get("cache_ttl_minutes", CacheTTL.SOCIAL // 60),
            timeout_seconds=cfg.get("timeout_seconds", 10)
        )
    
    def _build_sec_config(self, cfg: Dict[str, Any]) -> SECFilingsConfig:
        """Build SEC Filings configuration."""
        return SECFilingsConfig(
            enabled=cfg.get("enabled", True),
            user_agent=cfg.get("user_agent", "TradingBot/1.0"),
            base_url=cfg.get("base_url", "https://www.sec.gov"),
            filing_types=cfg.get("filing_types", ["10-K", "10-Q", "8-K", "4", "13F-HR"]),
            known_ciks=cfg.get("known_ciks", {}),
            rate_limit=cfg.get("rate_limit", RateLimit.SEC_REQUESTS_PER_SECOND),
            cache_ttl_minutes=cfg.get("cache_ttl_hours", 24) * 60,
            timeout_seconds=cfg.get("timeout_seconds", 15)
        )
    
    def _build_options_config(self, cfg: Dict[str, Any]) -> OptionsFlowConfig:
        """Build Options Flow configuration."""
        return OptionsFlowConfig(
            enabled=cfg.get("enabled", True),
            fmp_key=cfg.get("fmp_key"),
            volume_threshold=cfg.get("volume_threshold", 2.0),
            premium_threshold=cfg.get("premium_threshold", 100000),
            use_alternative=cfg.get("use_alternative", True),
            alternative_source=cfg.get("alternative_source", "marketbeat"),
            rate_limit=cfg.get("rate_limit", RateLimit.FMP_REQUESTS_PER_MINUTE),
            cache_ttl_minutes=cfg.get("cache_ttl_minutes", CacheTTL.OPTIONS // 60),
            timeout_seconds=cfg.get("timeout_seconds", 10)
        )
    
    def _build_macro_config(self, cfg: Dict[str, Any]) -> MacroDataConfig:
        """Build Macro Data configuration."""
        return MacroDataConfig(
            enabled=cfg.get("enabled", True),
            fred_api_key=cfg.get("fred_api_key"),
            indicators=cfg.get("indicators", {}),
            rate_limit=cfg.get("rate_limit", RateLimit.FRED_REQUESTS_PER_MINUTE),
            cache_ttl_minutes=cfg.get("cache_ttl_hours", 24) * 60,
            timeout_seconds=cfg.get("timeout_seconds", 15)
        )
    
    def _build_nlp_config(self, cfg: Dict[str, Any]) -> NLPExtractorConfig:
        """Build NLP Extractor configuration."""
        return NLPExtractorConfig(
            spacy_model=cfg.get("spacy_model", "en_core_web_sm"),
            entity_types=cfg.get("entity_types", [
                "PERSON", "ORG", "GPE", "MONEY", "DATE", "PERCENT"
            ])
        )
    
    def _build_regex_config(self, cfg: Dict[str, Any]) -> RegexExtractorConfig:
        """Build Regex Extractor configuration."""
        return RegexExtractorConfig(
            ticker_pattern=cfg.get("ticker_pattern", r'\b[A-Z]{1,5}\b'),
            exclude_tickers=cfg.get("exclude_tickers", [
                "A", "I", "CEO", "CFO", "CTO", "IPO", "ETF", "GDP", "CPI", "NYSE", "NASDAQ"
            ])
        )
    
    def _build_enricher_config(self, cfg: Dict[str, Any]) -> EnricherConfig:
        """Build Data Enricher configuration."""
        return EnricherConfig(
            cache_ttl_minutes=cfg.get("cache_ttl_minutes", 60),
            price_cache_ttl_minutes=cfg.get("price_cache_ttl_minutes", 5),
            price_source=cfg.get("price_source", "yahoo"),
            fundamentals_source=cfg.get("fundamentals_source", "sec")
        )
    
    def _create_default_config(self) -> DiscoveryConfig:
        """Create default configuration."""
        print("📝 Creating default configuration...")
        
        # Default Tavily config
        tavily = TavilyConfig(
            enabled=True,
            api_key=os.getenv("TAVILY_API_KEY"),
            base_url="https://api.tavily.com",
            search_depth=SearchDepth.BASIC,
            max_results=10,
            include_answer=True,
            include_raw_content=False,
            rate_limit=RateLimit.TAVILY_REQUESTS_PER_MINUTE,
            cache_ttl_minutes=CacheTTL.TAVILY // 60
        )
        
        # Default News config
        news = NewsAPIConfig(
            enabled=True,
            news_api_key=os.getenv("NEWS_API_KEY"),
            alpha_vantage_key=os.getenv("ALPHA_VANTAGE_KEY"),
            fmp_key=os.getenv("FMP_KEY"),
            lookback_days=7,
            rate_limit=RateLimit.NEWSAPI_REQUESTS_PER_MINUTE,
            cache_ttl_minutes=CacheTTL.NEWS // 60
        )
        
        # Default Social config
        social = SocialMediaConfig(
            enabled=True,
            reddit_user_agent="TradingBot/1.0",
            reddit_limit=100,
            reddit_time_filter=TimeFilter.WEEK,
            reddit_subreddits=["wallstreetbets", "stocks", "investing"],
            platforms=["reddit", "stocktwits"],
            rate_limit=RateLimit.REDDIT_REQUESTS_PER_MINUTE,
            cache_ttl_minutes=CacheTTL.SOCIAL // 60
        )
        
        # Default SEC config
        sec = SECFilingsConfig(
            enabled=True,
            user_agent="TradingBot/1.0",
            rate_limit=RateLimit.SEC_REQUESTS_PER_SECOND,
            cache_ttl_minutes=CacheTTL.DAY // 60
        )
        
        # Default Options config
        options = OptionsFlowConfig(
            enabled=True,
            fmp_key=os.getenv("FMP_KEY"),
            rate_limit=RateLimit.FMP_REQUESTS_PER_MINUTE,
            cache_ttl_minutes=CacheTTL.OPTIONS // 60
        )
        
        # Default Macro config
        macro = MacroDataConfig(
            enabled=True,
            fred_api_key=os.getenv("FRED_API_KEY"),
            rate_limit=RateLimit.FRED_REQUESTS_PER_MINUTE,
            cache_ttl_minutes=CacheTTL.DAY // 60
        )
        
        # Default extractor configs
        nlp = NLPExtractorConfig()
        regex = RegexExtractorConfig()
        enricher = EnricherConfig()
        
        return DiscoveryConfig(
            tavily=tavily,
            news=news,
            social=social,
            sec=sec,
            options=options,
            macro=macro,
            nlp=nlp,
            regex=regex,
            enricher=enricher,
            max_workers=5,
            cache_ttl_minutes=15
        )
    
    def reload(self) -> DiscoveryConfig:
        """Reload configuration from file."""
        self._config_cache = None
        return self.load()


# =============================================================================
# Global instance and convenience functions
# =============================================================================

# Singleton loader instance
_loader = DiscoveryConfigLoader()


def get_discovery_config() -> DiscoveryConfig:
    """
    Get discovery configuration.
    
    Returns:
        DiscoveryConfig object loaded from YAML or default
        
    Usage:
        from discovery.config.loader import get_discovery_config
        config = get_discovery_config()
        print(config.tavily.api_key)
    """
    return _loader.load()


def reload_discovery_config(config_dir: Optional[Union[str, Path]] = None) -> DiscoveryConfig:
    """
    Reload discovery configuration.
    
    Args:
        config_dir: Optional custom config directory
        
    Returns:
        Reloaded DiscoveryConfig object
    """
    if config_dir:
        new_loader = DiscoveryConfigLoader(config_dir)
        return new_loader.load()
    else:
        return _loader.reload()


def print_config_summary(config: Optional[DiscoveryConfig] = None) -> None:
    """
    Print a summary of the discovery configuration.
    
    Args:
        config: DiscoveryConfig object (uses default if None)
    """
    if config is None:
        config = get_discovery_config()
    
    print("\n" + "=" * 60)
    print(" DISCOVERY CONFIGURATION SUMMARY")
    print("=" * 60)
    
    # Tavily
    print(f"\n📡 Tavily:")
    print(f"   Enabled: {config.tavily.enabled}")
    print(f"   API Key: {'✓ Set' if config.tavily.api_key else '✗ Missing'}")
    print(f"   Max Results: {config.tavily.max_results}")
    print(f"   Search Depth: {config.tavily.search_depth}")
    
    # News
    print(f"\n📰 News:")
    print(f"   Enabled: {config.news.enabled}")
    print(f"   NewsAPI Key: {'✓ Set' if config.news.news_api_key else '✗ Missing'}")
    print(f"   Alpha Vantage: {'✓ Set' if config.news.alpha_vantage_key else '✗ Missing'}")
    print(f"   FMP Key: {'✓ Set' if config.news.fmp_key else '✗ Missing'}")
    print(f"   Lookback Days: {config.news.lookback_days}")
    
    # Social
    print(f"\n📱 Social Media:")
    print(f"   Enabled: {config.social.enabled}")
    print(f"   Platforms: {', '.join(config.social.platforms)}")
    print(f"   Reddit Subreddits: {len(config.social.reddit_subreddits)}")
    print(f"   Reddit Time Filter: {config.social.reddit_time_filter}")
    
    # SEC
    print(f"\n📄 SEC Filings:")
    print(f"   Enabled: {config.sec.enabled}")
    print(f"   Filing Types: {len(config.sec.filing_types)}")
    
    # Options
    print(f"\n📊 Options Flow:")
    print(f"   Enabled: {config.options.enabled}")
    print(f"   Volume Threshold: {config.options.volume_threshold}x")
    print(f"   Premium Threshold: ${config.options.premium_threshold:,.0f}")
    
    # Macro
    print(f"\n📈 Macro Data:")
    print(f"   Enabled: {config.macro.enabled}")
    print(f"   FRED Key: {'✓ Set' if config.macro.fred_api_key else '✗ Missing'}")
    print(f"   Indicators: {len(config.macro.indicators)}")
    
    # General
    print(f"\n⚙️ General:")
    print(f"   Max Workers: {config.max_workers}")
    print(f"   Cache TTL: {config.cache_ttl_minutes} minutes")
    
    print("=" * 60)


if __name__ == "__main__":
    # Test the config loader
    print("Testing Discovery Config Loader...\n")
    
    config = get_discovery_config()
    print_config_summary(config)
    
    print("\n✅ Config loaded successfully!")