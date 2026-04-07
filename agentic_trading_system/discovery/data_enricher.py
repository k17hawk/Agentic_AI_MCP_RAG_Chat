from pathlib import Path
# =============================================================================
# discovery/data_enricher.py (UPDATED)
# =============================================================================
"""
Data Enricher - Enriches discovered data with additional context
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import yfinance as yf
import asyncio

from agentic_trading_system.utils.logger import logger as logging

# Import from new config structure
from agentic_trading_system.constants import Source, EntityExtraction, ContentType
from agentic_trading_system.config.config__entity import EnricherConfig
from agentic_trading_system.discovery.entity_extractor.regex_extractor import RegexExtractor


class DataEnricher:
    """
    Enriches discovered data with additional context.

    Enrichments:
    - Company information (sector, industry, market cap)
    - Current stock price and performance
    - Related news and social sentiment
    - Historical context
    - Relevance scoring
    """

    # Known valid tickers cache
    _invalid_ticker_cache = set()

    # Common English words that look like tickers
    _COMMON_WORDS = {
        # Articles / pronouns / conjunctions
        'THE', 'AND', 'FOR', 'BUT', 'NOT', 'ITS', 'ARE', 'WAS',
        'HAS', 'ALL', 'NEW', 'TOP', 'BIG', 'OLD', 'OUR', 'GET',
        'GOT', 'LET', 'PUT', 'SET', 'USE', 'MAY', 'CAN', 'DID',
        'SAY', 'HIT', 'RUN', 'BUY', 'NOW', 'HOW', 'WHY', 'WHO',
        'THEY', 'THAN', 'BEEN', 'ALSO', 'MORE', 'WILL', 'YOUR',
        'HAVE', 'FROM', 'WITH', 'THAT', 'THIS', 'ABOUT', 'WOULD',
        'COULD', 'THEIR', 'THERE', 'THESE', 'THOSE', 'WHICH',

        # Business / finance abbreviations
        'INC', 'LLC', 'LTD', 'CORP', 'CO', 'PLC', 'LP', 'SA', 'AG', 'NV',
        'ETF', 'IPO', 'SEC', 'CEO', 'CFO', 'COO', 'CTO', 'CIO',
        'EPS', 'YOY', 'QOQ', 'TTM', 'ATH', 'ATL',
        'GDP', 'CPI', 'PMI', 'ESG', 'ROI', 'ROE', 'DCF', 'EBITDA',
        'SPAC', 'REIT', 'REITS',

        # Currencies
        'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD', 'CHF',
        'HKD', 'SGD', 'NZD', 'KRW', 'INR',

        # Exchanges / regulators
        'NYSE', 'NASDAQ', 'AMEX', 'LSE', 'TSX', 'HKEX', 'ASX',

        # Market / trading words
        'STOCK', 'STOCKS', 'SHARE', 'SHARES', 'PRICE', 'MARKET', 'MARKETS',
        'FUND', 'FUNDS', 'BOND', 'BONDS', 'TRADE', 'TRADES', 'CHART', 'CHARTS',
        'NEWS', 'DATA', 'HIGH', 'LOW', 'OPEN', 'CLOSE',
        'CALL', 'CALLS', 'BULL', 'BEAR', 'LONG', 'SHORT',

        # UI / meta words
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

    def __init__(self, config: EnricherConfig):
        """
        Initialize data enricher.
        
        Args:
            config: EnricherConfig object
        """
        self.config = config

        # Cache for company info
        self.company_cache: Dict = {}
        self.cache_ttl = config.cache_ttl_minutes * 60

        # Shorter TTL for price data
        self.price_cache_ttl = config.price_cache_ttl_minutes * 60

        # Entity extractor for ticker detection
        self.extractor = None  # Will be set later if needed

        logging.info(f"✅ DataEnricher initialized")

    def set_extractor(self, extractor: RegexExtractor):
        """Set the regex extractor for ticker detection."""
        self.extractor = extractor

    async def enrich(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single data item with additional context.
        
        Args:
            item: Raw data item
            
        Returns:
            Enriched data item
        """
        enriched = item.copy()

        # Extract tickers from content
        content = f"{item.get('title', '')} {item.get('content', '')}"
        raw_tickers = await self._extract_tickers(content)

        # Filter out common words and previously failed tickers
        tickers = [
            t for t in raw_tickers
            if t not in self._COMMON_WORDS
            and t not in self._invalid_ticker_cache
        ]

        if tickers:
            enriched["detected_tickers"] = tickers

            # Get company info for the first valid ticker
            primary_ticker = tickers[0]
            company_info = await self._get_company_info(primary_ticker)
            if company_info:
                enriched["company_info"] = company_info

                # Calculate relevance score
                enriched["relevance_score"] = self._calculate_relevance(item, company_info)

        # Add timestamp if not present
        if "timestamp" not in enriched:
            enriched["timestamp"] = datetime.now().isoformat()

        # Add content length and word count
        if "content" in enriched:
            enriched["content_length"] = len(enriched["content"])
            enriched["word_count"] = len(enriched["content"].split())

        # Add source authority score
        enriched["source_authority"] = self._get_source_authority(enriched.get("source", ""))

        # Add content type classification
        enriched["content_type"] = self._classify_content(enriched)

        return enriched

    async def enrich_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enrich multiple items in parallel.
        
        Args:
            items: List of raw data items
            
        Returns:
            List of enriched items
        """
        tasks = [self.enrich(item) for item in items]
        return await asyncio.gather(*tasks)

    async def _extract_tickers(self, text: str) -> List[str]:
        """Extract tickers using regex extractor."""
        if self.extractor:
            return await self.extractor.extract_tickers(text)
        return []

    async def _get_company_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get company information from yfinance.
        Skips invalid tickers and caches failures.
        """
        # Skip known invalid tickers
        if ticker in self._invalid_ticker_cache:
            return None

        # Check cache
        if ticker in self.company_cache:
            cached_time, info = self.company_cache[ticker]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                return info

        try:
            loop = asyncio.get_event_loop()

            def _get_info_blocking():
                stock = yf.Ticker(ticker)
                return stock.info

            info = await asyncio.wait_for(
                loop.run_in_executor(None, _get_info_blocking),
                timeout=self.config.price_cache_ttl_minutes
            )

            # Validate info
            if not info or not info.get("symbol") and not info.get("longName") and not info.get("shortName"):
                logging.debug(f"No valid company info for ticker: {ticker}")
                self._invalid_ticker_cache.add(ticker)
                return None

            company_info = {
                "name": info.get("longName", info.get("shortName", ticker)),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "country": info.get("country", "Unknown"),
                "website": info.get("website", ""),
                "market_cap": info.get("marketCap"),
                "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                "day_change": info.get("regularMarketChangePercent"),
                "year_high": info.get("fiftyTwoWeekHigh"),
                "year_low": info.get("fiftyTwoWeekLow"),
                "volume": info.get("volume"),
                "avg_volume": info.get("averageVolume"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "beta": info.get("beta"),
                "updated_at": datetime.now().isoformat()
            }

            # Cache valid result
            self.company_cache[ticker] = (datetime.now().timestamp(), company_info)
            return company_info

        except asyncio.TimeoutError:
            logging.debug(f"yfinance timeout for {ticker}")
            self._invalid_ticker_cache.add(ticker)
        except Exception as e:
            logging.debug(f"Error getting company info for {ticker}: {e}")
            self._invalid_ticker_cache.add(ticker)

        return None

    def _calculate_relevance(self, item: Dict[str, Any], company_info: Dict) -> float:
        """
        Calculate relevance score for this item.
        
        Returns:
            Score between 0 and 1
        """
        score = EntityExtraction.DEFAULT_RELEVANCE

        title = item.get("title", "").lower()
        content = item.get("content", "")[:500].lower()

        # Check for company name in title
        company_name = company_info.get("name", "").lower()
        if company_name and (company_name in title or company_name in content):
            score += 0.2

        # Check for sector/industry terms
        sector = company_info.get("sector", "").lower()
        industry = company_info.get("industry", "").lower()

        if sector and (sector in title or sector in content):
            score += 0.15
        if industry and (industry in title or industry in content):
            score += 0.1

        # Check for financial terms
        financial_terms = ["earnings", "revenue", "profit", "loss", "guidance", "forecast"]
        for term in financial_terms:
            if term in title or term in content:
                score += 0.05

        # Recency boost
        published = item.get("published_at")
        if published:
            try:
                if isinstance(published, str):
                    pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                else:
                    pub_date = published
                age_hours = (datetime.now() - pub_date).total_seconds() / 3600
                if age_hours < 24:
                    score += 0.1
                elif age_hours < 72:
                    score += 0.05
            except Exception:
                pass

        return float(min(EntityExtraction.MAX_RELEVANCE, max(EntityExtraction.MIN_RELEVANCE, score)))

    def _get_source_authority(self, source: str) -> float:
        """
        Get authority score for a source.
        
        Returns:
            Score between 0 and 1
        """
        source_lower = source.lower()

        for src in EntityExtraction.HIGH_AUTHORITY_SOURCES:
            if src in source_lower:
                return EntityExtraction.SOURCE_AUTHORITY_SCORES['high']

        for src in EntityExtraction.MEDIUM_AUTHORITY_SOURCES:
            if src in source_lower:
                return EntityExtraction.SOURCE_AUTHORITY_SCORES['medium']

        for src in EntityExtraction.LOW_AUTHORITY_SOURCES:
            if src in source_lower:
                return EntityExtraction.SOURCE_AUTHORITY_SCORES['low']

        return EntityExtraction.SOURCE_AUTHORITY_SCORES['default']

    def _classify_content(self, item: Dict[str, Any]) -> str:
        """
        Classify content type.
        
        Returns:
            Content type string
        """
        title = item.get("title", "").lower()
        content = item.get("content", "")[:200].lower()
        combined = title + " " + content

        if any(word in combined for word in ["earnings", "revenue", "profit", "loss"]):
            return ContentType.EARNINGS
        elif any(word in combined for word in ["merger", "acquisition", "buyout", "takeover"]):
            return ContentType.MA
        elif any(word in combined for word in ["upgrade", "downgrade", "rating", "target"]):
            return ContentType.ANALYST_RATING
        elif any(word in combined for word in ["ipo", "offering", "listing"]):
            return ContentType.IPO
        elif any(word in combined for word in ["dividend", "buyback", "split"]):
            return ContentType.CORPORATE_ACTION
        elif any(word in combined for word in ["sec", "investigation", "lawsuit", "regulatory"]):
            return ContentType.REGULATORY
        elif any(word in combined for word in ["ceo", "cfo", "executive", "management"]):
            return ContentType.MANAGEMENT
        elif any(word in combined for word in ["price", "target", "forecast", "prediction"]):
            return ContentType.PRICE_TARGET
        else:
            return ContentType.GENERAL

    async def clear_cache(self) -> None:
        """Clear all caches."""
        self.company_cache.clear()
        self._invalid_ticker_cache.clear()
        logging.info("🧹 Data enricher cache cleared")