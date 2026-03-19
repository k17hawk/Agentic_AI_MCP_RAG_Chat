"""
Data Enricher - Enriches discovered data with additional context
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import yfinance as yf
import asyncio

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.discovery.entity_extractor.regex_extractor import RegexExtractor

class DataEnricher:
    """
    Enriches discovered data with additional context

    Enrichments:
    - Company information (sector, industry, market cap)
    - Current stock price and performance
    - Related news and social sentiment
    - Historical context
    - Relevance scoring
    """

    # Known valid tickers cache — avoids repeat yfinance 404s for bad tickers
    _invalid_ticker_cache = set()

    # Common English words that look like tickers but are not.
    # This is the LAST safety net before a yfinance call — keep it comprehensive.
    # Words seen in logs to cause 404s (IDEAS, QUOTE, CCPA) are included explicitly.
    _COMMON_WORDS = {
        # Articles / pronouns / conjunctions
        'THE', 'AND', 'FOR', 'BUT', 'NOT', 'ITS', 'ARE', 'WAS',
        'HAS', 'ALL', 'NEW', 'TOP', 'BIG', 'OLD', 'OUR', 'GET',
        'GOT', 'LET', 'PUT', 'SET', 'USE', 'MAY', 'CAN', 'DID',
        'SAY', 'HIT', 'RUN', 'BUY', 'NOW', 'HOW', 'WHY', 'WHO',
        'THEY', 'THAN', 'BEEN', 'ALSO', 'MORE', 'WILL', 'YOUR',
        'HAVE', 'FROM', 'WITH', 'THAT', 'THIS', 'ABOUT', 'WOULD',
        'COULD', 'THEIR', 'THERE', 'THESE', 'THOSE', 'WHICH',
        'JUST', 'ONLY', 'EVEN', 'THEN', 'WHEN', 'BOTH', 'EACH',
        'SUCH', 'VERY', 'MUCH', 'MANY', 'MOST', 'SOME', 'INTO',
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
        'FINRA', 'CFTC', 'FDIC', 'FED', 'ECB',
        # Market / trading words
        'STOCK', 'STOCKS', 'SHARE', 'SHARES', 'PRICE', 'MARKET', 'MARKETS',
        'FUND', 'FUNDS', 'BOND', 'BONDS', 'TRADE', 'TRADES', 'CHART', 'CHARTS',
        'NEWS', 'DATA', 'HIGH', 'LOW', 'OPEN', 'CLOSE',
        'CALL', 'CALLS', 'BULL', 'BEAR', 'LONG', 'SHORT',
        # Price-action and analysis words that appear adjacent to stock/price/trading
        'GAIN', 'GAINS', 'LOSS', 'RALLY', 'SURGE', 'DROP', 'FALL',
        'RISE', 'JUMP', 'SLIDE', 'SPIKE', 'DIP', 'PEAK', 'FLAT',
        'UP', 'DOWN', 'RANGE', 'LEVEL', 'MARK', 'BASE', 'ZONE',
        'MOVE', 'SETUP', 'PLAY', 'IDEA', 'PICK', 'RANK', 'SCAN',
        'ALERT', 'WATCH', 'BREAK', 'CROSS', 'STOP', 'LIMIT', 'CAP',
        'FLOOR', 'GRAPH', 'TREND', 'INFO', 'SITE', 'PAGE', 'LIST', 'FULL',
        # Geopolitical / news words near trading/price in headlines
        'WAR', 'WARS', 'DEAL', 'RISK', 'FEAR', 'TALK', 'TALKS',
        'PLAN', 'BILL', 'ACT', 'LAW', 'RULE',
        # Time / recency words
        'WEEK', 'MONTH', 'YEAR', 'DAILY', 'LIVE', 'PRE', 'POST', 'LATE', 'EARLY',
        # UI / meta words
        'REAL', 'BEST', 'TOP', 'NEW', 'OLD', 'KEY', 'MAIN', 'PLUS', 'PRO', 'API', 'APP',
        # Common web / UI tokens present in scraped Tavily content
        # (seen in logs: IDEAS, QUOTE, CCPA)
        'IDEAS', 'IDEA', 'QUOTE', 'QUOTES', 'VIEW', 'VIEWS',
        'ALERT', 'ALERTS', 'LOGIN', 'SIGN', 'TERMS', 'HELP',
        'ABOUT', 'HOME', 'BACK', 'NEXT', 'PREV', 'READ',
        'SHOW', 'HIDE', 'FULL', 'LIVE', 'FREE', 'MORE', 'LESS',
        # Privacy / legal boilerplate
        'CCPA', 'GDPR', 'DMCA', 'EULA', 'TOS',
        'TOTAL', 'RETURN', 'VOLUME', 'VALUE', 'RATIO', 'SCORE', 'RATE',
        'VS', 'VERSUS', 'TODAY', 'YESTERDAY', 'TOMORROW', 'SINCE', 'AFTER', 'BEFORE', 'DURING', 'WHILE',
        'INDEX', 'YIELD', 'BETA', 'DELTA', 'GAMMA', 'THETA', 'SIGMA',
        # Full company names (the ticker is different)
        'APPLE', 'GOOGLE', 'META', 'TESLA', 'AMAZON', 'MICROSOFT',
        'NVIDIA', 'DISNEY', 'NETFLIX', 'INTEL', 'CISCO', 'ORACLE',
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # Cache for company info
        self.company_cache = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 60) * 60

        # Shorter TTL for price data
        self.price_cache_ttl = config.get("price_cache_ttl_minutes", 5) * 60

        # Entity extractor for ticker detection
        self.extractor = RegexExtractor(config)

        logging.info(f"✅ DataEnricher initialized")

    async def enrich(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enrich a single data item with additional context
        """
        enriched = item.copy()

        # Extract tickers from content
        content = f"{item.get('title', '')} {item.get('content', '')}"
        raw_tickers = await self.extractor.extract_tickers(content)

        # FIX: Filter out common words and previously failed tickers
        # before making any yfinance calls
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
        Enrich multiple items in parallel
        """
        tasks = [self.enrich(item) for item in items]
        return await asyncio.gather(*tasks)

    async def _get_company_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get company information from yfinance.
        Skips invalid tickers and caches failures to avoid repeat 404s.
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
            stock = yf.Ticker(ticker)
            info = stock.info

            # yfinance returns a minimal dict with just {trailingPegRatio: None}
            # for invalid tickers — check for a real field to confirm it's valid
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

        except Exception as e:
            logging.debug(f"Error getting company info for {ticker}: {e}")
            # Cache the failure so we don't retry on every search
            self._invalid_ticker_cache.add(ticker)

        return None

    def _calculate_relevance(self, item: Dict[str, Any], company_info: Dict) -> float:
        """
        Calculate relevance score for this item
        """
        score = 0.5  # Base score

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

        # Recency boost (if available)
        published = item.get("published_at")
        if published:
            try:
                pub_date = datetime.fromisoformat(published.replace("Z", "+00:00"))
                age_hours = (datetime.now() - pub_date).total_seconds() / 3600
                if age_hours < 24:
                    score += 0.1
                elif age_hours < 72:
                    score += 0.05
            except Exception:
                pass

        return float(min(1.0, score))

    def _get_source_authority(self, source: str) -> float:
        """
        Get authority score for a source
        """
        source_lower = source.lower()

        high_authority = {
            "reuters", "bloomberg", "wsj", "wall street journal",
            "financial times", "ft.com", "nytimes", "economist"
        }
        medium_authority = {
            "cnbc", "yahoo", "seeking alpha", "marketwatch",
            "forbes", "business insider", "investopedia"
        }
        low_authority = {
            "twitter", "reddit", "stocktwits", "facebook",
            "linkedin", "medium", "wordpress"
        }

        for src in high_authority:
            if src in source_lower:
                return 1.0
        for src in medium_authority:
            if src in source_lower:
                return 0.7
        for src in low_authority:
            if src in source_lower:
                return 0.3

        return 0.5  # Default

    def _classify_content(self, item: Dict[str, Any]) -> str:
        """
        Classify content type
        """
        title = item.get("title", "").lower()
        content = item.get("content", "")[:200].lower()
        combined = title + " " + content

        if any(word in combined for word in ["earnings", "revenue", "profit", "loss"]):
            return "earnings"
        elif any(word in combined for word in ["merger", "acquisition", "buyout", "takeover"]):
            return "ma"
        elif any(word in combined for word in ["upgrade", "downgrade", "rating", "target"]):
            return "analyst_rating"
        elif any(word in combined for word in ["ipo", "offering", "listing"]):
            return "ipo"
        elif any(word in combined for word in ["dividend", "buyback", "split"]):
            return "corporate_action"
        elif any(word in combined for word in ["sec", "investigation", "lawsuit", "regulatory"]):
            return "regulatory"
        elif any(word in combined for word in ["ceo", "cfo", "executive", "management"]):
            return "management"
        elif any(word in combined for word in ["price", "target", "forecast", "prediction"]):
            return "price_target"
        else:
            return "general"