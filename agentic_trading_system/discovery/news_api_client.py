# =============================================================================
# discovery/news_api_client.py (UPDATED - Partial to show key changes)
# =============================================================================
"""
News API Client - Financial news aggregation
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio
import re

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

# Import from new config structure
from agentic_trading_system.constants import Source, RateLimit, CacheTTL, ScoringWeights
from agentic_trading_system.config.config__entity import NewsAPIConfig


class NewsAPIClient:
    """
    Aggregates news from multiple financial news sources.
    Supports: NewsAPI, Alpha Vantage, Financial Modeling Prep
    """

    def __init__(self, config: NewsAPIConfig):
        """
        Initialize News API client.
        
        Args:
            config: NewsAPIConfig object
        """
        self.config = config

        # API keys
        self.news_api_key = config.news_api_key
        self.alpha_vantage_key = config.alpha_vantage_key
        self.fmp_key = config.fmp_key

        # Rate limiting
        self.rate_limit = config.rate_limit
        self.request_timestamps: List[float] = []

        # Caches
        self.cache: Dict = {}
        self.cache_ttl = config.cache_ttl_minutes * 60
        self.company_name_cache: Dict[str, Optional[str]] = {}

        # Common tickers
        self.common_tickers = self._build_common_tickers()

        logging.info("✅ NewsAPIClient initialized")

    def _build_common_tickers(self) -> Dict[str, str]:
        """Build common ticker mapping."""
        return {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google',
            'GOOG': 'Google', 'AMZN': 'Amazon', 'META': 'Meta',
            'TSLA': 'Tesla', 'NVDA': 'NVIDIA', 'NFLX': 'Netflix',
            'AMD': 'AMD', 'INTC': 'Intel', 'IBM': 'IBM',
            'ORCL': 'Oracle', 'CRM': 'Salesforce', 'ADBE': 'Adobe',
            'DIS': 'Disney', 'V': 'Visa', 'MA': 'Mastercard',
            'JPM': 'JPMorgan Chase', 'BAC': 'Bank of America',
            'WMT': 'Walmart', 'TGT': 'Target', 'KO': 'Coca-Cola',
            'PEP': 'PepsiCo', 'MCD': "McDonald's", 'SBUX': 'Starbucks',
            'NKE': 'Nike', 'LIN': 'Linde', 'UNH': 'UnitedHealth',
            'HD': 'Home Depot', 'PG': 'Procter & Gamble', 'CVX': 'Chevron',
            'MRK': 'Merck', 'ABBV': 'AbbVie', 'PFE': 'Pfizer',
            'CSCO': 'Cisco', 'CMCSA': 'Comcast', 'COST': 'Costco',
        }

    @retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """Search for news articles with smart fallbacks."""
        options = options or {}
        logging.info(f"📰 News search for: '{query}'")

        # Cache check
        cache_key = f"news_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached news results for '{query}'")
                return cached_result

        await self._rate_limit()

        all_articles: List[Dict] = []
        sources_used: List[str] = []
        errors: List[str] = []

        # Build task pairs
        task_pairs = []
        if self.news_api_key:
            task_pairs.append((Source.NEWS, self._fetch_from_newsapi(query, options)))
        if self.alpha_vantage_key:
            task_pairs.append((Source.ALPHA_VANTAGE, self._fetch_from_alphavantage(query, options)))
        if self.fmp_key:
            task_pairs.append((Source.FMP, self._fetch_from_fmp(query, options)))

        if not task_pairs:
            logging.warning("⚠️ No news API keys configured")
            return {"items": [], "metadata": {"error": "No API keys configured"}}

        source_names = [name for name, _ in task_pairs]
        tasks = [coro for _, coro in task_pairs]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for source_name, result in zip(source_names, results):
            if isinstance(result, Exception):
                msg = f"{source_name}: {result}"
                logging.warning(f"⚠️ News source error — {msg}")
                errors.append(msg)
            elif isinstance(result, list):
                sources_used.append(source_name)
                if result:
                    all_articles.extend(result)
                    logging.info(f"✅ {source_name}: {len(result)} articles")
                else:
                    logging.info(f"ℹ️ {source_name}: 0 articles returned")

        unique_articles = self._deduplicate(all_articles)
        unique_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)

        result = {
            "items": unique_articles[:options.get("max_results", 50)],
            "metadata": {
                "total_found": len(all_articles),
                "unique_count": len(unique_articles),
                "sources_used": sources_used,
                "errors": errors or None,
            },
        }

        self.cache[cache_key] = (datetime.now().timestamp(), result)
        logging.info(f"✅ News: {len(unique_articles)} unique articles from {sources_used}")
        return result

    async def _fetch_from_newsapi(self, query: str, options: Dict) -> List[Dict]:
        """Fetch from NewsAPI with query-variation fallback."""
        if not self.news_api_key:
            return []

        everything_url = "https://newsapi.org/v2/everything"
        headlines_url = "https://newsapi.org/v2/top-headlines"
        days_back = options.get("days_back", self.config.lookback_days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        target = options.get("min_articles", 10)

        use_headlines_only = False
        query_variations = await self._generate_query_variations(query, options)
        all_articles: List[Dict] = []

        source_mapping = {
            "reuters": "reuters", "bloomberg": "bloomberg",
            "cnbc": "cnbc", "wsj": "the-wall-street-journal",
            "financial-times": "financial-times",
            "yahoo-finance": "yahoo-finance",
            "seeking-alpha": "seeking-alpha",
            "marketwatch": "marketwatch",
        }

        for variation in query_variations:
            if len(all_articles) >= target:
                break

            if not use_headlines_only:
                params: Dict = {
                    "q": variation,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "language": options.get("language", self.config.language),
                    "sortBy": options.get("sort_by", "publishedAt"),
                    "pageSize": min(options.get("page_size", self.config.page_size), 100),
                    "apiKey": self.news_api_key,
                }
                
                if "news_sources" in options and options["news_sources"]:
                    mapped = [source_mapping[s] for s in options["news_sources"]
                              if s in source_mapping]
                    if mapped:
                        params["sources"] = ",".join(mapped)

                try:
                    articles = await self._newsapi_get(everything_url, params, variation)
                    all_articles.extend(articles)
                    await asyncio.sleep(0.15)
                    continue
                except Exception as e:
                    if "plan" in str(e).lower() or "upgrade" in str(e).lower():
                        logging.warning("⚠️ NewsAPI plan restricts /v2/everything — switching to headlines")
                        use_headlines_only = True
                    elif "429" in str(e):
                        break

            # Top headlines fallback
            hl_params: Dict = {
                "q": variation,
                "language": options.get("language", self.config.language),
                "pageSize": min(options.get("page_size", self.config.page_size), 100),
                "apiKey": self.news_api_key,
            }
            
            if "news_sources" in options and options["news_sources"]:
                mapped = [source_mapping[s] for s in options["news_sources"]
                          if s in source_mapping]
                if mapped:
                    hl_params["sources"] = ",".join(mapped)
            else:
                hl_params["country"] = "us"

            try:
                articles = await self._newsapi_get(headlines_url, hl_params, variation)
                all_articles.extend(articles)
            except Exception:
                break

            await asyncio.sleep(0.15)

        return all_articles

    async def _newsapi_get(self, url: str, params: Dict, variation: str) -> List[Dict]:
        """Single GET against NewsAPI endpoint."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") != "ok":
                            code = data.get("code", "unknown")
                            logging.warning(f"⚠️ NewsAPI error for '{variation}': {code}")
                            return []
                        
                        articles = self._parse_newsapi_articles(data.get("articles", []), variation)
                        logging.info(f"✅ NewsAPI '{variation}': {len(articles)} articles")
                        return articles

                    elif response.status == 429:
                        logging.warning(f"⚠️ NewsAPI rate limit for '{variation}'")
                        raise Exception("Rate limit")
                    else:
                        return []

        except asyncio.TimeoutError:
            logging.warning(f"⚠️ NewsAPI timeout for '{variation}'")
            return []
        except Exception as e:
            logging.warning(f"⚠️ NewsAPI exception: {e}")
            return []

    def _parse_newsapi_articles(self, raw_articles: List[Dict], variation: str) -> List[Dict]:
        """Parse and normalize NewsAPI articles."""
        articles = []
        for article in raw_articles:
            title = article.get("title") or ""
            if "[Removed]" in title or not title.strip():
                continue
            content = article.get("content") or article.get("description") or ""
            if content and "[+" in content:
                content = content.split("[+")[0].strip()
            articles.append({
                "title": title,
                "description": article.get("description", ""),
                "content": content,
                "url": article.get("url", ""),
                "source": article.get("source", {}).get("name", "unknown"),
                "published_at": article.get("publishedAt", ""),
                "author": article.get("author", ""),
                "image_url": article.get("urlToImage", ""),
                "api_source": Source.NEWS,
                "matched_query": variation,
            })
        return articles

    async def _generate_query_variations(self, query: str, options: Dict) -> List[str]:
        """Generate query variations for better recall."""
        original = query.strip()
        variations: List[str] = []

        is_ticker = original.isupper() and 1 <= len(original) <= 5 and original.isalpha()

        if is_ticker:
            ticker = original
            company_name = await self._get_company_name(ticker)

            if company_name:
                variations.append(f'"{company_name}"')
                variations.append(f"{company_name} stock")
                variations.append(f"{company_name} news")
                variations.append(f"{company_name} earnings")

            variations.append(f"{ticker} stock")
            variations.append(f"${ticker}")
            variations.append(ticker)

        elif " " in original:
            variations.append(f'"{original}"')
            variations.append(original)
            words = original.split()
            if len(words) > 2:
                variations.append(" ".join(words[:2]))
        else:
            variations.append(f'"{original}"')
            variations.append(original)

        seen: set = set()
        unique: List[str] = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique.append(v)

        return unique[:8]

    async def _get_company_name(self, ticker: str) -> Optional[str]:
        """Resolve company name from ticker."""
        ticker = ticker.upper()

        # Check common tickers
        if ticker in self.common_tickers:
            return self.common_tickers[ticker]

        # Check cache
        if ticker in self.company_name_cache:
            return self.company_name_cache[ticker]

        # Try Alpha Vantage
        if self.alpha_vantage_key:
            name = await self._get_company_from_alphavantage(ticker)
            if name:
                self.company_name_cache[ticker] = name
                return name

        # Try yfinance
        name = await self._get_company_from_yfinance(ticker)
        if name:
            self.company_name_cache[ticker] = name
            return name

        self.company_name_cache[ticker] = None
        return None

    async def _get_company_from_alphavantage(self, ticker: str) -> Optional[str]:
        """Get company name from Alpha Vantage."""
        url = "https://www.alphavantage.co/query"
        params = {"function": "OVERVIEW", "symbol": ticker, "apikey": self.alpha_vantage_key}

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        name = data.get("Name", "")
                        if name and name != "None":
                            return self._strip_legal_suffix(name)
        except Exception:
            pass
        return None

    async def _get_company_from_yfinance(self, ticker: str) -> Optional[str]:
        """Get company name from yfinance (runs in executor)."""
        try:
            loop = asyncio.get_event_loop()

            def _blocking_lookup() -> Optional[str]:
                import yfinance as yf
                info = yf.Ticker(ticker).info
                return info.get("longName") or info.get("shortName")

            name = await asyncio.wait_for(
                loop.run_in_executor(None, _blocking_lookup),
                timeout=5.0
            )
            if name:
                return self._strip_legal_suffix(name)
        except Exception:
            pass
        return None

    async def _fetch_from_alphavantage(self, query: str, options: Dict) -> List[Dict]:
        """Fetch from Alpha Vantage News Sentiment API."""
        if not self.alpha_vantage_key:
            return []

        ticker = self._extract_ticker(query)
        if not ticker:
            return []

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": self.alpha_vantage_key,
            "limit": options.get("limit", 50),
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as response:
                    if response.status == 200:
                        data = await response.json()

                        if "Information" in data:
                            logging.warning(f"⚠️ Alpha Vantage: {data['Information'][:120]}")
                            return []
                        if "Error Message" in data:
                            return []

                        feed = data.get("feed", [])
                        articles = []
                        for item in feed[:options.get("max_results", 20)]:
                            time_published = item.get("time_published", "")
                            try:
                                pub_date = datetime.strptime(time_published, "%Y%m%dT%H%M%S")
                                published_at = pub_date.isoformat()
                            except Exception:
                                published_at = datetime.now().isoformat()

                            articles.append({
                                "title": item.get("title", ""),
                                "content": item.get("summary", ""),
                                "url": item.get("url", ""),
                                "source": item.get("source", Source.ALPHA_VANTAGE),
                                "published_at": published_at,
                                "sentiment": item.get("overall_sentiment_score", 0),
                                "sentiment_label": item.get("overall_sentiment_label", "NEUTRAL"),
                                "api_source": Source.ALPHA_VANTAGE,
                            })

                        logging.info(f"✅ Alpha Vantage: {len(articles)} articles for {ticker}")
                        return articles
        except Exception as e:
            logging.warning(f"⚠️ Alpha Vantage exception: {e}")

        return []

    async def _fetch_from_fmp(self, query: str, options: Dict) -> List[Dict]:
        """Fetch from Financial Modeling Prep."""
        if not self.fmp_key:
            return []

        ticker = self._extract_ticker(query)
        if not ticker:
            return []

        limit = options.get("limit", 50)
        endpoints = [
            (f"https://financialmodelingprep.com/stable/news/stock",
             {"symbols": ticker, "limit": limit, "apikey": self.fmp_key}),
            (f"https://financialmodelingprep.com/api/v4/stock_news-sentiments-rss-feed",
             {"tickers": ticker, "limit": limit, "apikey": self.fmp_key}),
        ]

        for url, params in endpoints:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)) as response:
                        if response.status in (403, 404):
                            continue
                        if response.status != 200:
                            continue

                        data = await response.json()
                        if not isinstance(data, list):
                            continue

                        articles = []
                        for item in data[:options.get("max_results", 20)]:
                            articles.append({
                                "title": item.get("title", ""),
                                "content": item.get("text", "") or item.get("content", "") or item.get("summary", ""),
                                "url": item.get("url", ""),
                                "source": item.get("site", "") or item.get("source", "") or Source.FMP,
                                "published_at": item.get("publishedDate") or item.get("publishedAt") or item.get("date", ""),
                                "api_source": Source.FMP,
                            })

                        if articles:
                            logging.info(f"✅ FMP: {len(articles)} articles for {ticker}")
                            return articles
            except Exception as e:
                logging.debug(f"FMP error: {e}")
                continue

        return []

    def _extract_ticker(self, query: str) -> Optional[str]:
        """Extract ticker from query."""
        q = query.strip().upper()
        if q.isalpha() and 1 <= len(q) <= 5:
            return q

        match = re.search(r'\b([A-Z]{1,5})\b', query)
        if match:
            candidate = match.group(1)
            skip = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'NEWS', 'STOCK'}
            if candidate not in skip:
                return candidate
        return None

    @staticmethod
    def _strip_legal_suffix(name: str) -> str:
        """Remove legal suffixes from company name."""
        cleaned = re.sub(
            r'\s+(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC|L\.L\.C\.|PLC|SA|AG|NV|LP|Co\.?|Company)$',
            '', name.strip(), flags=re.IGNORECASE
        )
        return cleaned.strip() or name.strip()

    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles."""
        unique = []
        seen_titles = set()
        seen_urls = set()

        for article in articles:
            title = (article.get("title") or "").lower().strip()
            url = (article.get("url") or "").lower().strip()

            if not title and not url:
                continue
            if title and title in seen_titles:
                continue
            if url and url in seen_urls:
                continue

            if title:
                seen_titles.add(title)
            if url:
                seen_urls.add(url)

            unique.append(article)

        return unique

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = datetime.now().timestamp()
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < RateLimit.WINDOW_SECONDS]

        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = RateLimit.WINDOW_SECONDS - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(now)

    async def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        self.company_name_cache.clear()
        logging.info("🧹 News API cache cleared")