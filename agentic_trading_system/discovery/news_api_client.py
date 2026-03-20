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


# ---------------------------------------------------------------------------
# Private sentinel exceptions — used only inside NewsAPIClient to signal
# specific API failure modes up through _newsapi_get() to _fetch_from_newsapi()
# without relying on string matching on error messages.
# ---------------------------------------------------------------------------
class _NewsAPIPlanError(Exception):
    """Raised when NewsAPI returns a plan-restriction error code."""

class _NewsAPIRateLimit(Exception):
    """Raised when NewsAPI returns HTTP 429."""


class NewsAPIClient:
    """
    Aggregates news from multiple financial news sources.
    Supports: NewsAPI, Alpha Vantage, Financial Modeling Prep
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # API keys — handle both naming conventions
        self.news_api_key = config.get("news_api_key") or config.get("NEWS_API_KEY")
        self.alpha_vantage_key = config.get("alpha_vantage_key") or config.get("ALPHA_VANTAGE_KEY")
        self.fmp_key = config.get("fmp_key") or config.get("FMP_KEY")

        if self.news_api_key:
            logging.info(f"✅ NewsAPI key configured (starts with: {self.news_api_key[:5]}...)")
        else:
            logging.warning("⚠️ NewsAPI key not configured")
        if self.alpha_vantage_key:
            logging.info(f"✅ Alpha Vantage key configured (starts with: {self.alpha_vantage_key[:5]}...)")
        else:
            logging.warning("⚠️ Alpha Vantage key not configured")
        if self.fmp_key:
            logging.info(f"✅ FMP key configured (starts with: {self.fmp_key[:5]}...)")
        else:
            logging.warning("⚠️ FMP key not configured")

        # Common ticker → company name (instant, no I/O)
        self.common_tickers = {
            'AAPL': 'Apple', 'MSFT': 'Microsoft', 'GOOGL': 'Google',
            'GOOG': 'Google', 'AMZN': 'Amazon', 'META': 'Meta',
            'TSLA': 'Tesla', 'NVDA': 'NVIDIA', 'NFLX': 'Netflix',
            'AMD': 'AMD', 'INTC': 'Intel', 'IBM': 'IBM',
            'ORCL': 'Oracle', 'CRM': 'Salesforce', 'ADBE': 'Adobe',
            'DIS': 'Disney', 'V': 'Visa', 'MA': 'Mastercard',
            'JPM': 'JPMorgan Chase', 'BAC': 'Bank of America',
            'WMT': 'Walmart', 'TGT': 'Target', 'KO': 'Coca-Cola',
            'PEP': 'PepsiCo', 'MCD': "McDonald's", 'SBUX': 'Starbucks',
            'NKE': 'Nike', 'BRK.A': 'Berkshire Hathaway',
            'BRK.B': 'Berkshire Hathaway', 'LIN': 'Linde',
            'UNH': 'UnitedHealth', 'HD': 'Home Depot',
            'PG': 'Procter & Gamble', 'CVX': 'Chevron',
            'MRK': 'Merck', 'ABBV': 'AbbVie', 'PFE': 'Pfizer',
            'CSCO': 'Cisco', 'CMCSA': 'Comcast', 'COST': 'Costco',
            'TMUS': 'T-Mobile', 'AVGO': 'Broadcom',
            'TMO': 'Thermo Fisher', 'NEE': 'NextEra Energy',
            'ACN': 'Accenture', 'DHR': 'Danaher', 'ABT': 'Abbott',
            'WFC': 'Wells Fargo', 'QCOM': 'Qualcomm',
            'TXN': 'Texas Instruments', 'INTU': 'Intuit',
            'AMGN': 'Amgen', 'HON': 'Honeywell', 'UPS': 'UPS',
            'CAT': 'Caterpillar', 'UNP': 'Union Pacific',
            'LOW': "Lowe's", 'BA': 'Boeing', 'GE': 'General Electric',
            'MMM': '3M', 'AXP': 'American Express',
            'GS': 'Goldman Sachs', 'MS': 'Morgan Stanley',
            'C': 'Citigroup', 'SPY': 'SPDR S&P 500 ETF',
            'QQQ': 'Invesco QQQ Trust', 'IVV': 'iShares Core S&P 500 ETF',
            'VOO': 'Vanguard S&P 500 ETF', 'VTI': 'Vanguard Total Stock Market ETF',
            'BND': 'Vanguard Total Bond Market', 'VT': 'Vanguard Total World Stock',
        }

        # Rate limiting
        self.rate_limit = config.get("rate_limit", 30)
        self.request_timestamps: List[float] = []

        # Caches
        self.cache: Dict = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 15) * 60
        self.company_name_cache: Dict[str, Optional[str]] = {}

        logging.info("✅ NewsAPIClient initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # Build task pairs so source names stay in sync with their coroutines
        # regardless of which API keys are configured.
        task_pairs = []
        if self.news_api_key:
            task_pairs.append(("newsapi", self._fetch_from_newsapi(query, options)))
        if self.alpha_vantage_key:
            task_pairs.append(("alphavantage", self._fetch_from_alphavantage(query, options)))
        if self.fmp_key:
            task_pairs.append(("fmp", self._fetch_from_fmp(query, options)))

        if not task_pairs:
            logging.warning("⚠️ No news API keys configured")
            return {"items": [], "metadata": {"error": "No API keys configured",
                                               "total_found": 0, "unique_count": 0,
                                               "sources_used": []}}

        source_names = [name for name, _ in task_pairs]
        tasks        = [coro for _, coro in task_pairs]
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

    # ------------------------------------------------------------------
    # Query generation
    # ------------------------------------------------------------------

    async def _generate_query_variations(self, query: str, options: Dict) -> List[str]:
        """
        Generate query variations to maximise NewsAPI recall.

        For tickers the company name is looked up FIRST (from the in-memory
        dict or a fast async lookup) so the best-performing 'company name'
        variations are always included.  The ticker-only variation is kept as
        a fallback but deprioritised — direct ticker searches return fewer and
        less relevant results than company-name searches.
        """
        original = query.strip()
        variations: List[str] = []

        is_ticker = original.isupper() and 1 <= len(original) <= 5 and original.isalpha()

        if is_ticker:
            ticker = original

            # Resolve company name first so we can build the best variations
            company_name = await self._get_company_name(ticker)

            if company_name:
                # Company-name variations first — much higher recall on NewsAPI
                variations.append(f'"{company_name}"')
                variations.append(f"{company_name} stock")
                variations.append(f"{company_name} news")
                variations.append(f"{company_name} earnings")

            # Ticker variations as supplementary / fallback
            variations.append(f"{ticker} stock")
            variations.append(f"${ticker}")
            variations.append(ticker)          # bare ticker last — lowest recall

        elif " " in original:
            # Multi-word: exact phrase first, then individual words
            variations.append(f'"{original}"')
            variations.append(original)
            words = original.split()
            if len(words) > 2:
                variations.append(" ".join(words[:2]))
        else:
            # Single non-ticker word
            variations.append(f'"{original}"')
            variations.append(original)

        # Deduplicate while preserving order
        seen: set = set()
        unique: List[str] = []
        for v in variations:
            if v not in seen:
                seen.add(v)
                unique.append(v)

        return unique[:8]   # cap at 8 to avoid excessive API calls

    # ------------------------------------------------------------------
    # Source fetchers
    # ------------------------------------------------------------------

    # Error codes returned by NewsAPI when the plan doesn't allow /v2/everything
    # with date filtering.  When we see these we fall back to /v2/top-headlines.
    _NEWSAPI_PLAN_ERROR_CODES = frozenset({
        "parameterInvalid",
        "planUpgradeRequired",
        "rateLimited",
        "sourcesTooMany",
    })

    @staticmethod
    def _parse_newsapi_articles(raw_articles: List[Dict], variation: str) -> List[Dict]:
        """Shared article normaliser for both /everything and /top-headlines."""
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
                "api_source": "newsapi",
                "matched_query": variation,
            })
        return articles

    async def _newsapi_get(
        self,
        url: str,
        params: Dict,
        variation: str,
    ) -> List[Dict]:
        """
        Single GET against a NewsAPI endpoint.
        Returns a list of normalised articles, or [] on any error.
        Errors are always logged at WARNING level so they're never silently dropped.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        if data.get("status") != "ok":
                            code = data.get("code", "unknown")
                            msg  = data.get("message", "")
                            logging.warning(
                                f"⚠️ NewsAPI non-ok for '{variation}' "
                                f"[{url.split('/')[-1]}]: {code} — {msg}"
                            )
                            # Signal plan errors back to the caller via a special sentinel
                            if code in self._NEWSAPI_PLAN_ERROR_CODES:
                                raise _NewsAPIPlanError(code)
                            return []
                        articles = self._parse_newsapi_articles(
                            data.get("articles", []), variation
                        )
                        logging.info(
                            f"✅ NewsAPI [{url.split('/')[-1]}] '{variation}': "
                            f"{len(articles)} articles "
                            f"(totalResults={data.get('totalResults', '?')})"
                        )
                        return articles

                    elif response.status == 429:
                        logging.warning(f"⚠️ NewsAPI rate limit hit for '{variation}'")
                        raise _NewsAPIRateLimit()
                    else:
                        text = await response.text()
                        logging.warning(
                            f"⚠️ NewsAPI HTTP {response.status} for '{variation}': "
                            f"{text[:200]}"
                        )
                        return []

        except (_NewsAPIPlanError, _NewsAPIRateLimit):
            raise   # let caller handle these specifically
        except asyncio.TimeoutError:
            logging.warning(f"⚠️ NewsAPI timeout for '{variation}'")
            return []
        except Exception as e:
            logging.warning(
                f"⚠️ NewsAPI exception for '{variation}': {type(e).__name__}: {e}"
            )
            return []

    async def _fetch_from_newsapi(self, query: str, options: Dict) -> List[Dict]:
        """
        Fetch from NewsAPI with query-variation fallback.

        Strategy:
          1. Try /v2/everything with date filtering (paid plans).
          2. On a plan-restriction error, fall back to /v2/top-headlines (free plans).
          3. Rate-limit hit aborts further requests for this call.
        """
        if not self.news_api_key:
            return []

        everything_url  = "https://newsapi.org/v2/everything"
        headlines_url   = "https://newsapi.org/v2/top-headlines"
        days_back       = options.get("days_back", 30)
        end_date        = datetime.now()
        start_date      = end_date - timedelta(days=days_back)
        target          = options.get("min_articles", 10)

        # True once we've confirmed /v2/everything is blocked for this plan
        use_headlines_only = False

        query_variations = await self._generate_query_variations(query, options)
        logging.info(f"📰 NewsAPI query variations: {query_variations}")

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

            # --- /v2/everything (paid / higher-tier plans) ---
            if not use_headlines_only:
                params: Dict = {
                    "q":        variation,
                    "from":     start_date.strftime("%Y-%m-%d"),
                    "to":       end_date.strftime("%Y-%m-%d"),
                    "language": options.get("language", "en"),
                    "sortBy":   options.get("sort_by", "publishedAt"),
                    "pageSize": min(options.get("page_size", 50), 100),
                    "apiKey":   self.news_api_key,
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
                except _NewsAPIPlanError:
                    logging.warning(
                        "⚠️ NewsAPI plan restricts /v2/everything — "
                        "switching to /v2/top-headlines for all remaining variations"
                    )
                    use_headlines_only = True
                except _NewsAPIRateLimit:
                    break   # stop all further requests this call

            # --- /v2/top-headlines fallback (works on free / developer plans) ---
            # top-headlines does not support from/to or sortBy — keep params minimal
            hl_params: Dict = {
                "q":        variation,
                "language": options.get("language", "en"),
                "pageSize": min(options.get("page_size", 50), 100),
                "apiKey":   self.news_api_key,
            }
            # top-headlines accepts "sources" OR "country"/"category", not both
            if "news_sources" in options and options["news_sources"]:
                mapped = [source_mapping[s] for s in options["news_sources"]
                          if s in source_mapping]
                if mapped:
                    hl_params["sources"] = ",".join(mapped)
            else:
                hl_params["country"] = "us"   # default to US headlines

            try:
                articles = await self._newsapi_get(headlines_url, hl_params, variation)
                all_articles.extend(articles)
            except _NewsAPIRateLimit:
                break
            except _NewsAPIPlanError:
                # top-headlines shouldn't raise this, but guard anyway
                logging.warning("⚠️ NewsAPI plan error even on /v2/top-headlines")
                break

            await asyncio.sleep(0.15)

        return all_articles

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
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        # Rate-limit / plan message comes back as {"Information": "..."}
                        if "Information" in data:
                            logging.warning(f"⚠️ Alpha Vantage: {data['Information'][:120]}")
                            return []

                        # Unexpected error key
                        if "Error Message" in data:
                            logging.warning(f"⚠️ Alpha Vantage error: {data['Error Message'][:120]}")
                            return []

                        feed = data.get("feed")
                        if not isinstance(feed, list):
                            logging.warning(
                                f"⚠️ Alpha Vantage unexpected response shape for {ticker}: "
                                f"{list(data.keys())}"
                            )
                            return []

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
                                "source": item.get("source", "alphavantage"),
                                "published_at": published_at,
                                "sentiment": item.get("overall_sentiment_score", 0),
                                "sentiment_label": item.get("overall_sentiment_label", "NEUTRAL"),
                                "tickers": [t["ticker"] for t in item.get("ticker_sentiment", [])],
                                "api_source": "alphavantage",
                            })

                        logging.info(f"✅ Alpha Vantage: {len(articles)} articles for {ticker}")
                        return articles

                    else:
                        text = await response.text()
                        logging.warning(
                            f"⚠️ Alpha Vantage HTTP {response.status} for {ticker}: {text[:200]}"
                        )

        except asyncio.TimeoutError:
            logging.warning(f"⚠️ Alpha Vantage timeout for {ticker}")
        except Exception as e:
            logging.warning(f"⚠️ Alpha Vantage exception: {type(e).__name__}: {e}")

        return []

    async def _fetch_from_fmp(self, query: str, options: Dict) -> List[Dict]:
        """
        Fetch from Financial Modeling Prep.

        Endpoint priority (most current first):
          1. /stable/news/stock          — current FMP versioning scheme (post-Aug 2025)
          2. /api/v4/stock_news-sentiments-rss-feed — v4 fallback
          (v3/stock_news is omitted: deprecated Aug 31 2025, returns 403 for new accounts)

        Return sentinel convention inside _fmp_get:
          list  — success (may be empty)
          None  — endpoint blocked/deprecated, try next in list
        """
        if not self.fmp_key:
            return []

        ticker = self._extract_ticker(query)
        if not ticker:
            return []

        limit = options.get("limit", 50)

        endpoints = [
            (
                "https://financialmodelingprep.com/stable/news/stock",
                {"symbols": ticker, "limit": limit, "apikey": self.fmp_key},
            ),
            (
                "https://financialmodelingprep.com/api/v4/stock_news-sentiments-rss-feed",
                {"tickers": ticker, "limit": limit, "apikey": self.fmp_key},
            ),
        ]

        async def _fmp_get(url: str, params: Dict):
            """
            One FMP GET.  Returns list (success), None (blocked → try next), or [] (hard stop).
            """
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    short_path = url.split("financialmodelingprep.com")[-1].split("?")[0]

                    # 403/404 = endpoint deprecated or not available on this plan → try next
                    if response.status in (403, 404):
                        text = await response.text()
                        logging.warning(
                            f"⚠️ FMP HTTP {response.status} on {short_path} "
                            f"for {ticker} — trying next endpoint. "
                            f"Detail: {text[:150]}"
                        )
                        return None

                    if response.status != 200:
                        text = await response.text()
                        logging.warning(
                            f"⚠️ FMP HTTP {response.status} for {ticker}: {text[:200]}"
                        )
                        return []   # hard failure, stop

                    data = await response.json()

                    # HTTP 200 with an error dict (FMP does this sometimes)
                    if isinstance(data, dict):
                        err = (data.get("Error Message") or data.get("message")
                               or data.get("error") or str(data))
                        logging.warning(f"⚠️ FMP error body on {short_path}: {err[:200]}")
                        return None   # treat as blocked → try next endpoint

                    if not isinstance(data, list):
                        logging.warning(
                            f"⚠️ FMP unexpected response type "
                            f"{type(data).__name__} on {short_path}"
                        )
                        return None

                    return data

        raw: Optional[List] = None
        try:
            for idx, (url, params) in enumerate(endpoints):
                raw = await _fmp_get(url, params)
                if raw is not None:
                    break
                if idx < len(endpoints) - 1:
                    logging.info(
                        f"ℹ️ FMP endpoint {idx + 1}/{len(endpoints)} unavailable, "
                        f"trying next fallback"
                    )
        except asyncio.TimeoutError:
            logging.warning(f"⚠️ FMP timeout for {ticker}")
            return []
        except Exception as e:
            logging.warning(f"⚠️ FMP exception: {type(e).__name__}: {e}")
            return []

        if not raw:
            logging.info(f"ℹ️ FMP: 0 articles for {ticker}")
            return []

        articles = []
        for item in raw[:options.get("max_results", 20)]:
            articles.append({
                "title": item.get("title", ""),
                "content": item.get("text", "") or item.get("content", "") or item.get("summary", ""),
                "url": item.get("url", ""),
                "source": item.get("site", "") or item.get("source", "") or "fmp",
                "published_at": (item.get("publishedDate") or item.get("publishedAt")
                                 or item.get("date", "")),
                "image": item.get("image", ""),
                "api_source": "fmp",
            })

        logging.info(f"✅ FMP: {len(articles)} articles for {ticker}")
        return articles

    # ------------------------------------------------------------------
    # Company name resolution
    # ------------------------------------------------------------------

    async def _get_company_name(self, ticker: str) -> Optional[str]:
        """
        Resolve a ticker to a company name.

        Lookup order:
          1. In-memory common_tickers dict  (instant)
          2. In-process cache               (instant, avoids repeat API calls)
          3. Alpha Vantage OVERVIEW         (async, already authenticated)
          4. yfinance                       (run in executor — blocking I/O)
          5. SEC EDGAR company_tickers.json (async, free, no key required)
        """
        ticker = ticker.upper()

        # 1. Common dict
        if ticker in self.common_tickers:
            return self.common_tickers[ticker]

        # 2. Cache (including previously-failed lookups stored as None)
        if ticker in self.company_name_cache:
            return self.company_name_cache[ticker]

        company_name: Optional[str] = None

        # 3. Alpha Vantage OVERVIEW
        if not company_name and self.alpha_vantage_key:
            company_name = await self._get_company_from_alphavantage(ticker)

        # 4. yfinance — run in executor so it doesn't block the event loop
        if not company_name:
            company_name = await self._get_company_from_yfinance(ticker)

        # 5. SEC EDGAR — free, no key, async
        if not company_name:
            company_name = await self._get_company_from_sec(ticker)

        # Cache (store None too, to avoid repeating failed lookups)
        self.company_name_cache[ticker] = company_name
        return company_name

    async def _get_company_from_alphavantage(self, ticker: str) -> Optional[str]:
        """Resolve company name via Alpha Vantage OVERVIEW endpoint."""
        if not self.alpha_vantage_key:
            return None

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
        except Exception as e:
            logging.debug(f"Alpha Vantage OVERVIEW error for {ticker}: {e}")

        return None

    async def _get_company_from_yfinance(self, ticker: str) -> Optional[str]:
        """
        Resolve company name via yfinance.

        yfinance performs blocking HTTP I/O, so it is run in a thread-pool
        executor to avoid stalling the asyncio event loop.
        """
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

        except asyncio.TimeoutError:
            logging.debug(f"yfinance timeout for {ticker}")
        except Exception as e:
            logging.debug(f"yfinance error for {ticker}: {e}")

        return None

    async def _get_company_from_sec(self, ticker: str) -> Optional[str]:
        """
        Resolve company name via SEC EDGAR company_tickers.json.
        Free, no API key required.
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://www.sec.gov/files/company_tickers.json"
                headers = {"User-Agent": "TradingBot/1.0 (news-lookup; contact@example.com)"}
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json(content_type=None)
                        for entry in data.values():
                            if entry.get("ticker", "").upper() == ticker:
                                return self._strip_legal_suffix(entry.get("title", ""))
        except asyncio.TimeoutError:
            logging.debug(f"SEC EDGAR timeout for {ticker}")
        except Exception as e:
            logging.debug(f"SEC EDGAR error for {ticker}: {e}")

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_ticker(self, query: str) -> Optional[str]:
        """
        Extract a ticker symbol from a query string.
        Returns None if the query doesn't look like a ticker search.
        """
        # Direct ticker (all-caps 1-5 letters)
        q = query.strip().upper()
        if q.isalpha() and 1 <= len(q) <= 5:
            return q

        # First all-caps word in the query
        m = re.search(r'\b([A-Z]{1,5})\b', query)
        if m:
            candidate = m.group(1)
            # Basic sanity: skip obvious non-tickers
            skip = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'NEWS', 'STOCK'}
            if candidate not in skip:
                return candidate

        return None

    @staticmethod
    def _strip_legal_suffix(name: str) -> str:
        """Remove Inc., Corp., Ltd. etc. for cleaner search queries."""
        cleaned = re.sub(
            r'\s+(Inc\.?|Corp\.?|Corporation|Ltd\.?|Limited|LLC|L\.L\.C\.'
            r'|PLC|SA|AG|NV|LP|Co\.?|Company)$',
            '', name.strip(), flags=re.IGNORECASE
        )
        return cleaned.strip() or name.strip()

    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles by title and URL."""
        unique: List[Dict] = []
        seen_titles: set = set()
        seen_urls: set = set()

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

    async def _rate_limit(self) -> None:
        """Enforce per-minute request cap."""
        now = datetime.now().timestamp()
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]

        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(now)

    async def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        self.company_name_cache.clear()
        logging.info("🧹 News API cache cleared")