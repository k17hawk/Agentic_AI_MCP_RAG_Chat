"""
News API Client - Financial news aggregation
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

class NewsAPIClient:
    """
    Aggregates news from multiple financial news sources
    Supports: NewsAPI, Alpha Vantage, Financial Modeling Prep
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API keys - handle both naming conventions
        self.news_api_key = config.get("news_api_key") or config.get("NEWS_API_KEY")
        self.alpha_vantage_key = config.get("alpha_vantage_key") or config.get("ALPHA_VANTAGE_KEY")
        self.fmp_key = config.get("fmp_key") or config.get("FMP_KEY")
        
        # Debug API key status
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
        
        # News sources
        self.sources = config.get("sources", [
            "reuters", "bloomberg", "cnbc", "wsj", "financial-times",
            "seeking-alpha", "yahoo-finance", "marketwatch"
        ])
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 30)
        self.request_timestamps = []
        
        # Cache
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 15) * 60
        
        logging.info(f"✅ NewsAPIClient initialized with {len(self.sources)} sources")
    
    @retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search for news articles
        """
        options = options or {}
        logging.info(f"📰 News search for: '{query}'")
        
        # Check cache
        cache_key = f"news_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached news results for '{query}'")
                return cached_result
        
        await self._rate_limit()
        
        all_articles = []
        errors = []
        
        # Try multiple sources in parallel
        tasks = []
        if self.news_api_key:
            tasks.append(self._fetch_from_newsapi(query, options))
        if self.alpha_vantage_key:
            tasks.append(self._fetch_from_alphavantage(query, options))
        if self.fmp_key:
            tasks.append(self._fetch_from_fmp(query, options))
        
        # If no API keys configured, log warning
        if not tasks:
            logging.warning("⚠️ No news API keys configured. Please add NEWS_API_KEY, ALPHA_VANTAGE_KEY, or FMP_KEY to your .env file")
            return {"items": [], "metadata": {"error": "No API keys configured", "total_found": 0, "unique_count": 0, "sources_used": 0}}
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                error_msg = f"Error from news source {i}: {result}"
                logging.warning(f"⚠️ {error_msg}")
                errors.append(error_msg)
            elif isinstance(result, list):
                all_articles.extend(result)
                logging.info(f"✅ Got {len(result)} articles from source {i}")
            elif isinstance(result, dict) and "articles" in result:
                all_articles.extend(result["articles"])
                logging.info(f"✅ Got {len(result['articles'])} articles from source {i}")
        
        # Deduplicate
        unique_articles = self._deduplicate(all_articles)
        
        # Sort by recency
        unique_articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        
        result = {
            "items": unique_articles[:options.get("max_results", 50)],
            "metadata": {
                "total_found": len(all_articles),
                "unique_count": len(unique_articles),
                "sources_used": len([t for t in tasks if t is not None]),
                "errors": errors if errors else None
            }
        }
        
        # Cache result
        self.cache[cache_key] = (datetime.now().timestamp(), result)
        
        logging.info(f"✅ News found {len(unique_articles)} unique articles")
        return result
    
    async def _fetch_from_newsapi(self, query: str, options: Dict) -> List[Dict]:
        """Fetch from NewsAPI"""
        if not self.news_api_key:
            logging.warning("⚠️ NewsAPI key missing")
            return []
        
        url = "https://newsapi.org/v2/everything"
        
        # Calculate date range
        days_back = options.get("days_back", 7)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Clean query - remove "stock" if present as it might limit results
        clean_query = query.replace(" stock", "").replace(" stocks", "").strip()
        
        params = {
            "q": clean_query,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": options.get("language", "en"),
            "sortBy": options.get("sort_by", "relevancy"),
            "pageSize": min(options.get("page_size", 50), 100),
            "apiKey": self.news_api_key
        }
        
        # Add specific sources if requested
        if "sources" in options:
            params["sources"] = ",".join(options["sources"])
        
        try:
            logging.info(f"📡 Calling NewsAPI with query: '{clean_query}'")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("status") != "ok":
                            logging.warning(f"⚠️ NewsAPI error: {data.get('message', 'Unknown error')}")
                            return []
                        
                        articles = []
                        for article in data.get("articles", []):
                            articles.append({
                                "title": article.get("title"),
                                "description": article.get("description"),
                                "content": article.get("content"),
                                "url": article.get("url"),
                                "source": article.get("source", {}).get("name", "unknown"),
                                "published_at": article.get("publishedAt"),
                                "author": article.get("author"),
                                "image_url": article.get("urlToImage"),
                                "api_source": "newsapi"
                            })
                        
                        logging.info(f"✅ NewsAPI returned {len(articles)} articles")
                        return articles
                    else:
                        error_text = await response.text()
                        logging.warning(f"⚠️ NewsAPI returned status {response.status}: {error_text[:200]}")
                        return []
        except Exception as e:
            logging.warning(f"⚠️ NewsAPI exception: {e}")
            return []
    
    async def _fetch_from_alphavantage(self, query: str, options: Dict) -> List[Dict]:
        """Fetch from Alpha Vantage News API"""
        if not self.alpha_vantage_key:
            return []
        
        # Extract ticker from query (assumes query is a ticker symbol)
        ticker = query.split()[0].upper() if query else ""
        
        url = "https://www.alphavantage.co/query"
        
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": self.alpha_vantage_key,
            "limit": options.get("limit", 50)
        }
        
        try:
            logging.info(f"📡 Calling Alpha Vantage for ticker: {ticker}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if "Information" in data and "demo" in data["Information"].lower():
                            logging.warning("⚠️ Alpha Vantage: API key requires premium tier for news")
                            return []
                        
                        articles = []
                        for item in data.get("feed", []):
                            articles.append({
                                "title": item.get("title"),
                                "summary": item.get("summary"),
                                "content": item.get("summary"),
                                "url": item.get("url"),
                                "source": item.get("source", "alphavantage"),
                                "published_at": item.get("time_published"),
                                "sentiment": item.get("overall_sentiment_score"),
                                "sentiment_label": item.get("overall_sentiment_label"),
                                "tickers": [t["ticker"] for t in item.get("ticker_sentiment", [])],
                                "api_source": "alphavantage"
                            })
                        
                        logging.info(f"✅ Alpha Vantage returned {len(articles)} articles")
                        return articles
                    else:
                        logging.warning(f"⚠️ Alpha Vantage returned status {response.status}")
                        return []
        except Exception as e:
            logging.warning(f"⚠️ Alpha Vantage exception: {e}")
            return []
    
    async def _fetch_from_fmp(self, query: str, options: Dict) -> List[Dict]:
        """Fetch from Financial Modeling Prep"""
        if not self.fmp_key:
            return []
        
        # Extract ticker from query
        ticker = query.split()[0].upper() if query else ""
        
        url = f"https://financialmodelingprep.com/api/v3/stock_news"
        
        params = {
            "tickers": ticker,
            "limit": options.get("limit", 50),
            "apikey": self.fmp_key
        }
        
        try:
            logging.info(f"📡 Calling FMP for ticker: {ticker}")
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if not isinstance(data, list):
                            logging.warning(f"⚠️ FMP returned unexpected data format")
                            return []
                        
                        articles = []
                        for item in data:
                            articles.append({
                                "title": item.get("title"),
                                "content": item.get("text"),
                                "url": item.get("url"),
                                "source": item.get("site", "fmp"),
                                "published_at": item.get("publishedDate"),
                                "image": item.get("image"),
                                "api_source": "fmp"
                            })
                        
                        logging.info(f"✅ FMP returned {len(articles)} articles")
                        return articles
                    else:
                        logging.warning(f"⚠️ FMP returned status {response.status}")
                        return []
        except Exception as e:
            logging.warning(f"⚠️ FMP exception: {e}")
            return []
    
    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles by title"""
        unique = []
        seen_titles = set()
        
        for article in articles:
            title = article.get("title", "").lower().strip()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique.append(article)
        
        return unique
    
    async def _rate_limit(self):
        """Rate limiting to avoid API throttling"""
        now = datetime.now().timestamp()
        
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if now - ts < 60]
        
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(now)