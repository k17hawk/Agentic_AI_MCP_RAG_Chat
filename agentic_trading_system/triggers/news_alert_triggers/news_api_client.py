"""
News API Client - Fetches news from multiple sources
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import aiohttp
import asyncio

from loguru import logger
import hashlib

class NewsAPIClient:
    """
    Fetches news from multiple APIs:
    - NewsAPI
    - Alpha Vantage (news sentiment)
    - Financial Modeling Prep
    - Custom RSS feeds
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # API keys
        self.news_api_key = config.get("news_api_key")
        self.alpha_vantage_key = config.get("alpha_vantage_key")
        self.fmp_key = config.get("fmp_key")
        
        # Sources to query
        self.sources = config.get("news_sources", [
            "bloomberg",
            "reuters",
            "cnbc",
            "financial-times",
            "wsj",
            "seeking-alpha"
        ])
        
        # Rate limiting
        self.request_timestamps = []
        self.max_requests_per_minute = 30
        
        # Cache
        self.cache = {}
        
        logger.info("NewsAPIClient initialized")
    
    async def get_recent_news(self, 
                              lookback_minutes: int = 15, 
                              limit: int = 50) -> List[Dict]:
        """
        Get recent news articles from all sources
        """
        all_articles = []
        
        # Calculate time range
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(minutes=lookback_minutes)
        
        # Try NewsAPI first
        if self.news_api_key:
            try:
                articles = await self._fetch_newsapi(start_time, end_time, limit)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"NewsAPI error: {e}")
        
        # Try Alpha Vantage (if configured)
        if self.alpha_vantage_key:
            try:
                articles = await self._fetch_alpha_vantage(start_time, end_time)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"Alpha Vantage error: {e}")
        
        # Try FMP (if configured)
        if self.fmp_key:
            try:
                articles = await self._fetch_fmp(start_time, end_time)
                all_articles.extend(articles)
            except Exception as e:
                logger.error(f"FMP error: {e}")
        
        # Deduplicate by title
        unique_articles = self._deduplicate(all_articles)
        
        # Sort by recency
        unique_articles.sort(
            key=lambda x: x.get("publishedAt", ""), 
            reverse=True
        )
        
        return unique_articles[:limit]
    
    async def _fetch_newsapi(self, 
                            start_time: datetime, 
                            end_time: datetime,
                            limit: int) -> List[Dict]:
        """
        Fetch from NewsAPI
        """
        await self._rate_limit()
        
        url = "https://newsapi.org/v2/everything"
        
        params = {
            "q": "stock OR market OR earnings OR IPO",
            "sources": ",".join(self.sources),
            "from": start_time.isoformat(),
            "to": end_time.isoformat(),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100),
            "apiKey": self.news_api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    articles = data.get("articles", [])
                    
                    # Format articles
                    formatted = []
                    for article in articles:
                        formatted.append({
                            "id": self._generate_id(article),
                            "title": article.get("title", ""),
                            "description": article.get("description", ""),
                            "content": article.get("content", ""),
                            "source": article.get("source", {}).get("name", "unknown"),
                            "url": article.get("url", ""),
                            "publishedAt": article.get("publishedAt", ""),
                            "api_source": "newsapi"
                        })
                    
                    return formatted
                else:
                    logger.error(f"NewsAPI error: {response.status}")
                    return []
    
    async def _fetch_alpha_vantage(self, 
                                  start_time: datetime, 
                                  end_time: datetime) -> List[Dict]:
        """
        Fetch from Alpha Vantage News Sentiment API
        """
        await self._rate_limit()
        
        url = "https://www.alphavantage.co/query"
        
        params = {
            "function": "NEWS_SENTIMENT",
            "apikey": self.alpha_vantage_key,
            "topics": "earnings,ipo,merger",
            "limit": 50,
            "sort": "LATEST"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    feed = data.get("feed", [])
                    
                    formatted = []
                    for item in feed:
                        # Parse time
                        time_published = item.get("time_published", "")
                        if time_published:
                            # Alpha Vantage format: YYYYMMDDTHHMMSS
                            try:
                                pub_time = datetime.strptime(
                                    time_published, 
                                    "%Y%m%dT%H%M%S"
                                )
                            except:
                                pub_time = datetime.utcnow()
                        else:
                            pub_time = datetime.utcnow()
                        
                        # Check if within time range
                        if start_time <= pub_time <= end_time:
                            formatted.append({
                                "id": self._generate_id(item),
                                "title": item.get("title", ""),
                                "description": item.get("summary", ""),
                                "content": item.get("summary", ""),
                                "source": item.get("source", "unknown"),
                                "url": item.get("url", ""),
                                "publishedAt": pub_time.isoformat(),
                                "sentiment": item.get("overall_sentiment_score", 0),
                                "tickers": item.get("ticker_sentiment", []),
                                "api_source": "alphavantage"
                            })
                    
                    return formatted
                else:
                    return []
    
    async def _fetch_fmp(self, 
                        start_time: datetime, 
                        end_time: datetime) -> List[Dict]:
        """
        Fetch from Financial Modeling Prep
        """
        await self._rate_limit()
        
        url = f"https://financialmodelingprep.com/api/v4/general_news"
        
        params = {
            "page": 0,
            "apikey": self.fmp_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    formatted = []
                    for item in data[:50]:  # Limit to 50
                        # Parse time
                        pub_time = datetime.fromisoformat(
                            item.get("publishedDate", "").replace("Z", "+00:00")
                        ) if item.get("publishedDate") else datetime.utcnow()
                        
                        # Check if within time range
                        if start_time <= pub_time <= end_time:
                            formatted.append({
                                "id": self._generate_id(item),
                                "title": item.get("title", ""),
                                "description": item.get("text", ""),
                                "content": item.get("text", ""),
                                "source": item.get("site", "unknown"),
                                "url": item.get("url", ""),
                                "publishedAt": pub_time.isoformat(),
                                "api_source": "fmp"
                            })
                    
                    return formatted
                else:
                    return []
    
    def _generate_id(self, article: Dict) -> str:
        """Generate unique ID for article"""
        title = article.get("title", "")
        source = article.get("source", "")
        published = article.get("publishedAt", "")
        
        id_string = f"{title}{source}{published}"
        return hashlib.md5(id_string.encode()).hexdigest()[:16]
    
    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles by title similarity"""
        unique = []
        seen_titles = set()
        
        for article in articles:
            title = article.get("title", "").lower()
            
            # Simple dedupe - check if similar title exists
            is_duplicate = False
            for seen in seen_titles:
                if self._similarity(title, seen) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                seen_titles.add(title)
                unique.append(article)
        
        return unique
    
    def _similarity(self, s1: str, s2: str) -> float:
        """Simple string similarity (Jaccard)"""
        set1 = set(s1.split())
        set2 = set(s2.split())
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0
    
    async def _rate_limit(self):
        """Rate limiting to avoid API throttling"""
        now = datetime.utcnow()
        
        # Remove old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < timedelta(minutes=1)
        ]
        
        # Check limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_timestamps[0]).seconds
            logger.warning(f"Rate limit reached, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
        
        self.request_timestamps.append(now)