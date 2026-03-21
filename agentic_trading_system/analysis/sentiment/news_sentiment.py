"""
News Sentiment - Analyzes sentiment from news articles
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
import numpy as np
from agentic_trading_system.utils.logger import logger as  logging

class NewsSentiment:
    """
    Fetches and analyzes sentiment from news articles
    Uses multiple news sources and NLP sentiment analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API keys (from config)
        self.news_api_key = config.get("news_api_key")
        self.fmp_api_key = config.get("fmp_api_key")
        
        # Sources to query
        self.sources = config.get("sources", [
            "reuters", "bloomberg", "cnbc", "wsj", "financial-times"
        ])
        
        # Lookback period
        self.lookback_days = config.get("lookback_days", 7)
        
        # Cache
        self.cache = {}
        self.cache_ttl = timedelta(minutes=30)
        
        # Rate limiting
        self.request_timestamps = []
        self.max_requests_per_minute = 30
        
        logging.info(f"📰 NewsSentiment initialized with {len(self.sources)} sources")
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment score from news articles for a symbol
        """
        logging.info(f"📰 Fetching news sentiment for {symbol}")
        
        # Check cache
        cache_key = f"news_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        try:
            # Fetch news articles
            articles = await self._fetch_news(symbol)
            
            if not articles:
                return {
                    "score": 0.5,
                    "confidence": 0.3,
                    "details": {"error": "No news found"},
                    "source": "news"
                }
            
            # Analyze sentiment for each article
            sentiments = []
            article_details = []
            
            for article in articles:
                sentiment = self._analyze_article_sentiment(article)
                sentiments.append(sentiment["score"])
                article_details.append({
                    "title": article.get("title", "")[:100],
                    "source": article.get("source", "unknown"),
                    "sentiment": sentiment["score"],
                    "relevance": sentiment["relevance"],
                    "url": article.get("url", "")
                })
            
            # Calculate aggregate sentiment
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.5
            
            # Calculate confidence based on:
            # - Number of articles
            # - Consistency of sentiment
            # - Recency of articles
            confidence = self._calculate_confidence(sentiments, articles)
            
            # Determine trend
            trend = self._detect_trend(article_details)
            
            result = {
                "score": float(avg_sentiment),
                "confidence": float(confidence),
                "source": "news",
                "details": {
                    "article_count": len(articles),
                    "sentiment_std": float(np.std(sentiments)) if len(sentiments) > 1 else 0,
                    "trend": trend,
                    "recent_articles": article_details[:5],  # Top 5 most recent
                    "sources_found": list(set(a["source"] for a in article_details))
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in news sentiment for {symbol}: {e}")
            return {
                "score": 0.5,
                "confidence": 0.3,
                "source": "news",
                "details": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    async def _fetch_news(self, symbol: str) -> List[Dict]:
        """Fetch news articles for symbol from various sources"""
        articles = []
        
        # Try multiple sources in parallel
        tasks = [
            self._fetch_from_newsapi(symbol),
            self._fetch_from_fmp(symbol),
            self._fetch_from_alphavantage(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
        
        # Remove duplicates by title similarity
        articles = self._deduplicate(articles)
        
        # Sort by recency
        articles.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        
        return articles[:20]  # Limit to 20 most recent
    
    async def _fetch_from_newsapi(self, symbol: str) -> List[Dict]:
        """Fetch from NewsAPI"""
        if not self.news_api_key:
            return []
        
        await self._rate_limit()
        
        url = "https://newsapi.org/v2/everything"
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        params = {
            "q": f"({symbol}) AND (stock OR share OR earnings OR market)",
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 50,
            "apiKey": self.news_api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        articles = data.get("articles", [])
                        
                        return [{
                            "title": a.get("title", ""),
                            "description": a.get("description", ""),
                            "content": a.get("content", ""),
                            "source": a.get("source", {}).get("name", "unknown"),
                            "url": a.get("url", ""),
                            "published_at": a.get("publishedAt", ""),
                            "api_source": "newsapi"
                        } for a in articles]
        except Exception as e:
            logging.debug(f"NewsAPI error: {e}")
        
        return []
    
    async def _fetch_from_fmp(self, symbol: str) -> List[Dict]:
        """Fetch from Financial Modeling Prep"""
        if not self.fmp_api_key:
            return []
        
        await self._rate_limit()
        
        url = f"https://financialmodelingprep.com/api/v3/stock_news"
        
        params = {
            "tickers": symbol,
            "limit": 50,
            "apikey": self.fmp_api_key
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        return [{
                            "title": item.get("title", ""),
                            "description": item.get("text", ""),
                            "content": item.get("text", ""),
                            "source": item.get("site", "unknown"),
                            "url": item.get("url", ""),
                            "published_at": item.get("publishedDate", ""),
                            "api_source": "fmp"
                        } for item in data]
        except Exception as e:
            logging.debug(f"FMP error: {e}")
        
        return []
    
    async def _fetch_from_alphavantage(self, symbol: str) -> List[Dict]:
        """Fetch from Alpha Vantage News API"""
        # Placeholder for Alpha Vantage integration
        return []
    
    def _analyze_article_sentiment(self, article: Dict) -> Dict[str, float]:
        """
        Analyze sentiment of a single article
        Uses simple keyword-based sentiment (can be upgraded to ML)
        """
        text = f"{article.get('title', '')} {article.get('description', '')} {article.get('content', '')}".lower()
        
        # Financial sentiment lexicons
        positive_words = {
            'beat', 'exceed', 'surge', 'soar', 'jump', 'gain', 'rise',
            'upgrade', 'bullish', 'outperform', 'growth', 'profit',
            'record', 'high', 'strong', 'positive', 'momentum',
            'breakthrough', 'approval', 'partnership', 'expansion',
            'dividend', 'buyback', 'acquisition', 'synergy', 'opportunity'
        }
        
        negative_words = {
            'miss', 'fall', 'drop', 'decline', 'downgrade', 'bearish',
            'underperform', 'loss', 'weak', 'negative', 'risk',
            'lawsuit', 'investigation', 'probe', 'scandal', 'fraud',
            'bankruptcy', 'default', 'layoff', 'cut', 'warning',
            'downgrade', 'sell', 'pressure', 'volatile', 'concern'
        }
        
        # Count sentiment words
        words = text.split()
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        
        # Calculate sentiment score
        total = positive_count + negative_count
        if total > 0:
            sentiment = (positive_count - negative_count) / total
            # Scale from -1..1 to 0..1
            sentiment = (sentiment + 1) / 2
        else:
            sentiment = 0.5
        
        # Calculate relevance (how many times symbol appears)
        symbol = article.get('title', '').split()[0] if article.get('title') else ''
        relevance = min(1.0, text.count(symbol.lower()) / 10) if symbol else 0.3
        
        return {
            "score": float(min(1.0, max(0.0, sentiment))),
            "relevance": float(relevance)
        }
    
    def _calculate_confidence(self, sentiments: List[float], articles: List[Dict]) -> float:
        """Calculate confidence in sentiment score"""
        if not sentiments:
            return 0.3
        
        # Number of articles factor
        article_factor = min(1.0, len(sentiments) / 10)
        
        # Consistency factor (lower std = higher confidence)
        if len(sentiments) > 1:
            std = np.std(sentiments)
            consistency = 1 - min(1.0, std * 2)
        else:
            consistency = 0.5
        
        # Recency factor (more recent = higher confidence)
        if articles:
            recent_count = sum(1 for a in articles[:5] 
                             if a.get("published_at", ""))
            recency = recent_count / 5
        else:
            recency = 0.5
        
        confidence = (
            article_factor * 0.4 +
            consistency * 0.4 +
            recency * 0.2
        )
        
        return float(min(1.0, max(0.3, confidence)))
    
    def _detect_trend(self, articles: List[Dict]) -> str:
        """Detect sentiment trend over recent articles"""
        if len(articles) < 3:
            return "neutral"
        
        # Sort by date
        sorted_articles = sorted(articles, 
                               key=lambda x: x.get("published_at", ""))
        
        # Split into recent and older
        recent = sorted_articles[-3:]  # Last 3
        older = sorted_articles[:3]    # First 3
        
        recent_sentiment = np.mean([a["sentiment"] for a in recent])
        older_sentiment = np.mean([a["sentiment"] for a in older])
        
        if recent_sentiment > older_sentiment + 0.1:
            return "improving"
        elif recent_sentiment < older_sentiment - 0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def _deduplicate(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles by title similarity"""
        unique = []
        seen_titles = set()
        
        for article in articles:
            title = article.get("title", "").lower()
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique.append(article)
        
        return unique
    
    async def _rate_limit(self):
        """Rate limiting to avoid API throttling"""
        now = datetime.now()
        
        # Remove old timestamps
        self.request_timestamps = [
            ts for ts in self.request_timestamps 
            if now - ts < timedelta(minutes=1)
        ]
        
        # Check limit
        if len(self.request_timestamps) >= self.max_requests_per_minute:
            wait_time = 60 - (now - self.request_timestamps[0]).seconds
            await asyncio.sleep(wait_time)
        
        self.request_timestamps.append(now)