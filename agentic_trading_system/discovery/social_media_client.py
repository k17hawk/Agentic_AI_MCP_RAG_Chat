"""
Social Media Client - Aggregates social media data
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio

from utils.logger import logger as logging
from utils.decorators import retry

class SocialMediaClient:
    """
    Aggregates data from social media platforms
    Supports: Twitter, Reddit, StockTwits
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API keys
        self.twitter_bearer_token = config.get("twitter_bearer_token")
        self.reddit_client_id = config.get("reddit_client_id")
        self.reddit_client_secret = config.get("reddit_client_secret")
        self.stocktwits_token = config.get("stocktwits_token")
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 20)
        self.request_timestamps = []
        
        # Cache
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 10) * 60
        
        logging.info(f"✅ SocialMediaClient initialized")
    
    @retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search social media for mentions
        """
        options = options or {}
        logging.info(f"📱 Social media search for: '{query}'")
        
        # Check cache
        cache_key = f"social_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                return cached_result
        
        await self._rate_limit()
        
        all_posts = []
        
        # Query platforms in parallel
        tasks = []
        platforms = options.get("platforms", ["twitter", "reddit", "stocktwits"])
        
        if "twitter" in platforms and self.twitter_bearer_token:
            tasks.append(self._search_twitter(query, options))
        if "reddit" in platforms and self.reddit_client_id:
            tasks.append(self._search_reddit(query, options))
        if "stocktwits" in platforms and self.stocktwits_token:
            tasks.append(self._search_stocktwits(query, options))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_posts.extend(result)
        
        # Deduplicate
        unique_posts = self._deduplicate(all_posts)
        
        # Sort by engagement/relevance
        unique_posts.sort(key=lambda x: x.get("engagement_score", 0), reverse=True)
        
        # Calculate sentiment
        sentiment = self._calculate_sentiment(unique_posts)
        
        result = {
            "items": unique_posts[:options.get("max_results", 100)],
            "metadata": {
                "total_found": len(all_posts),
                "unique_count": len(unique_posts),
                "sentiment": sentiment,
                "platforms_used": [p for p in platforms if self._has_api_key(p)]
            }
        }
        
        self.cache[cache_key] = (datetime.now().timestamp(), result)
        
        logging.info(f"✅ Social media found {len(unique_posts)} unique posts")
        return result
    
    async def _search_twitter(self, query: str, options: Dict) -> List[Dict]:
        """Search Twitter/X for mentions"""
        if not self.twitter_bearer_token:
            return []
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Calculate time range
        hours_back = options.get("hours_back", 24)
        start_time = (datetime.now() - timedelta(hours=hours_back)).isoformat() + "Z"
        
        params = {
            "query": query,
            "max_results": min(options.get("limit", 50), 100),
            "tweet.fields": "created_at,public_metrics,author_id,lang",
            "user.fields": "name,username,verified",
            "expansions": "author_id",
            "start_time": start_time
        }
        
        headers = {
            "Authorization": f"Bearer {self.twitter_bearer_token}"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        posts = []
                        for tweet in data.get("data", []):
                            metrics = tweet.get("public_metrics", {})
                            engagement = (
                                metrics.get("like_count", 0) * 1 +
                                metrics.get("retweet_count", 0) * 2 +
                                metrics.get("reply_count", 0) * 1.5
                            )
                            
                            posts.append({
                                "id": tweet.get("id"),
                                "platform": "twitter",
                                "author": tweet.get("author_id"),
                                "content": tweet.get("text"),
                                "created_at": tweet.get("created_at"),
                                "url": f"https://twitter.com/i/web/status/{tweet.get('id')}",
                                "metrics": metrics,
                                "engagement_score": engagement,
                                "sentiment": self._analyze_sentiment(tweet.get("text", "")),
                                "language": tweet.get("lang")
                            })
                        
                        return posts
        except Exception as e:
            logging.debug(f"Twitter search error: {e}")
        
        return []
    
    async def _search_reddit(self, query: str, options: Dict) -> List[Dict]:
        """Search Reddit for mentions"""
        if not self.reddit_client_id:
            return []
        
        # Mock implementation - in production would use PRAW
        # This is a simplified version using aiohttp
        url = "https://www.reddit.com/search.json"
        
        params = {
            "q": query,
            "limit": options.get("limit", 50),
            "sort": options.get("sort", "relevance"),
            "t": options.get("timeframe", "day"),
            "restrict_sr": options.get("subreddit", "") != ""
        }
        
        if "subreddit" in options:
            url = f"https://www.reddit.com/r/{options['subreddit']}/search.json"
        
        headers = {
            "User-Agent": "TradingBot/1.0"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        posts = []
                        for child in data.get("data", {}).get("children", []):
                            post = child.get("data", {})
                            
                            engagement = (
                                post.get("score", 0) * 1 +
                                post.get("num_comments", 0) * 2
                            )
                            
                            posts.append({
                                "id": post.get("id"),
                                "platform": "reddit",
                                "author": post.get("author"),
                                "title": post.get("title"),
                                "content": post.get("selftext"),
                                "created_at": datetime.fromtimestamp(post.get("created_utc", 0)).isoformat(),
                                "url": f"https://reddit.com{post.get('permalink')}",
                                "subreddit": post.get("subreddit"),
                                "score": post.get("score"),
                                "num_comments": post.get("num_comments"),
                                "engagement_score": engagement,
                                "sentiment": self._analyze_sentiment(post.get("title", "") + " " + post.get("selftext", ""))
                            })
                        
                        return posts
        except Exception as e:
            logging.debug(f"Reddit search error: {e}")
        
        return []
    
    async def _search_stocktwits(self, query: str, options: Dict) -> List[Dict]:
        """Search StockTwits for mentions"""
        if not self.stocktwits_token:
            return []
        
        # StockTwits API - simplified
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{query}.json"
        
        params = {
            "limit": options.get("limit", 30)
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        posts = []
                        for message in data.get("messages", []):
                            posts.append({
                                "id": message.get("id"),
                                "platform": "stocktwits",
                                "author": message.get("user", {}).get("username"),
                                "content": message.get("body"),
                                "created_at": message.get("created_at"),
                                "url": message.get("url"),
                                "sentiment": message.get("entities", {}).get("sentiment", {}).get("basic"),
                                "engagement_score": message.get("likes", {}).get("total", 0)
                            })
                        
                        return posts
        except Exception as e:
            logging.debug(f"StockTwits search error: {e}")
        
        return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis for social media posts
        Returns score from -1 (negative) to 1 (positive)
        """
        text_lower = text.lower()
        
        positive_words = {
            'bullish', 'moon', 'rocket', '🚀', 'buy', 'long', 'calls',
            'gain', 'profit', 'green', 'up', 'ath', 'rip', 'hodl',
            'diamond', 'hands', '💎', '🙌', 'yolo', 'undervalued'
        }
        
        negative_words = {
            'bearish', 'crash', 'dump', 'sell', 'short', 'puts',
            'loss', 'red', 'down', 'bagholder', 'rekt', 'panic',
            'overvalued', 'scam', 'fraud', 'lawsuit', 'bankrupt'
        }
        
        words = set(text_lower.split())
        
        positive_count = sum(1 for w in words if w in positive_words)
        negative_count = sum(1 for w in words if w in negative_words)
        
        if positive_count + negative_count > 0:
            return (positive_count - negative_count) / (positive_count + negative_count)
        
        return 0.0
    
    def _calculate_sentiment(self, posts: List[Dict]) -> Dict[str, float]:
        """
        Calculate aggregate sentiment from all posts
        """
        if not posts:
            return {"average": 0.0, "positive": 0, "negative": 0, "neutral": 0}
        
        sentiments = [p.get("sentiment", 0) for p in posts if p.get("sentiment") is not None]
        
        if not sentiments:
            return {"average": 0.0, "positive": 0, "negative": 0, "neutral": len(posts)}
        
        avg_sentiment = sum(sentiments) / len(sentiments)
        positive = sum(1 for s in sentiments if s > 0.1)
        negative = sum(1 for s in sentiments if s < -0.1)
        neutral = len(sentiments) - positive - negative
        
        return {
            "average": float(avg_sentiment),
            "positive": positive,
            "negative": negative,
            "neutral": neutral
        }
    
    def _deduplicate(self, posts: List[Dict]) -> List[Dict]:
        """Remove duplicate posts"""
        unique = []
        seen_content = set()
        
        for post in posts:
            content = post.get("content", "")[:100].lower()
            if content and content not in seen_content:
                seen_content.add(content)
                unique.append(post)
        
        return unique
    
    def _has_api_key(self, platform: str) -> bool:
        """Check if API key exists for platform"""
        keys = {
            "twitter": self.twitter_bearer_token,
            "reddit": self.reddit_client_id,
            "stocktwits": self.stocktwits_token
        }
        return bool(keys.get(platform))
    
    async def _rate_limit(self):
        """Rate limiting"""
        now = datetime.now().timestamp()
        
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if now - ts < 60]
        
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(now)