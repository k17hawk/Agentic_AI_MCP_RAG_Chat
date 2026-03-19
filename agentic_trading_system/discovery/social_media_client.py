"""
Social Media Client - Aggregates social media data
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio
import re

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

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
        platform_results = {}
        
        # Query platforms in parallel
        tasks = []
        platforms = options.get("platforms", ["reddit", "stocktwits"])  # Twitter removed by default (requires paid tier)
        
        if "twitter" in platforms and self.twitter_bearer_token:
            tasks.append(self._search_twitter(query, options))
        if "reddit" in platforms:
            tasks.append(self._search_reddit(query, options))
        if "stocktwits" in platforms:
            tasks.append(self._search_stocktwits(query, options))
        
        if not tasks:
            logging.warning("⚠️ No social media platforms configured")
            return {
                "items": [],
                "metadata": {
                    "total_found": 0,
                    "unique_count": 0,
                    "sentiment": {"average": 0.0, "positive": 0, "negative": 0, "neutral": 0},
                    "platforms_used": []
                }
            }
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        platform_names = []
        for i, result in enumerate(results):
            platform = platforms[i] if i < len(platforms) else f"platform_{i}"
            if isinstance(result, Exception):
                logging.warning(f"⚠️ Error from {platform}: {result}")
                platform_results[platform] = {"status": "error", "count": 0}
            elif isinstance(result, list):
                all_posts.extend(result)
                platform_results[platform] = {"status": "success", "count": len(result)}
                platform_names.append(platform)
                logging.info(f"✅ {platform} returned {len(result)} posts")
            else:
                platform_results[platform] = {"status": "error", "count": 0}
        
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
                "platforms_used": platform_names,
                "platform_results": platform_results
            }
        }
        
        self.cache[cache_key] = (datetime.now().timestamp(), result)
        
        logging.info(f"✅ Social media found {len(unique_posts)} unique posts")
        return result
    
    async def _search_twitter(self, query: str, options: Dict) -> List[Dict]:
        """Search Twitter/X for mentions"""
        if not self.twitter_bearer_token:
            return []
        
        # Extract ticker or clean query for Twitter
        search_query = self._format_twitter_query(query)
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Calculate time range
        hours_back = options.get("hours_back", 24)
        start_time = (datetime.now() - timedelta(hours=hours_back)).isoformat() + "Z"
        
        params = {
            "query": search_query,
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
                    elif response.status == 429:
                        logging.warning("⚠️ Twitter rate limit hit")
                        return []
                    else:
                        logging.debug(f"Twitter returned status {response.status}")
                        return []
        except Exception as e:
            logging.debug(f"Twitter search error: {e}")
        
        return []
    
    def _format_twitter_query(self, query: str) -> str:
        """Format query for Twitter search"""
        # Extract ticker if present
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', query.upper())
        if ticker_match:
            ticker = ticker_match.group()
            return f"${ticker} OR {ticker} stock OR {ticker} trading"
        
        # For company names
        words = query.split()[:3]  # Take first 3 words max
        return " OR ".join(words)
    
    async def _search_reddit(self, query: str, options: Dict) -> List[Dict]:
        """
        Search Reddit for mentions using public JSON endpoints.
        """
        # Determine subreddits to search
        subreddits = options.get("subreddits", ["wallstreetbets", "stocks", "investing", "finance"])
        
        all_posts = []
        
        for subreddit in subreddits[:2]:  # Limit to first 2 to avoid rate limits
            posts = await self._search_reddit_subreddit(query, subreddit, options)
            all_posts.extend(posts)
            await asyncio.sleep(1)  # Rate limit between subreddits
        
        return all_posts
    
    async def _search_reddit_subreddit(self, query: str, subreddit: str, options: Dict) -> List[Dict]:
        """Search a specific subreddit"""
        url = f"https://www.reddit.com/r/{subreddit}/search.json"
        
        # Format query for Reddit
        search_query = self._format_reddit_query(query)
        
        params = {
            "q": search_query,
            "limit": options.get("limit", 25),
            "sort": options.get("sort", "relevance"),
            "t": options.get("timeframe", "week"),
            "restrict_sr": True
        }
        
        headers = {
            "User-Agent": "TradingBot/1.0 (market sentiment aggregator)"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        posts = []
                        for child in data.get("data", {}).get("children", []):
                            post = child.get("data", {})
                            
                            # Skip if removed/deleted
                            if post.get("removed_by_category") or post.get("selftext") == "[removed]":
                                continue
                            
                            engagement = (
                                post.get("score", 0) * 1 +
                                post.get("num_comments", 0) * 2 +
                                post.get("upvote_ratio", 0.5) * 10
                            )
                            
                            posts.append({
                                "id": post.get("id"),
                                "platform": "reddit",
                                "subreddit": subreddit,
                                "author": post.get("author"),
                                "title": post.get("title"),
                                "content": post.get("selftext", "")[:500],
                                "created_at": datetime.fromtimestamp(post.get("created_utc", 0)).isoformat(),
                                "url": f"https://reddit.com{post.get('permalink')}",
                                "score": post.get("score"),
                                "num_comments": post.get("num_comments"),
                                "upvote_ratio": post.get("upvote_ratio"),
                                "engagement_score": engagement,
                                "sentiment": self._analyze_sentiment(
                                    post.get("title", "") + " " + post.get("selftext", "")
                                )
                            })
                        
                        return posts
        except Exception as e:
            logging.debug(f"Reddit search error for r/{subreddit}: {e}")
        
        return []
    
    def _format_reddit_query(self, query: str) -> str:
        """Format query for Reddit search"""
        # Extract ticker if present
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', query.upper())
        if ticker_match:
            ticker = ticker_match.group()
            return f"({ticker}) OR (${ticker}) OR (\"{ticker} stock\")"
        
        # For company names
        return f"\"{query}\""
    
    async def _search_stocktwits(self, query: str, options: Dict) -> List[Dict]:
        """
        Search StockTwits for mentions using public symbol stream endpoints.
        """
        # Extract ticker from query
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', query.upper())
        ticker = ticker_match.group() if ticker_match else query.upper().replace(" ", "")
        
        # Try multiple endpoints
        endpoints = [
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json",
            f"https://api.stocktwits.com/api/2/streams/symbol/{ticker.lower()}.json"
        ]
        
        for url in endpoints:
            params = {
                "limit": options.get("limit", 30)
            }
            
            headers = {}
            if self.stocktwits_token:
                params["access_token"] = self.stocktwits_token
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, headers=headers) as response:
                        if response.status == 200:
                            data = await response.json()
                            
                            posts = []
                            for message in data.get("messages", []):
                                # StockTwits sentiment
                                raw_sentiment = message.get("entities", {}).get("sentiment", {}).get("basic")
                                sentiment_score = (
                                    1.0 if raw_sentiment == "Bullish" else
                                    -1.0 if raw_sentiment == "Bearish" else
                                    0.0
                                )
                                
                                # Calculate engagement
                                likes = message.get("likes", {}).get("total", 0)
                                reshares = message.get("reshares", {}).get("total", 0)
                                engagement = likes + (reshares * 2)
                                
                                posts.append({
                                    "id": message.get("id"),
                                    "platform": "stocktwits",
                                    "author": message.get("user", {}).get("username"),
                                    "author_followers": message.get("user", {}).get("followers", 0),
                                    "content": message.get("body"),
                                    "created_at": message.get("created_at"),
                                    "url": message.get("url"),
                                    "sentiment": sentiment_score,
                                    "sentiment_label": raw_sentiment,
                                    "engagement_score": engagement,
                                    "likes": likes,
                                    "reshares": reshares
                                })
                            
                            if posts:
                                return posts
                            
            except Exception as e:
                logging.debug(f"StockTwits search error for {ticker}: {e}")
        
        return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple sentiment analysis for social media posts
        Returns score from -1 (negative) to 1 (positive)
        """
        if not text:
            return 0.0
            
        text_lower = text.lower()
        
        positive_words = {
            'bullish', 'moon', 'rocket', '🚀', 'buy', 'long', 'calls',
            'gain', 'profit', 'green', 'up', 'ath', 'hodl',
            'diamond', 'hands', '💎', '🙌', 'yolo', 'undervalued',
            'mooning', 'rip', 'tendies', 'squeeze', 'moonsoon',
            'good', 'great', 'awesome', 'excellent', 'amazing'
        }
        
        negative_words = {
            'bearish', 'crash', 'dump', 'sell', 'short', 'puts',
            'loss', 'red', 'down', 'bagholder', 'rekt', 'panic',
            'overvalued', 'scam', 'fraud', 'lawsuit', 'bankrupt',
            'fuck', 'shit', 'bad', 'terrible', 'awful', 'worse',
            'declining', 'plunge', 'tank', 'bloodbath'
        }
        
        words = set(re.findall(r'\b\w+\b', text_lower))
        
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
        """Check if platform is available"""
        keys = {
            "twitter": self.twitter_bearer_token,
            "reddit": True,
            "stocktwits": True,
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