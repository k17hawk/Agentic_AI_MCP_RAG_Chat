# =============================================================================
# discovery/social_media_client.py (UPDATED)
# =============================================================================
"""
Social Media Client - Aggregates social media sentiment
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio
import re

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

# Import from new config structure
from agentic_trading_system.constants import Source, RateLimit, CacheTTL, Sentiment
from agentic_trading_system.config.config__entity import SocialMediaConfig


class SocialMediaClient:
    """
    Aggregates social media content from Reddit, StockTwits, and Twitter.
    """

    def __init__(self, config: SocialMediaConfig):
        """
        Initialize social media client.
        
        Args:
            config: SocialMediaConfig object
        """
        self.config = config

        # API keys/tokens
        self.twitter_bearer_token = config.twitter_bearer_token

        # Reddit settings
        self.reddit_user_agent = config.reddit_user_agent
        self.reddit_limit = config.reddit_limit
        self.reddit_time_filter = config.reddit_time_filter.value if hasattr(config.reddit_time_filter, 'value') else config.reddit_time_filter
        self.reddit_subreddits = config.reddit_subreddits
        self.reddit_comment_limit = config.reddit_comment_limit
        self.reddit_sort = config.reddit_sort

        # StockTwits settings
        self.stocktwits_limit = config.stocktwits_limit

        # Platform selection
        self.platforms = config.platforms

        # Rate limiting
        self.rate_limit = config.rate_limit
        self.request_delay = config.request_delay
        self.request_timestamps: List[float] = []

        # Cache
        self.cache: Dict = {}
        self.cache_ttl = config.cache_ttl_minutes * 60

        # Sentiment cache
        self.sentiment_cache: Dict = {}

        logging.info(f"✅ SocialMediaClient initialized with platforms: {self.platforms}")

    @retry(max_attempts=2, delay=1.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search social media for content.
        
        Args:
            query: Search query
            options: Additional options
            
        Returns:
            Dict with 'items' and 'metadata' keys
        """
        options = options or {}
        logging.info(f"📱 Social media search for: '{query}'")

        # Check cache
        cache_key = f"social_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached social results for '{query}'")
                return cached_result

        await self._rate_limit()

        # Determine which platforms to query
        platforms = options.get("platforms", self.platforms)

        all_items: List[Dict] = []
        platform_results: Dict = {}

        # Build tasks
        tasks = []
        task_names = []

        if "reddit" in platforms:
            tasks.append(self._search_reddit(query, options))
            task_names.append("reddit")

        if "stocktwits" in platforms:
            tasks.append(self._search_stocktwits(query, options))
            task_names.append("stocktwits")

        if "twitter" in platforms and self.twitter_bearer_token:
            tasks.append(self._search_twitter(query, options))
            task_names.append("twitter")

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for name, result in zip(task_names, results):
                if isinstance(result, Exception):
                    logging.warning(f"⚠️ {name} error: {result}")
                    platform_results[name] = {"status": "error", "count": 0}
                else:
                    items = result.get("items", [])
                    all_items.extend(items)
                    platform_results[name] = {
                        "status": "success",
                        "count": len(items),
                        "metadata": result.get("metadata", {})
                    }

        # Sort by recency
        all_items.sort(key=lambda x: x.get("created_utc", 0), reverse=True)

        result = {
            "items": all_items[:options.get("max_results", 50)],
            "metadata": {
                "total_found": len(all_items),
                "platform_results": platform_results,
                "platforms_used": [p for p in platforms if p in task_names]
            }
        }

        self.cache[cache_key] = (datetime.now().timestamp(), result)

        logging.info(f"✅ Social media: {len(all_items)} items from {len(platform_results)} platforms")
        return result

    async def _search_reddit(self, query: str, options: Dict) -> Dict[str, Any]:
        """
        Search Reddit using the search endpoint.
        
        Args:
            query: Search query
            options: Search options
            
        Returns:
            Dict with items and metadata
        """
        # Get settings
        subreddits = options.get("subreddits", self.reddit_subreddits)
        limit = options.get("reddit_limit", self.reddit_limit)
        sort = options.get("reddit_sort", self.reddit_sort)
        time_filter = options.get("reddit_time_filter", self.reddit_time_filter)

        headers = {
            'User-Agent': self.reddit_user_agent
        }

        all_posts = []
        successful_subreddits = 0

        async with aiohttp.ClientSession() as session:
            for subreddit in subreddits:
                try:
                    # Use search endpoint
                    url = f"https://www.reddit.com/r/{subreddit}/search.json"
                    params = {
                        'q': query,
                        'restrict_sr': 'true',
                        'sort': sort,
                        't': time_filter,
                        'limit': limit
                    }

                    await asyncio.sleep(self.request_delay)  # Rate limiting

                    async with session.get(
                        url, headers=headers, params=params,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()

                            if 'data' in data and 'children' in data['data']:
                                posts = data['data']['children']
                                successful_subreddits += 1

                                for post in posts:
                                    post_data = post['data']
                                    all_posts.append(self._format_reddit_post(post_data, subreddit))

                        elif response.status == 429:
                            logging.warning(f"⚠️ Rate limited on r/{subreddit}")
                            await asyncio.sleep(5)
                        else:
                            logging.debug(f"Reddit search error on r/{subreddit}: {response.status}")

                except asyncio.TimeoutError:
                    logging.warning(f"⏰ Reddit timeout for r/{subreddit}")
                except Exception as e:
                    logging.warning(f"⚠️ Reddit error for r/{subreddit}: {e}")

        logging.info(f"✅ Reddit: {len(all_posts)} posts from {successful_subreddits}/{len(subreddits)} subreddits")

        return {
            "items": all_posts,
            "metadata": {
                "subreddits_searched": len(subreddits),
                "successful_subreddits": successful_subreddits,
                "query": query
            }
        }

    def _format_reddit_post(self, post_data: Dict, subreddit: str) -> Dict[str, Any]:
        """Format Reddit post for output."""
        created_utc = post_data.get('created_utc', 0)
        
        return {
            "title": post_data.get('title', ''),
            "content": post_data.get('selftext', '')[:500],
            "url": f"https://reddit.com{post_data.get('permalink', '')}",
            "source": Source.REDDIT,
            "platform": Source.REDDIT,
            "subreddit": subreddit,
            "score": post_data.get('score', 0),
            "num_comments": post_data.get('num_comments', 0),
            "created_utc": created_utc,
            "published_at": datetime.fromtimestamp(created_utc).isoformat() if created_utc else None,
            "author": post_data.get('author', ''),
            "upvote_ratio": post_data.get('upvote_ratio', 0),
            "relevance_score": 1.0,
            "type": "social_post"
        }

    async def _search_stocktwits(self, query: str, options: Dict) -> Dict[str, Any]:
        """
        Search StockTwits for messages about a ticker.
        
        Args:
            query: Search query (usually a ticker)
            options: Search options
            
        Returns:
            Dict with items and metadata
        """
        # Extract ticker from query
        ticker = self._extract_ticker(query)
        if not ticker:
            logging.debug(f"No ticker found in query: {query}")
            return {"items": [], "metadata": {"error": "No ticker found"}}

        limit = options.get("stocktwits_limit", self.stocktwits_limit)

        # StockTwits API is public read-only
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

        try:
            async with aiohttp.ClientSession() as session:
                await asyncio.sleep(self.request_delay)

                async with session.get(
                    url,
                    params={"limit": limit},
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        messages = data.get("messages", [])

                        items = []
                        for msg in messages[:limit]:
                            created_at = msg.get("created_at", "")
                            items.append({
                                "title": f"@{msg.get('user', {}).get('username', '')} on {ticker}",
                                "content": msg.get("body", ""),
                                "url": f"https://stocktwits.com/{msg.get('user', {}).get('username', '')}/message/{msg.get('id', '')}",
                                "source": Source.STOCKTWITS,
                                "platform": Source.STOCKTWITS,
                                "ticker": ticker,
                                "created_at": created_at,
                                "published_at": self._parse_stocktwits_date(created_at),
                                "sentiment": self._get_sentiment_from_message(msg),
                                "user": msg.get("user", {}).get("username", ""),
                                "likes": msg.get("likes", {}).get("total", 0),
                                "relevance_score": 0.8,
                                "type": "social_post"
                            })

                        logging.info(f"✅ StockTwits: {len(items)} messages for {ticker}")
                        return {
                            "items": items,
                            "metadata": {
                                "ticker": ticker,
                                "total_messages": len(messages)
                            }
                        }

        except asyncio.TimeoutError:
            logging.warning(f"⏰ StockTwits timeout for {ticker}")
        except Exception as e:
            logging.warning(f"⚠️ StockTwits error: {e}")

        return {"items": [], "metadata": {"error": "Failed to fetch"}}

    async def _search_twitter(self, query: str, options: Dict) -> Dict[str, Any]:
        """
        Search Twitter (X) for tweets.
        
        Args:
            query: Search query
            options: Search options
            
        Returns:
            Dict with items and metadata
        """
        if not self.twitter_bearer_token:
            return {"items": [], "metadata": {"error": "Twitter token not configured"}}

        url = "https://api.twitter.com/2/tweets/search/recent"
        headers = {
            "Authorization": f"Bearer {self.twitter_bearer_token}"
        }

        max_results = options.get("twitter_limit", 20)

        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,author_id",
            "user.fields": "username,name"
        }

        try:
            async with aiohttp.ClientSession() as session:
                await asyncio.sleep(self.request_delay)

                async with session.get(
                    url, headers=headers, params=params,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        tweets = data.get("data", [])

                        items = []
                        for tweet in tweets:
                            created_at = tweet.get("created_at", "")
                            items.append({
                                "title": f"Tweet from @{tweet.get('author_id', 'unknown')}",
                                "content": tweet.get("text", ""),
                                "url": f"https://twitter.com/i/web/status/{tweet.get('id', '')}",
                                "source": Source.TWITTER,
                                "platform": Source.TWITTER,
                                "published_at": created_at,
                                "like_count": tweet.get("public_metrics", {}).get("like_count", 0),
                                "retweet_count": tweet.get("public_metrics", {}).get("retweet_count", 0),
                                "reply_count": tweet.get("public_metrics", {}).get("reply_count", 0),
                                "relevance_score": 0.7,
                                "type": "social_post"
                            })

                        logging.info(f"✅ Twitter: {len(items)} tweets for '{query}'")
                        return {
                            "items": items,
                            "metadata": {
                                "query": query,
                                "total_tweets": len(tweets)
                            }
                        }

        except asyncio.TimeoutError:
            logging.warning(f"⏰ Twitter timeout for '{query}'")
        except Exception as e:
            logging.warning(f"⚠️ Twitter error: {e}")

        return {"items": [], "metadata": {"error": "Failed to fetch"}}

    def _format_reddit_post(self, post_data: Dict, subreddit: str) -> Dict[str, Any]:
        """Format Reddit post for output."""
        created_utc = post_data.get('created_utc', 0)

        return {
            "title": post_data.get('title', ''),
            "content": post_data.get('selftext', '')[:500],
            "url": f"https://reddit.com{post_data.get('permalink', '')}",
            "source": Source.REDDIT,
            "platform": Source.REDDIT,
            "subreddit": subreddit,
            "score": post_data.get('score', 0),
            "num_comments": post_data.get('num_comments', 0),
            "created_utc": created_utc,
            "published_at": datetime.fromtimestamp(created_utc).isoformat() if created_utc else None,
            "author": post_data.get('author', ''),
            "upvote_ratio": post_data.get('upvote_ratio', 0),
            "relevance_score": 1.0,
            "type": "social_post"
        }

    def _extract_ticker(self, query: str) -> Optional[str]:
        """Extract ticker symbol from query."""
        query_upper = query.upper().strip()

        # Direct ticker match
        if re.match(r'^[A-Z]{1,5}$', query_upper):
            skip = {'A', 'I', 'THE', 'AND', 'FOR', 'NEWS', 'STOCK'}
            if query_upper not in skip:
                return query_upper

        # Find first all-caps word
        match = re.search(r'\b([A-Z]{1,5})\b', query)
        if match:
            candidate = match.group(1)
            skip = {'THE', 'AND', 'FOR', 'WITH', 'FROM', 'NEWS', 'STOCK'}
            if candidate not in skip:
                return candidate

        return None

    def _parse_stocktwits_date(self, date_str: str) -> Optional[str]:
        """Parse StockTwits date format."""
        if not date_str:
            return None
        try:
            # StockTwits format: "2024-01-15T10:30:00Z"
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.isoformat()
        except Exception:
            return datetime.now().isoformat()

    def _get_sentiment_from_message(self, message: Dict) -> str:
        """Extract sentiment from StockTwits message."""
        # Check for sentiment from message body
        body = message.get("body", "").lower()

        # Simple sentiment detection
        positive_words = ['bullish', '🚀', 'moon', 'gain', 'up', 'higher', 'breakout']
        negative_words = ['bearish', 'dump', 'crash', 'down', 'lower', 'rejection']

        pos_count = sum(1 for w in positive_words if w in body)
        neg_count = sum(1 for w in negative_words if w in body)

        if pos_count > neg_count:
            return Sentiment.POSITIVE
        elif neg_count > pos_count:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.NEUTRAL

    async def _rate_limit(self):
        """Enforce rate limiting."""
        now = datetime.now().timestamp()
        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if now - ts < RateLimit.WINDOW_SECONDS]

        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = RateLimit.WINDOW_SECONDS - (now - self.request_timestamps[0])
            if sleep_time > 0:
                logging.warning(f"⏳ Rate limit reached, waiting {sleep_time:.1f}s")
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(now)

    async def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        self.sentiment_cache.clear()
        logging.info("🧹 Social media cache cleared")