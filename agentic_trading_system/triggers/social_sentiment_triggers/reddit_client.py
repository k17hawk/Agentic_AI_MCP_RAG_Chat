
"""
Reddit Client - Fetches posts from finance subreddits
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
import re
from loguru import logger

class RedditClient:
    """
    Fetches posts from finance-related subreddits
    Subreddits: wallstreetbets, investing, stocks, options
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.client_id = config.get("reddit_client_id")
        self.client_secret = config.get("reddit_client_secret")
        self.user_agent = config.get("reddit_user_agent", "TradingBot/1.0")
        
        # Subreddits to monitor
        self.subreddits = config.get("subreddits", [
            "wallstreetbets",
            "stocks",
            "investing",
            "options",
            "stockmarket",
            "pennystocks"
        ])
        
        # Rate limiting
        self.request_timestamps = []
        self.max_requests_per_minute = 30
        
        # Cache
        self.cache = {}
        self.access_token = None
        self.token_expiry = None
        
        # Sentiment lexicon (similar to Twitter but with Reddit-specific terms)
        self.positive_terms = {
            'bullish', 'moon', 'rocket', '🚀', 'buy', 'long', 'calls',
            'undervalued', 'gem', 'breakout', 'squeeze', 'tendies',
            'gain', 'profit', 'green', 'up', 'ath', 'rip', 'hodl',
            'diamond', 'hands', '💎', '🙌', 'yolo'
        }
        
        self.negative_terms = {
            'bearish', 'crash', 'dump', 'sell', 'short', 'puts',
            'overvalued', 'scam', 'fraud', 'lawsuit', 'bankrupt',
            'loss', 'red', 'down', 'bagholder', 'rekt', 'panic',
            'fuck', 'shit', 'gay', 'cancer'
        }
        
        logger.info("RedditClient initialized")
    
    async def scan(self, watchlist: List[str], lookback_minutes: int) -> Dict[str, Dict]:
        """
        Scan Reddit for mentions of watched stocks
        """
        results = {symbol: {"mentions": 0, "sentiment": 0} for symbol in watchlist}
        
        try:
            # Get access token if needed
            await self._ensure_token()
            
            # Fetch posts from each subreddit
            all_posts = []
            for subreddit in self.subreddits:
                posts = await self._fetch_subreddit_posts(subreddit, lookback_minutes)
                all_posts.extend(posts)
                await asyncio.sleep(0.5)  # Be nice to Reddit
            
            # Analyze each post
            for post in all_posts:
                symbol = self._extract_symbol(post, watchlist)
                if symbol:
                    sentiment = self._analyze_sentiment(post)
                    
                    results[symbol]["mentions"] += 1
                    results[symbol]["sentiment"] += sentiment
                    
                    # Store post data
                    if "posts" not in results[symbol]:
                        results[symbol]["posts"] = []
                    
                    results[symbol]["posts"].append({
                        "title": post.get("title", "")[:100],
                        "sentiment": sentiment,
                        "score": post.get("score", 0),
                        "num_comments": post.get("num_comments", 0),
                        "subreddit": post.get("subreddit"),
                        "created_utc": post.get("created_utc")
                    })
            
            # Calculate average sentiment
            for symbol in results:
                if results[symbol]["mentions"] > 0:
                    results[symbol]["sentiment"] /= results[symbol]["mentions"]
                    
                    # Sort posts by engagement
                    if "posts" in results[symbol]:
                        results[symbol]["posts"].sort(
                            key=lambda x: x["score"] + x["num_comments"],
                            reverse=True
                        )
                        results[symbol]["top_post"] = results[symbol]["posts"][0]
            
        except Exception as e:
            logger.error(f"Reddit scan error: {e}")
        
        return results
    
    async def _ensure_token(self):
        """Get or refresh access token"""
        if self.access_token and self.token_expiry and datetime.utcnow() < self.token_expiry:
            return
        
        if not self.client_id or not self.client_secret:
            logger.warning("Reddit API credentials not configured")
            return
        
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        
        data = {
            "grant_type": "client_credentials"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://www.reddit.com/api/v1/access_token",
                auth=auth,
                data=data,
                headers={"User-Agent": self.user_agent}
            ) as response:
                if response.status == 200:
                    token_data = await response.json()
                    self.access_token = token_data["access_token"]
                    self.token_expiry = datetime.utcnow() + timedelta(seconds=token_data["expires_in"])
                    logger.info("Reddit access token obtained")
                else:
                    logger.error(f"Failed to get Reddit token: {response.status}")
    
    async def _fetch_subreddit_posts(self, subreddit: str, lookback_minutes: int) -> List[Dict]:
        """Fetch posts from a specific subreddit"""
        await self._rate_limit()
        
        if not self.access_token:
            return []
        
        url = f"https://oauth.reddit.com/r/{subreddit}/new"
        
        params = {
            "limit": 50  # Max per request
        }
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "User-Agent": self.user_agent
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    posts = data.get("data", {}).get("children", [])
                    
                    # Filter by time
                    cutoff_time = datetime.utcnow() - timedelta(minutes=lookback_minutes)
                    
                    filtered_posts = []
                    for post in posts:
                        post_data = post["data"]
                        created_time = datetime.utcfromtimestamp(post_data["created_utc"])
                        
                        if created_time > cutoff_time:
                            filtered_posts.append(post_data)
                    
                    return filtered_posts
                else:
                    logger.error(f"Reddit API error {response.status} for r/{subreddit}")
                    return []
    
    def _extract_symbol(self, post: Dict, watchlist: List[str]) -> Optional[str]:
        """Extract stock symbol from Reddit post"""
        title = post.get("title", "").upper()
        selftext = post.get("selftext", "").upper()
        
        combined = title + " " + selftext
        
        for symbol in watchlist:
            # Look for $SYMBOL
            if f"${symbol}" in combined:
                return symbol
            
            # Look for symbol in brackets [SYMBOL]
            if f"[{symbol}]" in combined:
                return symbol
            
            # Look for symbol in parentheses (SYMBOL)
            if f"({symbol})" in combined:
                return symbol
            
            # Simple word boundary check
            pattern = r'\b' + symbol + r'\b'
            if re.search(pattern, combined):
                return symbol
        
        return None
    
    def _analyze_sentiment(self, post: Dict) -> float:
        """Analyze sentiment of Reddit post"""
        title = post.get("title", "").lower()
        selftext = post.get("selftext", "").lower()
        
        combined = title + " " + selftext
        words = combined.split()
        
        positive_count = 0
        negative_count = 0
        
        # Count sentiment words
        for word in words:
            if word in self.positive_terms:
                positive_count += 1
            elif word in self.negative_terms:
                negative_count += 1
        
        # Check for emojis
        if '🚀' in combined:
            positive_count += 2
        if '💎' in combined and '🙌' in combined:
            positive_count += 2
        if '🖕' in combined:
            negative_count += 1
        
        # Calculate sentiment (-1 to 1)
        total = positive_count + negative_count
        if total > 0:
            sentiment = (positive_count - negative_count) / total
        else:
            sentiment = 0
        
        # Adjust by post score
        score = post.get("score", 0)
        if abs(score) > 10:
            sentiment *= (1 + min(0.5, abs(score) / 100))  # Max 50% boost
        
        return sentiment
    
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
            logger.warning(f"Reddit rate limit reached, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
        
        self.request_timestamps.append(now)