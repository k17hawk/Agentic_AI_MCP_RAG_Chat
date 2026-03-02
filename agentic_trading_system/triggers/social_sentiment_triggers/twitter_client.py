"""
Twitter Client - Fetches tweets and analyzes sentiment
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import asyncio
import aiohttp
import re
from loguru import logger


class TwitterClient:
    """
    Fetches tweets about stocks and analyzes sentiment
    Uses Twitter API v2
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.bearer_token = config.get("twitter_bearer_token")
        
        # Rate limiting
        self.request_timestamps = []
        self.max_requests_per_minute = 30
        
        # Cache
        self.cache = {}
        
        # Sentiment lexicon for tweets
        self.positive_terms = {
            'bullish', 'moon', 'rocket', '🚀', 'buy', 'long', 'calls',
            'undervalued', 'gem', 'breakout', 'squeeze', 'tendies',
            'gain', 'profit', 'green', 'up', 'ath', 'rip', 'hodl'
        }
        
        self.negative_terms = {
            'bearish', 'crash', 'dump', 'sell', 'short', 'puts',
            'overvalued', 'scam', 'fraud', 'lawsuit', 'bankrupt',
            'loss', 'red', 'down', 'bagholder', 'rekt', 'panic'
        }
        
        logger.info("TwitterClient initialized")
    
    async def scan(self, watchlist: List[str], lookback_minutes: int) -> Dict[str, Dict]:
        """
        Scan Twitter for mentions of watched stocks
        """
        results = {symbol: {"mentions": 0, "sentiment": 0} for symbol in watchlist}
        
        if not self.bearer_token:
            logger.warning("Twitter API token not configured")
            return results
        
        try:
            # Build query
            query = self._build_query(watchlist)
            
            # Fetch recent tweets
            tweets = await self._fetch_tweets(query, lookback_minutes)
            
            # Analyze each tweet
            for tweet in tweets:
                symbol = self._extract_symbol(tweet["text"], watchlist)
                if symbol:
                    sentiment = self._analyze_sentiment(tweet["text"])
                    
                    results[symbol]["mentions"] += 1
                    results[symbol]["sentiment"] += sentiment
                    
                    # Store tweet data
                    if "tweets" not in results[symbol]:
                        results[symbol]["tweets"] = []
                    
                    results[symbol]["tweets"].append({
                        "text": tweet["text"][:100],  # Truncate for storage
                        "sentiment": sentiment,
                        "likes": tweet.get("public_metrics", {}).get("like_count", 0),
                        "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
                        "author": tweet.get("author_id"),
                        "created_at": tweet.get("created_at")
                    })
            
            # Calculate average sentiment
            for symbol in results:
                if results[symbol]["mentions"] > 0:
                    results[symbol]["sentiment"] /= results[symbol]["mentions"]
                    
                    # Sort tweets by engagement
                    if "tweets" in results[symbol]:
                        results[symbol]["tweets"].sort(
                            key=lambda x: x["likes"] + x["retweets"],
                            reverse=True
                        )
                        results[symbol]["top_tweet"] = results[symbol]["tweets"][0]
            
        except Exception as e:
            logger.error(f"Twitter scan error: {e}")
        
        return results
    
    def _build_query(self, watchlist: List[str]) -> str:
        """Build Twitter API query from watchlist"""
        # Create OR query for all symbols
        symbols_query = " OR ".join([f"${s}" for s in watchlist])
        
        # Add filters
        query = f"({symbols_query}) -is:retweet lang:en"
        
        return query
    
    async def _fetch_tweets(self, query: str, lookback_minutes: int) -> List[Dict]:
        """Fetch tweets from Twitter API"""
        await self._rate_limit()
        
        url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Calculate start time
        start_time = (datetime.utcnow() - timedelta(minutes=lookback_minutes)).isoformat() + "Z"
        
        params = {
            "query": query,
            "max_results": 100,
            "tweet.fields": "created_at,public_metrics,author_id",
            "start_time": start_time
        }
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("data", [])
                else:
                    error_text = await response.text()
                    logger.error(f"Twitter API error {response.status}: {error_text}")
                    return []
    
    def _extract_symbol(self, text: str, watchlist: List[str]) -> Optional[str]:
        """Extract stock symbol from tweet text"""
        text_upper = text.upper()
        
        for symbol in watchlist:
            # Look for $SYMBOL or #SYMBOL or just SYMBOL in context
            if f"${symbol}" in text or f"#{symbol}" in text:
                return symbol
            
            # Simple word boundary check
            pattern = r'\b' + symbol + r'\b'
            if re.search(pattern, text_upper):
                return symbol
        
        return None
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment of tweet text"""
        text_lower = text.lower()
        words = text_lower.split()
        
        positive_count = 0
        negative_count = 0
        
        # Count sentiment words
        for word in words:
            if word in self.positive_terms:
                positive_count += 1
            elif word in self.negative_terms:
                negative_count += 1
        
        # Check for emojis
        if '🚀' in text:
            positive_count += 2
        if '💎' in text and '🙌' in text:  # Diamond hands
            positive_count += 1
        
        # Calculate sentiment (-1 to 1)
        total = positive_count + negative_count
        if total > 0:
            sentiment = (positive_count - negative_count) / total
        else:
            sentiment = 0
        
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
            logger.warning(f"Twitter rate limit reached, waiting {wait_time}s")
            await asyncio.sleep(wait_time)
        
        self.request_timestamps.append(now)