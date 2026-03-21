"""
Social Sentiment - Analyzes sentiment from social media platforms
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import re
from agentic_trading_system.utils.logger import logger as logging
import numpy as np
class SocialSentiment:
    """
    Fetches and analyzes sentiment from social media
    Platforms: Twitter, Reddit, StockTwits
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API keys
        self.twitter_bearer_token = config.get("twitter_bearer_token")
        self.reddit_client_id = config.get("reddit_client_id")
        self.reddit_client_secret = config.get("reddit_client_secret")
        
        # Platforms to monitor
        self.platforms = config.get("platforms", ["twitter", "reddit"])
        
        # Subreddits to monitor
        self.subreddits = config.get("subreddits", [
            "wallstreetbets", "stocks", "investing", "options"
        ])
        
        # Lookback period
        self.lookback_hours = config.get("lookback_hours", 24)
        
        # Cache
        self.cache = {}
        self.cache_ttl = timedelta(minutes=15)
        
        logging.info(f"📱 SocialSentiment initialized with platforms: {self.platforms}")
    
    async def get_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get sentiment score from social media for a symbol
        """
        logging.info(f"📱 Fetching social sentiment for {symbol}")
        
        # Check cache
        cache_key = f"social_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                return cached_result
        
        try:
            # Gather data from all platforms in parallel
            tasks = []
            
            if "twitter" in self.platforms:
                tasks.append(self._get_twitter_sentiment(symbol))
            
            if "reddit" in self.platforms:
                tasks.append(self._get_reddit_sentiment(symbol))
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            platform_scores = []
            platform_details = []
            
            for result in results:
                if isinstance(result, Exception):
                    logging.warning(f"Social platform error: {result}")
                    continue
                
                if result and result.get("score") is not None:
                    platform_scores.append(result["score"])
                    platform_details.append(result)
            
            if not platform_scores:
                return {
                    "score": 0.5,
                    "confidence": 0.3,
                    "details": {"error": "No social data found"},
                    "source": "social"
                }
            
            # Calculate aggregate
            avg_score = sum(platform_scores) / len(platform_scores)
            
            # Calculate confidence
            confidence = self._calculate_confidence(platform_details)
            
            # Detect trending topics
            trending = await self._detect_trending(symbol, platform_details)
            
            result = {
                "score": float(avg_score),
                "confidence": float(confidence),
                "source": "social",
                "details": {
                    "platforms_analyzed": len(platform_details),
                    "platform_breakdown": platform_details,
                    "total_mentions": sum(p.get("mention_count", 0) for p in platform_details),
                    "trending": trending,
                    "sentiment_velocity": self._calculate_velocity(platform_details)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache result
            self.cache[cache_key] = (datetime.now(), result)
            
            return result
            
        except Exception as e:
            logging.error(f"Error in social sentiment for {symbol}: {e}")
            return {
                "score": 0.5,
                "confidence": 0.3,
                "source": "social",
                "details": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    async def _get_twitter_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment from Twitter"""
        if not self.twitter_bearer_token:
            return None
        
        # Mock data for now (would integrate with Twitter API v2)
        # In production, implement actual Twitter API calls
        mock_sentiment = {
            "platform": "twitter",
            "score": 0.65 if symbol == "AAPL" else 0.55,
            "confidence": 0.6,
            "mention_count": 150,
            "unique_authors": 45,
            "sentiment_distribution": {
                "positive": 0.45,
                "neutral": 0.35,
                "negative": 0.20
            },
            "top_hashtags": [f"${symbol}", "stocks", "investing"],
            "timestamp": datetime.now().isoformat()
        }
        
        return mock_sentiment
    
    async def _get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get sentiment from Reddit"""
        # Mock data for now (would integrate with Reddit API)
        mock_sentiment = {
            "platform": "reddit",
            "score": 0.70 if symbol == "GME" else 0.52,
            "confidence": 0.55,
            "mention_count": 75,
            "unique_authors": 30,
            "subreddits": ["wallstreetbets", "stocks"],
            "top_posts": [
                {"title": f"{symbol} discussion thread", "score": 150},
                {"title": f"Why {symbol} could moon", "score": 89}
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return mock_sentiment
    
    def _calculate_confidence(self, platform_results: List[Dict]) -> float:
        """Calculate confidence in social sentiment"""
        if not platform_results:
            return 0.3
        
        # Average platform confidence
        avg_confidence = np.mean([p.get("confidence", 0.5) for p in platform_results])
        
        # Volume factor (more mentions = higher confidence)
        total_mentions = sum(p.get("mention_count", 0) for p in platform_results)
        volume_factor = min(1.0, total_mentions / 500)  # Cap at 500 mentions
        
        # Agreement between platforms
        if len(platform_results) > 1:
            scores = [p.get("score", 0.5) for p in platform_results]
            agreement = 1 - min(1.0, np.std(scores) * 2)
        else:
            agreement = 0.6
        
        confidence = (
            avg_confidence * 0.4 +
            volume_factor * 0.4 +
            agreement * 0.2
        )
        
        return float(min(1.0, max(0.3, confidence)))
    
    async def _detect_trending(self, symbol: str, platform_results: List[Dict]) -> bool:
        """Detect if symbol is trending on social media"""
        total_mentions = sum(p.get("mention_count", 0) for p in platform_results)
        
        # Simple threshold-based detection
        trending_threshold = self.config.get("trending_threshold", 100)
        
        return total_mentions > trending_threshold
    
    def _calculate_velocity(self, platform_results: List[Dict]) -> float:
        """Calculate sentiment velocity (rate of change)"""
        # In production, would compare with historical data
        # For now, return neutral
        return 0.5