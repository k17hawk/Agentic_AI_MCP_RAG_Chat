"""
Social Sentiment Trigger - Monitors social media for sentiment
"""
from typing import List, Optional, Dict
from datetime import datetime, timedelta
import asyncio
from logger import logging as logger

from triggers.base_trigger import BaseTrigger, TriggerEvent
from triggers.social_sentiment_triggers.twitter_client import TwitterClient
from triggers.social_sentiment_triggers.reddit_client import RedditClient

class SocialSentimentTrigger(BaseTrigger):
    """
    Monitors social media for sentiment signals
    - Twitter: Track mentions, hashtags, influencer tweets
    - Reddit: Monitor subreddits (wallstreetbets, investing, stocks)
    """
    
    def __init__(self, config: dict, memory_agent=None, message_bus=None):
        super().__init__(
            name="SocialSentimentTrigger",
            config=config,
            memory_agent=memory_agent,
            message_bus=message_bus
        )
        
        # Initialize clients
        self.twitter = TwitterClient(config)
        self.reddit = RedditClient(config)
        
        # Configuration
        self.watchlist = config.get("watchlist", [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
            "META", "TSLA", "GME", "AMC", "BB"
        ])
        
        self.min_mention_threshold = config.get("min_mention_threshold", 10)
        self.sentiment_threshold = config.get("sentiment_threshold", 0.6)
        self.lookback_minutes = config.get("lookback_minutes", 30)
        
        # Track mention counts
        self.mention_counts = {symbol: 0 for symbol in self.watchlist}
        self.last_reset = datetime.utcnow()
        
        logger.info("📱 SocialSentimentTrigger initialized")
    
    async def scan(self) -> List[TriggerEvent]:
        """
        Scan social media for mentions and sentiment
        """
        events = []
        
        # Reset counters if needed
        await self._reset_counters()
        
        try:
            # Get Twitter data
            twitter_data = await self.twitter.scan(self.watchlist, self.lookback_minutes)
            
            # Get Reddit data
            reddit_data = await self.reddit.scan(self.watchlist, self.lookback_minutes)
            
            # Combine and analyze
            combined = self._combine_sources(twitter_data, reddit_data)
            
            for symbol, data in combined.items():
                # Check if meets threshold
                if data["total_mentions"] >= self.min_mention_threshold:
                    # Update mention count
                    self.mention_counts[symbol] += data["total_mentions"]
                    
                    # Check if sentiment is significant
                    if abs(data["avg_sentiment"]) >= self.sentiment_threshold:
                        event = await self._create_event(symbol, data)
                        events.append(event)
                        
                        logger.info(f"📊 Social signal for {symbol}: "
                                  f"{data['total_mentions']} mentions, "
                                  f"sentiment: {data['avg_sentiment']:.2f}")
                
                await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error scanning social media: {e}")
        
        return events
    
    async def validate(self, event: TriggerEvent) -> bool:
        """Validate social sentiment events"""
        # Check confidence
        if event.confidence < self.config.min_confidence:
            return False
        
        # Check if we're getting too many mentions (possible spam/bot)
        mention_count = self.mention_counts.get(event.symbol, 0)
        if mention_count > 100:  # Too many mentions in short time
            logger.warning(f"Possible spam detected for {event.symbol}")
            return False
        
        return True
    
    def _combine_sources(self, twitter_data: Dict, reddit_data: Dict) -> Dict:
        """
        Combine data from multiple sources
        """
        combined = {}
        
        # All symbols
        all_symbols = set(twitter_data.keys()) | set(reddit_data.keys())
        
        for symbol in all_symbols:
            twitter = twitter_data.get(symbol, {"mentions": 0, "sentiment": 0})
            reddit = reddit_data.get(symbol, {"mentions": 0, "sentiment": 0})
            
            total_mentions = twitter["mentions"] + reddit["mentions"]
            
            if total_mentions > 0:
                # Weighted average sentiment
                weighted_sentiment = (
                    (twitter["mentions"] * twitter["sentiment"] +
                     reddit["mentions"] * reddit["sentiment"]) /
                    total_mentions
                )
            else:
                weighted_sentiment = 0
            
            combined[symbol] = {
                "total_mentions": total_mentions,
                "avg_sentiment": weighted_sentiment,
                "twitter_mentions": twitter["mentions"],
                "twitter_sentiment": twitter["sentiment"],
                "reddit_mentions": reddit["mentions"],
                "reddit_sentiment": reddit["sentiment"],
                "sources": []
            }
            
            if twitter["mentions"] > 0:
                combined[symbol]["sources"].append("twitter")
            if reddit["mentions"] > 0:
                combined[symbol]["sources"].append("reddit")
        
        return combined
    
    async def _reset_counters(self):
        """Reset mention counters periodically"""
        now = datetime.utcnow()
        if now - self.last_reset > timedelta(hours=1):
            self.mention_counts = {symbol: 0 for symbol in self.watchlist}
            self.last_reset = now
    
    async def _create_event(self, symbol: str, data: Dict) -> TriggerEvent:
        """Create a trigger event from social sentiment"""
        # Determine event type based on sentiment
        if data["avg_sentiment"] > 0:
            event_type = "SOCIAL_BUZZ_POSITIVE"
            confidence = min(1.0, data["avg_sentiment"] * 1.2)
        else:
            event_type = "SOCIAL_BUZZ_NEGATIVE"
            confidence = min(1.0, abs(data["avg_sentiment"]) * 1.2)
        
        return TriggerEvent(
            symbol=symbol,
            source_trigger=self.name,
            event_type=event_type,
            confidence=confidence,
            raw_data={
                "total_mentions": data["total_mentions"],
                "twitter_mentions": data["twitter_mentions"],
                "reddit_mentions": data["reddit_mentions"],
                "sources": data["sources"]
            },
            processed_data={
                "avg_sentiment": data["avg_sentiment"],
                "twitter_sentiment": data["twitter_sentiment"],
                "reddit_sentiment": data["reddit_sentiment"],
                "mention_velocity": data["total_mentions"] / self.lookback_minutes
            },
            timeframes_detected=["intraday"],
            primary_timeframe="intraday",
            market_regime=await self._get_market_regime(),
            correlation_id=f"social_{symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        )
    
    async def _get_market_regime(self) -> Optional[str]:
        """Get current market regime from memory"""
        if self.memory:
            return await self.memory.get("current_market_regime")
        return None