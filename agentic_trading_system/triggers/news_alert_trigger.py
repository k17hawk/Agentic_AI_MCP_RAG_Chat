"""
News Alert Trigger - Detects market-moving news events
"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
from logger import logging as logger
import json

from triggers.base_trigger import BaseTrigger, TriggerEvent
from triggers.news_alert_triggers.news_api_client import NewsAPIClient
from triggers.news_alert_triggers.sentiment_scorer import SentimentScorer

class NewsAlertTrigger(BaseTrigger):
    """
    Monitors news sources for market-moving events
    - Fetches news from multiple APIs
    - Performs sentiment analysis
    - Extracts ticker mentions
    - Scores impact potential
    """
    
    def __init__(self, config: dict, memory_agent=None, message_bus=None):
        super().__init__(
            name="NewsAlertTrigger",
            config=config,
            memory_agent=memory_agent,
            message_bus=message_bus
        )
        
        # Initialize components
        self.news_client = NewsAPIClient(config)
        self.sentiment_scorer = SentimentScorer(config)
        
        # Configuration
        self.impact_keywords = config.get("impact_keywords", [
            "earnings", "beat", "miss", "upgrade", "downgrade",
            "acquisition", "merger", "fda approval", "breakthrough",
            "lawsuit", "investigation", "ceo", "resign", "bankruptcy",
            "stock split", "dividend", "buyback"
        ])
        
        self.min_sentiment_score = config.get("min_sentiment_score", 0.6)
        self.max_articles_per_run = config.get("max_articles_per_run", 50)
        self.lookback_minutes = config.get("lookback_minutes", 15)
        
        # Cache for processed articles
        self.processed_articles = set()
        
        logger.info("📰 NewsAlertTrigger initialized")
    
    async def scan(self) -> List[TriggerEvent]:
        """
        Scan news sources for new articles
        """
        events = []
        
        try:
            # Fetch recent news
            articles = await self.news_client.get_recent_news(
                lookback_minutes=self.lookback_minutes,
                limit=self.max_articles_per_run
            )
            
            for article in articles:
                # Skip already processed
                if article["id"] in self.processed_articles:
                    continue
                
                # Analyze article
                analysis = await self._analyze_article(article)
                
                if analysis and analysis["trigger"]:
                    # Create events for each mentioned ticker
                    for ticker in analysis["tickers"]:
                        event = await self._create_event(ticker, article, analysis)
                        events.append(event)
                    
                    # Mark as processed
                    self.processed_articles.add(article["id"])
                
                await asyncio.sleep(0.1)  # Rate limiting
            
            # Clean old processed IDs
            self._clean_processed_ids()
            
        except Exception as e:
            logger.error(f"Error scanning news: {e}")
        
        return events
    
    async def validate(self, event: TriggerEvent) -> bool:
        """Validate news alert events"""
        # Check confidence
        if event.confidence < self.config.min_confidence:
            return False
        
        # Check if still relevant (not too old)
        event_age = datetime.utcnow() - event.timestamp
        if event_age > timedelta(minutes=30):
            return False
        
        return True
    
    async def _analyze_article(self, article: Dict) -> Optional[Dict]:
        """
        Analyze a single article for trading relevance
        """
        title = article.get("title", "")
        description = article.get("description", "")
        content = article.get("content", "")
        
        # Combine text for analysis
        full_text = f"{title} {description} {content}"
        
        # Extract tickers
        tickers = self._extract_tickers(full_text)
        
        if not tickers:
            return None
        
        # Sentiment analysis
        sentiment = await self.sentiment_scorer.analyze(full_text)
        
        # Check for impact keywords
        impact_score = self._calculate_impact_score(full_text)
        
        # Determine if this is market-moving
        trigger = (
            sentiment["score"] >= self.min_sentiment_score or
            impact_score >= 0.7
        )
        
        if trigger:
            return {
                "tickers": tickers,
                "sentiment": sentiment,
                "impact_score": impact_score,
                "title": title,
                "source": article.get("source", "unknown"),
                "url": article.get("url", ""),
                "published_at": article.get("publishedAt", datetime.utcnow().isoformat()),
                "trigger": True
            }
        
        return None
    
    def _extract_tickers(self, text: str) -> List[str]:
        """
        Extract stock tickers from text
        """
        # Simple regex for uppercase words (potential tickers)
        import re
        potential_tickers = re.findall(r'\b[A-Z]{2,5}\b', text)
        
        # Filter common false positives
        blacklist = {
            'THE', 'AND', 'FOR', 'WITH', 'FROM', 'THAT', 'THIS',
            'HAVE', 'WILL', 'YOUR', 'ABOUT', 'WOULD', 'COULD',
            'NYSE', 'NASDAQ', 'SEC', 'CEO', 'CFO', 'EPS', 'YOY'
        }
        
        tickers = [t for t in potential_tickers if t not in blacklist]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []
        for t in tickers:
            if t not in seen:
                seen.add(t)
                unique_tickers.append(t)
        
        return unique_tickers[:5]  # Limit to 5 tickers per article
    
    def _calculate_impact_score(self, text: str) -> float:
        """
        Calculate potential market impact score
        """
        text_lower = text.lower()
        score = 0.0
        matches = []
        
        # Check for impact keywords
        for keyword in self.impact_keywords:
            if keyword in text_lower:
                score += 0.2
                matches.append(keyword)
        
        # Cap at 1.0
        score = min(1.0, score)
        
        return score
    
    async def _create_event(self, ticker: str, article: Dict, analysis: Dict) -> TriggerEvent:
        """Create a trigger event from news"""
        return TriggerEvent(
            symbol=ticker,
            source_trigger=self.name,
            event_type="NEWS_IMPACT",
            confidence=analysis["sentiment"]["score"],
            raw_data={
                "title": analysis["title"],
                "source": analysis["source"],
                "url": analysis["url"],
                "published_at": analysis["published_at"]
            },
            processed_data={
                "sentiment": analysis["sentiment"],
                "impact_score": analysis["impact_score"],
                "keywords": analysis["sentiment"].get("keywords", [])
            },
            timeframes_detected=["intraday"],  # News is immediate impact
            primary_timeframe="intraday",
            market_regime=await self._get_market_regime(),
            correlation_id=f"news_{ticker}_{datetime.utcnow().strftime('%Y%m%d%H%M')}"
        )
    
    def _clean_processed_ids(self):
        """Remove old processed article IDs"""
        # Keep only last 1000
        if len(self.processed_articles) > 1000:
            self.processed_articles = set(list(self.processed_articles)[-1000:])
    
    async def _get_market_regime(self) -> Optional[str]:
        """Get current market regime from memory"""
        if self.memory:
            return await self.memory.get("current_market_regime")
        return None