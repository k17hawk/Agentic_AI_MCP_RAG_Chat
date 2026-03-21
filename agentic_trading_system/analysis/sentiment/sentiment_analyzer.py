"""
Sentiment Analyzer - Main orchestrator for sentiment analysis
Combines news, social media, analyst ratings, and insider activity
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import numpy as np

from agentic_trading_system.utils.logger import logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

# Import sentiment components
from analysis.sentiment.news_sentiment import NewsSentiment
from analysis.sentiment.social_sentiment import SocialSentiment
from analysis.sentiment.analyst_ratings import AnalystRatings
from analysis.sentiment.insider_activity import InsiderActivity
from analysis.sentiment.institutional_holdings import InstitutionalHoldings
from analysis.sentiment.sentiment_scorer import SentimentScorer

class SentimentAnalyzer(BaseAgent):
    """
    Main sentiment analysis agent
    Coordinates all sentiment sources and produces a unified sentiment score
    
    Sources:
    - News articles (weight: 35%)
    - Social media (weight: 20%)
    - Analyst ratings (weight: 25%)
    - Insider activity (weight: 10%)
    - Institutional holdings (weight: 10%)
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Comprehensive sentiment analysis from multiple sources",
            config=config
        )
        
        # Initialize all sentiment components
        self.news_sentiment = NewsSentiment(config.get("news_config", {}))
        self.social_sentiment = SocialSentiment(config.get("social_config", {}))
        self.analyst_ratings = AnalystRatings(config.get("analyst_config", {}))
        self.insider_activity = InsiderActivity(config.get("insider_config", {}))
        self.institutional_holdings = InstitutionalHoldings(config.get("institutional_config", {}))
        self.scorer = SentimentScorer(config.get("scorer_config", {}))
        
        # Source weights (configurable)
        self.source_weights = config.get("source_weights", {
            "news": 0.35,
            "social": 0.20,
            "analyst": 0.25,
            "insider": 0.10,
            "institutional": 0.10
        })
        
        # Minimum confidence thresholds
        self.min_source_confidence = config.get("min_source_confidence", 0.3)
        self.required_sources = config.get("required_sources", 2)
        
        # Cache for results
        self.cache = {}
        self.cache_ttl = timedelta(minutes=15)
        
        logging.info(f"✅ SentimentAnalyzer initialized with {len(self.source_weights)} sources")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process sentiment analysis requests
        """
        if message.message_type == "analysis_request":
            analysis_id = message.content.get("analysis_id")
            symbol = message.content.get("symbol")
            
            # Perform sentiment analysis
            score, details = await self.analyze(symbol)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="analysis_result",
                content={
                    "analysis_id": analysis_id,
                    "agent": self.name,
                    "score": score,
                    "details": details
                }
            )
        
        return None
    
    async def analyze(self, symbol: str) -> tuple[float, Dict]:
        """
        Perform comprehensive sentiment analysis for a symbol
        Returns score (0-1) and detailed breakdown
        """
        logging.info(f"🔍 Analyzing sentiment for {symbol}")
        
        # Check cache first
        cache_key = f"sentiment_{symbol}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached sentiment for {symbol}")
                return cached_result["score"], cached_result["details"]
        
        # Gather sentiment from all sources in parallel
        tasks = [
            self.news_sentiment.get_sentiment(symbol),
            self.social_sentiment.get_sentiment(symbol),
            self.analyst_ratings.get_sentiment(symbol),
            self.insider_activity.get_sentiment(symbol),
            self.institutional_holdings.get_sentiment(symbol)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        source_scores = {}
        source_confidences = {}
        source_details = {}
        
        source_names = ["news", "social", "analyst", "insider", "institutional"]
        
        for i, (source_name, result) in enumerate(zip(source_names, results)):
            if isinstance(result, Exception):
                logging.warning(f"⚠️ Error from {source_name} for {symbol}: {result}")
                continue
            
            if result and result.get("confidence", 0) >= self.min_source_confidence:
                source_scores[source_name] = result.get("score", 0.5)
                source_confidences[source_name] = result.get("confidence", 0.5)
                source_details[source_name] = result.get("details", {})
        
        # Check if we have enough sources
        if len(source_scores) < self.required_sources:
            logging.warning(f"⚠️ Insufficient sentiment sources for {symbol}: {len(source_scores)} < {self.required_sources}")
            return 0.5, {"error": "Insufficient sources", "sources_available": list(source_scores.keys())}
        
        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(source_scores, source_confidences)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(source_confidences, source_scores)
        
        # Detect sentiment trends
        trends = await self._detect_trends(symbol, source_scores)
        
        # Calculate momentum (how sentiment is changing)
        momentum = await self._calculate_momentum(symbol)
        
        # Prepare result
        result = {
            "score": weighted_score,
            "confidence": confidence,
            "trend": trends.get("direction", "neutral"),
            "trend_strength": trends.get("strength", 0.5),
            "momentum": momentum,
            "source_breakdown": {
                source: {
                    "score": source_scores.get(source),
                    "confidence": source_confidences.get(source),
                    "details": source_details.get(source, {})
                }
                for source in source_scores.keys()
            },
            "weights_used": {k: v for k, v in self.source_weights.items() 
                           if k in source_scores},
            "sources_available": list(source_scores.keys()),
            "timestamp": datetime.now().isoformat()
        }
        
        # Determine sentiment label
        if weighted_score >= 0.7:
            result["label"] = "very_positive"
        elif weighted_score >= 0.6:
            result["label"] = "positive"
        elif weighted_score >= 0.45:
            result["label"] = "neutral"
        elif weighted_score >= 0.3:
            result["label"] = "negative"
        else:
            result["label"] = "very_negative"
        
        # Cache result
        self.cache[cache_key] = (datetime.now(), result)
        
        logging.info(f"✅ Sentiment analysis complete for {symbol}: {result['label']} (score={weighted_score:.2f}, conf={confidence:.2f})")
        
        return weighted_score, result
    
    def _calculate_weighted_score(self, scores: Dict[str, float], 
                                  confidences: Dict[str, float]) -> float:
        """Calculate weighted score using source weights and confidences"""
        total_weight = 0
        weighted_sum = 0
        
        for source, score in scores.items():
            # Base weight from config
            base_weight = self.source_weights.get(source, 0.2)
            
            # Adjust weight by confidence
            confidence = confidences.get(source, 0.5)
            effective_weight = base_weight * confidence
            
            weighted_sum += score * effective_weight
            total_weight += effective_weight
        
        if total_weight == 0:
            return 0.5
        
        return float(weighted_sum / total_weight)
    
    def _calculate_confidence(self, confidences: Dict[str, float], 
                             scores: Dict[str, float]) -> float:
        """Calculate overall confidence in the sentiment score"""
        if not confidences:
            return 0.5
        
        # Average confidence
        avg_confidence = sum(confidences.values()) / len(confidences)
        
        # Agreement between sources (lower std = higher agreement)
        if len(scores) > 1:
            score_values = list(scores.values())
            std = np.std(score_values)
            agreement = 1 - min(1.0, std * 2)  # Normalize to 0-1
        else:
            agreement = 0.7  # Default when only one source
        
        # Number of sources factor
        source_factor = min(1.0, len(scores) / len(self.source_weights))
        
        # Combined confidence
        confidence = (
            avg_confidence * 0.5 +
            agreement * 0.3 +
            source_factor * 0.2
        )
        
        return float(min(1.0, max(0.0, confidence)))
    
    async def _detect_trends(self, symbol: str, current_scores: Dict) -> Dict[str, Any]:
        """Detect sentiment trends over time"""
        # In production, this would query historical data
        # For now, return neutral
        return {
            "direction": "neutral",
            "strength": 0.5,
            "description": "Insufficient historical data"
        }
    
    async def _calculate_momentum(self, symbol: str) -> float:
        """Calculate sentiment momentum (rate of change)"""
        # In production, this would compare with historical
        # For now, return neutral
        return 0.5
    
    def get_source_weights(self) -> Dict[str, float]:
        """Get current source weights"""
        return self.source_weights.copy()
    
    def update_source_weights(self, new_weights: Dict[str, float]):
        """Update source weights based on performance"""
        self.source_weights.update(new_weights)
        # Normalize to sum to 1
        total = sum(self.source_weights.values())
        if total > 0:
            self.source_weights = {k: v/total for k, v in self.source_weights.items()}
        logging.info(f"📊 Updated sentiment source weights: {self.source_weights}")