"""
Sentiment Scorer - Analyzes sentiment of news articles
"""
from typing import Dict, List, Optional
import re
from loguru import logger
import numpy as np

class SentimentScorer:
    """
    Analyzes text sentiment for financial news
    Uses:
    - Financial-specific lexicon
    - Simple rule-based scoring
    - Can be extended with ML models
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Financial sentiment lexicons
        self.positive_words = {
            'beat', 'exceed', 'surge', 'soar', 'jump', 'gain', 'rise',
            'upgrade', 'bullish', 'outperform', 'growth', 'profit',
            'record', 'high', 'strong', 'positive', 'momentum',
            'breakthrough', 'approval', 'partnership', 'expansion',
            'dividend', 'buyback', 'acquisition', 'synergy'
        }
        
        self.negative_words = {
            'miss', 'fall', 'drop', 'decline', 'downgrade', 'bearish',
            'underperform', 'loss', 'weak', 'negative', 'risk',
            'lawsuit', 'investigation', 'probe', 'scandal', 'fraud',
            'bankruptcy', 'default', 'layoff', 'cut', 'warning',
            'downgrade', 'sell', 'pressure', 'volatile'
        }
        
        self.intensifiers = {
            'very', 'extremely', 'highly', 'significantly',
            'substantially', 'dramatically', 'sharply'
        }
        
        # Cache for results
        self.cache = {}
        
        logger.info("SentimentScorer initialized")
    
    async def analyze(self, text: str) -> Dict:
        """
        Analyze sentiment of text
        Returns sentiment score (-1 to 1) and metadata
        """
        # Check cache
        cache_key = hash(text[:100])  # Simple cache key
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Clean text
        text_clean = self._clean_text(text)
        words = text_clean.split()
        
        # Calculate scores
        positive_score = self._count_positive(words)
        negative_score = self._count_negative(words)
        
        # Adjust for intensifiers
        positive_score = self._apply_intensifiers(text_clean, positive_score)
        negative_score = self._apply_intensifiers(text_clean, negative_score)
        
        # Calculate net sentiment
        total_words = len(words)
        if total_words > 0:
            net_score = (positive_score - negative_score) / total_words
        else:
            net_score = 0
        
        # Normalize to -1 to 1
        sentiment_score = max(-1, min(1, net_score * 10))  # Scale factor
        
        # Determine label
        if sentiment_score > 0.2:
            label = "positive"
        elif sentiment_score < -0.2:
            label = "negative"
        else:
            label = "neutral"
        
        # Extract keywords
        keywords = self._extract_keywords(text_clean)
        
        result = {
            "score": float(sentiment_score),
            "label": label,
            "confidence": float(abs(sentiment_score)),  # Simple confidence
            "positive_count": positive_score,
            "negative_count": negative_score,
            "keywords": keywords,
            "magnitude": float(abs(sentiment_score))
        }
        
        # Cache result
        self.cache[cache_key] = result
        
        # Clean cache periodically
        if len(self.cache) > 1000:
            self.cache.clear()
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _count_positive(self, words: List[str]) -> int:
        """Count positive words"""
        return sum(1 for word in words if word in self.positive_words)
    
    def _count_negative(self, words: List[str]) -> int:
        """Count negative words"""
        return sum(1 for word in words if word in self.negative_words)
    
    def _apply_intensifiers(self, text: str, score: int) -> float:
        """Apply intensifier multipliers"""
        adjusted = float(score)
        
        for intensifier in self.intensifiers:
            if intensifier in text:
                # Find how many times it appears
                count = text.count(intensifier)
                adjusted += count * 0.5 * score
        
        return adjusted
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords"""
        words = text.split()
        
        # Find all sentiment words
        keywords = []
        for word in words:
            if word in self.positive_words or word in self.negative_words:
                if word not in keywords:
                    keywords.append(word)
        
        return keywords[:10]  # Limit to 10 keywords
    
    def compare_articles(self, articles: List[Dict]) -> Dict:
        """
        Compare sentiment across multiple articles
        """
        if not articles:
            return {"consensus": "neutral", "dispersion": 0}
        
        scores = [a.get("sentiment", {}).get("score", 0) for a in articles]
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        # Determine consensus
        if avg_score > 0.2:
            consensus = "positive"
        elif avg_score < -0.2:
            consensus = "negative"
        else:
            consensus = "neutral"
        
        return {
            "consensus": consensus,
            "average_score": float(avg_score),
            "dispersion": float(std_score),
            "article_count": len(articles),
            "agreement": 1 - std_score  # Lower std = higher agreement
        }
    
    def get_sentiment_trend(self, articles: List[Dict], window: int = 5) -> List[float]:
        """
        Get sentiment trend over time
        """
        if not articles:
            return []
        
        # Sort by time
        sorted_articles = sorted(
            articles, 
            key=lambda x: x.get("publishedAt", "")
        )
        
        # Extract scores
        scores = []
        for article in sorted_articles[-window:]:
            sentiment = article.get("sentiment", {})
            scores.append(sentiment.get("score", 0))
        
        return scores