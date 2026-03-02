"""
Sentiment Scorer - Final scoring and normalization for sentiment analysis
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from utils.logger import logging

class SentimentScorer:
    """
    Final scoring engine for sentiment analysis
    Normalizes scores, applies weights, and generates final sentiment
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Score thresholds
        self.thresholds = config.get("thresholds", {
            "very_positive": 0.8,
            "positive": 0.65,
            "neutral_high": 0.55,
            "neutral_low": 0.45,
            "negative": 0.35,
            "very_negative": 0.2
        })
        
        # Weights for different aspects
        self.aspect_weights = config.get("aspect_weights", {
            "magnitude": 0.3,      # How strong is the sentiment
            "confidence": 0.3,      # How confident are we
            "agreement": 0.2,       # How much sources agree
            "volume": 0.1,          # How much chatter
            "trend": 0.1             # Direction of sentiment
        })
        
        logging.info(f"✅ SentimentScorer initialized")
    
    def score(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate final sentiment score from all sources
        """
        # Extract components
        raw_score = sentiment_data.get("score", 0.5)
        confidence = sentiment_data.get("confidence", 0.5)
        source_breakdown = sentiment_data.get("source_breakdown", {})
        
        # Calculate metrics
        magnitude = self._calculate_magnitude(raw_score, confidence)
        agreement = self._calculate_agreement(source_breakdown)
        volume = self._calculate_volume(source_breakdown)
        trend = sentiment_data.get("trend_strength", 0.5)
        
        # Calculate weighted score
        final_score = (
            magnitude * self.aspect_weights["magnitude"] +
            confidence * self.aspect_weights["confidence"] +
            agreement * self.aspect_weights["agreement"] +
            volume * self.aspect_weights["volume"] +
            trend * self.aspect_weights["trend"]
        )
        
        # Determine label
        label = self._get_label(final_score)
        
        # Calculate strength (0-4 scale)
        strength = self._get_strength(final_score)
        
        return {
            "score": float(final_score),
            "label": label,
            "strength": strength,
            "magnitude": float(magnitude),
            "agreement": float(agreement),
            "volume_score": float(volume),
            "trend_score": float(trend),
            "confidence": float(confidence),
            "component_breakdown": {
                "raw_score": raw_score,
                "magnitude": magnitude,
                "agreement": agreement,
                "volume": volume,
                "trend": trend
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_magnitude(self, score: float, confidence: float) -> float:
        """Calculate sentiment magnitude (how extreme it is)"""
        # Distance from neutral (0.5)
        distance = abs(score - 0.5) * 2  # Scale to 0-1
        
        # Adjust by confidence
        magnitude = distance * confidence
        
        return float(min(1.0, magnitude))
    
    def _calculate_agreement(self, source_breakdown: Dict) -> float:
        """Calculate how much sources agree"""
        if not source_breakdown:
            return 0.5
        
        scores = []
        for source, data in source_breakdown.items():
            if isinstance(data, dict) and "score" in data:
                scores.append(data["score"])
        
        if len(scores) < 2:
            return 0.6  # Default when few sources
        
        # Lower standard deviation = higher agreement
        std = np.std(scores)
        agreement = 1 - min(1.0, std * 2)
        
        return float(agreement)
    
    def _calculate_volume(self, source_breakdown: Dict) -> float:
        """Calculate volume/activity score"""
        if not source_breakdown:
            return 0.5
        
        # Count active sources
        active_sources = len(source_breakdown)
        
        # Ideal number of sources (configurable)
        ideal_sources = self.config.get("ideal_sources", 5)
        
        volume = min(1.0, active_sources / ideal_sources)
        
        return float(volume)
    
    def _get_label(self, score: float) -> str:
        """Get sentiment label based on score"""
        if score >= self.thresholds["very_positive"]:
            return "very_positive"
        elif score >= self.thresholds["positive"]:
            return "positive"
        elif score >= self.thresholds["neutral_high"]:
            return "neutral_positive"
        elif score >= self.thresholds["neutral_low"]:
            return "neutral"
        elif score >= self.thresholds["negative"]:
            return "negative"
        else:
            return "very_negative"
    
    def _get_strength(self, score: float) -> int:
        """Get sentiment strength on 0-4 scale"""
        if score >= 0.8:
            return 4  # Very strong
        elif score >= 0.65:
            return 3  # Strong
        elif score >= 0.55:
            return 2  # Moderate
        elif score >= 0.45:
            return 1  # Weak
        else:
            return 0  # Very weak
    
    def normalize_score(self, score: float, min_val: float = 0, 
                       max_val: float = 1) -> float:
        """Normalize score to 0-1 range"""
        return float((score - min_val) / (max_val - min_val))
    
    def combine_scores(self, scores: List[float], weights: List[float] = None) -> float:
        """Combine multiple scores with optional weights"""
        if not scores:
            return 0.5
        
        if weights is None:
            weights = [1.0] * len(scores)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        
        return float(weighted_sum)
    
    def get_sentiment_trend(self, historical_scores: List[float]) -> Dict[str, Any]:
        """Analyze sentiment trend over time"""
        if len(historical_scores) < 2:
            return {"direction": "stable", "strength": 0.5}
        
        # Calculate slope
        x = np.arange(len(historical_scores))
        slope = np.polyfit(x, historical_scores, 1)[0]
        
        # Determine direction
        if slope > 0.05:
            direction = "improving"
        elif slope < -0.05:
            direction = "deteriorating"
        else:
            direction = "stable"
        
        # Calculate strength (normalized slope)
        strength = min(1.0, abs(slope) * 10)
        
        return {
            "direction": direction,
            "strength": float(strength),
            "slope": float(slope),
            "start_score": historical_scores[0],
            "end_score": historical_scores[-1]
        }