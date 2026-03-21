"""
Weighted Score Engine - Combines all analysis scores with dynamic weights
Adapts weights based on market regime and signal quality
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from enum import Enum

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.agents.base_agent import BaseAgent, AgentMessage

class ScoreCategory(Enum):
    """Categories of scores"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    TIMEFRAME = "timeframe"
    RISK = "risk"

class WeightedScoreEngine(BaseAgent):
    """
    Combines all analysis scores with dynamic weights
    
    Features:
    - Regime-based weight adjustment
    - Signal quality weighting
    - Historical performance tracking
    - Confidence scoring
    - Recommendation generation
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Combines all analysis scores with dynamic weights",
            config=config
        )
        
        # Base weights (can be overridden by config)
        self.base_weights = config.get("base_weights", {
            ScoreCategory.TECHNICAL: 0.35,
            ScoreCategory.FUNDAMENTAL: 0.25,
            ScoreCategory.SENTIMENT: 0.15,
            ScoreCategory.TIMEFRAME: 0.15,
            ScoreCategory.RISK: 0.10
        })
        
        # Regime-based adjustments
        self.regime_adjustments = config.get("regime_adjustments", {
            "strong_bull_trending": {
                ScoreCategory.TECHNICAL: +0.15,
                ScoreCategory.FUNDAMENTAL: -0.05,
                ScoreCategory.SENTIMENT: +0.05,
                ScoreCategory.TIMEFRAME: -0.05,
                ScoreCategory.RISK: -0.10
            },
            "bull_trending": {
                ScoreCategory.TECHNICAL: +0.10,
                ScoreCategory.FUNDAMENTAL: -0.05,
                ScoreCategory.SENTIMENT: +0.05,
                ScoreCategory.TIMEFRAME: -0.05,
                ScoreCategory.RISK: -0.05
            },
            "strong_bear_trending": {
                ScoreCategory.TECHNICAL: +0.05,
                ScoreCategory.FUNDAMENTAL: +0.15,
                ScoreCategory.SENTIMENT: -0.10,
                ScoreCategory.TIMEFRAME: -0.05,
                ScoreCategory.RISK: +0.05
            },
            "bear_trending": {
                ScoreCategory.TECHNICAL: +0.05,
                ScoreCategory.FUNDAMENTAL: +0.10,
                ScoreCategory.SENTIMENT: -0.10,
                ScoreCategory.TIMEFRAME: -0.05,
                ScoreCategory.RISK: +0.00
            },
            "high_volatility": {
                ScoreCategory.TECHNICAL: -0.10,
                ScoreCategory.FUNDAMENTAL: +0.00,
                ScoreCategory.SENTIMENT: +0.15,
                ScoreCategory.TIMEFRAME: -0.10,
                ScoreCategory.RISK: +0.15
            },
            "extreme_volatility_ranging": {
                ScoreCategory.TECHNICAL: -0.15,
                ScoreCategory.FUNDAMENTAL: +0.05,
                ScoreCategory.SENTIMENT: +0.20,
                ScoreCategory.TIMEFRAME: -0.15,
                ScoreCategory.RISK: +0.15
            },
            "ranging": {
                ScoreCategory.TECHNICAL: +0.15,
                ScoreCategory.FUNDAMENTAL: -0.10,
                ScoreCategory.SENTIMENT: -0.05,
                ScoreCategory.TIMEFRAME: +0.10,
                ScoreCategory.RISK: -0.10
            },
            "low_volatility_ranging": {
                ScoreCategory.TECHNICAL: +0.10,
                ScoreCategory.FUNDAMENTAL: +0.00,
                ScoreCategory.SENTIMENT: -0.05,
                ScoreCategory.TIMEFRAME: +0.05,
                ScoreCategory.RISK: -0.10
            },
            "bull_ranging": {
                ScoreCategory.TECHNICAL: +0.10,
                ScoreCategory.FUNDAMENTAL: -0.05,
                ScoreCategory.SENTIMENT: +0.00,
                ScoreCategory.TIMEFRAME: +0.05,
                ScoreCategory.RISK: -0.10
            },
            "bear_ranging": {
                ScoreCategory.TECHNICAL: +0.05,
                ScoreCategory.FUNDAMENTAL: +0.05,
                ScoreCategory.SENTIMENT: -0.05,
                ScoreCategory.TIMEFRAME: +0.00,
                ScoreCategory.RISK: +0.05
            },
            "transition": {
                ScoreCategory.TECHNICAL: -0.10,
                ScoreCategory.FUNDAMENTAL: +0.10,
                ScoreCategory.SENTIMENT: +0.10,
                ScoreCategory.TIMEFRAME: -0.10,
                ScoreCategory.RISK: +0.10
            },
            "panic": {
                ScoreCategory.TECHNICAL: -0.20,
                ScoreCategory.FUNDAMENTAL: +0.10,
                ScoreCategory.SENTIMENT: +0.25,
                ScoreCategory.TIMEFRAME: -0.20,
                ScoreCategory.RISK: +0.25
            }
        })
        
        # Quality thresholds
        self.min_quality_score = config.get("min_quality_score", 0.3)
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        
        # Historical performance tracking
        self.performance_history = []
        self.max_history = config.get("max_history", 1000)
        self.category_performance = {
            ScoreCategory.TECHNICAL: [],
            ScoreCategory.FUNDAMENTAL: [],
            ScoreCategory.SENTIMENT: [],
            ScoreCategory.TIMEFRAME: [],
            ScoreCategory.RISK: []
        }
        
        # Learning parameters
        self.learning_rate = config.get("learning_rate", 0.01)
        self.min_weight = config.get("min_weight", 0.05)
        self.max_weight = config.get("max_weight", 0.6)
        
        logging.info(f"✅ WeightedScoreEngine initialized")
        logging.info(f"   Base weights: {self._format_weights(self.base_weights)}")
    
    def _format_weights(self, weights: Dict) -> str:
        """Format weights for logging"""
        return {k.value if hasattr(k, 'value') else k: round(v, 3) for k, v in weights.items()}
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process score combination requests
        """
        if message.message_type == "combine_scores":
            analysis_id = message.content.get("analysis_id")
            scores = message.content.get("scores", {})
            regime = message.content.get("regime", "unknown")
            
            result = await self.combine_scores(analysis_id, scores, regime)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="combined_scores",
                content=result
            )
        
        elif message.message_type == "update_performance":
            # Update performance tracking with actual outcomes
            analysis_id = message.content.get("analysis_id")
            outcome = message.content.get("outcome")  # 'win' or 'loss'
            actual_return = message.content.get("return", 0)
            
            await self._update_performance(analysis_id, outcome, actual_return)
            
            # Optionally return updated weights
            updated_weights = self._learn_from_outcome(analysis_id, outcome, actual_return)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="weights_updated",
                content={"updated_weights": self._format_weights(updated_weights)}
            )
        
        elif message.message_type == "get_weights":
            # Return current weights
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="current_weights",
                content={
                    "base_weights": self._format_weights(self.base_weights),
                    "performance_stats": self.get_performance_stats()
                }
            )
        
        return None
    
    async def combine_scores(self, analysis_id: str, scores: Dict, 
                            regime: str) -> Dict[str, Any]:
        """
        Combine all scores into final recommendations
        """
        logging.info(f"🧮 Combining scores for analysis {analysis_id} under regime: {regime}")
        
        # Get weights for current regime
        weights = self._get_weights_for_regime(regime)
        
        # Extract individual scores and confidences
        technical_score, technical_conf = self._extract_score_and_conf(scores, "technical", 0.5, 0.7)
        fundamental_score, fundamental_conf = self._extract_score_and_conf(scores, "fundamental", 0.5, 0.7)
        sentiment_score, sentiment_conf = self._extract_score_and_conf(scores, "sentiment", 0.5, 0.6)
        timeframe_score, timeframe_conf = self._extract_score_and_conf(scores, "timeframe", 0.5, 0.7)
        risk_score, risk_conf = self._extract_score_and_conf(scores, "risk", 0.5, 0.8)
        
        # Calculate quality scores
        quality_scores = self._calculate_quality_scores(scores)
        
        # Adjust weights based on quality
        adjusted_weights = self._adjust_weights_by_quality(weights, quality_scores)
        
        # Calculate weighted total with confidence weighting
        weighted_total = 0
        total_conf_weight = 0
        
        score_map = {
            ScoreCategory.TECHNICAL: (technical_score, technical_conf),
            ScoreCategory.FUNDAMENTAL: (fundamental_score, fundamental_conf),
            ScoreCategory.SENTIMENT: (sentiment_score, sentiment_conf),
            ScoreCategory.TIMEFRAME: (timeframe_score, timeframe_conf),
            ScoreCategory.RISK: (risk_score, risk_conf)
        }
        
        for category, (score, conf) in score_map.items():
            weight = adjusted_weights.get(category, 0)
            weighted_total += score * weight * conf
            total_conf_weight += weight * conf
        
        # Normalize final score
        if total_conf_weight > 0:
            final_score = weighted_total / total_conf_weight
        else:
            final_score = 0.5
        
        final_score = max(0.0, min(1.0, final_score))
        
        # Calculate confidence
        confidence = self._calculate_confidence(scores, quality_scores, adjusted_weights)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(final_score, confidence, scores, regime)
        
        # Calculate risk-adjusted score
        risk_adjusted_score = self._risk_adjust_score(final_score, risk_score, confidence)
        
        # Get timeframe consensus if available
        timeframe_consensus = scores.get("timeframe", {}).get("consensus", {})
        timeframe_alignment = scores.get("timeframe", {}).get("alignment", 0.5)
        
        # Calculate composite quality
        composite_quality = float(np.mean(list(quality_scores.values()))) if quality_scores else 0.5
        
        # Determine if this is a high-conviction signal
        high_conviction = confidence > 0.7 and final_score > 0.65 and risk_score > 0.5
        
        result = {
            "analysis_id": analysis_id,
            "timestamp": datetime.now().isoformat(),
            "regime": regime,
            "individual_scores": {
                "technical": float(technical_score),
                "fundamental": float(fundamental_score),
                "sentiment": float(sentiment_score),
                "timeframe": float(timeframe_score),
                "risk": float(risk_score)
            },
            "individual_confidences": {
                "technical": float(technical_conf),
                "fundamental": float(fundamental_conf),
                "sentiment": float(sentiment_conf),
                "timeframe": float(timeframe_conf),
                "risk": float(risk_conf)
            },
            "weights_used": {k.value if hasattr(k, 'value') else k: round(v, 3) 
                            for k, v in adjusted_weights.items()},
            "quality_scores": {k: round(v, 3) for k, v in quality_scores.items()},
            "composite_quality": float(composite_quality),
            "final_score": float(final_score),
            "confidence": float(confidence),
            "risk_adjusted_score": float(risk_adjusted_score),
            "recommendation": recommendation,
            "timeframe_consensus": timeframe_consensus,
            "timeframe_alignment": float(timeframe_alignment),
            "high_conviction": high_conviction,
            "action": recommendation["action"],
            "action_strength": recommendation["strength"]
        }
        
        # Store for learning
        await self._store_for_learning(analysis_id, result)
        
        log_action = recommendation['action']
        log_score = f"{final_score:.2f}"
        log_conf = f"{confidence:.2f}"
        logging.info(f"✅ Score combination complete: {log_action} (score={log_score}, conf={log_conf})")
        
        return result
    
    def _extract_score_and_conf(self, scores: Dict, key: str, 
                               default_score: float, default_conf: float) -> tuple:
        """Extract score and confidence from scores dict"""
        data = scores.get(key, {})
        score = data.get("score", default_score)
        conf = data.get("confidence", default_conf)
        return float(score), float(conf)
    
    def _get_weights_for_regime(self, regime: str) -> Dict[ScoreCategory, float]:
        """Get base weights adjusted for market regime"""
        # Start with base weights
        weights = self.base_weights.copy()
        
        # Apply regime adjustments if available
        if regime in self.regime_adjustments:
            adjustments = self.regime_adjustments[regime]
            for category, adjustment in adjustments.items():
                if category in weights:
                    new_weight = weights[category] + adjustment
                    weights[category] = max(self.min_weight, min(self.max_weight, new_weight))
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _calculate_quality_scores(self, scores: Dict) -> Dict[str, float]:
        """Calculate quality scores for each category"""
        quality = {}
        
        for category, data in scores.items():
            if isinstance(data, dict):
                # Check data completeness
                details = data.get("details", {})
                if isinstance(details, dict):
                    # Count non-None values
                    non_none = sum(1 for v in details.values() if v is not None)
                    total = max(len(details), 1)
                    completeness = non_none / total
                else:
                    completeness = 0.7
                
                # Check recency (if timestamp available)
                recency = 1.0
                if "timestamp" in data:
                    try:
                        timestamp = datetime.fromisoformat(data["timestamp"])
                        age = datetime.now() - timestamp
                        # Decay over 1 hour (3600 seconds)
                        recency = max(0.3, 1 - (age.total_seconds() / 3600))
                    except:
                        recency = 0.8
                
                # Check confidence if provided
                confidence = data.get("confidence", 0.5)
                
                # Check data source diversity
                if "details" in data and isinstance(data["details"], dict):
                    # Look for source indicators in keys
                    source_keys = [k for k in data["details"].keys() 
                                 if "source" in k.lower() or "provider" in k.lower()]
                    source_count = len(source_keys)
                    diversity = min(1.0, source_count / 3)
                else:
                    diversity = 0.5
                
                # Combined quality score
                quality[category] = (
                    completeness * 0.35 + 
                    recency * 0.25 + 
                    confidence * 0.25 +
                    diversity * 0.15
                )
        
        return quality
    
    def _adjust_weights_by_quality(self, weights: Dict[ScoreCategory, float],
                                   quality_scores: Dict[str, float]) -> Dict[ScoreCategory, float]:
        """Adjust weights based on signal quality"""
        adjusted = weights.copy()
        
        for category, weight in weights.items():
            category_str = category.value
            if category_str in quality_scores:
                quality = quality_scores[category_str]
                
                # Adjust weight based on quality
                if quality < self.min_quality_score:
                    adjusted[category] = weight * 0.3
                elif quality < 0.5:
                    adjusted[category] = weight * 0.7
                elif quality > 0.8:
                    adjusted[category] = weight * 1.2  # Bonus for high quality
                
                # Apply min/max constraints
                adjusted[category] = max(self.min_weight, min(self.max_weight, adjusted[category]))
        
        # Renormalize
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}
        
        return adjusted
    
    def _calculate_confidence(self, scores: Dict, quality_scores: Dict,
                             weights: Dict[ScoreCategory, float]) -> float:
        """Calculate overall confidence in the result"""
        # Average quality score
        if quality_scores:
            avg_quality = sum(quality_scores.values()) / len(quality_scores)
        else:
            avg_quality = 0.5
        
        # Signal agreement (how well scores align)
        values = []
        for cat in ScoreCategory:
            cat_value = scores.get(cat.value, {}).get("score")
            if cat_value is not None:
                values.append(cat_value)
        
        if len(values) > 1:
            std = np.std(values)
            agreement = 1 - min(1.0, std * 1.5)  # Lower std = higher agreement
        else:
            agreement = 0.7
        
        # Data completeness
        categories_present = len([s for s in scores.values() if s.get("score") is not None])
        completeness = categories_present / len(ScoreCategory)
        
        # Weight distribution (more balanced = higher confidence)
        weight_values = list(weights.values())
        if weight_values:
            # Calculate entropy-based balance
            weight_entropy = -sum(w * np.log(w) for w in weight_values if w > 0)
            max_entropy = -np.log(1/len(weights))
            balance = weight_entropy / max_entropy if max_entropy > 0 else 0.5
        else:
            balance = 0.5
        
        # Combined confidence
        confidence = (
            avg_quality * 0.35 +
            agreement * 0.25 +
            completeness * 0.20 +
            balance * 0.20
        )
        
        return float(min(1.0, max(0.0, confidence)))
    
    def _generate_recommendation(self, score: float, confidence: float,
                                scores: Dict, regime: str) -> Dict[str, Any]:
        """Generate trading recommendation based on scores"""
        # Combine score and confidence
        effective_score = score * (0.6 + confidence * 0.4)
        
        # Apply regime-based multipliers
        regime_multipliers = {
            "strong_bull_trending": 1.15,
            "bull_trending": 1.10,
            "strong_bear_trending": 0.85,
            "bear_trending": 0.90,
            "panic": 0.60,
            "high_volatility": 0.80,
            "extreme_volatility_ranging": 0.75,
            "low_volatility_ranging": 1.10,
            "transition": 0.70
        }
        
        multiplier = regime_multipliers.get(regime, 1.0)
        adjusted_score = effective_score * multiplier
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        # Determine action
        if adjusted_score >= 0.8:
            action = "STRONG_BUY"
            strength = 4
            color = "dark_green"
        elif adjusted_score >= 0.65:
            action = "BUY"
            strength = 3
            color = "green"
        elif adjusted_score >= 0.5:
            action = "WATCH"
            strength = 2
            color = "yellow"
        elif adjusted_score >= 0.35:
            action = "SELL"
            strength = 1
            color = "orange"
        else:
            action = "STRONG_SELL"
            strength = 0
            color = "red"
        
        # Get reasoning from individual scores
        reasons = []
        concerns = []
        
        for category, data in scores.items():
            if isinstance(data, dict):
                cat_score = data.get("score", 0.5)
                cat_conf = data.get("confidence", 0.5)
                
                if cat_score > 0.7 and cat_conf > 0.6:
                    reasons.append(f"Strong {category} signal ({cat_score:.2f})")
                elif cat_score < 0.3 and cat_conf > 0.6:
                    concerns.append(f"Weak {category} signal ({cat_score:.2f})")
        
        # Add timeframe alignment if available
        timeframe_data = scores.get("timeframe", {})
        alignment = timeframe_data.get("alignment", 0.5)
        if alignment > 0.7:
            reasons.append(f"Strong timeframe alignment ({alignment:.2f})")
        elif alignment < 0.3:
            concerns.append(f"Poor timeframe alignment ({alignment:.2f})")
        
        # Add quality insights
        quality_scores = self._calculate_quality_scores(scores)
        if quality_scores:
            avg_quality = np.mean(list(quality_scores.values()))
            if avg_quality > 0.8:
                reasons.append("High quality data across all sources")
            elif avg_quality < 0.4:
                concerns.append("Low quality data - proceed with caution")
        
        # Add regime context
        if regime:
            reasons.append(f"Market regime: {regime}")
        
        return {
            "action": action,
            "strength": strength,
            "color": color,
            "effective_score": float(effective_score),
            "adjusted_score": float(adjusted_score),
            "reasons": reasons[:5],  # Top 5 reasons
            "concerns": concerns[:5],  # Top 5 concerns
            "score_breakdown": {k: v.get("score") for k, v in scores.items() 
                               if isinstance(v, dict)}
        }
    
    def _risk_adjust_score(self, score: float, risk_score: float, 
                          confidence: float) -> float:
        """Adjust score based on risk"""
        # Higher risk = lower effective score
        risk_penalty = (1 - risk_score) * 0.4
        
        # Lower confidence = higher penalty
        confidence_penalty = (1 - confidence) * 0.2
        
        adjusted = score * (1 - risk_penalty - confidence_penalty)
        
        return float(max(0.0, min(1.0, adjusted)))
    
    async def _store_for_learning(self, analysis_id: str, result: Dict):
        """Store results for future learning"""
        # Store in memory for performance tracking
        self.performance_history.append({
            "analysis_id": analysis_id,
            "timestamp": datetime.now(),
            "result": result,
            "outcome": None,  # Will be updated when trade completes
            "return": None
        })
        
        # Keep only last N records
        if len(self.performance_history) > self.max_history:
            self.performance_history = self.performance_history[-self.max_history:]
    
    async def _update_performance(self, analysis_id: str, outcome: str, actual_return: float):
        """Update performance tracking with actual outcome"""
        for record in self.performance_history:
            if record["analysis_id"] == analysis_id:
                record["outcome"] = outcome
                record["return"] = actual_return
                
                result = record["result"]
                
                # Update category performance based on outcome
                for category_str, score in result.get("individual_scores", {}).items():
                    # Convert string to enum if needed
                    category = None
                    for cat in ScoreCategory:
                        if cat.value == category_str:
                            category = cat
                            break
                    
                    if category:
                        if outcome == "win":
                            # If score was high and we won, that's good
                            self.category_performance[category].append(1 if score > 0.6 else 0)
                        else:
                            # If score was high and we lost, that's bad
                            self.category_performance[category].append(1 if score < 0.4 else 0)
                
                # Keep rolling window
                for category in self.category_performance:
                    if len(self.category_performance[category]) > 100:
                        self.category_performance[category] = self.category_performance[category][-100:]
                
                logging.info(f"📊 Updated performance for {analysis_id}: {outcome} ({actual_return:+.2f}%)")
                break
    
    def _learn_from_outcome(self, analysis_id: str, outcome: str, 
                           actual_return: float) -> Dict[ScoreCategory, float]:
        """
        Learn from trade outcome and adjust base weights
        Simple reinforcement learning approach
        """
        # Find the record
        record = None
        for r in self.performance_history:
            if r["analysis_id"] == analysis_id:
                record = r
                break
        
        if not record:
            return self.base_weights
        
        result = record["result"]
        predicted_action = result.get("action", "WATCH")
        confidence = result.get("confidence", 0.5)
        
        # Determine if prediction was correct
        correct = False
        if outcome == "win" and predicted_action in ["STRONG_BUY", "BUY"]:
            correct = True
        elif outcome == "loss" and predicted_action in ["STRONG_SELL", "SELL"]:
            correct = True
        elif outcome == "neutral" and predicted_action == "WATCH":
            correct = True
        
        # Adjust weights based on correctness
        if correct:
            # Reward categories that had high scores
            for category, score in result.get("individual_scores", {}).items():
                cat_enum = None
                for cat in ScoreCategory:
                    if cat.value == category:
                        cat_enum = cat
                        break
                
                if cat_enum and cat_enum in self.base_weights:
                    if score > 0.6:
                        # Increase weight for categories that were right
                        self.base_weights[cat_enum] += self.learning_rate * confidence
                    elif score < 0.4:
                        # Slightly decrease weight for categories that were wrong but we still won
                        self.base_weights[cat_enum] -= self.learning_rate * 0.5 * confidence
        else:
            # Penalize categories that had high scores but were wrong
            for category, score in result.get("individual_scores", {}).items():
                cat_enum = None
                for cat in ScoreCategory:
                    if cat.value == category:
                        cat_enum = cat
                        break
                
                if cat_enum and cat_enum in self.base_weights:
                    if score > 0.6:
                        # Decrease weight for categories that were wrong
                        self.base_weights[cat_enum] -= self.learning_rate * confidence
                    elif score < 0.4:
                        # Increase weight for categories that correctly predicted opposite
                        self.base_weights[cat_enum] += self.learning_rate * 0.5 * confidence
        
        # Apply min/max constraints
        for cat in self.base_weights:
            self.base_weights[cat] = max(self.min_weight, min(self.max_weight, self.base_weights[cat]))
        
        # Renormalize
        total = sum(self.base_weights.values())
        if total > 0:
            self.base_weights = {k: v / total for k, v in self.base_weights.items()}
        
        logging.info(f"📈 Updated base weights based on outcome: {self._format_weights(self.base_weights)}")
        
        return self.base_weights
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for learning"""
        if not self.performance_history:
            return {}
        
        # Filter records with outcomes
        completed = [r for r in self.performance_history if r["outcome"] is not None]
        
        if not completed:
            return {"message": "No completed trades yet"}
        
        # Calculate win rate by score range
        wins_by_score = {
            "very_high": {"count": 0, "wins": 0},
            "high": {"count": 0, "wins": 0},
            "medium": {"count": 0, "wins": 0},
            "low": {"count": 0, "wins": 0}
        }
        
        total_wins = 0
        total_losses = 0
        total_return = 0
        
        for record in completed:
            score = record["result"].get("final_score", 0.5)
            outcome = record["outcome"]
            ret = record.get("return", 0)
            
            total_return += ret
            
            if outcome == "win":
                total_wins += 1
            elif outcome == "loss":
                total_losses += 1
            
            # Categorize by score
            if score >= 0.8:
                category = "very_high"
            elif score >= 0.65:
                category = "high"
            elif score >= 0.5:
                category = "medium"
            else:
                category = "low"
            
            wins_by_score[category]["count"] += 1
            if outcome == "win":
                wins_by_score[category]["wins"] += 1
        
        # Calculate win rates
        for category in wins_by_score:
            if wins_by_score[category]["count"] > 0:
                wins_by_score[category]["win_rate"] = (
                    wins_by_score[category]["wins"] / wins_by_score[category]["count"]
                )
            else:
                wins_by_score[category]["win_rate"] = 0
        
        # Calculate category effectiveness
        category_effectiveness = {}
        for category, performances in self.category_performance.items():
            if performances:
                category_effectiveness[category.value] = {
                    "accuracy": float(np.mean(performances)),
                    "samples": len(performances)
                }
        
        total_trades = total_wins + total_losses
        win_rate = total_wins / total_trades if total_trades > 0 else 0
        avg_return = total_return / total_trades if total_trades > 0 else 0
        
        return {
            "total_trades": total_trades,
            "wins": total_wins,
            "losses": total_losses,
            "win_rate": float(win_rate),
            "avg_return": float(avg_return),
            "wins_by_score": wins_by_score,
            "category_effectiveness": category_effectiveness,
            "current_weights": self._format_weights(self.base_weights)
        }
    
    def get_current_weights(self) -> Dict[str, float]:
        """Get current base weights"""
        return self._format_weights(self.base_weights)
    
    def reset_weights(self):
        """Reset weights to default values"""
        self.base_weights = {
            ScoreCategory.TECHNICAL: 0.35,
            ScoreCategory.FUNDAMENTAL: 0.25,
            ScoreCategory.SENTIMENT: 0.15,
            ScoreCategory.TIMEFRAME: 0.15,
            ScoreCategory.RISK: 0.10
        }
        logging.info("🔄 Reset weights to default values")