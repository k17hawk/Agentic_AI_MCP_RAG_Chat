"""
Technical Scorer - Scores and ranks technical analysis results
"""
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from agentic_trading_system.utils.logger import logger as logging

class TechnicalScorer:
    """
    Scores technical analysis results and generates final technical score
    
    Combines:
    - Trend indicators (40%)
    - Momentum indicators (25%)
    - Volume indicators (20%)
    - Volatility indicators (15%)
    - Pattern recognition (bonus/malus)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Weights for different components
        self.weights = config.get("technical_weights", {
            "trend": 0.40,        # Trend is most important
            "momentum": 0.25,       # Momentum second
            "volume": 0.20,         # Volume confirmation
            "volatility": 0.15,      # Volatility context
            "pattern_bonus": 0.10    # Pattern bonus (added separately)
        })
        
        # Scoring thresholds
        self.strong_buy_threshold = config.get("strong_buy_threshold", 0.8)
        self.buy_threshold = config.get("buy_threshold", 0.65)
        self.sell_threshold = config.get("sell_threshold", 0.35)
        self.strong_sell_threshold = config.get("strong_sell_threshold", 0.2)
        
        logging.info(f"✅ TechnicalScorer initialized")
    
    def calculate_score(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate overall technical score from all analysis components
        """
        # Extract component scores
        trend_score = self._score_trend(analysis_results.get("trend", {}))
        momentum_score = self._score_momentum(analysis_results.get("momentum", {}))
        volume_score = self._score_volume(analysis_results.get("volume", {}))
        volatility_score = self._score_volatility(analysis_results.get("volatility", {}))
        pattern_score = self._score_patterns(analysis_results.get("patterns", []))
        
        # Calculate weighted base score
        base_score = (
            trend_score * self.weights["trend"] +
            momentum_score * self.weights["momentum"] +
            volume_score * self.weights["volume"] +
            volatility_score * self.weights["volatility"]
        )
        
        # Add pattern bonus (capped at max weight)
        pattern_bonus = pattern_score * self.weights["pattern_bonus"]
        final_score = min(1.0, base_score + pattern_bonus)
        
        # Determine signal
        signal = self._determine_signal(final_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(analysis_results)
        
        # Generate signals list
        signals = self._generate_signals(analysis_results, final_score, signal)
        
        return {
            "score": float(final_score),
            "base_score": float(base_score),
            "pattern_bonus": float(pattern_bonus),
            "signal": signal,
            "confidence": float(confidence),
            "signals": signals,
            "component_scores": {
                "trend": float(trend_score),
                "momentum": float(momentum_score),
                "volume": float(volume_score),
                "volatility": float(volatility_score),
                "patterns": float(pattern_score)
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _score_trend(self, trend_data: Dict) -> float:
        """
        Score trend indicators (0-1)
        """
        score = 0.5  # Start neutral
        
        # Direction scoring
        direction = trend_data.get("direction", "neutral")
        if direction == "strong_bullish":
            score += 0.4
        elif direction == "bullish":
            score += 0.2
        elif direction == "strong_bearish":
            score -= 0.4
        elif direction == "bearish":
            score -= 0.2
        
        # Moving average alignment
        price_vs_ma50 = trend_data.get("price_vs_ma50", 0)
        if price_vs_ma50 > 5:
            score += 0.1
        elif price_vs_ma50 < -5:
            score -= 0.1
        
        # Golden/Death cross
        if trend_data.get("golden_cross"):
            score += 0.15
        if trend_data.get("death_cross"):
            score -= 0.15
        
        # Slope strength
        slope = trend_data.get("slope_60d_pct", 0)  # Your 60-day metric
        if slope > 10:
            score += 0.1
        elif slope < -10:
            score -= 0.1
        
        return float(max(0.0, min(1.0, score)))
    
    def _score_momentum(self, momentum_data: Dict) -> float:
        """
        Score momentum indicators (0-1)
        """
        score = 0.5
        
        # RSI
        rsi = momentum_data.get("rsi", 50)
        if rsi < 30:
            score += 0.3  # Oversold - potential buy
        elif rsi < 40:
            score += 0.2
        elif rsi < 45:
            score += 0.1
        elif rsi > 70:
            score -= 0.3  # Overbought - potential sell
        elif rsi > 60:
            score -= 0.2
        elif rsi > 55:
            score -= 0.1
        
        # MACD
        macd_cross = momentum_data.get("macd_cross", "neutral")
        if macd_cross == "bullish":
            score += 0.15
        elif macd_cross == "bearish":
            score -= 0.15
        
        # Rate of Change (your 60-day metric)
        roc_60 = momentum_data.get("roc_60", 0)
        if roc_60 > 15:
            score += 0.2
        elif roc_60 > 8:
            score += 0.1
        elif roc_60 < -15:
            score -= 0.2
        elif roc_60 < -8:
            score -= 0.1
        
        return float(max(0.0, min(1.0, score)))
    
    def _score_volume(self, volume_data: Dict) -> float:
        """
        Score volume indicators (0-1)
        """
        score = 0.5
        
        # Volume spikes
        if volume_data.get("volume_spike"):
            if volume_data.get("obv_trend") == "up":
                score += 0.25  # Bullish volume spike
            elif volume_data.get("obv_trend") == "down":
                score -= 0.25  # Bearish volume spike
            else:
                score += 0.1  # Neutral spike
        
        # Volume ratios
        vol_ratio = volume_data.get("vol_ratio_20", 1.0)
        if vol_ratio > 2.0:
            score += 0.15
        elif vol_ratio > 1.5:
            score += 0.1
        
        # OBV trend
        obv_slope = volume_data.get("obv_slope", 0)
        if obv_slope > 2:
            score += 0.1
        elif obv_slope < -2:
            score -= 0.1
        
        return float(max(0.0, min(1.0, score)))
    
    def _score_volatility(self, volatility_data: Dict) -> float:
        """
        Score volatility indicators (0-1)
        Lower volatility is better for entries
        """
        score = 0.5
        
        # Bollinger Band position
        bb_position = volatility_data.get("bb_position", 50)
        if bb_position < 20:
            score += 0.15  # Near lower band - potential bounce
        elif bb_position > 80:
            score -= 0.15  # Near upper band - potential pullback
        
        # ATR (lower is better for entries)
        atr_pct = volatility_data.get("atr_pct", 2.0)
        if atr_pct < 1.5:
            score += 0.1
        elif atr_pct > 4.0:
            score -= 0.1
        
        # Volatility percentile
        hist_vol = volatility_data.get("hist_vol_60", 20)  # Your 60-day metric
        if hist_vol < 15:
            score += 0.1
        elif hist_vol > 30:
            score -= 0.1
        
        return float(max(0.0, min(1.0, score)))
    
    def _score_patterns(self, patterns: List[Dict]) -> float:
        """
        Score pattern recognition results (returns bonus 0-1)
        """
        if not patterns:
            return 0.0
        
        bullish_score = 0.0
        bearish_score = 0.0
        
        for pattern in patterns:
            confidence = pattern.get("confidence", 0.5)
            direction = pattern.get("direction", "neutral")
            
            if direction == "bullish":
                bullish_score += confidence
            elif direction == "bearish":
                bearish_score += confidence
        
        # Calculate net pattern score (-1 to 1)
        total = bullish_score + bearish_score
        if total > 0:
            net_score = (bullish_score - bearish_score) / total
        else:
            net_score = 0.0
        
        # Convert to 0-1 scale where 0.5 is neutral
        pattern_score = (net_score + 1) / 2
        
        return float(pattern_score)
    
    def _determine_signal(self, score: float) -> str:
        """
        Determine trading signal based on score
        """
        if score >= self.strong_buy_threshold:
            return "STRONG_BUY"
        elif score >= self.buy_threshold:
            return "BUY"
        elif score <= self.strong_sell_threshold:
            return "STRONG_SELL"
        elif score <= self.sell_threshold:
            return "SELL"
        else:
            return "NEUTRAL"
    
    def _calculate_confidence(self, analysis_results: Dict) -> float:
        """
        Calculate confidence level in the score
        """
        factors = []
        
        # Trend strength
        trend = analysis_results.get("trend", {})
        if trend.get("strength") == "strong":
            factors.append(0.9)
        elif trend.get("strength") == "moderate":
            factors.append(0.7)
        else:
            factors.append(0.5)
        
        # Volume confirmation
        volume = analysis_results.get("volume", {})
        if volume.get("volume_spike"):
            factors.append(0.8)
        else:
            factors.append(0.6)
        
        # Pattern confirmation
        patterns = analysis_results.get("patterns", [])
        if patterns:
            avg_confidence = np.mean([p.get("confidence", 0.5) for p in patterns])
            factors.append(avg_confidence)
        
        # Data quality (placeholder - would come from data quality check)
        factors.append(0.8)
        
        return float(np.mean(factors))
    
    def _generate_signals(self, analysis_results: Dict, score: float, 
                         signal: str) -> List[Dict]:
        """
        Generate list of specific trading signals
        """
        signals = []
        
        # Trend signals
        trend = analysis_results.get("trend", {})
        if trend.get("golden_cross"):
            signals.append({
                "type": "trend",
                "signal": "buy",
                "strength": 0.9,
                "description": "Golden Cross detected"
            })
        if trend.get("death_cross"):
            signals.append({
                "type": "trend",
                "signal": "sell",
                "strength": 0.9,
                "description": "Death Cross detected"
            })
        
        # RSI signals
        momentum = analysis_results.get("momentum", {})
        rsi = momentum.get("rsi", 50)
        if rsi < 30:
            signals.append({
                "type": "momentum",
                "signal": "buy",
                "strength": 0.7,
                "description": f"RSI Oversold ({rsi:.1f})"
            })
        elif rsi > 70:
            signals.append({
                "type": "momentum",
                "signal": "sell",
                "strength": 0.7,
                "description": f"RSI Overbought ({rsi:.1f})"
            })
        
        # Volume signals
        volume = analysis_results.get("volume", {})
        if volume.get("volume_spike"):
            if volume.get("obv_trend") == "up":
                signals.append({
                    "type": "volume",
                    "signal": "buy",
                    "strength": 0.8,
                    "description": "Volume spike with bullish OBV"
                })
            elif volume.get("obv_trend") == "down":
                signals.append({
                    "type": "volume",
                    "signal": "sell",
                    "strength": 0.8,
                    "description": "Volume spike with bearish OBV"
                })
        
        # 60-day breakout signals (your requirement)
        roc_60 = momentum.get("roc_60", 0)
        if abs(roc_60) > 20:
            signals.append({
                "type": "breakout",
                "signal": "buy" if roc_60 > 0 else "sell",
                "strength": 0.75,
                "description": f"60-day movement: {roc_60:+.1f}%"
            })
        
        return signals