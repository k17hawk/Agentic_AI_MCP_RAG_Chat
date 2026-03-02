"""
Timeframe Aggregator - Coordinates analysis across all timeframes
This is the CORE of your 60-day requirement!
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum

from utils.logger import logging
from analysis.technical.timeframe_analysis.intraday import IntradayAnalyzer
from analysis.technical.timeframe_analysis.daily import DailyAnalyzer
from analysis.technical.timeframe_analysis.weekly import WeeklyAnalyzer
from analysis.technical.timeframe_analysis.monthly import MonthlyAnalyzer

class Timeframe(Enum):
    """Available timeframes for analysis"""
    INTRADAY_5M = "5m"
    INTRADAY_15M = "15m"
    INTRADAY_1H = "1h"
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"

class TimeframeAggregator:
    """
    Aggregates technical analysis across multiple timeframes
    
    YOUR 60-DAY REQUIREMENT:
    - Daily timeframe (60 days) gets highest weight (40%)
    - Weekly and Monthly provide context
    - Intraday provides entry timing
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize timeframe analyzers
        self.intraday = IntradayAnalyzer(config.get("intraday_config", {}))
        self.daily = DailyAnalyzer(config.get("daily_config", {}))
        self.weekly = WeeklyAnalyzer(config.get("weekly_config", {}))
        self.monthly = MonthlyAnalyzer(config.get("monthly_config", {}))
        
        # Timeframe weights
        # YOUR 60-DAY WINDOW: Daily gets highest weight (40%)
        self.timeframe_weights = config.get("timeframe_weights", {
            Timeframe.INTRADAY_5M: 0.05,   # 5% - entry timing only
            Timeframe.INTRADAY_15M: 0.05,  # 5% - entry timing
            Timeframe.INTRADAY_1H: 0.10,    # 10% - intraday trend
            Timeframe.DAILY: 0.40,          # 40% - YOUR 60-DAY CORE!
            Timeframe.WEEKLY: 0.25,          # 25% - weekly context
            Timeframe.MONTHLY: 0.15          # 15% - monthly context
        })
        
        # Minimum timeframes required for analysis
        self.min_timeframes = config.get("min_timeframes", 3)
        
        # Alignment threshold
        self.alignment_threshold = config.get("alignment_threshold", 0.6)
        
        logging.info(f"✅ TimeframeAggregator initialized")
        logging.info(f"   Daily timeframe weight: {self.timeframe_weights[Timeframe.DAILY]:.2f} (60-day core)")
    
    def analyze_all(self, symbol: str, data_sets: Dict[Timeframe, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze all available timeframes and aggregate results
        """
        logging.info(f"🔄 Analyzing all timeframes for {symbol}")
        
        timeframe_results = {}
        available_timeframes = []
        
        # Analyze each timeframe
        for timeframe, data in data_sets.items():
            if data is not None and not data.empty:
                result = self._analyze_timeframe(timeframe, data)
                if result:
                    timeframe_results[timeframe.value] = result
                    available_timeframes.append(timeframe)
        
        if len(available_timeframes) < self.min_timeframes:
            return {
                "symbol": symbol,
                "error": f"Insufficient timeframes: {len(available_timeframes)} < {self.min_timeframes}",
                "available_timeframes": [tf.value for tf in available_timeframes]
            }
        
        # Calculate alignment between timeframes
        alignment = self._calculate_alignment(timeframe_results)
        
        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(timeframe_results, available_timeframes)
        
        # Determine primary timeframe (usually daily)
        primary_tf = self._determine_primary_timeframe(timeframe_results, available_timeframes)
        
        # Check for divergence between timeframes
        divergence = self._check_divergence(timeframe_results)
        
        # Generate consensus signal
        consensus = self._generate_consensus(timeframe_results, alignment, weighted_score)
        
        result = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "timeframes_analyzed": len(available_timeframes),
            "available_timeframes": [tf.value for tf in available_timeframes],
            "results": timeframe_results,
            "alignment": alignment,
            "weighted_score": weighted_score,
            "primary_timeframe": primary_tf,
            "divergence": divergence,
            "consensus": consensus,
            "confidence": self._calculate_confidence(alignment, weighted_score, divergence)
        }
        
        logging.info(f"✅ Timeframe aggregation complete: score={weighted_score:.2f}, alignment={alignment:.2f}")
        return result
    
    def _analyze_timeframe(self, timeframe: Timeframe, data: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze a single timeframe using the appropriate analyzer
        """
        try:
            if timeframe in [Timeframe.INTRADAY_5M, Timeframe.INTRADAY_15M, Timeframe.INTRADAY_1H]:
                return self.intraday.analyze(data, timeframe.value)
            elif timeframe == Timeframe.DAILY:
                return self.daily.analyze(data)
            elif timeframe == Timeframe.WEEKLY:
                return self.weekly.analyze(data)
            elif timeframe == Timeframe.MONTHLY:
                return self.monthly.analyze(data)
        except Exception as e:
            logging.error(f"Error analyzing {timeframe.value}: {e}")
        
        return None
    
    def _calculate_alignment(self, results: Dict) -> float:
        """
        Calculate how well timeframes align (0-1)
        Higher score means timeframes agree on trend direction
        """
        if not results:
            return 0.0
        
        # Count bullish vs bearish signals
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for tf, result in results.items():
            trend = result.get("trend", {}).get("direction", "neutral")
            if trend == "bullish":
                bullish_count += 1
            elif trend == "bearish":
                bearish_count += 1
            else:
                neutral_count += 1
        
        total = bullish_count + bearish_count + neutral_count
        if total == 0:
            return 0.5
        
        # Alignment is max agreement percentage
        alignment = max(bullish_count, bearish_count) / total
        
        return float(alignment)
    
    def _calculate_weighted_score(self, results: Dict, available_timeframes: List[Timeframe]) -> float:
        """
        Calculate weighted average score across timeframes
        """
        total_weight = 0
        weighted_sum = 0
        
        for timeframe in available_timeframes:
            tf_str = timeframe.value
            if tf_str in results:
                weight = self.timeframe_weights.get(timeframe, 0.1)
                score = results[tf_str].get("composite_score", 0.5)
                
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return float(weighted_sum / total_weight)
    
    def _determine_primary_timeframe(self, results: Dict, available_timeframes: List[Timeframe]) -> str:
        """
        Determine which timeframe is primary for analysis
        Daily is primary for 60-day analysis if available
        """
        # Daily is primary for 60-day analysis
        if Timeframe.DAILY.value in results:
            return Timeframe.DAILY.value
        
        # Otherwise use the highest weight timeframe available
        weighted_timeframes = [(tf, self.timeframe_weights.get(tf, 0)) 
                              for tf in available_timeframes]
        weighted_timeframes.sort(key=lambda x: x[1], reverse=True)
        
        if weighted_timeframes:
            return weighted_timeframes[0][0].value
        
        return "unknown"
    
    def _check_divergence(self, results: Dict) -> Dict[str, Any]:
        """
        Check for divergence between short-term and long-term timeframes
        """
        # Separate timeframes by term
        short_term = []
        medium_term = []
        long_term = []
        
        for tf_str, result in results.items():
            if tf_str in ["5m", "15m", "1h"]:
                short_term.append(result.get("composite_score", 0.5))
            elif tf_str in ["1d"]:
                medium_term.append(result.get("composite_score", 0.5))
            elif tf_str in ["1wk", "1mo"]:
                long_term.append(result.get("composite_score", 0.5))
        
        divergence = {
            "has_divergence": False,
            "short_vs_medium": 0.0,
            "medium_vs_long": 0.0,
            "type": "none"
        }
        
        # Check short vs medium
        if short_term and medium_term:
            avg_short = np.mean(short_term)
            avg_medium = np.mean(medium_term)
            diff = abs(avg_short - avg_medium)
            divergence["short_vs_medium"] = float(diff)
            
            if diff > 0.2:  # 20% difference is significant
                divergence["has_divergence"] = True
                if avg_short > avg_medium:
                    divergence["type"] = "short_term_bullish"
                else:
                    divergence["type"] = "short_term_bearish"
        
        # Check medium vs long
        if medium_term and long_term:
            avg_medium = np.mean(medium_term)
            avg_long = np.mean(long_term)
            diff = abs(avg_medium - avg_long)
            divergence["medium_vs_long"] = float(diff)
        
        return divergence
    
    def _generate_consensus(self, results: Dict, alignment: float, weighted_score: float) -> Dict[str, Any]:
        """
        Generate consensus signal based on all timeframes
        """
        consensus = {
            "signal": "neutral",
            "strength": 0.5,
            "reason": ""
        }
        
        # Determine overall trend direction
        bullish_count = 0
        bearish_count = 0
        
        for tf, result in results.items():
            trend = result.get("trend", {}).get("direction", "neutral")
            if trend == "bullish":
                bullish_count += 1
            elif trend == "bearish":
                bearish_count += 1
        
        total = bullish_count + bearish_count
        if total == 0:
            return consensus
        
        # Calculate signal based on alignment and weighted score
        if alignment > 0.7:  # Strong agreement
            if weighted_score > 0.7:
                consensus["signal"] = "strong_buy" if bullish_count > bearish_count else "strong_sell"
                consensus["strength"] = 0.9
                consensus["reason"] = "Strong alignment across timeframes"
            elif weighted_score > 0.6:
                consensus["signal"] = "buy" if bullish_count > bearish_count else "sell"
                consensus["strength"] = 0.7
                consensus["reason"] = "Good alignment across timeframes"
        
        elif alignment > 0.5:  # Moderate agreement
            if weighted_score > 0.65:
                consensus["signal"] = "buy" if bullish_count > bearish_count else "sell"
                consensus["strength"] = 0.6
                consensus["reason"] = "Moderate alignment with positive score"
            else:
                consensus["signal"] = "neutral"
                consensus["strength"] = 0.5
                consensus["reason"] = "Mixed signals across timeframes"
        
        else:  # Poor agreement
            consensus["signal"] = "neutral"
            consensus["strength"] = 0.4
            consensus["reason"] = "Timeframes showing conflicting signals"
        
        return consensus
    
    def _calculate_confidence(self, alignment: float, weighted_score: float, 
                             divergence: Dict) -> float:
        """
        Calculate overall confidence in the aggregation
        """
        confidence = (alignment * 0.4) + (weighted_score * 0.4)
        
        # Penalize divergence
        if divergence.get("has_divergence", False):
            confidence *= 0.8
        
        # Bonus for many timeframes
        confidence += 0.1  # Base bonus
        
        return float(min(1.0, confidence))
    
    def get_trend_alignment(self, results: Dict) -> Dict[str, Any]:
        """
        Get detailed trend alignment analysis
        """
        alignment = {
            "all_bullish": True,
            "all_bearish": True,
            "mixed": False,
            "primary_trend": "neutral",
            "secondary_trend": "neutral"
        }
        
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for tf, result in results.items():
            trend = result.get("trend", {}).get("direction", "neutral")
            if trend == "bullish":
                bullish_count += 1
                alignment["all_bearish"] = False
            elif trend == "bearish":
                bearish_count += 1
                alignment["all_bullish"] = False
            else:
                neutral_count += 1
                alignment["all_bullish"] = False
                alignment["all_bearish"] = False
        
        alignment["mixed"] = (bullish_count > 0 and bearish_count > 0)
        
        # Determine primary trend (most common)
        if bullish_count > bearish_count and bullish_count > neutral_count:
            alignment["primary_trend"] = "bullish"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            alignment["primary_trend"] = "bearish"
        else:
            alignment["primary_trend"] = "neutral"
        
        return alignment