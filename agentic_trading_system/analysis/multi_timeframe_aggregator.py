"""
Multi-Timeframe Aggregator - Combines signals across multiple timeframes
This is CRITICAL for your 60-day analysis requirement!
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum

from utils.logger import  logger as logging
from agents.base_agent import BaseAgent, AgentMessage

class Timeframe(Enum):
    """Available timeframes for analysis"""
    INTRADAY_5M = "5m"
    INTRADAY_15M = "15m"
    INTRADAY_1H = "1h"
    DAILY = "1d"
    WEEKLY = "1wk"
    MONTHLY = "1mo"
    QUARTERLY = "3mo"
    YEARLY = "1y"

class TimeframeConfig:
    """Configuration for each timeframe"""
    WEIGHTS = {
        Timeframe.INTRADAY_5M: 0.05,   # 5% - very short term
        Timeframe.INTRADAY_15M: 0.10,  # 10% - short term
        Timeframe.INTRADAY_1H: 0.15,   # 15% - intraday
        Timeframe.DAILY: 0.30,          # 30% - YOUR 60-DAY WINDOW!
        Timeframe.WEEKLY: 0.20,          # 20% - weekly trend
        Timeframe.MONTHLY: 0.15,         # 15% - monthly trend
        Timeframe.QUARTERLY: 0.03,       # 3% - quarterly context
        Timeframe.YEARLY: 0.02            # 2% - yearly context
    }
    
    # Required data periods for each timeframe
    REQUIRED_PERIODS = {
        Timeframe.INTRADAY_5M: "5d",
        Timeframe.INTRADAY_15M: "10d",
        Timeframe.INTRADAY_1H: "1mo",
        Timeframe.DAILY: "6mo",      # 6 months to cover 60 days
        Timeframe.WEEKLY: "2y",
        Timeframe.MONTHLY: "5y",
        Timeframe.QUARTERLY: "10y",
        Timeframe.YEARLY: "20y"
    }

class MultiTimeframeAggregator(BaseAgent):
    """
    Aggregates signals across multiple timeframes
    Ensures that short-term momentum aligns with medium-term trend
    
    YOUR 60-DAY REQUIREMENT: Daily timeframe (30% weight) is the CORE!
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Aggregates signals across multiple timeframes",
            config=config
        )
        
        # Timeframe weights (can be overridden by config)
        self.weights = config.get("timeframe_weights", TimeframeConfig.WEIGHTS)
        
        # Minimum number of timeframes required
        self.min_timeframes = config.get("min_timeframes", 3)
        
        # Alignment threshold (how much timeframes must agree)
        self.alignment_threshold = config.get("alignment_threshold", 0.6)
        
        # Cache for timeframe data
        self.data_cache = {}
        
        logging.info(f"✅ MultiTimeframeAggregator initialized with {len(self.weights)} timeframes")
        logging.info(f"   Daily timeframe weight: {self.weights.get(Timeframe.DAILY, 0):.2f} (60-day core)")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process aggregation requests
        """
        if message.message_type == "aggregate_request":
            symbol = message.content.get("symbol")
            signals = message.content.get("signals", {})
            
            result = await self.aggregate(symbol, signals)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="aggregate_result",
                content=result
            )
        
        return None
    
    async def aggregate(self, symbol: str, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Aggregate signals across all available timeframes
        """
        logging.info(f"🔄 Aggregating timeframes for {symbol}")
        
        timeframe_results = {}
        available_timeframes = []
        
        # Analyze each timeframe
        for timeframe in Timeframe:
            try:
                result = await self._analyze_timeframe(symbol, timeframe)
                if result:
                    timeframe_results[timeframe.value] = result
                    available_timeframes.append(timeframe)
            except Exception as e:
                logging.debug(f"Could not analyze {timeframe.value} for {symbol}: {e}")
        
        if len(available_timeframes) < self.min_timeframes:
            return {
                "symbol": symbol,
                "error": f"Insufficient timeframes: {len(available_timeframes)} < {self.min_timeframes}",
                "available_timeframes": [tf.value for tf in available_timeframes]
            }
        
        # Calculate alignment
        alignment = self._calculate_alignment(timeframe_results)
        
        # Calculate weighted score
        weighted_score = self._calculate_weighted_score(timeframe_results, available_timeframes)
        
        # Determine primary timeframe (usually daily for 60-day analysis)
        primary_tf = self._determine_primary_timeframe(timeframe_results)
        
        # Check for divergence (short-term vs long-term)
        divergence = self._check_divergence(timeframe_results)
        
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
            "confidence": self._calculate_confidence(alignment, weighted_score, divergence)
        }
        
        logging.info(f"✅ Timeframe aggregation complete for {symbol}: score={weighted_score:.2f}, alignment={alignment:.2f}")
        return result
    
    async def _analyze_timeframe(self, symbol: str, timeframe: Timeframe) -> Optional[Dict]:
        """
        Analyze a single timeframe
        """
        # Get data for this timeframe
        data = await self._get_timeframe_data(symbol, timeframe)
        
        if data is None or len(data) < self._get_min_bars(timeframe):
            return None
        
        # Calculate basic metrics
        current_price = data['Close'].iloc[-1]
        
        # Moving averages
        ma20 = data['Close'].rolling(20).mean().iloc[-1] if len(data) >= 20 else None
        ma50 = data['Close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else None
        ma200 = data['Close'].rolling(200).mean().iloc[-1] if len(data) >= 200 else None
        
        # Trend direction
        trend = "bullish" if current_price > ma50 else "bearish" if current_price < ma50 else "neutral"
        
        # Trend strength (using slope)
        if len(data) >= 20:
            x = np.arange(20)
            y = data['Close'].iloc[-20:].values
            slope = np.polyfit(x, y, 1)[0]
            strength = min(1.0, abs(slope) / current_price * 100)
        else:
            strength = 0.5
        
        # Momentum (rate of change)
        roc_5 = self._calculate_roc(data, 5)
        roc_10 = self._calculate_roc(data, 10)
        roc_20 = self._calculate_roc(data, 20)
        
        # Volume confirmation
        volume_ratio = self._calculate_volume_ratio(data)
        
        # Volatility
        returns = data['Close'].pct_change().dropna()
        volatility = returns.std() * 100 if len(returns) > 0 else 0
        
        return {
            "timeframe": timeframe.value,
            "current_price": float(current_price),
            "trend": trend,
            "trend_strength": float(strength),
            "momentum": {
                "roc_5": float(roc_5) if roc_5 else None,
                "roc_10": float(roc_10) if roc_10 else None,
                "roc_20": float(roc_20) if roc_20 else None,
            },
            "moving_averages": {
                "ma20": float(ma20) if ma20 else None,
                "ma50": float(ma50) if ma50 else None,
                "ma200": float(ma200) if ma200 else None,
            },
            "volume_ratio": float(volume_ratio),
            "volatility": float(volatility),
            "data_points": len(data),
            "score": self._calculate_timeframe_score(trend, strength, roc_20, volume_ratio)
        }
    
    def _calculate_timeframe_score(self, trend: str, strength: float, 
                                   roc_20: Optional[float], volume_ratio: float) -> float:
        """Calculate score for a single timeframe (0-1)"""
        score = 0.5  # Start neutral
        
        # Trend contribution
        if trend == "bullish":
            score += 0.15
        elif trend == "bearish":
            score -= 0.15
        
        # Trend strength
        score += strength * 0.1
        
        # Momentum
        if roc_20 and roc_20 > 0:
            score += min(0.2, roc_20 / 10)
        elif roc_20 and roc_20 < 0:
            score -= min(0.2, abs(roc_20) / 10)
        
        # Volume confirmation
        if volume_ratio > 1.5:
            score += 0.1
        elif volume_ratio < 0.5:
            score -= 0.1
        
        return float(max(0.0, min(1.0, score)))
    
    def _calculate_alignment(self, results: Dict) -> float:
        """Calculate how well timeframes align (0-1)"""
        if not results:
            return 0.0
        
        # Count bullish vs bearish signals
        bullish_count = sum(1 for r in results.values() if r.get("trend") == "bullish")
        bearish_count = sum(1 for r in results.values() if r.get("trend") == "bearish")
        total = len(results)
        
        if total == 0:
            return 0.5
        
        # Alignment is max agreement percentage
        alignment = max(bullish_count, bearish_count) / total
        
        return float(alignment)
    
    def _calculate_weighted_score(self, results: Dict, available_timeframes: List[Timeframe]) -> float:
        """Calculate weighted score across timeframes"""
        total_weight = 0
        weighted_sum = 0
        
        for timeframe in available_timeframes:
            tf_str = timeframe.value
            if tf_str in results:
                weight = self.weights.get(timeframe, 0.1)
                score = results[tf_str].get("score", 0.5)
                
                weighted_sum += score * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return float(weighted_sum / total_weight)
    
    def _determine_primary_timeframe(self, results: Dict) -> str:
        """Determine which timeframe is primary (usually daily)"""
        # Daily is usually primary for 60-day analysis
        if Timeframe.DAILY.value in results:
            return Timeframe.DAILY.value
        
        # Fallback to highest weight available
        available = list(results.keys())
        if available:
            return available[0]
        
        return "unknown"
    
    def _check_divergence(self, results: Dict) -> Dict[str, Any]:
        """Check for divergence between short and long timeframes"""
        # Get short-term (intraday) and long-term (weekly+) signals
        short_term = []
        long_term = []
        
        for tf_str, result in results.items():
            if tf_str in ["5m", "15m", "1h"]:
                short_term.append(result.get("score", 0.5))
            elif tf_str in ["1wk", "1mo", "3mo", "1y"]:
                long_term.append(result.get("score", 0.5))
        
        if not short_term or not long_term:
            return {"diverging": False, "strength": 0}
        
        avg_short = sum(short_term) / len(short_term)
        avg_long = sum(long_term) / len(long_term)
        
        diff = abs(avg_short - avg_long)
        diverging = diff > 0.2  # 20% difference is significant
        
        return {
            "diverging": diverging,
            "strength": float(diff),
            "short_term_score": float(avg_short),
            "long_term_score": float(avg_long),
            "direction": "bullish_short" if avg_short > avg_long else "bullish_long" if avg_long > avg_short else "aligned"
        }
    
    def _calculate_confidence(self, alignment: float, weighted_score: float, 
                              divergence: Dict) -> float:
        """Calculate overall confidence in the aggregation"""
        confidence = (alignment * 0.4) + (weighted_score * 0.4)
        
        # Penalize divergence
        if divergence.get("diverging", False):
            confidence *= 0.7
        
        return float(min(1.0, confidence))
    
    def _calculate_roc(self, data: pd.DataFrame, period: int) -> Optional[float]:
        """Calculate Rate of Change"""
        if len(data) < period + 1:
            return None
        
        current = data['Close'].iloc[-1]
        past = data['Close'].iloc[-period-1]
        
        return ((current - past) / past) * 100 if past != 0 else 0
    
    def _calculate_volume_ratio(self, data: pd.DataFrame) -> float:
        """Calculate volume ratio (current / average)"""
        if len(data) < 20:
            return 1.0
        
        avg_volume = data['Volume'].iloc[-20:].mean()
        current_volume = data['Volume'].iloc[-1]
        
        return current_volume / avg_volume if avg_volume > 0 else 1.0
    
    def _get_min_bars(self, timeframe: Timeframe) -> int:
        """Get minimum required bars for meaningful analysis"""
        requirements = {
            Timeframe.INTRADAY_5M: 50,
            Timeframe.INTRADAY_15M: 50,
            Timeframe.INTRADAY_1H: 50,
            Timeframe.DAILY: 60,  # YOUR 60-DAY REQUIREMENT!
            Timeframe.WEEKLY: 30,
            Timeframe.MONTHLY: 24,
            Timeframe.QUARTERLY: 20,
            Timeframe.YEARLY: 10
        }
        return requirements.get(timeframe, 30)
    
    async def _get_timeframe_data(self, symbol: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        """Get data for specific timeframe with caching"""
        import yfinance as yf
        
        cache_key = f"{symbol}_{timeframe.value}"
        
        # Check cache
        if cache_key in self.data_cache:
            cached_time, data = self.data_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=5):
                return data
        
        try:
            period = TimeframeConfig.REQUIRED_PERIODS.get(timeframe, "1mo")
            interval = timeframe.value
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if not data.empty:
                self.data_cache[cache_key] = (datetime.now(), data)
                return data
            
        except Exception as e:
            logging.debug(f"Error fetching {symbol} {timeframe.value}: {e}")
        
        return None