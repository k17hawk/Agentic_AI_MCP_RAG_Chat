"""
Weekly Analyzer - Analysis for weekly timeframe
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agentic_trading_system.utils.logger import logger as  logging

class WeeklyAnalyzer:
    """
    Analyzes weekly timeframe for longer-term trend context
    
    Use cases: Primary trend identification, major support/resistance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Weekly-specific settings
        self.required_weeks = config.get("required_weeks", 26)  # 6 months of weekly data
        
        logging.info(f"✅ WeeklyAnalyzer initialized")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze weekly data
        """
        if data.empty or len(data) < self.required_weeks:
            return {
                "error": f"Insufficient data: {len(data)} < {self.required_weeks} weeks",
                "required_weeks": self.required_weeks
            }
        
        results = {
            "current_price": float(data['Close'].iloc[-1]),
            "data_points": len(data),
            "date_range": {
                "start": data.index[0].strftime("%Y-%m-%d"),
                "end": data.index[-1].strftime("%Y-%m-%d")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Weekly trend analysis
        results["trend"] = self._analyze_trend(data)
        
        # Weekly momentum
        results["momentum"] = self._analyze_momentum(data)
        
        # Weekly volume
        results["volume"] = self._analyze_volume(data)
        
        # Major support/resistance
        results["levels"] = self._find_levels(data)
        
        # Composite score
        results["composite_score"] = self._calculate_composite_score(results)
        
        return results
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze weekly trend
        """
        close = data['Close']
        
        # Weekly moving averages
        ma10 = close.rolling(10).mean().iloc[-1]  # ~2.5 months
        ma20 = close.rolling(20).mean().iloc[-1]  # ~5 months
        ma40 = close.rolling(40).mean().iloc[-1]  # ~10 months
        
        current = close.iloc[-1]
        
        # Trend direction
        if current > ma20 > ma40:
            direction = "bullish"
            strength = "strong"
        elif current > ma20:
            direction = "bullish"
            strength = "moderate"
        elif current < ma20 < ma40:
            direction = "bearish"
            strength = "strong"
        elif current < ma20:
            direction = "bearish"
            strength = "moderate"
        else:
            direction = "neutral"
            strength = "weak"
        
        return {
            "direction": direction,
            "strength": strength,
            "ma10": float(ma10),
            "ma20": float(ma20),
            "ma40": float(ma40),
            "price_vs_ma20": float((current - ma20) / ma20 * 100)
        }
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze weekly momentum
        """
        close = data['Close']
        
        # Weekly RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # Rate of Change
        roc_4 = (close.iloc[-1] / close.iloc[-4] - 1) * 100 if len(close) >= 4 else 0  # 1 month
        roc_13 = (close.iloc[-1] / close.iloc[-13] - 1) * 100 if len(close) >= 13 else 0  # 3 months
        roc_26 = (close.iloc[-1] / close.iloc[-26] - 1) * 100 if len(close) >= 26 else 0  # 6 months
        
        return {
            "rsi": float(current_rsi),
            "rsi_signal": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral",
            "roc_4w": float(roc_4),
            "roc_13w": float(roc_13),
            "roc_26w": float(roc_26)
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze weekly volume patterns
        """
        volume = data['Volume']
        
        # Volume moving averages
        vol_ma10 = volume.rolling(10).mean().iloc[-1]
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        
        current_vol = volume.iloc[-1]
        
        # Volume trends
        vol_trend_10 = (vol_ma10 / vol_ma20 - 1) * 100
        
        return {
            "current_volume": float(current_vol),
            "vol_ma10": float(vol_ma10),
            "vol_ma20": float(vol_ma20),
            "vol_trend": "increasing" if vol_trend_10 > 0 else "decreasing",
            "volume_spike": current_vol > vol_ma20 * 1.5
        }
    
    def _find_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Find major weekly support/resistance
        """
        high = data['High']
        low = data['Low']
        close = data['Close'].iloc[-1]
        
        # Yearly highs/lows
        year_high = high.iloc[-52:].max() if len(high) >= 52 else high.max()
        year_low = low.iloc[-52:].min() if len(low) >= 52 else low.min()
        
        # All-time highs/lows for the period
        period_high = high.max()
        period_low = low.min()
        
        return {
            "year_high": float(year_high),
            "year_low": float(year_low),
            "period_high": float(period_high),
            "period_low": float(period_low),
            "near_year_high": close > year_high * 0.93,
            "near_year_low": close < year_low * 1.07
        }
    
    def _calculate_composite_score(self, analysis: Dict) -> float:
        """
        Calculate composite score for weekly timeframe
        """
        scores = []
        
        # Trend score
        trend = analysis["trend"]
        if trend["direction"] == "bullish":
            if trend["strength"] == "strong":
                scores.append(0.8)
            else:
                scores.append(0.7)
        elif trend["direction"] == "neutral":
            scores.append(0.5)
        else:
            if trend["strength"] == "strong":
                scores.append(0.2)
            else:
                scores.append(0.3)
        
        # RSI score
        rsi = analysis["momentum"]["rsi"]
        if rsi < 40:
            scores.append(0.7)
        elif rsi < 60:
            scores.append(0.5)
        else:
            scores.append(0.3)
        
        # Volume score
        volume = analysis["volume"]
        if volume["volume_spike"] and volume["vol_trend"] == "increasing":
            scores.append(0.7)
        else:
            scores.append(0.5)
        
        return float(np.mean(scores)) if scores else 0.5