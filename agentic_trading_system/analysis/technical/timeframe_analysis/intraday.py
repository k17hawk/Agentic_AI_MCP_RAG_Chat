"""
Intraday Analyzer - Analysis for intraday timeframes (5min, 15min, 1hr)
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agentic_trading_system.utils.logger import logger as logging

class IntradayAnalyzer:
    """
    Analyzes intraday timeframes for short-term trading signals
    
    Timeframes: 5min, 15min, 1hr
    Use cases: Entry timing, short-term momentum, day trading signals
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Intraday-specific settings
        self.rsi_period = config.get("rsi_period", 14)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        
        # Volume thresholds
        self.volume_spike_threshold = config.get("volume_spike_threshold", 1.5)
        
        logging.info(f"✅ IntradayAnalyzer initialized")
    
    def analyze(self, data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """
        Analyze intraday data for given timeframe
        """
        if data.empty or len(data) < 30:
            return {"error": "Insufficient data", "timeframe": timeframe}
        
        # Calculate basic indicators
        results = {
            "timeframe": timeframe,
            "current_price": float(data['Close'].iloc[-1]),
            "data_points": len(data),
            "timestamp": datetime.now().isoformat()
        }
        
        # Trend analysis
        results["trend"] = self._analyze_trend(data)
        
        # Momentum analysis
        results["momentum"] = self._analyze_momentum(data)
        
        # Volume analysis
        results["volume"] = self._analyze_volume(data)
        
        # Volatility analysis
        results["volatility"] = self._analyze_volatility(data)
        
        # Support/Resistance levels
        results["levels"] = self._find_levels(data)
        
        # Entry/Exit signals
        results["signals"] = self._generate_signals(data, results)
        
        # Composite score
        results["composite_score"] = self._calculate_composite_score(results)
        
        return results
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze trend for intraday timeframe
        """
        close = data['Close']
        
        # Moving averages (shorter periods for intraday)
        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]
        
        current = close.iloc[-1]
        
        # Trend direction
        if current > ma10 > ma20:
            direction = "bullish"
            strength = "strong"
        elif current > ma20:
            direction = "bullish"
            strength = "moderate"
        elif current < ma10 < ma20:
            direction = "bearish"
            strength = "strong"
        elif current < ma20:
            direction = "bearish"
            strength = "moderate"
        else:
            direction = "neutral"
            strength = "weak"
        
        # Slope (rate of change)
        slope_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100 if len(close) >= 5 else 0
        slope_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100 if len(close) >= 10 else 0
        
        return {
            "direction": direction,
            "strength": strength,
            "ma5": float(ma5),
            "ma10": float(ma10),
            "ma20": float(ma20),
            "slope_5": float(slope_5),
            "slope_10": float(slope_10),
            "price_vs_ma10": float((current - ma10) / ma10 * 100)
        }
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze momentum for intraday timeframe
        """
        close = data['Close']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        
        # MACD
        exp1 = close.ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = close.ewm(span=self.macd_slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=self.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        current_macd = macd_line.iloc[-1] if not pd.isna(macd_line.iloc[-1]) else 0
        current_signal = signal_line.iloc[-1] if not pd.isna(signal_line.iloc[-1]) else 0
        current_histogram = histogram.iloc[-1] if not pd.isna(histogram.iloc[-1]) else 0
        
        # Stochastic (for intraday)
        low_14 = data['Low'].rolling(14).min()
        high_14 = data['High'].rolling(14).max()
        stoch_k = 100 * ((close - low_14) / (high_14 - low_14))
        stoch_d = stoch_k.rolling(3).mean()
        
        current_k = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
        current_d = stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50
        
        return {
            "rsi": float(current_rsi),
            "rsi_signal": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral",
            "macd": float(current_macd),
            "macd_signal": float(current_signal),
            "macd_histogram": float(current_histogram),
            "macd_cross": "bullish" if current_macd > current_signal else "bearish" if current_macd < current_signal else "neutral",
            "stoch_k": float(current_k),
            "stoch_d": float(current_d),
            "stoch_signal": "overbought" if current_k > 80 else "oversold" if current_k < 20 else "neutral"
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volume patterns for intraday
        """
        volume = data['Volume']
        close = data['Close']
        
        # Volume moving averages
        vol_ma5 = volume.rolling(5).mean().iloc[-1]
        vol_ma10 = volume.rolling(10).mean().iloc[-1]
        current_vol = volume.iloc[-1]
        
        # Volume ratios
        vol_ratio_5 = current_vol / vol_ma5 if vol_ma5 > 0 else 1
        vol_ratio_10 = current_vol / vol_ma10 if vol_ma10 > 0 else 1
        
        # Volume price trend (simplified OBV)
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        obv_trend = "up" if obv[-1] > obv[-5] else "down" if obv[-1] < obv[-5] else "flat"
        
        return {
            "current_volume": float(current_vol),
            "vol_ma5": float(vol_ma5),
            "vol_ma10": float(vol_ma10),
            "vol_ratio_5": float(vol_ratio_5),
            "vol_ratio_10": float(vol_ratio_10),
            "volume_spike": vol_ratio_5 > self.volume_spike_threshold,
            "obv_trend": obv_trend
        }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volatility for intraday
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # Intraday range
        daily_range = (high - low) / close * 100
        current_range = daily_range.iloc[-1] if not pd.isna(daily_range.iloc[-1]) else 0
        avg_range = daily_range.rolling(10).mean().iloc[-1] if len(daily_range) >= 10 else current_range
        
        # ATR (simplified for intraday)
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1] if len(tr) >= 14 else tr.iloc[-1]
        atr_pct = (atr / close.iloc[-1]) * 100 if close.iloc[-1] != 0 else 0
        
        # Bollinger Bands (20,2)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = ma20 + (std20 * 2)
        bb_lower = ma20 - (std20 * 2)
        
        current_bb_upper = bb_upper.iloc[-1] if not pd.isna(bb_upper.iloc[-1]) else close.iloc[-1] * 1.02
        current_bb_lower = bb_lower.iloc[-1] if not pd.isna(bb_lower.iloc[-1]) else close.iloc[-1] * 0.98
        
        return {
            "current_range_pct": float(current_range),
            "avg_range_pct": float(avg_range),
            "range_ratio": float(current_range / avg_range if avg_range > 0 else 1),
            "atr": float(atr),
            "atr_pct": float(atr_pct),
            "bb_upper": float(current_bb_upper),
            "bb_lower": float(current_bb_lower),
            "bb_width": float((current_bb_upper - current_bb_lower) / close.iloc[-1] * 100)
        }
    
    def _find_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Find support and resistance levels for intraday
        """
        high = data['High'].iloc[-20:]  # Last 20 periods
        low = data['Low'].iloc[-20:]
        close = data['Close'].iloc[-1]
        
        # Pivot points
        pivot = (high.max() + low.min() + close) / 3
        r1 = 2 * pivot - low.min()
        s1 = 2 * pivot - high.max()
        r2 = pivot + (high.max() - low.min())
        s2 = pivot - (high.max() - low.min())
        
        # Find nearest levels
        levels = {
            "pivot": float(pivot),
            "r1": float(r1),
            "s1": float(s1),
            "r2": float(r2),
            "s2": float(s2)
        }
        
        # Distance to nearest levels
        distances = {}
        for level_name, level_price in levels.items():
            distances[f"dist_to_{level_name}"] = float((level_price - close) / close * 100)
        
        return {**levels, **distances}
    
    def _generate_signals(self, data: pd.DataFrame, analysis: Dict) -> List[Dict]:
        """
        Generate intraday trading signals
        """
        signals = []
        
        # RSI signals
        rsi = analysis["momentum"]["rsi"]
        if rsi < 30:
            signals.append({
                "type": "oversold",
                "signal": "buy",
                "strength": 0.7,
                "description": f"RSI oversold at {rsi:.1f}"
            })
        elif rsi > 70:
            signals.append({
                "type": "overbought",
                "signal": "sell",
                "strength": 0.7,
                "description": f"RSI overbought at {rsi:.1f}"
            })
        
        # MACD signals
        if analysis["momentum"]["macd_cross"] == "bullish":
            signals.append({
                "type": "macd_cross",
                "signal": "buy",
                "strength": 0.6,
                "description": "MACD bullish crossover"
            })
        elif analysis["momentum"]["macd_cross"] == "bearish":
            signals.append({
                "type": "macd_cross",
                "signal": "sell",
                "strength": 0.6,
                "description": "MACD bearish crossover"
            })
        
        # Volume spike signals
        if analysis["volume"]["volume_spike"]:
            if analysis["trend"]["direction"] == "bullish":
                signals.append({
                    "type": "volume_spike",
                    "signal": "buy",
                    "strength": 0.8,
                    "description": "Volume spike with bullish trend"
                })
            elif analysis["trend"]["direction"] == "bearish":
                signals.append({
                    "type": "volume_spike",
                    "signal": "sell",
                    "strength": 0.8,
                    "description": "Volume spike with bearish trend"
                })
        
        # Breakout signals
        current_price = analysis["current_price"]
        bb_upper = analysis["volatility"]["bb_upper"]
        bb_lower = analysis["volatility"]["bb_lower"]
        
        if current_price > bb_upper:
            signals.append({
                "type": "breakout",
                "signal": "buy" if analysis["trend"]["direction"] == "bullish" else "neutral",
                "strength": 0.6,
                "description": "Price above upper Bollinger Band"
            })
        elif current_price < bb_lower:
            signals.append({
                "type": "breakout",
                "signal": "sell" if analysis["trend"]["direction"] == "bearish" else "neutral",
                "strength": 0.6,
                "description": "Price below lower Bollinger Band"
            })
        
        return signals
    
    def _calculate_composite_score(self, analysis: Dict) -> float:
        """
        Calculate composite score for intraday timeframe
        """
        scores = []
        
        # Trend contribution
        trend = analysis["trend"]
        if trend["direction"] == "bullish":
            if trend["strength"] == "strong":
                scores.append(0.8)
            else:
                scores.append(0.7)
        elif trend["direction"] == "bearish":
            if trend["strength"] == "strong":
                scores.append(0.2)
            else:
                scores.append(0.3)
        else:
            scores.append(0.5)
        
        # Momentum contribution
        momentum = analysis["momentum"]
        if momentum["rsi"] < 40:  # Approaching oversold
            scores.append(0.7)
        elif momentum["rsi"] > 60:  # Approaching overbought
            scores.append(0.3)
        else:
            scores.append(0.5)
        
        # Volume contribution
        volume = analysis["volume"]
        if volume["volume_spike"]:
            if volume["obv_trend"] == "up":
                scores.append(0.8)
            else:
                scores.append(0.2)
        else:
            scores.append(0.5)
        
        # Volatility contribution (moderate volatility is good)
        volatility = analysis["volatility"]
        if 1.5 < volatility["range_ratio"] < 2.5:
            scores.append(0.7)
        elif volatility["range_ratio"] > 3:
            scores.append(0.3)
        else:
            scores.append(0.5)
        
        return float(np.mean(scores)) if scores else 0.5