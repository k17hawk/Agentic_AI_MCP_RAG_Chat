"""
Daily Analyzer - Analysis for daily timeframe
THIS IS YOUR 60-DAY CORE TIMEFRAME!
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agentic_trading_system.utils.logger import logger as  logging

class DailyAnalyzer:
    """
    Analyzes daily timeframe for medium-term trading signals
    
    YOUR 60-DAY REQUIREMENT:
    - Uses 60+ days of data
    - Primary timeframe for trend analysis
    - Highest weight in multi-timeframe aggregation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Daily-specific settings
        self.required_days = config.get("required_days", 60)  # YOUR 60-DAY REQUIREMENT!
        self.rsi_period = config.get("rsi_period", 14)
        self.macd_fast = config.get("macd_fast", 12)
        self.macd_slow = config.get("macd_slow", 26)
        self.macd_signal = config.get("macd_signal", 9)
        
        # Moving average periods
        self.ma_periods = config.get("ma_periods", [20, 50, 200])
        
        logging.info(f"✅ DailyAnalyzer initialized (requires {self.required_days} days)")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze daily data
        """
        if data.empty or len(data) < self.required_days:
            return {
                "error": f"Insufficient data: {len(data)} < {self.required_days}",
                "required_days": self.required_days
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
        
        # Trend analysis (YOUR 60-DAY FOCUS)
        results["trend"] = self._analyze_trend(data)
        
        # Momentum analysis
        results["momentum"] = self._analyze_momentum(data)
        
        # Volume analysis
        results["volume"] = self._analyze_volume(data)
        
        # Volatility analysis
        results["volatility"] = self._analyze_volatility(data)
        
        # Pattern analysis
        results["patterns"] = self._analyze_patterns(data)
        
        # Support/Resistance levels
        results["levels"] = self._find_levels(data)
        
        # Key levels (52-week high/low)
        results["key_levels"] = self._find_key_levels(data)
        
        # Signals
        results["signals"] = self._generate_signals(data, results)
        
        # Composite score
        results["composite_score"] = self._calculate_composite_score(results)
        
        return results
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze daily trend using 60-day window
        """
        close = data['Close']
        
        # Moving averages
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else ma20
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50
        
        current = close.iloc[-1]
        
        # Trend direction based on MA alignment
        if current > ma50 > ma200:
            direction = "strong_bullish"
            strength = 0.9
        elif current > ma50:
            direction = "bullish"
            strength = 0.7
        elif current < ma50 < ma200:
            direction = "strong_bearish"
            strength = 0.2
        elif current < ma50:
            direction = "bearish"
            strength = 0.3
        else:
            direction = "neutral"
            strength = 0.5
        
        # Golden/Death Cross detection
        ma50_series = close.rolling(50).mean()
        ma200_series = close.rolling(200).mean() if len(close) >= 200 else None
        
        golden_cross = False
        death_cross = False
        
        if ma200_series is not None and len(ma50_series) > 1 and len(ma200_series) > 1:
            golden_cross = (ma50_series.iloc[-1] > ma200_series.iloc[-1] and 
                           ma50_series.iloc[-2] <= ma200_series.iloc[-2])
            death_cross = (ma50_series.iloc[-1] < ma200_series.iloc[-1] and 
                          ma50_series.iloc[-2] >= ma200_series.iloc[-2])
        
        # 60-day slope (YOUR REQUIREMENT)
        if len(close) >= 60:
            x = np.arange(60)
            y = close.iloc[-60:].values
            slope = np.polyfit(x, y, 1)[0]
            slope_pct = (slope * 60 / close.iloc[-60]) * 100 if close.iloc[-60] != 0 else 0
        else:
            slope_pct = 0
        
        return {
            "direction": direction,
            "strength": strength,
            "ma20": float(ma20),
            "ma50": float(ma50),
            "ma200": float(ma200) if not pd.isna(ma200) else None,
            "price_vs_ma20": float((current - ma20) / ma20 * 100),
            "price_vs_ma50": float((current - ma50) / ma50 * 100),
            "price_vs_ma200": float((current - ma200) / ma200 * 100) if not pd.isna(ma200) else 0,
            "golden_cross": golden_cross,
            "death_cross": death_cross,
            "slope_60d_pct": float(slope_pct)
        }
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze daily momentum
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
        
        # Rate of Change (ROC) for different periods
        roc_5 = (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0
        roc_20 = (close.iloc[-1] / close.iloc[-20] - 1) * 100 if len(close) >= 20 else 0
        roc_60 = (close.iloc[-1] / close.iloc[-60] - 1) * 100 if len(close) >= 60 else 0  # YOUR 60-DAY
        
        return {
            "rsi": float(current_rsi),
            "rsi_signal": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral",
            "macd": float(current_macd),
            "macd_signal": float(current_signal),
            "macd_histogram": float(current_histogram),
            "macd_cross": "bullish" if current_macd > current_signal else "bearish" if current_macd < current_signal else "neutral",
            "roc_5": float(roc_5),
            "roc_20": float(roc_20),
            "roc_60": float(roc_60)  # YOUR 60-DAY METRIC
        }
    
    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze daily volume patterns
        """
        volume = data['Volume']
        close = data['Close']
        
        # Volume moving averages
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        vol_ma50 = volume.rolling(50).mean().iloc[-1] if len(volume) >= 50 else vol_ma20
        
        current_vol = volume.iloc[-1]
        
        # Volume ratios
        vol_ratio_20 = current_vol / vol_ma20 if vol_ma20 > 0 else 1
        vol_ratio_50 = current_vol / vol_ma50 if vol_ma50 > 0 else 1
        
        # On-Balance Volume (OBV)
        obv = [0]
        for i in range(1, len(close)):
            if close.iloc[i] > close.iloc[i-1]:
                obv.append(obv[-1] + volume.iloc[i])
            elif close.iloc[i] < close.iloc[i-1]:
                obv.append(obv[-1] - volume.iloc[i])
            else:
                obv.append(obv[-1])
        
        obv_series = pd.Series(obv)
        obv_ma = obv_series.rolling(20).mean()
        obv_slope = (obv_series.iloc[-1] - obv_series.iloc[-5]) / obv_series.iloc[-5] * 100 if len(obv_series) >= 5 else 0
        
        return {
            "current_volume": float(current_vol),
            "vol_ma20": float(vol_ma20),
            "vol_ma50": float(vol_ma50),
            "vol_ratio_20": float(vol_ratio_20),
            "vol_ratio_50": float(vol_ratio_50),
            "volume_spike": vol_ratio_20 > 1.5,
            "obv": float(obv_series.iloc[-1]),
            "obv_slope": float(obv_slope),
            "obv_trend": "up" if obv_slope > 0 else "down"
        }
    
    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze daily volatility
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # ATR
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': abs(high - close.shift()),
            'lc': abs(low - close.shift())
        }).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        atr_pct = (atr / close.iloc[-1]) * 100
        
        # Bollinger Bands (20,2)
        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_upper = ma20 + (std20 * 2)
        bb_lower = ma20 - (std20 * 2)
        
        current_bb_upper = bb_upper.iloc[-1]
        current_bb_lower = bb_lower.iloc[-1]
        bb_width = (current_bb_upper - current_bb_lower) / ma20.iloc[-1] * 100
        
        # Historical volatility (20-day)
        returns = close.pct_change().dropna() * 100
        hist_vol_20 = returns.iloc[-20:].std() if len(returns) >= 20 else 0
        hist_vol_60 = returns.iloc[-60:].std() if len(returns) >= 60 else 0  # YOUR 60-DAY
        
        return {
            "atr": float(atr),
            "atr_pct": float(atr_pct),
            "bb_upper": float(current_bb_upper),
            "bb_lower": float(current_bb_lower),
            "bb_width": float(bb_width),
            "bb_position": float((close.iloc[-1] - current_bb_lower) / (current_bb_upper - current_bb_lower) * 100),
            "hist_vol_20": float(hist_vol_20),
            "hist_vol_60": float(hist_vol_60)  # YOUR 60-DAY METRIC
        }
    
    def _analyze_patterns(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect common daily patterns
        """
        patterns = []
        close = data['Close']
        
        # Check for recent patterns (last 5 days)
        recent = data.iloc[-5:]
        
        # Bullish engulfing
        if (recent.iloc[-2]['Close'] < recent.iloc[-2]['Open'] and  # Previous bearish
            recent.iloc[-1]['Close'] > recent.iloc[-1]['Open'] and  # Current bullish
            recent.iloc[-1]['Open'] < recent.iloc[-2]['Close'] and
            recent.iloc[-1]['Close'] > recent.iloc[-2]['Open']):
            patterns.append({
                "name": "bullish_engulfing",
                "type": "reversal",
                "direction": "bullish",
                "confidence": 0.8
            })
        
        # Bearish engulfing
        elif (recent.iloc[-2]['Close'] > recent.iloc[-2]['Open'] and
              recent.iloc[-1]['Close'] < recent.iloc[-1]['Open'] and
              recent.iloc[-1]['Open'] > recent.iloc[-2]['Close'] and
              recent.iloc[-1]['Close'] < recent.iloc[-2]['Open']):
            patterns.append({
                "name": "bearish_engulfing",
                "type": "reversal",
                "direction": "bearish",
                "confidence": 0.8
            })
        
        # Doji
        body = abs(recent.iloc[-1]['Close'] - recent.iloc[-1]['Open'])
        range_hl = recent.iloc[-1]['High'] - recent.iloc[-1]['Low']
        if range_hl > 0 and body / range_hl < 0.1:
            patterns.append({
                "name": "doji",
                "type": "indecision",
                "direction": "neutral",
                "confidence": 0.6
            })
        
        return patterns
    
    def _find_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Find support and resistance levels
        """
        high = data['High']
        low = data['Low']
        close = data['Close'].iloc[-1]
        
        # Recent highs and lows (last 60 days - YOUR REQUIREMENT)
        recent_high = high.iloc[-60:].max()
        recent_low = low.iloc[-60:].min()
        
        # All-time high/low for the period
        period_high = high.max()
        period_low = low.min()
        
        # Pivot points
        pivot = (period_high + period_low + close) / 3
        
        return {
            "recent_high": float(recent_high),
            "recent_low": float(recent_low),
            "period_high": float(period_high),
            "period_low": float(period_low),
            "pivot": float(pivot),
            "distance_to_high": float((period_high - close) / close * 100),
            "distance_to_low": float((close - period_low) / close * 100)
        }
    
    def _find_key_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Find key levels like 52-week high/low
        """
        high = data['High']
        low = data['Low']
        close = data['Close'].iloc[-1]
        
        # 52-week (252 trading days) high/low
        if len(data) >= 252:
            week_high = high.iloc[-252:].max()
            week_low = low.iloc[-252:].min()
        else:
            week_high = high.max()
            week_low = low.min()
        
        return {
            "week_high": float(week_high),
            "week_low": float(week_low),
            "near_week_high": close > week_high * 0.95,
            "near_week_low": close < week_low * 1.05
        }
    
    def _generate_signals(self, data: pd.DataFrame, analysis: Dict) -> List[Dict]:
        """
        Generate daily trading signals
        """
        signals = []
        
        # Trend signals
        trend = analysis["trend"]
        if trend["golden_cross"]:
            signals.append({
                "type": "golden_cross",
                "signal": "buy",
                "strength": 0.9,
                "description": "Golden Cross detected - major bullish signal"
            })
        if trend["death_cross"]:
            signals.append({
                "type": "death_cross",
                "signal": "sell",
                "strength": 0.9,
                "description": "Death Cross detected - major bearish signal"
            })
        
        # RSI signals
        rsi = analysis["momentum"]["rsi"]
        if rsi < 30:
            signals.append({
                "type": "rsi_oversold",
                "signal": "buy",
                "strength": 0.7,
                "description": f"RSI oversold at {rsi:.1f}"
            })
        elif rsi > 70:
            signals.append({
                "type": "rsi_overbought",
                "signal": "sell",
                "strength": 0.7,
                "description": f"RSI overbought at {rsi:.1f}"
            })
        
        # Volume signals
        volume = analysis["volume"]
        if volume["volume_spike"]:
            if trend["direction"] in ["bullish", "strong_bullish"]:
                signals.append({
                    "type": "volume_spike",
                    "signal": "buy",
                    "strength": 0.8,
                    "description": "Volume spike confirming uptrend"
                })
            elif trend["direction"] in ["bearish", "strong_bearish"]:
                signals.append({
                    "type": "volume_spike",
                    "signal": "sell",
                    "strength": 0.8,
                    "description": "Volume spike confirming downtrend"
                })
        
        # 60-day breakout signals (YOUR REQUIREMENT)
        roc_60 = analysis["momentum"]["roc_60"]
        if roc_60 > 20:
            signals.append({
                "type": "breakout",
                "signal": "buy",
                "strength": 0.75,
                "description": f"Strong 60-day momentum: +{roc_60:.1f}%"
            })
        elif roc_60 < -20:
            signals.append({
                "type": "breakout",
                "signal": "sell",
                "strength": 0.75,
                "description": f"Strong 60-day decline: {roc_60:.1f}%"
            })
        
        return signals
    
    def _calculate_composite_score(self, analysis: Dict) -> float:
        """
        Calculate composite score for daily timeframe
        """
        scores = []
        
        # Trend score
        trend = analysis["trend"]
        if trend["direction"] == "strong_bullish":
            scores.append(0.9)
        elif trend["direction"] == "bullish":
            scores.append(0.7)
        elif trend["direction"] == "neutral":
            scores.append(0.5)
        elif trend["direction"] == "bearish":
            scores.append(0.3)
        else:  # strong_bearish
            scores.append(0.1)
        
        # RSI score
        rsi = analysis["momentum"]["rsi"]
        if rsi < 30:
            scores.append(0.8)  # Oversold - potential buy
        elif rsi < 45:
            scores.append(0.7)
        elif rsi < 55:
            scores.append(0.5)
        elif rsi < 70:
            scores.append(0.3)
        else:
            scores.append(0.2)  # Overbought
        
        # Volume score
        volume = analysis["volume"]
        if volume["volume_spike"] and volume["obv_trend"] == "up":
            scores.append(0.8)
        elif volume["volume_spike"] and volume["obv_trend"] == "down":
            scores.append(0.2)
        else:
            scores.append(0.5)
        
        # Volatility score (moderate is good)
        bb_position = analysis["volatility"]["bb_position"]
        if 20 < bb_position < 80:
            scores.append(0.7)
        else:
            scores.append(0.3)
        
        return float(np.mean(scores)) if scores else 0.5