"""
Trend Indicators - Moving averages, MACD, Ichimoku, etc.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from agentic_trading_system.utils.logger import logger as  logging

class TrendIndicators:
    """
    Comprehensive trend analysis indicators
    
    Indicators:
    - Simple Moving Average (SMA)
    - Exponential Moving Average (EMA)
    - Weighted Moving Average (WMA)
    - Moving Average Convergence Divergence (MACD)
    - Ichimoku Cloud
    - Parabolic SAR
    - ADX (Average Directional Index)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default periods
        self.sma_periods = config.get("sma_periods", [10, 20, 50, 200])
        self.ema_periods = config.get("ema_periods", [12, 26, 50])
        self.macd_config = config.get("macd", {"fast": 12, "slow": 26, "signal": 9})
        self.ichimoku_config = config.get("ichimoku", {
            "tenkan": 9,
            "kijun": 26,
            "senkou": 52
        })
        self.adx_period = config.get("adx_period", 14)
        
        logging.info(f"✅ TrendIndicators initialized")
    
    def calculate_all(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all trend indicators
        """
        results = {}
        
        # Moving Averages
        results.update(self.calculate_moving_averages(data))
        
        # MACD
        results.update(self.calculate_macd(data))
        
        # Ichimoku
        results.update(self.calculate_ichimoku(data))
        
        # Parabolic SAR
        results.update(self.calculate_parabolic_sar(data))
        
        # ADX
        results.update(self.calculate_adx(data))
        
        # Trend Strength
        results["trend_strength"] = self.calculate_trend_strength(data)
        
        # Signal Generation
        results["signals"] = self.generate_signals(data, results)
        
        return results
    
    def calculate_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate various moving averages
        """
        close = data['Close']
        results = {}
        
        # SMA
        for period in self.sma_periods:
            if len(close) >= period:
                sma = close.rolling(window=period).mean()
                results[f"sma_{period}"] = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else None
                results[f"sma_{period}_slope"] = self._calculate_slope(sma, 5)
        
        # EMA
        for period in self.ema_periods:
            if len(close) >= period:
                ema = close.ewm(span=period, adjust=False).mean()
                results[f"ema_{period}"] = float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else None
                results[f"ema_{period}_slope"] = self._calculate_slope(ema, 5)
        
        # Golden/Death Cross detection
        if len(close) >= 50:
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean() if len(close) >= 200 else None
            
            if sma_200 is not None:
                current_sma_50 = sma_50.iloc[-1]
                current_sma_200 = sma_200.iloc[-1]
                prev_sma_50 = sma_50.iloc[-2]
                prev_sma_200 = sma_200.iloc[-2]
                
                results["golden_cross"] = bool(
                    current_sma_50 > current_sma_200 and 
                    prev_sma_50 <= prev_sma_200
                )
                
                results["death_cross"] = bool(
                    current_sma_50 < current_sma_200 and 
                    prev_sma_50 >= prev_sma_200
                )
        
        # Price vs MAs
        current_price = close.iloc[-1]
        results["price_vs_sma_20"] = float(current_price / results.get("sma_20", current_price) - 1) if results.get("sma_20") else 0
        results["price_vs_sma_50"] = float(current_price / results.get("sma_50", current_price) - 1) if results.get("sma_50") else 0
        results["price_vs_sma_200"] = float(current_price / results.get("sma_200", current_price) - 1) if results.get("sma_200") else 0
        
        return results
    
    def calculate_macd(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        """
        close = data['Close']
        results = {}
        
        fast = self.macd_config["fast"]
        slow = self.macd_config["slow"]
        signal = self.macd_config["signal"]
        
        if len(close) >= slow:
            # Calculate MACD line
            ema_fast = close.ewm(span=fast, adjust=False).mean()
            ema_slow = close.ewm(span=slow, adjust=False).mean()
            macd_line = ema_fast - ema_slow
            
            # Calculate signal line
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            results["macd_line"] = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0
            results["macd_signal"] = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0
            results["macd_histogram"] = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0
            results["macd_histogram_slope"] = self._calculate_slope(histogram, 5)
            
            # Crossover detection
            prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else None
            prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else None
            
            if prev_macd is not None and prev_signal is not None:
                results["macd_bullish_cross"] = bool(
                    results["macd_line"] > results["macd_signal"] and 
                    prev_macd <= prev_signal
                )
                results["macd_bearish_cross"] = bool(
                    results["macd_line"] < results["macd_signal"] and 
                    prev_macd >= prev_signal
                )
        
        return results
    
    def calculate_ichimoku(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Ichimoku Cloud indicators
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        results = {}
        
        tenkan = self.ichimoku_config["tenkan"]
        kijun = self.ichimoku_config["kijun"]
        senkou = self.ichimoku_config["senkou"]
        
        if len(data) >= senkou * 2:
            # Tenkan-sen (Conversion Line): (highest high + lowest low)/2 for past 9 periods
            tenkan_high = high.rolling(window=tenkan).max()
            tenkan_low = low.rolling(window=tenkan).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line): (highest high + lowest low)/2 for past 26 periods
            kijun_high = high.rolling(window=kijun).max()
            kijun_low = low.rolling(window=kijun).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2, shifted forward 26 periods
            senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
            
            # Senkou Span B (Leading Span B): (highest high + lowest low)/2 for past 52 periods, shifted forward 26 periods
            senkou_high = high.rolling(window=senkou).max()
            senkou_low = low.rolling(window=senkou).min()
            senkou_b = ((senkou_high + senkou_low) / 2).shift(kijun)
            
            # Chikou Span (Lagging Span): Current close shifted backward 26 periods
            chikou = close.shift(-kijun)
            
            # Current values
            current_tenkan = tenkan_sen.iloc[-1] if not pd.isna(tenkan_sen.iloc[-1]) else None
            current_kijun = kijun_sen.iloc[-1] if not pd.isna(kijun_sen.iloc[-1]) else None
            current_senkou_a = senkou_a.iloc[-1] if not pd.isna(senkou_a.iloc[-1]) else None
            current_senkou_b = senkou_b.iloc[-1] if not pd.isna(senkou_b.iloc[-1]) else None
            current_chikou = chikou.iloc[-1] if not pd.isna(chikou.iloc[-1]) else None
            current_price = close.iloc[-1]
            
            results["ichimoku_tenkan"] = float(current_tenkan) if current_tenkan else None
            results["ichimoku_kijun"] = float(current_kijun) if current_kijun else None
            results["ichimoku_senkou_a"] = float(current_senkou_a) if current_senkou_a else None
            results["ichimoku_senkou_b"] = float(current_senkou_b) if current_senkou_b else None
            results["ichimoku_chikou"] = float(current_chikou) if current_chikou else None
            
            # Cloud position
            if current_senkou_a and current_senkou_b:
                results["ichimoku_cloud_top"] = float(max(current_senkou_a, current_senkou_b))
                results["ichimoku_cloud_bottom"] = float(min(current_senkou_a, current_senkou_b))
                results["ichimoku_price_vs_cloud"] = float(current_price - results["ichimoku_cloud_top"]) if current_price > results["ichimoku_cloud_top"] else float(current_price - results["ichimoku_cloud_bottom"]) if current_price < results["ichimoku_cloud_bottom"] else 0
                
                # Signals
                results["ichimoku_above_cloud"] = bool(current_price > results["ichimoku_cloud_top"])
                results["ichimoku_below_cloud"] = bool(current_price < results["ichimoku_cloud_bottom"])
                results["ichimoku_in_cloud"] = bool(not results["ichimoku_above_cloud"] and not results["ichimoku_below_cloud"])
        
        return results
    
    def calculate_parabolic_sar(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Parabolic SAR (Stop and Reverse)
        """
        high = data['High']
        low = data['Low']
        results = {}
        
        acceleration = self.config.get("parabolic_sar_acceleration", 0.02)
        maximum = self.config.get("parabolic_sar_maximum", 0.2)
        
        if len(data) > 2:
            sar = self._calculate_psar(high, low, acceleration, maximum)
            
            if sar is not None:
                results["parabolic_sar"] = float(sar[-1]) if not pd.isna(sar[-1]) else None
                
                # Trend direction
                if len(sar) > 2:
                    results["parabolic_sar_trend"] = "up" if sar[-1] < sar[-2] else "down"
        
        return results
    
    def calculate_adx(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate ADX (Average Directional Index) for trend strength
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        period = self.adx_period
        
        results = {}
        
        if len(data) > period * 2:
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Directional Movement
            up_move = high - high.shift()
            down_move = low.shift() - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Smooth
            atr = tr.rolling(window=period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(window=period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(window=period).mean() / atr)
            
            # ADX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(window=period).mean()
            
            results["adx"] = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0
            results["plus_di"] = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0
            results["minus_di"] = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0
            
            # Trend strength interpretation
            if results["adx"] >= 25:
                results["adx_trend_strength"] = "strong"
            elif results["adx"] >= 20:
                results["adx_trend_strength"] = "moderate"
            else:
                results["adx_trend_strength"] = "weak"
            
            # Trend direction
            if results["plus_di"] > results["minus_di"]:
                results["adx_trend_direction"] = "up"
            else:
                results["adx_trend_direction"] = "down"
        
        return results
    
    def calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate overall trend strength (0-1)
        """
        close = data['Close']
        
        if len(close) < 50:
            return 0.5
        
        # Linear regression slope
        x = np.arange(50)
        y = close.iloc[-50:].values
        slope = np.polyfit(x, y, 1)[0]
        
        # Normalize slope
        price_range = close.iloc[-50:].max() - close.iloc[-50:].min()
        normalized_slope = abs(slope) / price_range if price_range > 0 else 0
        
        # R-squared (trend consistency)
        correlation = np.corrcoef(x, y)[0, 1]
        r_squared = correlation ** 2
        
        # Combine
        strength = (normalized_slope * 0.5 + r_squared * 0.5)
        
        return float(min(1.0, strength))
    
    def generate_signals(self, data: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """
        Generate trading signals based on trend indicators
        """
        signals = []
        close = data['Close']
        current_price = close.iloc[-1]
        
        # Moving Average signals
        if indicators.get("golden_cross"):
            signals.append({
                "indicator": "moving_averages",
                "signal": "bullish",
                "strength": 0.8,
                "description": "Golden Cross detected - major bullish signal"
            })
        
        if indicators.get("death_cross"):
            signals.append({
                "indicator": "moving_averages",
                "signal": "bearish",
                "strength": 0.8,
                "description": "Death Cross detected - major bearish signal"
            })
        
        # Price vs MA signals
        if indicators.get("price_vs_sma_50", 0) > 0.05:
            signals.append({
                "indicator": "moving_averages",
                "signal": "bullish",
                "strength": 0.6,
                "description": f"Price 5% above 50-day SMA"
            })
        elif indicators.get("price_vs_sma_50", 0) < -0.05:
            signals.append({
                "indicator": "moving_averages",
                "signal": "bearish",
                "strength": 0.6,
                "description": f"Price 5% below 50-day SMA"
            })
        
        # MACD signals
        if indicators.get("macd_bullish_cross"):
            signals.append({
                "indicator": "macd",
                "signal": "bullish",
                "strength": 0.7,
                "description": "MACD bullish crossover"
            })
        
        if indicators.get("macd_bearish_cross"):
            signals.append({
                "indicator": "macd",
                "signal": "bearish",
                "strength": 0.7,
                "description": "MACD bearish crossover"
            })
        
        if indicators.get("macd_histogram", 0) > 0:
            signals.append({
                "indicator": "macd",
                "signal": "bullish",
                "strength": 0.5,
                "description": "MACD histogram positive"
            })
        else:
            signals.append({
                "indicator": "macd",
                "signal": "bearish",
                "strength": 0.5,
                "description": "MACD histogram negative"
            })
        
        # Ichimoku signals
        if indicators.get("ichimoku_above_cloud"):
            signals.append({
                "indicator": "ichimoku",
                "signal": "bullish",
                "strength": 0.75,
                "description": "Price above Ichimoku cloud"
            })
        elif indicators.get("ichimoku_below_cloud"):
            signals.append({
                "indicator": "ichimoku",
                "signal": "bearish",
                "strength": 0.75,
                "description": "Price below Ichimoku cloud"
            })
        
        # ADX signals
        adx = indicators.get("adx", 0)
        if adx > 25:
            if indicators.get("adx_trend_direction") == "up":
                signals.append({
                    "indicator": "adx",
                    "signal": "bullish",
                    "strength": 0.6,
                    "description": f"Strong uptrend (ADX: {adx:.1f})"
                })
            elif indicators.get("adx_trend_direction") == "down":
                signals.append({
                    "indicator": "adx",
                    "signal": "bearish",
                    "strength": 0.6,
                    "description": f"Strong downtrend (ADX: {adx:.1f})"
                })
        
        return signals
    
    def _calculate_slope(self, series: pd.Series, periods: int) -> float:
        """Calculate slope of series over given periods"""
        if len(series) < periods:
            return 0.0
        
        y = series.iloc[-periods:].values
        x = np.arange(len(y))
        
        if np.std(y) == 0:
            return 0.0
        
        slope = np.polyfit(x, y, 1)[0]
        return float(slope / np.mean(y) if np.mean(y) != 0 else 0)
    
    def _calculate_psar(self, high: pd.Series, low: pd.Series, 
                       acceleration: float, maximum: float) -> pd.Series:
        """Calculate Parabolic SAR"""
        length = len(high)
        sar = np.zeros(length)
        trend = np.zeros(length)
        ep = np.zeros(length)
        af = np.zeros(length)
        
        if length < 3:
            return None
        
        # Initialization
        if high.iloc[1] > high.iloc[0]:
            trend[1] = 1  # Uptrend
            sar[1] = low.iloc[0]
            ep[1] = high.iloc[1]
            af[1] = acceleration
        else:
            trend[1] = -1  # Downtrend
            sar[1] = high.iloc[0]
            ep[1] = low.iloc[1]
            af[1] = acceleration
        
        for i in range(2, length):
            sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
            
            if trend[i-1] == 1:  # Uptrend
                if sar[i] > low.iloc[i]:
                    # Trend reversal
                    trend[i] = -1
                    sar[i] = ep[i-1]
                    ep[i] = low.iloc[i]
                    af[i] = acceleration
                else:
                    trend[i] = 1
                    if high.iloc[i] > ep[i-1]:
                        ep[i] = high.iloc[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure SAR is below low
                    if sar[i] > low.iloc[i]:
                        sar[i] = low.iloc[i]
            
            else:  # Downtrend
                if sar[i] < high.iloc[i]:
                    # Trend reversal
                    trend[i] = 1
                    sar[i] = ep[i-1]
                    ep[i] = high.iloc[i]
                    af[i] = acceleration
                else:
                    trend[i] = -1
                    if low.iloc[i] < ep[i-1]:
                        ep[i] = low.iloc[i]
                        af[i] = min(af[i-1] + acceleration, maximum)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    # Ensure SAR is above high
                    if sar[i] < high.iloc[i]:
                        sar[i] = high.iloc[i]
        
        return pd.Series(sar, index=high.index)