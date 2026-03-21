"""
Volatility Indicators - Bollinger Bands, ATR, Keltner Channels, etc.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from agentic_trading_system.utils.logger import logger as  logging

class VolatilityIndicators:
    """
    Comprehensive volatility analysis indicators
    
    Indicators:
    - Bollinger Bands
    - ATR (Average True Range)
    - Keltner Channels
    - Historical Volatility
    - Volatility Ratio
    - Chaikin Volatility
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default periods
        self.bb_period = config.get("bb_period", 20)
        self.bb_std = config.get("bb_std", 2)
        self.atr_period = config.get("atr_period", 14)
        self.kc_period = config.get("kc_period", 20)
        self.kc_multiplier = config.get("kc_multiplier", 1.5)
        self.hv_periods = config.get("hv_periods", [10, 20, 30, 60])
        
        logging.info(f"✅ VolatilityIndicators initialized")
    
    def calculate_all(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all volatility indicators
        """
        results = {}
        
        # Bollinger Bands
        results.update(self.calculate_bollinger_bands(data))
        
        # ATR
        results.update(self.calculate_atr(data))
        
        # Keltner Channels
        results.update(self.calculate_keltner_channels(data))
        
        # Historical Volatility
        results.update(self.calculate_historical_volatility(data))
        
        # Volatility Ratio
        results.update(self.calculate_volatility_ratio(data))
        
        # Chaikin Volatility
        results.update(self.calculate_chaikin_volatility(data))
        
        # Volatility Regime
        results["volatility_regime"] = self.determine_volatility_regime(results)
        
        # Signal Generation
        results["signals"] = self.generate_signals(results)
        
        return results
    
    def calculate_bollinger_bands(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Bollinger Bands
        """
        close = data['Close']
        period = self.bb_period
        num_std = self.bb_std
        results = {}
        
        if len(close) >= period:
            # Middle Band (SMA)
            middle = close.rolling(window=period).mean()
            
            # Standard Deviation
            std = close.rolling(window=period).std()
            
            # Upper and Lower Bands
            upper = middle + (std * num_std)
            lower = middle - (std * num_std)
            
            current_price = close.iloc[-1]
            current_middle = middle.iloc[-1] if not pd.isna(middle.iloc[-1]) else current_price
            current_upper = upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else current_price * 1.05
            current_lower = lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else current_price * 0.95
            
            results["bb_middle"] = float(current_middle)
            results["bb_upper"] = float(current_upper)
            results["bb_lower"] = float(current_lower)
            
            # Band width
            results["bb_width"] = float((current_upper - current_lower) / current_middle)
            results["bb_width_pct"] = float(results["bb_width"] * 100)
            
            # Band position (0-100)
            if current_upper > current_lower:
                position = (current_price - current_lower) / (current_upper - current_lower) * 100
                results["bb_position"] = float(position)
            else:
                results["bb_position"] = 50.0
            
            # Squeeze detection
            results["bb_squeeze"] = bool(results["bb_width"] < results.get("bb_width_ma", results["bb_width"]))
            
            # Bollinger %B
            results["bb_percent_b"] = float((current_price - current_lower) / (current_upper - current_lower))
            
            # Conditions
            results["bb_above_upper"] = bool(current_price > current_upper)
            results["bb_below_lower"] = bool(current_price < current_lower)
            results["bb_near_upper"] = bool(current_price > current_middle + std.iloc[-1] * 1.5 if not pd.isna(std.iloc[-1]) else False)
            results["bb_near_lower"] = bool(current_price < current_middle - std.iloc[-1] * 1.5 if not pd.isna(std.iloc[-1]) else False)
        
        return results
    
    def calculate_atr(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Average True Range (ATR)
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        period = self.atr_period
        results = {}
        
        if len(data) > period:
            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # ATR
            atr = tr.rolling(window=period).mean()
            
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
            current_price = close.iloc[-1]
            
            results["atr"] = float(current_atr)
            results["atr_pct"] = float((current_atr / current_price) * 100) if current_price > 0 else 0
            results["atr_slope"] = self._calculate_slope(atr, 5)
            results["atr_trend"] = "increasing" if results["atr_slope"] > 0 else "decreasing"
            
            # ATR bands (simple)
            results["atr_upper_band"] = float(current_price + current_atr)
            results["atr_lower_band"] = float(current_price - current_atr)
        
        return results
    
    def calculate_keltner_channels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Keltner Channels
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        period = self.kc_period
        multiplier = self.kc_multiplier
        results = {}
        
        if len(data) > period:
            # Middle Line (EMA)
            ema = close.ewm(span=period, adjust=False).mean()
            
            # ATR for channel width
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            
            # Upper and Lower Channels
            upper = ema + (atr * multiplier)
            lower = ema - (atr * multiplier)
            
            current_price = close.iloc[-1]
            current_ema = ema.iloc[-1] if not pd.isna(ema.iloc[-1]) else current_price
            current_upper = upper.iloc[-1] if not pd.isna(upper.iloc[-1]) else current_price * 1.03
            current_lower = lower.iloc[-1] if not pd.isna(lower.iloc[-1]) else current_price * 0.97
            
            results["kc_middle"] = float(current_ema)
            results["kc_upper"] = float(current_upper)
            results["kc_lower"] = float(current_lower)
            results["kc_width"] = float((current_upper - current_lower) / current_ema)
            
            # Position
            if current_upper > current_lower:
                position = (current_price - current_lower) / (current_upper - current_lower) * 100
                results["kc_position"] = float(position)
            else:
                results["kc_position"] = 50.0
            
            # Conditions
            results["kc_above_upper"] = bool(current_price > current_upper)
            results["kc_below_lower"] = bool(current_price < current_lower)
        
        return results
    
    def calculate_historical_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Historical Volatility for different periods
        """
        close = data['Close']
        results = {}
        
        # Calculate daily returns
        returns = close.pct_change().dropna() * 100
        
        if len(returns) > 0:
            # Volatility for different periods
            for period in self.hv_periods:
                if len(returns) >= period:
                    vol = returns.iloc[-period:].std()
                    results[f"hv_{period}"] = float(vol)
                    
                    # Annualized
                    results[f"hv_{period}_annualized"] = float(vol * np.sqrt(252))
            
            # Volatility percentile (where current volatility ranks)
            if len(returns) >= 60:
                current_vol = returns.iloc[-20:].std()
                hist_vols = []
                
                for i in range(0, len(returns) - 20, 5):
                    window = returns.iloc[i:i+20]
                    if len(window) == 20:
                        hist_vols.append(window.std())
                
                if hist_vols:
                    percentile = sum(1 for v in hist_vols if v < current_vol) / len(hist_vols) * 100
                    results["volatility_percentile"] = float(percentile)
        
        return results
    
    def calculate_volatility_ratio(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Volatility Ratio (current volatility / average volatility)
        """
        close = data['Close']
        results = {}
        
        if len(close) > 30:
            returns = close.pct_change().dropna() * 100
            
            current_vol = returns.iloc[-5:].std() if len(returns) >= 5 else returns.std()
            avg_vol = returns.iloc[-30:].std() if len(returns) >= 30 else returns.std()
            
            if avg_vol > 0:
                results["volatility_ratio"] = float(current_vol / avg_vol)
                
                # Classification
                vr = results["volatility_ratio"]
                if vr > 1.5:
                    results["volatility_regime"] = "very_high"
                elif vr > 1.2:
                    results["volatility_regime"] = "high"
                elif vr > 0.8:
                    results["volatility_regime"] = "normal"
                elif vr > 0.5:
                    results["volatility_regime"] = "low"
                else:
                    results["volatility_regime"] = "very_low"
        
        return results
    
    def calculate_chaikin_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Chaikin Volatility
        """
        high = data['High']
        low = data['Low']
        period = 10
        roc_period = 10
        results = {}
        
        if len(data) > period + roc_period:
            # Moving average of high-low range
            hl_range = high - low
            hl_ma = hl_range.rolling(window=period).mean()
            
            # Rate of change
            chaikin_vol = ((hl_ma - hl_ma.shift(roc_period)) / hl_ma.shift(roc_period)) * 100
            
            results["chaikin_volatility"] = float(chaikin_vol.iloc[-1]) if not pd.isna(chaikin_vol.iloc[-1]) else 0
            results["chaikin_volatility_slope"] = self._calculate_slope(chaikin_vol, 5)
        
        return results
    
    def determine_volatility_regime(self, indicators: Dict) -> Dict[str, Any]:
        """
        Determine overall volatility regime
        """
        regime = {
            "level": "normal",
            "score": 0.5,
            "trend": "stable"
        }
        
        # Check multiple indicators
        signals = []
        
        # Bollinger width
        if "bb_width_pct" in indicators:
            width = indicators["bb_width_pct"]
            if width > 8:
                signals.append(("high", 0.8))
            elif width > 5:
                signals.append(("high", 0.6))
            elif width > 3:
                signals.append(("normal", 0.5))
            else:
                signals.append(("low", 0.3))
        
        # Volatility ratio
        if "volatility_ratio" in indicators:
            vr = indicators["volatility_ratio"]
            if vr > 1.5:
                signals.append(("high", 0.9))
            elif vr > 1.2:
                signals.append(("high", 0.7))
            elif vr > 0.8:
                signals.append(("normal", 0.5))
            elif vr > 0.5:
                signals.append(("low", 0.3))
            else:
                signals.append(("very_low", 0.2))
        
        # ATR percent
        if "atr_pct" in indicators:
            atr_pct = indicators["atr_pct"]
            if atr_pct > 5:
                signals.append(("high", 0.8))
            elif atr_pct > 3:
                signals.append(("normal", 0.5))
            elif atr_pct > 1.5:
                signals.append(("low", 0.3))
            else:
                signals.append(("very_low", 0.2))
        
        # Determine consensus
        if signals:
            levels = [s[0] for s in signals]
            scores = [s[1] for s in signals]
            
            # Most common level
            from collections import Counter
            level_counts = Counter(levels)
            regime["level"] = level_counts.most_common(1)[0][0]
            
            # Average score
            regime["score"] = float(np.mean(scores))
        
        # Trend
        if "atr_trend" in indicators:
            regime["trend"] = indicators["atr_trend"]
        
        return regime
    
    def generate_signals(self, indicators: Dict) -> List[Dict]:
        """
        Generate trading signals based on volatility indicators
        """
        signals = []
        
        # Bollinger Band signals
        if indicators.get("bb_above_upper"):
            signals.append({
                "indicator": "bollinger_bands",
                "signal": "bearish",
                "strength": 0.7,
                "description": "Price above upper Bollinger Band - extended"
            })
        elif indicators.get("bb_below_lower"):
            signals.append({
                "indicator": "bollinger_bands",
                "signal": "bullish",
                "strength": 0.7,
                "description": "Price below lower Bollinger Band - oversold"
            })
        
        if indicators.get("bb_near_upper"):
            signals.append({
                "indicator": "bollinger_bands",
                "signal": "bearish",
                "strength": 0.5,
                "description": "Price near upper Bollinger Band"
            })
        elif indicators.get("bb_near_lower"):
            signals.append({
                "indicator": "bollinger_bands",
                "signal": "bullish",
                "strength": 0.5,
                "description": "Price near lower Bollinger Band"
            })
        
        # Bollinger %B signals
        bb_pct = indicators.get("bb_percent_b", 0.5)
        if bb_pct > 0.9:
            signals.append({
                "indicator": "bollinger_bands",
                "signal": "bearish",
                "strength": 0.6,
                "description": f"Bollinger %B at {bb_pct:.2f} - overbought"
            })
        elif bb_pct < 0.1:
            signals.append({
                "indicator": "bollinger_bands",
                "signal": "bullish",
                "strength": 0.6,
                "description": f"Bollinger %B at {bb_pct:.2f} - oversold"
            })
        
        # ATR signals (volatility expansion)
        atr_slope = indicators.get("atr_slope", 0)
        if atr_slope > 0.05:
            signals.append({
                "indicator": "atr",
                "signal": "neutral",
                "strength": 0.5,
                "description": "Volatility expanding - wider stops recommended"
            })
        elif atr_slope < -0.05:
            signals.append({
                "indicator": "atr",
                "signal": "neutral",
                "strength": 0.4,
                "description": "Volatility contracting - tighter stops possible"
            })
        
        # Keltner Channel signals
        if indicators.get("kc_above_upper"):
            signals.append({
                "indicator": "keltner",
                "signal": "bearish",
                "strength": 0.6,
                "description": "Price above Keltner upper channel"
            })
        elif indicators.get("kc_below_lower"):
            signals.append({
                "indicator": "keltner",
                "signal": "bullish",
                "strength": 0.6,
                "description": "Price below Keltner lower channel"
            })
        
        # Volatility regime signals
        regime = indicators.get("volatility_regime", {}).get("level", "normal")
        if regime == "very_high":
            signals.append({
                "indicator": "volatility",
                "signal": "neutral",
                "strength": 0.7,
                "description": "Very high volatility - reduce position sizes"
            })
        elif regime == "very_low":
            signals.append({
                "indicator": "volatility",
                "signal": "neutral",
                "strength": 0.5,
                "description": "Very low volatility - potential breakout soon"
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