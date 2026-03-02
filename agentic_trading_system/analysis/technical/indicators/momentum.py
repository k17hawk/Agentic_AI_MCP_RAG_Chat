"""
Momentum Indicators - RSI, Stochastic, Williams %R, etc.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from utils.logger import logger as logging

class MomentumIndicators:
    """
    Comprehensive momentum analysis indicators
    
    Indicators:
    - RSI (Relative Strength Index)
    - Stochastic Oscillator
    - Williams %R
    - CCI (Commodity Channel Index)
    - ROC (Rate of Change)
    - Ultimate Oscillator
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default periods
        self.rsi_period = config.get("rsi_period", 14)
        self.stoch_period = config.get("stoch_period", 14)
        self.williams_period = config.get("williams_period", 14)
        self.cci_period = config.get("cci_period", 20)
        self.ultimate_periods = config.get("ultimate_periods", [7, 14, 28])
        
        # Overbought/oversold thresholds
        self.overbought_rsi = config.get("overbought_rsi", 70)
        self.oversold_rsi = config.get("oversold_rsi", 30)
        self.overbought_stoch = config.get("overbought_stoch", 80)
        self.oversold_stoch = config.get("oversold_stoch", 20)
        self.overbought_williams = config.get("overbought_williams", -20)
        self.oversold_williams = config.get("oversold_williams", -80)
        
        logging.info(f"✅ MomentumIndicators initialized")
    
    def calculate_all(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all momentum indicators
        """
        results = {}
        
        # RSI
        results.update(self.calculate_rsi(data))
        
        # Stochastic
        results.update(self.calculate_stochastic(data))
        
        # Williams %R
        results.update(self.calculate_williams_r(data))
        
        # CCI
        results.update(self.calculate_cci(data))
        
        # ROC
        results.update(self.calculate_roc(data))
        
        # Ultimate Oscillator
        results.update(self.calculate_ultimate_oscillator(data))
        
        # Momentum Score
        results["momentum_score"] = self.calculate_momentum_score(results)
        
        # Signal Generation
        results["signals"] = self.generate_signals(results)
        
        return results
    
    def calculate_rsi(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Relative Strength Index (RSI)
        """
        close = data['Close']
        period = self.rsi_period
        results = {}
        
        if len(close) > period:
            # Calculate price changes
            delta = close.diff()
            
            # Separate gains and losses
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # Average gains and losses
            avg_gain = gain.rolling(window=period).mean()
            avg_loss = loss.rolling(window=period).mean()
            
            # RS and RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            prev_rsi = rsi.iloc[-2] if len(rsi) > 1 and not pd.isna(rsi.iloc[-2]) else 50
            
            results["rsi"] = float(current_rsi)
            results["rsi_slope"] = float(current_rsi - prev_rsi)
            results["rsi_ma"] = float(rsi.rolling(5).mean().iloc[-1]) if len(rsi) > 5 else current_rsi
            
            # RSI conditions
            results["rsi_oversold"] = bool(current_rsi < self.oversold_rsi)
            results["rsi_overbought"] = bool(current_rsi > self.overbought_rsi)
            results["rsi_bullish_divergence"] = self._check_bullish_divergence(data, rsi)
            results["rsi_bearish_divergence"] = self._check_bearish_divergence(data, rsi)
        
        return results
    
    def calculate_stochastic(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Stochastic Oscillator (%K and %D)
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        period = self.stoch_period
        results = {}
        
        if len(data) > period:
            # Calculate %K
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            
            stoch_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            
            # Calculate %D (3-period SMA of %K)
            stoch_d = stoch_k.rolling(window=3).mean()
            
            current_k = stoch_k.iloc[-1] if not pd.isna(stoch_k.iloc[-1]) else 50
            current_d = stoch_d.iloc[-1] if not pd.isna(stoch_d.iloc[-1]) else 50
            prev_k = stoch_k.iloc[-2] if len(stoch_k) > 1 and not pd.isna(stoch_k.iloc[-2]) else 50
            prev_d = stoch_d.iloc[-2] if len(stoch_d) > 1 and not pd.isna(stoch_d.iloc[-2]) else 50
            
            results["stoch_k"] = float(current_k)
            results["stoch_d"] = float(current_d)
            results["stoch_k_slope"] = float(current_k - prev_k)
            results["stoch_d_slope"] = float(current_d - prev_d)
            
            # Conditions
            results["stoch_oversold"] = bool(current_k < self.oversold_stoch)
            results["stoch_overbought"] = bool(current_k > self.overbought_stoch)
            results["stoch_bullish_cross"] = bool(current_k > current_d and prev_k <= prev_d)
            results["stoch_bearish_cross"] = bool(current_k < current_d and prev_k >= prev_d)
        
        return results
    
    def calculate_williams_r(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Williams %R
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        period = self.williams_period
        results = {}
        
        if len(data) > period:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            
            williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
            
            current_wr = williams_r.iloc[-1] if not pd.isna(williams_r.iloc[-1]) else -50
            prev_wr = williams_r.iloc[-2] if len(williams_r) > 1 and not pd.isna(williams_r.iloc[-2]) else -50
            
            results["williams_r"] = float(current_wr)
            results["williams_r_slope"] = float(current_wr - prev_wr)
            
            # Conditions
            results["williams_oversold"] = bool(current_wr < self.oversold_williams)
            results["williams_overbought"] = bool(current_wr > self.overbought_williams)
        
        return results
    
    def calculate_cci(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Commodity Channel Index (CCI)
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        period = self.cci_period
        results = {}
        
        if len(data) > period:
            # Typical Price
            tp = (high + low + close) / 3
            
            # Simple Moving Average of TP
            sma_tp = tp.rolling(window=period).mean()
            
            # Mean Deviation
            md = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            
            # CCI
            cci = (tp - sma_tp) / (0.015 * md)
            
            current_cci = cci.iloc[-1] if not pd.isna(cci.iloc[-1]) else 0
            prev_cci = cci.iloc[-2] if len(cci) > 1 and not pd.isna(cci.iloc[-2]) else 0
            
            results["cci"] = float(current_cci)
            results["cci_slope"] = float(current_cci - prev_cci)
            
            # Conditions
            results["cci_oversold"] = bool(current_cci < -100)
            results["cci_overbought"] = bool(current_cci > 100)
            results["cci_bullish"] = bool(current_cci > -100 and prev_cci <= -100)
            results["cci_bearish"] = bool(current_cci < 100 and prev_cci >= 100)
        
        return results
    
    def calculate_roc(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Rate of Change (ROC)
        """
        close = data['Close']
        results = {}
        
        periods = [5, 10, 20, 60]
        
        for period in periods:
            if len(close) > period:
                roc = ((close - close.shift(period)) / close.shift(period)) * 100
                current_roc = roc.iloc[-1] if not pd.isna(roc.iloc[-1]) else 0
                results[f"roc_{period}"] = float(current_roc)
        
        # ROC moving average
        if len(close) > 20:
            roc_10 = ((close - close.shift(10)) / close.shift(10)) * 100
            roc_ma = roc_10.rolling(10).mean()
            results["roc_ma"] = float(roc_ma.iloc[-1]) if not pd.isna(roc_ma.iloc[-1]) else 0
        
        return results
    
    def calculate_ultimate_oscillator(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Ultimate Oscillator
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        periods = self.ultimate_periods
        results = {}
        
        if len(data) > max(periods):
            # Buying Pressure
            bp = close - np.minimum(low, close.shift())
            
            # True Range
            tr = np.maximum(high, close.shift()) - np.minimum(low, close.shift())
            
            # Averages for each period
            averages = []
            weights = [4, 2, 1]
            
            for period, weight in zip(periods, weights):
                avg_bp = bp.rolling(window=period).sum()
                avg_tr = tr.rolling(window=period).sum()
                avg = avg_bp / avg_tr * 100
                averages.append(avg * weight)
            
            # Ultimate Oscillator
            uo = sum(averages) / sum(weights)
            
            results["ultimate_oscillator"] = float(uo.iloc[-1]) if not pd.isna(uo.iloc[-1]) else 50
            
            # Conditions
            if results["ultimate_oscillator"] < 30:
                results["ultimate_signal"] = "oversold"
            elif results["ultimate_oscillator"] > 70:
                results["ultimate_signal"] = "overbought"
            else:
                results["ultimate_signal"] = "neutral"
        
        return results
    
    def calculate_momentum_score(self, indicators: Dict) -> float:
        """
        Calculate overall momentum score (0-1)
        """
        scores = []
        
        # RSI contribution
        if "rsi" in indicators:
            rsi = indicators["rsi"]
            # Convert RSI (0-100) to score (0-1)
            rsi_score = rsi / 100
            scores.append(rsi_score)
        
        # Stochastic contribution
        if "stoch_k" in indicators:
            stoch_score = indicators["stoch_k"] / 100
            scores.append(stoch_score)
        
        # CCI contribution (normalized)
        if "cci" in indicators:
            cci = indicators["cci"]
            # Normalize CCI (-200 to +200) to 0-1
            cci_score = (cci + 200) / 400
            cci_score = max(0, min(1, cci_score))
            scores.append(cci_score)
        
        # ROC contributions
        for period in [5, 10, 20]:
            roc_key = f"roc_{period}"
            if roc_key in indicators:
                # Normalize ROC (-20% to +20%) to 0-1
                roc = indicators[roc_key]
                roc_score = (roc + 20) / 40
                roc_score = max(0, min(1, roc_score))
                scores.append(roc_score)
        
        if scores:
            return float(np.mean(scores))
        
        return 0.5
    
    def generate_signals(self, indicators: Dict) -> List[Dict]:
        """
        Generate trading signals based on momentum indicators
        """
        signals = []
        
        # RSI signals
        if indicators.get("rsi_oversold"):
            signals.append({
                "indicator": "rsi",
                "signal": "bullish",
                "strength": 0.7,
                "description": f"RSI oversold ({indicators['rsi']:.1f}) - potential bounce"
            })
        elif indicators.get("rsi_overbought"):
            signals.append({
                "indicator": "rsi",
                "signal": "bearish",
                "strength": 0.7,
                "description": f"RSI overbought ({indicators['rsi']:.1f}) - potential pullback"
            })
        
        if indicators.get("rsi_bullish_divergence"):
            signals.append({
                "indicator": "rsi",
                "signal": "bullish",
                "strength": 0.8,
                "description": "Bullish RSI divergence - price making lower lows, RSI higher lows"
            })
        
        if indicators.get("rsi_bearish_divergence"):
            signals.append({
                "indicator": "rsi",
                "signal": "bearish",
                "strength": 0.8,
                "description": "Bearish RSI divergence - price making higher highs, RSI lower highs"
            })
        
        # Stochastic signals
        if indicators.get("stoch_bullish_cross"):
            signals.append({
                "indicator": "stochastic",
                "signal": "bullish",
                "strength": 0.6,
                "description": "Stochastic bullish crossover"
            })
        
        if indicators.get("stoch_bearish_cross"):
            signals.append({
                "indicator": "stochastic",
                "signal": "bearish",
                "strength": 0.6,
                "description": "Stochastic bearish crossover"
            })
        
        if indicators.get("stoch_oversold"):
            signals.append({
                "indicator": "stochastic",
                "signal": "bullish",
                "strength": 0.5,
                "description": f"Stochastic oversold ({indicators['stoch_k']:.1f})"
            })
        
        # Williams %R signals
        if indicators.get("williams_oversold"):
            signals.append({
                "indicator": "williams_r",
                "signal": "bullish",
                "strength": 0.6,
                "description": f"Williams %R oversold ({indicators['williams_r']:.1f})"
            })
        elif indicators.get("williams_overbought"):
            signals.append({
                "indicator": "williams_r",
                "signal": "bearish",
                "strength": 0.6,
                "description": f"Williams %R overbought ({indicators['williams_r']:.1f})"
            })
        
        # CCI signals
        if indicators.get("cci_bullish"):
            signals.append({
                "indicator": "cci",
                "signal": "bullish",
                "strength": 0.6,
                "description": "CCI moving out of oversold territory"
            })
        
        if indicators.get("cci_bearish"):
            signals.append({
                "indicator": "cci",
                "signal": "bearish",
                "strength": 0.6,
                "description": "CCI moving out of overbought territory"
            })
        
        # Ultimate Oscillator signals
        if indicators.get("ultimate_signal") == "oversold":
            signals.append({
                "indicator": "ultimate_oscillator",
                "signal": "bullish",
                "strength": 0.5,
                "description": "Ultimate Oscillator oversold"
            })
        elif indicators.get("ultimate_signal") == "overbought":
            signals.append({
                "indicator": "ultimate_oscillator",
                "signal": "bearish",
                "strength": 0.5,
                "description": "Ultimate Oscillator overbought"
            })
        
        return signals
    
    def _check_bullish_divergence(self, data: pd.DataFrame, rsi: pd.Series) -> bool:
        """Check for bullish divergence (price makes lower low, RSI makes higher low)"""
        if len(data) < 20:
            return False
        
        # Find recent lows
        price_lows = data['Low'].iloc[-20:].values
        rsi_values = rsi.iloc[-20:].values
        
        price_min_idx = np.argmin(price_lows)
        rsi_min_idx = np.argmin(rsi_values)
        
        # Check if price made lower low but RSI made higher low
        if price_min_idx > rsi_min_idx:  # Price low came after RSI low
            if price_lows[price_min_idx] < price_lows[rsi_min_idx]:
                if rsi_values[price_min_idx] > rsi_values[rsi_min_idx]:
                    return True
        
        return False
    
    def _check_bearish_divergence(self, data: pd.DataFrame, rsi: pd.Series) -> bool:
        """Check for bearish divergence (price makes higher high, RSI makes lower high)"""
        if len(data) < 20:
            return False
        
        # Find recent highs
        price_highs = data['High'].iloc[-20:].values
        rsi_values = rsi.iloc[-20:].values
        
        price_max_idx = np.argmax(price_highs)
        rsi_max_idx = np.argmax(rsi_values)
        
        # Check if price made higher high but RSI made lower high
        if price_max_idx > rsi_max_idx:  # Price high came after RSI high
            if price_highs[price_max_idx] > price_highs[rsi_max_idx]:
                if rsi_values[price_max_idx] < rsi_values[rsi_max_idx]:
                    return True
        
        return False