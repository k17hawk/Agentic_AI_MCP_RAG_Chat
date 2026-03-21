"""
Volume Indicators - OBV, MFI, VWAP, etc.
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from agentic_trading_system.utils.logger import logger as  logging

class VolumeIndicators:
    """
    Comprehensive volume analysis indicators
    
    Indicators:
    - OBV (On-Balance Volume)
    - MFI (Money Flow Index)
    - VWAP (Volume Weighted Average Price)
    - Volume Profile
    - Accumulation/Distribution Line
    - Chaikin Money Flow
    - Volume Ratio
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default periods
        self.mfi_period = config.get("mfi_period", 14)
        self.cmf_period = config.get("cmf_period", 20)
        self.volume_ma_periods = config.get("volume_ma_periods", [5, 10, 20, 50])
        
        # Thresholds
        self.volume_spike_threshold = config.get("volume_spike_threshold", 2.0)
        self.mfi_overbought = config.get("mfi_overbought", 80)
        self.mfi_oversold = config.get("mfi_oversold", 20)
        
        logging.info(f"✅ VolumeIndicators initialized")
    
    def calculate_all(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all volume indicators
        """
        results = {}
        
        # OBV
        results.update(self.calculate_obv(data))
        
        # MFI
        results.update(self.calculate_mfi(data))
        
        # VWAP
        results.update(self.calculate_vwap(data))
        
        # Volume Profile
        results.update(self.calculate_volume_profile(data))
        
        # Accumulation/Distribution
        results.update(self.calculate_ad_line(data))
        
        # Chaikin Money Flow
        results.update(self.calculate_cmf(data))
        
        # Volume Ratios and Moving Averages
        results.update(self.calculate_volume_metrics(data))
        
        # Signal Generation
        results["signals"] = self.generate_signals(results)
        
        return results
    
    def calculate_obv(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate On-Balance Volume (OBV)
        """
        close = data['Close']
        volume = data['Volume']
        results = {}
        
        if len(close) > 1:
            # Calculate OBV
            obv = np.zeros(len(close))
            
            for i in range(1, len(close)):
                if close.iloc[i] > close.iloc[i-1]:
                    obv[i] = obv[i-1] + volume.iloc[i]
                elif close.iloc[i] < close.iloc[i-1]:
                    obv[i] = obv[i-1] - volume.iloc[i]
                else:
                    obv[i] = obv[i-1]
            
            obv_series = pd.Series(obv, index=close.index)
            
            results["obv"] = float(obv_series.iloc[-1])
            results["obv_slope"] = self._calculate_slope(obv_series, 5)
            results["obv_trend"] = "up" if results["obv_slope"] > 0 else "down"
            
            # OBV vs Price
            price_slope = self._calculate_slope(close, 5)
            results["obv_divergence"] = bool(results["obv_slope"] * price_slope < 0)
            
            # OBV Moving Average
            obv_ma = pd.Series(obv).rolling(20).mean()
            results["obv_ma"] = float(obv_ma.iloc[-1]) if not pd.isna(obv_ma.iloc[-1]) else results["obv"]
            results["obv_vs_ma"] = float(results["obv"] - results["obv_ma"])
        
        return results
    
    def calculate_mfi(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Money Flow Index (MFI)
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        period = self.mfi_period
        results = {}
        
        if len(data) > period:
            # Typical Price
            tp = (high + low + close) / 3
            
            # Raw Money Flow
            rmf = tp * volume
            
            # Positive and Negative Money Flow
            pos_flow = []
            neg_flow = []
            
            for i in range(1, len(tp)):
                if tp.iloc[i] > tp.iloc[i-1]:
                    pos_flow.append(rmf.iloc[i])
                    neg_flow.append(0)
                elif tp.iloc[i] < tp.iloc[i-1]:
                    pos_flow.append(0)
                    neg_flow.append(rmf.iloc[i])
                else:
                    pos_flow.append(0)
                    neg_flow.append(0)
            
            pos_flow = pd.Series(pos_flow, index=tp.index[1:])
            neg_flow = pd.Series(neg_flow, index=tp.index[1:])
            
            # Money Flow Ratio
            pos_sum = pos_flow.rolling(window=period).sum()
            neg_sum = neg_flow.rolling(window=period).sum()
            
            mfr = pos_sum / neg_sum
            
            # MFI
            mfi = 100 - (100 / (1 + mfr))
            
            current_mfi = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50
            prev_mfi = mfi.iloc[-2] if len(mfi) > 1 and not pd.isna(mfi.iloc[-2]) else 50
            
            results["mfi"] = float(current_mfi)
            results["mfi_slope"] = float(current_mfi - prev_mfi)
            
            # Conditions
            results["mfi_oversold"] = bool(current_mfi < self.mfi_oversold)
            results["mfi_overbought"] = bool(current_mfi > self.mfi_overbought)
            
            # Divergence
            price_slope = self._calculate_slope(close, period)
            results["mfi_divergence"] = bool(results["mfi_slope"] * price_slope < 0)
        
        return results
    
    def calculate_vwap(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Volume Weighted Average Price (VWAP)
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        results = {}
        
        # Typical Price
        tp = (high + low + close) / 3
        
        # VWAP
        vwap = (tp * volume).cumsum() / volume.cumsum()
        
        current_vwap = vwap.iloc[-1] if not pd.isna(vwap.iloc[-1]) else close.iloc[-1]
        current_price = close.iloc[-1]
        
        results["vwap"] = float(current_vwap)
        results["price_vs_vwap"] = float((current_price - current_vwap) / current_vwap * 100)
        results["above_vwap"] = bool(current_price > current_vwap)
        results["vwap_slope"] = self._calculate_slope(vwap, 5)
        
        # VWAP bands (standard deviation)
        vwap_std = (tp * volume).rolling(20).std() / volume.rolling(20).mean()
        results["vwap_upper"] = float(current_vwap + vwap_std.iloc[-1] * 2) if not pd.isna(vwap_std.iloc[-1]) else current_vwap * 1.02
        results["vwap_lower"] = float(current_vwap - vwap_std.iloc[-1] * 2) if not pd.isna(vwap_std.iloc[-1]) else current_vwap * 0.98
        
        return results
    
    def calculate_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Volume Profile (price levels with highest volume)
        """
        close = data['Close']
        volume = data['Volume']
        results = {}
        
        if len(data) > 20:
            # Create price bins
            price_min = close.min()
            price_max = close.max()
            price_range = price_max - price_min
            
            if price_range > 0:
                # Simple volume profile with 10 bins
                bins = 10
                bin_size = price_range / bins
                
                volume_profile = {}
                for i in range(bins):
                    lower = price_min + i * bin_size
                    upper = lower + bin_size
                    
                    mask = (close >= lower) & (close < upper)
                    vol_sum = volume[mask].sum()
                    volume_profile[f"{lower:.2f}-{upper:.2f}"] = float(vol_sum)
                
                # Find high volume nodes
                max_vol = max(volume_profile.values())
                high_volume_nodes = [k for k, v in volume_profile.items() if v > max_vol * 0.8]
                
                results["volume_profile"] = volume_profile
                results["high_volume_nodes"] = high_volume_nodes
                results["value_area_low"] = self._find_value_area_low(volume_profile, close.iloc[-1])
                results["value_area_high"] = self._find_value_area_high(volume_profile, close.iloc[-1])
        
        return results
    
    def calculate_ad_line(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Accumulation/Distribution Line
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        results = {}
        
        if len(data) > 1:
            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            
            # Money Flow Volume
            mfv = mfm * volume
            
            # Accumulation/Distribution Line
            ad_line = mfv.cumsum()
            
            results["ad_line"] = float(ad_line.iloc[-1])
            results["ad_line_slope"] = self._calculate_slope(ad_line, 5)
            
            # Divergence
            price_slope = self._calculate_slope(close, 5)
            results["ad_divergence"] = bool(results["ad_line_slope"] * price_slope < 0)
        
        return results
    
    def calculate_cmf(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate Chaikin Money Flow
        """
        high = data['High']
        low = data['Low']
        close = data['Close']
        volume = data['Volume']
        period = self.cmf_period
        results = {}
        
        if len(data) > period:
            # Money Flow Multiplier
            mfm = ((close - low) - (high - close)) / (high - low)
            
            # Money Flow Volume
            mfv = mfm * volume
            
            # Chaikin Money Flow
            cmf = mfv.rolling(window=period).sum() / volume.rolling(window=period).sum()
            
            results["cmf"] = float(cmf.iloc[-1]) if not pd.isna(cmf.iloc[-1]) else 0
            results["cmf_slope"] = self._calculate_slope(cmf, 5)
            
            # Conditions
            results["cmf_positive"] = bool(results["cmf"] > 0)
            results["cmf_strong_buy"] = bool(results["cmf"] > 0.1)
            results["cmf_strong_sell"] = bool(results["cmf"] < -0.1)
        
        return results
    
    def calculate_volume_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate volume ratios and moving averages
        """
        volume = data['Volume']
        results = {}
        
        if len(volume) > 50:
            # Volume moving averages
            for period in self.volume_ma_periods:
                vol_ma = volume.rolling(period).mean()
                results[f"volume_ma_{period}"] = float(vol_ma.iloc[-1]) if not pd.isna(vol_ma.iloc[-1]) else 0
            
            # Current volume
            current_volume = volume.iloc[-1]
            results["current_volume"] = float(current_volume)
            
            # Volume ratios
            for period in self.volume_ma_periods[:3]:  # Use shorter periods for ratios
                ma_key = f"volume_ma_{period}"
                if ma_key in results and results[ma_key] > 0:
                    ratio = current_volume / results[ma_key]
                    results[f"volume_ratio_{period}"] = float(ratio)
                    
                    # Volume spike detection
                    if ratio > self.volume_spike_threshold:
                        results[f"volume_spike_{period}"] = True
            
            # Volume trend
            vol_trend = self._calculate_slope(volume, 10)
            results["volume_trend"] = "increasing" if vol_trend > 0 else "decreasing"
            results["volume_trend_strength"] = float(abs(vol_trend))
        
        return results
    
    def generate_signals(self, indicators: Dict) -> List[Dict]:
        """
        Generate trading signals based on volume indicators
        """
        signals = []
        
        # OBV signals
        if indicators.get("obv_trend") == "up":
            signals.append({
                "indicator": "obv",
                "signal": "bullish",
                "strength": 0.6,
                "description": "OBV trending up - volume confirming price"
            })
        
        if indicators.get("obv_divergence"):
            if indicators.get("obv_slope", 0) > 0:
                signals.append({
                    "indicator": "obv",
                    "signal": "bullish",
                    "strength": 0.8,
                    "description": "Bullish OBV divergence - price down, volume up"
                })
            else:
                signals.append({
                    "indicator": "obv",
                    "signal": "bearish",
                    "strength": 0.8,
                    "description": "Bearish OBV divergence - price up, volume down"
                })
        
        # MFI signals
        if indicators.get("mfi_oversold"):
            signals.append({
                "indicator": "mfi",
                "signal": "bullish",
                "strength": 0.7,
                "description": f"MFI oversold ({indicators['mfi']:.1f})"
            })
        elif indicators.get("mfi_overbought"):
            signals.append({
                "indicator": "mfi",
                "signal": "bearish",
                "strength": 0.7,
                "description": f"MFI overbought ({indicators['mfi']:.1f})"
            })
        
        # Volume spike signals
        for period in [5, 10, 20]:
            spike_key = f"volume_spike_{period}"
            if indicators.get(spike_key):
                ratio_key = f"volume_ratio_{period}"
                ratio = indicators.get(ratio_key, 0)
                signals.append({
                    "indicator": "volume",
                    "signal": "neutral",
                    "strength": 0.6,
                    "description": f"Volume spike ({ratio:.1f}x {period}-day average)"
                })
        
        # VWAP signals
        if indicators.get("above_vwap"):
            signals.append({
                "indicator": "vwap",
                "signal": "bullish",
                "strength": 0.5,
                "description": f"Price above VWAP ({indicators['price_vs_vwap']:.2f}%)"
            })
        else:
            signals.append({
                "indicator": "vwap",
                "signal": "bearish",
                "strength": 0.5,
                "description": f"Price below VWAP ({indicators['price_vs_vwap']:.2f}%)"
            })
        
        # CMF signals
        if indicators.get("cmf_strong_buy"):
            signals.append({
                "indicator": "cmf",
                "signal": "bullish",
                "strength": 0.7,
                "description": f"Strong Chaikin Money Flow ({indicators['cmf']:.2f})"
            })
        elif indicators.get("cmf_strong_sell"):
            signals.append({
                "indicator": "cmf",
                "signal": "bearish",
                "strength": 0.7,
                "description": f"Strong Chaikin Money Flow negative ({indicators['cmf']:.2f})"
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
    
    def _find_value_area_low(self, volume_profile: Dict, current_price: float) -> float:
        """Find lower bound of value area"""
        # Simplified - find nearest high volume node below current price
        levels = []
        for price_range in volume_profile.keys():
            try:
                lower = float(price_range.split('-')[0])
                if lower < current_price:
                    levels.append(lower)
            except:
                continue
        
        return max(levels) if levels else current_price * 0.95
    
    def _find_value_area_high(self, volume_profile: Dict, current_price: float) -> float:
        """Find upper bound of value area"""
        # Simplified - find nearest high volume node above current price
        levels = []
        for price_range in volume_profile.keys():
            try:
                upper = float(price_range.split('-')[1])
                if upper > current_price:
                    levels.append(upper)
            except:
                continue
        
        return min(levels) if levels else current_price * 1.05