"""
Volatility Adjuster - Dynamic thresholds based on market volatility
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from loguru import logger
class VolatilityAdjuster:
    """
    Calculates dynamic thresholds based on volatility
    Instead of fixed 2%, threshold = f(current_volatility)
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.base_threshold = config.get("base_threshold", 2.0)
        self.volatility_multiplier = config.get("volatility_multiplier", 1.5)
        self.atr_period = config.get("atr_period", 14)
        self.volatility_lookback = config.get("volatility_lookback", 60)
        
        logger.info("VolatilityAdjuster initialized")
    
    def calculate(self, data: pd.DataFrame) -> Dict:
        """
        Calculate volatility-adjusted metrics
        """
        if data.empty or len(data) < 20:
            return {
                "current_volatility": 0,
                "historical_volatility": 0,
                "dynamic_threshold": self.base_threshold,
                "atr": 0,
                "atr_pct": 0
            }
        
        # Calculate returns
        returns = data['Close'].pct_change().dropna() * 100
        
        # Current volatility (last 20 days)
        current_vol = returns.iloc[-20:].std() if len(returns) >= 20 else returns.std()
        
        # Historical volatility (full window)
        historical_vol = returns.iloc[-self.volatility_lookback:].std() if len(returns) >= self.volatility_lookback else returns.std()
        
        # Dynamic threshold
        dynamic_threshold = max(
            self.base_threshold,
            current_vol * self.volatility_multiplier
        )
        
        # ATR (Average True Range)
        atr = self._calculate_atr(data)
        atr_pct = (atr / data['Close'].iloc[-1]) * 100 if atr and data['Close'].iloc[-1] > 0 else 0
        
        # Bollinger Band width
        bb_width = self._calculate_bb_width(data)
        
        # Volatility percentile
        vol_percentile = self._calculate_volatility_percentile(returns)
        
        return {
            "current_volatility": float(current_vol),
            "historical_volatility": float(historical_vol),
            "dynamic_threshold": float(dynamic_threshold),
            "atr": float(atr),
            "atr_pct": float(atr_pct),
            "bb_width": float(bb_width),
            "volatility_percentile": float(vol_percentile),
            "is_high_volatility": current_vol > historical_vol * 1.5,
            "is_low_volatility": current_vol < historical_vol * 0.5,
            "threshold_multiplier": float(current_vol / historical_vol) if historical_vol > 0 else 1.0
        }
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = None) -> float:
        """Calculate Average True Range"""
        if period is None:
            period = self.atr_period
        
        if len(data) < period + 1:
            return 0
        
        high = data['High']
        low = data['Low']
        close = data['Close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return float(atr) if not pd.isna(atr) else 0
    
    def _calculate_bb_width(self, data: pd.DataFrame, period: int = 20) -> float:
        """Calculate Bollinger Band width"""
        if len(data) < period:
            return 0
        
        close = data['Close']
        
        # Calculate Bollinger Bands
        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        
        # Band width as percentage of price
        current_price = close.iloc[-1]
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        
        if current_price == 0 or pd.isna(current_upper) or pd.isna(current_lower):
            return 0
        
        band_width = (current_upper - current_lower) / current_price * 100
        
        return float(band_width)
    
    def _calculate_volatility_percentile(self, returns: pd.Series) -> float:
        """Calculate where current volatility ranks historically"""
        if len(returns) < 60:
            return 50
        
        # Current volatility (20 days)
        current_vol = returns.iloc[-20:].std()
        
        # Historical volatilities
        hist_vols = []
        for i in range(0, len(returns) - 20, 5):  # Step every 5 days
            window = returns.iloc[i:i+20]
            if len(window) == 20:
                hist_vols.append(window.std())
        
        if not hist_vols:
            return 50
        
        # Calculate percentile
        count_less = sum(1 for v in hist_vols if v < current_vol)
        percentile = (count_less / len(hist_vols)) * 100
        
        return float(percentile)
    
    def get_position_size_multiplier(self, data: pd.DataFrame) -> float:
        """
        Get position size multiplier based on volatility
        Higher volatility = smaller position
        """
        metrics = self.calculate(data)
        
        # Base multiplier
        if metrics["is_high_volatility"]:
            return 0.5  # Half size
        elif metrics["is_low_volatility"]:
            return 1.5  # 1.5x size
        else:
            return 1.0  # Normal size
    
    def get_stop_loss_distance(self, data: pd.DataFrame) -> float:
        """
        Get recommended stop loss distance based on ATR
        """
        metrics = self.calculate(data)
        
        # Use ATR * multiplier
        atr_multiplier = self.config.get("stop_loss_atr_multiplier", 1.5)
        stop_distance = metrics["atr_pct"] * atr_multiplier
        
        return float(stop_distance)