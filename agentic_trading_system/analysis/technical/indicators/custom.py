"""
Custom Indicators - Composite and proprietary indicators
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from agentic_trading_system.utils.logger import logger as logging

class CustomIndicators:
    """
    Custom and composite indicators
    
    Indicators:
    - Composite Trend Score
    - Composite Momentum Score
    - Composite Volume Score
    - Volatility-Adjusted Momentum
    - Market Regime Indicator
    - Signal Quality Score
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Import other indicator classes
        from analysis.technical.indicators.trend import TrendIndicators
        from analysis.technical.indicators.momentum import MomentumIndicators
        from analysis.technical.indicators.volume import VolumeIndicators
        from analysis.technical.indicators.volatility import VolatilityIndicators
        
        self.trend = TrendIndicators(config)
        self.momentum = MomentumIndicators(config)
        self.volume = VolumeIndicators(config)
        self.volatility = VolatilityIndicators(config)
        
        # Weights for composite scores
        self.composite_weights = config.get("composite_weights", {
            "trend": 0.3,
            "momentum": 0.25,
            "volume": 0.2,
            "volatility": 0.15,
            "pattern": 0.1
        })
        
        logging.info(f"✅ CustomIndicators initialized")
    
    def calculate_all(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all custom indicators
        """
        results = {}
        
        # Calculate base indicators
        trend_results = self.trend.calculate_all(data)
        momentum_results = self.momentum.calculate_all(data)
        volume_results = self.volume.calculate_all(data)
        volatility_results = self.volatility.calculate_all(data)
        
        # Composite Scores
        results["composite_trend_score"] = self.calculate_composite_trend_score(trend_results)
        results["composite_momentum_score"] = self.calculate_composite_momentum_score(momentum_results)
        results["composite_volume_score"] = self.calculate_composite_volume_score(volume_results)
        results["composite_volatility_score"] = self.calculate_composite_volatility_score(volatility_results)
        
        # Overall Composite Score
        results["composite_score"] = self.calculate_composite_score(results)
        
        # Volatility-Adjusted Momentum
        results["vol_adjusted_momentum"] = self.calculate_vol_adjusted_momentum(
            results["composite_momentum_score"],
            volatility_results.get("volatility_regime", {}).get("score", 0.5)
        )
        
        # Market Regime
        results["market_regime"] = self.determine_market_regime(
            trend_results,
            momentum_results,
            volatility_results
        )
        
        # Signal Quality
        results["signal_quality"] = self.calculate_signal_quality(results)
        
        # Buy/Sell Pressure
        results["buy_pressure"], results["sell_pressure"] = self.calculate_pressure(
            volume_results,
            momentum_results
        )
        
        # Divergence Score
        results["divergence_score"] = self.calculate_divergence_score(
            trend_results,
            momentum_results,
            volume_results
        )
        
        # All Signals
        all_signals = (
            trend_results.get("signals", []) +
            momentum_results.get("signals", []) +
            volume_results.get("signals", []) +
            volatility_results.get("signals", [])
        )
        results["signals"] = all_signals
        
        # Top Signals (highest strength)
        if all_signals:
            results["top_signals"] = sorted(
                all_signals,
                key=lambda x: x.get("strength", 0),
                reverse=True
            )[:5]
        
        return results
    
    def calculate_composite_trend_score(self, trend_results: Dict) -> float:
        """
        Calculate composite trend score (0-1)
        """
        scores = []
        
        # Moving average alignment
        ma_signals = 0
        if trend_results.get("golden_cross"):
            scores.append(0.9)
        elif trend_results.get("death_cross"):
            scores.append(0.1)
        
        # Price vs MAs
        price_vs_ma50 = trend_results.get("price_vs_sma_50", 0)
        if price_vs_ma50 > 0.05:
            scores.append(0.7)
        elif price_vs_ma50 > 0:
            scores.append(0.6)
        elif price_vs_ma50 > -0.05:
            scores.append(0.4)
        else:
            scores.append(0.3)
        
        # ADX trend strength
        adx = trend_results.get("adx", 0)
        if adx > 25:
            if trend_results.get("adx_trend_direction") == "up":
                scores.append(0.8)
            else:
                scores.append(0.2)
        elif adx > 20:
            scores.append(0.6)
        else:
            scores.append(0.4)
        
        # MACD
        if trend_results.get("macd_bullish_cross"):
            scores.append(0.8)
        elif trend_results.get("macd_histogram", 0) > 0:
            scores.append(0.6)
        elif trend_results.get("macd_bearish_cross"):
            scores.append(0.2)
        else:
            scores.append(0.5)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def calculate_composite_momentum_score(self, momentum_results: Dict) -> float:
        """
        Calculate composite momentum score (0-1)
        """
        scores = []
        
        # RSI
        rsi = momentum_results.get("rsi", 50)
        if rsi < 30:
            scores.append(0.8)  # Oversold - bullish
        elif rsi < 40:
            scores.append(0.7)
        elif rsi < 60:
            scores.append(0.5)
        elif rsi < 70:
            scores.append(0.4)
        else:
            scores.append(0.2)  # Overbought - bearish
        
        # Stochastic
        stoch_k = momentum_results.get("stoch_k", 50)
        if stoch_k < 20:
            scores.append(0.8)
        elif stoch_k < 40:
            scores.append(0.6)
        elif stoch_k < 60:
            scores.append(0.5)
        elif stoch_k < 80:
            scores.append(0.4)
        else:
            scores.append(0.2)
        
        # Stochastic crossovers
        if momentum_results.get("stoch_bullish_cross"):
            scores.append(0.7)
        elif momentum_results.get("stoch_bearish_cross"):
            scores.append(0.3)
        
        # ROC
        roc_10 = momentum_results.get("roc_10", 0)
        if roc_10 > 5:
            scores.append(0.8)
        elif roc_10 > 2:
            scores.append(0.7)
        elif roc_10 > -2:
            scores.append(0.5)
        elif roc_10 > -5:
            scores.append(0.3)
        else:
            scores.append(0.2)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def calculate_composite_volume_score(self, volume_results: Dict) -> float:
        """
        Calculate composite volume score (0-1)
        """
        scores = []
        
        # Volume ratio
        for period in [5, 10, 20]:
            ratio_key = f"volume_ratio_{period}"
            if ratio_key in volume_results:
                ratio = volume_results[ratio_key]
                if ratio > 2:
                    scores.append(0.8)
                elif ratio > 1.5:
                    scores.append(0.7)
                elif ratio > 1:
                    scores.append(0.6)
                else:
                    scores.append(0.4)
        
        # OBV trend
        obv_trend = volume_results.get("obv_trend")
        if obv_trend == "up":
            scores.append(0.7)
        elif obv_trend == "down":
            scores.append(0.3)
        
        # MFI
        mfi = volume_results.get("mfi", 50)
        if mfi < 20:
            scores.append(0.8)
        elif mfi > 80:
            scores.append(0.2)
        elif mfi < 40:
            scores.append(0.6)
        elif mfi > 60:
            scores.append(0.4)
        else:
            scores.append(0.5)
        
        # CMF
        cmf = volume_results.get("cmf", 0)
        if cmf > 0.1:
            scores.append(0.7)
        elif cmf > 0:
            scores.append(0.6)
        elif cmf > -0.1:
            scores.append(0.4)
        else:
            scores.append(0.3)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def calculate_composite_volatility_score(self, volatility_results: Dict) -> float:
        """
        Calculate composite volatility score (lower is better for entry)
        """
        scores = []
        
        # Bollinger width (narrower is better)
        bb_width = volatility_results.get("bb_width_pct", 5)
        if bb_width < 2:
            scores.append(0.8)  # Very narrow - good entry
        elif bb_width < 3.5:
            scores.append(0.7)
        elif bb_width < 5:
            scores.append(0.5)
        elif bb_width < 7:
            scores.append(0.3)
        else:
            scores.append(0.1)  # Very wide - avoid entry
        
        # ATR percent (lower is better)
        atr_pct = volatility_results.get("atr_pct", 3)
        if atr_pct < 1.5:
            scores.append(0.8)
        elif atr_pct < 2.5:
            scores.append(0.7)
        elif atr_pct < 4:
            scores.append(0.5)
        elif atr_pct < 6:
            scores.append(0.3)
        else:
            scores.append(0.1)
        
        # Volatility ratio (near 1 is normal)
        vol_ratio = volatility_results.get("volatility_ratio", 1)
        if 0.8 <= vol_ratio <= 1.2:
            scores.append(0.7)
        elif 0.6 <= vol_ratio <= 1.4:
            scores.append(0.5)
        else:
            scores.append(0.3)
        
        return float(np.mean(scores)) if scores else 0.5
    
    def calculate_composite_score(self, results: Dict) -> float:
        """
        Calculate overall composite score (0-1)
        """
        weighted_sum = 0
        total_weight = 0
        
        components = [
            ("composite_trend_score", self.composite_weights["trend"]),
            ("composite_momentum_score", self.composite_weights["momentum"]),
            ("composite_volume_score", self.composite_weights["volume"]),
            ("composite_volatility_score", self.composite_weights["volatility"])
        ]
        
        for key, weight in components:
            if key in results:
                weighted_sum += results[key] * weight
                total_weight += weight
        
        if total_weight > 0:
            return float(weighted_sum / total_weight)
        
        return 0.5
    
    def calculate_vol_adjusted_momentum(self, momentum_score: float, 
                                       volatility_score: float) -> float:
        """
        Adjust momentum score by volatility
        """
        # Higher volatility reduces confidence in momentum
        vol_adjustment = 1 - (volatility_score * 0.3)
        adjusted = momentum_score * vol_adjustment
        
        return float(min(1.0, max(0.0, adjusted)))
    
    def determine_market_regime(self, trend: Dict, momentum: Dict, 
                               volatility: Dict) -> Dict[str, Any]:
        """
        Determine market regime from indicators
        """
        regime = {
            "type": "unknown",
            "strength": 0.5,
            "description": ""
        }
        
        # Get trend direction
        trend_direction = "neutral"
        if trend.get("adx_trend_direction") == "up":
            trend_direction = "bullish"
        elif trend.get("adx_trend_direction") == "down":
            trend_direction = "bearish"
        
        # Get trend strength
        trend_strength = "weak"
        adx = trend.get("adx", 0)
        if adx > 25:
            trend_strength = "strong"
        elif adx > 20:
            trend_strength = "moderate"
        
        # Get volatility regime
        vol_regime = volatility.get("volatility_regime", {}).get("level", "normal")
        
        # Determine market regime
        if trend_strength == "strong":
            if trend_direction == "bullish":
                if vol_regime in ["low", "normal"]:
                    regime["type"] = "strong_bull_trend"
                    regime["strength"] = 0.9
                    regime["description"] = "Strong bull trend with normal volatility"
                else:
                    regime["type"] = "bull_trend_high_vol"
                    regime["strength"] = 0.7
                    regime["description"] = "Bull trend with high volatility - cautious"
            elif trend_direction == "bearish":
                if vol_regime in ["low", "normal"]:
                    regime["type"] = "strong_bear_trend"
                    regime["strength"] = 0.2
                    regime["description"] = "Strong bear trend"
                else:
                    regime["type"] = "bear_trend_high_vol"
                    regime["strength"] = 0.3
                    regime["description"] = "Bear trend with high volatility"
        else:
            if vol_regime == "high":
                regime["type"] = "high_volatility_ranging"
                regime["strength"] = 0.4
                regime["description"] = "High volatility ranging market"
            elif vol_regime == "low":
                regime["type"] = "low_volatility_ranging"
                regime["strength"] = 0.6
                regime["description"] = "Low volatility ranging market - potential breakout"
            else:
                regime["type"] = "ranging"
                regime["strength"] = 0.5
                regime["description"] = "Ranging market"
        
        return regime
    
    def calculate_signal_quality(self, results: Dict) -> Dict[str, Any]:
        """
        Calculate overall signal quality score
        """
        quality = {
            "score": 0.5,
            "factors": {}
        }
        
        # Component agreement (lower std = higher quality)
        scores = [
            results.get("composite_trend_score", 0.5),
            results.get("composite_momentum_score", 0.5),
            results.get("composite_volume_score", 0.5)
        ]
        
        if scores:
            std = np.std(scores)
            agreement = 1 - min(1.0, std * 2)
            quality["factors"]["agreement"] = float(agreement)
        
        # Divergence penalty
        divergence = results.get("divergence_score", 0)
        quality["factors"]["divergence"] = float(1 - divergence)
        
        # Volatility factor (moderate volatility is good)
        vol_score = results.get("composite_volatility_score", 0.5)
        # Invert so moderate volatility (0.5) scores highest
        vol_factor = 1 - abs(vol_score - 0.5) * 2
        quality["factors"]["volatility"] = float(vol_factor)
        
        # Calculate overall quality
        factors = list(quality["factors"].values())
        if factors:
            quality["score"] = float(np.mean(factors))
        
        return quality
    
    def calculate_pressure(self, volume: Dict, momentum: Dict) -> tuple:
        """
        Calculate buy vs sell pressure
        """
        buy_pressure = 0.5
        sell_pressure = 0.5
        
        # Volume indicators
        if volume.get("cmf", 0) > 0:
            buy_pressure += 0.1
        else:
            sell_pressure += 0.1
        
        if volume.get("obv_trend") == "up":
            buy_pressure += 0.1
        elif volume.get("obv_trend") == "down":
            sell_pressure += 0.1
        
        # Momentum indicators
        if momentum.get("rsi", 50) > 60:
            buy_pressure += 0.1
        elif momentum.get("rsi", 50) < 40:
            sell_pressure += 0.1
        
        if momentum.get("stoch_k", 50) > 70:
            buy_pressure += 0.1
        elif momentum.get("stoch_k", 50) < 30:
            sell_pressure += 0.1
        
        # Normalize
        total = buy_pressure + sell_pressure
        if total > 0:
            buy_pressure /= total
            sell_pressure /= total
        
        return float(buy_pressure), float(sell_pressure)
    
    def calculate_divergence_score(self, trend: Dict, momentum: Dict, 
                                  volume: Dict) -> float:
        """
        Calculate divergence score (0-1, higher = more divergence)
        """
        divergences = []
        
        # Price vs RSI
        if momentum.get("rsi_bullish_divergence"):
            divergences.append(0.8)
        elif momentum.get("rsi_bearish_divergence"):
            divergences.append(0.8)
        
        # Price vs OBV
        if volume.get("obv_divergence"):
            if volume.get("obv_trend") == "up" and trend.get("price_vs_sma_50", 0) < 0:
                divergences.append(0.7)
            elif volume.get("obv_trend") == "down" and trend.get("price_vs_sma_50", 0) > 0:
                divergences.append(0.7)
        
        # MACD divergence
        if (trend.get("macd_bullish_cross") and 
            trend.get("price_vs_sma_50", 0) < 0):
            divergences.append(0.6)
        elif (trend.get("macd_bearish_cross") and 
              trend.get("price_vs_sma_50", 0) > 0):
            divergences.append(0.6)
        
        return float(np.mean(divergences)) if divergences else 0.0