"""
Candlestick Patterns - Detection of Japanese candlestick patterns
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from utils.logger import logger as  logging

class CandlestickPatterns:
    """
    Comprehensive candlestick pattern detection
    
    Patterns:
    - Single Candle: Doji, Hammer, Shooting Star, Marubozu, Spinning Top
    - Two Candle: Engulfing, Harami, Piercing, Dark Cloud, Tweezer
    - Three Candle: Morning Star, Evening Star, Three Soldiers, Three Crows
    - Four+ Candle: Three Inside Up/Down, Three Outside Up/Down
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Pattern confidence thresholds
        self.min_pattern_confidence = config.get("min_pattern_confidence", 0.6)
        
        # Require volume confirmation
        self.require_volume = config.get("require_volume_confirmation", True)
        
        # Pattern weights for scoring
        self.pattern_weights = {
            # Major reversal patterns
            "morning_star": 0.9,
            "evening_star": 0.9,
            "bullish_engulfing": 0.85,
            "bearish_engulfing": 0.85,
            "hammer": 0.8,
            "shooting_star": 0.8,
            
            # Moderate reversal patterns
            "piercing_line": 0.75,
            "dark_cloud_cover": 0.75,
            "bullish_harami": 0.7,
            "bearish_harami": 0.7,
            "doji": 0.6,
            "spinning_top": 0.6,
            
            # Continuation patterns
            "three_white_soldiers": 0.85,
            "three_black_crows": 0.85,
            "rising_three": 0.75,
            "falling_three": 0.75,
            
            # Indecision patterns
            "long_legged_doji": 0.6,
            "dragonfly_doji": 0.65,
            "gravestone_doji": 0.65
        }
        
        logging.info(f"✅ CandlestickPatterns initialized with {len(self.pattern_weights)} patterns")
    
    def detect_all(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect all candlestick patterns in the data
        """
        patterns = []
        
        if data.empty or len(data) < 10:
            return patterns
        
        # Calculate candle properties
        candles = self._calculate_candle_properties(data)
        
        # Detect single candle patterns
        single_patterns = self._detect_single_candle(candles)
        patterns.extend(single_patterns)
        
        # Detect two candle patterns
        two_candle_patterns = self._detect_two_candle(candles)
        patterns.extend(two_candle_patterns)
        
        # Detect three candle patterns
        three_candle_patterns = self._detect_three_candle(candles)
        patterns.extend(three_candle_patterns)
        
        # Detect multi-candle patterns
        multi_patterns = self._detect_multi_candle(candles)
        patterns.extend(multi_patterns)
        
        # Filter by confidence
        patterns = [p for p in patterns if p["confidence"] >= self.min_pattern_confidence]
        
        # Add volume confirmation if required
        if self.require_volume:
            patterns = self._add_volume_confirmation(patterns, data)
        
        logging.info(f"📊 Detected {len(patterns)} candlestick patterns")
        return patterns
    
    def _calculate_candle_properties(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate properties for each candle
        """
        candles = data.copy()
        
        # Basic properties
        candles['body'] = abs(candles['Close'] - candles['Open'])
        candles['upper_shadow'] = candles['High'] - candles[['Close', 'Open']].max(axis=1)
        candles['lower_shadow'] = candles[['Close', 'Open']].min(axis=1) - candles['Low']
        candles['total_range'] = candles['High'] - candles['Low']
        candles['body_percent'] = (candles['body'] / candles['total_range'] * 100).fillna(0)
        candles['is_bullish'] = candles['Close'] > candles['Open']
        candles['is_bearish'] = candles['Close'] < candles['Open']
        
        # Midpoint
        candles['midpoint'] = (candles['High'] + candles['Low']) / 2
        
        # Body center
        candles['body_center'] = (candles['Open'] + candles['Close']) / 2
        
        # Shadow ratios
        candles['upper_shadow_ratio'] = (candles['upper_shadow'] / candles['total_range'] * 100).fillna(0)
        candles['lower_shadow_ratio'] = (candles['lower_shadow'] / candles['total_range'] * 100).fillna(0)
        
        # Relative size compared to recent candles
        candles['avg_body'] = candles['body'].rolling(10).mean()
        candles['body_ratio'] = candles['body'] / candles['avg_body']
        
        return candles
    
    def _detect_single_candle(self, candles: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect single candle patterns
        """
        patterns = []
        last = candles.iloc[-1]
        prev = candles.iloc[-2] if len(candles) > 1 else None
        
        # Doji (open and close very close)
        if last['body_percent'] < 10:
            # Dragonfly Doji (long lower shadow, little or no upper shadow)
            if (last['lower_shadow_ratio'] > 60 and 
                last['upper_shadow_ratio'] < 10):
                patterns.append({
                    "name": "dragonfly_doji",
                    "type": "single",
                    "direction": "bullish",
                    "confidence": 0.65,
                    "description": "Dragonfly Doji - potential bullish reversal after downtrend",
                    "position": "current"
                })
            
            # Gravestone Doji (long upper shadow, little or no lower shadow)
            elif (last['upper_shadow_ratio'] > 60 and 
                  last['lower_shadow_ratio'] < 10):
                patterns.append({
                    "name": "gravestone_doji",
                    "type": "single",
                    "direction": "bearish",
                    "confidence": 0.65,
                    "description": "Gravestone Doji - potential bearish reversal after uptrend",
                    "position": "current"
                })
            
            # Long-legged Doji (both shadows long)
            elif (last['upper_shadow_ratio'] > 30 and 
                  last['lower_shadow_ratio'] > 30):
                patterns.append({
                    "name": "long_legged_doji",
                    "type": "single",
                    "direction": "neutral",
                    "confidence": 0.6,
                    "description": "Long-legged Doji - market indecision",
                    "position": "current"
                })
            
            # Standard Doji
            else:
                patterns.append({
                    "name": "doji",
                    "type": "single",
                    "direction": "neutral",
                    "confidence": 0.6,
                    "description": "Doji - market indecision",
                    "position": "current"
                })
        
        # Hammer (small body, long lower shadow, little upper shadow)
        if (last['lower_shadow_ratio'] > 60 and 
            last['upper_shadow_ratio'] < 10 and
            last['body_percent'] < 40):
            
            # Determine if it's a hammer (after downtrend) or hanging man (after uptrend)
            if prev and prev['Close'] < prev['Open']:  # Previous bearish
                patterns.append({
                    "name": "hammer",
                    "type": "single",
                    "direction": "bullish",
                    "confidence": 0.8,
                    "description": "Hammer - bullish reversal after downtrend",
                    "position": "current"
                })
            else:
                patterns.append({
                    "name": "hanging_man",
                    "type": "single",
                    "direction": "bearish",
                    "confidence": 0.75,
                    "description": "Hanging Man - potential bearish reversal after uptrend",
                    "position": "current"
                })
        
        # Shooting Star (small body, long upper shadow, little lower shadow)
        if (last['upper_shadow_ratio'] > 60 and 
            last['lower_shadow_ratio'] < 10 and
            last['body_percent'] < 40 and
            last['is_bullish']):
            
            patterns.append({
                "name": "shooting_star",
                "type": "single",
                "direction": "bearish",
                "confidence": 0.8,
                "description": "Shooting Star - bearish reversal after uptrend",
                "position": "current"
            })
        
        # Marubozu (no shadows, strong trend)
        if last['upper_shadow_ratio'] < 5 and last['lower_shadow_ratio'] < 5:
            if last['is_bullish']:
                patterns.append({
                    "name": "white_marubozu",
                    "type": "single",
                    "direction": "bullish",
                    "confidence": 0.75,
                    "description": "White Marubozu - strong bullish momentum",
                    "position": "current"
                })
            else:
                patterns.append({
                    "name": "black_marubozu",
                    "type": "single",
                    "direction": "bearish",
                    "confidence": 0.75,
                    "description": "Black Marubozu - strong bearish momentum",
                    "position": "current"
                })
        
        # Spinning Top (small body, shadows on both sides)
        if (20 <= last['body_percent'] <= 40 and
            last['upper_shadow_ratio'] > 20 and
            last['lower_shadow_ratio'] > 20):
            
            patterns.append({
                "name": "spinning_top",
                "type": "single",
                "direction": "neutral",
                "confidence": 0.6,
                "description": "Spinning Top - market indecision with small range",
                "position": "current"
            })
        
        return patterns
    
    def _detect_two_candle(self, candles: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect two-candle patterns
        """
        patterns = []
        
        if len(candles) < 2:
            return patterns
        
        current = candles.iloc[-1]
        previous = candles.iloc[-2]
        
        # Bullish Engulfing
        if (previous['is_bearish'] and current['is_bullish'] and
            current['Open'] < previous['Close'] and
            current['Close'] > previous['Open'] and
            current['body'] > previous['body'] * 1.2):  # Body should be significantly larger
            
            patterns.append({
                "name": "bullish_engulfing",
                "type": "two_candle",
                "direction": "bullish",
                "confidence": 0.85,
                "description": "Bullish Engulfing - strong buying pressure reversing downtrend",
                "position": "current"
            })
        
        # Bearish Engulfing
        elif (previous['is_bullish'] and current['is_bearish'] and
              current['Open'] > previous['Close'] and
              current['Close'] < previous['Open'] and
              current['body'] > previous['body'] * 1.2):
            
            patterns.append({
                "name": "bearish_engulfing",
                "type": "two_candle",
                "direction": "bearish",
                "confidence": 0.85,
                "description": "Bearish Engulfing - strong selling pressure reversing uptrend",
                "position": "current"
            })
        
        # Bullish Harami
        elif (previous['is_bearish'] and current['is_bullish'] and
              current['Open'] > previous['Close'] and
              current['Close'] < previous['Open'] and
              current['body'] < previous['body'] * 0.7):  # Body should be significantly smaller
            
            patterns.append({
                "name": "bullish_harami",
                "type": "two_candle",
                "direction": "bullish",
                "confidence": 0.7,
                "description": "Bullish Harami - potential reversal with decreasing momentum",
                "position": "current"
            })
        
        # Bearish Harami
        elif (previous['is_bullish'] and current['is_bearish'] and
              current['Open'] < previous['Close'] and
              current['Close'] > previous['Open'] and
              current['body'] < previous['body'] * 0.7):
            
            patterns.append({
                "name": "bearish_harami",
                "type": "two_candle",
                "direction": "bearish",
                "confidence": 0.7,
                "description": "Bearish Harami - potential reversal with decreasing momentum",
                "position": "current"
            })
        
        # Piercing Line
        elif (previous['is_bearish'] and current['is_bullish'] and
              current['Open'] < previous['Low'] and
              current['Close'] > (previous['Open'] + previous['Close']) / 2 and
              current['Close'] < previous['Open']):
            
            patterns.append({
                "name": "piercing_line",
                "type": "two_candle",
                "direction": "bullish",
                "confidence": 0.75,
                "description": "Piercing Line - bullish reversal signal",
                "position": "current"
            })
        
        # Dark Cloud Cover
        elif (previous['is_bullish'] and current['is_bearish'] and
              current['Open'] > previous['High'] and
              current['Close'] < (previous['Open'] + previous['Close']) / 2 and
              current['Close'] > previous['Open']):
            
            patterns.append({
                "name": "dark_cloud_cover",
                "type": "two_candle",
                "direction": "bearish",
                "confidence": 0.75,
                "description": "Dark Cloud Cover - bearish reversal signal",
                "position": "current"
            })
        
        # Tweezer Bottom
        elif (previous['is_bearish'] and current['is_bullish'] and
              abs(previous['Low'] - current['Low']) / previous['Low'] < 0.01 and  # Same low
              previous['Close'] < previous['Open'] and  # Previous bearish
              current['Close'] > current['Open']):  # Current bullish
            
            patterns.append({
                "name": "tweezer_bottom",
                "type": "two_candle",
                "direction": "bullish",
                "confidence": 0.7,
                "description": "Tweezer Bottom - double bottom reversal",
                "position": "current"
            })
        
        # Tweezer Top
        elif (previous['is_bullish'] and current['is_bearish'] and
              abs(previous['High'] - current['High']) / previous['High'] < 0.01 and  # Same high
              previous['Close'] > previous['Open'] and  # Previous bullish
              current['Close'] < current['Open']):  # Current bearish
            
            patterns.append({
                "name": "tweezer_top",
                "type": "two_candle",
                "direction": "bearish",
                "confidence": 0.7,
                "description": "Tweezer Top - double top reversal",
                "position": "current"
            })
        
        return patterns
    
    def _detect_three_candle(self, candles: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect three-candle patterns
        """
        patterns = []
        
        if len(candles) < 3:
            return patterns
        
        c1 = candles.iloc[-3]  # First candle
        c2 = candles.iloc[-2]  # Middle candle
        c3 = candles.iloc[-1]  # Last candle
        
        # Morning Star (bullish reversal)
        if (c1['is_bearish'] and  # First: long bearish
            abs(c2['body_percent']) < 30 and  # Second: small body (star)
            c3['is_bullish'] and  # Third: bullish
            c3['Close'] > (c1['Open'] + c1['Close']) / 2):  # Closes into first candle
            
            # Check for gap
            has_gap = c2['Low'] > c1['Close'] or c2['High'] < c1['Open']
            confidence = 0.9 if has_gap else 0.8
            
            patterns.append({
                "name": "morning_star",
                "type": "three_candle",
                "direction": "bullish",
                "confidence": confidence,
                "description": "Morning Star - strong bullish reversal after downtrend",
                "position": "current",
                "has_gap": has_gap
            })
        
        # Evening Star (bearish reversal)
        elif (c1['is_bullish'] and  # First: long bullish
              abs(c2['body_percent']) < 30 and  # Second: small body (star)
              c3['is_bearish'] and  # Third: bearish
              c3['Close'] < (c1['Open'] + c1['Close']) / 2):  # Closes into first candle
            
            # Check for gap
            has_gap = c2['Low'] > c1['Close'] or c2['High'] < c1['Open']
            confidence = 0.9 if has_gap else 0.8
            
            patterns.append({
                "name": "evening_star",
                "type": "three_candle",
                "direction": "bearish",
                "confidence": confidence,
                "description": "Evening Star - strong bearish reversal after uptrend",
                "position": "current",
                "has_gap": has_gap
            })
        
        # Three White Soldiers (strong bullish continuation)
        if (c1['is_bullish'] and c2['is_bullish'] and c3['is_bullish'] and
            c1['Close'] > c1['Open'] and
            c2['Close'] > c2['Open'] and
            c3['Close'] > c3['Open'] and
            c2['Close'] > c1['Close'] and
            c3['Close'] > c2['Close'] and
            c2['body'] > c1['body'] * 0.8 and  # Consistent size
            c3['body'] > c2['body'] * 0.8):
            
            patterns.append({
                "name": "three_white_soldiers",
                "type": "three_candle",
                "direction": "bullish",
                "confidence": 0.85,
                "description": "Three White Soldiers - strong bullish continuation",
                "position": "current"
            })
        
        # Three Black Crows (strong bearish continuation)
        elif (c1['is_bearish'] and c2['is_bearish'] and c3['is_bearish'] and
              c1['Close'] < c1['Open'] and
              c2['Close'] < c2['Open'] and
              c3['Close'] < c3['Open'] and
              c2['Close'] < c1['Close'] and
              c3['Close'] < c2['Close'] and
              c2['body'] > c1['body'] * 0.8 and
              c3['body'] > c2['body'] * 0.8):
            
            patterns.append({
                "name": "three_black_crows",
                "type": "three_candle",
                "direction": "bearish",
                "confidence": 0.85,
                "description": "Three Black Crows - strong bearish continuation",
                "position": "current"
            })
        
        # Three Inside Up (bullish)
        if self._is_bullish_harami(c1, c2) and c3['is_bullish'] and c3['Close'] > c2['High']:
            patterns.append({
                "name": "three_inside_up",
                "type": "three_candle",
                "direction": "bullish",
                "confidence": 0.8,
                "description": "Three Inside Up - bullish reversal confirmation",
                "position": "current"
            })
        
        # Three Inside Down (bearish)
        elif self._is_bearish_harami(c1, c2) and c3['is_bearish'] and c3['Close'] < c2['Low']:
            patterns.append({
                "name": "three_inside_down",
                "type": "three_candle",
                "direction": "bearish",
                "confidence": 0.8,
                "description": "Three Inside Down - bearish reversal confirmation",
                "position": "current"
            })
        
        # Three Outside Up (bullish)
        if self._is_bullish_engulfing(c1, c2) and c3['is_bullish'] and c3['Close'] > c2['High']:
            patterns.append({
                "name": "three_outside_up",
                "type": "three_candle",
                "direction": "bullish",
                "confidence": 0.85,
                "description": "Three Outside Up - strong bullish reversal",
                "position": "current"
            })
        
        # Three Outside Down (bearish)
        elif self._is_bearish_engulfing(c1, c2) and c3['is_bearish'] and c3['Close'] < c2['Low']:
            patterns.append({
                "name": "three_outside_down",
                "type": "three_candle",
                "direction": "bearish",
                "confidence": 0.85,
                "description": "Three Outside Down - strong bearish reversal",
                "position": "current"
            })
        
        return patterns
    
    def _detect_multi_candle(self, candles: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect patterns with 4+ candles
        """
        patterns = []
        
        if len(candles) < 5:
            return patterns
        
        # Rising Three Methods (bullish continuation)
        # Long bullish, then 3 small bearish inside range, then long bullish
        if self._detect_rising_three(candles):
            patterns.append({
                "name": "rising_three_methods",
                "type": "multi_candle",
                "direction": "bullish",
                "confidence": 0.75,
                "description": "Rising Three Methods - bullish continuation pattern",
                "position": "current"
            })
        
        # Falling Three Methods (bearish continuation)
        if self._detect_falling_three(candles):
            patterns.append({
                "name": "falling_three_methods",
                "type": "multi_candle",
                "direction": "bearish",
                "confidence": 0.75,
                "description": "Falling Three Methods - bearish continuation pattern",
                "position": "current"
            })
        
        return patterns
    
    def _is_bullish_harami(self, c1: pd.Series, c2: pd.Series) -> bool:
        """Check for bullish harami pattern"""
        return (c1['is_bearish'] and c2['is_bullish'] and
                c2['Open'] > c1['Close'] and
                c2['Close'] < c1['Open'] and
                c2['body'] < c1['body'] * 0.7)
    
    def _is_bearish_harami(self, c1: pd.Series, c2: pd.Series) -> bool:
        """Check for bearish harami pattern"""
        return (c1['is_bullish'] and c2['is_bearish'] and
                c2['Open'] < c1['Close'] and
                c2['Close'] > c1['Open'] and
                c2['body'] < c1['body'] * 0.7)
    
    def _is_bullish_engulfing(self, c1: pd.Series, c2: pd.Series) -> bool:
        """Check for bullish engulfing pattern"""
        return (c1['is_bearish'] and c2['is_bullish'] and
                c2['Open'] < c1['Close'] and
                c2['Close'] > c1['Open'] and
                c2['body'] > c1['body'] * 1.2)
    
    def _is_bearish_engulfing(self, c1: pd.Series, c2: pd.Series) -> bool:
        """Check for bearish engulfing pattern"""
        return (c1['is_bullish'] and c2['is_bearish'] and
                c2['Open'] > c1['Close'] and
                c2['Close'] < c1['Open'] and
                c2['body'] > c1['body'] * 1.2)
    
    def _detect_rising_three(self, candles: pd.DataFrame) -> bool:
        """Detect Rising Three Methods pattern"""
        if len(candles) < 5:
            return False
        
        # Get last 5 candles
        c1 = candles.iloc[-5]
        c2 = candles.iloc[-4]
        c3 = candles.iloc[-3]
        c4 = candles.iloc[-2]
        c5 = candles.iloc[-1]
        
        # First: long bullish
        if not c1['is_bullish'] or c1['body_ratio'] < 1.5:
            return False
        
        # Middle 3: small bearish inside first candle's range
        for c in [c2, c3, c4]:
            if not c['is_bearish']:
                return False
            if c['High'] > c1['High'] or c['Low'] < c1['Low']:
                return False
            if c['body_ratio'] > 0.8:  # Should be smaller
                return False
        
        # Last: long bullish breaking above
        if not c5['is_bullish'] or c5['Close'] <= c1['High']:
            return False
        if c5['body_ratio'] < 1.2:
            return False
        
        return True
    
    def _detect_falling_three(self, candles: pd.DataFrame) -> bool:
        """Detect Falling Three Methods pattern"""
        if len(candles) < 5:
            return False
        
        # Get last 5 candles
        c1 = candles.iloc[-5]
        c2 = candles.iloc[-4]
        c3 = candles.iloc[-3]
        c4 = candles.iloc[-2]
        c5 = candles.iloc[-1]
        
        # First: long bearish
        if not c1['is_bearish'] or c1['body_ratio'] < 1.5:
            return False
        
        # Middle 3: small bullish inside first candle's range
        for c in [c2, c3, c4]:
            if not c['is_bullish']:
                return False
            if c['High'] > c1['High'] or c['Low'] < c1['Low']:
                return False
            if c['body_ratio'] > 0.8:
                return False
        
        # Last: long bearish breaking below
        if not c5['is_bearish'] or c5['Close'] >= c1['Low']:
            return False
        if c5['body_ratio'] < 1.2:
            return False
        
        return True
    
    def _add_volume_confirmation(self, patterns: List[Dict], data: pd.DataFrame) -> List[Dict]:
        """
        Add volume confirmation to patterns
        """
        if len(data) < 20:
            return patterns
        
        volume = data['Volume']
        avg_volume = volume.rolling(20).mean()
        current_volume = volume.iloc[-1]
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
        
        for pattern in patterns:
            # Adjust confidence based on volume
            if volume_ratio > 1.5:
                pattern["volume_confirmed"] = True
                pattern["confidence"] = min(1.0, pattern["confidence"] * 1.1)
                pattern["volume_ratio"] = float(volume_ratio)
            elif volume_ratio < 0.7:
                pattern["volume_confirmed"] = False
                pattern["confidence"] = pattern["confidence"] * 0.8
                pattern["volume_ratio"] = float(volume_ratio)
            else:
                pattern["volume_confirmed"] = None
                pattern["volume_ratio"] = float(volume_ratio)
        
        return patterns