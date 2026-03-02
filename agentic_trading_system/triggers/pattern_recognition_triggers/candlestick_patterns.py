"""
Candlestick Patterns - Detects Japanese candlestick patterns
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger
class CandlestickPatterns:
    """
    Detects common candlestick patterns:
    - Single candle: Doji, Hammer, Shooting Star, Marubozu
    - Two candle: Engulfing, Harami, Piercing, Dark Cloud
    - Three candle: Morning Star, Evening Star, Three White Soldiers
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Pattern definitions with confidence weights
        self.patterns = {
            # Bullish patterns
            "hammer": {"bullish": True, "confidence": 0.7, "candles": 1},
            "morning_star": {"bullish": True, "confidence": 0.9, "candles": 3},
            "bullish_engulfing": {"bullish": True, "confidence": 0.8, "candles": 2},
            "piercing_line": {"bullish": True, "confidence": 0.7, "candles": 2},
            "three_white_soldiers": {"bullish": True, "confidence": 0.85, "candles": 3},
            "bullish_harami": {"bullish": True, "confidence": 0.6, "candles": 2},
            
            # Bearish patterns
            "shooting_star": {"bullish": False, "confidence": 0.7, "candles": 1},
            "evening_star": {"bullish": False, "confidence": 0.9, "candles": 3},
            "bearish_engulfing": {"bullish": False, "confidence": 0.8, "candles": 2},
            "dark_cloud_cover": {"bullish": False, "confidence": 0.7, "candles": 2},
            "three_black_crows": {"bullish": False, "confidence": 0.85, "candles": 3},
            "bearish_harami": {"bullish": False, "confidence": 0.6, "candles": 2},
            
            # Reversal signals
            "doji": {"bullish": None, "confidence": 0.5, "candles": 1},
            "dragonfly_doji": {"bullish": True, "confidence": 0.6, "candles": 1},
            "gravestone_doji": {"bullish": False, "confidence": 0.6, "candles": 1},
            "long_legged_doji": {"bullish": None, "confidence": 0.5, "candles": 1}
        }
        
        logger.info("CandlestickPatterns initialized")
    
    def detect(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect all candlestick patterns in the data
        """
        patterns = []
        
        if data.empty or len(data) < 10:
            return patterns
        
        # Get recent candles
        recent = data.iloc[-10:].copy()
        
        # Calculate candle properties
        recent['body'] = abs(recent['Close'] - recent['Open'])
        recent['upper_shadow'] = recent['High'] - recent[['Close', 'Open']].max(axis=1)
        recent['lower_shadow'] = recent[['Close', 'Open']].min(axis=1) - recent['Low']
        recent['is_bullish'] = recent['Close'] > recent['Open']
        recent['range'] = recent['High'] - recent['Low']
        recent['body_percent'] = recent['body'] / recent['range'] * 100
        
        # Detect single candle patterns
        patterns.extend(self._detect_single_candle(recent))
        
        # Detect two candle patterns
        patterns.extend(self._detect_two_candle(recent))
        
        # Detect three candle patterns
        patterns.extend(self._detect_three_candle(recent))
        
        return patterns
    
    def _detect_single_candle(self, data: pd.DataFrame) -> List[Dict]:
        """Detect single candle patterns"""
        patterns = []
        last = data.iloc[-1]
        
        # Doji (open and close very close)
        if last['body_percent'] < 10:
            # Dragonfly Doji (long lower shadow)
            if last['lower_shadow'] > last['body'] * 3 and last['upper_shadow'] < last['body']:
                patterns.append({
                    "name": "dragonfly_doji",
                    "direction": "bullish",
                    "confidence": 0.6,
                    "description": "Dragonfly Doji - potential bullish reversal",
                    "signals": ["reversal", "bullish"]
                })
            
            # Gravestone Doji (long upper shadow)
            elif last['upper_shadow'] > last['body'] * 3 and last['lower_shadow'] < last['body']:
                patterns.append({
                    "name": "gravestone_doji",
                    "direction": "bearish",
                    "confidence": 0.6,
                    "description": "Gravestone Doji - potential bearish reversal",
                    "signals": ["reversal", "bearish"]
                })
            
            # Long-legged Doji (both shadows long)
            elif (last['upper_shadow'] > last['body'] * 2 and 
                  last['lower_shadow'] > last['body'] * 2):
                patterns.append({
                    "name": "long_legged_doji",
                    "direction": "neutral",
                    "confidence": 0.5,
                    "description": "Long-legged Doji - market indecision",
                    "signals": ["indecision"]
                })
            
            # Regular Doji
            else:
                patterns.append({
                    "name": "doji",
                    "direction": "neutral",
                    "confidence": 0.5,
                    "description": "Doji - market indecision",
                    "signals": ["indecision"]
                })
        
        # Hammer (small body, long lower shadow, little upper shadow)
        if (last['lower_shadow'] > last['body'] * 2 and
            last['upper_shadow'] < last['body'] and
            last['body_percent'] < 50 and
            not last['is_bullish']):  # Usually bearish candle becomes hammer
            
            patterns.append({
                "name": "hammer",
                "direction": "bullish",
                "confidence": 0.7,
                "description": "Hammer - potential bullish reversal after downtrend",
                "signals": ["reversal", "bullish"]
            })
        
        # Shooting Star (small body, long upper shadow, little lower shadow)
        if (last['upper_shadow'] > last['body'] * 2 and
            last['lower_shadow'] < last['body'] and
            last['body_percent'] < 50 and
            last['is_bullish']):  # Usually bullish candle becomes shooting star
            
            patterns.append({
                "name": "shooting_star",
                "direction": "bearish",
                "confidence": 0.7,
                "description": "Shooting Star - potential bearish reversal after uptrend",
                "signals": ["reversal", "bearish"]
            })
        
        return patterns
    
    def _detect_two_candle(self, data: pd.DataFrame) -> List[Dict]:
        """Detect two-candle patterns"""
        patterns = []
        
        if len(data) < 2:
            return patterns
        
        current = data.iloc[-1]
        previous = data.iloc[-2]
        
        # Bullish Engulfing
        if (not previous['is_bullish'] and current['is_bullish'] and
            current['Open'] < previous['Close'] and
            current['Close'] > previous['Open']):
            
            patterns.append({
                "name": "bullish_engulfing",
                "direction": "bullish",
                "confidence": 0.8,
                "description": "Bullish Engulfing - strong buying pressure",
                "signals": ["reversal", "bullish", "strong"]
            })
        
        # Bearish Engulfing
        elif (previous['is_bullish'] and not current['is_bullish'] and
              current['Open'] > previous['Close'] and
              current['Close'] < previous['Open']):
            
            patterns.append({
                "name": "bearish_engulfing",
                "direction": "bearish",
                "confidence": 0.8,
                "description": "Bearish Engulfing - strong selling pressure",
                "signals": ["reversal", "bearish", "strong"]
            })
        
        # Bullish Harami
        elif (not previous['is_bullish'] and current['is_bullish'] and
              current['Open'] > previous['Close'] and
              current['Close'] < previous['Open'] and
              current['body'] < previous['body']):
            
            patterns.append({
                "name": "bullish_harami",
                "direction": "bullish",
                "confidence": 0.6,
                "description": "Bullish Harami - potential reversal with caution",
                "signals": ["reversal", "bullish", "weak"]
            })
        
        # Bearish Harami
        elif (previous['is_bullish'] and not current['is_bullish'] and
              current['Open'] < previous['Close'] and
              current['Close'] > previous['Open'] and
              current['body'] < previous['body']):
            
            patterns.append({
                "name": "bearish_harami",
                "direction": "bearish",
                "confidence": 0.6,
                "description": "Bearish Harami - potential reversal with caution",
                "signals": ["reversal", "bearish", "weak"]
            })
        
        # Piercing Line
        elif (not previous['is_bullish'] and current['is_bullish'] and
              current['Open'] < previous['Low'] and
              current['Close'] > (previous['Open'] + previous['Close']) / 2):
            
            patterns.append({
                "name": "piercing_line",
                "direction": "bullish",
                "confidence": 0.7,
                "description": "Piercing Line - bullish reversal signal",
                "signals": ["reversal", "bullish"]
            })
        
        # Dark Cloud Cover
        elif (previous['is_bullish'] and not current['is_bullish'] and
              current['Open'] > previous['High'] and
              current['Close'] < (previous['Open'] + previous['Close']) / 2):
            
            patterns.append({
                "name": "dark_cloud_cover",
                "direction": "bearish",
                "confidence": 0.7,
                "description": "Dark Cloud Cover - bearish reversal signal",
                "signals": ["reversal", "bearish"]
            })
        
        return patterns
    
    def _detect_three_candle(self, data: pd.DataFrame) -> List[Dict]:
        """Detect three-candle patterns"""
        patterns = []
        
        if len(data) < 3:
            return patterns
        
        c1 = data.iloc[-3]
        c2 = data.iloc[-2]
        c3 = data.iloc[-1]
        
        # Morning Star (bullish reversal)
        if (not c1['is_bullish'] and  # First: long bearish
            abs(c2['body_percent']) < 30 and  # Second: small body (star)
            c3['is_bullish'] and  # Third: bullish
            c3['Close'] > (c1['Open'] + c1['Close']) / 2):  # Closes into first candle
            
            patterns.append({
                "name": "morning_star",
                "direction": "bullish",
                "confidence": 0.9,
                "description": "Morning Star - strong bullish reversal",
                "signals": ["reversal", "bullish", "strong"]
            })
        
        # Evening Star (bearish reversal)
        elif (c1['is_bullish'] and  # First: long bullish
              abs(c2['body_percent']) < 30 and  # Second: small body (star)
              not c3['is_bullish'] and  # Third: bearish
              c3['Close'] < (c1['Open'] + c1['Close']) / 2):  # Closes into first candle
            
            patterns.append({
                "name": "evening_star",
                "direction": "bearish",
                "confidence": 0.9,
                "description": "Evening Star - strong bearish reversal",
                "signals": ["reversal", "bearish", "strong"]
            })
        
        # Three White Soldiers (strong bullish)
        elif (c1['is_bullish'] and c2['is_bullish'] and c3['is_bullish'] and
              c1['Close'] > c1['Open'] and
              c2['Close'] > c2['Open'] and
              c3['Close'] > c3['Open'] and
              c2['Close'] > c1['Close'] and
              c3['Close'] > c2['Close']):
            
            patterns.append({
                "name": "three_white_soldiers",
                "direction": "bullish",
                "confidence": 0.85,
                "description": "Three White Soldiers - strong bullish continuation",
                "signals": ["continuation", "bullish", "strong"]
            })
        
        # Three Black Crows (strong bearish)
        elif (not c1['is_bullish'] and not c2['is_bullish'] and not c3['is_bullish'] and
              c1['Close'] < c1['Open'] and
              c2['Close'] < c2['Open'] and
              c3['Close'] < c3['Open'] and
              c2['Close'] < c1['Close'] and
              c3['Close'] < c2['Close']):
            
            patterns.append({
                "name": "three_black_crows",
                "direction": "bearish",
                "confidence": 0.85,
                "description": "Three Black Crows - strong bearish continuation",
                "signals": ["continuation", "bearish", "strong"]
            })
        
        return patterns