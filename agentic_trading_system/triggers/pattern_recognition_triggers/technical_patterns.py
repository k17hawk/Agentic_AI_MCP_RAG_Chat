
"""
Technical Patterns - Detects chart patterns like head & shoulders, double tops, etc.
"""
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from loguru import logger


class TechnicalPatterns:
    """
    Detects technical chart patterns:
    - Head and Shoulders (top/bottom)
    - Double Top/Bottom
    - Triple Top/Bottom
    - Rising/Falling Wedge
    - Symmetrical Triangle
    - Flag/Pennant
    """
    
    def __init__(self, config: dict):
        self.config = config
        
        # Pattern detection parameters
        self.order = config.get("pattern_order", 5)  # Local extrema window
        self.min_pattern_size = config.get("min_pattern_size", 10)  # Minimum candles
        self.max_pattern_size = config.get("max_pattern_size", 50)  # Maximum candles
        
        logger.info("TechnicalPatterns initialized")
    
    def detect(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect all technical patterns in the data
        """
        patterns = []
        
        if data.empty or len(data) < 30:
            return patterns
        
        # Find local maxima and minima
        highs = data['High'].values
        lows = data['Low'].values
        
        # Find local maxima
        max_idx = argrelextrema(highs, np.greater, order=self.order)[0]
        
        # Find local minima
        min_idx = argrelextrema(lows, np.less, order=self.order)[0]
        
        # Detect Head and Shoulders
        hs_patterns = self._detect_head_shoulders(data, highs, lows, max_idx, min_idx)
        patterns.extend(hs_patterns)
        
        # Detect Double Top/Bottom
        double_patterns = self._detect_double_patterns(data, highs, lows, max_idx, min_idx)
        patterns.extend(double_patterns)
        
        # Detect Triangles
        triangle_patterns = self._detect_triangles(data, highs, lows)
        patterns.extend(triangle_patterns)
        
        # Detect Support/Resistance
        sr_patterns = self._detect_support_resistance(data, highs, lows, max_idx, min_idx)
        patterns.extend(sr_patterns)
        
        return patterns
    
    def _detect_head_shoulders(self, data: pd.DataFrame, 
                               highs: np.array, 
                               lows: np.array,
                               max_idx: np.array,
                               min_idx: np.array) -> List[Dict]:
        """Detect Head and Shoulders patterns"""
        patterns = []
        
        # Head and Shoulders Top (bearish)
        if len(max_idx) >= 3:
            for i in range(len(max_idx) - 2):
                left_shoulder = max_idx[i]
                head = max_idx[i + 1]
                right_shoulder = max_idx[i + 2]
                
                # Check if head is higher than shoulders
                if (highs[head] > highs[left_shoulder] and 
                    highs[head] > highs[right_shoulder]):
                    
                    # Shoulders should be roughly equal height
                    shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder]) / highs[left_shoulder] if highs[left_shoulder] != 0 else 1
                    
                    if shoulder_diff < 0.1:  # Within 10%
                        # Check neckline (low between shoulders)
                        neckline_lows = [l for l in min_idx if left_shoulder < l < right_shoulder]
                        
                        if len(neckline_lows) > 0:
                            neckline = min(lows[l] for l in neckline_lows)
                            
                            # Check if price has broken neckline
                            current_price = data['Close'].iloc[-1]
                            
                            if current_price < neckline:
                                patterns.append({
                                    "name": "head_and_shoulders_top",
                                    "direction": "bearish",
                                    "confidence": 0.8,
                                    "description": "Head and Shoulders Top - bearish reversal",
                                    "signals": ["reversal", "bearish", "strong"],
                                    "neckline": float(neckline),
                                    "target": float(highs[head] - (highs[head] - neckline)),
                                    "needs_confirmation": True
                                })
        
        # Inverse Head and Shoulders (bullish)
        if len(min_idx) >= 3:
            for i in range(len(min_idx) - 2):
                left_shoulder = min_idx[i]
                head = min_idx[i + 1]
                right_shoulder = min_idx[i + 2]
                
                # Check if head is lower than shoulders
                if (lows[head] < lows[left_shoulder] and 
                    lows[head] < lows[right_shoulder]):
                    
                    # Shoulders should be roughly equal depth
                    shoulder_diff = abs(lows[left_shoulder] - lows[right_shoulder]) / abs(lows[left_shoulder]) if lows[left_shoulder] != 0 else 1
                    
                    if shoulder_diff < 0.1:  # Within 10%
                        # Check neckline (high between shoulders)
                        neckline_highs = [h for h in max_idx if left_shoulder < h < right_shoulder]
                        
                        if len(neckline_highs) > 0:
                            neckline = max(highs[h] for h in neckline_highs)
                            
                            # Check if price has broken neckline
                            current_price = data['Close'].iloc[-1]
                            
                            if current_price > neckline:
                                patterns.append({
                                    "name": "inverse_head_and_shoulders",
                                    "direction": "bullish",
                                    "confidence": 0.8,
                                    "description": "Inverse Head and Shoulders - bullish reversal",
                                    "signals": ["reversal", "bullish", "strong"],
                                    "neckline": float(neckline),
                                    "target": float(neckline + (neckline - lows[head])),
                                    "needs_confirmation": True
                                })
        
        return patterns
    
    def _detect_double_patterns(self, data: pd.DataFrame,
                                highs: np.array,
                                lows: np.array,
                                max_idx: np.array,
                                min_idx: np.array) -> List[Dict]:
        """Detect Double Top and Double Bottom patterns"""
        patterns = []
        
        # Double Top (bearish)
        if len(max_idx) >= 2:
            for i in range(len(max_idx) - 1):
                first = max_idx[i]
                second = max_idx[i + 1]
                
                # Check distance between tops (not too close)
                if second - first > 5 and second - first < 20:
                    # Tops should be similar height
                    height_diff = abs(highs[first] - highs[second]) / highs[first] if highs[first] != 0 else 1
                    
                    if height_diff < 0.03:  # Within 3%
                        # Find the trough between them
                        troughs = [l for l in min_idx if first < l < second]
                        
                        if len(troughs) > 0:
                            trough = max(lows[l] for l in troughs)  # Highest trough
                            
                            # Check for breakdown
                            current_price = data['Close'].iloc[-1]
                            
                            if current_price < trough:
                                patterns.append({
                                    "name": "double_top",
                                    "direction": "bearish",
                                    "confidence": 0.75,
                                    "description": "Double Top - bearish reversal",
                                    "signals": ["reversal", "bearish"],
                                    "resistance": float(highs[first]),
                                    "support": float(trough),
                                    "target": float(trough - (highs[first] - trough)),
                                    "needs_confirmation": True
                                })
        
        # Double Bottom (bullish)
        if len(min_idx) >= 2:
            for i in range(len(min_idx) - 1):
                first = min_idx[i]
                second = min_idx[i + 1]
                
                # Check distance between bottoms
                if second - first > 5 and second - first < 20:
                    # Bottoms should be similar depth
                    depth_diff = abs(lows[first] - lows[second]) / abs(lows[first]) if lows[first] != 0 else 1
                    
                    if depth_diff < 0.03:  # Within 3%
                        # Find the peak between them
                        peaks = [h for h in max_idx if first < h < second]
                        
                        if len(peaks) > 0:
                            peak = min(highs[h] for h in peaks)  # Lowest peak
                            
                            # Check for breakout
                            current_price = data['Close'].iloc[-1]
                            
                            if current_price > peak:
                                patterns.append({
                                    "name": "double_bottom",
                                    "direction": "bullish",
                                    "confidence": 0.75,
                                    "description": "Double Bottom - bullish reversal",
                                    "signals": ["reversal", "bullish"],
                                    "support": float(lows[first]),
                                    "resistance": float(peak),
                                    "target": float(peak + (peak - lows[first])),
                                    "needs_confirmation": True
                                })
        
        return patterns
    
    def _detect_triangles(self, data: pd.DataFrame, highs: np.array, lows: np.array) -> List[Dict]:
        """Detect triangle patterns"""
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        # Get recent data (last 20-30 candles)
        recent = data.iloc[-30:]
        recent_highs = recent['High'].values
        recent_lows = recent['Low'].values
        x = np.arange(len(recent))
        
        # Linear regression on highs and lows
        high_slope, high_intercept = np.polyfit(x, recent_highs, 1)
        low_slope, low_intercept = np.polyfit(x, recent_lows, 1)
        
        # Calculate convergence
        high_low_diff = (high_slope - low_slope)
        
        # Symmetrical Triangle (converging)
        if abs(high_low_diff) < 0.01 and high_slope < 0 and low_slope > 0:
            patterns.append({
                "name": "symmetrical_triangle",
                "direction": "neutral",
                "confidence": 0.6,
                "description": "Symmetrical Triangle - breakout pending",
                "signals": ["consolidation", "breakout_soon"],
                "high_slope": float(high_slope),
                "low_slope": float(low_slope),
                "apex": float(len(recent) + (high_intercept - low_intercept) / (low_slope - high_slope) if (low_slope - high_slope) != 0 else 0)
            })
        
        # Ascending Triangle (bullish)
        elif abs(high_slope) < 0.001 and low_slope > 0:  # Flat top, rising bottom
            patterns.append({
                "name": "ascending_triangle",
                "direction": "bullish",
                "confidence": 0.7,
                "description": "Ascending Triangle - bullish continuation",
                "signals": ["bullish", "breakout_soon"],
                "resistance": float(high_intercept),
                "support_slope": float(low_slope)
            })
        
        # Descending Triangle (bearish)
        elif high_slope < 0 and abs(low_slope) < 0.001:  # Falling top, flat bottom
            patterns.append({
                "name": "descending_triangle",
                "direction": "bearish",
                "confidence": 0.7,
                "description": "Descending Triangle - bearish continuation",
                "signals": ["bearish", "breakdown_soon"],
                "support": float(low_intercept),
                "resistance_slope": float(high_slope)
            })
        
        return patterns
    
    def _detect_support_resistance(self, data: pd.DataFrame,
                                   highs: np.array,
                                   lows: np.array,
                                   max_idx: np.array,
                                   min_idx: np.array) -> List[Dict]:
        """Detect support and resistance levels"""
        patterns = []
        
        if len(max_idx) == 0 or len(min_idx) == 0:
            return patterns
        
        current_price = data['Close'].iloc[-1]
        
        # Find nearest resistance
        resistances = [highs[i] for i in max_idx if highs[i] > current_price]
        if resistances:
            nearest_resistance = min(resistances)
            distance = (nearest_resistance - current_price) / current_price * 100 if current_price != 0 else 0
            
            if distance < 3:  # Within 3%
                patterns.append({
                    "name": "near_resistance",
                    "direction": "bearish",
                    "confidence": 0.6,
                    "description": f"Near resistance at ${nearest_resistance:.2f}",
                    "signals": ["resistance", "ceiling"],
                    "level": float(nearest_resistance),
                    "distance": float(distance)
                })
        
        # Find nearest support
        supports = [lows[i] for i in min_idx if lows[i] < current_price]
        if supports:
            nearest_support = max(supports)
            distance = (current_price - nearest_support) / current_price * 100 if current_price != 0 else 0
            
            if distance < 3:  # Within 3%
                patterns.append({
                    "name": "near_support",
                    "direction": "bullish",
                    "confidence": 0.6,
                    "description": f"Near support at ${nearest_support:.2f}",
                    "signals": ["support", "floor"],
                    "level": float(nearest_support),
                    "distance": float(distance)
                })
        
        return patterns