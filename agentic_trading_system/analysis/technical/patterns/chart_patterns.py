"""
Chart Patterns - Detection of classical chart patterns
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from agentic_trading_system.utils.logger import logger as logging

class ChartPatterns:
    """
    Comprehensive chart pattern detection
    
    Patterns:
    - Head and Shoulders (top/bottom)
    - Double Top/Bottom
    - Triple Top/Bottom
    - Rounding Bottom/Top
    - Flags and Pennants
    - Wedges (Rising/Falling)
    - Triangles (Symmetrical, Ascending, Descending)
    - Channels
    - Support and Resistance Levels
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Pattern detection parameters
        self.extrema_order = config.get("extrema_order", 5)  # For finding local maxima/minima
        self.min_pattern_size = config.get("min_pattern_size", 10)  # Minimum candles for pattern
        self.max_pattern_size = config.get("max_pattern_size", 50)  # Maximum candles for pattern
        self.tolerance = config.get("pattern_tolerance", 0.03)  # 3% tolerance for price levels
        
        logging.info(f"✅ ChartPatterns initialized")
    
    def detect_all(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect all chart patterns in the data
        """
        patterns = []
        
        if data.empty or len(data) < 30:
            return patterns
        
        # Find local maxima and minima
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        
        # Find local maxima (peaks)
        max_idx = argrelextrema(highs, np.greater, order=self.extrema_order)[0]
        
        # Find local minima (troughs)
        min_idx = argrelextrema(lows, np.less, order=self.extrema_order)[0]
        
        # Detect Head and Shoulders
        hs_patterns = self._detect_head_shoulders(data, highs, lows, max_idx, min_idx)
        patterns.extend(hs_patterns)
        
        # Detect Double Top/Bottom
        double_patterns = self._detect_double_patterns(data, highs, lows, max_idx, min_idx)
        patterns.extend(double_patterns)
        
        # Detect Triple Top/Bottom
        triple_patterns = self._detect_triple_patterns(data, highs, lows, max_idx, min_idx)
        patterns.extend(triple_patterns)
        
        # Detect Triangles
        triangle_patterns = self._detect_triangles(data)
        patterns.extend(triangle_patterns)
        
        # Detect Flags and Pennants
        flag_patterns = self._detect_flags(data)
        patterns.extend(flag_patterns)
        
        # Detect Wedges
        wedge_patterns = self._detect_wedges(data)
        patterns.extend(wedge_patterns)
        
        # Detect Channels
        channel_patterns = self._detect_channels(data)
        patterns.extend(channel_patterns)
        
        # Detect Support/Resistance
        sr_levels = self._find_support_resistance(data, highs, lows, max_idx, min_idx)
        if sr_levels:
            patterns.append({
                "name": "support_resistance_levels",
                "type": "levels",
                "direction": "neutral",
                "confidence": 0.7,
                "description": "Key support and resistance levels detected",
                "levels": sr_levels,
                "position": "current"
            })
        
        logging.info(f"📊 Detected {len(patterns)} chart patterns")
        return patterns
    
    def _detect_head_shoulders(self, data: pd.DataFrame, 
                               highs: np.array, 
                               lows: np.array,
                               max_idx: np.array,
                               min_idx: np.array) -> List[Dict[str, Any]]:
        """
        Detect Head and Shoulders patterns
        """
        patterns = []
        closes = data['Close'].values
        
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
                    shoulder_diff = abs(highs[left_shoulder] - highs[right_shoulder]) / highs[left_shoulder]
                    
                    if shoulder_diff < 0.05:  # Within 5%
                        # Find neckline (lows between shoulders)
                        neckline_lows = [l for l in min_idx if left_shoulder < l < right_shoulder]
                        
                        if len(neckline_lows) >= 1:
                            neckline = np.mean([lows[l] for l in neckline_lows])
                            
                            # Check if pattern is valid
                            pattern_height = highs[head] - neckline
                            
                            # Check if price has broken or is near neckline
                            current_price = closes[-1]
                            distance_to_neckline = (current_price - neckline) / neckline
                            
                            # Determine pattern status
                            if current_price < neckline:
                                status = "broken"
                                confidence = 0.9
                            elif abs(distance_to_neckline) < 0.02:
                                status = "testing"
                                confidence = 0.8
                            else:
                                status = "forming"
                                confidence = 0.7
                            
                            patterns.append({
                                "name": "head_and_shoulders_top",
                                "type": "reversal",
                                "direction": "bearish",
                                "confidence": confidence,
                                "description": f"Head and Shoulders Top - bearish reversal pattern ({status})",
                                "left_shoulder": float(highs[left_shoulder]),
                                "head": float(highs[head]),
                                "right_shoulder": float(highs[right_shoulder]),
                                "neckline": float(neckline),
                                "target": float(neckline - pattern_height),
                                "status": status,
                                "position": "current"
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
                    shoulder_diff = abs(lows[left_shoulder] - lows[right_shoulder]) / abs(lows[left_shoulder])
                    
                    if shoulder_diff < 0.05:
                        # Find neckline (highs between shoulders)
                        neckline_highs = [h for h in max_idx if left_shoulder < h < right_shoulder]
                        
                        if len(neckline_highs) >= 1:
                            neckline = np.mean([highs[h] for h in neckline_highs])
                            
                            # Calculate pattern height
                            pattern_height = neckline - lows[head]
                            
                            # Check if price has broken or is near neckline
                            current_price = closes[-1]
                            distance_to_neckline = (current_price - neckline) / neckline
                            
                            if current_price > neckline:
                                status = "broken"
                                confidence = 0.9
                            elif abs(distance_to_neckline) < 0.02:
                                status = "testing"
                                confidence = 0.8
                            else:
                                status = "forming"
                                confidence = 0.7
                            
                            patterns.append({
                                "name": "inverse_head_and_shoulders",
                                "type": "reversal",
                                "direction": "bullish",
                                "confidence": confidence,
                                "description": f"Inverse Head and Shoulders - bullish reversal pattern ({status})",
                                "left_shoulder": float(lows[left_shoulder]),
                                "head": float(lows[head]),
                                "right_shoulder": float(lows[right_shoulder]),
                                "neckline": float(neckline),
                                "target": float(neckline + pattern_height),
                                "status": status,
                                "position": "current"
                            })
        
        return patterns
    
    def _detect_double_patterns(self, data: pd.DataFrame,
                                highs: np.array,
                                lows: np.array,
                                max_idx: np.array,
                                min_idx: np.array) -> List[Dict[str, Any]]:
        """
        Detect Double Top and Double Bottom patterns
        """
        patterns = []
        closes = data['Close'].values
        
        # Double Top (bearish)
        if len(max_idx) >= 2:
            for i in range(len(max_idx) - 1):
                first = max_idx[i]
                second = max_idx[i + 1]
                
                # Check distance between tops (not too close, not too far)
                candle_distance = second - first
                if 5 <= candle_distance <= 30:
                    # Tops should be similar height
                    height_diff = abs(highs[first] - highs[second]) / highs[first]
                    
                    if height_diff < 0.02:  # Within 2%
                        # Find the trough between them
                        troughs = [l for l in min_idx if first < l < second]
                        
                        if len(troughs) > 0:
                            trough = min([lows[l] for l in troughs])  # Lowest trough
                            
                            # Calculate pattern metrics
                            pattern_height = highs[first] - trough
                            current_price = closes[-1]
                            
                            # Check if pattern is valid
                            if current_price < trough:
                                status = "broken"
                                confidence = 0.85
                            elif abs(current_price - trough) / trough < 0.02:
                                status = "testing"
                                confidence = 0.75
                            else:
                                status = "forming"
                                confidence = 0.7
                            
                            patterns.append({
                                "name": "double_top",
                                "type": "reversal",
                                "direction": "bearish",
                                "confidence": confidence,
                                "description": f"Double Top - bearish reversal pattern ({status})",
                                "first_top": float(highs[first]),
                                "second_top": float(highs[second]),
                                "valley": float(trough),
                                "target": float(trough - pattern_height),
                                "status": status,
                                "position": "current"
                            })
        
        # Double Bottom (bullish)
        if len(min_idx) >= 2:
            for i in range(len(min_idx) - 1):
                first = min_idx[i]
                second = min_idx[i + 1]
                
                # Check distance between bottoms
                if 5 <= second - first <= 30:
                    # Bottoms should be similar depth
                    depth_diff = abs(lows[first] - lows[second]) / abs(lows[first])
                    
                    if depth_diff < 0.02:
                        # Find the peak between them
                        peaks = [h for h in max_idx if first < h < second]
                        
                        if len(peaks) > 0:
                            peak = max([highs[h] for h in peaks])  # Highest peak
                            
                            # Calculate pattern metrics
                            pattern_height = peak - lows[first]
                            current_price = closes[-1]
                            
                            if current_price > peak:
                                status = "broken"
                                confidence = 0.85
                            elif abs(current_price - peak) / peak < 0.02:
                                status = "testing"
                                confidence = 0.75
                            else:
                                status = "forming"
                                confidence = 0.7
                            
                            patterns.append({
                                "name": "double_bottom",
                                "type": "reversal",
                                "direction": "bullish",
                                "confidence": confidence,
                                "description": f"Double Bottom - bullish reversal pattern ({status})",
                                "first_bottom": float(lows[first]),
                                "second_bottom": float(lows[second]),
                                "peak": float(peak),
                                "target": float(peak + pattern_height),
                                "status": status,
                                "position": "current"
                            })
        
        return patterns
    
    def _detect_triple_patterns(self, data: pd.DataFrame,
                                highs: np.array,
                                lows: np.array,
                                max_idx: np.array,
                                min_idx: np.array) -> List[Dict[str, Any]]:
        """
        Detect Triple Top and Triple Bottom patterns
        """
        patterns = []
        closes = data['Close'].values
        
        # Triple Top (bearish)
        if len(max_idx) >= 3:
            for i in range(len(max_idx) - 2):
                first = max_idx[i]
                second = max_idx[i + 1]
                third = max_idx[i + 2]
                
                # Check spacing
                if 3 <= second - first <= 20 and 3 <= third - second <= 20:
                    # All tops should be similar
                    heights = [highs[first], highs[second], highs[third]]
                    mean_height = np.mean(heights)
                    max_deviation = max(abs(h - mean_height) for h in heights) / mean_height
                    
                    if max_deviation < 0.03:  # Within 3%
                        # Find the two troughs
                        troughs = [l for l in min_idx if first < l < third]
                        
                        if len(troughs) >= 2:
                            support = np.mean([lows[l] for l in troughs])
                            
                            patterns.append({
                                "name": "triple_top",
                                "type": "reversal",
                                "direction": "bearish",
                                "confidence": 0.8,
                                "description": "Triple Top - strong bearish reversal pattern",
                                "tops": [float(h) for h in heights],
                                "support": float(support),
                                "position": "current"
                            })
        
        # Triple Bottom (bullish)
        if len(min_idx) >= 3:
            for i in range(len(min_idx) - 2):
                first = min_idx[i]
                second = min_idx[i + 1]
                third = min_idx[i + 2]
                
                if 3 <= second - first <= 20 and 3 <= third - second <= 20:
                    depths = [lows[first], lows[second], lows[third]]
                    mean_depth = np.mean(depths)
                    max_deviation = max(abs(d - mean_depth) for d in depths) / abs(mean_depth)
                    
                    if max_deviation < 0.03:
                        peaks = [h for h in max_idx if first < h < third]
                        
                        if len(peaks) >= 2:
                            resistance = np.mean([highs[h] for h in peaks])
                            
                            patterns.append({
                                "name": "triple_bottom",
                                "type": "reversal",
                                "direction": "bullish",
                                "confidence": 0.8,
                                "description": "Triple Bottom - strong bullish reversal pattern",
                                "bottoms": [float(d) for d in depths],
                                "resistance": float(resistance),
                                "position": "current"
                            })
        
        return patterns
    
    def _detect_triangles(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect triangle patterns
        """
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        # Get recent data
        recent = data.iloc[-30:]
        highs = recent['High'].values
        lows = recent['Low'].values
        closes = recent['Close'].values
        x = np.arange(len(recent))
        
        # Linear regression on highs and lows
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Calculate convergence
        slope_diff = high_slope - low_slope
        
        # Symmetrical Triangle (converging)
        if abs(slope_diff) < 0.01 and high_slope < 0 and low_slope > 0:
            # Calculate apex
            if low_slope - high_slope != 0:
                apex_x = (high_intercept - low_intercept) / (low_slope - high_slope)
                apex_price = high_slope * apex_x + high_intercept
                
                # Determine breakout direction
                current_price = closes[-1]
                if current_price > (high_intercept + low_intercept) / 2:
                    direction = "bullish"
                else:
                    direction = "bearish"
                
                patterns.append({
                    "name": "symmetrical_triangle",
                    "type": "continuation",
                    "direction": direction,
                    "confidence": 0.7,
                    "description": f"Symmetrical Triangle - {direction} breakout expected",
                    "high_slope": float(high_slope),
                    "low_slope": float(low_slope),
                    "apex_distance": float(apex_x - len(recent)),
                    "apex_price": float(apex_price),
                    "position": "current"
                })
        
        # Ascending Triangle (bullish)
        elif abs(high_slope) < 0.001 and low_slope > 0.001:  # Flat top, rising bottom
            resistance = high_intercept
            patterns.append({
                "name": "ascending_triangle",
                "type": "continuation",
                "direction": "bullish",
                "confidence": 0.75,
                "description": "Ascending Triangle - bullish continuation pattern",
                "resistance": float(resistance),
                "support_slope": float(low_slope),
                "position": "current"
            })
        
        # Descending Triangle (bearish)
        elif high_slope < -0.001 and abs(low_slope) < 0.001:  # Falling top, flat bottom
            support = low_intercept
            patterns.append({
                "name": "descending_triangle",
                "type": "continuation",
                "direction": "bearish",
                "confidence": 0.75,
                "description": "Descending Triangle - bearish continuation pattern",
                "support": float(support),
                "resistance_slope": float(high_slope),
                "position": "current"
            })
        
        return patterns
    
    def _detect_flags(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect flag and pennant patterns
        """
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        # Look for strong move followed by consolidation
        returns = data['Close'].pct_change() * 100
        
        # Find strong move (5%+ in 5 days)
        for i in range(10, len(data) - 10):
            move_returns = returns.iloc[i-5:i].sum()
            
            if abs(move_returns) > 10:  # 10% move
                # Check subsequent consolidation (5-15 days)
                consolidation = returns.iloc[i:i+10]
                consolidation_range = consolidation.max() - consolidation.min()
                
                if consolidation_range < 5:  # Tight consolidation
                    # Determine flag or pennant
                    high_slope, _ = np.polyfit(
                        range(10), 
                        data['High'].iloc[i:i+10].values, 
                        1
                    )
                    low_slope, _ = np.polyfit(
                        range(10), 
                        data['Low'].iloc[i:i+10].values, 
                        1
                    )
                    
                    if abs(high_slope) < 0.1 and abs(low_slope) < 0.1:
                        pattern_type = "flag"
                        confidence = 0.7
                    else:
                        pattern_type = "pennant"
                        confidence = 0.65
                    
                    patterns.append({
                        "name": f"{pattern_type}_{'bull' if move_returns > 0 else 'bear'}",
                        "type": "continuation",
                        "direction": "bullish" if move_returns > 0 else "bearish",
                        "confidence": confidence,
                        "description": f"{pattern_type.title()} - continuation pattern after strong move",
                        "move_pct": float(move_returns),
                        "consolidation_days": 10,
                        "position": "recent"
                    })
        
        return patterns
    
    def _detect_wedges(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect rising and falling wedges
        """
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        recent = data.iloc[-20:]
        highs = recent['High'].values
        lows = recent['Low'].values
        x = np.arange(len(recent))
        
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Rising Wedge (bearish)
        if high_slope > 0 and low_slope > 0 and high_slope > low_slope:
            patterns.append({
                "name": "rising_wedge",
                "type": "reversal",
                "direction": "bearish",
                "confidence": 0.7,
                "description": "Rising Wedge - bearish reversal pattern",
                "high_slope": float(high_slope),
                "low_slope": float(low_slope),
                "position": "current"
            })
        
        # Falling Wedge (bullish)
        elif high_slope < 0 and low_slope < 0 and high_slope < low_slope:
            patterns.append({
                "name": "falling_wedge",
                "type": "reversal",
                "direction": "bullish",
                "confidence": 0.7,
                "description": "Falling Wedge - bullish reversal pattern",
                "high_slope": float(high_slope),
                "low_slope": float(low_slope),
                "position": "current"
            })
        
        return patterns
    
    def _detect_channels(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect trading channels
        """
        patterns = []
        
        if len(data) < 30:
            return patterns
        
        recent = data.iloc[-30:]
        highs = recent['High'].values
        lows = recent['Low'].values
        x = np.arange(len(recent))
        
        high_slope, high_intercept = np.polyfit(x, highs, 1)
        low_slope, low_intercept = np.polyfit(x, lows, 1)
        
        # Check if channel is parallel
        slope_diff = abs(high_slope - low_slope)
        
        if slope_diff < 0.1:  # Parallel lines
            channel_type = "horizontal" if abs(high_slope) < 0.001 else "trending"
            
            if high_slope > 0:
                direction = "up"
            elif high_slope < 0:
                direction = "down"
            else:
                direction = "horizontal"
            
            current_price = data['Close'].iloc[-1]
            
            # Determine position in channel
            channel_top = high_slope * len(recent) + high_intercept
            channel_bottom = low_slope * len(recent) + low_intercept
            channel_height = channel_top - channel_bottom
            
            if channel_height > 0:
                position_pct = (current_price - channel_bottom) / channel_height * 100
                
                patterns.append({
                    "name": f"{direction}_channel",
                    "type": "continuation",
                    "direction": "bullish" if direction == "up" else "bearish" if direction == "down" else "neutral",
                    "confidence": 0.65,
                    "description": f"{direction.title()} Channel - price trading within channel",
                    "channel_top": float(channel_top),
                    "channel_bottom": float(channel_bottom),
                    "position_pct": float(position_pct),
                    "near_top": position_pct > 80,
                    "near_bottom": position_pct < 20,
                    "position": "current"
                })
        
        return patterns
    
    def _find_support_resistance(self, data: pd.DataFrame,
                                 highs: np.array,
                                 lows: np.array,
                                 max_idx: np.array,
                                 min_idx: np.array) -> Dict[str, List[float]]:
        """
        Find key support and resistance levels
        """
        support_levels = []
        resistance_levels = []
        
        current_price = data['Close'].iloc[-1]
        
        # Get all pivot highs (resistance)
        for idx in max_idx[-10:]:  # Last 10 pivots
            level = highs[idx]
            distance = abs(level - current_price) / current_price * 100
            
            # Only include if within 10% of current price
            if distance < 10:
                resistance_levels.append({
                    "level": float(level),
                    "distance": float(distance),
                    "strength": "major" if distance < 3 else "minor"
                })
        
        # Get all pivot lows (support)
        for idx in min_idx[-10:]:
            level = lows[idx]
            distance = abs(current_price - level) / current_price * 100
            
            if distance < 10:
                support_levels.append({
                    "level": float(level),
                    "distance": float(distance),
                    "strength": "major" if distance < 3 else "minor"
                })
        
        # Sort by distance
        resistance_levels.sort(key=lambda x: x["distance"])
        support_levels.sort(key=lambda x: x["distance"])
        
        return {
            "support": support_levels[:5],  # Top 5 closest supports
            "resistance": resistance_levels[:5]  # Top 5 closest resistances
        }