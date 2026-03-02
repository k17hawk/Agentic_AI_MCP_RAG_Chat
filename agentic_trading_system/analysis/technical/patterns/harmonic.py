"""
Harmonic Patterns - Detection of harmonic price patterns
"""
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from utils.logger import logger as  logging

class HarmonicPatterns:
    """
    Comprehensive harmonic pattern detection
    
    Patterns:
    - Gartley
    - Butterfly
    - Bat
    - Crab
    - Deep Crab
    - Shark
    - Cypher
    - ABCD
    - Three Drives
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Fibonacci ratios for harmonic patterns
        self.fib_ratios = {
            "gartley": {
                "XA": 1.0,
                "AB": 0.618,  # Retracement of XA
                "BC": 0.382,  # Retracement of AB
                "CD": 1.272   # Extension of BC
            },
            "butterfly": {
                "XA": 1.0,
                "AB": 0.786,
                "BC": 0.382,
                "CD": 1.618
            },
            "bat": {
                "XA": 1.0,
                "AB": 0.382,
                "BC": 0.382,
                "CD": 2.618
            },
            "crab": {
                "XA": 1.0,
                "AB": 0.382,
                "BC": 0.382,
                "CD": 3.618
            },
            "deep_crab": {
                "XA": 1.0,
                "AB": 0.886,
                "BC": 0.382,
                "CD": 2.618
            },
            "shark": {
                "XA": 1.0,
                "AB": 0.500,
                "BC": 0.500,
                "CD": 1.130
            },
            "cypher": {
                "XA": 1.0,
                "AB": 0.382,
                "BC": 1.272,
                "CD": 0.786
            }
        }
        
        # Tolerance for ratio matching
        self.tolerance = config.get("harmonic_tolerance", 0.1)  # 10% tolerance
        
        # Minimum number of pivots needed
        self.min_pivots = config.get("min_harmonic_pivots", 5)
        
        logging.info(f"✅ HarmonicPatterns initialized")
    
    def detect_all(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect all harmonic patterns in the data
        """
        patterns = []
        
        if data.empty or len(data) < 50:
            return patterns
        
        # Find swing points
        pivots = self._find_swing_points(data)
        
        if len(pivots) < self.min_pivots:
            return patterns
        
        # Detect ABCD pattern
        abcd_patterns = self._detect_abcd(pivots)
        patterns.extend(abcd_patterns)
        
        # Detect Gartley patterns
        for pattern_name in ["gartley", "butterfly", "bat", "crab", "deep_crab"]:
            detected = self._detect_harmonic_pattern(pivots, pattern_name)
            patterns.extend(detected)
        
        # Detect Three Drives
        three_drives = self._detect_three_drives(pivots)
        patterns.extend(three_drives)
        
        logging.info(f"📊 Detected {len(patterns)} harmonic patterns")
        return patterns
    
    def _find_swing_points(self, data: pd.DataFrame, order: int = 5) -> List[Dict]:
        """
        Find swing highs and lows
        """
        highs = data['High'].values
        lows = data['Low'].values
        closes = data['Close'].values
        dates = data.index
        
        # Find local maxima
        max_idx = argrelextrema(highs, np.greater, order=order)[0]
        
        # Find local minima
        min_idx = argrelextrema(lows, np.less, order=order)[0]
        
        # Combine and sort by index
        pivots = []
        
        for idx in max_idx:
            pivots.append({
                "index": idx,
                "date": dates[idx],
                "price": float(highs[idx]),
                "type": "high",
                "strength": self._calculate_pivot_strength(data, idx, "high")
            })
        
        for idx in min_idx:
            pivots.append({
                "index": idx,
                "date": dates[idx],
                "price": float(lows[idx]),
                "type": "low",
                "strength": self._calculate_pivot_strength(data, idx, "low")
            })
        
        # Sort by index
        pivots.sort(key=lambda x: x["index"])
        
        return pivots
    
    def _calculate_pivot_strength(self, data: pd.DataFrame, idx: int, 
                                  pivot_type: str) -> float:
        """
        Calculate pivot strength based on surrounding price action
        """
        if idx < 5 or idx > len(data) - 5:
            return 0.5
        
        if pivot_type == "high":
            # Check how many lower highs around it
            left_higher = sum(1 for i in range(idx-5, idx) 
                            if data['High'].iloc[i] > data['High'].iloc[idx])
            right_higher = sum(1 for i in range(idx+1, idx+6) 
                             if data['High'].iloc[i] > data['High'].iloc[idx])
            
            strength = 1.0 - (left_higher + right_higher) / 10
        
        else:  # low
            # Check how many higher lows around it
            left_lower = sum(1 for i in range(idx-5, idx) 
                           if data['Low'].iloc[i] < data['Low'].iloc[idx])
            right_lower = sum(1 for i in range(idx+1, idx+6) 
                            if data['Low'].iloc[i] < data['Low'].iloc[idx])
            
            strength = 1.0 - (left_lower + right_lower) / 10
        
        return float(max(0.3, min(1.0, strength)))
    
    def _detect_abcd(self, pivots: List[Dict]) -> List[Dict[str, Any]]:
        """
        Detect ABCD patterns
        """
        patterns = []
        
        for i in range(len(pivots) - 3):
            A = pivots[i]
            B = pivots[i + 1]
            C = pivots[i + 2]
            D = pivots[i + 3]
            
            # Check alternating pattern
            if A["type"] == B["type"] or B["type"] == C["type"] or C["type"] == D["type"]:
                continue
            
            # Calculate swings
            AB = abs(B["price"] - A["price"])
            BC = abs(C["price"] - B["price"])
            CD = abs(D["price"] - C["price"])
            
            # Check if valid
            if AB == 0 or BC == 0:
                continue
            
            # Calculate ratios
            BC_AB = BC / AB
            CD_BC = CD / BC
            
            # Check for ideal ABCD (BC should be 0.618 of AB, CD should equal AB)
            if (abs(BC_AB - 0.618) < self.tolerance and 
                abs(CD_BC - 1.618) < self.tolerance):
                
                # Determine direction
                if A["type"] == "low" and B["type"] == "high":
                    direction = "bearish"
                else:
                    direction = "bullish"
                
                patterns.append({
                    "name": "abcd",
                    "type": "harmonic",
                    "direction": direction,
                    "confidence": 0.7,
                    "description": f"ABCD Pattern - {direction} harmonic pattern",
                    "points": {
                        "A": {"price": A["price"], "index": int(A["index"])},
                        "B": {"price": B["price"], "index": int(B["index"])},
                        "C": {"price": C["price"], "index": int(C["index"])},
                        "D": {"price": D["price"], "index": int(D["index"])}
                    },
                    "ratios": {
                        "AB": float(AB),
                        "BC_AB": float(BC_AB),
                        "CD_BC": float(CD_BC)
                    },
                    "position": "current" if i == len(pivots) - 4 else "historical"
                })
        
        return patterns
    
    def _detect_harmonic_pattern(self, pivots: List[Dict], 
                                 pattern_name: str) -> List[Dict[str, Any]]:
        """
        Detect harmonic patterns (Gartley, Butterfly, Bat, etc.)
        """
        patterns = []
        ratios = self.fib_ratios[pattern_name]
        
        for i in range(len(pivots) - 4):
            X = pivots[i]
            A = pivots[i + 1]
            B = pivots[i + 2]
            C = pivots[i + 3]
            D = pivots[i + 4]
            
            # Check alternating pattern
            if (X["type"] == A["type"] or A["type"] == B["type"] or 
                B["type"] == C["type"] or C["type"] == D["type"]):
                continue
            
            # Calculate swings
            XA = abs(A["price"] - X["price"])
            AB = abs(B["price"] - A["price"])
            BC = abs(C["price"] - B["price"])
            CD = abs(D["price"] - C["price"])
            
            if XA == 0 or AB == 0 or BC == 0:
                continue
            
            # Calculate ratios
            AB_XA = AB / XA
            BC_AB = BC / AB
            CD_BC = CD / BC
            
            # Check if pattern matches
            if (abs(AB_XA - ratios["AB"]) < self.tolerance and
                abs(BC_AB - ratios["BC"]) < self.tolerance and
                abs(CD_BC - ratios["CD"]) < self.tolerance):
                
                # Determine direction
                if X["type"] == "low" and A["type"] == "high":  # X to A up
                    if B["type"] == "low":  # Then down to B
                        direction = "bullish"
                    else:
                        direction = "bearish"
                else:  # X to A down
                    if B["type"] == "high":
                        direction = "bearish"
                    else:
                        direction = "bullish"
                
                # Calculate confidence based on ratio accuracy
                ratio_accuracy = 1.0 - (
                    abs(AB_XA - ratios["AB"]) / ratios["AB"] +
                    abs(BC_AB - ratios["BC"]) / ratios["BC"] +
                    abs(CD_BC - ratios["CD"]) / ratios["CD"]
                ) / 3
                
                confidence = 0.6 + ratio_accuracy * 0.3
                
                patterns.append({
                    "name": pattern_name,
                    "type": "harmonic",
                    "direction": direction,
                    "confidence": float(min(0.95, confidence)),
                    "description": f"{pattern_name.title()} Pattern - {direction} harmonic pattern",
                    "points": {
                        "X": {"price": X["price"], "index": int(X["index"])},
                        "A": {"price": A["price"], "index": int(A["index"])},
                        "B": {"price": B["price"], "index": int(B["index"])},
                        "C": {"price": C["price"], "index": int(C["index"])},
                        "D": {"price": D["price"], "index": int(D["index"])}
                    },
                    "ratios": {
                        "AB_XA": float(AB_XA),
                        "BC_AB": float(BC_AB),
                        "CD_BC": float(CD_BC),
                        "ideal_AB": ratios["AB"],
                        "ideal_BC": ratios["BC"],
                        "ideal_CD": ratios["CD"]
                    },
                    "position": "current" if i == len(pivots) - 5 else "historical"
                })
        
        return patterns
    
    def _detect_three_drives(self, pivots: List[Dict]) -> List[Dict[str, Any]]:
        """
        Detect Three Drives pattern
        """
        patterns = []
        
        for i in range(len(pivots) - 5):
            # Need 6 points for three drives
            start = pivots[i]
            drive1 = pivots[i + 1]
            pullback1 = pivots[i + 2]
            drive2 = pivots[i + 3]
            pullback2 = pivots[i + 4]
            drive3 = pivots[i + 5]
            
            # Check alternating pattern
            if (start["type"] != drive1["type"] or 
                drive1["type"] == pullback1["type"] or
                pullback1["type"] != drive2["type"] or
                drive2["type"] == pullback2["type"] or
                pullback2["type"] != drive3["type"]):
                continue
            
            # Calculate moves
            move1 = abs(drive1["price"] - start["price"])
            retrace1 = abs(pullback1["price"] - drive1["price"])
            move2 = abs(drive2["price"] - pullback1["price"])
            retrace2 = abs(pullback2["price"] - drive2["price"])
            move3 = abs(drive3["price"] - pullback2["price"])
            
            if move1 == 0 or move2 == 0:
                continue
            
            # Check ratios
            retrace1_ratio = retrace1 / move1
            retrace2_ratio = retrace2 / move2
            move2_ratio = move2 / move1
            move3_ratio = move3 / move2
            
            # Ideal ratios: 1.272 or 1.618 for drives, 0.618 for retracements
            if (abs(move2_ratio - 1.272) < self.tolerance or 
                abs(move2_ratio - 1.618) < self.tolerance):
                
                if (abs(move3_ratio - 1.272) < self.tolerance or 
                    abs(move3_ratio - 1.618) < self.tolerance):
                    
                    if (abs(retrace1_ratio - 0.618) < self.tolerance and
                        abs(retrace2_ratio - 0.618) < self.tolerance):
                        
                        # Determine direction
                        if start["type"] == "low":
                            direction = "bullish"
                        else:
                            direction = "bearish"
                        
                        patterns.append({
                            "name": "three_drives",
                            "type": "harmonic",
                            "direction": direction,
                            "confidence": 0.75,
                            "description": f"Three Drives Pattern - {direction} reversal pattern",
                            "points": {
                                "start": {"price": start["price"], "index": int(start["index"])},
                                "drive1": {"price": drive1["price"], "index": int(drive1["index"])},
                                "pullback1": {"price": pullback1["price"], "index": int(pullback1["index"])},
                                "drive2": {"price": drive2["price"], "index": int(drive2["index"])},
                                "pullback2": {"price": pullback2["price"], "index": int(pullback2["index"])},
                                "drive3": {"price": drive3["price"], "index": int(drive3["index"])}
                            },
                            "ratios": {
                                "move2_ratio": float(move2_ratio),
                                "move3_ratio": float(move3_ratio),
                                "retrace1_ratio": float(retrace1_ratio),
                                "retrace2_ratio": float(retrace2_ratio)
                            },
                            "position": "current"
                        })
        
        return patterns