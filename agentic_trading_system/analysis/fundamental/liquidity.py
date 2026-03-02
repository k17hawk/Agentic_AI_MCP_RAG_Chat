"""
Liquidity Analyzer - Analyzes liquidity metrics
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import  logger as  logging

class LiquidityAnalyzer:
    """
    Analyzes liquidity metrics:
    - Current Ratio
    - Quick Ratio
    - Cash Ratio
    - Operating Cash Flow Ratio
    - Working Capital
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Liquidity thresholds
        self.current_ratio_ideal = config.get("current_ratio_ideal", 2.0)
        self.current_ratio_min = config.get("current_ratio_min", 1.0)
        
        self.quick_ratio_ideal = config.get("quick_ratio_ideal", 1.0)
        self.quick_ratio_min = config.get("quick_ratio_min", 0.5)
        
        logging.info(f"✅ LiquidityAnalyzer initialized")
    
    async def analyze(self, info: Dict, benchmarks: Dict) -> Dict[str, Any]:
        """
        Analyze liquidity metrics
        """
        results = {}
        scores = []
        
        # Current Ratio
        current_ratio = info.get("currentRatio")
        if current_ratio:
            analysis = self._analyze_current_ratio(current_ratio, benchmarks.get("current_ratio_benchmark", 1.5))
            results["current_ratio"] = analysis
            scores.append(analysis["score"])
        
        # Quick Ratio
        quick_ratio = info.get("quickRatio")
        if quick_ratio:
            analysis = self._analyze_quick_ratio(quick_ratio)
            results["quick_ratio"] = analysis
            scores.append(analysis["score"])
        
        # Cash Ratio
        cash_ratio = self._calculate_cash_ratio(info)
        if cash_ratio:
            analysis = self._analyze_cash_ratio(cash_ratio)
            results["cash_ratio"] = analysis
            scores.append(analysis["score"])
        
        # Operating Cash Flow Ratio
        ocf_ratio = self._calculate_ocf_ratio(info)
        if ocf_ratio:
            analysis = self._analyze_ocf_ratio(ocf_ratio)
            results["ocf_ratio"] = analysis
            scores.append(analysis["score"])
        
        # Working Capital
        working_capital = self._calculate_working_capital(info)
        if working_capital:
            results["working_capital"] = {
                "score": 0.5,  # Placeholder - would need industry context
                "value": float(working_capital),
                "signal": "neutral",
                "details": f"Working capital: ${working_capital:,.0f}"
            }
            scores.append(0.5)
        
        # Calculate composite score
        if scores:
            composite_score = np.mean(scores)
        else:
            composite_score = 0.5
        
        # Determine overall signal
        if composite_score >= 0.7:
            signal = "strong_liquidity"
        elif composite_score >= 0.5:
            signal = "adequate_liquidity"
        else:
            signal = "liquidity_concern"
        
        return {
            "score": float(composite_score),
            "signal": signal,
            "details": results
        }
    
    def _analyze_current_ratio(self, ratio: float, benchmark: float) -> Dict:
        """Analyze Current Ratio"""
        if ratio > benchmark * 1.3:
            score = 0.9
            signal = "very_strong"
            details = f"Current ratio very strong at {ratio:.2f}"
        elif ratio > benchmark:
            score = 0.8
            signal = "strong"
            details = f"Current ratio strong at {ratio:.2f}"
        elif ratio > self.current_ratio_min:
            score = 0.7
            signal = "adequate"
            details = f"Current ratio adequate at {ratio:.2f}"
        elif ratio > 0.8:
            score = 0.4
            signal = "weak"
            details = f"Current ratio weak at {ratio:.2f} - potential liquidity concerns"
        else:
            score = 0.2
            signal = "critical"
            details = f"Current ratio critical at {ratio:.2f} - serious liquidity risk"
        
        return {
            "score": float(score),
            "value": float(ratio),
            "signal": signal,
            "details": details
        }
    
    def _analyze_quick_ratio(self, ratio: float) -> Dict:
        """Analyze Quick Ratio (Acid Test)"""
        if ratio > self.quick_ratio_ideal * 1.3:
            score = 0.9
            signal = "very_strong"
            details = f"Quick ratio very strong at {ratio:.2f}"
        elif ratio > self.quick_ratio_ideal:
            score = 0.8
            signal = "strong"
            details = f"Quick ratio strong at {ratio:.2f}"
        elif ratio > self.quick_ratio_min:
            score = 0.7
            signal = "adequate"
            details = f"Quick ratio adequate at {ratio:.2f}"
        elif ratio > 0.3:
            score = 0.4
            signal = "weak"
            details = f"Quick ratio weak at {ratio:.2f} - liquidity concerns"
        else:
            score = 0.2
            signal = "critical"
            details = f"Quick ratio critical at {ratio:.2f} - immediate liquidity risk"
        
        return {
            "score": float(score),
            "value": float(ratio),
            "signal": signal,
            "details": details
        }
    
    def _analyze_cash_ratio(self, ratio: float) -> Dict:
        """Analyze Cash Ratio (most conservative)"""
        if ratio > 0.5:
            score = 0.9
            signal = "very_strong"
            details = f"Cash ratio very strong at {ratio:.2f}"
        elif ratio > 0.3:
            score = 0.8
            signal = "strong"
            details = f"Cash ratio strong at {ratio:.2f}"
        elif ratio > 0.2:
            score = 0.7
            signal = "adequate"
            details = f"Cash ratio adequate at {ratio:.2f}"
        elif ratio > 0.1:
            score = 0.5
            signal = "moderate"
            details = f"Cash ratio moderate at {ratio:.2f}"
        elif ratio > 0:
            score = 0.3
            signal = "low"
            details = f"Cash ratio low at {ratio:.2f}"
        else:
            score = 0.1
            signal = "very_low"
            details = f"Cash ratio very low at {ratio:.2f}"
        
        return {
            "score": float(score),
            "value": float(ratio),
            "signal": signal,
            "details": details
        }
    
    def _analyze_ocf_ratio(self, ratio: float) -> Dict:
        """Analyze Operating Cash Flow Ratio"""
        if ratio > 1.2:
            score = 0.9
            signal = "very_strong"
            details = f"Operating cash flow ratio very strong at {ratio:.2f}"
        elif ratio > 1.0:
            score = 0.8
            signal = "strong"
            details = f"Operating cash flow ratio strong at {ratio:.2f}"
        elif ratio > 0.8:
            score = 0.7
            signal = "adequate"
            details = f"Operating cash flow ratio adequate at {ratio:.2f}"
        elif ratio > 0.5:
            score = 0.5
            signal = "moderate"
            details = f"Operating cash flow ratio moderate at {ratio:.2f}"
        elif ratio > 0:
            score = 0.3
            signal = "weak"
            details = f"Operating cash flow ratio weak at {ratio:.2f}"
        else:
            score = 0.1
            signal = "negative"
            details = f"Operating cash flow ratio negative at {ratio:.2f}"
        
        return {
            "score": float(score),
            "value": float(ratio),
            "signal": signal,
            "details": details
        }
    
    def _calculate_cash_ratio(self, info: Dict) -> Optional[float]:
        """
        Calculate Cash Ratio = (Cash + Equivalents) / Current Liabilities
        """
        cash = info.get("totalCash")
        liabilities = info.get("totalCurrentLiabilities")
        
        if cash and liabilities and liabilities > 0:
            return cash / liabilities
        
        return None
    
    def _calculate_ocf_ratio(self, info: Dict) -> Optional[float]:
        """
        Calculate Operating Cash Flow Ratio = Operating Cash Flow / Current Liabilities
        """
        ocf = info.get("operatingCashFlow")
        liabilities = info.get("totalCurrentLiabilities")
        
        if ocf and liabilities and liabilities > 0:
            return ocf / liabilities
        
        return None
    
    def _calculate_working_capital(self, info: Dict) -> Optional[float]:
        """
        Calculate Working Capital = Current Assets - Current Liabilities
        """
        current_assets = info.get("totalCurrentAssets")
        current_liabilities = info.get("totalCurrentLiabilities")
        
        if current_assets and current_liabilities:
            return current_assets - current_liabilities
        
        return None