"""
Growth Analyzer - Analyzes growth metrics
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as  logging

class GrowthAnalyzer:
    """
    Analyzes growth metrics:
    - Revenue Growth (QoQ, YoY)
    - EPS Growth (QoQ, YoY)
    - EBITDA Growth
    - Free Cash Flow Growth
    - Analyst Growth Estimates
    - Growth Stability
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Growth thresholds
        self.strong_growth = config.get("strong_growth", 20)  # 20%+
        self.moderate_growth = config.get("moderate_growth", 10)  # 10-20%
        self.low_growth = config.get("low_growth", 5)  # 5-10%
        
        logging.info(f"✅ GrowthAnalyzer initialized")
    
    async def analyze(self, info: Dict, benchmarks: Dict) -> Dict[str, Any]:
        """
        Analyze growth metrics
        """
        results = {}
        scores = []
        
        # Revenue Growth
        revenue_growth = info.get("revenueGrowth")
        if revenue_growth is not None:
            analysis = self._analyze_revenue_growth(revenue_growth * 100)
            results["revenue_growth"] = analysis
            scores.append(analysis["score"])
        
        # EPS Growth
        eps_growth = info.get("earningsGrowth")
        if eps_growth is not None:
            analysis = self._analyze_eps_growth(eps_growth * 100)
            results["eps_growth"] = analysis
            scores.append(analysis["score"])
        
        # Quarterly Revenue Growth
        q_revenue_growth = info.get("revenueQuarterlyGrowth")
        if q_revenue_growth is not None:
            analysis = self._analyze_quarterly_growth(q_revenue_growth * 100, "revenue")
            results["quarterly_revenue_growth"] = analysis
            scores.append(analysis["score"])
        
        # Quarterly EPS Growth
        q_eps_growth = info.get("earningsQuarterlyGrowth")
        if q_eps_growth is not None:
            analysis = self._analyze_quarterly_growth(q_eps_growth * 100, "eps")
            results["quarterly_eps_growth"] = analysis
            scores.append(analysis["score"])
        
        # EBITDA Growth (if available)
        ebitda_growth = self._extract_ebitda_growth(info)
        if ebitda_growth is not None:
            analysis = self._analyze_ebitda_growth(ebitda_growth)
            results["ebitda_growth"] = analysis
            scores.append(analysis["score"])
        
        # FCF Growth (if available)
        fcf_growth = self._extract_fcf_growth(info)
        if fcf_growth is not None:
            analysis = self._analyze_fcf_growth(fcf_growth)
            results["fcf_growth"] = analysis
            scores.append(analysis["score"])
        
        # Analyst Growth Estimates
        analyst_growth = self._extract_analyst_growth(info)
        if analyst_growth is not None:
            analysis = self._analyze_analyst_growth(analyst_growth)
            results["analyst_growth"] = analysis
            scores.append(analysis["score"])
        
        # Growth Stability
        stability = self._analyze_growth_stability(scores)
        results["growth_stability"] = stability
        
        # Calculate composite score
        if scores:
            composite_score = np.mean(scores)
        else:
            composite_score = 0.5
        
        # Determine overall signal
        if composite_score >= 0.7:
            signal = "strong_growth"
        elif composite_score >= 0.6:
            signal = "moderate_growth"
        elif composite_score >= 0.4:
            signal = "stable"
        else:
            signal = "declining"
        
        return {
            "score": float(composite_score),
            "signal": signal,
            "details": results
        }
    
    def _analyze_revenue_growth(self, growth_pct: float) -> Dict:
        """Analyze revenue growth percentage"""
        if growth_pct > self.strong_growth:
            score = 0.9
            signal = "exceptional"
            details = f"Revenue growth exceptional at {growth_pct:.1f}%"
        elif growth_pct > self.moderate_growth:
            score = 0.8
            signal = "strong"
            details = f"Revenue growth strong at {growth_pct:.1f}%"
        elif growth_pct > self.low_growth:
            score = 0.7
            signal = "moderate"
            details = f"Revenue growth moderate at {growth_pct:.1f}%"
        elif growth_pct > 0:
            score = 0.6
            signal = "positive"
            details = f"Revenue growth positive at {growth_pct:.1f}%"
        elif growth_pct > -5:
            score = 0.4
            signal = "slightly_negative"
            details = f"Revenue slightly declining at {growth_pct:.1f}%"
        else:
            score = 0.2
            signal = "declining"
            details = f"Revenue declining at {growth_pct:.1f}%"
        
        return {
            "score": float(score),
            "value": float(growth_pct),
            "signal": signal,
            "details": details
        }
    
    def _analyze_eps_growth(self, growth_pct: float) -> Dict:
        """Analyze EPS growth percentage"""
        if growth_pct > self.strong_growth:
            score = 0.9
            signal = "exceptional"
            details = f"EPS growth exceptional at {growth_pct:.1f}%"
        elif growth_pct > self.moderate_growth:
            score = 0.8
            signal = "strong"
            details = f"EPS growth strong at {growth_pct:.1f}%"
        elif growth_pct > self.low_growth:
            score = 0.7
            signal = "moderate"
            details = f"EPS growth moderate at {growth_pct:.1f}%"
        elif growth_pct > 0:
            score = 0.6
            signal = "positive"
            details = f"EPS growth positive at {growth_pct:.1f}%"
        elif growth_pct > -10:
            score = 0.3
            signal = "negative"
            details = f"EPS declining at {growth_pct:.1f}%"
        else:
            score = 0.1
            signal = "severely_negative"
            details = f"EPS severely declining at {growth_pct:.1f}%"
        
        return {
            "score": float(score),
            "value": float(growth_pct),
            "signal": signal,
            "details": details
        }
    
    def _analyze_quarterly_growth(self, growth_pct: float, metric: str) -> Dict:
        """Analyze quarterly growth"""
        if growth_pct > self.moderate_growth:
            score = 0.8
            signal = "accelerating"
            details = f"Quarterly {metric} growth accelerating at {growth_pct:.1f}%"
        elif growth_pct > self.low_growth:
            score = 0.7
            signal = "positive"
            details = f"Quarterly {metric} growth positive at {growth_pct:.1f}%"
        elif growth_pct > 0:
            score = 0.6
            signal = "slightly_positive"
            details = f"Quarterly {metric} growth slightly positive at {growth_pct:.1f}%"
        elif growth_pct > -5:
            score = 0.4
            signal = "slightly_negative"
            details = f"Quarterly {metric} slightly declining at {growth_pct:.1f}%"
        else:
            score = 0.2
            signal = "declining"
            details = f"Quarterly {metric} declining at {growth_pct:.1f}%"
        
        return {
            "score": float(score),
            "value": float(growth_pct),
            "signal": signal,
            "details": details
        }
    
    def _analyze_ebitda_growth(self, growth_pct: float) -> Dict:
        """Analyze EBITDA growth"""
        if growth_pct > self.strong_growth:
            score = 0.9
            signal = "exceptional"
            details = f"EBITDA growth exceptional at {growth_pct:.1f}%"
        elif growth_pct > self.moderate_growth:
            score = 0.8
            signal = "strong"
            details = f"EBITDA growth strong at {growth_pct:.1f}%"
        elif growth_pct > self.low_growth:
            score = 0.7
            signal = "moderate"
            details = f"EBITDA growth moderate at {growth_pct:.1f}%"
        elif growth_pct > 0:
            score = 0.6
            signal = "positive"
            details = f"EBITDA growth positive at {growth_pct:.1f}%"
        else:
            score = 0.3
            signal = "negative"
            details = f"EBITDA declining at {growth_pct:.1f}%"
        
        return {
            "score": float(score),
            "value": float(growth_pct),
            "signal": signal,
            "details": details
        }
    
    def _analyze_fcf_growth(self, growth_pct: float) -> Dict:
        """Analyze Free Cash Flow growth"""
        if growth_pct > self.strong_growth:
            score = 0.9
            signal = "exceptional"
            details = f"FCF growth exceptional at {growth_pct:.1f}%"
        elif growth_pct > self.moderate_growth:
            score = 0.8
            signal = "strong"
            details = f"FCF growth strong at {growth_pct:.1f}%"
        elif growth_pct > self.low_growth:
            score = 0.7
            signal = "moderate"
            details = f"FCF growth moderate at {growth_pct:.1f}%"
        elif growth_pct > 0:
            score = 0.6
            signal = "positive"
            details = f"FCF growth positive at {growth_pct:.1f}%"
        else:
            score = 0.3
            signal = "negative"
            details = f"FCF declining at {growth_pct:.1f}%"
        
        return {
            "score": float(score),
            "value": float(growth_pct),
            "signal": signal,
            "details": details
        }
    
    def _analyze_analyst_growth(self, growth_pct: float) -> Dict:
        """Analyze analyst growth estimates"""
        if growth_pct > self.strong_growth:
            score = 0.8
            signal = "strong_outlook"
            details = f"Analysts expect strong growth of {growth_pct:.1f}%"
        elif growth_pct > self.moderate_growth:
            score = 0.7
            signal = "positive_outlook"
            details = f"Analysts expect growth of {growth_pct:.1f}%"
        elif growth_pct > self.low_growth:
            score = 0.6
            signal = "moderate_outlook"
            details = f"Analysts expect moderate growth of {growth_pct:.1f}%"
        elif growth_pct > 0:
            score = 0.5
            signal = "slightly_positive_outlook"
            details = f"Analysts expect slight growth of {growth_pct:.1f}%"
        else:
            score = 0.3
            signal = "negative_outlook"
            details = f"Analysts expect decline of {growth_pct:.1f}%"
        
        return {
            "score": float(score),
            "value": float(growth_pct),
            "signal": signal,
            "details": details
        }
    
    def _analyze_growth_stability(self, scores: List[float]) -> Dict:
        """Analyze stability of growth across metrics"""
        if not scores:
            return {"score": 0.5, "signal": "unknown", "details": "Insufficient data"}
        
        # Calculate consistency (lower std = more stable)
        if len(scores) > 1:
            std = np.std(scores)
            consistency = 1 - min(1.0, std)
        else:
            consistency = 0.7
        
        # Determine signal
        if consistency >= 0.8:
            signal = "very_stable"
            details = "Growth metrics very consistent"
        elif consistency >= 0.6:
            signal = "stable"
            details = "Growth metrics reasonably consistent"
        else:
            signal = "volatile"
            details = "Growth metrics show volatility"
        
        return {
            "score": float(consistency),
            "signal": signal,
            "details": details
        }
    
    def _extract_ebitda_growth(self, info: Dict) -> Optional[float]:
        """Extract EBITDA growth from info dict"""
        # Try different possible field names
        candidates = ["ebitdaGrowth", "ebitdaQuarterlyGrowth"]
        
        for field in candidates:
            value = info.get(field)
            if value is not None:
                return value * 100
        
        return None
    
    def _extract_fcf_growth(self, info: Dict) -> Optional[float]:
        """Extract Free Cash Flow growth"""
        candidates = ["freeCashflowGrowth", "cashFlowGrowth"]
        
        for field in candidates:
            value = info.get(field)
            if value is not None:
                return value * 100
        
        return None
    
    def _extract_analyst_growth(self, info: Dict) -> Optional[float]:
        """Extract analyst growth estimates"""
        candidates = [
            "earningsGrowth",  # This is often forward-looking
            "revenueGrowth"    # Sometimes includes estimates
        ]
        
        for field in candidates:
            value = info.get(field)
            if value is not None and value > 0:
                return value * 100
        
        return None