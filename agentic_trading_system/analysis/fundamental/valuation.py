"""
Valuation Analyzer - Analyzes valuation ratios
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as logging

class ValuationAnalyzer:
    """
    Analyzes valuation metrics:
    - P/E Ratio (Price to Earnings)
    - Forward P/E
    - P/B Ratio (Price to Book)
    - P/S Ratio (Price to Sales)
    - P/CF Ratio (Price to Cash Flow)
    - EV/EBITDA
    - EV/Revenue
    - PEG Ratio (P/E to Growth)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Valuation thresholds
        self.pe_undervalued = config.get("pe_undervalued", 15)
        self.pe_fair = config.get("pe_fair", 25)
        self.pe_overvalued = config.get("pe_overvalued", 30)
        
        self.pb_undervalued = config.get("pb_undervalued", 1.0)
        self.pb_fair = config.get("pb_fair", 3.0)
        self.pb_overvalued = config.get("pb_overvalued", 5.0)
        
        self.ps_undervalued = config.get("ps_undervalued", 1.0)
        self.ps_fair = config.get("ps_fair", 3.0)
        self.ps_overvalued = config.get("ps_overvalued", 5.0)
        
        logging.info(f"✅ ValuationAnalyzer initialized")
    
    async def analyze(self, info: Dict, benchmarks: Dict) -> Dict[str, Any]:
        """
        Analyze valuation metrics
        """
        results = {}
        scores = []
        
        # P/E Ratio
        pe = info.get("trailingPE")
        if pe:
            analysis = self._analyze_pe_ratio(pe, benchmarks.get("pe_benchmark", 20))
            results["pe_ratio"] = analysis
            scores.append(analysis["score"])
        else:
            results["pe_ratio"] = {"score": 0.5, "value": None, "signal": "neutral", "details": "Not available"}
        
        # Forward P/E
        forward_pe = info.get("forwardPE")
        if forward_pe:
            analysis = self._analyze_forward_pe(forward_pe, pe)
            results["forward_pe"] = analysis
            scores.append(analysis["score"])
        
        # P/B Ratio
        pb = info.get("priceToBook")
        if pb:
            analysis = self._analyze_pb_ratio(pb, benchmarks.get("pb_benchmark", 3))
            results["pb_ratio"] = analysis
            scores.append(analysis["score"])
        else:
            results["pb_ratio"] = {"score": 0.5, "value": None, "signal": "neutral", "details": "Not available"}
        
        # P/S Ratio
        ps = info.get("priceToSalesTrailing12Months")
        if ps:
            analysis = self._analyze_ps_ratio(ps, benchmarks.get("ps_benchmark", 3))
            results["ps_ratio"] = analysis
            scores.append(analysis["score"])
        else:
            results["ps_ratio"] = {"score": 0.5, "value": None, "signal": "neutral", "details": "Not available"}
        
        # P/CF Ratio
        pcf = info.get("priceToBook")  # Placeholder - would need actual P/CF
        if pcf:
            results["pcf_ratio"] = self._analyze_pcf_ratio(pcf)
            scores.append(results["pcf_ratio"]["score"])
        
        # EV/EBITDA
        ev_to_ebitda = info.get("enterpriseToEbitda")
        if ev_to_ebitda:
            analysis = self._analyze_ev_ebitda(ev_to_ebitda)
            results["ev_ebitda"] = analysis
            scores.append(analysis["score"])
        
        # EV/Revenue
        ev_to_revenue = info.get("enterpriseToRevenue")
        if ev_to_revenue:
            analysis = self._analyze_ev_revenue(ev_to_revenue)
            results["ev_revenue"] = analysis
            scores.append(analysis["score"])
        
        # PEG Ratio
        peg = info.get("pegRatio")
        if peg:
            analysis = self._analyze_peg_ratio(peg)
            results["peg_ratio"] = analysis
            scores.append(analysis["score"])
        
        # Calculate composite score
        if scores:
            composite_score = np.mean(scores)
        else:
            composite_score = 0.5
        
        # Determine overall signal
        if composite_score >= 0.7:
            signal = "undervalued"
        elif composite_score <= 0.3:
            signal = "overvalued"
        else:
            signal = "fair_value"
        
        return {
            "score": float(composite_score),
            "signal": signal,
            "details": results
        }
    
    def _analyze_pe_ratio(self, pe: float, benchmark: float) -> Dict:
        """Analyze P/E ratio"""
        if pe <= 0:
            return {"score": 0.5, "value": pe, "signal": "neutral", "details": "Negative earnings"}
        
        # Compare to benchmark
        if pe < benchmark * 0.7:
            score = 0.8
            signal = "undervalued"
            details = f"P/E significantly below sector average ({benchmark:.1f})"
        elif pe < benchmark * 0.9:
            score = 0.7
            signal = "moderately_undervalued"
            details = f"P/E below sector average ({benchmark:.1f})"
        elif pe < benchmark * 1.1:
            score = 0.5
            signal = "fair_value"
            details = f"P/E in line with sector average ({benchmark:.1f})"
        elif pe < benchmark * 1.3:
            score = 0.3
            signal = "moderately_overvalued"
            details = f"P/E above sector average ({benchmark:.1f})"
        else:
            score = 0.2
            signal = "overvalued"
            details = f"P/E significantly above sector average ({benchmark:.1f})"
        
        return {
            "score": float(score),
            "value": float(pe),
            "signal": signal,
            "details": details
        }
    
    def _analyze_forward_pe(self, forward_pe: float, trailing_pe: Optional[float]) -> Dict:
        """Analyze forward P/E ratio"""
        if forward_pe <= 0:
            return {"score": 0.5, "value": forward_pe, "signal": "neutral", "details": "Not available"}
        
        # Compare to trailing P/E if available
        if trailing_pe and trailing_pe > 0:
            ratio = forward_pe / trailing_pe
            
            if ratio < 0.8:
                score = 0.8
                signal = "improving"
                details = f"Forward P/E significantly lower than trailing ({ratio:.2f}x)"
            elif ratio < 0.95:
                score = 0.7
                signal = "moderately_improving"
                details = f"Forward P/E lower than trailing ({ratio:.2f}x)"
            elif ratio < 1.05:
                score = 0.5
                signal = "stable"
                details = f"Forward P/E similar to trailing ({ratio:.2f}x)"
            elif ratio < 1.2:
                score = 0.3
                signal = "moderately_worsening"
                details = f"Forward P/E higher than trailing ({ratio:.2f}x)"
            else:
                score = 0.2
                signal = "worsening"
                details = f"Forward P/E significantly higher than trailing ({ratio:.2f}x)"
        else:
            # Just use absolute value
            if forward_pe < 15:
                score = 0.7
                signal = "attractive"
                details = f"Forward P/E attractive at {forward_pe:.1f}"
            elif forward_pe < 25:
                score = 0.5
                signal = "fair"
                details = f"Forward P/E fair at {forward_pe:.1f}"
            else:
                score = 0.3
                signal = "expensive"
                details = f"Forward P/E expensive at {forward_pe:.1f}"
        
        return {
            "score": float(score),
            "value": float(forward_pe),
            "signal": signal,
            "details": details
        }
    
    def _analyze_pb_ratio(self, pb: float, benchmark: float) -> Dict:
        """Analyze Price to Book ratio"""
        if pb <= 0:
            if pb < 0:
                return {"score": 0.8, "value": pb, "signal": "undervalued", "details": "Negative book value - potentially distressed but could be value"}
            return {"score": 0.5, "value": pb, "signal": "neutral", "details": "Book value not available"}
        
        if pb < 1.0:
            score = 0.9
            signal = "deep_value"
            details = "Trading below book value - potential deep value"
        elif pb < benchmark * 0.7:
            score = 0.8
            signal = "undervalued"
            details = f"P/B significantly below sector average ({benchmark:.1f})"
        elif pb < benchmark * 0.9:
            score = 0.7
            signal = "moderately_undervalued"
            details = f"P/B below sector average ({benchmark:.1f})"
        elif pb < benchmark * 1.1:
            score = 0.5
            signal = "fair_value"
            details = f"P/B in line with sector average ({benchmark:.1f})"
        elif pb < benchmark * 1.3:
            score = 0.3
            signal = "moderately_overvalued"
            details = f"P/B above sector average ({benchmark:.1f})"
        else:
            score = 0.2
            signal = "overvalued"
            details = f"P/B significantly above sector average ({benchmark:.1f})"
        
        return {
            "score": float(score),
            "value": float(pb),
            "signal": signal,
            "details": details
        }
    
    def _analyze_ps_ratio(self, ps: float, benchmark: float) -> Dict:
        """Analyze Price to Sales ratio"""
        if ps <= 0:
            return {"score": 0.5, "value": ps, "signal": "neutral", "details": "Not available"}
        
        if ps < benchmark * 0.5:
            score = 0.8
            signal = "undervalued"
            details = f"P/S significantly below sector average ({benchmark:.1f})"
        elif ps < benchmark * 0.8:
            score = 0.7
            signal = "moderately_undervalued"
            details = f"P/S below sector average ({benchmark:.1f})"
        elif ps < benchmark * 1.2:
            score = 0.5
            signal = "fair_value"
            details = f"P/S in line with sector average ({benchmark:.1f})"
        elif ps < benchmark * 1.5:
            score = 0.3
            signal = "moderately_overvalued"
            details = f"P/S above sector average ({benchmark:.1f})"
        else:
            score = 0.2
            signal = "overvalued"
            details = f"P/S significantly above sector average ({benchmark:.1f})"
        
        return {
            "score": float(score),
            "value": float(ps),
            "signal": signal,
            "details": details
        }
    
    def _analyze_pcf_ratio(self, pcf: float) -> Dict:
        """Analyze Price to Cash Flow ratio"""
        if pcf <= 0:
            return {"score": 0.5, "value": pcf, "signal": "neutral", "details": "Not available"}
        
        if pcf < 10:
            score = 0.8
            signal = "undervalued"
            details = f"P/CF attractive at {pcf:.1f}"
        elif pcf < 15:
            score = 0.7
            signal = "moderately_undervalued"
            details = f"P/CF reasonable at {pcf:.1f}"
        elif pcf < 20:
            score = 0.5
            signal = "fair_value"
            details = f"P/CF fair at {pcf:.1f}"
        elif pcf < 25:
            score = 0.3
            signal = "moderately_overvalued"
            details = f"P/CF elevated at {pcf:.1f}"
        else:
            score = 0.2
            signal = "overvalued"
            details = f"P/CF expensive at {pcf:.1f}"
        
        return {
            "score": float(score),
            "value": float(pcf),
            "signal": signal,
            "details": details
        }
    
    def _analyze_ev_ebitda(self, ev_ebitda: float) -> Dict:
        """Analyze EV/EBITDA ratio"""
        if ev_ebitda <= 0:
            return {"score": 0.5, "value": ev_ebitda, "signal": "neutral", "details": "Not available"}
        
        if ev_ebitda < 8:
            score = 0.8
            signal = "undervalued"
            details = f"EV/EBITDA attractive at {ev_ebitda:.1f}"
        elif ev_ebitda < 12:
            score = 0.7
            signal = "moderately_undervalued"
            details = f"EV/EBITDA reasonable at {ev_ebitda:.1f}"
        elif ev_ebitda < 15:
            score = 0.5
            signal = "fair_value"
            details = f"EV/EBITDA fair at {ev_ebitda:.1f}"
        elif ev_ebitda < 20:
            score = 0.3
            signal = "moderately_overvalued"
            details = f"EV/EBITDA elevated at {ev_ebitda:.1f}"
        else:
            score = 0.2
            signal = "overvalued"
            details = f"EV/EBITDA expensive at {ev_ebitda:.1f}"
        
        return {
            "score": float(score),
            "value": float(ev_ebitda),
            "signal": signal,
            "details": details
        }
    
    def _analyze_ev_revenue(self, ev_revenue: float) -> Dict:
        """Analyze EV/Revenue ratio"""
        if ev_revenue <= 0:
            return {"score": 0.5, "value": ev_revenue, "signal": "neutral", "details": "Not available"}
        
        if ev_revenue < 1:
            score = 0.8
            signal = "undervalued"
            details = f"EV/Revenue attractive at {ev_revenue:.1f}"
        elif ev_revenue < 2:
            score = 0.7
            signal = "moderately_undervalued"
            details = f"EV/Revenue reasonable at {ev_revenue:.1f}"
        elif ev_revenue < 3:
            score = 0.5
            signal = "fair_value"
            details = f"EV/Revenue fair at {ev_revenue:.1f}"
        elif ev_revenue < 5:
            score = 0.3
            signal = "moderately_overvalued"
            details = f"EV/Revenue elevated at {ev_revenue:.1f}"
        else:
            score = 0.2
            signal = "overvalued"
            details = f"EV/Revenue expensive at {ev_revenue:.1f}"
        
        return {
            "score": float(score),
            "value": float(ev_revenue),
            "signal": signal,
            "details": details
        }
    
    def _analyze_peg_ratio(self, peg: float) -> Dict:
        """Analyze PEG ratio"""
        if peg <= 0:
            return {"score": 0.5, "value": peg, "signal": "neutral", "details": "Not available"}
        
        if peg < 0.5:
            score = 0.9
            signal = "very_attractive"
            details = f"PEG extremely attractive at {peg:.2f}"
        elif peg < 1.0:
            score = 0.8
            signal = "attractive"
            details = f"PEG attractive at {peg:.2f}"
        elif peg < 1.5:
            score = 0.6
            signal = "fair"
            details = f"PEG fair at {peg:.2f}"
        elif peg < 2.0:
            score = 0.4
            signal = "expensive"
            details = f"PEG elevated at {peg:.2f}"
        else:
            score = 0.2
            signal = "very_expensive"
            details = f"PEG expensive at {peg:.2f}"
        
        return {
            "score": float(score),
            "value": float(peg),
            "signal": signal,
            "details": details
        }