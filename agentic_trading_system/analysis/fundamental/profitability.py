"""
Profitability Analyzer - Analyzes profitability metrics
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as  logging

class ProfitabilityAnalyzer:
    """
    Analyzes profitability metrics:
    - ROE (Return on Equity)
    - ROA (Return on Assets)
    - ROIC (Return on Invested Capital)
    - Gross Margin
    - Operating Margin
    - Net Profit Margin
    - EBITDA Margin
    - FCF Margin
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Profitability thresholds
        self.roe_excellent = config.get("roe_excellent", 20)  # 20%+
        self.roe_good = config.get("roe_good", 15)  # 15-20%
        self.roe_fair = config.get("roe_fair", 10)  # 10-15%
        
        self.margin_excellent = config.get("margin_excellent", 20)  # 20%+
        self.margin_good = config.get("margin_good", 15)  # 15-20%
        self.margin_fair = config.get("margin_fair", 10)  # 10-15%
        
        logging.info(f"✅ ProfitabilityAnalyzer initialized")
    
    async def analyze(self, info: Dict, benchmarks: Dict) -> Dict[str, Any]:
        """
        Analyze profitability metrics
        """
        results = {}
        scores = []
        
        # Return on Equity
        roe = info.get("returnOnEquity")
        if roe is not None:
            analysis = self._analyze_roe(roe * 100, benchmarks.get("roe_benchmark", 15))
            results["roe"] = analysis
            scores.append(analysis["score"])
        
        # Return on Assets
        roa = info.get("returnOnAssets")
        if roa is not None:
            analysis = self._analyze_roa(roa * 100)
            results["roa"] = analysis
            scores.append(analysis["score"])
        
        # Return on Invested Capital
        roic = info.get("returnOnCapital")
        if roic is not None:
            analysis = self._analyze_roic(roic * 100)
            results["roic"] = analysis
            scores.append(analysis["score"])
        
        # Gross Margin
        gross_margin = info.get("grossMargins")
        if gross_margin is not None:
            analysis = self._analyze_gross_margin(gross_margin * 100)
            results["gross_margin"] = analysis
            scores.append(analysis["score"])
        
        # Operating Margin
        op_margin = info.get("operatingMargins")
        if op_margin is not None:
            analysis = self._analyze_operating_margin(op_margin * 100)
            results["operating_margin"] = analysis
            scores.append(analysis["score"])
        
        # Net Profit Margin
        net_margin = info.get("profitMargins")
        if net_margin is not None:
            analysis = self._analyze_net_margin(net_margin * 100)
            results["net_margin"] = analysis
            scores.append(analysis["score"])
        
        # EBITDA Margin
        ebitda_margin = info.get("ebitdaMargins")
        if ebitda_margin is not None:
            analysis = self._analyze_ebitda_margin(ebitda_margin * 100)
            results["ebitda_margin"] = analysis
            scores.append(analysis["score"])
        
        # FCF Margin
        fcf_margin = self._calculate_fcf_margin(info)
        if fcf_margin is not None:
            analysis = self._analyze_fcf_margin(fcf_margin)
            results["fcf_margin"] = analysis
            scores.append(analysis["score"])
        
        # Margin Trend (if we can calculate)
        margin_trend = self._analyze_margin_trend(info)
        if margin_trend:
            results["margin_trend"] = margin_trend
            scores.append(margin_trend["score"])
        
        # Calculate composite score
        if scores:
            composite_score = np.mean(scores)
        else:
            composite_score = 0.5
        
        # Determine overall signal
        if composite_score >= 0.7:
            signal = "highly_profitable"
        elif composite_score >= 0.6:
            signal = "profitable"
        elif composite_score >= 0.4:
            signal = "moderately_profitable"
        else:
            signal = "low_profitability"
        
        return {
            "score": float(composite_score),
            "signal": signal,
            "details": results
        }
    
    def _analyze_roe(self, roe: float, benchmark: float) -> Dict:
        """Analyze Return on Equity"""
        if roe > benchmark * 1.5:
            score = 0.9
            signal = "exceptional"
            details = f"ROE exceptional at {roe:.1f}%"
        elif roe > benchmark:
            score = 0.8
            signal = "strong"
            details = f"ROE strong at {roe:.1f}%"
        elif roe > benchmark * 0.7:
            score = 0.7
            signal = "good"
            details = f"ROE good at {roe:.1f}%"
        elif roe > benchmark * 0.5:
            score = 0.6
            signal = "fair"
            details = f"ROE fair at {roe:.1f}%"
        elif roe > 0:
            score = 0.5
            signal = "positive"
            details = f"ROE positive at {roe:.1f}%"
        elif roe > -10:
            score = 0.3
            signal = "negative"
            details = f"ROE negative at {roe:.1f}%"
        else:
            score = 0.1
            signal = "severely_negative"
            details = f"ROE severely negative at {roe:.1f}%"
        
        return {
            "score": float(score),
            "value": float(roe),
            "signal": signal,
            "details": details
        }
    
    def _analyze_roa(self, roa: float) -> Dict:
        """Analyze Return on Assets"""
        if roa > 15:
            score = 0.9
            signal = "exceptional"
            details = f"ROA exceptional at {roa:.1f}%"
        elif roa > 10:
            score = 0.8
            signal = "strong"
            details = f"ROA strong at {roa:.1f}%"
        elif roa > 7:
            score = 0.7
            signal = "good"
            details = f"ROA good at {roa:.1f}%"
        elif roa > 5:
            score = 0.6
            signal = "fair"
            details = f"ROA fair at {roa:.1f}%"
        elif roa > 0:
            score = 0.5
            signal = "positive"
            details = f"ROA positive at {roa:.1f}%"
        elif roa > -5:
            score = 0.3
            signal = "negative"
            details = f"ROA negative at {roa:.1f}%"
        else:
            score = 0.1
            signal = "severely_negative"
            details = f"ROA severely negative at {roa:.1f}%"
        
        return {
            "score": float(score),
            "value": float(roa),
            "signal": signal,
            "details": details
        }
    
    def _analyze_roic(self, roic: float) -> Dict:
        """Analyze Return on Invested Capital"""
        if roic > 20:
            score = 0.9
            signal = "exceptional"
            details = f"ROIC exceptional at {roic:.1f}%"
        elif roic > 15:
            score = 0.8
            signal = "strong"
            details = f"ROIC strong at {roic:.1f}%"
        elif roic > 10:
            score = 0.7
            signal = "good"
            details = f"ROIC good at {roic:.1f}%"
        elif roic > 7:
            score = 0.6
            signal = "fair"
            details = f"ROIC fair at {roic:.1f}%"
        elif roic > 0:
            score = 0.5
            signal = "positive"
            details = f"ROIC positive at {roic:.1f}%"
        else:
            score = 0.2
            signal = "negative"
            details = f"ROIC negative at {roic:.1f}%"
        
        return {
            "score": float(score),
            "value": float(roic),
            "signal": signal,
            "details": details
        }
    
    def _analyze_gross_margin(self, margin: float) -> Dict:
        """Analyze Gross Margin"""
        if margin > 60:
            score = 0.9
            signal = "exceptional"
            details = f"Gross margin exceptional at {margin:.1f}%"
        elif margin > 50:
            score = 0.8
            signal = "strong"
            details = f"Gross margin strong at {margin:.1f}%"
        elif margin > 40:
            score = 0.7
            signal = "good"
            details = f"Gross margin good at {margin:.1f}%"
        elif margin > 30:
            score = 0.6
            signal = "fair"
            details = f"Gross margin fair at {margin:.1f}%"
        elif margin > 20:
            score = 0.5
            signal = "moderate"
            details = f"Gross margin moderate at {margin:.1f}%"
        elif margin > 10:
            score = 0.4
            signal = "low"
            details = f"Gross margin low at {margin:.1f}%"
        else:
            score = 0.2
            signal = "very_low"
            details = f"Gross margin very low at {margin:.1f}%"
        
        return {
            "score": float(score),
            "value": float(margin),
            "signal": signal,
            "details": details
        }
    
    def _analyze_operating_margin(self, margin: float) -> Dict:
        """Analyze Operating Margin"""
        if margin > 30:
            score = 0.9
            signal = "exceptional"
            details = f"Operating margin exceptional at {margin:.1f}%"
        elif margin > 20:
            score = 0.8
            signal = "strong"
            details = f"Operating margin strong at {margin:.1f}%"
        elif margin > 15:
            score = 0.7
            signal = "good"
            details = f"Operating margin good at {margin:.1f}%"
        elif margin > 10:
            score = 0.6
            signal = "fair"
            details = f"Operating margin fair at {margin:.1f}%"
        elif margin > 5:
            score = 0.5
            signal = "moderate"
            details = f"Operating margin moderate at {margin:.1f}%"
        elif margin > 0:
            score = 0.4
            signal = "low"
            details = f"Operating margin low at {margin:.1f}%"
        else:
            score = 0.1
            signal = "negative"
            details = f"Operating margin negative at {margin:.1f}%"
        
        return {
            "score": float(score),
            "value": float(margin),
            "signal": signal,
            "details": details
        }
    
    def _analyze_net_margin(self, margin: float) -> Dict:
        """Analyze Net Profit Margin"""
        if margin > 25:
            score = 0.9
            signal = "exceptional"
            details = f"Net margin exceptional at {margin:.1f}%"
        elif margin > 20:
            score = 0.8
            signal = "strong"
            details = f"Net margin strong at {margin:.1f}%"
        elif margin > 15:
            score = 0.7
            signal = "good"
            details = f"Net margin good at {margin:.1f}%"
        elif margin > 10:
            score = 0.6
            signal = "fair"
            details = f"Net margin fair at {margin:.1f}%"
        elif margin > 5:
            score = 0.5
            signal = "moderate"
            details = f"Net margin moderate at {margin:.1f}%"
        elif margin > 0:
            score = 0.4
            signal = "low"
            details = f"Net margin low at {margin:.1f}%"
        else:
            score = 0.1
            signal = "negative"
            details = f"Net margin negative at {margin:.1f}%"
        
        return {
            "score": float(score),
            "value": float(margin),
            "signal": signal,
            "details": details
        }
    
    def _analyze_ebitda_margin(self, margin: float) -> Dict:
        """Analyze EBITDA Margin"""
        if margin > 35:
            score = 0.9
            signal = "exceptional"
            details = f"EBITDA margin exceptional at {margin:.1f}%"
        elif margin > 25:
            score = 0.8
            signal = "strong"
            details = f"EBITDA margin strong at {margin:.1f}%"
        elif margin > 20:
            score = 0.7
            signal = "good"
            details = f"EBITDA margin good at {margin:.1f}%"
        elif margin > 15:
            score = 0.6
            signal = "fair"
            details = f"EBITDA margin fair at {margin:.1f}%"
        elif margin > 10:
            score = 0.5
            signal = "moderate"
            details = f"EBITDA margin moderate at {margin:.1f}%"
        elif margin > 5:
            score = 0.4
            signal = "low"
            details = f"EBITDA margin low at {margin:.1f}%"
        else:
            score = 0.2
            signal = "very_low"
            details = f"EBITDA margin very low at {margin:.1f}%"
        
        return {
            "score": float(score),
            "value": float(margin),
            "signal": signal,
            "details": details
        }
    
    def _analyze_fcf_margin(self, margin: float) -> Dict:
        """Analyze Free Cash Flow Margin"""
        if margin > 20:
            score = 0.9
            signal = "exceptional"
            details = f"FCF margin exceptional at {margin:.1f}%"
        elif margin > 15:
            score = 0.8
            signal = "strong"
            details = f"FCF margin strong at {margin:.1f}%"
        elif margin > 10:
            score = 0.7
            signal = "good"
            details = f"FCF margin good at {margin:.1f}%"
        elif margin > 5:
            score = 0.6
            signal = "fair"
            details = f"FCF margin fair at {margin:.1f}%"
        elif margin > 0:
            score = 0.5
            signal = "positive"
            details = f"FCF margin positive at {margin:.1f}%"
        else:
            score = 0.2
            signal = "negative"
            details = f"FCF margin negative at {margin:.1f}%"
        
        return {
            "score": float(score),
            "value": float(margin),
            "signal": signal,
            "details": details
        }
    
    def _analyze_margin_trend(self, info: Dict) -> Optional[Dict]:
        """
        Analyze trend in margins
        This would ideally use historical data
        """
        # Placeholder - would need historical margin data
        return {
            "score": 0.5,
            "signal": "stable",
            "details": "Margin trend analysis not available"
        }
    
    def _calculate_fcf_margin(self, info: Dict) -> Optional[float]:
        """Calculate Free Cash Flow Margin"""
        # FCF Margin = Free Cash Flow / Revenue
        fcf = info.get("freeCashflow")
        revenue = info.get("totalRevenue")
        
        if fcf and revenue and revenue > 0:
            return (fcf / revenue) * 100
        
        return None