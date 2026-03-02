"""
Solvency Analyzer - Analyzes solvency and leverage metrics
"""
from typing import Dict, List, Optional, Any
import numpy as np
from utils.logger import logger as  logging

class SolvencyAnalyzer:
    """
    Analyzes solvency and leverage metrics:
    - Debt to Equity Ratio
    - Debt to Assets Ratio
    - Interest Coverage Ratio
    - Debt to EBITDA
    - Financial Leverage
    - Altman Z-Score (bankruptcy risk)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Solvency thresholds
        self.de_ideal = config.get("de_ideal", 0.5)
        self.de_max = config.get("de_max", 1.5)
        
        self.interest_coverage_min = config.get("interest_coverage_min", 3)
        self.interest_coverage_ideal = config.get("interest_coverage_ideal", 5)
        
        self.debt_ebitda_ideal = config.get("debt_ebitda_ideal", 2)
        self.debt_ebitda_max = config.get("debt_ebitda_max", 4)
        
        logging.info(f"✅ SolvencyAnalyzer initialized")
    
    async def analyze(self, info: Dict, benchmarks: Dict) -> Dict[str, Any]:
        """
        Analyze solvency metrics
        """
        results = {}
        scores = []
        
        # Debt to Equity
        de = info.get("debtToEquity")
        if de is not None:
            analysis = self._analyze_de_ratio(de, benchmarks.get("de_benchmark", 1.0))
            results["de_ratio"] = analysis
            scores.append(analysis["score"])
        
        # Debt to Assets
        da = self._calculate_debt_to_assets(info)
        if da is not None:
            analysis = self._analyze_da_ratio(da)
            results["da_ratio"] = analysis
            scores.append(analysis["score"])
        
        # Interest Coverage
        interest_coverage = info.get("interestCoverage")
        if interest_coverage is not None:
            analysis = self._analyze_interest_coverage(interest_coverage)
            results["interest_coverage"] = analysis
            scores.append(analysis["score"])
        
        # Debt to EBITDA
        debt_ebitda = info.get("debtToEbitda")
        if debt_ebitda is not None:
            analysis = self._analyze_debt_ebitda(debt_ebitda)
            results["debt_ebitda"] = analysis
            scores.append(analysis["score"])
        
        # Financial Leverage
        leverage = info.get("financialLeverage")
        if leverage is not None:
            analysis = self._analyze_leverage(leverage)
            results["leverage"] = analysis
            scores.append(analysis["score"])
        
        # Altman Z-Score
        z_score = self._calculate_altman_z(info)
        if z_score is not None:
            analysis = self._analyze_z_score(z_score)
            results["z_score"] = analysis
            scores.append(analysis["score"])
        
        # Calculate composite score
        if scores:
            composite_score = np.mean(scores)
        else:
            composite_score = 0.5
        
        # Determine overall signal
        if composite_score >= 0.7:
            signal = "very_strong"
        elif composite_score >= 0.6:
            signal = "strong"
        elif composite_score >= 0.4:
            signal = "moderate"
        else:
            signal = "high_risk"
        
        return {
            "score": float(composite_score),
            "signal": signal,
            "details": results
        }
    
    def _analyze_de_ratio(self, de: float, benchmark: float) -> Dict:
        """Analyze Debt to Equity ratio"""
        if de < 0:
            # Negative equity - unusual situation
            if de > -0.5:
                return {
                    "score": 0.3,
                    "value": float(de),
                    "signal": "negative_equity",
                    "details": f"Negative equity (D/E = {de:.2f}) - high risk"
                }
            else:
                return {
                    "score": 0.1,
                    "value": float(de),
                    "signal": "severely_negative",
                    "details": f"Severely negative equity (D/E = {de:.2f}) - extreme risk"
                }
        
        if de < benchmark * 0.5:
            score = 0.9
            signal = "very_low"
            details = f"Debt to Equity very low at {de:.2f}"
        elif de < benchmark:
            score = 0.8
            signal = "low"
            details = f"Debt to Equity low at {de:.2f}"
        elif de < benchmark * 1.5:
            score = 0.6
            signal = "moderate"
            details = f"Debt to Equity moderate at {de:.2f}"
        elif de < benchmark * 2:
            score = 0.4
            signal = "elevated"
            details = f"Debt to Equity elevated at {de:.2f}"
        else:
            score = 0.2
            signal = "high"
            details = f"Debt to Equity high at {de:.2f} - leverage concerns"
        
        return {
            "score": float(score),
            "value": float(de),
            "signal": signal,
            "details": details
        }
    
    def _analyze_da_ratio(self, da: float) -> Dict:
        """Analyze Debt to Assets ratio"""
        if da < 0.2:
            score = 0.9
            signal = "very_low"
            details = f"Debt to Assets very low at {da:.2f}"
        elif da < 0.3:
            score = 0.8
            signal = "low"
            details = f"Debt to Assets low at {da:.2f}"
        elif da < 0.4:
            score = 0.6
            signal = "moderate"
            details = f"Debt to Assets moderate at {da:.2f}"
        elif da < 0.5:
            score = 0.4
            signal = "elevated"
            details = f"Debt to Assets elevated at {da:.2f}"
        else:
            score = 0.2
            signal = "high"
            details = f"Debt to Assets high at {da:.2f} - leverage concerns"
        
        return {
            "score": float(score),
            "value": float(da),
            "signal": signal,
            "details": details
        }
    
    def _analyze_interest_coverage(self, coverage: float) -> Dict:
        """Analyze Interest Coverage ratio"""
        if coverage > self.interest_coverage_ideal * 2:
            score = 0.9
            signal = "exceptional"
            details = f"Interest coverage exceptional at {coverage:.2f}x"
        elif coverage > self.interest_coverage_ideal:
            score = 0.8
            signal = "strong"
            details = f"Interest coverage strong at {coverage:.2f}x"
        elif coverage > self.interest_coverage_min:
            score = 0.7
            signal = "adequate"
            details = f"Interest coverage adequate at {coverage:.2f}x"
        elif coverage > 1.5:
            score = 0.5
            signal = "weak"
            details = f"Interest coverage weak at {coverage:.2f}x - some risk"
        elif coverage > 0:
            score = 0.3
            signal = "very_weak"
            details = f"Interest coverage very weak at {coverage:.2f}x - high risk"
        else:
            score = 0.1
            signal = "critical"
            details = f"Interest coverage negative - extreme risk"
        
        return {
            "score": float(score),
            "value": float(coverage),
            "signal": signal,
            "details": details
        }
    
    def _analyze_debt_ebitda(self, debt_ebitda: float) -> Dict:
        """Analyze Debt to EBITDA ratio"""
        if debt_ebitda < 0:
            # Negative EBITDA
            return {
                "score": 0.1,
                "value": float(debt_ebitda),
                "signal": "negative_ebitda",
                "details": f"Negative EBITDA - extreme risk"
            }
        
        if debt_ebitda < 1:
            score = 0.9
            signal = "very_low"
            details = f"Debt to EBITDA very low at {debt_ebitda:.2f}x"
        elif debt_ebitda < self.debt_ebitda_ideal:
            score = 0.8
            signal = "low"
            details = f"Debt to EBITDA low at {debt_ebitda:.2f}x"
        elif debt_ebitda < self.debt_ebitda_max:
            score = 0.6
            signal = "moderate"
            details = f"Debt to EBITDA moderate at {debt_ebitda:.2f}x"
        elif debt_ebitda < 5:
            score = 0.4
            signal = "elevated"
            details = f"Debt to EBITDA elevated at {debt_ebitda:.2f}x"
        else:
            score = 0.2
            signal = "high"
            details = f"Debt to EBITDA high at {debt_ebitda:.2f}x - leverage concerns"
        
        return {
            "score": float(score),
            "value": float(debt_ebitda),
            "signal": signal,
            "details": details
        }
    
    def _analyze_leverage(self, leverage: float) -> Dict:
        """Analyze Financial Leverage"""
        if leverage < 1.5:
            score = 0.9
            signal = "very_low"
            details = f"Financial leverage very low at {leverage:.2f}x"
        elif leverage < 2:
            score = 0.8
            signal = "low"
            details = f"Financial leverage low at {leverage:.2f}x"
        elif leverage < 2.5:
            score = 0.6
            signal = "moderate"
            details = f"Financial leverage moderate at {leverage:.2f}x"
        elif leverage < 3:
            score = 0.4
            signal = "elevated"
            details = f"Financial leverage elevated at {leverage:.2f}x"
        else:
            score = 0.2
            signal = "high"
            details = f"Financial leverage high at {leverage:.2f}x - risk concerns"
        
        return {
            "score": float(score),
            "value": float(leverage),
            "signal": signal,
            "details": details
        }
    
    def _analyze_z_score(self, z_score: float) -> Dict:
        """Analyze Altman Z-Score for bankruptcy risk"""
        if z_score > 3:
            score = 0.9
            signal = "safe"
            details = f"Z-Score {z_score:.2f} - safe zone, very low bankruptcy risk"
        elif z_score > 2.7:
            score = 0.8
            signal = "safe"
            details = f"Z-Score {z_score:.2f} - safe zone, low bankruptcy risk"
        elif z_score > 1.8:
            score = 0.5
            signal = "grey"
            details = f"Z-Score {z_score:.2f} - grey zone, moderate bankruptcy risk"
        else:
            score = 0.2
            signal = "distress"
            details = f"Z-Score {z_score:.2f} - distress zone, high bankruptcy risk"
        
        return {
            "score": float(score),
            "value": float(z_score),
            "signal": signal,
            "details": details
        }
    
    def _calculate_debt_to_assets(self, info: Dict) -> Optional[float]:
        """
        Calculate Debt to Assets = Total Debt / Total Assets
        """
        debt = info.get("totalDebt")
        assets = info.get("totalAssets")
        
        if debt and assets and assets > 0:
            return debt / assets
        
        return None
    
    def _calculate_altman_z(self, info: Dict) -> Optional[float]:
        """
        Calculate Altman Z-Score (simplified version)
        Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
        where:
        A = Working Capital / Total Assets
        B = Retained Earnings / Total Assets
        C = EBIT / Total Assets
        D = Market Value of Equity / Total Liabilities
        E = Sales / Total Assets
        """
        try:
            # Get required data
            current_assets = info.get("totalCurrentAssets", 0)
            current_liabilities = info.get("totalCurrentLiabilities", 0)
            total_assets = info.get("totalAssets", 1)  # Avoid division by zero
            retained_earnings = info.get("retainedEarnings", 0)
            ebit = info.get("ebit", info.get("operatingIncome", 0))
            market_cap = info.get("marketCap", 0)
            total_liabilities = info.get("totalLiabilities", 1)
            revenue = info.get("totalRevenue", 0)
            
            # Calculate components
            A = (current_assets - current_liabilities) / total_assets
            B = retained_earnings / total_assets
            C = ebit / total_assets
            D = market_cap / total_liabilities
            E = revenue / total_assets
            
            # Calculate Z-Score
            z_score = (1.2 * A) + (1.4 * B) + (3.3 * C) + (0.6 * D) + (1.0 * E)
            
            return float(z_score)
            
        except Exception as e:
            logging.debug(f"Could not calculate Z-Score: {e}")
            return None