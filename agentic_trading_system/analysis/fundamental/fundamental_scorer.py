"""
Fundamental Scorer - Scores and ranks fundamental analysis results
"""
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from utils.logger import logger as logging

class FundamentalScorer:
    """
    Scores fundamental analysis results and generates final fundamental score
    
    Weights:
    - Valuation: 25%
    - Growth: 25%
    - Profitability: 20%
    - Solvency: 15%
    - Liquidity: 10%
    - Efficiency: 5%
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Category weights
        self.weights = config.get("fundamental_weights", {
            "valuation": 0.25,
            "growth": 0.25,
            "profitability": 0.20,
            "solvency": 0.15,
            "liquidity": 0.10,
            "efficiency": 0.05
        })
        
        # Score thresholds
        self.strong_buy_threshold = config.get("strong_buy_threshold", 0.8)
        self.buy_threshold = config.get("buy_threshold", 0.65)
        self.watch_threshold = config.get("watch_threshold", 0.5)
        self.sell_threshold = config.get("sell_threshold", 0.35)
        
        # Quality thresholds
        self.min_data_quality = config.get("min_data_quality", 0.5)
        
        logging.info(f"✅ FundamentalScorer initialized")
    
    def calculate_score(self, analysis_results: Dict[str, Any], 
                       sector: str, benchmarks: Dict) -> tuple[float, Dict]:
        """
        Calculate overall fundamental score from all analysis components
        """
        # Extract component scores
        valuation = analysis_results.get("valuation", {})
        growth = analysis_results.get("growth", {})
        profitability = analysis_results.get("profitability", {})
        solvency = analysis_results.get("solvency", {})
        liquidity = analysis_results.get("liquidity", {})
        efficiency = analysis_results.get("efficiency", {})
        dcf = analysis_results.get("dcf", {})
        
        # Get individual scores
        valuation_score = valuation.get("score", 0.5)
        growth_score = growth.get("score", 0.5)
        profitability_score = profitability.get("score", 0.5)
        solvency_score = solvency.get("score", 0.5)
        liquidity_score = liquidity.get("score", 0.5)
        efficiency_score = efficiency.get("score", 0.5)
        dcf_score = dcf.get("score", 0.5)
        
        # Calculate weighted base score
        base_score = (
            valuation_score * self.weights["valuation"] +
            growth_score * self.weights["growth"] +
            profitability_score * self.weights["profitability"] +
            solvency_score * self.weights["solvency"] +
            liquidity_score * self.weights["liquidity"] +
            efficiency_score * self.weights["efficiency"]
        )
        
        # DCF bonus (if available)
        dcf_bonus = (dcf_score - 0.5) * 0.1  # Max +/- 0.05
        final_score = min(1.0, max(0.0, base_score + dcf_bonus))
        
        # Determine signal
        signal = self._determine_signal(final_score, dcf_score)
        
        # Calculate confidence
        confidence = self._calculate_confidence(analysis_results)
        
        # Calculate quality score
        quality = self._calculate_quality(analysis_results)
        
        # Generate strengths and weaknesses
        strengths, weaknesses = self._analyze_strengths_weaknesses(analysis_results)
        
        # Generate key metrics summary
        key_metrics = self._extract_key_metrics(analysis_results)
        
        # Compare to sector benchmarks
        sector_comparison = self._compare_to_sector(analysis_results, benchmarks)
        
        details = {
            "final_score": float(final_score),
            "base_score": float(base_score),
            "dcf_bonus": float(dcf_bonus),
            "signal": signal,
            "confidence": float(confidence),
            "quality": float(quality),
            "component_scores": {
                "valuation": float(valuation_score),
                "growth": float(growth_score),
                "profitability": float(profitability_score),
                "solvency": float(solvency_score),
                "liquidity": float(liquidity_score),
                "efficiency": float(efficiency_score),
                "dcf": float(dcf_score)
            },
            "strengths": strengths,
            "weaknesses": weaknesses,
            "key_metrics": key_metrics,
            "sector_comparison": sector_comparison,
            "sector": sector,
            "timestamp": datetime.now().isoformat()
        }
        
        return final_score, details
    
    def _determine_signal(self, score: float, dcf_score: float) -> str:
        """
        Determine fundamental signal based on score
        """
        if score >= self.strong_buy_threshold:
            if dcf_score > 0.6:
                return "STRONG_BUY"
            else:
                return "BUY"
        elif score >= self.buy_threshold:
            return "BUY"
        elif score >= self.watch_threshold:
            return "WATCH"
        elif score >= self.sell_threshold:
            return "SELL"
        else:
            return "STRONG_SELL"
    
    def _calculate_confidence(self, analysis_results: Dict) -> float:
        """
        Calculate confidence level in the score
        """
        confidence_factors = []
        
        # Check data availability
        categories = ["valuation", "growth", "profitability", "solvency", "liquidity", "efficiency"]
        available = sum(1 for cat in categories if cat in analysis_results)
        confidence_factors.append(available / len(categories))
        
        # Check score consistency
        scores = []
        for cat in categories:
            if cat in analysis_results:
                scores.append(analysis_results[cat].get("score", 0.5))
        
        if len(scores) > 1:
            consistency = 1 - np.std(scores)
            confidence_factors.append(consistency)
        
        # DCF availability bonus
        if "dcf" in analysis_results and analysis_results["dcf"].get("value"):
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        return float(np.mean(confidence_factors))
    
    def _calculate_quality(self, analysis_results: Dict) -> float:
        """
        Calculate quality of fundamental data
        """
        quality_factors = []
        
        # Check depth of analysis for each category
        for category, data in analysis_results.items():
            if isinstance(data, dict):
                details = data.get("details", {})
                if details:
                    # Count number of metrics available
                    metric_count = len([v for v in details.values() if v is not None])
                    quality_factors.append(min(1.0, metric_count / 5))
        
        if quality_factors:
            return float(np.mean(quality_factors))
        
        return 0.5
    
    def _analyze_strengths_weaknesses(self, analysis_results: Dict) -> tuple[List, List]:
        """
        Identify key strengths and weaknesses
        """
        strengths = []
        weaknesses = []
        
        # Valuation strengths/weaknesses
        valuation = analysis_results.get("valuation", {}).get("details", {})
        for metric, data in valuation.items():
            if isinstance(data, dict):
                score = data.get("score", 0.5)
                if score >= 0.7:
                    strengths.append(f"Attractive {metric.replace('_', ' ')}")
                elif score <= 0.3:
                    weaknesses.append(f"Expensive {metric.replace('_', ' ')}")
        
        # Growth strengths/weaknesses
        growth = analysis_results.get("growth", {}).get("details", {})
        for metric, data in growth.items():
            if isinstance(data, dict):
                score = data.get("score", 0.5)
                if score >= 0.7:
                    strengths.append(f"Strong {metric.replace('_', ' ')}")
                elif score <= 0.3:
                    weaknesses.append(f"Weak {metric.replace('_', ' ')}")
        
        # Profitability strengths/weaknesses
        profitability = analysis_results.get("profitability", {}).get("details", {})
        for metric, data in profitability.items():
            if isinstance(data, dict):
                score = data.get("score", 0.5)
                if score >= 0.7:
                    strengths.append(f"High {metric.replace('_', ' ')}")
                elif score <= 0.3:
                    weaknesses.append(f"Low {metric.replace('_', ' ')}")
        
        # Solvency strengths/weaknesses
        solvency = analysis_results.get("solvency", {}).get("details", {})
        for metric, data in solvency.items():
            if isinstance(data, dict):
                score = data.get("score", 0.5)
                if score >= 0.7:
                    strengths.append(f"Low {metric.replace('_', ' ')}")
                elif score <= 0.3:
                    weaknesses.append(f"High {metric.replace('_', ' ')}")
        
        return strengths[:5], weaknesses[:5]  # Top 5 each
    
    def _extract_key_metrics(self, analysis_results: Dict) -> Dict[str, Any]:
        """
        Extract key fundamental metrics for display
        """
        metrics = {}
        
        # Valuation metrics
        valuation = analysis_results.get("valuation", {}).get("details", {})
        if "pe_ratio" in valuation:
            metrics["P/E"] = valuation["pe_ratio"].get("value")
        if "pb_ratio" in valuation:
            metrics["P/B"] = valuation["pb_ratio"].get("value")
        
        # Growth metrics
        growth = analysis_results.get("growth", {}).get("details", {})
        if "revenue_growth" in growth:
            metrics["Revenue Growth"] = growth["revenue_growth"].get("value")
        if "eps_growth" in growth:
            metrics["EPS Growth"] = growth["eps_growth"].get("value")
        
        # Profitability metrics
        profitability = analysis_results.get("profitability", {}).get("details", {})
        if "roe" in profitability:
            metrics["ROE"] = profitability["roe"].get("value")
        if "net_margin" in profitability:
            metrics["Net Margin"] = profitability["net_margin"].get("value")
        
        # Solvency metrics
        solvency = analysis_results.get("solvency", {}).get("details", {})
        if "de_ratio" in solvency:
            metrics["D/E"] = solvency["de_ratio"].get("value")
        
        # DCF metrics
        dcf = analysis_results.get("dcf", {})
        if "margin_of_safety" in dcf:
            metrics["Margin of Safety"] = dcf.get("margin_of_safety")
        if "intrinsic_value" in dcf:
            metrics["Intrinsic Value"] = dcf.get("intrinsic_value")
        
        return metrics
    
    def _compare_to_sector(self, analysis_results: Dict, benchmarks: Dict) -> Dict[str, Any]:
        """
        Compare company metrics to sector benchmarks
        """
        comparison = {}
        
        # P/E comparison
        valuation = analysis_results.get("valuation", {}).get("details", {})
        if "pe_ratio" in valuation and "pe_benchmark" in benchmarks:
            pe = valuation["pe_ratio"].get("value")
            bench = benchmarks["pe_benchmark"]
            if pe and bench:
                comparison["P/E vs Sector"] = {
                    "company": float(pe),
                    "sector": float(bench),
                    "ratio": float(pe / bench if bench > 0 else 1)
                }
        
        # P/B comparison
        if "pb_ratio" in valuation and "pb_benchmark" in benchmarks:
            pb = valuation["pb_ratio"].get("value")
            bench = benchmarks["pb_benchmark"]
            if pb and bench:
                comparison["P/B vs Sector"] = {
                    "company": float(pb),
                    "sector": float(bench),
                    "ratio": float(pb / bench if bench > 0 else 1)
                }
        
        # ROE comparison
        profitability = analysis_results.get("profitability", {}).get("details", {})
        if "roe" in profitability and "roe_benchmark" in benchmarks:
            roe = profitability["roe"].get("value")
            bench = benchmarks["roe_benchmark"]
            if roe and bench:
                comparison["ROE vs Sector"] = {
                    "company": float(roe),
                    "sector": float(bench),
                    "ratio": float(roe / bench if bench > 0 else 1)
                }
        
        return comparison
    
    def get_investment_thesis(self, score: float, details: Dict) -> str:
        """
        Generate investment thesis based on fundamental analysis
        """
        signal = details.get("signal", "WATCH")
        strengths = details.get("strengths", [])
        weaknesses = details.get("weaknesses", [])
        
        if signal in ["STRONG_BUY", "BUY"]:
            thesis = f"Strong fundamental case with {len(strengths)} key strengths: "
            thesis += ", ".join(strengths[:3])
            if weaknesses:
                thesis += f". Key risks: {', '.join(weaknesses[:2])}"
        
        elif signal in ["SELL", "STRONG_SELL"]:
            thesis = f"Weak fundamentals with {len(weaknesses)} key concerns: "
            thesis += ", ".join(weaknesses[:3])
        
        else:
            thesis = "Mixed fundamentals. "
            if strengths:
                thesis += f"Strengths: {', '.join(strengths[:2])}. "
            if weaknesses:
                thesis += f"Concerns: {', '.join(weaknesses[:2])}"
        
        return thesis
    
    def get_rating_summary(self, score: float) -> Dict[str, Any]:
        """
        Get rating summary with recommendation
        """
        if score >= self.strong_buy_threshold:
            return {
                "rating": "A",
                "recommendation": "Strong Buy",
                "description": "Exceptionally strong fundamentals"
            }
        elif score >= self.buy_threshold:
            return {
                "rating": "B",
                "recommendation": "Buy",
                "description": "Attractive fundamentals"
            }
        elif score >= self.watch_threshold:
            return {
                "rating": "C",
                "recommendation": "Watch",
                "description": "Fair fundamentals"
            }
        elif score >= self.sell_threshold:
            return {
                "rating": "D",
                "recommendation": "Sell",
                "description": "Weak fundamentals"
            }
        else:
            return {
                "rating": "F",
                "recommendation": "Strong Sell",
                "description": "Very weak fundamentals"
            }