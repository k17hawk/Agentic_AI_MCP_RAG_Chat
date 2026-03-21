"""
Diversification Score - Measures portfolio diversification
"""
from typing import Dict, Any, Optional, List
import numpy as np
from agentic_trading_system.utils.logger import logger as logging

class DiversificationScore:
    """
    Diversification Score - Measures how well diversified a portfolio is
    
    Factors:
    - Number of assets
    - Correlation between assets
    - Sector concentration
    - Position size concentration
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Weights for different factors
        self.weights = config.get("weights", {
            "num_assets": 0.3,
            "correlation": 0.3,
            "sector_concentration": 0.2,
            "position_concentration": 0.2
        })
        
        # Targets
        self.ideal_num_assets = config.get("ideal_num_assets", 20)
        self.max_sector_exposure = config.get("max_sector_exposure", 0.3)  # 30% max per sector
        
        logging.info(f"✅ DiversificationScore initialized")
    
    def calculate(self, positions: List[Dict], correlations: np.ndarray = None,
                 sectors: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Calculate diversification score (0-100)
        """
        if not positions:
            return {"score": 0, "message": "No positions"}
        
        # Calculate component scores
        num_assets_score = self._score_num_assets(len(positions))
        correlation_score = self._score_correlation(positions, correlations)
        sector_score = self._score_sector_concentration(positions, sectors)
        concentration_score = self._score_position_concentration(positions)
        
        # Weighted average
        total_score = (
            num_assets_score * self.weights["num_assets"] +
            correlation_score * self.weights["correlation"] +
            sector_score * self.weights["sector_concentration"] +
            concentration_score * self.weights["position_concentration"]
        )
        
        # Calculate individual metrics
        metrics = {
            "num_assets": len(positions),
            "herfindahl_index": self._calculate_herfindahl(positions),
            "max_position": max(p.get("weight", 0) for p in positions) if positions else 0,
            "min_position": min(p.get("weight", 0) for p in positions) if positions else 0
        }
        
        if sectors:
            metrics["sector_exposures"] = self._calculate_sector_exposures(positions, sectors)
        
        return {
            "score": float(total_score),
            "components": {
                "num_assets_score": float(num_assets_score),
                "correlation_score": float(correlation_score),
                "sector_score": float(sector_score),
                "concentration_score": float(concentration_score)
            },
            "metrics": metrics,
            "weights_used": self.weights,
            "interpretation": self._interpret_score(total_score)
        }
    
    def _score_num_assets(self, num_assets: int) -> float:
        """Score based on number of assets"""
        if num_assets >= self.ideal_num_assets:
            return 1.0
        elif num_assets >= self.ideal_num_assets * 0.5:
            return 0.7
        elif num_assets >= self.ideal_num_assets * 0.25:
            return 0.4
        elif num_assets >= 5:
            return 0.2
        else:
            return 0.0
    
    def _score_correlation(self, positions: List[Dict], 
                          correlations: np.ndarray = None) -> float:
        """
        Score based on average correlation between assets
        """
        if correlations is None or len(correlations) < 2:
            return 0.5  # Neutral if no correlation data
        
        # Calculate average absolute correlation
        triu_indices = np.triu_indices_from(correlations, k=1)
        avg_corr = np.mean(np.abs(correlations[triu_indices]))
        
        # Score: lower correlation is better
        if avg_corr < 0.3:
            return 1.0
        elif avg_corr < 0.5:
            return 0.7
        elif avg_corr < 0.7:
            return 0.4
        else:
            return 0.1
    
    def _score_sector_concentration(self, positions: List[Dict],
                                   sectors: Dict[str, str] = None) -> float:
        """
        Score based on sector concentration
        """
        if not sectors:
            return 0.5
        
        sector_exposures = self._calculate_sector_exposures(positions, sectors)
        
        # Check if any sector exceeds max exposure
        max_exposure = max(sector_exposures.values()) if sector_exposures else 0
        
        if max_exposure <= self.max_sector_exposure:
            return 1.0
        elif max_exposure <= self.max_sector_exposure * 1.5:
            return 0.6
        elif max_exposure <= self.max_sector_exposure * 2:
            return 0.3
        else:
            return 0.0
    
    def _score_position_concentration(self, positions: List[Dict]) -> float:
        """
        Score based on position size concentration using Herfindahl index
        """
        herfindahl = self._calculate_herfindahl(positions)
        
        # Lower Herfindahl = better diversification
        if herfindahl < 0.1:
            return 1.0
        elif herfindahl < 0.2:
            return 0.8
        elif herfindahl < 0.3:
            return 0.6
        elif herfindahl < 0.4:
            return 0.4
        elif herfindahl < 0.5:
            return 0.2
        else:
            return 0.0
    
    def _calculate_herfindahl(self, positions: List[Dict]) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration
        """
        weights = [p.get("weight", 0) for p in positions]
        total = sum(weights)
        
        if total == 0:
            return 1.0
        
        hhi = sum((w / total) ** 2 for w in weights)
        return float(hhi)
    
    def _calculate_sector_exposures(self, positions: List[Dict],
                                   sectors: Dict[str, str]) -> Dict[str, float]:
        """
        Calculate exposure by sector
        """
        sector_exposures = {}
        total_weight = 0
        
        for position in positions:
            symbol = position.get("symbol")
            weight = position.get("weight", 0)
            
            if symbol in sectors:
                sector = sectors[symbol]
                sector_exposures[sector] = sector_exposures.get(sector, 0) + weight
                total_weight += weight
        
        # Normalize
        if total_weight > 0:
            for sector in sector_exposures:
                sector_exposures[sector] /= total_weight
        
        return sector_exposures
    
    def _interpret_score(self, score: float) -> str:
        """Get human-readable interpretation"""
        if score >= 0.8:
            return "Excellent diversification"
        elif score >= 0.6:
            return "Good diversification"
        elif score >= 0.4:
            return "Moderate diversification"
        elif score >= 0.2:
            return "Poor diversification - concentrated"
        else:
            return "Very poor diversification - highly concentrated"