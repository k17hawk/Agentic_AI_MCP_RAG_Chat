"""
Correlation Matrix - Calculates correlations between assets
"""
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from agentic_trading_system.utils.logger import logger as logging

class CorrelationMatrix:
    """
    Correlation Matrix - Calculates and analyzes asset correlations
    
    Used for:
    - Diversification analysis
    - Risk concentration
    - Pair trading
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default parameters
        self.lookback_period = config.get("lookback_period", 60)  # 60 days
        self.min_history = config.get("min_history", 30)
        
        logging.info(f"✅ CorrelationMatrix initialized")
    
    def calculate(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Calculate correlation matrix from price data
        """
        # Calculate returns for each asset
        returns_dict = {}
        valid_assets = []
        
        for symbol, data in price_data.items():
            if data is not None and len(data) >= self.min_history:
                returns = data['Close'].pct_change().dropna()
                if len(returns) >= self.min_history:
                    returns_dict[symbol] = returns[-self.lookback_period:]
                    valid_assets.append(symbol)
        
        if len(valid_assets) < 2:
            return {
                "error": "Insufficient assets for correlation",
                "matrix": None
            }
        
        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        # Calculate average correlation
        triu_indices = np.triu_indices_from(corr_matrix, k=1)
        avg_correlation = np.mean(corr_matrix.values[triu_indices])
        
        # Find highly correlated pairs
        high_corr_pairs = []
        symbols = corr_matrix.columns
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.8:
                    high_corr_pairs.append({
                        "asset1": symbols[i],
                        "asset2": symbols[j],
                        "correlation": float(corr)
                    })
        
        # Calculate diversification score
        diversification_score = 1 - avg_correlation
        
        return {
            "matrix": corr_matrix.to_dict(),
            "average_correlation": float(avg_correlation),
            "diversification_score": float(diversification_score),
            "high_correlation_pairs": high_corr_pairs,
            "assets": valid_assets,
            "num_assets": len(valid_assets),
            "lookback_period": self.lookback_period
        }
    
    def get_pair_correlation(self, symbol1: str, symbol2: str,
                            price_data: Dict[str, pd.DataFrame]) -> float:
        """
        Get correlation between two specific assets
        """
        if symbol1 not in price_data or symbol2 not in price_data:
            return 0.0
        
        data1 = price_data[symbol1]
        data2 = price_data[symbol2]
        
        if data1 is None or data2 is None or len(data1) < self.min_history or len(data2) < self.min_history:
            return 0.0
        
        returns1 = data1['Close'].pct_change().dropna()[-self.lookback_period:]
        returns2 = data2['Close'].pct_change().dropna()[-self.lookback_period:]
        
        # Align dates
        common_dates = returns1.index.intersection(returns2.index)
        if len(common_dates) < self.min_history:
            return 0.0
        
        aligned_returns1 = returns1[common_dates]
        aligned_returns2 = returns2[common_dates]
        
        correlation = aligned_returns1.corr(aligned_returns2)
        
        return float(correlation) if not pd.isna(correlation) else 0.0