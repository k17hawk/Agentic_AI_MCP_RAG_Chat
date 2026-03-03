"""
Factor Attribution - Analyzes performance attribution to risk factors
"""
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime
from utils.logger import logger as logging

class FactorAttribution:
    """
    Factor Attribution - Analyzes performance attribution to risk factors
    
    Features:
    - Factor model analysis
    - Beta exposure
    - Factor returns
    - Residual analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Default factors
        self.default_factors = config.get("default_factors", [
            "market", "size", "value", "momentum", "quality", "volatility"
        ])
        
        # Risk-free rate
        self.risk_free_rate = config.get("risk_free_rate", 0.02)
        
        logging.info(f"✅ FactorAttribution initialized")
    
    def calculate_factor_exposure(self, returns: List[float],
                                  factor_returns: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate factor exposures (betas) using linear regression
        """
        if len(returns) < len(self.default_factors) + 5:
            return {"error": "Insufficient data for factor analysis"}
        
        import statsmodels.api as sm
        
        # Prepare data
        y = np.array(returns)
        X_dict = {name: np.array(factor_returns[name]) for name in self.default_factors}
        
        # Ensure all factors have same length
        min_len = min(len(y), *[len(x) for x in X_dict.values()])
        y = y[-min_len:]
        X = np.column_stack([X_dict[name][-min_len:] for name in self.default_factors])
        
        # Add constant for alpha
        X = sm.add_constant(X)
        
        # Fit regression
        model = sm.OLS(y, X).fit()
        
        # Extract results
        exposures = {}
        for i, factor in enumerate(self.default_factors):
            exposures[factor] = {
                "beta": float(model.params[i+1]),
                "p_value": float(model.pvalues[i+1]),
                "significant": model.pvalues[i+1] < 0.05
            }
        
        # Calculate factor contributions
        factor_contributions = {}
        for i, factor in enumerate(self.default_factors):
            factor_return = np.mean(factor_returns[factor][-min_len:]) * 252  # Annualized
            exposure = exposures[factor]["beta"]
            factor_contributions[factor] = exposure * factor_return
        
        return {
            "alpha": float(model.params[0]),
            "alpha_p_value": float(model.pvalues[0]),
            "alpha_significant": model.pvalues[0] < 0.05,
            "r_squared": float(model.rsquared),
            "adjusted_r_squared": float(model.rsquared_adj),
            "exposures": exposures,
            "factor_contributions": factor_contributions,
            "residual_std": float(np.std(model.resid)),
            "num_periods": min_len,
            "f_statistic": float(model.fvalue),
            "f_p_value": float(model.f_pvalue)
        }
    
    def calculate_rolling_factor_exposure(self, returns: List[float],
                                         factor_returns: Dict[str, List[float]],
                                         window: int = 60) -> List[Dict[str, Any]]:
        """
        Calculate rolling factor exposures over time
        """
        if len(returns) < window + 10:
            return []
        
        rolling_exposures = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            window_factor_returns = {
                name: fr[i-window:i] for name, fr in factor_returns.items()
            }
            
            exposure = self.calculate_factor_exposure(
                window_returns, window_factor_returns
            )
            
            rolling_exposures.append({
                "date": i,
                "exposure": exposure
            })
        
        return rolling_exposures
    
    def attribute_performance(self, portfolio_returns: List[float],
                             benchmark_returns: List[float],
                             factor_returns: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Attribute portfolio performance to factors and selection/ timing
        """
        # Ensure same length
        min_len = min(len(portfolio_returns), len(benchmark_returns))
        portfolio_returns = portfolio_returns[-min_len:]
        benchmark_returns = benchmark_returns[-min_len:]
        
        # Calculate excess returns
        excess_returns = np.array(portfolio_returns) - np.array(benchmark_returns)
        
        # Factor attribution
        factor_attr = self.calculate_factor_exposure(
            excess_returns.tolist(), factor_returns
        )
        
        # Calculate selection and timing effects (simplified)
        # This is a basic implementation - would be more sophisticated in production
        
        # Selection effect: ability to pick winning assets within factors
        selection_effect = factor_attr.get("alpha", 0)
        
        # Timing effect: ability to time factor exposures
        # Simplified - would need rolling betas
        timing_effect = 0
        
        # Interaction effect
        interaction_effect = 0
        
        return {
            "total_excess_return": float(np.sum(excess_returns)),
            "annualized_excess": float(np.mean(excess_returns) * 252),
            "factor_attribution": factor_attr,
            "brinson_attribution": {
                "selection_effect": float(selection_effect),
                "timing_effect": float(timing_effect),
                "interaction_effect": float(interaction_effect),
                "total_active_return": float(selection_effect + timing_effect + interaction_effect)
            }
        }
    
    def calculate_factor_correlation(self, factor_returns: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Calculate correlation between factors
        """
        factor_names = list(factor_returns.keys())
        n = len(factor_names)
        
        if n < 2:
            return {}
        
        # Create returns matrix
        returns_matrix = []
        for name in factor_names:
            returns_matrix.append(factor_returns[name])
        
        returns_array = np.array(returns_matrix)
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_array)
        
        # Find highly correlated factors
        high_correlations = []
        for i in range(n):
            for j in range(i+1, n):
                corr = corr_matrix[i, j]
                if abs(corr) > 0.7:
                    high_correlations.append({
                        "factor1": factor_names[i],
                        "factor2": factor_names[j],
                        "correlation": float(corr)
                    })
        
        return {
            "factors": factor_names,
            "correlation_matrix": corr_matrix.tolist(),
            "high_correlations": high_correlations,
            "avg_correlation": float(np.mean(corr_matrix[np.triu_indices(n, k=1)]))
        }
    
    def get_exposure_summary(self, portfolio_returns: List[float],
                            factor_returns: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        Get summary of factor exposures
        """
        exposure = self.calculate_factor_exposure(portfolio_returns, factor_returns)
        
        # Categorize exposures
        significant_exposures = {
            f: data for f, data in exposure.get("exposures", {}).items()
            if data.get("significant", False)
        }
        
        # Calculate net exposure (simplified)
        net_exposure = sum(
            data["beta"] for data in exposure.get("exposures", {}).values()
        )
        
        return {
            "r_squared": exposure.get("r_squared", 0),
            "alpha": exposure.get("alpha", 0),
            "alpha_significant": exposure.get("alpha_significant", False),
            "significant_factors": len(significant_exposures),
            "factor_exposures": significant_exposures,
            "net_factor_exposure": float(net_exposure),
            "primary_factors": sorted(
                significant_exposures.items(),
                key=lambda x: abs(x[1]["beta"]),
                reverse=True
            )[:3]
        }