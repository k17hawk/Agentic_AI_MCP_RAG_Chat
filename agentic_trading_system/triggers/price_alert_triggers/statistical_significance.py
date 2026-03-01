"""
Statistical Significance - Tests if price movements are statistically significant
This implements your 60-day requirement for distinguishing signal from noise
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from scipy import stats
from logger import logging as logger



class StatisticalSignificance:
    """
    Statistical tests to determine if price movements are significant
    Uses:
    - Z-score analysis
    - T-tests
    - Confidence intervals
    - Monte Carlo simulations
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.z_threshold = config.get("z_score_threshold", 2.0)
        self.p_threshold = config.get("p_value_threshold", 0.05)
        self.min_sample = config.get("min_sample_size", 30)  # YOUR 60-DAY MINIMUM
        self.confidence_level = config.get("confidence_level", 0.95)
        
        logger.info("StatisticalSignificance initialized")
    
    def calculate_significance(self, data: pd.DataFrame) -> Dict:
        """
        Calculate statistical significance for current price movement
        """
        if len(data) < self.min_sample:
            return {
                "significant": False,
                "error": f"Insufficient data: {len(data)} < {self.min_sample}",
                "sample_size": len(data)
            }
        
        # Get current and previous prices
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        
        # Calculate current return
        current_return = (current_price - prev_price) / prev_price * 100
        
        # Get historical returns for baseline
        historical_returns = data['Close'].pct_change().dropna() * 100
        historical_returns = historical_returns.iloc[-self.min_sample:]  # Use last N days
        
        # Z-score test
        z_score_results = self._z_score_test(current_return, historical_returns)
        
        # T-test
        t_test_results = self._t_test(current_return, historical_returns)
        
        # Confidence interval
        ci_results = self._confidence_interval(historical_returns)
        
        # Outlier detection
        outlier_results = self._detect_outlier(current_return, historical_returns)
        
        # Bootstrap test
        bootstrap_results = self._bootstrap_test(current_return, historical_returns)
        
        # Combine results
        significant = (
            z_score_results["significant"] and
            t_test_results["p_value"] < self.p_threshold and
            outlier_results["is_outlier"]
        )
        
        # Calculate effect size
        effect_size = self._calculate_effect_size(current_return, historical_returns)
        
        return {
            "significant": significant,
            "z_score": z_score_results["z_score"],
            "p_value": t_test_results["p_value"],
            "t_statistic": t_test_results["t_statistic"],
            "confidence_interval": ci_results,
            "is_outlier": outlier_results["is_outlier"],
            "outlier_score": outlier_results["score"],
            "bootstrap_probability": bootstrap_results["probability"],
            "effect_size": effect_size,
            "sample_size": len(historical_returns),
            "mean_return": float(historical_returns.mean()),
            "std_return": float(historical_returns.std()),
            "current_return": float(current_return)
        }
    
    def _z_score_test(self, current_value: float, historical: pd.Series) -> Dict:
        """
        Calculate Z-score: (x - μ) / σ
        How many standard deviations from mean?
        """
        mean = historical.mean()
        std = historical.std()
        
        if std == 0:
            return {"z_score": 0, "significant": False}
        
        z_score = (current_value - mean) / std
        
        # Significant if |z| > threshold
        significant = abs(z_score) > self.z_threshold
        
        return {
            "z_score": float(z_score),
            "significant": significant,
            "mean": float(mean),
            "std": float(std)
        }
    
    def _t_test(self, current_value: float, historical: pd.Series) -> Dict:
        """
        One-sample t-test: Is this value significantly different from historical mean?
        """
        # Remove current value from historical for independence
        historical = historical[:-1] if len(historical) > 1 else historical
        
        # Perform t-test
        t_statistic, p_value = stats.ttest_1samp(historical, current_value)
        
        return {
            "t_statistic": float(t_statistic),
            "p_value": float(p_value),
            "significant": p_value < self.p_threshold,
            "degrees_freedom": len(historical) - 1
        }
    
    def _confidence_interval(self, data: pd.Series) -> Dict:
        """
        Calculate confidence interval for the mean
        """
        mean = data.mean()
        std = data.std()
        n = len(data)
        
        # Standard error
        se = std / np.sqrt(n)
        
        # Z-score for confidence level
        if self.confidence_level == 0.95:
            z = 1.96
        elif self.confidence_level == 0.99:
            z = 2.576
        else:
            z = 1.96
        
        # Confidence interval
        margin = z * se
        ci_lower = mean - margin
        ci_upper = mean + margin
        
        return {
            "mean": float(mean),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "margin": float(margin),
            "level": self.confidence_level
        }
    
    def _detect_outlier(self, current_value: float, historical: pd.Series) -> Dict:
        """
        Detect if current value is an outlier using IQR method
        """
        # Calculate quartiles
        q1 = historical.quantile(0.25)
        q3 = historical.quantile(0.75)
        iqr = q3 - q1
        
        # Outlier boundaries
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Check if outlier
        is_outlier = current_value < lower_bound or current_value > upper_bound
        
        # Calculate outlier score (how far beyond bounds)
        if is_outlier:
            if current_value < lower_bound:
                score = (lower_bound - current_value) / iqr if iqr != 0 else 0
            else:
                score = (current_value - upper_bound) / iqr if iqr != 0 else 0
        else:
            score = 0
        
        return {
            "is_outlier": is_outlier,
            "score": float(score),
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr)
        }
    
    def _bootstrap_test(self, current_value: float, historical: pd.Series, 
                        n_iterations: int = 1000) -> Dict:
        """
        Bootstrap test: probability of observing such an extreme value by chance
        """
        n = len(historical)
        extreme_count = 0
        
        for _ in range(n_iterations):
            # Sample with replacement
            sample = np.random.choice(historical, size=n, replace=True)
            sample_mean = np.mean(sample)
            
            # Check if sample mean is as extreme as current
            if abs(sample_mean) >= abs(current_value):
                extreme_count += 1
        
        probability = extreme_count / n_iterations
        
        return {
            "probability": float(probability),
            "significant": probability < 0.05,
            "iterations": n_iterations
        }
    
    def _calculate_effect_size(self, current_value: float, historical: pd.Series) -> Dict:
        """
        Calculate effect size (Cohen's d)
        How large is the effect?
        """
        mean = historical.mean()
        std = historical.std()
        
        if std == 0:
            return {"d": 0, "magnitude": "none"}
        
        # Cohen's d
        d = abs(current_value - mean) / std
        
        # Interpret magnitude
        if d < 0.2:
            magnitude = "negligible"
        elif d < 0.5:
            magnitude = "small"
        elif d < 0.8:
            magnitude = "medium"
        else:
            magnitude = "large"
        
        return {
            "d": float(d),
            "magnitude": magnitude,
            "interpretation": f"{magnitude} effect size"
        }
    
    def compare_distributions(self, data1: pd.Series, data2: pd.Series) -> Dict:
        """
        Compare two distributions (e.g., pre and post event)
        """
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(data1, data2)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.ks_2samp(data1, data2)
        
        return {
            "t_test": {"statistic": float(t_stat), "p_value": float(p_value)},
            "mann_whitney": {"statistic": float(u_stat), "p_value": float(u_p_value)},
            "ks_test": {"statistic": float(ks_stat), "p_value": float(ks_p_value)},
            "different": p_value < 0.05 or u_p_value < 0.05 or ks_p_value < 0.05
        }
    
    def calculate_var(self, returns: pd.Series, confidence: float = 0.95) -> Dict:
        """
        Calculate Value at Risk
        """
        if len(returns) < 30:
            return {"var": 0, "cvar": 0}
        
        # Sort returns
        sorted_returns = returns.sort_values()
        
        # VaR at confidence level
        var_index = int((1 - confidence) * len(sorted_returns))
        var = abs(sorted_returns.iloc[var_index])
        
        # CVaR (Expected Shortfall)
        cvar = abs(sorted_returns.iloc[:var_index].mean()) if var_index > 0 else 0
        
        return {
            "var": float(var),
            "cvar": float(cvar),
            "confidence": confidence,
            "method": "historical"
        }