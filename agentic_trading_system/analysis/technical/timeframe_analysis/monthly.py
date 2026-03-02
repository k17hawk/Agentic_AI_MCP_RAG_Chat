"""
Monthly Analyzer - Analysis for monthly timeframe
"""
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.logger import logger as logging

class MonthlyAnalyzer:
    """
    Analyzes monthly timeframe for long-term structural trends
    
    Use cases: Major trend direction, long-term investing signals
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Monthly-specific settings
        self.required_months = config.get("required_months", 12)  # 1 year minimum
        
        logging.info(f"✅ MonthlyAnalyzer initialized")
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze monthly data
        """
        if data.empty or len(data) < self.required_months:
            return {
                "error": f"Insufficient data: {len(data)} < {self.required_months} months",
                "required_months": self.required_months
            }
        
        results = {
            "current_price": float(data['Close'].iloc[-1]),
            "data_points": len(data),
            "date_range": {
                "start": data.index[0].strftime("%Y-%m"),
                "end": data.index[-1].strftime("%Y-%m")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Long-term trend analysis
        results["trend"] = self._analyze_trend(data)
        
        # Monthly momentum
        results["momentum"] = self._analyze_momentum(data)
        
        # Seasonal patterns
        results["seasonal"] = self._analyze_seasonal(data)
        
        # Major structural levels
        results["levels"] = self._find_levels(data)
        
        # Composite score
        results["composite_score"] = self._calculate_composite_score(results)
        
        return results
    
    def _analyze_trend(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze long-term monthly trend
        """
        close = data['Close']
        
        # Long-term moving averages
        ma12 = close.rolling(12).mean().iloc[-1]  # 1 year
        ma24 = close.rolling(24).mean().iloc[-1]  # 2 years
        ma60 = close.rolling(60).mean().iloc[-1] if len(close) >= 60 else ma24  # 5 years
        
        current = close.iloc[-1]
        
        # Trend direction
        if current > ma24 > ma60:
            direction = "bullish"
            strength = "strong"
        elif current > ma24:
            direction = "bullish"
            strength = "moderate"
        elif current < ma24 < ma60:
            direction = "bearish"
            strength = "strong"
        elif current < ma24:
            direction = "bearish"
            strength = "moderate"
        else:
            direction = "neutral"
            strength = "weak"
        
        return {
            "direction": direction,
            "strength": strength,
            "ma12": float(ma12),
            "ma24": float(ma24),
            "ma60": float(ma60) if not pd.isna(ma60) else None,
            "price_vs_ma24": float((current - ma24) / ma24 * 100)
        }
    
    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze monthly momentum
        """
        close = data['Close']
        
        # Rate of Change for longer periods
        roc_3 = (close.iloc[-1] / close.iloc[-3] - 1) * 100 if len(close) >= 3 else 0  # 3 months
        roc_6 = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0  # 6 months
        roc_12 = (close.iloc[-1] / close.iloc[-12] - 1) * 100 if len(close) >= 12 else 0  # 1 year
        roc_24 = (close.iloc[-1] / close.iloc[-24] - 1) * 100 if len(close) >= 24 else 0  # 2 years
        
        return {
            "roc_3m": float(roc_3),
            "roc_6m": float(roc_6),
            "roc_12m": float(roc_12),
            "roc_24m": float(roc_24),
            "momentum_trend": "accelerating" if roc_3 > roc_6 > roc_12 else "decelerating" if roc_3 < roc_6 < roc_12 else "mixed"
        }
    
    def _analyze_seasonal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze seasonal patterns (monthly)
        """
        # Get month of last candle
        last_date = data.index[-1]
        current_month = last_date.month
        
        # Calculate average returns by month
        monthly_returns = {}
        
        for month in range(1, 13):
            month_data = data[data.index.month == month]
            if len(month_data) >= 3:  # At least 3 years of data for that month
                returns = month_data['Close'].pct_change().dropna() * 100
                monthly_returns[month] = {
                    "avg_return": float(returns.mean()),
                    "positive_ratio": float((returns > 0).sum() / len(returns))
                }
        
        # Current month stats
        current_stats = monthly_returns.get(current_month, {"avg_return": 0, "positive_ratio": 0.5})
        
        return {
            "current_month": current_month,
            "month_name": last_date.strftime("%B"),
            "avg_return_this_month": float(current_stats["avg_return"]),
            "positive_ratio_this_month": float(current_stats["positive_ratio"]),
            "seasonal_bias": "bullish" if current_stats["avg_return"] > 0 else "bearish",
            "monthly_stats": monthly_returns
        }
    
    def _find_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Find major structural levels
        """
        high = data['High']
        low = data['Low']
        close = data['Close'].iloc[-1]
        
        # All-time highs/lows
        all_time_high = high.max()
        all_time_low = low.min()
        
        # Multi-year highs/lows
        if len(data) >= 60:  # 5 years
            five_year_high = high.iloc[-60:].max()
            five_year_low = low.iloc[-60:].min()
        else:
            five_year_high = all_time_high
            five_year_low = all_time_low
        
        # Round numbers (psychological levels)
        round_number = round(close, -2)  # Round to nearest 100
        
        return {
            "all_time_high": float(all_time_high),
            "all_time_low": float(all_time_low),
            "five_year_high": float(five_year_high),
            "five_year_low": float(five_year_low),
            "nearest_round_number": float(round_number),
            "distance_to_ath": float((all_time_high - close) / close * 100),
            "distance_to_round": float((round_number - close) / close * 100)
        }
    
    def _calculate_composite_score(self, analysis: Dict) -> float:
        """
        Calculate composite score for monthly timeframe
        """
        scores = []
        
        # Trend score
        trend = analysis["trend"]
        if trend["direction"] == "bullish":
            if trend["strength"] == "strong":
                scores.append(0.9)
            else:
                scores.append(0.7)
        elif trend["direction"] == "neutral":
            scores.append(0.5)
        else:
            if trend["strength"] == "strong":
                scores.append(0.1)
            else:
                scores.append(0.3)
        
        # Momentum score
        momentum = analysis["momentum"]
        roc_12 = momentum["roc_12m"]
        if roc_12 > 20:
            scores.append(0.8)
        elif roc_12 > 10:
            scores.append(0.7)
        elif roc_12 > 0:
            scores.append(0.6)
        elif roc_12 > -10:
            scores.append(0.4)
        else:
            scores.append(0.2)
        
        # Seasonal score
        seasonal = analysis["seasonal"]
        if seasonal["seasonal_bias"] == "bullish" and seasonal["positive_ratio_this_month"] > 0.6:
            scores.append(0.7)
        elif seasonal["seasonal_bias"] == "bearish" and seasonal["positive_ratio_this_month"] < 0.4:
            scores.append(0.3)
        else:
            scores.append(0.5)
        
        return float(np.mean(scores)) if scores else 0.5