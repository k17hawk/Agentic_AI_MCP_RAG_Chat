"""
Data Quality Checker - Validates quality of available data
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import yfinance as yf
import pandas as pd

from agentic_trading_system.utils.logger import logger as logging

class DataQualityChecker:
    """
    Validates that sufficient quality data is available
    
    Checks:
    - Historical data availability
    - Data completeness
    - Freshness of data
    - Corporate actions (splits, dividends)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Historical data requirements
        self.min_history_days = config.get("min_history_days", 60)  # YOUR 60-DAY REQUIREMENT!
        self.min_data_points = config.get("min_data_points", 50)
        
        # Freshness requirements
        self.max_data_age_hours = config.get("max_data_age_hours", 24)
        
        # Volume of data points per period
        self.expected_data_frequency = config.get("expected_data_frequency", {
            "1d": 1,  # 1 data point per day for daily data
            "1h": 6.5,  # ~6.5 hours of trading per day
            "1m": 390  # 390 minutes per trading day
        })
        
        logging.info(f"✅ DataQualityChecker initialized")
    
    async def validate(self, ticker: str, info: Dict) -> Dict[str, Any]:
        """
        Validate data quality
        """
        issues = []
        warnings = []
        
        # Check if ticker exists and has basic info
        if not info or len(info) < 10:
            issues.append("Insufficient basic information")
        
        # Check price data
        price = info.get("current_price")
        if price is None or price <= 0:
            issues.append("Invalid or missing price data")
        
        # Check volume data
        volume = info.get("volume")
        if volume is None:
            issues.append("Missing volume data")
        
        # Check historical data availability
        hist_data = await self._check_historical_data(ticker)
        if hist_data["has_issues"]:
            issues.extend(hist_data["issues"])
        
        # Check data freshness
        freshness = await self._check_data_freshness(ticker)
        if freshness["stale"]:
            warnings.append(f"Data may be stale: {freshness['age_hours']:.1f} hours old")
        
        # Check for recent corporate actions
        corporate_actions = await self._check_corporate_actions(ticker)
        if corporate_actions["has_splits"]:
            warnings.append(f"Recent stock split: {corporate_actions['split_date']}")
        if corporate_actions["has_dividends"]:
            warnings.append("Recent dividend payment")
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(
            issues, warnings, hist_data, freshness
        )
        
        # Determine if passed
        passed = len(issues) == 0 and quality_score >= 50
        
        return {
            "passed": passed,
            "quality_score": quality_score,
            "issues": issues,
            "warnings": warnings,
            "historical_data": hist_data,
            "freshness": freshness,
            "corporate_actions": corporate_actions,
            "has_sufficient_history": hist_data["days_available"] >= self.min_history_days
        }
    
    async def _check_historical_data(self, ticker: str) -> Dict[str, Any]:
        """
        Check historical data availability
        """
        issues = []
        
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="3mo")
            
            if hist.empty:
                issues.append("No historical data available")
                return {
                    "has_issues": True,
                    "issues": issues,
                    "days_available": 0,
                    "data_points": 0
                }
            
            days_available = len(hist)
            data_points = hist['Close'].count()
            
            # Check minimum history
            if days_available < self.min_history_days:
                issues.append(f"Insufficient history: {days_available} days < {self.min_history_days}")
            
            # Check for gaps
            date_diff = hist.index.to_series().diff().dt.days
            gaps = date_diff[date_diff > 3].count()  # Gaps of more than 3 days
            if gaps > 0:
                issues.append(f"Data gaps detected: {gaps} gaps")
            
            # Check for constant values (possible stale data)
            if hist['Close'].std() < 0.01:
                issues.append("Suspiciously constant prices")
            
            return {
                "has_issues": len(issues) > 0,
                "issues": issues,
                "days_available": days_available,
                "data_points": int(data_points),
                "gaps_detected": int(gaps),
                "start_date": hist.index[0].strftime("%Y-%m-%d") if not hist.empty else None,
                "end_date": hist.index[-1].strftime("%Y-%m-%d") if not hist.empty else None
            }
            
        except Exception as e:
            return {
                "has_issues": True,
                "issues": [f"Error checking historical data: {str(e)}"],
                "days_available": 0,
                "data_points": 0
            }
    
    async def _check_data_freshness(self, ticker: str) -> Dict[str, Any]:
        """
        Check if data is fresh (not stale)
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="5d")
            
            if hist.empty:
                return {
                    "stale": True,
                    "age_hours": 999,
                    "last_trade": None
                }
            
            last_trade = hist.index[-1]
            now = datetime.now(last_trade.tzinfo)
            age_hours = (now - last_trade).total_seconds() / 3600
            
            # Check if market is open
            is_market_hours = self._is_market_hours(now)
            
            stale = False
            if is_market_hours and age_hours > 1:  # During market hours, data should be < 1hr old
                stale = True
            elif not is_market_hours and age_hours > self.max_data_age_hours:
                stale = True
            
            return {
                "stale": stale,
                "age_hours": age_hours,
                "last_trade": last_trade.strftime("%Y-%m-%d %H:%M:%S"),
                "is_market_hours": is_market_hours,
                "market_status": "open" if is_market_hours else "closed"
            }
            
        except Exception as e:
            return {
                "stale": True,
                "age_hours": 999,
                "last_trade": None,
                "error": str(e)
            }
    
    async def _check_corporate_actions(self, ticker: str) -> Dict[str, Any]:
        """
        Check for recent corporate actions
        """
        try:
            stock = yf.Ticker(ticker)
            
            # Get splits
            splits = stock.splits
            recent_splits = splits[splits.index > (datetime.now() - pd.Timedelta(days=30))] if not splits.empty else pd.Series()
            
            # Get dividends
            dividends = stock.dividends
            recent_dividends = dividends[dividends.index > (datetime.now() - pd.Timedelta(days=30))] if not dividends.empty else pd.Series()
            
            return {
                "has_splits": not recent_splits.empty,
                "split_date": recent_splits.index[-1].strftime("%Y-%m-%d") if not recent_splits.empty else None,
                "split_ratio": float(recent_splits.iloc[-1]) if not recent_splits.empty else None,
                "has_dividends": not recent_dividends.empty,
                "dividend_date": recent_dividends.index[-1].strftime("%Y-%m-%d") if not recent_dividends.empty else None,
                "dividend_amount": float(recent_dividends.iloc[-1]) if not recent_dividends.empty else None
            }
            
        except Exception:
            return {
                "has_splits": False,
                "has_dividends": False
            }
    
    def _is_market_hours(self, dt: datetime) -> bool:
        """
        Check if given time is during market hours (US Eastern Time)
        Market hours: 9:30 AM - 4:00 PM ET, Monday-Friday
        """
        # Convert to US Eastern Time
        try:
            import pytz
            eastern = pytz.timezone('US/Eastern')
            dt_eastern = dt.astimezone(eastern)
            
            # Check weekday (Monday=0, Sunday=6)
            if dt_eastern.weekday() >= 5:  # Saturday or Sunday
                return False
            
            # Check time
            market_open = dt_eastern.replace(hour=9, minute=30, second=0)
            market_close = dt_eastern.replace(hour=16, minute=0, second=0)
            
            return market_open <= dt_eastern <= market_close
            
        except:
            # If pytz not available, assume always market hours for simplicity
            return True
    
    def _calculate_quality_score(self, issues: List[str], warnings: List[str],
                                 hist_data: Dict, freshness: Dict) -> float:
        """
        Calculate overall data quality score (0-100)
        """
        score = 100
        
        # Deduct for issues
        score -= len(issues) * 15
        
        # Deduct for warnings
        score -= len(warnings) * 5
        
        # Historical data quality
        if hist_data.get("days_available", 0) > 0:
            coverage_ratio = min(1.0, hist_data["days_available"] / self.min_history_days)
            score = score * (0.5 + 0.5 * coverage_ratio)
        
        # Freshness penalty
        if freshness.get("stale", False):
            score *= 0.7
        
        return max(0, min(100, score))