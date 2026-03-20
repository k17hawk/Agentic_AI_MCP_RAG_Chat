"""
Price Range Checker - Validates stock price range
"""
from typing import Dict, List, Optional, Any
from agentic_trading_system.utils.logger import logger as logging
class PriceRangeChecker:
    """
    Validates that stock price is within acceptable range
    
    Filters out:
    - Penny stocks (below minimum price)
    - Extremely expensive stocks (above maximum price)
    - Stocks with suspicious price movements
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Price thresholds
        self.min_price = config.get("min_price", 1.0)  # $1 minimum
        self.max_price = config.get("max_price", 10000.0)  # $10k maximum
        
        # Price change thresholds
        self.max_daily_change = config.get("max_daily_change", 100.0)  # 100% max daily move
        self.max_weekly_change = config.get("max_weekly_change", 300.0)  # 300% max weekly move
        
        # Gap thresholds
        self.max_gap_percent = config.get("max_gap_percent", 20.0)  # 20% max gap
        
        logging.info(f"✅ PriceRangeChecker initialized")
    
    async def validate(self, ticker: str, info: Dict) -> Dict[str, Any]:
        """
        Validate stock price range
        """
        current_price = info.get("current_price")
        if current_price is None:
            current_price = info.get("regularMarketPrice")
        
        if current_price is None:
            return {
                "passed": False,
                "reason": "Could not determine current price"
            }
        
        # Check minimum price
        if current_price < self.min_price:
            return {
                "passed": False,
                "price": current_price,
                "reason": f"Price too low: ${current_price:.2f} < ${self.min_price:.2f}"
            }
        
        # Check maximum price
        if current_price > self.max_price:
            return {
                "passed": False,
                "price": current_price,
                "reason": f"Price too high: ${current_price:.2f} > ${self.max_price:.2f}"
            }
        
        # Check daily change if available
        previous_close = info.get("previous_close")
        if previous_close and previous_close > 0:
            daily_change = ((current_price - previous_close) / previous_close) * 100
            
            if abs(daily_change) > self.max_daily_change:
                return {
                    "passed": False,
                    "price": current_price,
                    "daily_change": daily_change,
                    "reason": f"Daily change too large: {daily_change:.2f}% > {self.max_daily_change}%"
                }
        
        # Check 52-week range if available
        year_high = info.get("fifty_two_week_high")
        year_low = info.get("fifty_two_week_low")
        
        if year_high and year_low and year_high > year_low:
            year_range = year_high - year_low
            year_range_pct = (year_range / year_low) * 100
            
            if year_range_pct > self.max_weekly_change * 4:  # Scale weekly to yearly
                return {
                    "passed": False,
                    "price": current_price,
                    "year_range_pct": year_range_pct,
                    "reason": f"Yearly range too large: {year_range_pct:.2f}%"
                }
        
        # Check gap from previous close
        if previous_close and previous_close > 0:
            gap_pct = ((current_price - previous_close) / previous_close) * 100
            if abs(gap_pct) > self.max_gap_percent:
                return {
                    "passed": True,  # Allow gaps, but note them
                    "price": current_price,
                    "gap_pct": gap_pct,
                    "warning": f"Large gap: {gap_pct:.2f}%",
                    "gap_detected": True
                }
        
        return {
            "passed": True,
            "price": current_price,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "daily_change": daily_change if 'daily_change' in locals() else None,
            "gap_detected": 'gap_pct' in locals() and abs(gap_pct) > 5
        }