"""
DCF Analyzer - Discounted Cash Flow valuation
"""
from typing import Dict, List, Optional, Any
import numpy as np
from datetime import datetime
from utils.logger import logger as  logging

class DCFAnalyzer:
    """
    Performs Discounted Cash Flow (DCF) valuation
    
    Calculates:
    - Intrinsic Value
    - Margin of Safety
    - Fair Value Estimate
    - Upside/Downside Potential
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # DCF parameters
        self.growth_rate = config.get("growth_rate", 0.10)  # 10% initial growth
        self.terminal_growth = config.get("terminal_growth", 0.03)  # 3% perpetual growth
        self.discount_rate = config.get("discount_rate", 0.10)  # 10% WACC
        self.projection_years = config.get("projection_years", 5)
        
        # Margin of safety thresholds
        self.mos_strong = config.get("mos_strong", 0.30)  # 30%+
        self.mos_good = config.get("mos_good", 0.20)  # 20-30%
        self.mos_fair = config.get("mos_fair", 0.10)  # 10-20%
        
        logging.info(f"✅ DCFAnalyzer initialized")
    
    async def analyze(self, symbol: str, info: Dict) -> Dict[str, Any]:
        """
        Perform DCF valuation
        """
        try:
            # Extract financial data
            fcf = self._get_free_cash_flow(info)
            revenue = info.get("totalRevenue")
            ebitda = info.get("ebitda")
            debt = info.get("totalDebt", 0)
            cash = info.get("totalCash", 0)
            shares = info.get("sharesOutstanding")
            current_price = info.get("currentPrice", info.get("regularMarketPrice", 0))
            
            if not fcf or not shares or not current_price:
                return {
                    "score": 0.5,
                    "value": None,
                    "signal": "insufficient_data",
                    "details": "Insufficient data for DCF calculation"
                }
            
            # Calculate growth rate based on historical/analyst estimates
            growth_rate = self._estimate_growth_rate(info)
            
            # Project future cash flows
            projected_fcf = self._project_cash_flows(fcf, growth_rate)
            
            # Calculate terminal value
            terminal_value = self._calculate_terminal_value(projected_fcf[-1])
            
            # Calculate present value
            pv_fcf = self._calculate_present_value(projected_fcf)
            pv_terminal = self._calculate_present_value([terminal_value], self.projection_years)
            
            # Calculate enterprise value
            enterprise_value = pv_fcf + pv_terminal
            
            # Calculate equity value
            equity_value = enterprise_value - debt + cash
            
            # Calculate intrinsic value per share
            intrinsic_value = equity_value / shares if shares > 0 else 0
            
            # Calculate margin of safety
            if current_price > 0:
                mos = (intrinsic_value - current_price) / current_price
            else:
                mos = 0
            
            # Determine signal
            if mos >= self.mos_strong:
                signal = "strong_buy"
                score = 0.9
            elif mos >= self.mos_good:
                signal = "buy"
                score = 0.8
            elif mos >= self.mos_fair:
                signal = "fair_value"
                score = 0.7
            elif mos >= 0:
                signal = "slightly_undervalued"
                score = 0.6
            elif mos >= -0.1:
                signal = "fairly_valued"
                score = 0.5
            elif mos >= -0.2:
                signal = "slightly_overvalued"
                score = 0.4
            else:
                signal = "overvalued"
                score = 0.2
            
            return {
                "score": float(score),
                "value": float(intrinsic_value),
                "current_price": float(current_price),
                "margin_of_safety": float(mos),
                "signal": signal,
                "details": {
                    "intrinsic_value": float(intrinsic_value),
                    "current_price": float(current_price),
                    "upside": float(mos * 100),
                    "enterprise_value": float(enterprise_value),
                    "equity_value": float(equity_value),
                    "projected_fcf": [float(x) for x in projected_fcf],
                    "terminal_value": float(terminal_value),
                    "growth_rate_used": float(growth_rate),
                    "discount_rate_used": float(self.discount_rate),
                    "terminal_growth_used": float(self.terminal_growth),
                    "debt": float(debt),
                    "cash": float(cash),
                    "shares": float(shares)
                }
            }
            
        except Exception as e:
            logging.error(f"Error in DCF calculation: {e}")
            return {
                "score": 0.5,
                "value": None,
                "signal": "error",
                "details": {"error": str(e)}
            }
    
    def _get_free_cash_flow(self, info: Dict) -> Optional[float]:
        """Extract Free Cash Flow from info"""
        # Try different fields
        candidates = [
            "freeCashflow",
            "operatingCashFlow"
        ]
        
        for field in candidates:
            value = info.get(field)
            if value:
                return float(value)
        
        return None
    
    def _estimate_growth_rate(self, info: Dict) -> float:
        """Estimate growth rate from multiple sources"""
        growth_rates = []
        
        # Analyst growth estimates
        earnings_growth = info.get("earningsGrowth")
        if earnings_growth:
            growth_rates.append(earnings_growth)
        
        # Revenue growth
        revenue_growth = info.get("revenueGrowth")
        if revenue_growth:
            growth_rates.append(revenue_growth)
        
        # Industry growth (placeholder)
        growth_rates.append(0.08)  # 8% industry average
        
        if growth_rates:
            # Use max of available estimates, but cap at reasonable levels
            estimated = max(growth_rates)
            return min(0.20, max(0.02, estimated))
        
        return self.growth_rate
    
    def _project_cash_flows(self, base_fcf: float, growth_rate: float) -> List[float]:
        """
        Project future cash flows
        """
        projected = []
        current = base_fcf
        
        for year in range(1, self.projection_years + 1):
            # Gradually decay growth rate to terminal rate
            decay = 1 - (year / self.projection_years) * 0.5
            year_growth = growth_rate * decay
            
            current = current * (1 + year_growth)
            projected.append(current)
        
        return projected
    
    def _calculate_terminal_value(self, last_fcf: float) -> float:
        """
        Calculate terminal value using Gordon Growth Model
        TV = FCF * (1 + g) / (r - g)
        """
        if self.discount_rate <= self.terminal_growth:
            # Fallback to multiple method
            return last_fcf * 15  # 15x multiple
        
        terminal = last_fcf * (1 + self.terminal_growth) / (self.discount_rate - self.terminal_growth)
        return terminal
    
    def _calculate_present_value(self, cash_flows: List[float], 
                                 start_year: int = 1) -> float:
        """
        Calculate present value of cash flows
        """
        pv = 0
        
        for i, cf in enumerate(cash_flows):
            year = start_year + i
            pv += cf / ((1 + self.discount_rate) ** year)
        
        return pv