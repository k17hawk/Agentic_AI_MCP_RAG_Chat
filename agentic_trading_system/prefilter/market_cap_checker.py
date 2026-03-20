"""
Market Cap Checker - Validates market capitalization
"""
from typing import Dict, List, Optional, Any
from agentic_trading_system.utils.logger import logger as logging
class MarketCapChecker:
    """
    Validates that stock has sufficient market capitalization
    
    Categories:
    - Mega Cap: $200B+
    - Large Cap: $10B - $200B
    - Mid Cap: $2B - $10B
    - Small Cap: $300M - $2B
    - Micro Cap: $50M - $300M
    - Nano Cap: < $50M
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Market cap thresholds
        self.min_market_cap = config.get("min_market_cap", 50_000_000)  # $50M minimum
        self.max_market_cap = config.get("max_market_cap", float('inf'))
        
        # Preferred market cap ranges (for scoring)
        self.preferred_min = config.get("preferred_min", 300_000_000)  # $300M (Small Cap+)
        self.preferred_max = config.get("preferred_max", 200_000_000_000)  # $200B (Large Cap)
        
        # Category thresholds
        self.categories = {
            "mega": 200_000_000_000,
            "large": 10_000_000_000,
            "mid": 2_000_000_000,
            "small": 300_000_000,
            "micro": 50_000_000,
            "nano": 0
        }
        
        logging.info(f"✅ MarketCapChecker initialized")
    
    async def validate(self, ticker: str, info: Dict) -> Dict[str, Any]:
        """
        Validate market capitalization
        """
        market_cap = info.get("market_cap")
        if market_cap is None:
            return {
                "passed": False,
                "reason": "Could not determine market cap"
            }
        
        # Check minimum
        if market_cap < self.min_market_cap:
            return {
                "passed": False,
                "market_cap": market_cap,
                "category": self._get_category(market_cap),
                "reason": f"Market cap too small: ${market_cap:,.0f} < ${self.min_market_cap:,.0f}"
            }
        
        # Check maximum
        if market_cap > self.max_market_cap:
            return {
                "passed": False,
                "market_cap": market_cap,
                "category": self._get_category(market_cap),
                "reason": f"Market cap too large: ${market_cap:,.0f} > ${self.max_market_cap:,.0f}"
            }
        
        # Determine category
        category = self._get_category(market_cap)
        
        # Calculate score based on market cap
        score = self._calculate_score(market_cap, category)
        
        # Check if in preferred range
        in_preferred = self.preferred_min <= market_cap <= self.preferred_max
        
        return {
            "passed": True,
            "market_cap": market_cap,
            "category": category,
            "score": score,
            "in_preferred_range": in_preferred,
            "min_threshold": self.min_market_cap,
            "max_threshold": self.max_market_cap
        }
    
    def _get_category(self, market_cap: float) -> str:
        """Get market cap category"""
        if market_cap >= self.categories["mega"]:
            return "mega"
        elif market_cap >= self.categories["large"]:
            return "large"
        elif market_cap >= self.categories["mid"]:
            return "mid"
        elif market_cap >= self.categories["small"]:
            return "small"
        elif market_cap >= self.categories["micro"]:
            return "micro"
        else:
            return "nano"
    
    def _calculate_score(self, market_cap: float, category: str) -> float:
        """
        Calculate score based on market cap (0-100)
        """
        scores = {
            "mega": 90,
            "large": 100,
            "mid": 80,
            "small": 70,
            "micro": 50,
            "nano": 30
        }
        
        base_score = scores.get(category, 50)
        
        # Adjust score based on position within category
        if category == "mega":
            # Higher score for larger mega caps
            ratio = min(1.0, market_cap / 1000_000_000_000)  # Cap at $1T
            return base_score + (ratio * 10)
        elif category == "large":
            # Peak at mid-large
            return base_score
        elif category == "mid":
            # Slightly higher for larger mid caps
            ratio = (market_cap - self.categories["mid"]) / (self.categories["large"] - self.categories["mid"])
            return base_score + (ratio * 10)
        elif category == "small":
            # Slightly higher for larger small caps
            ratio = (market_cap - self.categories["small"]) / (self.categories["mid"] - self.categories["small"])
            return base_score + (ratio * 10)
        else:
            # Smaller gets lower scores
            return base_score
    
    def get_category_description(self, category: str) -> str:
        """Get description of market cap category"""
        descriptions = {
            "mega": "Mega Cap (>$200B) - Established global leaders",
            "large": "Large Cap ($10B-$200B) - Established companies",
            "mid": "Mid Cap ($2B-$10B) - Growing companies",
            "small": "Small Cap ($300M-$2B) - Emerging growth",
            "micro": "Micro Cap ($50M-$300M) - Speculative",
            "nano": "Nano Cap (<$50M) - Highly speculative"
        }
        return descriptions.get(category, "Unknown")