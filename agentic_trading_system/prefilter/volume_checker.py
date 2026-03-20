"""
Volume Checker - Validates trading volume
"""
from typing import Dict, List, Optional, Any
from agentic_trading_system.utils.logger import logger as logging

class VolumeChecker:
    """
    Validates that stock has sufficient trading volume
    
    Ensures:
    - Minimum daily volume
    - Minimum average volume
    - Not too thinly traded
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Volume thresholds
        self.min_volume = config.get("min_volume", 100000)  # 100k shares minimum
        self.min_avg_volume = config.get("min_avg_volume", 50000)  # 50k average
        self.min_dollar_volume = config.get("min_dollar_volume", 1000000)  # $1M minimum
        
        # Volume spike detection
        self.volume_spike_threshold = config.get("volume_spike_threshold", 5.0)  # 5x normal
        
        logging.info(f"✅ VolumeChecker initialized")
    
    async def validate(self, ticker: str, info: Dict) -> Dict[str, Any]:
        """
        Validate trading volume
        """
        volume = info.get("volume")
        if volume is None:
            return {
                "passed": False,
                "reason": "Could not determine current volume"
            }
        
        # Check minimum volume
        if volume < self.min_volume:
            return {
                "passed": False,
                "volume": volume,
                "reason": f"Volume too low: {volume:,} < {self.min_volume:,}"
            }
        
        # Check average volume
        avg_volume = info.get("average_volume")
        if avg_volume is None:
            avg_volume = info.get("average_volume_10d")
        
        if avg_volume and avg_volume < self.min_avg_volume:
            return {
                "passed": False,
                "volume": volume,
                "avg_volume": avg_volume,
                "reason": f"Average volume too low: {avg_volume:,} < {self.min_avg_volume:,}"
            }
        
        # Calculate dollar volume
        price = info.get("current_price")
        if price and volume:
            dollar_volume = price * volume
            
            if dollar_volume < self.min_dollar_volume:
                return {
                    "passed": False,
                    "volume": volume,
                    "dollar_volume": dollar_volume,
                    "reason": f"Dollar volume too low: ${dollar_volume:,.0f} < ${self.min_dollar_volume:,.0f}"
                }
        
        # Check for volume spike
        if avg_volume and avg_volume > 0:
            volume_ratio = volume / avg_volume
            
            if volume_ratio > self.volume_spike_threshold:
                return {
                    "passed": True,
                    "volume": volume,
                    "avg_volume": avg_volume,
                    "volume_ratio": volume_ratio,
                    "volume_spike": True,
                    "warning": f"Volume spike: {volume_ratio:.1f}x average"
                }
        
        # Check bid/ask liquidity
        bid = info.get("bid")
        ask = info.get("ask")
        bid_size = info.get("bid_size")
        ask_size = info.get("ask_size")
        
        liquidity_score = self._calculate_liquidity_score(volume, avg_volume, bid, ask, bid_size, ask_size)
        
        return {
            "passed": True,
            "volume": volume,
            "avg_volume": avg_volume,
            "volume_ratio": volume_ratio if 'volume_ratio' in locals() else 1.0,
            "dollar_volume": dollar_volume if 'dollar_volume' in locals() else None,
            "volume_spike": 'volume_ratio' in locals() and volume_ratio > 2.0,
            "liquidity_score": liquidity_score,
            "bid": bid,
            "ask": ask,
            "spread": (ask - bid) / price * 100 if ask and bid and price else None
        }
    
    def _calculate_liquidity_score(self, volume: int, avg_volume: Optional[int],
                                   bid: Optional[float], ask: Optional[float],
                                   bid_size: Optional[int], ask_size: Optional[int]) -> float:
        """
        Calculate liquidity score (0-100)
        """
        score = 50  # Base score
        
        # Volume contribution
        if volume > 1_000_000:
            score += 20
        elif volume > 500_000:
            score += 10
        elif volume < 100_000:
            score -= 20
        
        # Average volume contribution
        if avg_volume:
            if avg_volume > 500_000:
                score += 10
            elif avg_volume < 100_000:
                score -= 10
        
        # Spread contribution
        if bid and ask and bid > 0:
            spread_pct = (ask - bid) / bid * 100
            if spread_pct < 0.1:
                score += 15
            elif spread_pct < 0.5:
                score += 10
            elif spread_pct < 1.0:
                score += 5
            elif spread_pct > 5.0:
                score -= 15
        
        # Market depth contribution
        if bid_size and ask_size:
            total_depth = bid_size + ask_size
            if total_depth > 50_000:
                score += 10
            elif total_depth < 5_000:
                score -= 10
        
        return max(0, min(100, score))