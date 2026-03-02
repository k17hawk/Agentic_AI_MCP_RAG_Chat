"""
Venue Analyzer - Analyzes and compares trading venues
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
from utils.logger import logger as logging

class VenueAnalyzer:
    """
    Venue Analyzer - Analyzes and compares trading venues
    
    Analyzes:
    - Liquidity
    - Spread
    - Fill rates
    - Latency
    - Fees
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Venue definitions
        self.venues = config.get("venues", {
            "NYSE": {
                "name": "New York Stock Exchange",
                "type": "exchange",
                "asset_classes": ["stocks", "etfs"],
                "latency_ms": 5,
                "maker_fee": -0.0001,  # Negative = rebate
                "taker_fee": 0.0002,
                "liquidity_score": 0.95
            },
            "NASDAQ": {
                "name": "NASDAQ",
                "type": "exchange",
                "asset_classes": ["stocks", "etfs"],
                "latency_ms": 4,
                "maker_fee": -0.0001,
                "taker_fee": 0.0002,
                "liquidity_score": 0.94
            },
            "BATS": {
                "name": "BATS Global Markets",
                "type": "exchange",
                "asset_classes": ["stocks", "etfs"],
                "latency_ms": 3,
                "maker_fee": -0.00015,
                "taker_fee": 0.00025,
                "liquidity_score": 0.85
            },
            "IEX": {
                "name": "IEX",
                "type": "exchange",
                "asset_classes": ["stocks", "etfs"],
                "latency_ms": 10,
                "maker_fee": 0.0,
                "taker_fee": 0.0,
                "liquidity_score": 0.70
            },
            "DARK_POOL_A": {
                "name": "Dark Pool A",
                "type": "dark_pool",
                "asset_classes": ["stocks"],
                "latency_ms": 15,
                "maker_fee": -0.00005,
                "taker_fee": 0.00015,
                "liquidity_score": 0.60,
                "min_size": 1000
            }
        })
        
        # Performance history
        self.venue_performance = {venue: {
            "fill_rate": 1.0,
            "avg_latency": data["latency_ms"],
            "slippage_avg": 0.0,
            "total_orders": 0,
            "successful_orders": 0
        } for venue, data in self.venues.items()}
        
        logging.info(f"✅ VenueAnalyzer initialized with {len(self.venues)} venues")
    
    def get_best_venue(self, order: Dict[str, Any], market_data: Dict = None) -> Dict[str, Any]:
        """
        Get best venue for order execution
        """
        symbol = order["symbol"]
        quantity = order["quantity"]
        side = order["side"]
        order_type = order.get("order_type", "MARKET")
        
        # Score each venue
        venue_scores = []
        
        for venue_name, venue_data in self.venues.items():
            score = self._score_venue(venue_name, venue_data, order, market_data)
            venue_scores.append({
                "venue": venue_name,
                "data": venue_data,
                "score": score,
                "estimated_cost": self._estimate_cost(venue_name, venue_data, order)
            })
        
        # Sort by score
        venue_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "best_venue": venue_scores[0]["venue"],
            "all_venues": venue_scores,
            "order": order,
            "timestamp": datetime.now().isoformat()
        }
    
    def update_performance(self, venue: str, success: bool, latency_ms: float, 
                          slippage: float):
        """
        Update venue performance metrics
        """
        if venue not in self.venue_performance:
            return
        
        perf = self.venue_performance[venue]
        perf["total_orders"] += 1
        if success:
            perf["successful_orders"] += 1
        
        # Update moving averages
        perf["fill_rate"] = perf["successful_orders"] / perf["total_orders"]
        perf["avg_latency"] = perf["avg_latency"] * 0.9 + latency_ms * 0.1
        perf["slippage_avg"] = perf["slippage_avg"] * 0.9 + slippage * 0.1
    
    def _score_venue(self, venue_name: str, venue_data: Dict, 
                    order: Dict, market_data: Dict = None) -> float:
        """
        Score a venue for a specific order
        """
        score = 0.0
        
        # Liquidity score (0-40)
        score += venue_data.get("liquidity_score", 0) * 40
        
        # Latency score (0-20) - lower latency = higher score
        latency_ms = venue_data.get("latency_ms", 10)
        latency_score = max(0, 20 - latency_ms)
        score += latency_score
        
        # Fee score (0-20) - lower fees = higher score
        if order["side"] == "BUY":
            fee = venue_data.get("taker_fee", 0.0002)  # Buyers usually pay taker fees
        else:
            fee = venue_data.get("maker_fee", -0.0001)  # Sellers can get rebates
        
        fee_score = 20 - (fee * 10000)  # Scale fee (0.0002 = 2 basis points)
        score += max(0, fee_score)
        
        # Performance score (0-20) based on historical fill rate
        perf = self.venue_performance.get(venue_name, {})
        fill_rate = perf.get("fill_rate", 1.0)
        score += fill_rate * 20
        
        return score
    
    def _estimate_cost(self, venue_name: str, venue_data: Dict, 
                      order: Dict) -> Dict[str, float]:
        """
        Estimate execution cost at venue
        """
        quantity = order["quantity"]
        price = order.get("limit_price", 100.0)  # Default price for estimation
        
        if order["side"] == "BUY":
            fee_rate = venue_data.get("taker_fee", 0.0002)
        else:
            fee_rate = venue_data.get("maker_fee", -0.0001)
        
        fees = quantity * price * fee_rate
        
        # Estimate slippage based on liquidity
        liquidity = venue_data.get("liquidity_score", 0.5)
        expected_slippage = (1 - liquidity) * 0.001 * quantity * price
        
        total_cost = fees + expected_slippage
        
        return {
            "fees": fees,
            "expected_slippage": expected_slippage,
            "total_cost": total_cost,
            "cost_per_share": total_cost / quantity if quantity > 0 else 0
        }
    
    def get_venue_stats(self) -> Dict[str, Any]:
        """
        Get statistics for all venues
        """
        return {
            "venues": self.venues,
            "performance": self.venue_performance,
            "best_overall": max(self.venue_performance.items(), 
                               key=lambda x: x[1]["fill_rate"] * x[1].get("liquidity_score", 0))[0]
        }