"""
Smart Order Routing - Routes orders to best venues
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
from utils.logger import logger as logging
from execution.routing.venue_analyzer import VenueAnalyzer

class SmartOrderRouting:
    """
    Smart Order Routing - Routes orders to best venues
    
    Strategies:
    - Best price
    - Lowest cost
    - Fastest execution
    - Dark pool preference
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize venue analyzer
        self.venue_analyzer = VenueAnalyzer(config.get("venue_config", {}))
        
        # Routing strategies
        self.strategies = {
            "best_price": self._route_best_price,
            "lowest_cost": self._route_lowest_cost,
            "fastest": self._route_fastest,
            "dark_pool_first": self._route_dark_pool_first,
            "lit_only": self._route_lit_only
        }
        
        # Default strategy
        self.default_strategy = config.get("default_strategy", "lowest_cost")
        
        # Order splitting
        self.max_splits = config.get("max_splits", 5)
        self.min_split_size = config.get("min_split_size", 100)
        
        logging.info(f"✅ SmartOrderRouting initialized")
    
    async def route_order(self, order: Dict[str, Any], strategy: str = None) -> Dict[str, Any]:
        """
        Route an order to the best venue(s)
        """
        if strategy is None:
            strategy = self.default_strategy
        
        if strategy not in self.strategies:
            strategy = self.default_strategy
        
        logging.info(f"🔄 Routing order with strategy: {strategy}")
        
        # Get routing function
        route_func = self.strategies[strategy]
        
        # Route the order
        routing_plan = await route_func(order)
        
        # Add metadata
        routing_plan["strategy"] = strategy
        routing_plan["order"] = order
        routing_plan["timestamp"] = datetime.now().isoformat()
        
        return routing_plan
    
    async def _route_best_price(self, order: Dict) -> Dict[str, Any]:
        """
        Route to venue with best price
        """
        # Get venue analysis
        venue_analysis = self.venue_analyzer.get_best_venue(order)
        
        # Simple routing - send all to best venue
        return {
            "type": "single",
            "venue": venue_analysis["best_venue"],
            "venue_details": venue_analysis["all_venues"][0],
            "quantity": order["quantity"],
            "splits": [{
                "venue": venue_analysis["best_venue"],
                "quantity": order["quantity"],
                "order_type": order.get("order_type", "MARKET")
            }]
        }
    
    async def _route_lowest_cost(self, order: Dict) -> Dict[str, Any]:
        """
        Route to minimize total cost (fees + slippage)
        """
        quantity = order["quantity"]
        
        # Get all venues with cost estimates
        venues_with_cost = []
        for venue_name, venue_data in self.venue_analyzer.venues.items():
            cost = self.venue_analyzer._estimate_cost(venue_name, venue_data, order)
            venues_with_cost.append({
                "venue": venue_name,
                "data": venue_data,
                "cost": cost
            })
        
        # Sort by total cost
        venues_with_cost.sort(key=lambda x: x["cost"]["total_cost"])
        
        # Check if we need to split order (if best venue can't handle full quantity)
        best_venue = venues_with_cost[0]
        max_size = best_venue["data"].get("max_order_size", float('inf'))
        
        if quantity <= max_size:
            # Single venue
            return {
                "type": "single",
                "venue": best_venue["venue"],
                "venue_details": best_venue,
                "quantity": quantity,
                "splits": [{
                    "venue": best_venue["venue"],
                    "quantity": quantity,
                    "estimated_cost": best_venue["cost"]
                }]
            }
        else:
            # Split across venues
            splits = []
            remaining = quantity
            total_cost = 0
            
            for venue_info in venues_with_cost:
                if remaining <= 0:
                    break
                
                venue_max = min(venue_info["data"].get("max_order_size", remaining), remaining)
                split_qty = min(venue_max, remaining)
                
                splits.append({
                    "venue": venue_info["venue"],
                    "quantity": split_qty,
                    "estimated_cost": venue_info["cost"]["total_cost"] * (split_qty / quantity)
                })
                
                remaining -= split_qty
                total_cost += venue_info["cost"]["total_cost"] * (split_qty / quantity)
            
            return {
                "type": "split",
                "splits": splits,
                "total_estimated_cost": total_cost,
                "quantity": quantity
            }
    
    async def _route_fastest(self, order: Dict) -> Dict[str, Any]:
        """
        Route to fastest venue (lowest latency)
        """
        # Sort venues by latency
        venues_by_latency = sorted(
            self.venue_analyzer.venues.items(),
            key=lambda x: x[1].get("latency_ms", float('inf'))
        )
        
        fastest = venues_by_latency[0]
        
        return {
            "type": "single",
            "venue": fastest[0],
            "venue_details": {
                "name": fastest[0],
                "data": fastest[1],
                "latency_ms": fastest[1].get("latency_ms")
            },
            "quantity": order["quantity"],
            "splits": [{
                "venue": fastest[0],
                "quantity": order["quantity"]
            }]
        }
    
    async def _route_dark_pool_first(self, order: Dict) -> Dict[str, Any]:
        """
        Route to dark pools first, then lit exchanges
        """
        quantity = order["quantity"]
        
        # Separate dark pools and lit exchanges
        dark_pools = []
        lit_exchanges = []
        
        for venue_name, venue_data in self.venue_analyzer.venues.items():
            if venue_data.get("type") == "dark_pool":
                dark_pools.append((venue_name, venue_data))
            else:
                lit_exchanges.append((venue_name, venue_data))
        
        # Try dark pools first
        splits = []
        remaining = quantity
        
        for venue_name, venue_data in dark_pools:
            if remaining <= 0:
                break
            
            # Check if order meets minimum size for dark pool
            min_size = venue_data.get("min_size", 0)
            if remaining >= min_size:
                split_qty = min(remaining, venue_data.get("max_order_size", remaining))
                splits.append({
                    "venue": venue_name,
                    "quantity": split_qty,
                    "type": "dark_pool"
                })
                remaining -= split_qty
        
        # Send remaining to lit exchanges
        if remaining > 0:
            # Use best lit exchange for remaining
            best_lit = lit_exchanges[0] if lit_exchanges else None
            if best_lit:
                splits.append({
                    "venue": best_lit[0],
                    "quantity": remaining,
                    "type": "lit"
                })
        
        return {
            "type": "split" if len(splits) > 1 else "single",
            "splits": splits,
            "quantity": quantity,
            "dark_pool_quantity": quantity - remaining,
            "lit_quantity": remaining
        }
    
    async def _route_lit_only(self, order: Dict) -> Dict[str, Any]:
        """
        Route only to lit exchanges (no dark pools)
        """
        # Filter out dark pools
        lit_venues = {
            name: data for name, data in self.venue_analyzer.venues.items()
            if data.get("type") != "dark_pool"
        }
        
        # Use best price among lit venues
        best_venue = min(lit_venues.items(), 
                        key=lambda x: x[1].get("taker_fee", 0.0002))
        
        return {
            "type": "single",
            "venue": best_venue[0],
            "venue_details": {
                "name": best_venue[0],
                "data": best_venue[1]
            },
            "quantity": order["quantity"],
            "splits": [{
                "venue": best_venue[0],
                "quantity": order["quantity"]
            }]
        }