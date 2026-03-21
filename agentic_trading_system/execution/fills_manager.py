"""
Fills Manager - Tracks and manages order fills
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
from agentic_trading_system.utils.logger import logger as logging

class FillsManager:
    """
    Fills Manager - Tracks and manages order fills
    
    Responsibilities:
    - Record fills
    - Calculate average fill prices
    - Track partial fills
    - Manage fill allocations
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Fill storage
        self.fills = []  # All fills
        self.order_fills = {}  # Fills by order ID
        
        # Statistics
        self.stats = {
            "total_fills": 0,
            "total_volume": 0,
            "total_value": 0.0
        }
        
        logging.info(f"✅ FillsManager initialized")
    
    def add_fill(self, fill: Dict[str, Any]) -> str:
        """
        Add a fill record
        """
        fill_id = f"fill_{datetime.now().timestamp()}_{fill.get('order_id', 'unknown')}"
        
        fill_record = {
            "fill_id": fill_id,
            "timestamp": datetime.now().isoformat(),
            **fill
        }
        
        self.fills.append(fill_record)
        
        # Group by order
        order_id = fill.get("order_id")
        if order_id:
            if order_id not in self.order_fills:
                self.order_fills[order_id] = []
            self.order_fills[order_id].append(fill_record)
        
        # Update stats
        self.stats["total_fills"] += 1
        self.stats["total_volume"] += fill.get("quantity", 0)
        self.stats["total_value"] += fill.get("value", 0)
        
        logging.info(f"✅ Fill recorded: {fill.get('quantity')} {fill.get('symbol')} @ ${fill.get('price', 0):.2f}")
        
        return fill_id
    
    def get_fills_for_order(self, order_id: str) -> List[Dict[str, Any]]:
        """
        Get all fills for an order
        """
        return self.order_fills.get(order_id, [])
    
    def get_average_fill_price(self, order_id: str) -> Optional[float]:
        """
        Get average fill price for an order
        """
        fills = self.order_fills.get(order_id, [])
        
        if not fills:
            return None
        
        total_quantity = sum(f.get("quantity", 0) for f in fills)
        total_value = sum(f.get("value", 0) for f in fills)
        
        if total_quantity == 0:
            return None
        
        return total_value / total_quantity
    
    def get_total_filled_quantity(self, order_id: str) -> int:
        """
        Get total filled quantity for an order
        """
        fills = self.order_fills.get(order_id, [])
        return sum(f.get("quantity", 0) for f in fills)
    
    def is_order_complete(self, order_id: str, expected_quantity: int) -> bool:
        """
        Check if order is completely filled
        """
        filled = self.get_total_filled_quantity(order_id)
        return filled >= expected_quantity
    
    def get_recent_fills(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get most recent fills
        """
        return sorted(self.fills, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_fills_by_symbol(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get fills for a specific symbol
        """
        symbol_fills = [f for f in self.fills if f.get("symbol") == symbol]
        return sorted(symbol_fills, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_fills_by_date(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Get fills within a date range
        """
        return [
            f for f in self.fills
            if start_date <= datetime.fromisoformat(f["timestamp"]) <= end_date
        ]
    
    def clear_fills(self, order_id: str = None):
        """
        Clear fills (for testing)
        """
        if order_id:
            if order_id in self.order_fills:
                # Remove from main list too
                order_fills = self.order_fills[order_id]
                self.fills = [f for f in self.fills if f not in order_fills]
                del self.order_fills[order_id]
        else:
            self.fills = []
            self.order_fills = {}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get fill statistics
        """
        return {
            **self.stats,
            "fills_by_order": len(self.order_fills),
            "average_fill_value": self.stats["total_value"] / self.stats["total_fills"] if self.stats["total_fills"] > 0 else 0
        }