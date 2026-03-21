"""
Open Positions - Tracks current open positions
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import yfinance as yf
from agentic_trading_system.utils.logger import logger as  logging

class OpenPositions:
    """
    Open Positions - Tracks and manages open positions
    
    Responsibilities:
    - Track current positions
    - Calculate unrealized P&L
    - Monitor position limits
    - Generate position reports
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Position storage
        self.positions = {}  # symbol -> position details
        
        # Position limits
        self.max_positions = config.get("max_positions", 20)
        self.max_position_size = config.get("max_position_size", 0.25)  # 25% of portfolio
        self.max_sector_exposure = config.get("max_sector_exposure", 0.30)  # 30% per sector
        
        # Market data cache
        self.price_cache = {}
        
        logging.info(f"✅ OpenPositions initialized")
    
    def add_position(self, position: Dict[str, Any]) -> bool:
        """
        Add a new position
        """
        symbol = position["symbol"]
        
        # Check position limits
        if len(self.positions) >= self.max_positions:
            logging.warning(f"❌ Cannot add {symbol}: max positions ({self.max_positions}) reached")
            return False
        
        # Check if already have position
        if symbol in self.positions:
            # Average in
            existing = self.positions[symbol]
            total_shares = existing["shares"] + position["shares"]
            total_cost = existing["cost_basis"] + position["cost_basis"]
            
            existing["shares"] = total_shares
            existing["cost_basis"] = total_cost
            existing["avg_price"] = total_cost / total_shares
            existing["last_updated"] = datetime.now().isoformat()
            
            logging.info(f"📈 Added to position {symbol}: now {total_shares} shares")
        else:
            # New position
            self.positions[symbol] = {
                "symbol": symbol,
                "shares": position["shares"],
                "avg_price": position["avg_price"],
                "cost_basis": position["cost_basis"],
                "entry_date": position.get("entry_date", datetime.now().isoformat()),
                "last_updated": datetime.now().isoformat(),
                "sector": position.get("sector", "Unknown"),
                "notes": position.get("notes", "")
            }
            logging.info(f"📈 New position opened: {symbol} - {position['shares']} shares")
        
        return True
    
    def remove_position(self, symbol: str, shares: int = None) -> Optional[Dict[str, Any]]:
        """
        Remove shares from a position (or entire position)
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        
        if shares is None or shares >= position["shares"]:
            # Close entire position
            closed_position = position.copy()
            del self.positions[symbol]
            logging.info(f"📉 Position closed: {symbol}")
            return closed_position
        else:
            # Partial close
            # Calculate cost basis of sold shares
            sold_value = (position["cost_basis"] / position["shares"]) * shares
            
            position["shares"] -= shares
            position["cost_basis"] -= sold_value
            position["last_updated"] = datetime.now().isoformat()
            
            logging.info(f"📉 Reduced position {symbol}: now {position['shares']} shares")
            
            return {
                "symbol": symbol,
                "shares": shares,
                "avg_price": position["avg_price"],
                "cost_basis": sold_value
            }
    
    async def update_prices(self):
        """
        Update current prices for all positions
        """
        for symbol in self.positions:
            await self.update_position_price(symbol)
    
    async def update_position_price(self, symbol: str) -> Optional[float]:
        """
        Update current price for a single position
        """
        if symbol not in self.positions:
            return None
        
        try:
            # Get current price
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                current_price = float(data['Close'].iloc[-1])
            else:
                # Fallback to last known price
                current_price = self.positions[symbol].get("current_price", self.positions[symbol]["avg_price"])
            
            self.positions[symbol]["current_price"] = current_price
            self.positions[symbol]["market_value"] = self.positions[symbol]["shares"] * current_price
            self.positions[symbol]["unrealized_pl"] = self.positions[symbol]["market_value"] - self.positions[symbol]["cost_basis"]
            self.positions[symbol]["unrealized_plpc"] = (self.positions[symbol]["unrealized_pl"] / self.positions[symbol]["cost_basis"]) * 100
            
            return current_price
            
        except Exception as e:
            logging.error(f"Error updating price for {symbol}: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position details
        """
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Dict[str, Any]]:
        """
        Get all positions
        """
        return list(self.positions.values())
    
    def get_total_exposure(self) -> float:
        """
        Get total market value of all positions
        """
        return sum(p.get("market_value", p["shares"] * p["avg_price"]) for p in self.positions.values())
    
    def get_sector_exposure(self) -> Dict[str, float]:
        """
        Get exposure by sector
        """
        sector_exposure = {}
        total = self.get_total_exposure()
        
        for position in self.positions.values():
            sector = position.get("sector", "Unknown")
            value = position.get("market_value", position["shares"] * position["avg_price"])
            sector_exposure[sector] = sector_exposure.get(sector, 0) + value
        
        # Convert to percentages
        if total > 0:
            for sector in sector_exposure:
                sector_exposure[sector] = (sector_exposure[sector] / total) * 100
        
        return sector_exposure
    
    def check_limits(self) -> Dict[str, Any]:
        """
        Check if any position limits are exceeded
        """
        violations = []
        total = self.get_total_exposure()
        
        # Check individual position sizes
        for symbol, position in self.positions.items():
            value = position.get("market_value", position["shares"] * position["avg_price"])
            pct = (value / total) * 100 if total > 0 else 0
            
            if pct > self.max_position_size * 100:
                violations.append({
                    "type": "position_size",
                    "symbol": symbol,
                    "current": pct,
                    "limit": self.max_position_size * 100,
                    "message": f"Position {symbol} exceeds max size: {pct:.1f}% > {self.max_position_size*100:.1f}%"
                })
        
        # Check sector exposure
        sector_exp = self.get_sector_exposure()
        for sector, pct in sector_exp.items():
            if pct > self.max_sector_exposure * 100:
                violations.append({
                    "type": "sector_exposure",
                    "sector": sector,
                    "current": pct,
                    "limit": self.max_sector_exposure * 100,
                    "message": f"Sector {sector} exceeds max exposure: {pct:.1f}% > {self.max_sector_exposure*100:.1f}%"
                })
        
        return {
            "within_limits": len(violations) == 0,
            "violations": violations,
            "total_positions": len(self.positions),
            "total_exposure": total
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate position report
        """
        total_value = self.get_total_exposure()
        total_cost = sum(p["cost_basis"] for p in self.positions.values())
        total_pl = total_value - total_cost
        total_pl_pct = (total_pl / total_cost) * 100 if total_cost > 0 else 0
        
        positions_list = []
        for symbol, pos in self.positions.items():
            current_price = pos.get("current_price", pos["avg_price"])
            market_value = pos["shares"] * current_price
            pl = market_value - pos["cost_basis"]
            pl_pct = (pl / pos["cost_basis"]) * 100 if pos["cost_basis"] > 0 else 0
            
            positions_list.append({
                "symbol": symbol,
                "shares": pos["shares"],
                "avg_price": pos["avg_price"],
                "current_price": current_price,
                "cost_basis": pos["cost_basis"],
                "market_value": market_value,
                "unrealized_pl": pl,
                "unrealized_pl_pct": pl_pct,
                "sector": pos.get("sector", "Unknown"),
                "entry_date": pos["entry_date"]
            })
        
        # Sort by largest position
        positions_list.sort(key=lambda x: x["market_value"], reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_positions": len(self.positions),
            "total_cost": total_cost,
            "total_value": total_value,
            "total_unrealized_pl": total_pl,
            "total_unrealized_pl_pct": total_pl_pct,
            "sector_exposure": self.get_sector_exposure(),
            "positions": positions_list,
            "limits_check": self.check_limits()
        }