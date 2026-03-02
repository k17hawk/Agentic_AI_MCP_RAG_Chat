"""
Paper Trading - Simulated trading for testing
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import random
import yfinance as yf
from utils.logger import logger as logging
import asyncio
class PaperTrading:
    """
    Paper Trading - Simulated trading environment
    
    Features:
    - Virtual account with fake money
    - Simulated fills with realistic delays
    - Market data integration
    - Performance tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Account parameters
        self.initial_capital = config.get("initial_capital", 100000.0)
        self.cash = self.initial_capital
        self.positions = {}
        self.order_history = []
        
        # Simulation parameters
        self.fill_delay_seconds = config.get("fill_delay_seconds", 1)
        self.slippage_model = config.get("slippage_model", "fixed")  # fixed, percentage, market
        self.fixed_slippage = config.get("fixed_slippage", 0.01)  # $0.01 fixed slippage
        self.percentage_slippage = config.get("percentage_slippage", 0.001)  # 0.1% slippage
        
        # Market data cache
        self.price_cache = {}
        
        logging.info(f"✅ PaperTrading initialized with ${self.initial_capital:,.2f}")
    
    async def get_account(self) -> Dict[str, Any]:
        """
        Get paper trading account information
        """
        # Calculate total value
        total_value = self.cash
        for symbol, pos in self.positions.items():
            current_price = await self._get_current_price(symbol)
            pos_value = pos["quantity"] * current_price
            total_value += pos_value
            
            # Update unrealized P&L
            pos["current_price"] = current_price
            pos["market_value"] = pos_value
            pos["unrealized_pl"] = pos_value - pos["cost_basis"]
            pos["unrealized_plpc"] = (pos["unrealized_pl"] / pos["cost_basis"]) if pos["cost_basis"] > 0 else 0
        
        return {
            "account_id": "paper_001",
            "currency": "USD",
            "cash": self.cash,
            "portfolio_value": total_value,
            "equity": total_value,
            "initial_capital": self.initial_capital,
            "total_pl": total_value - self.initial_capital,
            "total_pl_percent": ((total_value - self.initial_capital) / self.initial_capital) * 100,
            "buying_power": self.cash * 2,  # 2x leverage
            "positions": len(self.positions),
            "is_paper": True
        }
    
    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit a paper trading order
        """
        symbol = order["symbol"]
        quantity = order["quantity"]
        side = order["side"]
        order_type = order.get("order_type", "MARKET")
        
        # Get current price
        current_price = await self._get_current_price(symbol)
        
        # Apply slippage
        execution_price = self._apply_slippage(current_price, side, order_type, order)
        
        # Calculate cost
        cost = quantity * execution_price
        
        # Check if we have enough cash for buys
        if side == "BUY" and cost > self.cash:
            return {
                "success": False,
                "error": "Insufficient funds",
                "cash_available": self.cash,
                "cost": cost
            }
        
        # Simulate fill delay
        await asyncio.sleep(self.fill_delay_seconds)
        
        # Create order response
        order_response = {
            "order_id": f"paper_{datetime.now().timestamp()}",
            "symbol": symbol,
            "quantity": quantity,
            "filled_quantity": quantity,
            "side": side,
            "order_type": order_type,
            "limit_price": order.get("limit_price"),
            "stop_price": order.get("stop_price"),
            "status": "filled",
            "filled_price": execution_price,
            "filled_at": datetime.now().isoformat(),
            "cost": cost,
            "success": True,
            "is_paper": True
        }
        
        # Update account
        if side == "BUY":
            self.cash -= cost
            
            # Update or create position
            if symbol in self.positions:
                pos = self.positions[symbol]
                # Average down cost basis
                total_shares = pos["quantity"] + quantity
                total_cost = pos["cost_basis"] + cost
                pos["quantity"] = total_shares
                pos["cost_basis"] = total_cost
                pos["avg_price"] = total_cost / total_shares
            else:
                self.positions[symbol] = {
                    "quantity": quantity,
                    "cost_basis": cost,
                    "avg_price": execution_price,
                    "entry_date": datetime.now().isoformat()
                }
        else:  # SELL
            self.cash += cost
            
            # Reduce or remove position
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos["quantity"] -= quantity
                
                # Reduce cost basis proportionally
                reduction_ratio = quantity / (pos["quantity"] + quantity)
                pos["cost_basis"] *= (1 - reduction_ratio)
                
                if pos["quantity"] <= 0:
                    del self.positions[symbol]
        
        # Record order
        self.order_history.append(order_response)
        
        logging.info(f"✅ Paper trade executed: {side} {quantity} {symbol} @ ${execution_price:.2f}")
        
        return order_response
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order (paper trading - orders fill instantly)
        """
        return False
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current paper trading positions
        """
        positions = []
        
        for symbol, pos in self.positions.items():
            current_price = await self._get_current_price(symbol)
            market_value = pos["quantity"] * current_price
            
            positions.append({
                "symbol": symbol,
                "quantity": pos["quantity"],
                "avg_entry_price": pos["avg_price"],
                "current_price": current_price,
                "cost_basis": pos["cost_basis"],
                "market_value": market_value,
                "unrealized_pl": market_value - pos["cost_basis"],
                "unrealized_plpc": (market_value - pos["cost_basis"]) / pos["cost_basis"] if pos["cost_basis"] > 0 else 0,
                "entry_date": pos["entry_date"]
            })
        
        return positions
    
    async def get_order_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order history
        """
        return self.order_history[-limit:]
    
    async def _get_current_price(self, symbol: str) -> float:
        """
        Get current price from cache or yfinance
        """
        # Check cache (30 second TTL)
        if symbol in self.price_cache:
            price, timestamp = self.price_cache[symbol]
            if (datetime.now() - timestamp).total_seconds() < 30:
                return price
        
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            
            if not data.empty:
                price = float(data['Close'].iloc[-1])
            else:
                # Fallback to random price movement
                price = 100.0 + random.uniform(-1, 1)
            
            self.price_cache[symbol] = (price, datetime.now())
            return price
            
        except Exception as e:
            logging.debug(f"Error getting price for {symbol}: {e}")
            # Return random price for testing
            return 100.0 + random.uniform(-1, 1)
    
    def _apply_slippage(self, price: float, side: str, order_type: str, 
                        order: Dict) -> float:
        """
        Apply slippage to execution price
        """
        if order_type == "LIMIT":
            # Limit orders execute at limit price or better
            limit = order.get("limit_price")
            if limit:
                if side == "BUY":
                    return min(price, limit)
                else:
                    return max(price, limit)
        
        if order_type == "STOP":
            # Stop orders become market orders when triggered
            pass
        
        # Apply slippage model
        if self.slippage_model == "fixed":
            if side == "BUY":
                return price + self.fixed_slippage
            else:
                return price - self.fixed_slippage
        
        elif self.slippage_model == "percentage":
            if side == "BUY":
                return price * (1 + self.percentage_slippage)
            else:
                return price * (1 - self.percentage_slippage)
        
        else:  # market
            # Random slippage based on liquidity
            random_slippage = random.uniform(-0.001, 0.001)
            return price * (1 + random_slippage)