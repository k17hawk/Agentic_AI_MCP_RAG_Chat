"""
Alpaca Client - Alpaca Markets API integration
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import os
import asyncio
import aiohttp
from utils.logger import logger as  logging

class AlpacaClient:
    """
    Alpaca Markets API client for real trading
    
    Supports:
    - Stock trading
    - Options trading (if enabled)
    - Real-time data
    - Account management
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Get credentials from environment or config
        self.api_key = os.getenv("ALPACA_API_KEY") or config.get("api_key")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY") or config.get("api_secret")
        self.base_url = config.get("base_url", "https://paper-api.alpaca.markets")  # Paper trading by default
        self.data_url = config.get("data_url", "https://data.alpaca.markets")
        
        # Headers for API requests
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json"
        }
        
        # Account info
        self.account = None
        self.last_update = None
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 200)  # requests per minute
        self.request_timestamps = []
        
        # Check if configured
        self.is_configured = bool(self.api_key and self.api_secret)
        
        if self.is_configured:
            logging.info(f"✅ AlpacaClient initialized (paper trading: 'paper' in {self.base_url})")
        else:
            logging.warning("⚠️ Alpaca credentials not found - client disabled")
    
    async def get_account(self) -> Dict[str, Any]:
        """
        Get account information
        """
        if not self.is_configured:
            return self._mock_account()
        
        await self._rate_limit()
        
        url = f"{self.base_url}/v2/account"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        self.account = data
                        self.last_update = datetime.now()
                        return self._format_account(data)
                    else:
                        error = await response.text()
                        logging.error(f"Alpaca account error: {error}")
                        return self._mock_account()
        except Exception as e:
            logging.error(f"Alpaca connection error: {e}")
            return self._mock_account()
    
    async def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit an order to Alpaca
        """
        if not self.is_configured:
            return self._mock_order_response(order)
        
        await self._rate_limit()
        
        url = f"{self.base_url}/v2/orders"
        
        # Map internal order to Alpaca format
        alpaca_order = self._to_alpaca_order(order)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=self.headers, json=alpaca_order) as response:
                    if response.status in [200, 201]:
                        data = await response.json()
                        logging.info(f"✅ Order submitted to Alpaca: {data.get('id')}")
                        return self._from_alpaca_order(data)
                    else:
                        error = await response.text()
                        logging.error(f"Alpaca order error: {error}")
                        return {
                            "success": False,
                            "error": error,
                            "order": order
                        }
        except Exception as e:
            logging.error(f"Alpaca order submission error: {e}")
            return {
                "success": False,
                "error": str(e),
                "order": order
            }
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an order
        """
        if not self.is_configured:
            return True
        
        await self._rate_limit()
        
        url = f"{self.base_url}/v2/orders/{order_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(url, headers=self.headers) as response:
                    if response.status == 204:
                        logging.info(f"✅ Order cancelled: {order_id}")
                        return True
                    else:
                        error = await response.text()
                        logging.error(f"Cancel order error: {error}")
                        return False
        except Exception as e:
            logging.error(f"Cancel order error: {e}")
            return False
    
    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order details
        """
        if not self.is_configured:
            return None
        
        await self._rate_limit()
        
        url = f"{self.base_url}/v2/orders/{order_id}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._from_alpaca_order(data)
                    else:
                        return None
        except Exception as e:
            logging.error(f"Get order error: {e}")
            return None
    
    async def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions
        """
        if not self.is_configured:
            return []
        
        await self._rate_limit()
        
        url = f"{self.base_url}/v2/positions"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        return [self._format_position(p) for p in data]
                    else:
                        return []
        except Exception as e:
            logging.error(f"Get positions error: {e}")
            return []
    
    async def get_bars(self, symbol: str, timeframe: str = "1D", 
                      limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get historical bars
        """
        if not self.is_configured:
            return []
        
        await self._rate_limit()
        
        url = f"{self.data_url}/v2/stocks/{symbol}/bars"
        
        params = {
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": "raw"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self.headers, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("bars", [])
                    else:
                        return []
        except Exception as e:
            logging.error(f"Get bars error: {e}")
            return []
    
    def _to_alpaca_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Convert internal order to Alpaca format"""
        alpaca_order = {
            "symbol": order["symbol"],
            "qty": str(order["quantity"]),
            "side": order["side"].lower(),
            "type": order["order_type"].lower(),
            "time_in_force": order.get("time_in_force", "day").lower()
        }
        
        # Add limit price if limit order
        if order["order_type"] == "LIMIT":
            alpaca_order["limit_price"] = str(order["limit_price"])
        
        # Add stop price if stop order
        if order["order_type"] == "STOP":
            alpaca_order["stop_price"] = str(order["stop_price"])
        
        # Add client order ID if provided
        if "client_order_id" in order:
            alpaca_order["client_order_id"] = order["client_order_id"]
        
        return alpaca_order
    
    def _from_alpaca_order(self, data: Dict) -> Dict[str, Any]:
        """Convert Alpaca order to internal format"""
        return {
            "broker_order_id": data.get("id"),
            "client_order_id": data.get("client_order_id"),
            "symbol": data.get("symbol"),
            "quantity": int(float(data.get("qty", 0))),
            "filled_quantity": int(float(data.get("filled_qty", 0))),
            "side": data.get("side", "").upper(),
            "order_type": data.get("type", "").upper(),
            "limit_price": float(data.get("limit_price", 0)) if data.get("limit_price") else None,
            "stop_price": float(data.get("stop_price", 0)) if data.get("stop_price") else None,
            "status": data.get("status"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "submitted_at": data.get("submitted_at"),
            "filled_at": data.get("filled_at"),
            "filled_avg_price": float(data.get("filled_avg_price", 0)) if data.get("filled_avg_price") else None,
            "success": True
        }
    
    def _format_account(self, data: Dict) -> Dict[str, Any]:
        """Format account data"""
        return {
            "id": data.get("id"),
            "account_number": data.get("account_number"),
            "status": data.get("status"),
            "currency": data.get("currency"),
            "cash": float(data.get("cash", 0)),
            "portfolio_value": float(data.get("portfolio_value", 0)),
            "buying_power": float(data.get("buying_power", 0)),
            "daytrade_count": data.get("daytrade_count", 0),
            "equity": float(data.get("equity", 0)),
            "last_equity": float(data.get("last_equity", 0)),
            "multiplier": data.get("multiplier", 1),
            "pattern_day_trader": data.get("pattern_day_trader", False)
        }
    
    def _format_position(self, data: Dict) -> Dict[str, Any]:
        """Format position data"""
        return {
            "symbol": data.get("symbol"),
            "quantity": int(float(data.get("qty", 0))),
            "avg_entry_price": float(data.get("avg_entry_price", 0)),
            "current_price": float(data.get("current_price", 0)),
            "market_value": float(data.get("market_value", 0)),
            "cost_basis": float(data.get("cost_basis", 0)),
            "unrealized_pl": float(data.get("unrealized_pl", 0)),
            "unrealized_plpc": float(data.get("unrealized_plpc", 0)),
            "current_day_pl": float(data.get("current_day_pl", 0)),
            "change_today": float(data.get("change_today", 0))
        }
    
    def _mock_account(self) -> Dict[str, Any]:
        """Return mock account for testing"""
        return {
            "status": "ACTIVE",
            "currency": "USD",
            "cash": 100000.0,
            "portfolio_value": 100000.0,
            "buying_power": 200000.0,
            "equity": 100000.0,
            "pattern_day_trader": False,
            "mock": True
        }
    
    def _mock_order_response(self, order: Dict) -> Dict[str, Any]:
        """Return mock order response"""
        return {
            "broker_order_id": f"mock_{datetime.now().timestamp()}",
            "symbol": order["symbol"],
            "quantity": order["quantity"],
            "side": order["side"],
            "order_type": order["order_type"],
            "status": "accepted",
            "success": True,
            "mock": True
        }
    
    async def _rate_limit(self):
        """Rate limiting"""
        now = datetime.now().timestamp()
        
        self.request_timestamps = [ts for ts in self.request_timestamps if now - ts < 60]
        
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(now)