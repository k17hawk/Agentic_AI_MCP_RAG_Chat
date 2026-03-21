"""
Exchange Validator - Validates stock exchange
"""
from typing import Dict, List, Optional, Any
from agentic_trading_system.utils.logger import logger as logging


class ExchangeValidator:
    """
    Validates that a stock trades on allowed exchanges
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Allowed exchanges (case-insensitive - convert to uppercase for matching)
        self.allowed_exchanges = [ex.upper() for ex in config.get("allowed_exchanges", [
            "NYSE", "NASDAQ", "AMEX", "BATS", "IEX",
            "NYSE ARCA", "NYSE AMERICAN", "CBOE", "CBOE BZX"
        ])]
        
        # Also allow common exchange codes (from Yahoo Finance)
        self.exchange_codes = {
            "NMS": "NASDAQ",      # NASDAQ Global Select Market
            "NGM": "NASDAQ",      # NASDAQ Global Market
            "NCM": "NASDAQ",      # NASDAQ Capital Market
            "NYQ": "NYSE",        # New York Stock Exchange
            "ASE": "AMEX",        # NYSE American
            "BAT": "BATS",        # BATS Global Markets
            "IEX": "IEX"          # IEX
        }
        
        # Explicitly blocked exchanges
        self.blocked_exchanges = config.get("blocked_exchanges", [
            "OTC", "PINK", "GREY", "OTHER OTC", "YHD"
        ])
        
        # Exchange metadata
        self.exchange_metadata = {
            "NYSE": {"name": "New York Stock Exchange", "country": "USA", "type": "primary"},
            "NASDAQ": {"name": "NASDAQ", "country": "USA", "type": "primary"},
            "AMEX": {"name": "NYSE American", "country": "USA", "type": "primary"},
            "BATS": {"name": "BATS Global Markets", "country": "USA", "type": "primary"},
            "IEX": {"name": "Investors Exchange", "country": "USA", "type": "primary"}
        }
        
        logging.info(f"✅ ExchangeValidator initialized with {len(self.allowed_exchanges)} allowed exchanges")
    
    async def validate(self, ticker: str, info: Dict) -> Dict[str, Any]:
        """
        Validate if stock trades on allowed exchange
        """
        # Get exchange from info
        exchange_raw = info.get("exchange", "")
        
        # Convert to uppercase for case-insensitive comparison
        exchange_upper = exchange_raw.upper() if exchange_raw else ""
        
        # Map exchange code to actual exchange name
        mapped_exchange = self.exchange_codes.get(exchange_upper, exchange_upper)
        
        logging.debug(f"Exchange validation for {ticker}: raw='{exchange_raw}', code='{exchange_upper}', mapped='{mapped_exchange}'")
        
        # Check if exchange is in blocked list
        for blocked in self.blocked_exchanges:
            if blocked in exchange_upper or blocked in mapped_exchange:
                return {
                    "passed": False,
                    "exchange": exchange_raw,
                    "reason": f"Stock trades on blocked exchange: {exchange_raw}"
                }
        
        # Check if mapped exchange is allowed
        if mapped_exchange in self.allowed_exchanges:
            metadata = self.exchange_metadata.get(mapped_exchange, {})
            return {
                "passed": True,
                "exchange": exchange_raw,
                "exchange_code": exchange_upper,
                "mapped_exchange": mapped_exchange,
                "exchange_name": metadata.get("name", mapped_exchange),
                "exchange_country": metadata.get("country", "Unknown"),
                "exchange_type": metadata.get("type", "Unknown")
            }
        
        # Check if original exchange is allowed (for cases where mapping didn't apply)
        if exchange_upper in self.allowed_exchanges:
            metadata = self.exchange_metadata.get(exchange_upper, {})
            return {
                "passed": True,
                "exchange": exchange_raw,
                "exchange_code": exchange_upper,
                "mapped_exchange": exchange_upper,
                "exchange_name": metadata.get("name", exchange_upper),
                "exchange_country": metadata.get("country", "Unknown"),
                "exchange_type": metadata.get("type", "Unknown")
            }
        
        # Not in allowed list
        return {
            "passed": False,
            "exchange": exchange_raw,
            "exchange_code": exchange_upper,
            "reason": f"Exchange not in allowed list: {exchange_raw} (code: {exchange_upper}, mapped: {mapped_exchange}). Allowed: {', '.join(self.allowed_exchanges[:5])}..."
        }
    
    def add_allowed_exchange(self, exchange: str):
        """Add an exchange to allowed list"""
        exchange_upper = exchange.upper()
        if exchange_upper not in self.allowed_exchanges:
            self.allowed_exchanges.append(exchange_upper)
            logging.info(f"➕ Added exchange to allowed list: {exchange}")
    
    def remove_allowed_exchange(self, exchange: str):
        """Remove an exchange from allowed list"""
        exchange_upper = exchange.upper()
        if exchange_upper in self.allowed_exchanges:
            self.allowed_exchanges.remove(exchange_upper)
            logging.info(f"➖ Removed exchange from allowed list: {exchange}")