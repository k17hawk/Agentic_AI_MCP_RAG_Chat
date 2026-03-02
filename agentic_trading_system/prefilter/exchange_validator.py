"""
Exchange Validator - Validates stock exchange
"""
from typing import Dict, List, Optional, Any
from utils.logger import logger as  logging

class ExchangeValidator:
    """
    Validates that a stock trades on allowed exchanges
    
    Allowed exchanges typically include major US exchanges:
    - NYSE (New York Stock Exchange)
    - NASDAQ
    - AMEX (American Stock Exchange)
    - BATS (Better Alternative Trading System)
    - IEX (Investors Exchange)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Allowed exchanges (case-insensitive)
        self.allowed_exchanges = config.get("allowed_exchanges", [
            "nyse", "nasdaq", "amex", "bats", "iex",
            "nyse arca", "nyse american", "cboe", "cboe bzx"
        ])
        
        # Explicitly blocked exchanges
        self.blocked_exchanges = config.get("blocked_exchanges", [
            "otc", "pink", "grey", "other otc"
        ])
        
        # Exchange metadata
        self.exchange_metadata = {
            "nyse": {"name": "New York Stock Exchange", "country": "USA", "type": "primary"},
            "nasdaq": {"name": "NASDAQ", "country": "USA", "type": "primary"},
            "amex": {"name": "American Stock Exchange", "country": "USA", "type": "primary"},
            "bats": {"name": "BATS Global Markets", "country": "USA", "type": "primary"},
            "iex": {"name": "Investors Exchange", "country": "USA", "type": "primary"},
            "tsx": {"name": "Toronto Stock Exchange", "country": "Canada", "type": "primary"},
            "lse": {"name": "London Stock Exchange", "country": "UK", "type": "primary"},
            "hkex": {"name": "Hong Kong Exchange", "country": "Hong Kong", "type": "primary"}
        }
        
        logging.info(f"✅ ExchangeValidator initialized with {len(self.allowed_exchanges)} allowed exchanges")
    
    async def validate(self, ticker: str, info: Dict) -> Dict[str, Any]:
        """
        Validate if stock trades on allowed exchange
        """
        exchange = info.get("exchange", "").lower()
        
        # Check if exchange is in blocked list
        for blocked in self.blocked_exchanges:
            if blocked in exchange:
                return {
                    "passed": False,
                    "exchange": exchange,
                    "reason": f"Stock trades on blocked exchange: {exchange}"
                }
        
        # Check if exchange is allowed
        for allowed in self.allowed_exchanges:
            if allowed in exchange:
                metadata = self.exchange_metadata.get(allowed, {})
                return {
                    "passed": True,
                    "exchange": exchange,
                    "exchange_name": metadata.get("name", exchange),
                    "exchange_country": metadata.get("country", "Unknown"),
                    "exchange_type": metadata.get("type", "Unknown")
                }
        
        # Not in allowed list
        return {
            "passed": False,
            "exchange": exchange,
            "reason": f"Exchange not in allowed list: {exchange}"
        }
    
    def add_allowed_exchange(self, exchange: str):
        """Add an exchange to allowed list"""
        exchange_lower = exchange.lower()
        if exchange_lower not in self.allowed_exchanges:
            self.allowed_exchanges.append(exchange_lower)
            logging.info(f"➕ Added exchange to allowed list: {exchange}")
    
    def remove_allowed_exchange(self, exchange: str):
        """Remove an exchange from allowed list"""
        exchange_lower = exchange.lower()
        if exchange_lower in self.allowed_exchanges:
            self.allowed_exchanges.remove(exchange_lower)
            logging.info(f"➖ Removed exchange from allowed list: {exchange}")