"""
Exceptions - Custom exception classes
"""
from typing import Any, Dict, Optional

class TradingBaseError(Exception):
    """Base exception for all trading system errors"""
    
    def __init__(self, message: str, details: Optional[Dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict:
        """Convert exception to dictionary"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }

class ConfigurationError(TradingBaseError):
    """Configuration related errors"""
    pass

class DataError(TradingBaseError):
    """Data fetching/processing errors"""
    pass

class ValidationError(TradingBaseError):
    """Input validation errors"""
    pass

class InsufficientFundsError(TradingBaseError):
    """Insufficient funds for trading"""
    
    def __init__(self, required: float, available: float, 
                 message: str = None, details: Dict = None):
        if message is None:
            message = f"Insufficient funds. Required: {required}, Available: {available}"
        super().__init__(message, details or {
            "required": required,
            "available": available,
            "shortfall": required - available
        })

class OrderError(TradingBaseError):
    """Order execution errors"""
    pass

class BrokerError(TradingBaseError):
    """Broker API errors"""
    pass

class RateLimitError(TradingBaseError):
    """Rate limit exceeded"""
    pass

class MarketClosedError(TradingBaseError):
    """Market is closed"""
    pass

class InvalidSymbolError(TradingBaseError):
    """Invalid trading symbol"""
    pass

class PositionError(TradingBaseError):
    """Position management errors"""
    pass

class RiskLimitError(TradingBaseError):
    """Risk limit exceeded"""
    pass

class TimeoutError(TradingBaseError):
    """Operation timeout"""
    pass

class AuthenticationError(TradingBaseError):
    """Authentication failed"""
    pass

class DatabaseError(TradingBaseError):
    """Database operation errors"""
    pass

class CacheError(TradingBaseError):
    """Cache operation errors"""
    pass

class SignalError(TradingBaseError):
    """Signal processing errors"""
    pass

class AnalysisError(TradingBaseError):
    """Analysis errors"""
    pass

def handle_exception(e: Exception) -> Dict:
    """
    Standard exception handler for API responses
    """
    if isinstance(e, TradingBaseError):
        return {
            "success": False,
            "error": e.message,
            "error_type": e.__class__.__name__,
            "details": e.details
        }
    else:
        return {
            "success": False,
            "error": str(e),
            "error_type": "UnexpectedError",
            "details": {}
        }