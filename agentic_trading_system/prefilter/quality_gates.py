"""
Quality Gates - Main orchestrator for prefiltering stocks
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import asyncio
import yfinance as yf


from agents.base_agent import BaseAgent, AgentMessage

# Import all validators
from agentic_trading_system.prefilter.exchange_validator import ExchangeValidator
from agentic_trading_system.prefilter.price_range_checker import PriceRangeChecker
from agentic_trading_system.prefilter.volume_checker import VolumeChecker
from agentic_trading_system.prefilter.market_cap_checker import MarketCapChecker
from agentic_trading_system.prefilter.data_quality_checker import DataQualityChecker
from agentic_trading_system.prefilter.rejected_logger import RejectedLogger
from agentic_trading_system.prefilter.passed_queue import PassedQueue
from agentic_trading_system.utils.logger import logger as logging

class QualityGates(BaseAgent):
    """
    Main orchestrator for prefiltering stocks
    
    Responsibilities:
    - Run all quality checks on incoming tickers
    - Track rejection reasons
    - Maintain passed queue for analysis
    - Log all rejections for learning
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(
            name=name,
            description="Quality gates for stock prefiltering",
            config=config
        )
        
        # Initialize validators
        self.exchange_validator = ExchangeValidator(config.get("exchange_config", {}))
        self.price_checker = PriceRangeChecker(config.get("price_config", {}))
        self.volume_checker = VolumeChecker(config.get("volume_config", {}))
        self.market_cap_checker = MarketCapChecker(config.get("market_cap_config", {}))
        self.data_quality_checker = DataQualityChecker(config.get("data_quality_config", {}))
        
        # Initialize logger and queue
        self.rejected_logger = RejectedLogger(config.get("rejected_logger_config", {}))
        self.passed_queue = PassedQueue(config.get("queue_config", {}))
        
        # Cache for company info
        self.info_cache = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 5) * 60
        
        # Performance tracking
        self.stats = {
            "total_processed": 0,
            "passed": 0,
            "rejected": 0,
            "rejection_reasons": {}
        }
        
        logging.info(f"✅ QualityGates initialized")
    
    async def process(self, message: AgentMessage) -> Optional[AgentMessage]:
        """
        Process prefilter requests
        """
        if message.message_type == "prefilter_request":
            tickers = message.content.get("tickers", [])
            source = message.content.get("source", "unknown")
            
            results = await self.filter_tickers(tickers, source)
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="prefilter_result",
                content=results
            )
        
        elif message.message_type == "get_passed_queue":
            queue_items = await self.passed_queue.get_all()
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="passed_queue_result",
                content={"items": queue_items}
            )
        
        elif message.message_type == "get_rejection_stats":
            stats = self.rejected_logger.get_stats()
            
            return AgentMessage(
                sender=self.name,
                receiver=message.sender,
                message_type="rejection_stats",
                content=stats
            )
        
        return None
    
    async def filter_tickers(self, tickers: List[str], source: str) -> Dict[str, Any]:
        """
        Filter tickers through all quality gates
        """
        logging.info(f"🔍 Filtering {len(tickers)} tickers from {source}")
        
        passed = []
        rejected = []
        
        for ticker in tickers:
            self.stats["total_processed"] += 1
            
            # Get company info
            info = await self._get_company_info(ticker)
            if not info:
                rejected.append({
                    "ticker": ticker,
                    "reasons": ["Could not fetch company info"]
                })
                self.stats["rejected"] += 1
                self.stats["rejection_reasons"]["no_info"] = self.stats["rejection_reasons"].get("no_info", 0) + 1
                await self.rejected_logger.log(ticker, ["Could not fetch company info"], source)
                continue
            
            # Run all checks in parallel
            checks = await asyncio.gather(
                self.exchange_validator.validate(ticker, info),
                self.price_checker.validate(ticker, info),
                self.volume_checker.validate(ticker, info),
                self.market_cap_checker.validate(ticker, info),
                self.data_quality_checker.validate(ticker, info),
                return_exceptions=True
            )
            
            # Collect rejection reasons
            rejection_reasons = []
            
            for i, (check_name, check_result) in enumerate(zip(
                ["exchange", "price", "volume", "market_cap", "data_quality"],
                checks
            )):
                if isinstance(check_result, Exception):
                    rejection_reasons.append(f"{check_name}_check_error: {str(check_result)}")
                elif not check_result.get("passed", True):
                    rejection_reasons.append(check_result.get("reason", f"Failed {check_name} check"))
            
            if rejection_reasons:
                # Stock rejected
                rejected.append({
                    "ticker": ticker,
                    "reasons": rejection_reasons,
                    "info": info
                })
                self.stats["rejected"] += 1
                for reason in rejection_reasons:
                    self.stats["rejection_reasons"][reason] = self.stats["rejection_reasons"].get(reason, 0) + 1
                await self.rejected_logger.log(ticker, rejection_reasons, source, info)
            else:
                # Stock passed all checks
                passed_item = {
                    "ticker": ticker,
                    "info": info,
                    "timestamp": datetime.now().isoformat(),
                    "source": source,
                    "checks_passed": {
                        "exchange": checks[0],
                        "price": checks[1],
                        "volume": checks[2],
                        "market_cap": checks[3],
                        "data_quality": checks[4]
                    }
                }
                passed.append(passed_item)
                self.stats["passed"] += 1
                await self.passed_queue.add(passed_item)
        
        logging.info(f"✅ Prefilter complete: {len(passed)} passed, {len(rejected)} rejected")
        
        return {
            "timestamp": datetime.now().isoformat(),
            "source": source,
            "total_processed": len(tickers),
            "passed": passed,
            "passed_count": len(passed),
            "rejected": rejected,
            "rejected_count": len(rejected),
            "stats": self.stats
        }
    
    async def _get_company_info(self, ticker: str) -> Optional[Dict]:
        """
        Get company info with caching
        """
        # Check cache
        if ticker in self.info_cache:
            cached_time, info = self.info_cache[ticker]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                return info
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info:
                # Extract relevant fields
                clean_info = {
                    "symbol": ticker,
                    "name": info.get("longName", info.get("shortName", ticker)),
                    "exchange": info.get("exchange"),
                    "currency": info.get("currency", "USD"),
                    "current_price": info.get("currentPrice", info.get("regularMarketPrice")),
                    "previous_close": info.get("regularMarketPreviousClose"),
                    "day_high": info.get("dayHigh"),
                    "day_low": info.get("dayLow"),
                    "volume": info.get("volume"),
                    "average_volume": info.get("averageVolume"),
                    "average_volume_10d": info.get("averageVolume10days"),
                    "market_cap": info.get("marketCap"),
                    "shares_outstanding": info.get("sharesOutstanding"),
                    "float_shares": info.get("floatShares"),
                    "fifty_two_week_high": info.get("fiftyTwoWeekHigh"),
                    "fifty_two_week_low": info.get("fiftyTwoWeekLow"),
                    "sector": info.get("sector"),
                    "industry": info.get("industry"),
                    "country": info.get("country"),
                    "website": info.get("website"),
                    "bid": info.get("bid"),
                    "ask": info.get("ask"),
                    "bid_size": info.get("bidSize"),
                    "ask_size": info.get("askSize")
                }
                
                # Cache
                self.info_cache[ticker] = (datetime.now().timestamp(), clean_info)
                return clean_info
                
        except Exception as e:
            logging.debug(f"Error fetching info for {ticker}: {e}")
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        return {
            **self.stats,
            "queue_size": self.passed_queue.size(),
            "rejection_log_size": self.rejected_logger.size()
        }
    
    async def clear_queue(self):
        """Clear the passed queue"""
        await self.passed_queue.clear()
        logging.info("🧹 Cleared passed queue")