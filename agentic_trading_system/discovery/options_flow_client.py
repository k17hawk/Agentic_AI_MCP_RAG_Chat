"""
Options Flow Client - Detects unusual options activity
"""
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

class OptionsFlowClient:
    """
    Detects and analyzes unusual options activity
    
    Sources:
    - Unusual options volume
    - Large block trades
    - Sweep orders
    - Put/Call ratios
    - Open interest changes
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # API keys
        self.market_data_api = config.get("market_data_api")
        self.fmp_key = config.get("fmp_key")
        
        # Rate limiting
        self.rate_limit = config.get("rate_limit", 30)
        self.request_timestamps = []
        
        # Cache
        self.cache = {}
        self.cache_ttl = config.get("cache_ttl_minutes", 15) * 60
        
        # Thresholds
        self.volume_threshold = config.get("volume_threshold", 2.0)  # 2x average
        self.premium_threshold = config.get("premium_threshold", 100000)  # $100k
        
        logging.info(f"✅ OptionsFlowClient initialized")
    
    @retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search for unusual options activity
        """
        options = options or {}
        logging.info(f"📊 Options flow search for: '{query}'")
        
        # Check cache
        cache_key = f"options_{query}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                return cached_result
        
        await self._rate_limit()
        
        all_flows = []
        
        # Try multiple sources
        tasks = []
        if self.fmp_key:
            tasks.append(self._fetch_from_fmp(query, options))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                all_flows.extend(result)
        
        # Filter unusual activity
        unusual_flows = self._filter_unusual(all_flows)
        
        # Sort by significance
        unusual_flows.sort(key=lambda x: x.get("significance_score", 0), reverse=True)
        
        # Calculate aggregate metrics
        metrics = self._calculate_metrics(all_flows, unusual_flows)
        
        result = {
            "items": unusual_flows[:options.get("max_results", 50)],
            "all_flows_count": len(all_flows),
            "unusual_count": len(unusual_flows),
            "metrics": metrics,
            "metadata": {
                "ticker": query.upper(),
                "period_hours": options.get("hours_back", 24)
            }
        }
        
        self.cache[cache_key] = (datetime.now().timestamp(), result)
        
        logging.info(f"✅ Options flow: {len(unusual_flows)} unusual trades detected")
        return result
    
    async def _fetch_from_fmp(self, query: str, options: Dict) -> List[Dict]:
        """Fetch options data from Financial Modeling Prep"""
        if not self.fmp_key:
            return []
        
        # Get options chain
        url = f"https://financialmodelingprep.com/api/v3/stock-options/{query}"
        
        params = {
            "apikey": self.fmp_key
        }
        
        flows = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Get options chain
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Process puts and calls
                        for option_type in ["puts", "calls"]:
                            if option_type in data:
                                for option in data[option_type][:20]:  # Limit to top 20
                                    flow = self._analyze_option(option, option_type[:-1])
                                    if flow:
                                        flows.append(flow)
        except Exception as e:
            logging.debug(f"FMP options error: {e}")
        
        return flows
    
    def _analyze_option(self, option: Dict, option_type: str) -> Optional[Dict]:
        """
        Analyze a single option for unusual activity
        """
        try:
            strike = option.get("strike", 0)
            expiration = option.get("expirationDate", "")
            volume = option.get("volume", 0)
            open_interest = option.get("openInterest", 0)
            last_price = option.get("lastPrice", 0)
            implied_volatility = option.get("impliedVolatility", 0)
            
            # Calculate premium
            premium = volume * last_price * 100  # Each contract is 100 shares
            
            # Calculate volume ratio
            avg_volume = max(open_interest * 0.1, 1)  # Rough estimate
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0
            
            # Determine if unusual
            is_unusual = (
                volume_ratio > self.volume_threshold or
                premium > self.premium_threshold
            )
            
            # Calculate significance score
            significance = 0
            if is_unusual:
                significance = min(100, (
                    volume_ratio * 20 +
                    (premium / 100000) * 30 +
                    (implied_volatility * 20)
                ))
            
            return {
                "ticker": option.get("symbol", ""),
                "option_type": option_type,
                "strike": float(strike),
                "expiration": expiration,
                "volume": int(volume),
                "open_interest": int(open_interest),
                "last_price": float(last_price),
                "premium": float(premium),
                "volume_ratio": float(volume_ratio),
                "implied_volatility": float(implied_volatility),
                "is_unusual": is_unusual,
                "significance_score": float(significance),
                "source": "fmp",
                "timestamp": datetime.now().isoformat()
            }
        except:
            return None
    
    def _filter_unusual(self, flows: List[Dict]) -> List[Dict]:
        """Filter for unusual options activity"""
        unusual = []
        
        for flow in flows:
            if flow.get("is_unusual"):
                unusual.append(flow)
        
        return unusual
    
    def _calculate_metrics(self, all_flows: List[Dict], unusual_flows: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate options metrics"""
        if not all_flows:
            return {}
        
        # Put/Call ratio
        puts = sum(1 for f in all_flows if f.get("option_type") == "put")
        calls = sum(1 for f in all_flows if f.get("option_type") == "call")
        put_call_ratio = puts / calls if calls > 0 else float('inf')
        
        # Unusual put/call ratio
        unusual_puts = sum(1 for f in unusual_flows if f.get("option_type") == "put")
        unusual_calls = sum(1 for f in unusual_flows if f.get("option_type") == "call")
        unusual_pcr = unusual_puts / unusual_calls if unusual_calls > 0 else float('inf')
        
        # Total premiums
        total_premium = sum(f.get("premium", 0) for f in all_flows)
        unusual_premium = sum(f.get("premium", 0) for f in unusual_flows)
        
        return {
            "total_volume": len(all_flows),
            "puts": puts,
            "calls": calls,
            "put_call_ratio": float(put_call_ratio),
            "unusual_puts": unusual_puts,
            "unusual_calls": unusual_calls,
            "unusual_put_call_ratio": float(unusual_pcr) if unusual_calls > 0 else None,
            "total_premium": float(total_premium),
            "unusual_premium": float(unusual_premium),
            "unusual_percentage": len(unusual_flows) / len(all_flows) * 100 if all_flows else 0
        }
    
    async def _rate_limit(self):
        """Rate limiting"""
        now = datetime.now().timestamp()
        
        self.request_timestamps = [ts for ts in self.request_timestamps 
                                  if now - ts < 60]
        
        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_timestamps.append(now)