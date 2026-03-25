# =============================================================================
# discovery/options_flow_client.py (UPDATED)
# =============================================================================
"""
Options Flow Client - Detects unusual options activity
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import aiohttp
import asyncio
import yfinance as yf
import math
import re

from agentic_trading_system.utils.logger import logger as logging
from agentic_trading_system.utils.decorators import retry

# Import from new config structure
from agentic_trading_system.constants import Source, OptionFlowThresholds, RateLimit, CacheTTL
from agentic_trading_system.config.config__entity import OptionsFlowConfig


class OptionsFlowClient:
    """
    Detects and analyzes unusual options activity.

    Sources:
    - Unusual options volume
    - Large block trades
    - Sweep orders
    - Put/Call ratios
    - Open interest changes
    """

    def __init__(self, config: OptionsFlowConfig):
        """
        Initialize options flow client.
        
        Args:
            config: OptionsFlowConfig object
        """
        self.config = config

        # API keys
        self.fmp_key = config.fmp_key

        # Rate limiting
        self.rate_limit = config.rate_limit
        self.request_timestamps: List[float] = []

        # Cache
        self.cache: Dict = {}
        self.cache_ttl = config.cache_ttl_minutes * 60

        # Thresholds
        self.volume_threshold = config.volume_threshold
        self.premium_threshold = config.premium_threshold

        logging.info(f"✅ OptionsFlowClient initialized")

    @retry(max_attempts=3, delay=1.0)
    async def search(self, query: str, options: Dict = None) -> Dict[str, Any]:
        """
        Search for unusual options activity.
        
        Args:
            query: Stock ticker or company name
            options: Search options
            
        Returns:
            Dict with 'items' and 'metadata' keys
        """
        options = options or {}

        # Extract ticker from query
        ticker_match = re.search(r'\b[A-Z]{1,5}\b', query.upper())
        ticker = ticker_match.group() if ticker_match else query.upper().replace(" ", "")

        logging.info(f"📊 Options flow search for: '{ticker}'")

        # Check cache
        cache_key = f"options_{ticker}_{hash(str(options))}"
        if cache_key in self.cache:
            cached_time, cached_result = self.cache[cache_key]
            if datetime.now().timestamp() - cached_time < self.cache_ttl:
                logging.info(f"📦 Using cached options results for '{ticker}'")
                return cached_result

        await self._rate_limit()

        all_flows = []

        # Try multiple sources
        tasks = []
        if self.fmp_key:
            tasks.append(self._fetch_from_fmp(ticker, options))

        if not tasks:
            return {
                "items": [],
                "all_flows_count": 0,
                "unusual_count": 0,
                "metrics": {},
                "metadata": {
                    "ticker": ticker,
                    "period_hours": options.get("hours_back", 24),
                    "error": "No data sources configured"
                }
            }

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, list):
                all_flows.extend(result)
            elif isinstance(result, Exception):
                logging.warning(f"⚠️ Options flow error: {result}")

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
                "ticker": ticker,
                "period_hours": options.get("hours_back", 24)
            }
        }

        self.cache[cache_key] = (datetime.now().timestamp(), result)

        logging.info(f"✅ Options flow: {len(unusual_flows)} unusual trades detected for {ticker}")
        return result

    async def _fetch_from_fmp(self, ticker: str, options: Dict) -> List[Dict]:
        """Fetch options data from FMP with yfinance fallback."""
        flows = []

        # Attempt FMP
        if self.fmp_key:
            url = f"https://financialmodelingprep.com/api/v3/option-chain/{ticker}"
            params = {"apikey": self.fmp_key}

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        url, params=params,
                        timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            if isinstance(data, list) and data:
                                for option in data:
                                    opt_type = option.get("contractType", "").lower()
                                    if opt_type in ("call", "put"):
                                        flow = self._analyze_option(option, opt_type, ticker)
                                        if flow:
                                            flows.append(flow)
                                logging.info(f"✅ FMP options: {len(flows)} items for {ticker}")
                                return flows
                            else:
                                logging.debug(f"FMP returned no data for {ticker}")
            except asyncio.TimeoutError:
                logging.debug(f"FMP options timeout for {ticker}")
            except Exception as e:
                logging.debug(f"FMP options error: {e}")

        # Fallback to yfinance
        if self.config.use_alternative:
            logging.info(f"📡 Falling back to yfinance for options data on {ticker}")
            yf_flows = await self._fetch_options_from_yfinance(ticker)
            if yf_flows:
                flows.extend(yf_flows)
                logging.info(f"✅ yfinance options: {len(yf_flows)} items for {ticker}")

        return flows

    async def _fetch_options_from_yfinance(self, ticker: str) -> List[Dict]:
        """Fetch options data using yfinance (runs in thread pool)."""
        loop = asyncio.get_event_loop()

        def _get_options_blocking():
            try:
                stock = yf.Ticker(ticker)
                expirations = stock.options
                if not expirations:
                    return []

                # Use nearest expiration
                chain = stock.option_chain(expirations[0])
                flows = []

                # Process calls
                for opt in chain.calls.to_dict('records'):
                    opt["symbol"] = ticker
                    opt["expirationDate"] = expirations[0]
                    analyzed = self._analyze_option(opt, "call", ticker)
                    if analyzed:
                        analyzed["source"] = Source.YAHOO_FINANCE
                        flows.append(analyzed)

                # Process puts
                for opt in chain.puts.to_dict('records'):
                    opt["symbol"] = ticker
                    opt["expirationDate"] = expirations[0]
                    analyzed = self._analyze_option(opt, "put", ticker)
                    if analyzed:
                        analyzed["source"] = Source.YAHOO_FINANCE
                        flows.append(analyzed)

                return flows
            except Exception as e:
                logging.debug(f"yfinance options error for {ticker}: {e}")
                return []

        try:
            flows = await asyncio.wait_for(
                loop.run_in_executor(None, _get_options_blocking),
                timeout=self.config.timeout_seconds
            )
            return flows
        except asyncio.TimeoutError:
            logging.warning(f"⏰ yfinance options timeout for {ticker}")
            return []
        except Exception as e:
            logging.warning(f"⚠️ yfinance options exception: {e}")
            return []

    def _analyze_option(self, option: Dict, option_type: str, ticker: str = "") -> Optional[Dict]:
        """Analyze a single option for unusual activity."""
        try:
            # Safely extract values
            strike = self._safe_float(option.get("strike"))
            expiration = option.get("expirationDate", "")

            volume = self._safe_int(option.get("volume"))
            open_interest = self._safe_int(option.get("openInterest"))
            last_price = self._safe_float(option.get("lastPrice"))
            implied_volatility = self._safe_float(option.get("impliedVolatility"))

            ticker = ticker or option.get("symbol", "")

            # Calculate premium
            premium = volume * last_price * OptionFlowThresholds.CONTRACT_MULTIPLIER

            # Calculate volume ratio
            avg_volume = max(open_interest * 0.1, 1)
            volume_ratio = volume / avg_volume if avg_volume > 0 else 0

            # Determine if unusual
            is_unusual = (
                volume_ratio > self.volume_threshold or
                premium > self.premium_threshold
            )

            # Calculate significance score
            significance = 0.0
            if is_unusual:
                significance = min(100.0, (
                    volume_ratio * 20 +
                    (premium / self.premium_threshold) * 30 +
                    (implied_volatility * 20)
                ))

            # Build human-readable title and content
            action = "Unusual" if is_unusual else "Normal"
            opt_type_display = "CALL" if option_type == "call" else "PUT"
            title = f"{action} {opt_type_display} option on {ticker} at ${strike:.2f}"
            content = (
                f"Option Type: {opt_type_display}\n"
                f"Strike: ${strike:.2f}\n"
                f"Expiration: {expiration}\n"
                f"Volume: {volume:,}\n"
                f"Open Interest: {open_interest:,}\n"
                f"Last Price: ${last_price:.2f}\n"
                f"Premium: ${premium:,.2f}\n"
                f"Volume Ratio: {volume_ratio:.2f}x\n"
                f"Implied Volatility: {implied_volatility:.2%}\n"
                f"Significance Score: {significance:.1f}"
            )

            return {
                "ticker": ticker,
                "option_type": option_type,
                "strike": strike,
                "expiration": expiration,
                "volume": volume,
                "open_interest": open_interest,
                "last_price": last_price,
                "premium": premium,
                "volume_ratio": volume_ratio,
                "implied_volatility": implied_volatility,
                "is_unusual": is_unusual,
                "significance_score": significance,
                "source": Source.FMP,
                "timestamp": datetime.now().isoformat(),
                "title": title,
                "content": content,
                "relevance_score": 1.0 if is_unusual else 0.5,
                "type": "option_flow"
            }
        except Exception as e:
            logging.debug(f"Option analysis error: {e}")
            return None

    @staticmethod
    def _safe_float(value, default=0.0):
        """Convert value to float, return default on failure."""
        if value is None:
            return default
        try:
            f = float(value)
            return f if not math.isnan(f) else default
        except (ValueError, TypeError):
            return default

    @staticmethod
    def _safe_int(value, default=0):
        """Convert value to int, return default on failure."""
        if value is None:
            return default
        try:
            if isinstance(value, float) and value.is_integer():
                return int(value)
            return int(float(value))
        except (ValueError, TypeError):
            return default

    def _filter_unusual(self, flows: List[Dict]) -> List[Dict]:
        """Filter for unusual options activity."""
        return [flow for flow in flows if flow.get("is_unusual")]

    def _calculate_metrics(self, all_flows: List[Dict], unusual_flows: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate options metrics."""
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
        """Rate limiting."""
        now = datetime.now().timestamp()

        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if now - ts < RateLimit.WINDOW_SECONDS]

        if len(self.request_timestamps) >= self.rate_limit:
            sleep_time = RateLimit.WINDOW_SECONDS - (now - self.request_timestamps[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(now)

    async def clear_cache(self) -> None:
        """Clear all caches."""
        self.cache.clear()
        logging.info("🧹 Options flow cache cleared")